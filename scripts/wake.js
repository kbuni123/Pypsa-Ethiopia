/**
 * Wakes sleeping Streamlit Community Cloud apps.
 *
 * Community Cloud sleeps an app after 12 hours without traffic. A plain HTTP
 * request does not reset that timer -- Streamlit counts the websocket session
 * a real browser opens. So this loads each app in headless Chromium, clicks the
 * wake button if the sleep screen is showing, and waits for the app to boot.
 *
 * Success is judged by the sleep screen being gone, not by any internal
 * Streamlit selector, since those change between releases.
 *
 * Diagnostics (screenshots + page text) are written to ./diagnostics.
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const APPS = (process.env.APP_URLS || '').split(/[\s,]+/).filter(Boolean);
const OUT_DIR = path.join(process.cwd(), 'diagnostics');

// Phrases that mean "this app is asleep", across Streamlit's wordings.
const SLEEP_MARKERS = [
  /get this app back up/i,
  /has gone to sleep/i,
  /is currently sleeping/i,
  /wake.{0,10}app/i,
];

const BOOT_TIMEOUT_MS = 420_000; // cold starts reinstall dependencies
const NAV_TIMEOUT_MS = 90_000;
const SETTLE_MS = 10_000;

const slug = (url) => url.replace(/^https?:\/\//, '').replace(/[^a-z0-9]+/gi, '_').slice(0, 60);

async function pageText(page) {
  return page
    .evaluate(() => document.body?.innerText || '')
    .catch(() => '');
}

async function looksAsleep(page) {
  const text = await pageText(page);
  return SLEEP_MARKERS.some((re) => re.test(text));
}

async function snap(page, name, label) {
  const file = path.join(OUT_DIR, `${name}__${label}.png`);
  await page.screenshot({ path: file, fullPage: true }).catch(() => {});
}

async function visit(browser, url) {
  const name = slug(url);
  const log = (msg) => console.log(`    ${msg}`);
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();

  try {
    console.log(`\n=== ${url}`);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS });
    await page.waitForTimeout(5_000); // let the shell render

    log(`title: ${JSON.stringify(await page.title())}`);
    await snap(page, name, '1-landed');

    const initialText = await pageText(page);
    fs.writeFileSync(path.join(OUT_DIR, `${name}__landed.txt`), initialText);
    log(`first 200 chars: ${JSON.stringify(initialText.slice(0, 200))}`);

    const buttons = await page
      .locator('button, [role="button"], a')
      .allInnerTexts()
      .catch(() => []);
    const labels = buttons.map((b) => b.trim()).filter(Boolean).slice(0, 15);
    log(`clickable labels: ${JSON.stringify(labels)}`);

    let asleep = await looksAsleep(page);
    log(`sleep screen detected: ${asleep}`);

    if (asleep) {
      // Click whichever control carries a wake phrase; fall back to the only button.
      let clicked = false;
      for (const re of SLEEP_MARKERS) {
        const el = page.locator('button, [role="button"], a').filter({ hasText: re }).first();
        if (await el.count().then((n) => n > 0).catch(() => false)) {
          await el.click({ timeout: 10_000 }).catch(() => {});
          clicked = true;
          log(`clicked control matching ${re}`);
          break;
        }
      }
      if (!clicked) {
        const only = page.locator('button').first();
        if (await only.count().then((n) => n > 0).catch(() => false)) {
          await only.click({ timeout: 10_000 }).catch(() => {});
          clicked = true;
          log('clicked first button on page (no phrase matched)');
        }
      }
      if (!clicked) log('WARNING: sleep screen detected but no control to click');

      // Wait for the sleep screen to go away.
      const deadline = Date.now() + BOOT_TIMEOUT_MS;
      while (Date.now() < deadline) {
        await page.waitForTimeout(10_000);
        if (!(await looksAsleep(page))) break;
      }
      asleep = await looksAsleep(page);
      log(`still asleep after wait: ${asleep}`);
    }

    // Hold the websocket open so the visit registers as real traffic.
    await page.waitForTimeout(SETTLE_MS);
    await snap(page, name, '2-final');
    fs.writeFileSync(path.join(OUT_DIR, `${name}__final.txt`), await pageText(page));

    return { url, ok: !asleep, state: asleep ? 'STILL ASLEEP' : 'up' };
  } catch (err) {
    await snap(page, name, '3-error');
    return { url, ok: false, state: err.message.split('\n')[0] };
  } finally {
    await context.close();
  }
}

(async () => {
  if (APPS.length === 0) {
    console.error('No URLs given. Set APP_URLS to a whitespace- or comma-separated list.');
    process.exit(1);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const browser = await chromium.launch();
  const results = [];
  for (const url of APPS) {
    results.push(await visit(browser, url));
  }
  await browser.close();

  console.log('\n--- summary ---');
  for (const r of results) {
    console.log(`${r.ok ? 'ok  ' : 'FAIL'}  ${r.state.padEnd(16)}  ${r.url}`);
  }

  const failed = results.filter((r) => !r.ok);
  if (failed.length) {
    console.error(`\n${failed.length} of ${results.length} app(s) did not come up.`);
    process.exit(1);
  }
  console.log(`\nAll ${results.length} app(s) up.`);
})();
