/**
 * Wakes sleeping Streamlit Community Cloud apps.
 *
 * Community Cloud sleeps an app after 12 hours without traffic. A plain HTTP
 * request does not reset that timer -- Streamlit counts the websocket session
 * a real browser opens. So this loads each app in headless Chromium, clicks
 * the "Yes, get this app back up!" button when it appears, and waits for the
 * app to finish booting.
 */

const { chromium } = require('playwright');

const APPS = process.env.APP_URLS
  ? process.env.APP_URLS.split(/[\s,]+/).filter(Boolean)
  : [];

const WAKE_BUTTON = /get this app back up/i;
const APP_READY = '[data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp';

const BOOT_TIMEOUT_MS = 240_000; // cold starts can install dependencies
const NAV_TIMEOUT_MS = 90_000;

async function visit(browser, url) {
  const context = await browser.newContext({
    userAgent:
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36',
    viewport: { width: 1280, height: 900 },
  });
  const page = await context.newPage();

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS });

    const wakeButton = page.getByText(WAKE_BUTTON).first();
    const wasAsleep = await wakeButton
      .waitFor({ state: 'visible', timeout: 10_000 })
      .then(() => true)
      .catch(() => false);

    if (wasAsleep) {
      await wakeButton.click();
    }

    await page.waitForSelector(APP_READY, { timeout: BOOT_TIMEOUT_MS });

    // Hold the websocket open briefly so the session registers as real traffic.
    await page.waitForTimeout(8_000);

    return { url, ok: true, state: wasAsleep ? 'woken' : 'already awake' };
  } catch (err) {
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

  const browser = await chromium.launch();
  const results = [];

  for (const url of APPS) {
    const result = await visit(browser, url);
    results.push(result);
    console.log(`${result.ok ? 'ok  ' : 'FAIL'}  ${result.state.padEnd(14)}  ${url}`);
  }

  await browser.close();

  const failed = results.filter((r) => !r.ok);
  if (failed.length) {
    console.error(`\n${failed.length} of ${results.length} app(s) did not come up.`);
    process.exit(1);
  }
  console.log(`\nAll ${results.length} app(s) up.`);
})();
