# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 19:32:23 2025

@author: kumab
"""

"""
Standalone Data Bundle Downloader for PyPSA-Ethiopia
Extracted from PyPSA-Earth retrieve_databundle_light.py to work without Snakemake
"""

import os
import re
import datetime as dt
import requests
import zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import yaml
from tqdm import tqdm
from urllib.request import urlretrieve
import logging

class DataBundleDownloader:
    """Standalone data bundle downloader for PyPSA-Ethiopia"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ethiopia-specific configuration
        self.ethiopia_config = {
            'countries': ['ET'],  # Ethiopia country code
            'tutorial': False,
            'data_dir': self.base_dir / "data",
            'cutouts_dir': self.base_dir / "cutouts",
            'resources_dir': self.base_dir / "resources"
        }
        
        # Create necessary directories
        for dir_path in [self.ethiopia_config['data_dir'], 
                        self.ethiopia_config['cutouts_dir'], 
                        self.ethiopia_config['resources_dir']]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(self, block_num, block_size, total_size):
        """Progress hook for urlretrieve"""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
    
    def download_file(self, url, filepath, headers=None):
        """Download a file from URL with progress bar"""
        try:
            if headers:
                # Use requests for custom headers
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
            else:
                # Use urlretrieve for simple downloads
                urlretrieve(url, filepath, reporthook=self.progress_hook)
            
            print()  # New line after progress
            return True
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_path}: {e}")
            return False
    
    def download_zenodo_bundle(self, bundle_name, zenodo_url, destination):
        """Download and extract bundle from Zenodo"""
        self.logger.info(f"Downloading {bundle_name} from Zenodo...")
        
        temp_file = self.base_dir / f"{bundle_name}_temp.zip"
        
        # Download
        if self.download_file(zenodo_url, temp_file):
            # Extract
            if self.extract_zip(temp_file, destination):
                # Clean up
                temp_file.unlink()
                self.logger.info(f"Successfully downloaded and extracted {bundle_name}")
                return True
            else:
                temp_file.unlink()
                return False
        return False
    
    def download_hydrobasins(self, level=6):
        """Download HydroBASINS data for Africa"""
        base_url = "https://data.hydrosheds.org/file/HydroBASINS/standard/"
        regions = ["af"]  # Africa region for Ethiopia
        
        hydrobasins_dir = self.ethiopia_config['data_dir'] / "hydrobasins"
        hydrobasins_dir.mkdir(exist_ok=True)
        
        all_files = []
        
        for region in regions:
            filename = f"hybas_{region}_lev{level:02d}_v1c.zip"
            url = base_url + filename
            filepath = hydrobasins_dir / filename
            
            self.logger.info(f"Downloading HydroBASINS {region} level {level}...")
            
            if self.download_file(url, filepath, headers={"User-agent": "Mozilla/5.0"}):
                # Extract
                if self.extract_zip(filepath, hydrobasins_dir):
                    all_files.append(hydrobasins_dir / f"hybas_{region}_lev{level:02d}_v1c.shp")
                    filepath.unlink()  # Remove zip after extraction
        
        # Merge all regional files into a single world file
        if all_files:
            self.merge_hydrobasins(all_files, hydrobasins_dir / "hybas_world.shp")
            return True
        return False
    
    def merge_hydrobasins(self, shp_files, output_file):
        """Merge multiple hydrobasins shapefiles into one"""
        self.logger.info("Merging HydroBASINS files...")
        
        gdfs = []
        for shp_file in shp_files:
            if shp_file.exists():
                gdf = gpd.read_file(shp_file)
                gdfs.append(gdf)
        
        if gdfs:
            merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
            merged_gdf = merged_gdf.drop_duplicates(subset="HYBAS_ID", ignore_index=True)
            merged_gdf.to_file(output_file, driver="ESRI Shapefile")
            self.logger.info(f"Merged HydroBASINS saved to {output_file}")
    
    def download_ethiopia_essential_data(self):
        """Download essential data bundles for Ethiopia"""
        
        bundles = {
            # Core geographical data
            "africa_data": {
                "url": "https://sandbox.zenodo.org/records/13539/files/bundle_data_earth.zip?download=1",
                "destination": self.ethiopia_config['data_dir'],
                "description": "Core geographical data for Africa including Ethiopia"
            },
            
            # Land cover and protected areas
            "natura_protected": {
                "url": "https://sandbox.zenodo.org/records/13539/files/natura_wpda_CC0.zip?download=1", 
                "destination": self.ethiopia_config['data_dir'] / "natura",
                "description": "Protected areas and natura data"
            },
            
            # Weather data cutouts for Ethiopia region
            "africa_cutouts": {
                "url": "https://sandbox.zenodo.org/records/13539/files/bundle_cutouts_africa.zip?download=1",
                "destination": self.ethiopia_config['cutouts_dir'],
                "description": "Weather data cutouts for Africa"
            }
        }
        
        success_count = 0
        total_count = len(bundles) + 1  # +1 for hydrobasins
        
        # Download each bundle
        for bundle_name, bundle_info in bundles.items():
            destination = Path(bundle_info['destination'])
            destination.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📦 {bundle_info['description']}")
            if self.download_zenodo_bundle(bundle_name, bundle_info['url'], destination):
                success_count += 1
                print(f"✅ {bundle_name} downloaded successfully")
            else:
                print(f"❌ Failed to download {bundle_name}")
        
        # Download HydroBASINS separately
        print(f"\n🌊 Downloading HydroBASINS data...")
        if self.download_hydrobasins():
            success_count += 1
            print(f"✅ HydroBASINS downloaded successfully")
        else:
            print(f"❌ Failed to download HydroBASINS")
        
        print(f"\n📊 Downloaded {success_count}/{total_count} data bundles")
        return success_count == total_count
    
    def check_data_availability(self):
        """Check which data files are already available"""
        essential_files = [
            self.ethiopia_config['data_dir'] / "eez" / "eez_v11.gpkg",
            self.ethiopia_config['data_dir'] / "gebco" / "GEBCO_2021_TID.nc",
            self.ethiopia_config['data_dir'] / "copernicus" / "PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif",
            self.ethiopia_config['data_dir'] / "ssp2-2.6" / "2030" / "era5_2013" / "Africa.nc",
            self.ethiopia_config['data_dir'] / "hydrobasins" / "hybas_world.shp",
            self.ethiopia_config['cutouts_dir'] / "cutout-2013-era5.nc"
        ]
        
        available_files = []
        missing_files = []
        
        for file_path in essential_files:
            if file_path.exists():
                available_files.append(file_path.name)
            else:
                missing_files.append(file_path.name)
        
        return available_files, missing_files
    
    def get_download_status(self):
        """Get detailed status of data downloads"""
        available, missing = self.check_data_availability()
        
        status = {
            'total_files': len(available) + len(missing),
            'available_files': len(available),
            'missing_files': len(missing),
            'completion_percentage': len(available) / (len(available) + len(missing)) * 100,
            'available_list': available,
            'missing_list': missing,
            'ready_for_modeling': len(missing) == 0
        }
        
        return status


# Helper functions for integration with existing app
def create_ethiopia_data_downloader():
    """Factory function to create downloader instance"""
    return DataBundleDownloader()

def download_ethiopia_data_bundle():
    """Main function to download all Ethiopia data"""
    downloader = create_ethiopia_data_downloader()
    return downloader.download_ethiopia_essential_data()

def check_ethiopia_data_status():
    """Check status of Ethiopia data"""
    downloader = create_ethiopia_data_downloader()
    return downloader.get_download_status()


if __name__ == "__main__":
    # CLI interface for standalone usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PyPSA-Ethiopia data bundles")
    parser.add_argument("--check", action="store_true", help="Check data availability")
    parser.add_argument("--download", action="store_true", help="Download missing data")
    args = parser.parse_args()
    
    downloader = create_ethiopia_data_downloader()
    
    if args.check:
        status = downloader.get_download_status()
        print(f"\n📊 Data Status:")
        print(f"Available: {status['available_files']}/{status['total_files']} files")
        print(f"Completion: {status['completion_percentage']:.1f}%")
        print(f"Ready for modeling: {'Yes' if status['ready_for_modeling'] else 'No'}")
        
        if status['missing_files'] > 0:
            print(f"\n❌ Missing files:")
            for file in status['missing_list']:
                print(f"  - {file}")
    
    if args.download:
        success = downloader.download_ethiopia_essential_data()
        if success:
            print(f"\n🎉 All data downloaded successfully!")
        else:
            print(f"\n⚠️ Some downloads failed. Check logs for details.")