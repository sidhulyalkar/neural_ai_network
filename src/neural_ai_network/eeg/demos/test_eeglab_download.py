#!/usr/bin/env python
# test_eeglab_download.py

import os
import glob
import zipfile
import requests
from tqdm import tqdm

def download_eeglab_sample(dataset_name):
    """
    Download and extract an EEGLAB sample dataset with careful error handling.
    
    Args:
        dataset_name: Name of the sample dataset
        
    Returns:
        Path to downloaded dataset or None if failed
    """
    print(f"Downloading EEGLAB sample dataset: {dataset_name}")
    
    # Define URLs for different sample datasets
    urls = {
        "eeglab_sample": "https://sccn.ucsd.edu/mediawiki/images/9/9c/Eeglab_sample_data.zip",
        "eeglab_sleep_sample": "https://sccn.ucsd.edu/mediawiki/images/3/3e/Sleep_data.zip",
        "eeglab_erp_sample": "https://sccn.ucsd.edu/mediawiki/images/5/5e/EEG_data_ch32.zip"
    }
    
    if dataset_name not in urls:
        print(f"Unknown dataset name: {dataset_name}")
        print(f"Available datasets: {list(urls.keys())}")
        return None
    
    url = urls[dataset_name]
    
    # Create directories
    cache_dir = "./data/cache/eeg"
    output_dir = f"./data/raw/eeg/eeglab/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download file path
    zip_path = os.path.join(cache_dir, f"{dataset_name}.zip")
    
    # Download the file
    try:
        # Check if file already exists
        if os.path.exists(zip_path):
            print(f"Removing existing file: {zip_path}")
            os.remove(zip_path)  # Remove potentially corrupted file
        
        print(f"Downloading from URL: {url}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc=f"Downloading {dataset_name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
        
        print(f"Download completed: {zip_path}")
        
        # Verify it's a zip file
        if not zipfile.is_zipfile(zip_path):
            print(f"Error: Downloaded file is not a valid ZIP file: {zip_path}")
            return None
        
        # Extract the zip file
        print(f"Extracting to: {output_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Verify extraction
        files = glob.glob(os.path.join(output_dir, "*"))
        print(f"Extracted {len(files)} files/directories to {output_dir}")
        
        return output_dir
    
    except Exception as e:
        print(f"Error downloading/extracting dataset: {e}")
        return None

def main():
    # Test downloading EEGLAB sample data
    dataset_dir = download_eeglab_sample("eeglab_sample")
    
    if dataset_dir:
        print(f"Successfully downloaded and extracted to: {dataset_dir}")
        
        # List files
        files = glob.glob(os.path.join(dataset_dir, "**/*.*"), recursive=True)
        print(f"Found {len(files)} files:")
        for file in files[:10]:  # Show first 10 files
            print(f"- {os.path.basename(file)}")
        
        if len(files) > 10:
            print(f"...and {len(files) - 10} more")
    else:
        print("Failed to download dataset")

if __name__ == "__main__":
    main()