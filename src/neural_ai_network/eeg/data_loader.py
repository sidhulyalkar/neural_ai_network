# eeg_data_loader.py
import os
import glob
import logging
import mne
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import zipfile
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm


class EEGDataLoader:
    """
    Data loader for EEG data from various sources including Temple University
    EEG Corpus and EEGLAB sample datasets.
    
    This class handles downloading, extracting, and loading EEG data in various
    formats, preparing it for preprocessing and analysis.
    """
    
    def __init__(self, data_dir: str = "./data/raw/eeg", 
                 cache_dir: str = "./data/cache/eeg"):
        """
        Initialize the EEG data loader.
        
        Args:
            data_dir: Directory for storing raw EEG data
            cache_dir: Directory for caching downloaded files
        """
        self.logger = self._setup_logging()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary of available EEGLAB sample datasets
        self.eeglab_samples = {
            "eeglab_sample": {
                "url": "https://sccn.ucsd.edu/mediawiki/images/9/9c/Eeglab_sample_data.zip",
                "description": "Standard EEGLAB sample dataset (continuous)"
            },
            "eeglab_sleep_sample": {
                "url": "https://sccn.ucsd.edu/mediawiki/images/3/3e/Sleep_data.zip",
                "description": "Sleep stage data sample"
            },
            "eeglab_erp_sample": {
                "url": "https://sccn.ucsd.edu/mediawiki/images/5/5e/EEG_data_ch32.zip",
                "description": "32-channel ERP data"
            }
        }
        
        # Temple University EEG Corpus info
        self.temple_eeg_info = {
            "base_url": "https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/",
            "versions": ["v1.2.0", "v1.1.0", "v1.0.0"],
            "description": "Clinical EEG data from Temple University Hospital"
        }
        
        self.logger.info("EEG Data Loader initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("EEGDataLoader")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def list_available_datasets(self) -> Dict:
        """
        List all available datasets.
        
        Returns:
            Dictionary of available datasets with descriptions
        """
        datasets = {
            "eeglab": self.eeglab_samples,
            "temple_eeg": {
                "versions": self.temple_eeg_info["versions"],
                "description": self.temple_eeg_info["description"]
            }
        }
        
        # Add any local datasets already downloaded
        local_datasets = self._scan_local_datasets()
        if local_datasets:
            datasets["local"] = local_datasets
        
        return datasets
    
    def _scan_local_datasets(self) -> Dict:
        """
        Scan for local datasets already in the data directory.
        
        Returns:
            Dictionary of local datasets with file counts
        """
        local_datasets = {}
        
        # Check for EEGLAB datasets
        eeglab_dir = os.path.join(self.data_dir, "eeglab")
        if os.path.exists(eeglab_dir):
            for sample_name in self.eeglab_samples.keys():
                sample_dir = os.path.join(eeglab_dir, sample_name)
                if os.path.exists(sample_dir):
                    files = glob.glob(os.path.join(sample_dir, "*.set")) + \
                            glob.glob(os.path.join(sample_dir, "*.edf"))
                    local_datasets[f"eeglab_{sample_name}"] = {
                        "path": sample_dir,
                        "file_count": len(files)
                    }
        
        # Check for Temple EEG data
        temple_dir = os.path.join(self.data_dir, "temple_eeg")
        if os.path.exists(temple_dir):
            for version in self.temple_eeg_info["versions"]:
                version_dir = os.path.join(temple_dir, version)
                if os.path.exists(version_dir):
                    edf_files = glob.glob(os.path.join(version_dir, "**/*.edf"), recursive=True)
                    local_datasets[f"temple_eeg_{version}"] = {
                        "path": version_dir,
                        "file_count": len(edf_files)
                    }
        
        # Check for other EEG files
        other_files = glob.glob(os.path.join(self.data_dir, "*.edf")) + \
                     glob.glob(os.path.join(self.data_dir, "*.bdf")) + \
                     glob.glob(os.path.join(self.data_dir, "*.set"))
        if other_files:
            local_datasets["other_eeg_files"] = {
                "path": self.data_dir,
                "file_count": len(other_files)
            }
        
        return local_datasets
    
    def download_eeglab_sample(self, sample_name: str) -> str:
        """
        Download an EEGLAB sample dataset.
        
        Args:
            sample_name: Name of the sample dataset
            
        Returns:
            Path to downloaded dataset
        """
        if sample_name not in self.eeglab_samples:
            raise ValueError(f"Unknown EEGLAB sample: {sample_name}")
        
        sample_info = self.eeglab_samples[sample_name]
        url = sample_info["url"]
        
        # Create output directory
        output_dir = os.path.join(self.data_dir, "eeglab", sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Download and extract
        self.logger.info(f"Downloading EEGLAB sample: {sample_name}")
        zip_path = self._download_file(url, os.path.join(self.cache_dir, f"{sample_name}.zip"))
        
        self.logger.info(f"Extracting to {output_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Check for extracted files
        files = glob.glob(os.path.join(output_dir, "**/*.*"), recursive=True)
        self.logger.info(f"Extracted {len(files)} files")
        
        return output_dir
    
    def download_temple_eeg(self, version: str = "v1.2.0", subset: str = "sample") -> str:
        """
        Download Temple University EEG Corpus data.
        
        Note: This function provides instructions for downloading, as the full
        dataset requires authentication and is very large. It downloads a
        small publicly available sample if specified.
        
        Args:
            version: Version of the dataset
            subset: Subset to download ('sample', 'full', or specific subset name)
            
        Returns:
            Path to download instructions or downloaded data
        """
        if version not in self.temple_eeg_info["versions"]:
            raise ValueError(f"Unknown Temple EEG version: {version}")
        
        # Create output directory
        output_dir = os.path.join(self.data_dir, "temple_eeg", version)
        os.makedirs(output_dir, exist_ok=True)
        
        # For the sample subset, download directly
        if subset.lower() == "sample":
            self.logger.info(f"Downloading Temple EEG sample ({version})")
            
            # Sample URL for public access
            sample_url = f"{self.temple_eeg_info['base_url']}{version}/tuh_eeg_sample.tar.gz"
            
            try:
                tar_path = self._download_file(sample_url, os.path.join(self.cache_dir, f"tuh_eeg_sample_{version}.tar.gz"))
                
                self.logger.info(f"Extracting to {output_dir}")
                with tarfile.open(tar_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(output_dir)
                
                # Check for extracted files
                edf_files = glob.glob(os.path.join(output_dir, "**/*.edf"), recursive=True)
                self.logger.info(f"Extracted {len(edf_files)} EDF files")
                
                return output_dir
            except Exception as e:
                self.logger.error(f"Error downloading sample: {e}")
                
                # Create instructions file
                instructions_path = os.path.join(output_dir, "download_instructions.txt")
                with open(instructions_path, 'w') as f:
                    f.write(f"Temple University EEG Corpus ({version})\n")
                    f.write("="*50 + "\n\n")
                    f.write("This dataset requires registration and authentication.\n\n")
                    f.write("To download the full dataset or specific subsets:\n")
                    f.write("1. Register at https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php\n")
                    f.write("2. Once approved, download from https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/\n")
                    f.write("3. Extract the downloaded files to this directory\n")
                
                return instructions_path
        else:
            # Create instructions file for full dataset or specific subsets
            instructions_path = os.path.join(output_dir, "download_instructions.txt")
            with open(instructions_path, 'w') as f:
                f.write(f"Temple University EEG Corpus ({version}) - {subset} subset\n")
                f.write("="*50 + "\n\n")
                f.write("This dataset requires registration and authentication.\n\n")
                f.write("To download the full dataset or specific subsets:\n")
                f.write("1. Register at https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php\n")
                f.write("2. Once approved, download from https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/\n")
                f.write("3. Extract the downloaded files to this directory\n")
            
            return instructions_path
    
    def load_eeg_file(self, file_path: str) -> mne.io.Raw:
        """
        Load an EEG file using MNE.
        
        Args:
            file_path: Path to EEG file
            
        Returns:
            MNE Raw object containing the EEG data
        """
        self.logger.info(f"Loading EEG file: {file_path}")
        
        # Determine file type and use appropriate loader
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.edf':
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif ext == '.bdf':
            raw = mne.io.read_raw_bdf(file_path, preload=True)
        elif ext == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        elif ext in ['.fif', '.fiff']:
            raw = mne.io.read_raw_fif(file_path, preload=True)
        elif ext == '.vhdr':
            raw = mne.io.read_raw_brainvision(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        self.logger.info(f"Loaded data: {len(raw.ch_names)} channels, {raw.n_times} samples at {raw.info['sfreq']} Hz")
        return raw
    
    def load_eeglab_dataset(self, sample_name: str, file_index: int = 0) -> mne.io.Raw:
        """
        Load a specific file from an EEGLAB sample dataset.
        
        Args:
            sample_name: Name of the sample dataset
            file_index: Index of the file to load (if multiple files present)
            
        Returns:
            MNE Raw object containing the EEG data
        """
        dataset_dir = os.path.join(self.data_dir, "eeglab", sample_name)
        
        if not os.path.exists(dataset_dir):
            self.logger.info(f"Dataset not found locally, downloading: {sample_name}")
            dataset_dir = self.download_eeglab_sample(sample_name)
        
        # Find EEG files
        set_files = glob.glob(os.path.join(dataset_dir, "**/*.set"), recursive=True)
        edf_files = glob.glob(os.path.join(dataset_dir, "**/*.edf"), recursive=True)
        
        all_files = set_files + edf_files
        if not all_files:
            raise ValueError(f"No EEG files found in {dataset_dir}")
        
        # Select file by index
        if file_index >= len(all_files):
            raise ValueError(f"File index {file_index} out of range (0-{len(all_files)-1})")
        
        selected_file = all_files[file_index]
        self.logger.info(f"Selected file: {selected_file}")
        
        # Load the file
        return self.load_eeg_file(selected_file)
    
    def load_temple_eeg_file(self, version: str = "v1.2.0", 
                            file_pattern: str = None, 
                            random_selection: bool = False) -> mne.io.Raw:
        """
        Load an EEG file from the Temple University EEG Corpus.
        
        Args:
            version: Version of the dataset
            file_pattern: Pattern to match specific files
            random_selection: Whether to randomly select a file
            
        Returns:
            MNE Raw object containing the EEG data
        """
        dataset_dir = os.path.join(self.data_dir, "temple_eeg", version)
        
        if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
            self.logger.info(f"Dataset not found locally, downloading sample: {version}")
            dataset_dir = self.download_temple_eeg(version, "sample")
        
        # Find EEG files
        if file_pattern:
            edf_files = glob.glob(os.path.join(dataset_dir, "**", file_pattern), recursive=True)
        else:
            edf_files = glob.glob(os.path.join(dataset_dir, "**/*.edf"), recursive=True)
        
        if not edf_files:
            raise ValueError(f"No EDF files found in {dataset_dir}")
        
        # Select file
        if random_selection:
            selected_file = np.random.choice(edf_files)
        else:
            selected_file = edf_files[0]
        
        self.logger.info(f"Selected file: {selected_file}")
        
        # Load the file
        return self.load_eeg_file(selected_file)
    
    def load_file_batch(self, directory: str, pattern: str = "*.edf", 
                       limit: int = None) -> List[Dict]:
        """
        Load a batch of EEG files from a directory.
        
        Args:
            directory: Directory containing EEG files
            pattern: File pattern to match
            limit: Maximum number of files to load
            
        Returns:
            List of dictionaries with file info and MNE Raw objects
        """
        self.logger.info(f"Loading batch of files from {directory} matching {pattern}")
        
        # Find matching files
        files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        
        if not files:
            raise ValueError(f"No files matching {pattern} found in {directory}")
        
        # Limit number of files if specified
        if limit and limit < len(files):
            files = files[:limit]
        
        # Load each file
        batch = []
        for file_path in tqdm(files, desc="Loading files"):
            try:
                raw = self.load_eeg_file(file_path)
                
                # Extract basic info
                info = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "n_channels": len(raw.ch_names),
                    "duration": raw.times[-1],
                    "sfreq": raw.info["sfreq"],
                    "raw": raw
                }
                
                batch.append(info)
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(batch)} files successfully")
        return batch
    
    def _download_file(self, url: str, output_path: str) -> str:
        """
        Download a file from a URL and show progress.
        
        Args:
            url: URL to download
            output_path: Path to save the file
            
        Returns:
            Path to downloaded file
        """
        if os.path.exists(output_path):
            self.logger.info(f"File already exists: {output_path}")
            return output_path
        
        self.logger.info(f"Downloading {url} to {output_path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        
        return output_path


# Example usage
if __name__ == "__main__":
    loader = EEGDataLoader()
    
    # List available datasets
    datasets = loader.list_available_datasets()
    print("Available datasets:")
    for category, category_data in datasets.items():
        print(f"\n{category.upper()}:")
        for name, info in category_data.items():
            if isinstance(info, dict) and "description" in info:
                print(f"  - {name}: {info['description']}")
            elif isinstance(info, dict) and "file_count" in info:
                print(f"  - {name}: {info['file_count']} files at {info['path']}")
            else:
                print(f"  - {name}")
    
    # Example: Download and load EEGLAB sample
    try:
        raw = loader.load_eeglab_dataset("eeglab_sample")
        print(f"\nLoaded EEGLAB sample: {len(raw.ch_names)} channels, {raw.n_times} samples")
    except Exception as e:
        print(f"Error loading EEGLAB sample: {e}")
    
    # Example: Download and load Temple EEG sample
    try:
        raw = loader.load_temple_eeg_file()
        print(f"\nLoaded Temple EEG sample: {len(raw.ch_names)} channels, {raw.n_times} samples")
    except Exception as e:
        print(f"Error loading Temple EEG sample: {e}")