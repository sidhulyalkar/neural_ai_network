# fmri_data_loader.py
import os
import glob
import logging
import nibabel as nib
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


class FMRIDataLoader:
    """
    Data loader for fMRI data from various sources including OpenNeuro,
    sample datasets, and standard databases.
    
    This class handles downloading, extracting, and loading fMRI data in various
    formats, preparing it for preprocessing and analysis.
    """
    
    def __init__(self, data_dir: str = "./data/raw/fmri", 
                 cache_dir: str = "./data/cache/fmri"):
        """
        Initialize the fMRI data loader.
        
        Args:
            data_dir: Directory for storing raw fMRI data
            cache_dir: Directory for caching downloaded files
        """
        self.logger = self._setup_logging()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary of available sample datasets
        self.sample_datasets = {
            "auditory": {
                "url": "https://openneuro.org/crn/datasets/ds000116/snapshots/1.0.0/files/sub-01:download",
                "description": "Auditory task fMRI dataset (single subject)"
            },
            "visual": {
                "url": "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.4/files/sub-01:download",
                "description": "Visual faces task fMRI dataset (single subject)"
            },
            "resting_state": {
                "url": "https://openneuro.org/crn/datasets/ds000102/snapshots/00001/files/sub-01:download",
                "description": "Resting-state fMRI dataset (single subject)"
            },
            "ds002": {
                "url": "https://openneuro.org/crn/datasets/ds002/snapshots/00001/files/CHANGES:download",
                "base_url": "https://openneuro.org/crn/datasets/ds002/snapshots/00001/files",
                "description": "Classification learning task fMRI dataset (OpenfMRI ds002)"
            }
        }
        
        self.logger.info("fMRI Data Loader initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("FMRIDataLoader")
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
            "sample": self.sample_datasets
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
        
        # Check for downloaded sample datasets
        for sample_name in self.sample_datasets.keys():
            sample_dir = os.path.join(self.data_dir, sample_name)
            if os.path.exists(sample_dir):
                # Count NIfTI files
                nifti_files = glob.glob(os.path.join(sample_dir, "**/*.nii*"), recursive=True)
                local_datasets[sample_name] = {
                    "path": sample_dir,
                    "file_count": len(nifti_files)
                }
        
        # Check for BIDS datasets
        bids_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d)) 
                    and os.path.exists(os.path.join(self.data_dir, d, "dataset_description.json"))]
        
        for bids_dir in bids_dirs:
            full_path = os.path.join(self.data_dir, bids_dir)
            # Count subjects
            subjects = [d for d in os.listdir(full_path) if d.startswith("sub-")]
            
            # Count functional and anatomical files
            func_files = glob.glob(os.path.join(full_path, "**/*_bold.nii*"), recursive=True)
            anat_files = glob.glob(os.path.join(full_path, "**/*_T1w.nii*"), recursive=True)
            
            local_datasets[bids_dir] = {
                "path": full_path,
                "subjects": len(subjects),
                "functional_files": len(func_files),
                "anatomical_files": len(anat_files)
            }
        
        # Check for other fMRI files
        other_nifti = glob.glob(os.path.join(self.data_dir, "*.nii*"))
        if other_nifti:
            local_datasets["other_nifti_files"] = {
                "path": self.data_dir,
                "file_count": len(other_nifti)
            }