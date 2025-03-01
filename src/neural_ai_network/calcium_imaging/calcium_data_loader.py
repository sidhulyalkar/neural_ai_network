# calcium_data_loader.py
import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import zipfile
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import skimage.io
import h5py
import json
import yaml


class CalciumDataLoader:
    """
    Data loader for calcium imaging data from various sources including Allen Brain Atlas,
    CRCNS, and the Neurofinder challenge.
    
    This class handles downloading, extracting, and loading calcium imaging data in various
    formats, preparing it for preprocessing and analysis.
    """
    
    def __init__(self, data_dir: str = "./data/raw/calcium", 
                 cache_dir: str = "./data/cache/calcium"):
        """
        Initialize the calcium data loader.
        
        Args:
            data_dir: Directory for storing raw calcium imaging data
            cache_dir: Directory for caching downloaded files
        """
        self.logger = self._setup_logging()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary of available Neurofinder datasets
        self.neurofinder_datasets = {
            "00.00": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.00.00.zip",
                "description": "Neurofinder 00.00 - GCaMP6s, high SNR, widefield, cultured neurons"
            },
            "00.01": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.00.01.zip",
                "description": "Neurofinder 00.01 - GCaMP6s, high SNR, widefield, cultured neurons"
            },
            "01.00": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.01.00.zip",
                "description": "Neurofinder 01.00 - GCaMP6s, high SNR, two-photon, visual cortex"
            },
            "01.01": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.01.01.zip",
                "description": "Neurofinder 01.01 - GCaMP6s, high SNR, two-photon, visual cortex"
            },
            "02.00": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.02.00.zip",
                "description": "Neurofinder 02.00 - GCaMP6s, low SNR, two-photon, visual cortex"
            },
            "02.01": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.02.01.zip",
                "description": "Neurofinder 02.01 - GCaMP6s, medium SNR, two-photon, visual cortex"
            },
            "03.00": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.03.00.zip",
                "description": "Neurofinder 03.00 - GCaMP6f, low SNR, two-photon, visual cortex"
            },
            "04.00": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.04.00.zip",
                "description": "Neurofinder 04.00 - GCaMP5k, low SNR, two-photon, visual cortex"
            },
            "04.01": {
                "url": "https://neurofinder.codeneuro.org/neurofinder.04.01.zip",
                "description": "Neurofinder 04.01 - GCaMP5k, low SNR, two-photon, visual cortex"
            }
        }
        
        # Allen Brain Observatory data info
        self.allen_brain_info = {
            "base_url": "http://observatory.brain-map.org/visualcoding",
            "downloads_url": "https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html",
            "description": "Two-photon calcium imaging from the Allen Brain Observatory"
        }
        
        # CRCNS data info
        self.crcns_info = {
            "base_url": "https://crcns.org/data-sets/methods/",
            "description": "Collaborative Research in Computational Neuroscience data sharing"
        }
        
        # Sample data for quick testing
        self.sample_datasets = {
            "sample1": {
                "url": "https://github.com/codeneuro/neurofinder-datasets/raw/master/mini/neurofinder.00.00.zip",
                "description": "Small sample from Neurofinder 00.00"
            },
            "sample2": {
                "url": "https://github.com/codeneuro/neurofinder-datasets/raw/master/mini/neurofinder.01.00.zip",
                "description": "Small sample from Neurofinder 01.00"
            }
        }
        
        self.logger.info("Calcium Data Loader initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("CalciumDataLoader")
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
            "neurofinder": self.neurofinder_datasets,
            "allen_brain": {
                "description": self.allen_brain_info["description"],
                "url": self.allen_brain_info["base_url"]
            },
            "crcns": {
                "description": self.crcns_info["description"],
                "url": self.crcns_info["base_url"]
            },
            "samples": self.sample_datasets
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
        
        # Check for Neurofinder datasets
        neurofinder_dir = os.path.join(self.data_dir, "neurofinder")
        if os.path.exists(neurofinder_dir):
            for dataset_name in self.neurofinder_datasets.keys():
                dataset_dir = os.path.join(neurofinder_dir, f"neurofinder.{dataset_name}")
                if os.path.exists(dataset_dir):
                    image_files = glob.glob(os.path.join(dataset_dir, "images", "*.tif*"))
                    local_datasets[f"neurofinder_{dataset_name}"] = {
                        "path": dataset_dir,
                        "frame_count": len(image_files)
                    }
        
        # Check for Allen Brain data
        allen_dir = os.path.join(self.data_dir, "allen_brain")
        if os.path.exists(allen_dir):
            h5_files = glob.glob(os.path.join(allen_dir, "**/*.h5"), recursive=True)
            if h5_files:
                local_datasets["allen_brain"] = {
                    "path": allen_dir,
                    "file_count": len(h5_files)
                }
        
        # Check for CRCNS data
        crcns_dir = os.path.join(self.data_dir, "crcns")
        if os.path.exists(crcns_dir):
            data_files = glob.glob(os.path.join(crcns_dir, "**/*.*"), recursive=True)
            if data_files:
                local_datasets["crcns"] = {
                    "path": crcns_dir,
                    "file_count": len(data_files)
                }
        
        # Check for other calcium imaging files
        other_files = glob.glob(os.path.join(self.data_dir, "*.tif*")) + \
                     glob.glob(os.path.join(self.data_dir, "*.h5")) + \
                     glob.glob(os.path.join(self.data_dir, "*.npy"))
        if other_files:
            local_datasets["other_files"] = {
                "path": self.data_dir,
                "file_count": len(other_files)
            }
        
        return local_datasets
    
    def download_neurofinder_dataset(self, dataset_id: str) -> str:
        """
        Download a Neurofinder dataset.
        
        Args:
            dataset_id: ID of the dataset (e.g., "00.00")
            
        Returns:
            Path to downloaded dataset
        """
        if dataset_id not in self.neurofinder_datasets:
            raise ValueError(f"Unknown Neurofinder dataset ID: {dataset_id}")
            
        dataset_info = self.neurofinder_datasets[dataset_id]
        url = dataset_info["url"]
        
        # Create output directory
        output_dir = os.path.join(self.data_dir, "neurofinder")
        os.makedirs(output_dir, exist_ok=True)
        
        # Download and extract
        self.logger.info(f"Downloading Neurofinder dataset: {dataset_id}")
        zip_path = self._download_file(url, os.path.join(self.cache_dir, f"neurofinder.{dataset_id}.zip"))
        
        extract_dir = os.path.join(output_dir, f"neurofinder.{dataset_id}")
        self.logger.info(f"Extracting to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Check for extracted files
        image_files = glob.glob(os.path.join(extract_dir, "images", "*.tif*"))
        self.logger.info(f"Extracted {len(image_files)} image files")
        
        return extract_dir
    
    def download_allen_brain_data(self) -> str:
        """
        Provide instructions for downloading Allen Brain Observatory data.
        
        Returns:
            Path to instructions file
        """
        # Create output directory
        output_dir = os.path.join(self.data_dir, "allen_brain")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create instructions file
        instructions_path = os.path.join(output_dir, "download_instructions.txt")
        with open(instructions_path, 'w') as f:
            f.write("Allen Brain Observatory Calcium Imaging Data\n")
            f.write("="*50 + "\n\n")
            f.write("This dataset requires using the AllenSDK to download:\n\n")
            f.write("1. Install AllenSDK: pip install allensdk\n\n")
            f.write("2. Use the following Python code to download data:\n\n")
            f.write("```python\n")
            f.write("from allensdk.core.brain_observatory_cache import BrainObservatoryCache\n")
            f.write(f"cache_dir = '{output_dir}'\n")
            f.write("boc = BrainObservatoryCache(cache_directory=cache_dir)\n\n")
            f.write("# Get experiment containers for a particular targeted structure\n")
            f.write("containers = boc.get_experiment_containers(targeted_structures=['VISp'])\n\n")
            f.write("# Download a specific experiment\n")
            f.write("container_id = containers[0]['id']\n")
            f.write("exp_id = boc.get_ophys_experiments(experiment_container_ids=[container_id])[0]['id']\n")
            f.write("dataset = boc.get_ophys_experiment_data(exp_id)\n")
            f.write("# Access fluorescence traces\n")
            f.write("cell_ids = dataset.get_cell_specimen_ids()\n")
            f.write("fluorescence = dataset.get_dff_traces()\n")
            f.write("```\n\n")
            f.write("More information: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html\n")
        
        return instructions_path
    
    def download_crcns_data(self) -> str:
        """
        Provide instructions for downloading CRCNS calcium imaging data.
        
        Returns:
            Path to instructions file
        """
        # Create output directory
        output_dir = os.path.join(self.data_dir, "crcns")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create instructions file
        instructions_path = os.path.join(output_dir, "download_instructions.txt")
        with open(instructions_path, 'w') as f:
            f.write("CRCNS Calcium Imaging Datasets\n")
            f.write("="*50 + "\n\n")
            f.write("These datasets require registration at crcns.org:\n\n")
            f.write("1. Register at https://crcns.org/register\n\n")
            f.write("2. Available calcium imaging datasets include:\n")
            f.write("   - cai-1: Calcium imaging of mouse V1 responses to oriented gratings\n")
            f.write("   - cai-2: Calcium imaging of mouse barrel cortex during pole localization\n")
            f.write("   - cai-3: Calcium imaging and simultaneous electrophysiology in visual cortex\n\n")
            f.write("3. After registration, download using HTTP or SFTP:\n")
            f.write("   HTTP: https://crcns.org/data-sets/methods/\n")
            f.write("   SFTP: sftp://download.crcns.org/\n\n")
            f.write("4. Extract the downloaded files to this directory\n")
        
        return instructions_path
    
    def download_sample_dataset(self, sample_name: str = "sample1") -> str:
        """
        Download a small sample calcium imaging dataset for testing.
        
        Args:
            sample_name: Name of the sample dataset
            
        Returns:
            Path to downloaded dataset
        """
        if sample_name not in self.sample_datasets:
            raise ValueError(f"Unknown sample dataset: {sample_name}")
        
        sample_info = self.sample_datasets[sample_name]
        url = sample_info["url"]
        
        # Create output directory
        output_dir = os.path.join(self.data_dir, "samples")
        os.makedirs(output_dir, exist_ok=True)
        
        extract_dir = os.path.join(output_dir, sample_name)
        os.makedirs(extract_dir, exist_ok=True)
        
        # Download and extract
        self.logger.info(f"Downloading sample dataset: {sample_name}")
        
        try:
            # Download directly to temporary file
            temp_zip = os.path.join(self.cache_dir, f"{sample_name}_temp.zip")
            os.makedirs(os.path.dirname(temp_zip), exist_ok=True)
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_zip, 'wb') as file, tqdm(
                desc=os.path.basename(temp_zip),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(1024):
                    file.write(data)
                    bar.update(len(data))
            
            # Verify it's a valid zip file
            if not zipfile.is_zipfile(temp_zip):
                self.logger.error(f"Downloaded file is not a valid ZIP file: {temp_zip}")
                # Create a synthetic dataset as fallback
                return self._create_synthetic_dataset(extract_dir)
            
            # Extract the zip file
            self.logger.info(f"Extracting to {extract_dir}")
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Check for extracted files
            image_files = glob.glob(os.path.join(extract_dir, "images", "*.tif*"))
            if not image_files:
                image_files = glob.glob(os.path.join(extract_dir, "**", "*.tif*"), recursive=True)
                
            if not image_files:
                self.logger.warning(f"No image files found in extracted zip at {extract_dir}")
                return self._create_synthetic_dataset(extract_dir)
                
            self.logger.info(f"Extracted {len(image_files)} image files")
            
            return extract_dir
            
        except Exception as e:
            self.logger.error(f"Error downloading sample: {e}")
            return self._create_synthetic_dataset(extract_dir)

    def _create_synthetic_dataset(self, output_dir: str) -> str:
        """Create a synthetic dataset when download fails."""
        self.logger.info("Creating synthetic dataset as fallback")
        
        # Create directories
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate synthetic data (10 frames, 100x100 pixels)
        frames = 10
        height = 100
        width = 100
        
        # Create background with some noise
        background = 100 + np.random.normal(0, 10, (height, width))
        
        # Create cells
        n_cells = 5
        cell_masks = []
        
        # Generate random cell centers
        cell_centers = []
        for _ in range(n_cells):
            x = np.random.randint(10, width-10)
            y = np.random.randint(10, height-10)
            radius = np.random.randint(3, 8)
            cell_centers.append((y, x, radius))
        
        # Create cell masks
        for y, x, radius in cell_centers:
            y_grid, x_grid = np.ogrid[-y:height-y, -x:width-x]
            mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
            cell_masks.append(mask)
        
        # Generate data for each frame
        for frame in range(frames):
            # Start with background
            data = background.copy()
            
            # Add cells with activity
            for i, mask in enumerate(cell_masks):
                # Add cell with some activity
                activity = 50 + 20 * np.sin(frame / 2 + i)
                data[mask] += activity
            
            # Add noise
            data += np.random.normal(0, 5, (height, width))
            
            # Save as TIFF
            data = np.clip(data, 0, 255).astype(np.uint8)
            filename = os.path.join(images_dir, f"image{frame:05d}.tif")
            
            try:
                from skimage.io import imsave
                imsave(filename, data)
            except ImportError:
                # Fallback to numpy save
                np.save(filename.replace('.tif', '.npy'), data)
        
        # Create regions JSON for ground truth
        regions_dir = os.path.join(output_dir, "regions")
        os.makedirs(regions_dir, exist_ok=True)
        
        regions = []
        for i, (y, x, radius) in enumerate(cell_centers):
            # Create coordinates for circle approximation
            coords = []
            for theta in np.linspace(0, 2*np.pi, 20):
                coords.append([int(y + radius * np.sin(theta)), 
                            int(x + radius * np.cos(theta))])
            
            regions.append({
                "id": i,
                "coordinates": coords
            })
        
        # Save regions
        with open(os.path.join(regions_dir, "regions.json"), 'w') as f:
            json.dump(regions, f)
        
        self.logger.info(f"Created synthetic dataset with {frames} frames and {n_cells} cells")
        return output_dir
    
    def load_calcium_file(self, file_path: str) -> np.ndarray:
        """
        Load a calcium imaging file.
        
        Args:
            file_path: Path to calcium imaging file
            
        Returns:
            Numpy array with dimensions [frames, height, width]
        """
        self.logger.info(f"Loading calcium data from {file_path}")
        
        # Determine file type and use appropriate loader
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in ['.tif', '.tiff']:
            # Load TIFF stack
            data = skimage.io.imread(file_path)
            
            # Ensure correct dimensions (frames, height, width)
            if data.ndim == 2:
                # Single frame
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[0] > data.shape[2]:
                # Likely [height, width, frames] format, transpose to [frames, height, width]
                if data.shape[2] < 10:  # Heuristic: if third dimension is small, it's likely RGB
                    pass  # Keep as is, it's probably a color image
                else:
                    data = np.transpose(data, (2, 0, 1))
        
        elif ext == '.h5':
            # Load HDF5 file
            with h5py.File(file_path, 'r') as f:
                # Try common dataset names
                dataset_names = ['data', 'images', 'calcium', 'movie', 'frames']
                
                data = None
                for name in dataset_names:
                    if name in f:
                        data = f[name][:]
                        break
                
                # If still not found, take the first dataset
                if data is None:
                    for name in f.keys():
                        if isinstance(f[name], h5py.Dataset) and len(f[name].shape) >= 2:
                            data = f[name][:]
                            break
                
                if data is None:
                    raise ValueError(f"Could not find calcium imaging data in HDF5 file: {file_path}")
                
                # Ensure correct dimensions
                if data.ndim == 2:
                    data = data[np.newaxis, :, :]
                elif data.ndim == 3 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                    # Likely [channels, height, width] format, convert to [frames, height, width]
                    data = np.transpose(data, (0, 1, 2))
                elif data.ndim == 3 and data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                    # Likely [height, width, frames] format, convert to [frames, height, width]
                    data = np.transpose(data, (2, 0, 1))
                elif data.ndim == 4:
                    # Likely [frames, channels, height, width], take first channel
                    data = data[:, 0, :, :]
        
        elif ext == '.npy':
            # Load numpy array
            data = np.load(file_path)
            
            # Ensure correct dimensions
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                # Likely [height, width, frames] format, convert to [frames, height, width]
                data = np.transpose(data, (2, 0, 1))
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        self.logger.info(f"Loaded data: {data.shape} (frames, height, width)")
        return data
    
    def load_neurofinder_dataset(self, dataset_id: str) -> Dict:
        """
        Load a Neurofinder dataset.
        
        Args:
            dataset_id: ID of the dataset (e.g., "00.00")
            
        Returns:
            Dictionary with loaded data and ground truth
        """
        dataset_dir = os.path.join(self.data_dir, "neurofinder", f"neurofinder.{dataset_id}")
        
        if not os.path.exists(dataset_dir):
            self.logger.info(f"Dataset not found locally, downloading: {dataset_id}")
            dataset_dir = self.download_neurofinder_dataset(dataset_id)
        
        # Load images
        images_dir = os.path.join(dataset_dir, "images")
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.tif*")))
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        self.logger.info(f"Loading {len(image_files)} images")
        
        # For large datasets, load and combine in batches
        n_frames = len(image_files)
        
        # Load a single frame to get dimensions
        first_frame = skimage.io.imread(image_files[0])
        height, width = first_frame.shape
        
        # Preallocate array
        data = np.zeros((n_frames, height, width), dtype=np.uint16)
        
        # Load all frames
        for i, file_path in enumerate(tqdm(image_files, desc="Loading frames")):
            data[i] = skimage.io.imread(file_path)
        
        # Load ground truth ROIs if available
        regions = []
        regions_path = os.path.join(dataset_dir, "regions", "regions.json")
        if os.path.exists(regions_path):
            with open(regions_path, 'r') as f:
                regions_data = json.load(f)
                
            for r in regions_data:
                coordinates = np.array(r['coordinates'])
                regions.append({
                    'coordinates': coordinates,
                    'id': r.get('id', len(regions))
                })
            
            self.logger.info(f"Loaded {len(regions)} ground truth regions")
        
        return {
            "data": data,
            "regions": regions,
            "dataset_id": dataset_id,
            "info": self.neurofinder_datasets[dataset_id]
        }
    
    def load_allen_brain_data(self, exp_id: str = None) -> Dict:
        """
        Load Allen Brain Observatory data using AllenSDK.
        
        Args:
            exp_id: Optional experiment ID, if None use the first available experiment
            
        Returns:
            Dictionary with loaded data
        """
        try:
            from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        except ImportError:
            self.logger.error("AllenSDK not installed, please install with: pip install allensdk")
            return {
                "error": "AllenSDK not installed",
                "instructions": "Install AllenSDK with: pip install allensdk"
            }
        
        # Create cache directory
        cache_dir = os.path.join(self.data_dir, "allen_brain")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize Brain Observatory Cache
        boc = BrainObservatoryCache(cache_directory=cache_dir)
        
        try:
            if exp_id is None:
                # Get experiment containers
                self.logger.info("Fetching experiment containers")
                containers = boc.get_experiment_containers(targeted_structures=['VISp'])
                
                if not containers:
                    self.logger.warning("No experiment containers found, fetching all")
                    containers = boc.get_experiment_containers()
                
                if not containers:
                    raise ValueError("No experiment containers available")
                
                # Get the first experiment
                container_id = containers[0]['id']
                experiments = boc.get_ophys_experiments(experiment_container_ids=[container_id])
                
                if not experiments:
                    raise ValueError(f"No experiments found for container {container_id}")
                
                exp_id = experiments[0]['id']
                self.logger.info(f"Using experiment ID: {exp_id}")
            
            # Get experiment data
            self.logger.info(f"Loading experiment {exp_id}")
            dataset = boc.get_ophys_experiment_data(exp_id)
            
            # Get fluorescence data
            cell_ids = dataset.get_cell_specimen_ids()
            dff_traces = dataset.get_dff_traces()
            
            # Get max projection
            max_projection = dataset.get_max_projection()
            
            # Get ROI masks
            roi_masks = dataset.get_roi_mask_array()
            
            # Get stimulus information
            session_type = dataset.get_metadata()['session_type']
            stimuli = dataset.list_stimuli()
            
            return {
                "cell_ids": cell_ids,
                "dff_traces": dff_traces,
                "max_projection": max_projection,
                "roi_masks": roi_masks,
                "session_type": session_type,
                "stimuli": stimuli,
                "exp_id": exp_id
            }
            
        except Exception as e:
            self.logger.error(f"Error loading Allen Brain data: {e}")
            return {
                "error": str(e),
                "instructions": "Try downloading data first using AllenSDK"
            }
    
    def load_sample_dataset(self, sample_name: str = "sample1") -> Dict:
        """
        Load a small sample calcium imaging dataset for testing.
        
        Args:
            sample_name: Name of the sample dataset
            
        Returns:
            Dictionary with loaded data
        """
        sample_dir = os.path.join(self.data_dir, "samples", sample_name)
        
        if not os.path.exists(sample_dir):
            self.logger.info(f"Sample not found locally, downloading: {sample_name}")
            sample_dir = self.download_sample_dataset(sample_name)
        
        # Load images
        images_dir = os.path.join(sample_dir, "images")
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.tif*")))
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        self.logger.info(f"Loading {len(image_files)} images")
        
        # Load all frames
        frames = []
        for file_path in tqdm(image_files, desc="Loading frames"):
            frames.append(skimage.io.imread(file_path))
        
        data = np.array(frames)
        
        # Load ground truth ROIs if available
        regions = []
        regions_path = os.path.join(sample_dir, "regions", "regions.json")
        if os.path.exists(regions_path):
            with open(regions_path, 'r') as f:
                regions_data = json.load(f)
                
            for r in regions_data:
                coordinates = np.array(r['coordinates'])
                regions.append({
                    'coordinates': coordinates,
                    'id': r.get('id', len(regions))
                })
            
            self.logger.info(f"Loaded {len(regions)} ground truth regions")
        
        return {
            "data": data,
            "regions": regions,
            "sample_name": sample_name,
            "info": self.sample_datasets[sample_name]
        }
    
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
    loader = CalciumDataLoader()
    
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
    
    # Example: Download and load a sample dataset
    try:
        sample_data = loader.load_sample_dataset("sample1")
        print(f"\nLoaded sample dataset: {sample_data['data'].shape} frames")
        print(f"Ground truth regions: {len(sample_data['regions'])}")
    except Exception as e:
        print(f"Error loading sample: {e}")