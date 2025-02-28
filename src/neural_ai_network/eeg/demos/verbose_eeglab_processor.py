#!/usr/bin/env python
# verbose_eeglab_processor.py

import os
import logging
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import mne

# Set up more verbose logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerboseEEGLABProcessor")

# Import your package modules
from neural_ai_network.eeg.data_loader import EEGDataLoader
from neural_ai_network.eeg.preprocessing import EEGPreprocessor, PreprocessingConfig

def main():
    """Main execution with forced arguments and verbose output."""
    print("="*50)
    print("Starting VERBOSE EEGLAB batch processor")
    print("="*50)
    
    # Force arguments instead of parsing from command line
    args = argparse.Namespace(
        datasets=["eeglab_sample"],  # Just process one dataset
        config=None,                # Use default config
        max_files=1,                # Process only one file
        parallel=False,             # Sequential processing
        output_dir="./verbose_eeglab_output",  # Custom output dir
        save_plots=True             # Save plots
    )
    
    print(f"Using arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Initialize data loader with verbose output
    print("Initializing data loader...")
    data_loader = EEGDataLoader()
    print(f"Data loader initialized with data_dir: {data_loader.data_dir}")
    
    # Load preprocessing configuration
    print("Creating preprocessing configuration...")
    config = PreprocessingConfig()
    # Use minimal configuration for testing
    config.lowpass_freq = 45.0
    config.highpass_freq = 1.0
    config.apply_notch = True
    config.notch_freq = 60.0
    config.resample = True
    config.resample_freq = 250.0
    config.detect_bad_channels = True
    config.reject_artifacts = True
    config.create_epochs = True
    config.extract_features = True
    config.feature_types = ["bandpower"]  # Just use one feature type for testing
    
    print("Configuration created")
    
    # Save configuration for reference
    config_path = os.path.join(args.output_dir, "preprocessing_config.json")
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Process one dataset
    dataset_name = args.datasets[0]
    print(f"Processing dataset: {dataset_name}")
    
    # Download dataset (with explicit handling)
    print(f"Downloading dataset: {dataset_name}")
    try:
        dataset_dir = data_loader.download_eeglab_sample(dataset_name)
        print(f"Dataset downloaded to: {dataset_dir}")
        
        # Find files
        print("Searching for EEG files...")
        set_files = glob.glob(os.path.join(dataset_dir, "**/*.set"), recursive=True)
        edf_files = glob.glob(os.path.join(dataset_dir, "**/*.edf"), recursive=True)
        
        all_files = set_files + edf_files
        print(f"Found {len(all_files)} files")
        
        if all_files:
            # Process just the first file
            file_path = all_files[0]
            print(f"Processing file: {file_path}")
            
            try:
                # Load the file
                print("Loading EEG file...")
                raw = data_loader.load_eeg_file(file_path)
                print(f"Loaded EEG data: {len(raw.ch_names)} channels, {raw.n_times} samples")
                
                # Create output directory for this file
                file_output_dir = os.path.join(args.output_dir, dataset_name, "file_0")
                os.makedirs(file_output_dir, exist_ok=True)
                print(f"Created file output directory: {file_output_dir}")
                
                # Update config for this file
                file_config = PreprocessingConfig()
                for key, value in vars(config).items():
                    setattr(file_config, key, value)
                
                file_config.interim_dir = os.path.join(file_output_dir, "interim")
                file_config.save_interim = True
                
                # Initialize preprocessor
                print("Initializing preprocessor...")
                preprocessor = EEGPreprocessor(file_config)
                
                # Process data with timing
                print("Starting preprocessing...")
                start_time = time.time()
                
                result = preprocessor.preprocess(raw, file_id="test_file")
                
                processing_time = time.time() - start_time
                print(f"Preprocessing completed in {processing_time:.2f} seconds")
                
                # Save a simple summary
                print("Saving summary...")
                summary_path = os.path.join(file_output_dir, "summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"EEG PROCESSING SUMMARY\n")
                    f.write("====================\n\n")
                    f.write(f"Processing time: {processing_time:.2f} seconds\n")
                    if "raw_info" in result:
                        f.write(f"Channels: {result['raw_info'].get('n_channels', 'N/A')}\n")
                
                print(f"Summary saved to: {summary_path}")
                print("Processing completed successfully")
                
            except Exception as e:
                print(f"Error processing file: {e}")
        else:
            print("No files found in the dataset")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    
    print("="*50)
    print("VERBOSE EEGLAB batch processor completed")
    print("="*50)


if __name__ == "__main__":
    main()