#!/usr/bin/env python
# debug_eeglab_processor.py

import os
import sys
import logging

# Set up basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DebugEEGLABProcessor")

def main():
    """Simple debug function to test execution."""
    print("Starting debug EEGLAB processor...")
    
    # Test importing required modules
    try:
        print("Importing modules...")
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        import mne
        print("Core modules imported successfully")
        
        # Test importing your package modules
        try:
            print("Importing package modules...")
            from neural_ai_network.eeg.data_loader import EEGDataLoader
            from neural_ai_network.eeg.preprocessing import EEGPreprocessor, PreprocessingConfig
            print("Package modules imported successfully")
            
            # Test initializing components
            print("Initializing components...")
            data_loader = EEGDataLoader()
            print(f"Data loader initialized with data_dir: {data_loader.data_dir}")
            
            # Test directory creation
            output_dir = "./debug_output"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
            
            # Test downloading a dataset
            print("Attempting to download EEGLAB sample dataset...")
            try:
                dataset_dir = data_loader.download_eeglab_sample("eeglab_sample")
                print(f"Dataset downloaded to: {dataset_dir}")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
            
            # Create a simple file to verify output is working
            with open(os.path.join(output_dir, "debug_output.txt"), "w") as f:
                f.write("Debug test completed successfully\n")
            print(f"Created debug output file: {os.path.join(output_dir, 'debug_output.txt')}")
            
        except ImportError as e:
            print(f"Error importing package modules: {e}")
            print("Check that your package is correctly installed and importable")
            
    except ImportError as e:
        print(f"Error importing core modules: {e}")
        print("Make sure required packages are installed")
    
    print("Debug process completed")

if __name__ == "__main__":
    main()