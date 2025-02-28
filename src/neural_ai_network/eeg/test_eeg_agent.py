# improved_test_eeg_agent.py
import os
import json
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from neural_ai_network.eeg.eeg_agent import EEGProcessingAgent
from neural_ai_network.core import NeuralDataOrchestrator

def setup_logging():
    """Set up logging for the test script."""
    logger = logging.getLogger("TestScript")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def test_eeg_agent_with_file(file_path, output_dir=None):
    """Test the EEG agent with a specific file."""
    logger = setup_logging()
    logger.info(f"Testing EEG Agent with file: {file_path}")
    
    # Initialize the agent
    agent = EEGProcessingAgent()
    
    # Process the file
    result = agent.process_data(file_path)
    
    if result["status"] == "error":
        logger.error(f"Error processing file: {result['error']}")
        return
    
    logger.info(f"Processing successful: {result['status']}")
    logger.info(f"Results saved to: {result['results_path']}")
    
    # Load the results for analysis
    with open(result["results_path"], 'r') as f:
        results_data = json.load(f)
    
    # Check what features were extracted
    logger.info(f"Extracted features: {list(results_data['features'].keys())}")
    
    # Analyze features
    if "features" in results_data:
        analysis = agent.analyze_features(results_data["features"])
        
        logger.info("\nEEG AGENT ANALYSIS:")
        logger.info("-" * 50)
        logger.info(analysis)
        
        # Test band-specific analyses
        if "bandpower" in results_data["features"]:
            for band in ["alpha", "beta", "theta", "delta"]:
                band_analysis = agent.analyze_band_activity(
                    results_data["features"]["bandpower"], 
                    band=band
                )
                logger.info(f"\n{band.upper()} BAND ANALYSIS:")
                logger.info("-" * 50)
                logger.info(band_analysis)
        
        # Visualize results if output directory is specified
        if output_dir and "bandpower" in results_data["features"]:
            try:
                visualize_results(results_data, output_dir, os.path.basename(file_path))
            except Exception as e:
                logger.error(f"Error visualizing results: {e}")
    
    return result

def test_eeg_agent_with_directory(input_dir, output_dir=None):
    """Test the EEG agent with all EEG files in a directory."""
    logger = setup_logging()
    logger.info(f"Testing EEG Agent with files from: {input_dir}")
    
    # Find all EEG files
    eeg_files = []
    for ext in [".edf", ".bdf", ".set", ".fif", ".vhdr"]:
        eeg_files.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))
    
    if not eeg_files:
        logger.error(f"No EEG files found in {input_dir}")
        return
    
    logger.info(f"Found {len(eeg_files)} EEG files")
    
    # Process each file
    for file_path in eeg_files:
        try:
            test_eeg_agent_with_file(file_path, output_dir)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

def test_orchestrator_eeg_integration(eeg_file):
    """Test the integration between the orchestrator and EEG agent."""
    logger = setup_logging()
    logger.info(f"Testing Orchestrator-EEG integration with file: {eeg_file}")
    
    # Initialize orchestrator
    orchestrator = NeuralDataOrchestrator()
    
    # Set up a result callback
    def result_callback(result):
        logger.info(f"Received result for job {result['job_id']}: {result['status']}")
    
    # Register callback
    try:
        orchestrator.register_result_callback(result_callback)
    except Exception as e:
        logger.warning(f"Could not register callback: {e}")
    
    # Process an EEG file
    job_id = orchestrator.process_data(eeg_file)
    logger.info(f"Started job: {job_id}")
    
    # Check job status
    status = orchestrator.get_job_status(job_id)
    logger.info(f"Initial job status: {status}")
    
    # Wait a bit and check again
    import time
    time.sleep(5)
    
    # Check status again
    status = orchestrator.get_job_status(job_id)
    logger.info(f"Updated job status: {status}")
    
    # Clean shutdown
    orchestrator.shutdown()

def visualize_results(results_data, output_dir, file_name_base):
    """Visualize the results of EEG processing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualize band powers
    if "bandpower" in results_data["features"]:
        bandpower = results_data["features"]["bandpower"]
        if "band_powers" in bandpower:
            bands = bandpower["band_powers"]
            channel_names = bandpower.get("channel_names", [])
            
            # Create figure for band powers
            plt.figure(figsize=(12, 8))
            
            # Plot each band power
            x = np.arange(len(channel_names))
            width = 0.15
            offsets = [-2, -1, 0, 1, 2]
            
            for i, (band_name, offset) in enumerate(zip(bands.keys(), offsets)):
                if band_name in ["delta", "theta", "alpha", "beta", "gamma"]:
                    plt.bar(
                        x + (width * offset),
                        bands[band_name],
                        width,
                        label=band_name.capitalize()
                    )
            
            plt.xlabel('Channels')
            plt.ylabel('Power (µV²/Hz)')
            plt.title('Band Powers by Channel')
            plt.xticks(x, channel_names, rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{file_name_base}_band_powers.png"))
            plt.close()
    
    # 2. Visualize connectivity matrix if available
    if "connectivity" in results_data["features"]:
        connectivity = results_data["features"]["connectivity"]
        
        if "connectivity_matrices" in connectivity:
            matrices = connectivity["connectivity_matrices"]
            channel_names = connectivity.get("channel_names", [])
            
            for band, matrix in matrices.items():
                plt.figure(figsize=(10, 8))
                plt.imshow(matrix, cmap='viridis')
                plt.colorbar(label=connectivity.get("method", "connectivity"))
                plt.title(f'{band.capitalize()} Band Connectivity')
                
                # Add channel labels if not too many
                if len(channel_names) <= 32:
                    plt.xticks(np.arange(len(channel_names)), channel_names, rotation=90)
                    plt.yticks(np.arange(len(channel_names)), channel_names)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_name_base}_{band}_connectivity.png"))
                plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file or directory
        path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"
        
        if os.path.isfile(path):
            # Test with a specific file
            test_eeg_agent_with_file(path, output_dir)
            
            # Test orchestrator integration if RabbitMQ is available
            try:
                test_orchestrator_eeg_integration(path)
            except Exception as e:
                print(f"Orchestrator test failed (RabbitMQ might not be running): {e}")
                
        elif os.path.isdir(path):
            # Test with all files in directory
            test_eeg_agent_with_directory(path, output_dir)
        else:
            print(f"Path not found: {path}")
    else:
        print("Usage: python improved_test_eeg_agent.py <file_or_directory> [output_dir]")