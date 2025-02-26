#!/usr/bin/env python
# eeg_pipeline_demo.py
"""
Neural AI Network - EEG Processing Pipeline Demonstration

This script demonstrates the EEG preprocessing pipeline, data loading, and
integration with the Neural AI Network framework.
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from eeg_data_loader import EEGDataLoader
from eeg_preprocessing import EEGPreprocessor, PreprocessingConfig
import mne


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"eeg_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger("EEGPipelineDemo")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EEG Processing Pipeline Demo")
    
    # Data source arguments
    parser.add_argument("--source", choices=["eeglab", "temple", "file", "simulate"], 
                       default="eeglab", help="Data source")
    parser.add_argument("--file", help="Path to specific EEG file if source is 'file'")
    parser.add_argument("--dataset", default="eeglab_sample", 
                       help="Dataset name for EEGLAB or Temple EEG")
    
    # Processing arguments
    parser.add_argument("--config", help="Path to preprocessing configuration JSON file")
    parser.add_argument("--skip-preprocessing", action="store_true", 
                       help="Skip preprocessing and use sample data")
    
    # Output arguments
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to output directory")
    
    return parser.parse_args()


def load_data(args, logger):
    """Load EEG data from specified source."""
    loader = EEGDataLoader()
    
    if args.source == "eeglab":
        logger.info(f"Loading EEGLAB dataset: {args.dataset}")
        try:
            raw = loader.load_eeglab_dataset(args.dataset)
            return raw
        except Exception as e:
            logger.error(f"Error loading EEGLAB dataset: {e}")
            logger.info("Falling back to simulated data")
            return simulate_data()
    
    elif args.source == "temple":
        logger.info(f"Loading Temple EEG dataset: {args.dataset}")
        try:
            raw = loader.load_temple_eeg_file()
            return raw
        except Exception as e:
            logger.error(f"Error loading Temple EEG dataset: {e}")
            logger.info("Falling back to simulated data")
            return simulate_data()
    
    elif args.source == "file":
        if not args.file:
            logger.error("No file specified. Use --file argument.")
            sys.exit(1)
        
        logger.info(f"Loading EEG file: {args.file}")
        try:
            raw = loader.load_eeg_file(args.file)
            return raw
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            logger.info("Falling back to simulated data")
            return simulate_data()
    
    elif args.source == "simulate":
        logger.info("Generating simulated EEG data")
        return simulate_data()
    
    else:
        logger.error(f"Unknown data source: {args.source}")
        sys.exit(1)


def simulate_data():
    """Create simulated EEG data."""
    # Create info object with standard 10-20 electrode positions
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 
               'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names),
        sfreq=256
    )
    
    # Simulate 60 seconds of data
    try:
        from mne.simulation import simulate_raw
        raw = simulate_raw(info, duration=60)
    except ImportError:
        # Fallback if mne.simulation is not available
        data = np.random.randn(len(ch_names), 256 * 60) * 10e-6  # ~10 µV
        
        # Add some alpha oscillations
        t = np.arange(0, 60, 1/256)
        alpha = np.sin(2 * np.pi * 10 * t) * 15e-6  # 10 Hz, ~15 µV
        data[ch_names.index('O1'), :] += alpha
        data[ch_names.index('O2'), :] += alpha
        
        # Add some 60 Hz noise
        line_noise = np.sin(2 * np.pi * 60 * t) * 5e-6  # 60 Hz, ~5 µV
        data += line_noise
        
        raw = mne.io.RawArray(data, info)
    
    return raw


def load_preprocessing_config(config_path=None):
    """Load preprocessing configuration or create default."""
    if config_path and os.path.exists(config_path):
        return PreprocessingConfig.load(config_path)
    
    # Create default configuration
    config = PreprocessingConfig()
    
    # Customize for demonstration
    config.lowpass_freq = 45.0
    config.highpass_freq = 1.0
    config.apply_notch = True
    config.notch_freq = 60.0
    config.resample = True
    config.resample_freq = 250.0
    config.detect_bad_channels = True
    config.reject_artifacts = True
    config.create_epochs = True
    config.epoch_duration_s = 1.0
    config.epoch_overlap_s = 0.5
    config.extract_features = True
    config.feature_types = ["bandpower", "connectivity", "time_domain"]
    
    return config


def preprocess_data(raw, config, output_dir, logger):
    """Apply preprocessing pipeline to raw data."""
    logger.info("Initializing preprocessor")
    
    # Update output directory in config
    config.interim_dir = os.path.join(output_dir, "interim")
    config.save_interim = True
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(config)
    
    # Process data
    logger.info("Starting preprocessing pipeline")
    start_time = time.time()
    
    result = preprocessor.preprocess(raw, file_id="demo")
    
    processing_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {processing_time:.2f} seconds")
    
    return result


def visualize_results(result, output_dir, save_plots=False):
    """Visualize processing results."""
    print("\n===== RESULTS VISUALIZATION =====\n")
    
    print("Original Data Information:")
    print(f"- Channels: {result['raw_info']['n_channels']}")
    print(f"- Channel names: {', '.join(result['raw_info']['ch_names'])}")
    print(f"- Duration: {result['raw_info']['duration']:.1f} seconds")
    print(f"- Sampling rate: {result['raw_info']['sfreq']} Hz")
    
    print("\nProcessing Steps Applied:")
    for i, step in enumerate(result["processing_steps"]):
        print(f"{i+1}. {step['step'].upper()}")
        for key, value in step["info"].items():
            print(f"   - {key}: {value}")
    
    print(f"\nBad Channels: {result['bad_channels'] if result['bad_channels'] else 'None'}")
    
    if "epoch_data" in result:
        epochs = result["epoch_data"]
        print(f"\nEpochs Created: {len(epochs)}")
        print(f"- Duration: {epochs.times[-1] - epochs.times[0]:.3f} seconds each")
    
    if "features" in result:
        print("\nExtracted Features:")
        for feature_type, features in result["features"].items():
            print(f"\n{feature_type.upper()}:")
            if feature_type == "bandpower":
                # Print band power features
                bands = [b for b in features.keys() if b not in ["channel_names"] and not b.startswith("rel_")]
                print(f"- Frequency bands: {', '.join(bands)}")
                
                if "epoch_data" in result:
                    # For epochs, show average band power
                    print("- Average band power across epochs (first 3 channels):")
                    for band in bands[:3]:  # Show first 3 bands
                        band_avg = np.mean(features[band], axis=0)
                        print(f"  {band}: {band_avg[:3]}")
                else:
                    # For continuous data
                    print("- Band power for first 3 channels:")
                    for band in bands[:3]:
                        print(f"  {band}: {features[band][:3]}")
            
            elif feature_type == "connectivity":
                # Print connectivity features
                method = features.get("method", "unknown")
                bands = [b for b in features.keys() if b not in ["channel_names", "method", "error"]]
                print(f"- Connectivity method: {method}")
                print(f"- Frequency bands: {', '.join(bands)}")
                
                if bands:
                    # Show connectivity strength range for first band
                    first_band = bands[0]
                    con_matrix = features[first_band]
                    print(f"- {first_band} connectivity range: {np.min(con_matrix):.3f} to {np.max(con_matrix):.3f}")
            
            elif feature_type == "time_domain":
                # Print time domain features
                metrics = [m for m in features.keys() if m != "channel_names"]
                print(f"- Metrics: {', '.join(metrics)}")
                
                if "mean" in features:
                    if "epoch_data" in result:
                        # For epochs, show average across epochs
                        print("- Average mean value across epochs (first 3 channels):")
                        mean_avg = np.mean(features["mean"], axis=0)
                        print(f"  {mean_avg[:3]}")
                    else:
                        # For continuous data
                        print("- Mean value for first 3 channels:")
                        print(f"  {features['mean'][:3]}")
    
    # Create and save visualizations
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot channel data
        try:
            if "raw_data" in result:
                # Plot raw data (first 10 seconds)
                plt.figure(figsize=(12, 8))
                result["raw_data"].copy().crop(0, 10).plot(n_channels=min(16, result["raw_info"]["n_channels"]), 
                                                          scalings='auto', title="Preprocessed EEG Data (first 10s)",
                                                          show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "preprocessed_eeg.png"))
                plt.close()
            
            if "epoch_data" in result:
                # Plot a few epochs
                plt.figure(figsize=(12, 8))
                result["epoch_data"].copy().crop(tmin=None, tmax=None).plot(n_channels=min(16, result["raw_info"]["n_channels"]), 
                                                                          n_epochs=3, scalings='auto', 
                                                                          title="Example Epochs", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "eeg_epochs.png"))
                plt.close()
            
            if "features" in result and "bandpower" in result["features"]:
                # Plot band powers
                band_powers = result["features"]["bandpower"]
                bands = [b for b in band_powers.keys() if b not in ["channel_names"] and not b.startswith("rel_")]
                
                plt.figure(figsize=(12, 8))
                
                if "epoch_data" in result:
                    # For epochs, plot average band power across epochs
                    x = np.arange(len(band_powers["channel_names"]))
                    width = 0.8 / len(bands)
                    
                    for i, band in enumerate(bands):
                        # Average across epochs
                        avg_power = np.mean(band_powers[band], axis=0)
                        plt.bar(x + i*width, avg_power, width, label=band)
                    
                    plt.xlabel('Channel')
                    plt.ylabel('Power (µV²/Hz)')
                    plt.title('Average Band Powers Across Epochs')
                    plt.xticks(x + width * (len(bands) - 1) / 2, band_powers["channel_names"], rotation=45)
                    plt.legend()
                else:
                    # For continuous data, plot band power per channel
                    x = np.arange(len(bands))
                    width = 0.8 / len(band_powers["channel_names"])
                    
                    # Limit to first 10 channels for visibility
                    channels = band_powers["channel_names"][:10]
                    
                    for i, channel in enumerate(channels):
                        ch_idx = band_powers["channel_names"].index(channel)
                        channel_powers = [band_powers[band][ch_idx] for band in bands]
                        plt.bar(x + i*width, channel_powers, width, label=channel)
                    
                    plt.xlabel('Frequency Band')
                    plt.ylabel('Power (µV²/Hz)')
                    plt.title('Band Powers by Channel')
                    plt.xticks(x + width * (len(channels) - 1) / 2, bands)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "band_powers.png"))
                plt.close()
            
            if "features" in result and "connectivity" in result["features"]:
                bands = [b for b in result["features"]["connectivity"].keys() 
                        if b not in ["channel_names", "method", "error"]]
                
                if bands:
                    # Plot connectivity matrix for the first band
                    band = bands[0]
                    con_matrix = result["features"]["connectivity"][band]
                    channel_names = result["features"]["connectivity"]["channel_names"]
                    
                    plt.figure(figsize=(10, 8))
                    im = plt.imshow(con_matrix, cmap='viridis', interpolation='none')
                    plt.colorbar(im, label=result["features"]["connectivity"]["method"])
                    plt.title(f'{band.capitalize()} Band Connectivity')
                    
                    # Add channel labels if not too many
                    if len(channel_names) <= 20:
                        plt.xticks(np.arange(len(channel_names)), channel_names, rotation=45)
                        plt.yticks(np.arange(len(channel_names)), channel_names)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"connectivity_{band}.png"))
                    plt.close()
                    
                    # Plot connectivity graph for the first band (if networkx is available)
                    try:
                        import networkx as nx
                        
                        # Create graph from connectivity matrix
                        G = nx.from_numpy_array(con_matrix)
                        
                        # Set threshold to keep only stronger connections
                        threshold = np.percentile(con_matrix[con_matrix > 0], 75)
                        
                        # Create position dictionary using circular layout
                        pos = nx.circular_layout(G)
                        
                        plt.figure(figsize=(10, 8))
                        
                        # Draw nodes
                        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
                        
                        # Draw edges with width based on weight, but only those above threshold
                        edges = [(i, j) for i, j in G.edges() if con_matrix[i, j] > threshold]
                        weights = [con_matrix[i, j] for i, j in edges]
                        
                        if edges:
                            nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*5 for w in weights], 
                                                alpha=0.7, edge_color='navy')
                        
                        # Add labels if not too many channels
                        if len(channel_names) <= 20:
                            labels = {i: channel_names[i] for i in range(len(channel_names))}
                            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
                        
                        plt.title(f'{band.capitalize()} Band Connectivity Network (top 25% connections)')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"connectivity_graph_{band}.png"))
                        plt.close()
                    except ImportError:
                        print("NetworkX not installed, skipping connectivity graph visualization")
            
            print(f"\nPlots saved to {output_dir}")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    return


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting EEG pipeline demonstration")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create preprocessing configuration
    config = load_preprocessing_config(args.config)
    
    # Save configuration for reference
    config_path = os.path.join(args.output_dir, "preprocessing_config.json")
    config.save(config_path)
    logger.info(f"Preprocessing configuration saved to {config_path}")
    
    if args.skip_preprocessing:
        logger.info("Skipping preprocessing, using sample data")
        
        # Load sample result if available
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "eeg_result.pkl")
        
        if os.path.exists(sample_path):
            import pickle
            with open(sample_path, 'rb') as f:
                result = pickle.load(f)
            logger.info("Loaded sample result data")
        else:
            logger.error("Sample data not found. Please run without --skip-preprocessing")
            sys.exit(1)
    else:
        # Load data
        raw = load_data(args, logger)
        logger.info(f"Loaded EEG data: {len(raw.ch_names)} channels, {raw.n_times} samples at {raw.info['sfreq']} Hz")
        
        # Apply preprocessing
        result = preprocess_data(raw, config, args.output_dir, logger)
        
        # Save result for later use
        import pickle
        os.makedirs(os.path.join(args.output_dir, "processed"), exist_ok=True)
        result_path = os.path.join(args.output_dir, "processed", "eeg_result.pkl")
        
        # Remove raw data objects before saving to save space
        result_save = {k: v for k, v in result.items() if k not in ['raw_data', 'epoch_data']}
        
        with open(result_path, 'wb') as f:
            pickle.dump(result_save, f)
        logger.info(f"Result data saved to {result_path}")
    
    # Visualize results
    visualize_results(result, args.output_dir, save_plots=args.save_plots)
    
    logger.info("EEG pipeline demonstration completed")


if __name__ == "__main__":
    main()