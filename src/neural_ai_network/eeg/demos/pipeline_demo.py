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
from neural_ai_network.eeg.data_loader import EEGDataLoader
from neural_ai_network.eeg.preprocessing import EEGPreprocessor, PreprocessingConfig

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
    """Create simulated EEG data with clearly visible signals."""
    # Create info object with standard 10-20 electrode positions
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 
               'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    
    # Create mne.Info object
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names),
        sfreq=256
    )
    
    # Generate 60 seconds of data with larger amplitude (50 µV baseline)
    data = np.random.randn(len(ch_names), 256 * 60) * 50e-6  # ~50 µV
    
    # Add more pronounced oscillations
    t = np.arange(0, 60, 1/256)
    
    # Add different frequencies to different channels
    # Alpha (8-12 Hz) in occipital channels
    alpha = np.sin(2 * np.pi * 10 * t) * 100e-6  # 100 µV
    data[ch_names.index('O1'), :] += alpha
    data[ch_names.index('O2'), :] += alpha
    
    # Beta (13-30 Hz) in frontal channels
    beta = np.sin(2 * np.pi * 20 * t) * 50e-6  # 50 µV
    data[ch_names.index('F3'), :] += beta
    data[ch_names.index('F4'), :] += beta
    
    # Theta (4-7 Hz) in temporal channels
    theta = np.sin(2 * np.pi * 6 * t) * 75e-6  # 75 µV
    data[ch_names.index('T7'), :] += theta
    data[ch_names.index('T8'), :] += theta
    
    # Add some 60 Hz noise
    line_noise = np.sin(2 * np.pi * 60 * t) * 20e-6  # 20 µV
    data += line_noise
    
    # Add a slow drift to some channels
    drift = np.sin(2 * np.pi * 0.1 * t) * 200e-6  # 200 µV very slow wave
    data[ch_names.index('Cz'), :] += drift
    
    # Create raw object
    raw = mne.io.RawArray(data, info)
    
    # Set montage with standard positions
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
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
    
    # Display basic information
    try:
        print("Original Data Information:")
        if "raw_info" in result:
            raw_info = result["raw_info"]
            print(f"- Channels: {raw_info.get('n_channels', 'N/A')}")
            print(f"- Channel names: {(', '.join(raw_info.get('ch_names', ['N/A']))[:100] + '...') if len(raw_info.get('ch_names', [])) > 10 else ', '.join(raw_info.get('ch_names', ['N/A']))}")
            print(f"- Duration: {raw_info.get('duration', 'N/A'):.1f} seconds")
            print(f"- Sampling rate: {raw_info.get('sfreq', 'N/A')} Hz")
        else:
            print("- Raw information not available")
    except Exception as e:
        print(f"Error displaying data information: {e}")
    
    # Display processing steps
    try:
        print("\nProcessing Steps Applied:")
        if "processing_steps" in result:
            for i, step in enumerate(result["processing_steps"]):
                if isinstance(step, dict) and "step" in step:
                    print(f"{i+1}. {step['step'].upper()}")
                    if "info" in step and isinstance(step["info"], dict):
                        for key, value in step["info"].items():
                            # Limit the length of printed values
                            if isinstance(value, (list, tuple)) and len(value) > 5:
                                value_str = f"{str(value[:5])[:-1]}, ...] ({len(value)} items)"
                            elif isinstance(value, str) and len(value) > 100:
                                value_str = f"{value[:100]}... (truncated)"
                            else:
                                value_str = str(value)
                            print(f"   - {key}: {value_str}")
                else:
                    print(f"{i+1}. Unknown processing step format")
        else:
            print("- No processing steps information available")
    except Exception as e:
        print(f"Error displaying processing steps: {e}")
    
    # Display bad channels
    try:
        print("\nBad Channels:", end=" ")
        if "bad_channels" in result:
            bad_channels = result["bad_channels"]
            if bad_channels and len(bad_channels) > 0:
                print(f"{bad_channels}")
            else:
                print("None")
        else:
            print("Information not available")
    except Exception as e:
        print(f"Error displaying bad channels: {e}")
    
    # Display epochs information
    try:
        if "epoch_data" in result:
            epochs = result["epoch_data"]
            print(f"\nEpochs Created: {len(epochs)}")
            print(f"- Duration: {epochs.times[-1] - epochs.times[0]:.3f} seconds each")
        elif "epoch_info" in result:
            epoch_info = result["epoch_info"]
            print(f"\nEpochs Created: {epoch_info.get('n_epochs', 'N/A')}")
            print(f"- Duration: {epoch_info.get('duration', 'N/A')} seconds each")
    except Exception as e:
        print(f"Error displaying epochs information: {e}")
    
    # Display features
    try:
        if "features" in result:
            print("\nExtracted Features:")
            for feature_type, features in result["features"].items():
                # Skip if features is not a dict
                if not isinstance(features, dict):
                    print(f"\n{feature_type.upper()}: Invalid format")
                    continue
                    
                print(f"\n{feature_type.upper()}:")
                
                # Band power features
                if feature_type == "bandpower":
                    # Get band names
                    bands = [b for b in features.keys() if b not in ["channel_names", "error"] and not b.startswith("rel_")]
                    if bands:
                        print(f"- Frequency bands: {', '.join(bands)}")
                        
                        # Check if we have epoch data
                        has_epochs = "epoch_data" in result
                        
                        if has_epochs:
                            # For epochs, show average band power
                            print("- Average band power across epochs (first 3 channels):")
                            for band in bands[:min(3, len(bands))]:
                                try:
                                    if band in features:
                                        band_data = features[band]
                                        if isinstance(band_data, (list, np.ndarray)) and len(band_data) > 0:
                                            # Check dimensionality
                                            if isinstance(band_data, np.ndarray) and band_data.ndim > 0:
                                                band_avg = np.mean(band_data, axis=0)
                                                # Show first few channels
                                                display_data = band_avg[:min(3, len(band_avg))]
                                                print(f"  {band}: {display_data}")
                                            elif isinstance(band_data, list) and len(band_data) > 0 and isinstance(band_data[0], (list, np.ndarray)):
                                                # Handle list of lists
                                                band_avg = np.mean(band_data, axis=0)
                                                display_data = band_avg[:min(3, len(band_avg))]
                                                print(f"  {band}: {display_data}")
                                            else:
                                                # Single value or flat list
                                                print(f"  {band}: {band_data}")
                                        else:
                                            print(f"  {band}: No data or empty array")
                                    else:
                                        print(f"  {band}: Not found in features")
                                except Exception as e:
                                    print(f"  {band}: Error processing data - {e}")
                        else:
                            # For continuous data
                            print("- Band power for first 3 channels:")
                            for band in bands[:min(3, len(bands))]:
                                try:
                                    if band in features:
                                        band_data = features[band]
                                        if isinstance(band_data, (list, np.ndarray)) and len(band_data) > 0:
                                            # Show first few channels
                                            display_data = band_data[:min(3, len(band_data))]
                                            print(f"  {band}: {display_data}")
                                        else:
                                            print(f"  {band}: No data or empty array")
                                    else:
                                        print(f"  {band}: Not found in features")
                                except Exception as e:
                                    print(f"  {band}: Error processing data - {e}")
                    else:
                        print("- No frequency bands found")
                
                # Connectivity features
                elif feature_type == "connectivity":
                    method = features.get("method", "unknown")
                    bands = [b for b in features.keys() if b not in ["channel_names", "method", "error"]]
                    
                    if bands:
                        print(f"- Connectivity method: {method}")
                        print(f"- Frequency bands: {', '.join(bands)}")
                        
                        # Show connectivity strength range for first band
                        first_band = bands[0]
                        try:
                            if first_band in features:
                                con_matrix = features[first_band]
                                if isinstance(con_matrix, (list, np.ndarray)) and len(con_matrix) > 0:
                                    # Convert to numpy if it's a list
                                    if isinstance(con_matrix, list):
                                        con_matrix = np.array(con_matrix)
                                    
                                    # Get min/max if it's a matrix
                                    if con_matrix.ndim >= 2:
                                        min_val = np.min(con_matrix)
                                        max_val = np.max(con_matrix)
                                        print(f"- {first_band} connectivity range: {min_val:.3f} to {max_val:.3f}")
                                    else:
                                        print(f"- {first_band} connectivity data is not a matrix")
                                else:
                                    print(f"- {first_band} has no valid connectivity data")
                            else:
                                print(f"- {first_band} not found in connectivity features")
                        except Exception as e:
                            print(f"- Error processing connectivity data: {e}")
                    else:
                        print("- No connectivity bands found")
                
                # Time domain features
                elif feature_type == "time_domain":
                    metrics = [m for m in features.keys() if m != "channel_names"]
                    if metrics:
                        print(f"- Metrics: {', '.join(metrics)}")
                        
                        # Show mean values as an example
                        if "mean" in features:
                            try:
                                mean_data = features["mean"]
                                has_epochs = "epoch_data" in result
                                
                                if has_epochs:
                                    # For epochs, show average across epochs
                                    print("- Average mean value across epochs (first 3 channels):")
                                    if isinstance(mean_data, (list, np.ndarray)) and len(mean_data) > 0:
                                        # Handle different data formats
                                        if isinstance(mean_data, np.ndarray) and mean_data.ndim > 0:
                                            mean_avg = np.mean(mean_data, axis=0)
                                            display_data = mean_avg[:min(3, len(mean_avg))]
                                            print(f"  {display_data}")
                                        elif isinstance(mean_data, list) and len(mean_data) > 0 and isinstance(mean_data[0], (list, np.ndarray)):
                                            mean_avg = np.mean(mean_data, axis=0)
                                            display_data = mean_avg[:min(3, len(mean_avg))]
                                            print(f"  {display_data}")
                                        else:
                                            print(f"  {mean_data}")
                                    else:
                                        print("  No valid mean data")
                                else:
                                    # For continuous data
                                    print("- Mean value for first 3 channels:")
                                    if isinstance(mean_data, (list, np.ndarray)) and len(mean_data) > 0:
                                        display_data = mean_data[:min(3, len(mean_data))]
                                        print(f"  {display_data}")
                                    else:
                                        print("  No valid mean data")
                            except Exception as e:
                                print(f"- Error processing mean data: {e}")
                    else:
                        print("- No time domain metrics found")
                
                # Other feature types
                else:
                    print(f"- Feature type details not displayed in console")
        else:
            print("\nNo features extracted or feature data not available")
    except Exception as e:
        print(f"Error displaying features: {e}")
    
    # Create and save visualizations
    if save_plots:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot channel data if available
            if "raw_data" in result:
                try:
                    plt.figure(figsize=(12, 8))
                    result["raw_data"].copy().crop(0, min(10, result["raw_data"].times[-1])).plot(
                        n_channels=min(16, len(result["raw_data"].ch_names)), 
                        scalings='auto', 
                        title="Preprocessed EEG Data (first 10s)",
                        show=False
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "preprocessed_eeg.png"))
                    plt.close()
                    print(f"\nSaved preprocessed EEG plot to {os.path.join(output_dir, 'preprocessed_eeg.png')}")
                except Exception as e:
                    print(f"Error creating raw data plot: {e}")
            
            # Plot epochs if available
            if "epoch_data" in result:
                try:
                    plt.figure(figsize=(12, 8))
                    result["epoch_data"].copy().plot(
                        n_channels=min(16, len(result["epoch_data"].ch_names)), 
                        n_epochs=min(3, len(result["epoch_data"])),
                        scalings='auto', 
                        title="Example Epochs", 
                        show=False
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "eeg_epochs.png"))
                    plt.close()
                    print(f"Saved epochs plot to {os.path.join(output_dir, 'eeg_epochs.png')}")
                except Exception as e:
                    print(f"Error creating epochs plot: {e}")
            
            # Plot band powers if available
            if "features" in result and "bandpower" in result["features"]:
                try:
                    band_powers = result["features"]["bandpower"]
                    bands = [b for b in band_powers.keys() if b not in ["channel_names", "error"] and not b.startswith("rel_")]
                    
                    if bands:
                        plt.figure(figsize=(12, 8))
                        
                        # For epochs/continuous data visualization
                        has_epochs = "epoch_data" in result
                        
                        if has_epochs and isinstance(band_powers.get(bands[0]), (list, np.ndarray)) and len(band_powers.get(bands[0])) > 0:
                            # Check if we have a proper structure
                            first_band_data = band_powers.get(bands[0])
                            if isinstance(first_band_data, np.ndarray) and first_band_data.ndim > 0:
                                # For epochs, plot average band power across epochs
                                x = np.arange(len(band_powers.get("channel_names", [])))
                                width = 0.8 / len(bands)
                                
                                for i, band in enumerate(bands):
                                    try:
                                        # Average across epochs
                                        if band in band_powers and isinstance(band_powers[band], (list, np.ndarray)) and len(band_powers[band]) > 0:
                                            band_data = np.array(band_powers[band])
                                            if band_data.ndim > 0:
                                                avg_power = np.mean(band_data, axis=0)
                                                plt.bar(x + i*width, avg_power, width, label=band)
                                    except Exception as e:
                                        print(f"Error plotting {band}: {e}")
                                
                                plt.xlabel('Channel')
                                plt.ylabel('Power (µV²/Hz)')
                                plt.title('Average Band Powers Across Epochs')
                                if x.size > 0 and len(bands) > 0:
                                    plt.xticks(
                                        x + width * (len(bands) - 1) / 2, 
                                        band_powers.get("channel_names", [])[:len(x)], 
                                        rotation=45
                                    )
                                plt.legend()
                            else:
                                print("Band power data structure not suitable for plotting")
                        else:
                            # For continuous data or simplified structure
                            channels = band_powers.get("channel_names", [])[:10]  # Limit to first 10 channels
                            
                            if channels and all(band in band_powers for band in bands):
                                x = np.arange(len(bands))
                                width = 0.8 / len(channels)
                                
                                for i, channel in enumerate(channels):
                                    try:
                                        if "channel_names" in band_powers:
                                            ch_idx = band_powers["channel_names"].index(channel)
                                            channel_powers = []
                                            
                                            for band in bands:
                                                if band in band_powers and isinstance(band_powers[band], (list, np.ndarray)):
                                                    # Make sure we have data and can index it
                                                    if len(band_powers[band]) > ch_idx:
                                                        channel_powers.append(band_powers[band][ch_idx])
                                                    else:
                                                        channel_powers.append(0)  # Placeholder if data missing
                                                else:
                                                    channel_powers.append(0)
                                            
                                            plt.bar(x + i*width, channel_powers, width, label=channel)
                                    except Exception as e:
                                        print(f"Error plotting channel {channel}: {e}")
                                
                                plt.xlabel('Frequency Band')
                                plt.ylabel('Power (µV²/Hz)')
                                plt.title('Band Powers by Channel')
                                if x.size > 0 and len(channels) > 0:
                                    plt.xticks(
                                        x + width * (len(channels) - 1) / 2, 
                                        bands, 
                                        rotation=0
                                    )
                                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
                            else:
                                print("Channel data not suitable for plotting")
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "band_powers.png"))
                        plt.close()
                        print(f"Saved band powers plot to {os.path.join(output_dir, 'band_powers.png')}")
                except Exception as e:
                    print(f"Error creating band powers plot: {e}")
            
            # Plot connectivity matrix if available
            if "features" in result and "connectivity" in result["features"]:
                try:
                    connectivity_features = result["features"]["connectivity"]
                    bands = [b for b in connectivity_features.keys() 
                            if b not in ["channel_names", "method", "error"]]
                    
                    if bands:
                        # Plot connectivity matrix for the first band
                        band = bands[0]
                        if band in connectivity_features:
                            con_matrix = connectivity_features[band]
                            if isinstance(con_matrix, (list, np.ndarray)) and len(con_matrix) > 0:
                                # Convert to numpy if it's a list
                                if isinstance(con_matrix, list):
                                    con_matrix = np.array(con_matrix)
                                
                                # Check if it's a proper matrix
                                if con_matrix.ndim >= 2:
                                    channel_names = connectivity_features.get("channel_names", [])
                                    
                                    plt.figure(figsize=(10, 8))
                                    im = plt.imshow(con_matrix, cmap='viridis', interpolation='none')
                                    plt.colorbar(im, label=connectivity_features.get("method", "connectivity"))
                                    plt.title(f'{band.capitalize()} Band Connectivity')
                                    
                                    # Add channel labels if not too many
                                    if len(channel_names) <= 20:
                                        plt.xticks(np.arange(len(channel_names)), channel_names, rotation=45)
                                        plt.yticks(np.arange(len(channel_names)), channel_names)
                                    
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(output_dir, f"connectivity_{band}.png"))
                                    plt.close()
                                    print(f"Saved connectivity matrix plot to {os.path.join(output_dir, f'connectivity_{band}.png')}")
                                    
                                    # Plot connectivity graph for the first band (if networkx is available)
                                    try:
                                        import networkx as nx
                                        
                                        # Create graph from connectivity matrix
                                        G = nx.from_numpy_array(con_matrix)
                                        
                                        # Set threshold to keep only stronger connections
                                        # Ensure we have positive values to calculate percentile
                                        pos_matrix = np.abs(con_matrix)
                                        pos_matrix_flat = pos_matrix[pos_matrix > 0]
                                        
                                        if len(pos_matrix_flat) > 0:
                                            threshold = np.percentile(pos_matrix_flat, 75)
                                            
                                            # Create position dictionary using circular layout
                                            pos = nx.circular_layout(G)
                                            
                                            plt.figure(figsize=(10, 8))
                                            
                                            # Draw nodes
                                            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
                                            
                                            # Draw edges with width based on weight, but only those above threshold
                                            edges = [(i, j) for i, j in G.edges() if pos_matrix[i, j] > threshold]
                                            weights = [pos_matrix[i, j] for i, j in edges]
                                            
                                            if edges:
                                                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                                                      width=[w*5 for w in weights], 
                                                                      alpha=0.7, edge_color='navy')
                                            
                                            # Add labels if not too many channels
                                            if len(channel_names) <= 20:
                                                labels = {i: channel_names[i] for i in range(min(len(channel_names), len(G.nodes())))}
                                                nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
                                            
                                            plt.title(f'{band.capitalize()} Band Connectivity Network (top 25% connections)')
                                            plt.axis('off')
                                            plt.tight_layout()
                                            plt.savefig(os.path.join(output_dir, f"connectivity_graph_{band}.png"))
                                            plt.close()
                                            print(f"Saved connectivity graph to {os.path.join(output_dir, f'connectivity_graph_{band}.png')}")
                                        else:
                                            print("No positive connectivity values to plot graph")
                                    except ImportError:
                                        print("NetworkX not installed, skipping connectivity graph visualization")
                                    except Exception as e:
                                        print(f"Error creating connectivity graph: {e}")
                                else:
                                    print(f"Connectivity data for {band} is not a proper matrix")
                            else:
                                print(f"No valid connectivity data for {band}")
                        else:
                            print(f"Band {band} not found in connectivity features")
                except Exception as e:
                    print(f"Error creating connectivity plots: {e}")
            
            print(f"\nAll plots saved to {output_dir}")
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