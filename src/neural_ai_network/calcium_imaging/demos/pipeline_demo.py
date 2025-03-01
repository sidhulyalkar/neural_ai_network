#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcium Imaging Pipeline Demo

This script demonstrates the calcium imaging processing pipeline
from data loading to cell detection, signal extraction, and analysis.
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from neural_ai_network.calcium_imaging.calcium_agent import CalciumProcessingAgent
from neural_ai_network.calcium_imaging.calcium_data_loader import CalciumDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CalciumPipelineDemo")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calcium Imaging Pipeline Demo')
    
    parser.add_argument('--dataset', type=str, default='sample1',
                        help='Dataset to process: "sample1", "sample2", or a Neurofinder ID like "00.00"')
    
    parser.add_argument('--data-type', type=str, default='sample',
                        choices=['sample', 'neurofinder', 'synthetic', 'file'],
                        help='Type of data to load')
    
    parser.add_argument('--file-path', type=str, default=None,
                        help='Path to custom calcium imaging file (for data-type=file)')
    
    parser.add_argument('--output-dir', type=str, default='./calcium_demo_output',
                        help='Directory to save results')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom configuration file')
    
    parser.add_argument('--frames', type=int, default=500,
                        help='Number of frames for synthetic data')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualizations during processing')
    
    return parser.parse_args()

def load_data(args):
    """Load calcium imaging data based on arguments."""
    loader = CalciumDataLoader()
    
    if args.data_type == 'sample':
        logger.info(f"Loading sample dataset: {args.dataset}")
        try:
            sample_data = loader.load_sample_dataset(args.dataset)
            return sample_data["data"]
        except Exception as e:
            logger.error(f"Error loading sample: {e}")
            logger.info("Downloading sample dataset...")
            try:
                sample_dir = loader.download_sample_dataset(args.dataset)
                sample_data = loader.load_sample_dataset(args.dataset)
                return sample_data["data"]
            except Exception as e:
                logger.error(f"Error loading sample after download: {e}")
                logger.info("Falling back to synthetic data")
                return create_synthetic_data(args)
    
    elif args.data_type == 'neurofinder':
        logger.info(f"Loading Neurofinder dataset: {args.dataset}")
        try:
            dataset = loader.load_neurofinder_dataset(args.dataset)
            return dataset["data"]
        except Exception as e:
            logger.error(f"Error loading Neurofinder dataset: {e}")
            logger.info("Downloading Neurofinder dataset...")
            try:
                dataset_dir = loader.download_neurofinder_dataset(args.dataset)
                dataset = loader.load_neurofinder_dataset(args.dataset)
                return dataset["data"]
            except Exception as e:
                logger.error(f"Error loading Neurofinder after download: {e}")
                logger.info("Falling back to synthetic data")
                return create_synthetic_data(args)
    
    elif args.data_type == 'synthetic':
        logger.info("Creating synthetic data")
        return create_synthetic_data(args)
    
    elif args.data_type == 'file':
        if not args.file_path:
            logger.error("No file path provided for 'file' data type")
            sys.exit(1)
        
        logger.info(f"Loading file: {args.file_path}")
        try:
            return loader.load_calcium_file(args.file_path)
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            sys.exit(1)
    
    else:
        logger.error(f"Unknown data type: {args.data_type}")
        sys.exit(1)

def create_synthetic_data(args):
    """Create synthetic calcium imaging data."""
    from neural_ai_network.calcium_imaging.test_calcium_agent import create_synthetic_calcium_data
    
    logger.info(f"Creating synthetic data with {args.frames} frames")
    data = create_synthetic_calcium_data(
        frames=args.frames,
        width=100,
        height=100,
        n_cells=15,
        noise_level=0.1
    )
    return data

def load_configuration(args):
    """Load configuration from file or use defaults."""
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            return json.load(f)
    else:
        logger.info("Using default configuration")
        return {}

def setup_output_directory(args):
    """Create output directory for results."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    return output_dir

def save_temp_data(data, output_dir):
    """Save data to temporary file for processing."""
    temp_file = output_dir / "temp_data.npy"
    np.save(temp_file, data)
    logger.info(f"Saved temporary data to {temp_file}")
    return temp_file

def process_data(agent, data_path, config):
    """Process data with the calcium agent."""
    logger.info("Processing data with calcium agent")
    result = agent.process_data(data_path, parameters=config)
    logger.info(f"Processing completed with status: {result['status']}")
    return result

def display_results(result, args):
    """Display and explain processing results."""
    if result['status'] != 'success':
        logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        return
    
    logger.info("=== Processing Results ===")
    logger.info(f"Cells detected: {result.get('cells_detected', 0)}")
    logger.info(f"Events detected: {result.get('events_detected', 0)}")
    
    if 'results_path' in result:
        results_dir = result['results_path']
        logger.info(f"Results saved to: {results_dir}")
        
        # List generated files
        result_files = list(Path(results_dir).glob("*.png"))
        if result_files:
            logger.info(f"Generated {len(result_files)} visualization files:")
            for f in result_files:
                logger.info(f"  - {f.name}")
            
            # Display images if requested
            if args.visualize:
                for image_file in sorted(result_files):
                    img = plt.imread(image_file)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.title(image_file.stem)
                    plt.tight_layout()
                    plt.show()

def analyze_results(agent, result):
    """Perform analysis on processing results."""
    if result['status'] != 'success' or 'results_path' not in result:
        return
    
    results_dir = result['results_path']
    signals_file = os.path.join(results_dir, "df_f_signals.npy")
    events_file = os.path.join(results_dir, "events.json")
    
    if os.path.exists(signals_file) and os.path.exists(events_file):
        logger.info("\n=== Analysis Results ===")
        
        # Load signals
        signals = {"df_f": np.load(signals_file)}
        
        # Get cell IDs
        cell_summary_file = os.path.join(results_dir, "cell_summary.json")
        if os.path.exists(cell_summary_file):
            with open(cell_summary_file, 'r') as f:
                cell_summary = json.load(f)
            signals["cell_ids"] = [c["id"] for c in cell_summary]
        else:
            signals["cell_ids"] = list(range(signals["df_f"].shape[0]))
        
        # Load events
        with open(events_file, 'r') as f:
            events = json.load(f)
        
        # Analyze activity
        logger.info("\nCALCIUM ACTIVITY ANALYSIS:")
        logger.info("=" * 40)
        activity_analysis = agent.analyze_activity(signals)
        logger.info(activity_analysis)
        
        # Analyze cell types
        logger.info("\nCELL TYPE ANALYSIS:")
        logger.info("=" * 40)
        cell_type_analysis = agent.analyze_cell_types(signals, events)
        logger.info(cell_type_analysis)

def main():
    """Main function to run the demo pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    logger.info("Starting Calcium Imaging Pipeline Demo")
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    
    # Load configuration
    config = load_configuration(args)
    
    # Initialize agent
    agent = CalciumProcessingAgent()
    
    try:
        # Load data
        data = load_data(args)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Save to temporary file for processing
        temp_file = save_temp_data(data, output_dir)
        
        # Process the data
        result = process_data(agent, temp_file, config)
        
        # Display results
        display_results(result, args)
        
        # Analyze results
        analyze_results(agent, result)
        
        logger.info("Pipeline demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()