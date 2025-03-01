#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Calcium Imaging Data Processing Script

This script downloads and processes real calcium imaging data from public sources
including Allen Brain Observatory and Neurofinder datasets.
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from neural_ai_network.calcium.calcium_agent import CalciumProcessingAgent
from neural_ai_network.calcium.calcium_data_loader import CalciumDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RealDataProcessor")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Real Calcium Imaging Data')
    
    parser.add_argument('--dataset-source', type=str, default='neurofinder',
                       choices=['neurofinder', 'allen', 'crcns'],
                       help='Source of the dataset')
    
    parser.add_argument('--dataset-id', type=str, default='00.00',
                       help='Dataset ID (e.g., "00.00" for Neurofinder)')
    
    parser.add_argument('--output-dir', type=str, default='./calcium_real_data_results',
                       help='Directory to save results')
    
    parser.add_argument('--frames-limit', type=int, default=500,
                       help='Maximum number of frames to process (for speed)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualizations during processing')
    
    return parser.parse_args()

def process_neurofinder_data(loader, dataset_id, frames_limit, output_dir):
    """Process a Neurofinder dataset."""
    logger.info(f"Processing Neurofinder dataset {dataset_id}")
    
    try:
        # Download if not already available
        try:
            dataset = loader.load_neurofinder_dataset(dataset_id)
        except Exception as e:
            logger.warning(f"Error loading dataset: {e}")
            logger.info(f"Downloading Neurofinder dataset {dataset_id}")
            dataset_dir = loader.download_neurofinder_dataset(dataset_id)
            dataset = loader.load_neurofinder_dataset(dataset_id)
        
        # Extract data and limit frames if needed
        data = dataset["data"]
        if frames_limit > 0 and data.shape[0] > frames_limit:
            logger.info(f"Limiting dataset from {data.shape[0]} to {frames_limit} frames")
            data = data[:frames_limit]
        
        # Get ground truth regions
        regions = dataset.get("regions", [])
        logger.info(f"Dataset has {len(regions)} ground truth regions")
        
        return data, regions
    
    except Exception as e:
        logger.error(f"Error processing Neurofinder data: {e}")
        return None, []

def process_allen_data(loader, exp_id, frames_limit, output_dir):
    """Process Allen Brain Observatory data."""
    logger.info(f"Processing Allen Brain data {exp_id}")
    
    try:
        # Load data from Allen Brain Observatory
        allen_data = loader.load_allen_brain_data(exp_id)
        
        if "error" in allen_data:
            logger.error(f"Error loading Allen data: {allen_data['error']}")
            logger.info(allen_data.get("instructions", ""))
            return None, []
        
        # Extract dF/F traces and ROI masks
        dff_traces = allen_data.get("dff_traces")
        roi_masks = allen_data.get("roi_masks")
        max_projection = allen_data.get("max_projection")
        
        if dff_traces is None or roi_masks is None:
            logger.error("Missing required data (traces or ROI masks)")
            return None, []
        
        # Convert from traces to movie format if needed
        # In Allen data, we might need to reconstruct the movie from ROIs and traces
        # This is a simplified approach - assumes we have the max projection
        if max_projection is not None:
            # Create a simple synthetic movie based on the max projection and traces
            n_frames = min(frames_limit, dff_traces[0].shape[0]) if frames_limit > 0 else dff_traces[0].shape[0]
            height, width = max_projection.shape
            
            # Initialize movie with base fluorescence from max projection
            movie = np.zeros((n_frames, height, width), dtype=np.float32)
            base_level = np.percentile(max_projection, 50)  # Use median as baseline
            
            # Add activity for each ROI
            for i, mask in enumerate(roi_masks):
                if i < len(dff_traces[0]):
                    # Scale activity to make it visible
                    trace = dff_traces[0][i, :n_frames]
                    for t in range(n_frames):
                        # Add normalized trace to the mask regions
                        activity = trace[t] * base_level * 0.5  # Scale factor
                        movie[t][mask] = max_projection[mask] + activity
            
            # Make sure values are positive and not too extreme
            movie = np.clip(movie, 0, np.percentile(max_projection, 99.9) * 2)
            
            data = movie
            
            # Create synthetic regions from ROI masks
            regions = []
            for i, mask in enumerate(roi_masks):
                # Extract the coordinates from the mask
                coords = np.array(np.where(mask)).T  # y, x format
                if len(coords) > 0:
                    regions.append({
                        "id": i,
                        "coordinates": coords.tolist()
                    })
            
            return data, regions
        else:
            logger.error("Missing max projection, cannot reconstruct movie")
            return None, []
    
    except Exception as e:
        logger.error(f"Error processing Allen data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, []

def process_crcns_data(loader, dataset_id, frames_limit, output_dir):
    """Process CRCNS dataset."""
    logger.info(f"Getting instructions for CRCNS data")
    
    # For CRCNS data, we just provide instructions as they require registration
    instructions_path = loader.download_crcns_data()
    logger.info(f"See instructions at: {instructions_path}")
    
    logger.info("Since CRCNS data requires registration, using a synthetic dataset instead")
    
    # Create a synthetic dataset with cell-like properties
    n_frames = 500 if frames_limit <= 0 else frames_limit
    height, width = 200, 200
    n_cells = 20
    
    # Create synthetic data with more realistic calcium dynamics
    from neural_ai_network.calcium.test_calcium_agent import create_synthetic_calcium_data
    data = create_synthetic_calcium_data(
        frames=n_frames,
        width=width,
        height=height,
        n_cells=n_cells,
        noise_level=0.05
    )
    
    # Create synthetic regions
    regions = []
    # We'd need positions from the synthetic data generation
    # For now, return empty list
    
    return data, regions

def save_temp_data(data, output_dir):
    """Save data to temporary file for processing."""
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, "real_data_temp.npy")
    np.save(temp_file, data)
    logger.info(f"Saved temporary data to {temp_file}")
    return temp_file

def process_with_agent(agent, data_path, output_dir):
    """Process data with the calcium agent."""
    logger.info("Processing data with calcium agent")
    
    # Custom parameters to improve processing of real data
    parameters = {
        "preprocessing": {
            "spatial_filter": "gaussian",
            "spatial_filter_size": 2,  # Smaller filter for real data
            "motion_correction": "ecc",
            "background_removal": "percentile",
            "background_percentile": 10.0
        },
        "cell_detection": {
            "method": "watershed",
            "min_cell_size": 20,
            "max_cell_size": 400,
            "threshold": 1.2,  # Slightly lower threshold for real data
            "min_distance": 8
        },
        "signal_extraction": {
            "roi_expansion": 2,
            "neuropil_correction": True
        },
        "event_detection": {
            "threshold_std": 2.0,  # More sensitive event detection
            "min_duration": 2
        },
        "storage": {
            "save_interim": True,
            "interim_dir": os.path.join(output_dir, "interim"),
            "processed_dir": os.path.join(output_dir, "processed"),
            "results_dir": os.path.join(output_dir, "results")
        }
    }
    
    # Process the data
    start_time = time.time()
    result = agent.process_data(data_path, parameters=parameters)
    processing_time = time.time() - start_time
    
    logger.info(f"Processing completed in {processing_time:.1f} seconds with status: {result['status']}")
    return result

def compare_with_ground_truth(result, ground_truth_regions, output_dir):
    """Compare detected cells with ground truth if available."""
    if not ground_truth_regions or 'results_path' not in result:
        logger.info("No ground truth comparison available")
        return
    
    logger.info(f"Comparing detected cells with {len(ground_truth_regions)} ground truth regions")
    
    # Load detected cell information
    cell_summary_file = os.path.join(result['results_path'], "cell_summary.json")
    if not os.path.exists(cell_summary_file):
        logger.warning("Cell summary file not found, cannot compare with ground truth")
        return
    
    import json
    with open(cell_summary_file, 'r') as f:
        detected_cells = json.load(f)
    
    # Load cell masks to get the exact cell shapes
    cell_masks_file = os.path.join(result['results_path'], "cell_masks.npy")
    if os.path.exists(cell_masks_file):
        cell_masks = np.load(cell_masks_file)
    else:
        logger.warning("Cell masks file not found, using center points only")
        cell_masks = None
    
    # Create a visualization of ground truth vs detected cells
    try:
        # Load standard deviation projection for background
        std_projection_file = os.path.join(result['results_path'], "std_projection.npy")
        if os.path.exists(std_projection_file):
            background = np.load(std_projection_file)
        else:
            logger.warning("Standard deviation projection not found, using blank background")
            # Estimate size from the first ground truth region
            if ground_truth_regions and 'coordinates' in ground_truth_regions[0]:
                coords = np.array(ground_truth_regions[0]['coordinates'])
                height = int(np.max(coords[:, 0])) + 50
                width = int(np.max(coords[:, 1])) + 50
            else:
                height, width = 512, 512
            background = np.zeros((height, width))
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Show background
        plt.imshow(background, cmap='gray')
        
        # Plot ground truth regions
        for region in ground_truth_regions:
            if 'coordinates' in region:
                coords = np.array(region['coordinates'])
                if coords.shape[1] >= 2:  # Make sure we have x,y coordinates
                    plt.scatter(coords[:, 1], coords[:, 0], s=1, color='blue', alpha=0.5)
        
        # Plot detected cells
        for cell in detected_cells:
            plt.scatter(cell['x'], cell['y'], s=30, facecolors='none', edgecolors='red')
            circle = plt.Circle((cell['x'], cell['y']), cell['radius'], 
                              fill=False, color='red', linewidth=1)
            plt.gca().add_patch(circle)
        
        plt.title(f"Cell Detection Comparison\nBlue: Ground Truth ({len(ground_truth_regions)} regions) | Red: Detected ({len(detected_cells)} cells)")
        plt.tight_layout()
        
        # Save the figure
        comparison_file = os.path.join(output_dir, "ground_truth_comparison.png")
        plt.savefig(comparison_file, dpi=150)
        logger.info(f"Saved ground truth comparison to {comparison_file}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating ground truth comparison: {e}")

def main():
    """Main function to run real data processing."""
    # Parse arguments
    args = parse_arguments()
    
    logger.info(f"Starting Real Calcium Imaging Data Processing: {args.dataset_source}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    # Initialize data loader and agent
    loader = CalciumDataLoader()
    agent = CalciumProcessingAgent()
    
    try:
        # Process data based on source
        if args.dataset_source == 'neurofinder':
            data, ground_truth = process_neurofinder_data(
                loader, args.dataset_id, args.frames_limit, output_dir)
        elif args.dataset_source == 'allen':
            data, ground_truth = process_allen_data(
                loader, args.dataset_id, args.frames_limit, output_dir)
        elif args.dataset_source == 'crcns':
            data, ground_truth = process_crcns_data(
                loader, args.dataset_id, args.frames_limit, output_dir)
        else:
            logger.error(f"Unknown dataset source: {args.dataset_source}")
            sys.exit(1)
        
        if data is None:
            logger.error("Failed to load data")
            sys.exit(1)
        
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Save data to temporary file
        temp_file = save_temp_data(data, output_dir)
        
        # Process with agent
        result = process_with_agent(agent, temp_file, output_dir)
        
        # Compare with ground truth if available
        if ground_truth:
            compare_with_ground_truth(result, ground_truth, output_dir)
        
        # Display results
        if result['status'] == 'success':
            logger.info(f"Cells detected: {result.get('cells_detected', 0)}")
            logger.info(f"Events detected: {result.get('events_detected', 0)}")
            
            if 'results_path' in result:
                logger.info(f"Results saved to: {result['results_path']}")
                
                # List generated files
                if os.path.exists(result['results_path']):
                    result_files = list(Path(result['results_path']).glob("*.png"))
                    if result_files:
                        logger.info(f"Generated {len(result_files)} visualization files")
                        for f in result_files:
                            logger.info(f"  - {f.name}")
        else:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        
        logger.info("Real data processing completed")
        
    except Exception as e:
        logger.error(f"Error processing real data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()