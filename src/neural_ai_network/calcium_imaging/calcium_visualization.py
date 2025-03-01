#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Calcium Imaging Visualizations

This module provides advanced visualization tools for calcium imaging data,
including activity maps, correlation matrices, event raster plots, and more.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from typing import Dict, List, Tuple, Union, Optional
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CalciumVisualization")

class CalciumVisualizer:
    """Advanced visualization tools for calcium imaging data."""
    
    def __init__(self, output_dir: str = './calcium_visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger("CalciumVisualizer")
    
    def create_visualizations(self, results_path: str) -> List[str]:
        """
        Create a comprehensive set of visualizations from calcium processing results.
        
        Args:
            results_path: Path to results directory from calcium agent
            
        Returns:
            List of paths to created visualization files
        """
        self.logger.info(f"Creating visualizations from {results_path}")
        
        # Ensure the results path exists
        if not os.path.exists(results_path):
            self.logger.error(f"Results path does not exist: {results_path}")
            return []
        
        # Load necessary files
        cell_masks = self._load_npy_file(os.path.join(results_path, "cell_masks.npy"))
        std_projection = self._load_npy_file(os.path.join(results_path, "std_projection.npy"))
        max_projection = self._load_npy_file(os.path.join(results_path, "max_projection.npy"))
        raw_signals = self._load_npy_file(os.path.join(results_path, "raw_signals.npy"))
        df_f_signals = self._load_npy_file(os.path.join(results_path, "df_f_signals.npy"))
        
        # Load cell summary and events
        cell_summary = self._load_json_file(os.path.join(results_path, "cell_summary.json"))
        events = self._load_json_file(os.path.join(results_path, "events.json"))
        analysis = self._load_json_file(os.path.join(results_path, "analysis.json"))
        
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_dir, os.path.basename(results_path))
        os.makedirs(vis_dir, exist_ok=True)
        
        # Track created files
        created_files = []
        
        # Generate visualizations
        if std_projection is not None and cell_summary:
            # Activity map visualization
            activity_map_file = self.create_activity_map(
                std_projection, cell_summary, output_dir=vis_dir, 
                filename="activity_map.png")
            created_files.append(activity_map_file)
        
        if df_f_signals is not None and cell_summary:
            # Cell raster plot
            raster_file = self.create_raster_plot(
                df_f_signals, cell_summary, events, output_dir=vis_dir, 
                filename="calcium_raster.png")
            created_files.append(raster_file)
            
            # Correlation matrix
            corr_matrix_file = self.create_correlation_matrix(
                df_f_signals, cell_summary, output_dir=vis_dir, 
                filename="correlation_matrix.png")
            created_files.append(corr_matrix_file)
            
            # Activity trace plot
            trace_file = self.create_activity_traces(
                df_f_signals, cell_summary, events, output_dir=vis_dir, 
                filename="activity_traces.png")
            created_files.append(trace_file)
        
        if analysis and "temporal_statistics" in analysis:
            # Population activity plot
            pop_activity_file = self.create_population_activity_plot(
                analysis["temporal_statistics"], output_dir=vis_dir, 
                filename="population_activity.png")
            created_files.append(pop_activity_file)
        
        if cell_masks is not None and max_projection is not None:
            # ROI map visualization
            roi_map_file = self.create_roi_map(
                max_projection, cell_masks, output_dir=vis_dir, 
                filename="roi_map.png")
            created_files.append(roi_map_file)
        
        return created_files
    
    def create_activity_map(self, background: np.ndarray, cell_summary: List[Dict], 
                           output_dir: str = None, filename: str = "activity_map.png") -> str:
        """
        Create an activity map showing cells colored by their activity level.
        
        Args:
            background: Background image (typically std projection or max projection)
            cell_summary: List of cell dictionaries with activity information
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating activity map visualization")
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Show background
        plt.imshow(background, cmap='gray')
        
        # Define colormap for activity (blue to red)
        activity_cmap = plt.cm.plasma
        
        # Get activity values (max_df_f or event_count)
        max_df_f_values = [cell.get("max_df_f", 0) for cell in cell_summary]
        if max(max_df_f_values) > 0:
            activity_values = max_df_f_values
            activity_label = "Max ΔF/F"
        else:
            activity_values = [cell.get("event_count", 0) for cell in cell_summary]
            activity_label = "Event Count"
        
        # Normalize activity for color mapping
        max_activity = max(activity_values) if activity_values else 1
        norm_activity = [a / max_activity for a in activity_values]
        
        # Plot cell circles colored by activity
        for i, cell in enumerate(cell_summary):
            activity = norm_activity[i]
            x, y = cell["x"], cell["y"]
            radius = cell.get("radius", 5)
            
            # Create circle with activity color
            circle = plt.Circle((x, y), radius, fill=True, alpha=0.7,
                               color=activity_cmap(activity))
            plt.gca().add_patch(circle)
            
            # Add border
            border = plt.Circle((x, y), radius, fill=False, alpha=1,
                               edgecolor='white', linewidth=1)
            plt.gca().add_patch(border)
            
            # Add cell ID for larger figures
            if len(cell_summary) < 50:
                plt.text(x, y, str(cell["id"]), color='white', 
                        fontsize=8, ha='center', va='center')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=activity_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label(activity_label)
        
        # Add title and labels
        plt.title(f"Cell Activity Map (n={len(cell_summary)} cells)")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved activity map to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def create_raster_plot(self, df_f: np.ndarray, cell_summary: List[Dict], 
                          events: Dict, output_dir: str = None, 
                          filename: str = "calcium_raster.png") -> str:
        """
        Create a raster plot showing calcium activity and events for all cells.
        
        Args:
            df_f: ΔF/F signals array [cells, time]
            cell_summary: List of cell dictionaries
            events: Dictionary of events for each cell
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating calcium raster plot")
        
        n_cells, n_frames = df_f.shape
        
        # Create figure - make height adaptive to number of cells
        height = max(6, min(20, n_cells * 0.25))
        plt.figure(figsize=(14, height))
        
        # Create custom colormap for the raster
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
        n_bins = 100
        cmap_name = 'calcium_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        # Normalize ΔF/F values for the colormap
        vmin = np.percentile(df_f, 1)
        vmax = np.percentile(df_f, 99)
        
        # Plot the raster
        plt.imshow(df_f, aspect='auto', interpolation='nearest', 
                  cmap=cm, vmin=vmin, vmax=vmax)
        
        # Mark events if available
        if events:
            for cell_idx, cell in enumerate(cell_summary):
                cell_id = str(cell["id"])
                if cell_id in events:
                    for event in events[cell_id]:
                        start = event["start_frame"]
                        end = event["end_frame"]
                        # Draw a line at the start of each event
                        plt.axvline(x=start, ymin=(cell_idx)/n_cells, 
                                   ymax=(cell_idx+1)/n_cells, 
                                   color='green', linewidth=1, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('ΔF/F')
        
        # Set ticks and labels
        plt.xlabel('Frame')
        plt.ylabel('Cell')
        
        # Set y-ticks to cell IDs
        plt.yticks(range(n_cells), [str(cell["id"]) for cell in cell_summary])
        
        # If too many cells, limit the y-ticks
        if n_cells > 20:
            plt.yticks(np.arange(0, n_cells, 5), [str(cell_summary[i]["id"]) for i in range(0, n_cells, 5)])
        
        # Set title
        plt.title(f"Calcium Activity Raster (n={n_cells} cells, {n_frames} frames)")
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved raster plot to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def create_correlation_matrix(self, df_f: np.ndarray, cell_summary: List[Dict], 
                                 output_dir: str = None, 
                                 filename: str = "correlation_matrix.png") -> str:
        """
        Create a correlation matrix visualization of cell-cell activity correlations.
        
        Args:
            df_f: ΔF/F signals array [cells, time]
            cell_summary: List of cell dictionaries
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating correlation matrix visualization")
        
        n_cells = df_f.shape[0]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(df_f)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot correlation matrix
        im = plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation')
        
        # Set ticks and labels
        plt.xticks(range(n_cells), [str(cell["id"]) for cell in cell_summary], rotation=90)
        plt.yticks(range(n_cells), [str(cell["id"]) for cell in cell_summary])
        
        # If too many cells, limit the ticks
        if n_cells > 20:
            step = max(1, n_cells // 20)
            plt.xticks(range(0, n_cells, step), 
                      [str(cell_summary[i]["id"]) for i in range(0, n_cells, step)], 
                      rotation=90)
            plt.yticks(range(0, n_cells, step), 
                      [str(cell_summary[i]["id"]) for i in range(0, n_cells, step)])
        
        # Add title
        plt.title(f"Cell-Cell Activity Correlation Matrix (n={n_cells} cells)")
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved correlation matrix to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def create_activity_traces(self, df_f: np.ndarray, cell_summary: List[Dict],
                              events: Dict, output_dir: str = None,
                              filename: str = "activity_traces.png") -> str:
        """
        Create activity trace plots for selected cells.
        
        Args:
            df_f: ΔF/F signals array [cells, time]
            cell_summary: List of cell dictionaries
            events: Dictionary of events for each cell
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating activity trace visualization")
        
        n_cells, n_frames = df_f.shape
        
        # Select a subset of cells to display (max 10)
        max_cells = min(10, n_cells)
        
        # Try to select cells with events if possible
        cells_with_events = []
        for cell_idx, cell in enumerate(cell_summary):
            cell_id = str(cell["id"])
            if cell_id in events and events[cell_id]:
                cells_with_events.append(cell_idx)
        
        # If we have enough cells with events, use those
        if len(cells_with_events) >= max_cells:
            selected_indices = cells_with_events[:max_cells]
        else:
            # Otherwise select a mix of cells with and without events
            selected_indices = cells_with_events + list(range(max_cells - len(cells_with_events)))
            selected_indices = selected_indices[:max_cells]
        
        # Create figure
        plt.figure(figsize=(12, 2 * max_cells))
        
        # Plot each selected cell
        for i, cell_idx in enumerate(selected_indices):
            plt.subplot(max_cells, 1, i + 1)
            
            # Get cell ID and trace
            cell_id = str(cell_summary[cell_idx]["id"])
            trace = df_f[cell_idx]
            
            # Plot trace
            plt.plot(trace, 'b-')
            
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Mark events if available
            if cell_id in events:
                for event in events[cell_id]:
                    start = event["start_frame"]
                    end = event["end_frame"]
                    plt.axvspan(start, end, color='r', alpha=0.3)
                    
                    # Mark peak
                    if "peak_frame" in event and event["peak_frame"] is not None:
                        plt.plot(event["peak_frame"], trace[event["peak_frame"]], 'ro', markersize=4)
            
            # Add cell ID label
            plt.ylabel(f"Cell {cell_id}")
            
            # Remove x-axis labels except for the last subplot
            if i < max_cells - 1:
                plt.xticks([])
            else:
                plt.xlabel("Frame")
        
        # Add title
        plt.suptitle("Calcium Activity Traces with Detected Events", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved activity traces to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def create_population_activity_plot(self, temporal_stats: Dict, 
                                       output_dir: str = None,
                                       filename: str = "population_activity.png") -> str:
        """
        Create a plot of population activity over time.
        
        Args:
            temporal_stats: Temporal statistics from analysis results
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating population activity plot")
        
        # Extract activity profile from temporal statistics
        activity_profile = temporal_stats.get("activity_profile", [])
        n_frames = len(activity_profile)
        
        if n_frames == 0:
            self.logger.warning("No activity profile data available")
            return ""
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot activity profile
        plt.plot(activity_profile, 'b-', linewidth=1.5)
        
        # Add a horizontal line at the mean activity
        mean_activity = temporal_stats.get("mean_activity", 0)
        plt.axhline(y=mean_activity, color='k', linestyle='--', 
                   linewidth=1, label=f"Mean ({mean_activity:.3f})")
        
        # Mark population events if available
        pop_events = temporal_stats.get("population_event_frames", [])
        if pop_events:
            for event_frame in pop_events:
                plt.axvline(x=event_frame, color='r', linestyle='-', linewidth=1, alpha=0.6)
            
            # Add annotation for the first few events
            for i, event_frame in enumerate(pop_events[:min(5, len(pop_events))]):
                if event_frame < n_frames:
                    plt.annotate(f"Event {i+1}", 
                                xy=(event_frame, activity_profile[event_frame]),
                                xytext=(10, 10), textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        # Add labels and title
        plt.xlabel("Frame")
        plt.ylabel("Fraction of Active Cells")
        plt.title("Population Activity Over Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved population activity plot to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def create_roi_map(self, background: np.ndarray, cell_masks: np.ndarray,
                      output_dir: str = None, filename: str = "roi_map.png") -> str:
        """
        Create a visualization of ROI masks for all detected cells.
        
        Args:
            background: Background image (max projection)
            cell_masks: Cell mask array with cell IDs
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating ROI map visualization")
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Display background
        plt.imshow(background, cmap='gray')
        
        # Create colormap for cell masks
        mask_cmap = plt.cm.nipy_spectral
        
        # Create transparent mask overlay
        mask_overlay = np.zeros((*background.shape, 4))  # RGBA
        
        # Get unique cell IDs (excluding 0 which is background)
        cell_ids = np.unique(cell_masks)
        cell_ids = cell_ids[cell_ids > 0]
        
        # Assign colors to each cell
        for i, cell_id in enumerate(cell_ids):
            # Get mask for this cell
            mask = cell_masks == cell_id
            
            # Calculate centroid
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                centroid_y = int(np.mean(y_indices))
                centroid_x = int(np.mean(x_indices))
                
                # Assign color to this cell
                color = mask_cmap(i / max(1, len(cell_ids) - 1))
                
                # Add to overlay with transparency
                mask_overlay[mask] = (*color[:3], 0.5)  # RGB + alpha
                
                # Add cell ID text
                plt.text(centroid_x, centroid_y, str(cell_id), color='white', 
                        fontsize=8, ha='center', va='center')
        
        # Add the overlay
        plt.imshow(mask_overlay)
        
        # Add title
        plt.title(f"Cell ROI Map (n={len(cell_ids)} cells)")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved ROI map to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def create_event_detection_visualization(self, trace: np.ndarray, events: List[Dict],
                                           deconvolved: Optional[np.ndarray] = None,
                                           output_dir: str = None, 
                                           filename: str = "event_detection.png") -> str:
        """
        Create a visualization of detected events on a calcium trace.
        
        Args:
            trace: Calcium ΔF/F trace
            events: List of detected events
            deconvolved: Optional deconvolved trace
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Creating event detection visualization")
        
        # Create figure
        fig_height = 6 if deconvolved is None else 9
        plt.figure(figsize=(12, fig_height))
        
        # Plot original trace
        if deconvolved is None:
            plt.subplot(1, 1, 1)
        else:
            plt.subplot(2, 1, 1)
            
        plt.plot(trace, 'b-', linewidth=1.5)
        
        # Mark events
        for event in events:
            start_frame = event["start_frame"]
            end_frame = event["end_frame"]
            peak_frame = event.get("peak_frame", None)
            
            # Shade event region
            plt.axvspan(start_frame, end_frame, color='r', alpha=0.3)
            
            # Mark peak if available
            if peak_frame is not None:
                plt.plot(peak_frame, trace[peak_frame], 'ro', markersize=5)
                
                # Add annotation with amplitude for first few events
                if events.index(event) < 5:
                    plt.annotate(f"{event.get('amplitude', 0):.2f}", 
                                xy=(peak_frame, trace[peak_frame]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8)
        
        # Add title and labels
        plt.title("Calcium Trace with Detected Events")
        plt.xlabel("Frame")
        plt.ylabel("ΔF/F")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add deconvolved trace if provided
        if deconvolved is not None:
            plt.subplot(2, 1, 2)
            plt.plot(deconvolved, 'g-', linewidth=1.5)
            
            # Mark event onsets on deconvolved trace
            for event in events:
                plt.axvline(x=event["start_frame"], color='r', linestyle='--', linewidth=1)
            
            plt.title("Deconvolved Trace")
            plt.xlabel("Frame")
            plt.ylabel("Deconvolved Signal")
            plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved event detection visualization to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close()
            return ""
    
    def _load_npy_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load NumPy file safely."""
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            return np.load(file_path)
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def _load_json_file(self, file_path: str) -> Optional[Dict]:
        """Load JSON file safely."""
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        visualizer = CalciumVisualizer()
        created_files = visualizer.create_visualizations(results_path)
        
        print(f"Created {len(created_files)} visualization files:")
        for file_path in created_files:
            print(f"  - {file_path}")
    else:
        print("Please provide a path to calcium processing results")