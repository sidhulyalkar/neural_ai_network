# neural_visualizer.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import mne
import nibabel as nib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
from scipy import signal

class NeuralDataVisualizer:
    """
    Visualization module for neural data analysis.
    
    This module provides visualization capabilities for different neural data modalities,
    including EEG, fMRI, calcium imaging, and more.
    """
    
    def __init__(self, config_path: str = "visualizer_config.json"):
        """
        Initialize the visualizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Set default styling
        sns.set(style="whitegrid")
        
        # Ensure output directory exists
        os.makedirs(self.config["storage"]["visualizations"], exist_ok=True)
        
        self.logger.info("Neural Data Visualizer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("Visualizer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                "styling": {
                    "color_palette": "viridis",
                    "figure_size": [10, 6],
                    "dpi": 100,
                    "font_size": 12
                },
                "eeg": {
                    "time_series": {
                        "n_channels": 10,
                        "duration": 5
                    },
                    "topographic": {
                        "times": "auto",
                        "resolution": 300
                    },
                    "spectral": {
                        "fmin": 0,
                        "fmax": 50
                    }
                },
                "fmri": {
                    "slice_view": {
                        "n_cuts": 5,
                        "orientation": "axial"
                    },
                    "volumetric": {
                        "threshold": 0.5,
                        "colormap": "hot"
                    }
                },
                "calcium_imaging": {
                    "cell_map": {
                        "alpha": 0.7,
                        "contour_width": 1.5
                    },
                    "activity_trace": {
                        "n_cells": 10,
                        "window": 60
                    }
                },
                "storage": {
                    "visualizations": "./visualizations"
                }
            }
    
    def visualize(self, data_path: str, modality: str, vis_type: str, 
                 output_format: str = "png", parameters: Dict = None) -> str:
        """
        Generate visualization for neural data.
        
        Args:
            data_path: Path to data file or directory
            modality: Neural data modality ('eeg', 'fmri', etc.)
            vis_type: Type of visualization
            output_format: Output format ('png', 'svg', 'html', 'base64')
            parameters: Optional visualization parameters
            
        Returns:
            Path to generated visualization or base64 string
        """
        self.logger.info(f"Generating {vis_type} visualization for {modality} data from {data_path}")
        
        # Merge configuration with provided parameters
        config = self.config[modality].get(vis_type, {}).copy()
        if parameters:
            for key, value in parameters.items():
                config[key] = value
        
        # Generate visualization based on modality
        if modality == "eeg":
            result = self._visualize_eeg(data_path, vis_type, config, output_format)
        elif modality == "fmri":
            result = self._visualize_fmri(data_path, vis_type, config, output_format)
        elif modality == "calcium_imaging":
            result = self._visualize_calcium(data_path, vis_type, config, output_format)
        elif modality == "connectivity":
            result = self._visualize_connectivity(data_path, vis_type, config, output_format)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        return result
    
    def visualize_results(self, results_path: str, vis_type: str, 
                         output_format: str = "png", parameters: Dict = None) -> str:
        """
        Generate visualization for analysis results.
        
        Args:
            results_path: Path to results file
            vis_type: Type of visualization
            output_format: Output format ('png', 'svg', 'html', 'base64')
            parameters: Optional visualization parameters
            
        Returns:
            Path to generated visualization or base64 string
        """
        self.logger.info(f"Generating {vis_type} visualization for results from {results_path}")
        
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Determine modality from results
        modality = results.get("metadata", {}).get("modality", "unknown")
        
        # Generate visualization based on type
        if vis_type == "feature_importance":
            return self._visualize_feature_importance(results, output_format, parameters)
        elif vis_type == "confusion_matrix":
            return self._visualize_confusion_matrix(results, output_format, parameters)
        elif vis_type == "performance_metrics":
            return self._visualize_performance_metrics(results, output_format, parameters)
        elif vis_type == "training_history":
            return self._visualize_training_history(results, output_format, parameters)
        else:
            # Try modality-specific visualization
            return self.visualize(results_path, modality, vis_type, output_format, parameters)
    
    def _visualize_eeg(self, data_path: str, vis_type: str, config: Dict, 
                      output_format: str) -> str:
        """
        Generate visualization for EEG data.
        
        Args:
            data_path: Path to EEG data file
            vis_type: Type of visualization
            config: Visualization configuration
            output_format: Output format
            
        Returns:
            Path to generated visualization or base64 string
        """
        # Load data
        data = self._load_eeg_data(data_path)
        
        if vis_type == "time_series":
            return self._visualize_eeg_time_series(data, config, output_format)
        elif vis_type == "topographic":
            return self._visualize_eeg_topographic(data, config, output_format)
        elif vis_type == "spectral":
            return self._visualize_eeg_spectral(data, config, output_format)
        elif vis_type == "erp":
            return self._visualize_eeg_erp(data, config, output_format)
        else:
            raise ValueError(f"Unsupported EEG visualization type: {vis_type}")
    
    def _load_eeg_data(self, data_path: str) -> Union[mne.io.Raw, Dict]:
        """
        Load EEG data for visualization.
        
        Args:
            data_path: Path to EEG data file
            
        Returns:
            MNE Raw object or dictionary with EEG data
        """
        # Handle different file types
        if data_path.endswith('.edf') or data_path.endswith('.bdf'):
            return mne.io.read_raw_edf(data_path, preload=True)
        elif data_path.endswith('.fif'):
            return mne.io.read_raw_fif(data_path, preload=True)
        elif data_path.endswith('.json'):
            # Load processed results
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def _visualize_eeg_time_series(self, data: Union[mne.io.Raw, Dict], 
                                  config: Dict, output_format: str) -> str:
        """
        Generate time series visualization for EEG data.
        
        Args:
            data: EEG data
            config: Visualization configuration
            output_format: Output format
            
        Returns:
            Path to generated visualization or base64 string
        """
        # Create figure
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        if isinstance(data, mne.io.Raw):
            # MNE Raw object
            n_channels = min(config.get("n_channels", 10), len(data.ch_names))
            duration = config.get("duration", 5)
            
            # Get selected channels
            channels = data.ch_names[:n_channels]
            start_time = 0
            
            # Plot time series
            data_subset = data.copy().pick_channels(channels)
            data_subset.plot(
                duration=duration,
                start=start_time,
                n_channels=n_channels,
                scalings='auto',
                title='EEG Time Series',
                show=False,
                fig=fig
            )
        else:
            # Processed data in dictionary
            if "features" in data and "erp" in data["features"]:
                # Plot ERP data
                erp_data = data["features"]["erp"]
                times = erp_data["times"]
                channel_data = erp_data["evoked_data"]
                channels = erp_data["channel_names"][:config.get("n_channels", 10)]
                
                for i, channel in enumerate(channels):
                    plt.plot(times, channel_data[i], label=channel)
                
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (µV)')
                plt.title('EEG Event-Related Potentials')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
            else:
                # Fallback to band powers
                if "features" in data and "band_power" in data["features"]:
                    band_powers = data["features"]["band_power"]["band_powers"]
                    channels = data["features"]["band_power"]["channel_names"]
                    
                    # Create bar plot of band powers
                    bands = list(band_powers.keys())
                    n_bands = len(bands)
                    n_channels = min(config.get("n_channels", 10), len(channels))
                    
                    fig, ax = plt.subplots(figsize=tuple(self.config["styling"]["figure_size"]))
                    x = np.arange(n_channels)
                    width = 0.8 / n_bands
                    
                    for i, band in enumerate(bands):
                        powers = band_powers[band][:n_channels]
                        ax.bar(x + i*width, powers, width, label=band)
                    
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Power')
                    ax.set_title('EEG Band Powers')
                    ax.set_xticks(x + width * (n_bands - 1) / 2)
                    ax.set_xticklabels(channels[:n_channels])
                    ax.legend()
                    plt.tight_layout()
                else:
                    plt.text(0.5, 0.5, "No suitable EEG data found for visualization",
                            ha='center', va='center', fontsize=12)
        
        # Save or return visualization
        return self._save_or_return_figure(fig, "eeg_time_series", output_format)
    
    def _visualize_eeg_topographic(self, data: Union[mne.io.Raw, Dict], 
                                  config: Dict, output_format: str) -> str:
        """
        Generate topographic map visualization for EEG data.
        
        This is a placeholder implementation. In a real system, you would use
        MNE's plotting functions with actual channel positions.
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        if isinstance(data, mne.io.Raw):
            # Check for montage information
            if data.info['dig'] is None:
                # Try to set a standard montage
                try:
                    montage = mne.channels.make_standard_montage('standard_1020')
                    data.set_montage(montage)
                except Exception as e:
                    self.logger.warning(f"Could not set standard montage: {e}")
                    plt.text(0.5, 0.5, "No channel position information available for topographic plot",
                            ha='center', va='center', fontsize=12)
                    return self._save_or_return_figure(fig, "eeg_topographic", output_format)
            
            # Create evoked data for plotting topography (use average of all data)
            epoch_data = mne.make_fixed_length_epochs(data, duration=1.0).average()
            
            times = config.get("times", "auto")
            if times == "auto":
                times = [epoch_data.times[len(epoch_data.times) // 2]]
            
            # Plot topography
            epoch_data.plot_topomap(
                times=times,
                ch_type='eeg',
                time_unit='s',
                title='EEG Topographic Map',
                show=False,
                fig=fig
            )
        else:
            # Simplified topographic plot using band powers if available
            if "features" in data and "band_power" in data["features"]:
                plt.text(0.5, 0.5, "Topographic plot requires channel position information",
                        ha='center', va='center', fontsize=12)
            else:
                plt.text(0.5, 0.5, "No suitable EEG data found for topographic visualization",
                        ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "eeg_topographic", output_format)
    
    def _visualize_eeg_spectral(self, data: Union[mne.io.Raw, Dict], 
                               config: Dict, output_format: str) -> str:
        """
        Generate spectral visualization for EEG data.
        
        Args:
            data: EEG data
            config: Visualization configuration
            output_format: Output format
            
        Returns:
            Path to generated visualization or base64 string
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        if isinstance(data, mne.io.Raw):
            # Calculate PSDs
            fmin = config.get("fmin", 0)
            fmax = config.get("fmax", 50)
            n_channels = min(config.get("n_channels", 10), len(data.ch_names))
            
            psds, freqs = mne.time_frequency.psd_welch(
                data.pick_channels(data.ch_names[:n_channels]),
                fmin=fmin,
                fmax=fmax,
                n_fft=int(data.info['sfreq'] * 2),
                n_overlap=int(data.info['sfreq'])
            )
            
            # Plot PSDs
            plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
            for i in range(n_channels):
                plt.semilogy(freqs, psds[i], label=data.ch_names[i])
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (µV²/Hz)')
            plt.title('EEG Power Spectrum')
            plt.legend()
            plt.xlim([fmin, fmax])
            plt.tight_layout()
        else:
            # Placeholder for processed data
            plt.text(0.5, 0.5, "Spectral visualization requires raw EEG data",
                    ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "eeg_spectral", output_format)
    
    def _visualize_eeg_erp(self, data: Union[mne.io.Raw, Dict], 
                          config: Dict, output_format: str) -> str:
        """
        Generate ERP visualization for EEG data.
        
        Args:
            data: EEG data
            config: Visualization configuration
            output_format: Output format
            
        Returns:
            Path to generated visualization or base64 string
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        if isinstance(data, dict) and "features" in data and "erp" in data["features"]:
            # Plot ERP data from processed results
            erp_data = data["features"]["erp"]
            times = erp_data["times"]
            channel_data = erp_data["evoked_data"]
            channels = erp_data["channel_names"]
            
            n_channels = min(config.get("n_channels", 5), len(channels))
            
            plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
            for i in range(n_channels):
                plt.plot(times, channel_data[i], label=channels[i])
            
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (µV)')
            plt.title('Event-Related Potentials')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        elif isinstance(data, mne.io.Raw):
            # Create mock epochs for visualization
            events = mne.make_fixed_length_events(data, duration=1.0)
            epochs = mne.Epochs(data, events, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0), preload=True)
            evoked = epochs.average()
            
            # Plot ERP
            evoked.plot(titles='Event-Related Potential', show=False, fig=fig)
        else:
            plt.text(0.5, 0.5, "No ERP data available for visualization",
                    ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "eeg_erp", output_format)
    
    def _visualize_fmri(self, data_path: str, vis_type: str, config: Dict, 
                       output_format: str) -> str:
        """
        Generate visualization for fMRI data.
        
        This is a placeholder implementation. In a real system, you would use
        neuroimaging libraries like nilearn for proper visualization.
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        # Placeholder for fMRI visualization
        plt.text(0.5, 0.5, f"fMRI {vis_type} visualization (placeholder)",
                ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, f"fmri_{vis_type}", output_format)
    
    def _visualize_calcium(self, data_path: str, vis_type: str, config: Dict, 
                          output_format: str) -> str:
        """
        Generate visualization for calcium imaging data.
        
        This is a placeholder implementation.
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        # Placeholder for calcium imaging visualization
        plt.text(0.5, 0.5, f"Calcium imaging {vis_type} visualization (placeholder)",
                ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, f"calcium_{vis_type}", output_format)
    
    def _visualize_connectivity(self, data_path: str, vis_type: str, config: Dict, 
                               output_format: str) -> str:
        """
        Generate visualization for connectivity data.
        
        Args:
            data_path: Path to connectivity data
            vis_type: Type of visualization
            config: Visualization configuration
            output_format: Output format
            
        Returns:
            Path to generated visualization or base64 string
        """
        # Load data
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format for connectivity: {data_path}")
        
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        if vis_type == "matrix":
            # Connectivity matrix visualization
            if "features" in data and "connectivity" in data["features"]:
                conn_data = data["features"]["connectivity"]
                matrix = np.array(conn_data["connectivity_matrix"])
                channels = conn_data["channel_names"]
                
                # Plot connectivity matrix
                plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
                sns.heatmap(
                    matrix,
                    xticklabels=channels,
                    yticklabels=channels,
                    cmap=self.config["styling"].get("color_palette", "viridis"),
                    annot=False
                )
                plt.title(f"Neural Connectivity ({conn_data.get('measure', 'correlation')})")
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, "No connectivity data found",
                        ha='center', va='center', fontsize=12)
        elif vis_type == "graph":
            # Graph visualization placeholder
            plt.text(0.5, 0.5, "Graph visualization requires networkx (placeholder)",
                    ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, f"Unsupported connectivity visualization: {vis_type}",
                    ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, f"connectivity_{vis_type}", output_format)
    
    def _visualize_feature_importance(self, results: Dict, output_format: str, 
                                     parameters: Dict = None) -> str:
        """
        Generate feature importance visualization.
        
        This is a placeholder implementation.
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        # Placeholder for feature importance visualization
        plt.text(0.5, 0.5, "Feature importance visualization (placeholder)",
                ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "feature_importance", output_format)
    
    def _visualize_confusion_matrix(self, results: Dict, output_format: str, 
                                   parameters: Dict = None) -> str:
        """
        Generate confusion matrix visualization.
        
        This is a placeholder implementation.
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        # Placeholder for confusion matrix visualization
        plt.text(0.5, 0.5, "Confusion matrix visualization (placeholder)",
                ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "confusion_matrix", output_format)
    
    def _visualize_performance_metrics(self, results: Dict, output_format: str, 
                                      parameters: Dict = None) -> str:
        """
        Generate performance metrics visualization.
        
        Args:
            results: Model training or evaluation results
            output_format: Output format
            parameters: Optional visualization parameters
            
        Returns:
            Path to generated visualization or base64 string
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        # Check if evaluation metrics exist
        if "evaluation" in results:
            metrics = results["evaluation"]
            
            # Create bar plot of metrics
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
            plt.bar(metric_names, metric_values, color=sns.color_palette("viridis", len(metric_names)))
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)  # Baseline for accuracy metrics
            plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)  # Perfect score
            plt.ylim(0, 1.1)
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Model Performance Metrics')
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "No performance metrics found in results",
                    ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "performance_metrics", output_format)
    
    def _visualize_training_history(self, results: Dict, output_format: str, 
                                   parameters: Dict = None) -> str:
        """
        Generate training history visualization.
        
        Args:
            results: Model training results
            output_format: Output format
            parameters: Optional visualization parameters
            
        Returns:
            Path to generated visualization or base64 string
        """
        fig = plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
        
        # Check if training history exists
        if "training" in results and "history" in results["training"]:
            history = results["training"]["history"]
            
            # Plot training and validation loss
            plt.figure(figsize=tuple(self.config["styling"]["figure_size"]))
            plt.subplot(1, 2, 1)
            plt.plot(history["loss"], label="Training Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            
            # Plot training and validation accuracy, if available
            if "accuracy" in history and "val_accuracy" in history:
                plt.subplot(1, 2, 2)
                plt.plot(history["accuracy"], label="Training Accuracy")
                plt.plot(history["val_accuracy"], label="Validation Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("Training and Validation Accuracy")
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "No training history found in results",
                    ha='center', va='center', fontsize=12)
        
        return self._save_or_return_figure(fig, "training_history", output_format)
    
    def _save_or_return_figure(self, fig: plt.Figure, name: str, output_format: str) -> str:
        """
        Save figure to file or return as base64 string.
        
        Args:
            fig: Matplotlib figure
            name: Base name for saved file
            output_format: Output format ('png', 'svg', 'html', 'base64')
            
        Returns:
            Path to saved file or base64 string
        """
        if output_format == "base64":
            # Return as base64 string
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.config["styling"]["dpi"])
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str
        else:
            # Save to file
            output_dir = self.config["storage"]["visualizations"]
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = int(time.time())
            filename = os.path.join(output_dir, f"{name}_{timestamp}.{output_format}")
            
            # Save figure
            fig.savefig(filename, dpi=self.config["styling"]["dpi"], bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Saved visualization to {filename}")
            return filename


# Example usage
if __name__ == "__main__":
    import time
    import sys
    
    visualizer = NeuralDataVisualizer()
    
    if len(sys.argv) > 3:
        data_path = sys.argv[1]
        modality = sys.argv[2]
        vis_type = sys.argv[3]
        
        print(f"Generating {vis_type} visualization for {modality} data from {data_path}")
        result = visualizer.visualize(data_path, modality, vis_type)
        print(f"Generated visualization: {result}")
    else:
        print("Usage: python neural_visualizer.py <data_path> <modality> <vis_type>")
        print("Example: python neural_visualizer.py data/eeg/subject001.edf eeg time_series")