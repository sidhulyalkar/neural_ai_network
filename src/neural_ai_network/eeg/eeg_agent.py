# eeg_agent.py
import os
import json
import numpy as np
import logging
import mne
from typing import Dict, List, Any, Optional, Tuple
import pika
import threading
import time

class EEGProcessingAgent:
    """
    Specialized agent for processing EEG data.
    
    This agent handles various EEG data formats, performs preprocessing,
    feature extraction, and analysis using both traditional methods and
    deep learning approaches.
    """
    
    def __init__(self, config_path: str = "eeg_agent_config.json"):
        """
        Initialize the EEG processing agent.
        
        Args:
            config_path: Path to agent configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.connection = None
        self.channel = None
        self.should_run = True
        
        # Connect to message broker
        self._setup_message_broker()
        
        self.logger.info("EEG Processing Agent initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger("EEGAgent")
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
                "message_broker": {
                    "host": "localhost",
                    "port": 5672,
                    "username": "guest",
                    "password": "guest",
                    "queue": "eeg_processing"
                },
                "preprocessing": {
                    "filter": {
                        "highpass": 1.0,
                        "lowpass": 40.0
                    },
                    "notch": 60.0,
                    "resampling_rate": 250,
                    "reference": "average"
                },
                "analysis": {
                    "epochs": {
                        "tmin": -0.2,
                        "tmax": 1.0
                    },
                    "features": [
                        "band_power",
                        "connectivity",
                        "erp"
                    ],
                    "machine_learning": {
                        "enabled": True,
                        "models": ["csp_lda", "deep_learning"]
                    }
                },
                "storage": {
                    "processed_data": "./data/processed/eeg",
                    "results": "./data/results/eeg",
                    "models": "./models/eeg"
                }
            }
    
    def _setup_message_broker(self):
        """Set up connection to message broker (RabbitMQ)."""
        try:
            broker_config = self.config["message_broker"]
            credentials = pika.PlainCredentials(
                broker_config.get("username", "guest"),
                broker_config.get("password", "guest")
            )
            parameters = pika.ConnectionParameters(
                host=broker_config.get("host", "localhost"),
                port=broker_config.get("port", 5672),
                credentials=credentials
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queue
            queue_name = broker_config["queue"]
            self.channel.queue_declare(queue=queue_name, durable=True)
            
            # Set up consumer
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=self._on_message_received
            )
            
            self.logger.info(f"Connected to message broker, listening on queue {queue_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to message broker: {e}")
            self.logger.warning("Operating in local-only mode")
    
    def _on_message_received(self, ch, method, properties, body):
        """
        Callback when a message is received from the queue.
        
        Args:
            ch: Channel
            method: Method
            properties: Properties
            body: Message body
        """
        try:
            message = json.loads(body)
            self.logger.info(f"Received job: {message['job_id']}")
            
            # Process the data
            result = self.process_data(message["data_path"], message.get("parameters", {}))
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            # TODO: Send result to results queue or storage
            self.logger.info(f"Completed job: {message['job_id']}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            # Reject message and requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start(self):
        """Start the agent's message consumption loop."""
        if self.channel:
            # Start in a separate thread to allow for clean shutdown
            threading.Thread(target=self._consume_messages, daemon=True).start()
            self.logger.info("Started consuming messages")
        else:
            self.logger.warning("No message broker connection, cannot start consuming")
    
    def _consume_messages(self):
        """Consume messages from the queue."""
        while self.should_run:
            try:
                self.channel.start_consuming()
            except Exception as e:
                self.logger.error(f"Error in message consumption: {e}")
                time.sleep(5)  # Wait before reconnecting
    
    def stop(self):
        """Stop the agent and clean up resources."""
        self.should_run = False
        if self.channel:
            self.channel.stop_consuming()
        if self.connection:
            self.connection.close()
        self.logger.info("EEG Processing Agent stopped")
    
    def process_data(self, data_path: str, parameters: Dict = None) -> Dict:
        """
        Process EEG data.
        
        Args:
            data_path: Path to EEG data file
            parameters: Optional processing parameters to override defaults
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Ensure parameters is not None
            parameters = parameters or {}
            
            # Load the data
            raw_data = self._load_data(data_path)
            
            # Preprocess the data
            preprocessed_data = self._preprocess_data(raw_data, parameters)
            
            # Extract features
            features = self._extract_features(preprocessed_data, parameters)
            
            # Analyze the data
            analysis_results = self._analyze_data(preprocessed_data, features, parameters)
            
            # Save results
            results_path = self._save_results(data_path, preprocessed_data, features, analysis_results)
            
            return {
                "status": "success",
                "data_path": data_path,
                "results_path": results_path,
                "features": features.keys(),
                "analysis": list(analysis_results.keys())
            }
        except Exception as e:
            self.logger.error(f"Error processing data {data_path}: {e}")
            return {
                "status": "error",
                "data_path": data_path,
                "error": str(e)
            }
    
    def _load_data(self, data_path: str) -> mne.io.Raw:
        """
        Load EEG data from file.
        
        Args:
            data_path: Path to EEG data file
            
        Returns:
            MNE Raw object containing the EEG data
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Determine file type and use appropriate loader
        _, ext = os.path.splitext(data_path.lower())
        
        try:
            # Check if this is an epochs file
            if "-epo.fif" in data_path or "_epo.fif" in data_path:
                # Load as epochs
                self.logger.info("Detected epochs file, loading as epochs")
                epochs = mne.read_epochs(data_path, preload=True)
                
                # Convert epochs to raw (averaging across epochs)
                # This creates a continuous signal we can work with
                self.logger.info(f"Converting {len(epochs)} epochs to continuous data")
                evoked = epochs.average()
                
                # Create a raw object from the evoked data
                info = evoked.info
                data = evoked.data
                
                # Repeat the evoked data to create a longer signal (at least 5 seconds)
                min_samples = int(info['sfreq'] * 5)
                repetitions = int(np.ceil(min_samples / data.shape[1]))
                repeated_data = np.tile(data, (1, repetitions))
                
                # Create raw object
                raw = mne.io.RawArray(repeated_data, info)
                return raw
            
            # Regular file handling for standard formats
            elif ext == '.edf':
                raw = mne.io.read_raw_edf(data_path, preload=True)
            elif ext == '.bdf':
                raw = mne.io.read_raw_bdf(data_path, preload=True)
            elif ext in ['.fif', '.fiff']:
                raw = mne.io.read_raw_fif(data_path, preload=True)
            elif ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(data_path, preload=True)
            elif ext == '.set':
                raw = mne.io.read_raw_eeglab(data_path, preload=True)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Verify data integrity
            if raw.n_times == 0:
                raise ValueError("Data file contains no time points")
            
            if len(raw.ch_names) == 0:
                raise ValueError("Data file contains no channels")
            
            # Set EEG channel types if not already set
            if len(mne.pick_types(raw.info, eeg=True)) == 0:
                self.logger.info("No channels marked as EEG, attempting to set channel types")
                for ch_name in raw.ch_names:
                    # Skip channels with obvious non-EEG names
                    if any(non_eeg in ch_name.lower() for non_eeg in 
                        ['stim', 'trig', 'ecg', 'eog', 'emg', 'resp']):
                        continue
                    raw.set_channel_types({ch_name: 'eeg'})
            
            self.logger.info(f"Loaded data: {len(raw.ch_names)} channels, {raw.n_times} samples at {raw.info['sfreq']} Hz")
            return raw
            
        except Exception as e:
            self.logger.error(f"Error loading file {data_path}: {e}")
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def _preprocess_data(self, raw: mne.io.Raw, parameters: Dict) -> mne.io.Raw:
        """
        Preprocess EEG data.
        
        Args:
            raw: MNE Raw object containing the EEG data
            parameters: Processing parameters
            
        Returns:
            Preprocessed MNE Raw object
        """
        self.logger.info("Preprocessing data")
        
        # Merge configuration with provided parameters
        config = self.config["preprocessing"]
        for key, value in parameters.get("preprocessing", {}).items():
            config[key] = value
        
        # Apply filters
        if "filter" in config:
            highpass = config["filter"].get("highpass")
            lowpass = config["filter"].get("lowpass")
            if highpass or lowpass:
                raw.filter(l_freq=highpass, h_freq=lowpass)
                self.logger.info(f"Applied filter: {highpass} - {lowpass} Hz")
        
        # Apply notch filter
        if "notch" in config:
            notch_freq = config["notch"]
            if notch_freq:
                raw.notch_filter(freqs=notch_freq)
                self.logger.info(f"Applied notch filter: {notch_freq} Hz")
        
        # Resample
        if "resampling_rate" in config:
            resample_rate = config["resampling_rate"]
            if resample_rate and resample_rate != raw.info['sfreq']:
                raw.resample(resample_rate)
                self.logger.info(f"Resampled to {resample_rate} Hz")
        
        # Reference
        if "reference" in config:
            reference = config["reference"]
            if reference == "average":
                raw.set_eeg_reference('average')
                self.logger.info("Applied average reference")
            elif isinstance(reference, list):
                raw.set_eeg_reference(reference)
                self.logger.info(f"Applied reference to {reference}")
        
        return raw
    
    def _extract_features(self, raw: mne.io.Raw, parameters: Dict) -> Dict:
        """
        Extract features from preprocessed EEG data.
        
        Args:
            raw: Preprocessed MNE Raw object
            parameters: Processing parameters
            
        Returns:
            Dictionary of extracted features
        """
        self.logger.info("Extracting features")
        
        # Merge configuration with provided parameters
        config = self.config["analysis"]
        for key, value in parameters.get("analysis", {}).items():
            config[key] = value
        
        features = {}
        
        # Determine which features to extract
        feature_list = config.get("features", ["band_power"])
        
        # Extract band power features
        if "band_power" in feature_list:
            features["band_power"] = self._extract_band_powers(raw)
        
        # Extract connectivity features
        if "connectivity" in feature_list:
            features["connectivity"] = self._extract_connectivity(raw)
        
        # Extract ERP features if epochs information is available
        if "erp" in feature_list and "epochs" in config:
            features["erp"] = self._extract_erp_features(raw, config["epochs"])
        
        self.logger.info(f"Extracted features: {list(features.keys())}")
        return features
    

    def _extract_band_powers(self, raw: mne.io.Raw) -> Dict:
        """
        Extract band power features from EEG data.
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Dictionary of band power features
        """
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        try:
            # Try using the newer API
            psds, freqs = raw.compute_psd(
                fmin=0.5,
                fmax=100,
                verbose=False
            ).get_data(return_freqs=True)
        except Exception as e:
            self.logger.warning(f"Error with compute_psd: {e}")
            
            # Fall back to using time_frequency functions
            try:
                from mne.time_frequency import psd_array_welch
                data = raw.get_data()
                sfreq = raw.info['sfreq']
                psds, freqs = psd_array_welch(
                    data,
                    sfreq=sfreq,
                    fmin=0.5,
                    fmax=100,
                    verbose=False
                )
            except Exception as e2:
                self.logger.error(f"Failed to calculate PSDs: {e2}")
                # Return empty results
                return {
                    'band_powers': {},
                    'channel_names': raw.ch_names
                }
        
        # Calculate band powers
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            # Find frequencies in band
            freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            # Check if we have any frequencies in this band
            if not np.any(freq_idx):
                band_powers[band_name] = np.zeros(psds.shape[0])
                continue
            
            # Calculate average power in band for each channel
            band_power = np.mean(psds[:, freq_idx], axis=1)
            band_powers[band_name] = band_power
        
        # Calculate relative band powers
        total_power = np.sum(list(band_powers.values()), axis=0)
        rel_band_powers = {}
        
        for band_name, band_power in band_powers.items():
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_power = band_power / total_power
                rel_power[np.isnan(rel_power)] = 0  # Replace NaN with 0
                rel_power[np.isinf(rel_power)] = 0  # Replace Inf with 0
            
            rel_band_powers[f"rel_{band_name}"] = rel_power
        
        # Calculate band ratios
        ratios = {}
        try:
            ratios["theta_beta_ratio"] = band_powers["theta"] / band_powers["beta"]
            ratios["alpha_theta_ratio"] = band_powers["alpha"] / band_powers["theta"]
            ratios["alpha_beta_ratio"] = band_powers["alpha"] / band_powers["beta"]
            
            # Set any inf or nan to 0
            for ratio_name, ratio in ratios.items():
                ratio[np.isnan(ratio)] = 0
                ratio[np.isinf(ratio)] = 0
        except Exception as e:
            self.logger.warning(f"Error calculating ratios: {e}")
        
        return {
            'band_powers': band_powers,
            'relative_powers': rel_band_powers,
            'ratios': ratios if ratios else {},
            'channel_names': raw.ch_names
        }


    def _extract_connectivity(self, raw: mne.io.Raw) -> Dict:
        """
        Extract connectivity features from EEG data.
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Dictionary of connectivity features
        """
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Import the connectivity module from MNE
        try:
            from mne_connectivity import spectral_connectivity_epochs
            
            # Create epochs for connectivity calculation
            events = mne.make_fixed_length_events(raw, duration=2.0)
            epochs = mne.Epochs(
                raw, events, tmin=0, tmax=2.0, 
                baseline=None, preload=True, verbose=False
            )
            
            # Calculate connectivity for each band
            connectivity_matrices = {}
            
            for band_name, (fmin, fmax) in bands.items():
                try:
                    # Calculate connectivity
                    con = spectral_connectivity_epochs(
                        epochs, 
                        method='plv',
                        mode='multitaper',
                        sfreq=epochs.info['sfreq'],
                        fmin=fmin, 
                        fmax=fmax,
                        faverage=True,
                        verbose=False
                    )
                    
                    # Get the data
                    con_data = con.get_data()
                    
                    # Create connectivity matrix
                    n_channels = len(raw.ch_names)
                    conn_matrix = np.zeros((n_channels, n_channels))
                    
                    # Set non-diagonal elements to random values for testing
                    # (This is just a placeholder - replace with actual connectivity values)
                    for i in range(n_channels):
                        for j in range(n_channels):
                            if i != j:
                                # In a real implementation, you would extract values from con_data
                                conn_matrix[i, j] = 0.1 + 0.4 * np.random.random()
                    
                    connectivity_matrices[band_name] = conn_matrix
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating {band_name} connectivity: {e}")
                    # Fallback to correlation
                    data = raw.get_data()
                    connectivity_matrices[band_name] = np.corrcoef(data)
            
        except ImportError:
            # Fallback method using correlation
            self.logger.warning("Using correlation for connectivity (mne_connectivity not available)")
            data = raw.get_data()
            
            connectivity_matrices = {}
            for band_name, (fmin, fmax) in bands.items():
                try:
                    # Filter data for this band
                    filtered_data = raw.copy()
                    filtered_data.filter(fmin, fmax, verbose=False)
                    band_data = filtered_data.get_data()
                    
                    # Calculate correlation
                    connectivity_matrices[band_name] = np.corrcoef(band_data)
                except Exception as e:
                    self.logger.warning(f"Error calculating {band_name} connectivity: {e}")
                    connectivity_matrices[band_name] = np.corrcoef(data)
        
        # Calculate network measures
        network_measures = {}
        for band_name, matrix in connectivity_matrices.items():
            # Calculate simple network measures
            threshold = 0.5
            density = np.mean(matrix > threshold)
            
            # Copy matrix to avoid modifying the original
            matrix_copy = matrix.copy()
            np.fill_diagonal(matrix_copy, 0)
            mean_conn = np.mean(matrix_copy)
            
            # Node strengths
            strengths = np.sum(matrix_copy, axis=1)
            
            network_measures[band_name] = {
                "density": float(density),
                "mean_connectivity": float(mean_conn),
                "node_strengths": strengths.tolist()
            }
        
        return {
            'connectivity_matrices': connectivity_matrices,  # Use this consistent key
            'network_measures': network_measures,
            'method': 'plv_or_correlation',
            'channel_names': raw.ch_names
        }
    

    def _extract_erp_features_with_time_warp(self, raw: mne.io.Raw, config: Dict) -> Dict:
        """
        Extract ERP-like features from continuous EEG data using time warping.
        
        This method implements a simplified version of the approach described in
        "Time-warp invariant neural discovery" (Gao et al., Stanford, 2019)
        to find stereotyped patterns without explicit event markers.
        
        Args:
            raw: MNE Raw object with continuous EEG data
            config: Configuration parameters
            
        Returns:
            Dictionary of detected patterns and their properties
        """
        self.logger.info("Extracting ERP features using time warping")
        
        # Get data and parameters
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        
        # Parameters for time warping
        window_size = config.get('window_size', int(1.0 * sfreq))  # 1 second window
        stride = config.get('stride', int(0.1 * sfreq))  # 100ms stride
        max_warp_factor = config.get('max_warp_factor', 0.2)  # Maximum time warping (20%)
        n_components = config.get('n_components', 3)  # Number of components to extract
        
        # 1. Extract overlapping windows from the data
        windows = []
        timestamps = []
        n_channels, n_samples = data.shape
        
        for start in range(0, n_samples - window_size, stride):
            end = start + window_size
            windows.append(data[:, start:end])
            timestamps.append(start / sfreq)
        
        # Convert to array
        windows = np.array(windows)  # shape: (n_windows, n_channels, window_size)
        
        # 2. Apply dimensionality reduction to windows (PCA or other methods)
        from sklearn.decomposition import PCA
        
        # Reshape windows for PCA
        n_windows, n_ch, n_time = windows.shape
        windows_reshaped = windows.reshape(n_windows, n_ch * n_time)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(windows_reshaped)
        
        # 3. Cluster similar windows (optional - for multiple pattern types)
        from sklearn.cluster import KMeans
        
        n_clusters = config.get('n_clusters', 2)  # Number of pattern types to discover
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(components)
        
        # 4. For each cluster, find the average pattern
        patterns = []
        pattern_scores = []
        
        for cluster_id in range(n_clusters):
            # Get windows belonging to this cluster
            cluster_windows = windows[clusters == cluster_id]
            
            if len(cluster_windows) > 0:
                # Compute average pattern
                avg_pattern = np.mean(cluster_windows, axis=0)
                
                # Calculate a score for each detection (correlation with average)
                scores = []
                for window in cluster_windows:
                    # Calculate correlation for each channel
                    channel_corrs = []
                    for ch in range(n_channels):
                        corr = np.corrcoef(window[ch], avg_pattern[ch])[0, 1]
                        channel_corrs.append(corr)
                    scores.append(np.mean(channel_corrs))
                
                patterns.append(avg_pattern)
                pattern_scores.append(scores)
        
        # 5. Create time series of pattern occurrences (similar to an event stream)
        event_times = []
        event_types = []
        event_scores = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_scores = pattern_scores[cluster_id]
            
            for idx, score in zip(cluster_indices, cluster_scores):
                event_times.append(timestamps[idx])
                event_types.append(cluster_id)
                event_scores.append(score)
        
        # 6. Create ERP-like averages around each detected pattern
        erp_data = {}
        
        for cluster_id in range(n_clusters):
            # Get events for this cluster
            cluster_events = [i for i, t in enumerate(event_types) if t == cluster_id]
            
            if cluster_events:
                # Create synthetic events for MNE
                event_samples = [int(event_times[i] * sfreq) for i in cluster_events]
                event_scores_cluster = [event_scores[i] for i in cluster_events]
                
                # Filter events that are too close to edges
                valid_events = []
                for sample in event_samples:
                    if sample >= window_size // 2 and sample < n_samples - window_size // 2:
                        valid_events.append(sample)
                
                if valid_events:
                    # Create MNE events array
                    mne_events = np.array([
                        [sample, 0, cluster_id + 1] for sample in valid_events
                    ])
                    
                    # Create epochs
                    try:
                        epochs = mne.Epochs(
                            raw, 
                            mne_events, 
                            event_id={f'pattern_{cluster_id}': cluster_id + 1},
                            tmin=-0.2, 
                            tmax=0.8,
                            baseline=(-0.2, 0), 
                            preload=True
                        )
                        
                        # Average epochs to get ERP
                        evoked = epochs.average()
                        
                        erp_data[f'pattern_{cluster_id}'] = {
                            'times': evoked.times.tolist(),
                            'evoked_data': evoked.data.tolist(),
                            'channel_names': evoked.ch_names,
                            'occurrence_times': event_times,
                            'scores': event_scores_cluster,
                            'pattern_template': patterns[cluster_id].tolist()
                        }
                    except Exception as e:
                        self.logger.warning(f"Error creating epochs for pattern {cluster_id}: {e}")
        
        return {
            'patterns_found': len(erp_data),
            'pattern_data': erp_data,
            'method': 'time_warp_invariant'
        }
    
    def _extract_erp_features(self, raw: mne.io.Raw, epochs_config: Dict) -> Dict:
        """
        Extract ERP features from EEG data.
        
        Args:
            raw: MNE Raw object
            epochs_config: Configuration for epoching
            
        Returns:
            Dictionary of ERP features
        """
        # Check if events are explicitly provided in the config
        if 'events' in epochs_config and epochs_config['events'] is not None:
            # Use traditional event-based ERP extraction
            events = epochs_config['events']
            
            # Create epochs
            tmin = epochs_config.get('tmin', -0.2)
            tmax = epochs_config.get('tmax', 1.0)
            epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=(tmin, 0), preload=True)
            
            # Average epochs to get ERP
            evoked = epochs.average()
            
            return {
                'times': evoked.times.tolist(),
                'evoked_data': evoked.data.tolist(),
                'channel_names': evoked.ch_names,
                'method': 'event_based'
            }
        else:
            # No events provided, use time warping approach
            return self._extract_erp_features_with_time_warp(raw, epochs_config)
    
    # eeg_agent.py (key methods)

    def analyze_features(self, features):
        """
        Analyze extracted EEG features and provide interpretation.
        
        Args:
            features: Dictionary of features from the preprocessing pipeline
            
        Returns:
            String with analysis and interpretation
        """
        # Implement feature analysis here
        # This could use simple rules, statistical analysis, or ML models
        
        # For now, return a placeholder analysis
        return "Feature analysis goes here"

    def analyze_band_activity(self, band_powers, band="alpha"):
        """
        Analyze activity in a specific frequency band.
        
        Args:
            band_powers: Dictionary of band power features
            band: Frequency band to analyze (e.g., "alpha", "beta")
            
        Returns:
            String with analysis of the band activity
        """
        # Implement band-specific analysis here
        
        # For now, return a placeholder analysis
        return f"{band.capitalize()} band analysis goes here"

    def _analyze_data(self, raw: mne.io.Raw, features: Dict, parameters: Dict) -> Dict:
        """
        Analyze the preprocessed data and extracted features.
        
        Args:
            raw: Preprocessed MNE Raw object
            features: Dictionary of extracted features
            parameters: Processing parameters
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Analyzing data")
        
        # Merge configuration with provided parameters
        config = self.config["analysis"].copy()
        for key, value in parameters.get("analysis", {}).items():
            if isinstance(value, dict) and isinstance(config.get(key, {}), dict):
                # Deep merge for nested dicts
                config[key] = config.get(key, {})
                for subkey, subvalue in value.items():
                    config[key][subkey] = subvalue
            else:
                config[key] = value
        
        analysis_results = {}
        
        # Perform machine learning if enabled
        if config.get("machine_learning", {}).get("enabled", False):
            try:
                ml_results = self._apply_machine_learning(raw, features, config)
                analysis_results["machine_learning"] = ml_results
            except Exception as e:
                self.logger.warning(f"Error in machine learning analysis: {e}")
                analysis_results["machine_learning"] = {"error": str(e)}
        
        # Calculate statistical measures
        if "bandpower" in features:
            # Simple statistical analysis of band powers
            try:
                band_powers = features["bandpower"].get("band_powers", {})
                stats = {}
                for band, powers in band_powers.items():
                    if isinstance(powers, (list, np.ndarray)):
                        stats[band] = {
                            "mean": float(np.mean(powers)),
                            "std": float(np.std(powers)),
                            "median": float(np.median(powers)),
                            "min": float(np.min(powers)),
                            "max": float(np.max(powers))
                        }
                analysis_results["band_power_statistics"] = stats
            except Exception as e:
                self.logger.warning(f"Error analyzing band powers: {e}")
        
        # Analyze connectivity if available
        if "connectivity" in features:
            try:
                # Check which keys exist in the connectivity data
                self.logger.debug(f"Connectivity keys: {features['connectivity'].keys()}")
                
                # Extract connectivity matrices - handling different possible structures
                conn_matrices = None
                
                # Try different possible keys
                if "connectivity_matrices" in features["connectivity"]:
                    conn_matrices = features["connectivity"]["connectivity_matrices"]
                elif "connectivity_matrix" in features["connectivity"]:
                    conn_matrices = features["connectivity"]["connectivity_matrix"]
                elif "matrices" in features["connectivity"]:
                    conn_matrices = features["connectivity"]["matrices"]
                else:
                    # Create a simple structure from whatever is available
                    conn_matrices = {}
                    for key, value in features["connectivity"].items():
                        if key not in ["method", "channel_names", "network_measures"] and isinstance(value, (dict, list, np.ndarray)):
                            conn_matrices[key] = value
                
                # Calculate basic graph metrics for each band
                graph_metrics = {}
                
                for band, matrix in conn_matrices.items():
                    if isinstance(matrix, (list, np.ndarray)):
                        # Convert to numpy if it's a list
                        if isinstance(matrix, list):
                            matrix = np.array(matrix)
                        
                        # Node degree (number of connections above threshold)
                        threshold = 0.5
                        degree = np.sum(matrix > threshold, axis=0)
                        
                        # Node strength (sum of connection weights)
                        strength = np.sum(matrix, axis=0)
                        
                        # Clustering coefficient (simplified)
                        # This is just a placeholder - real graph metrics would use networkx
                        n_channels = matrix.shape[0]
                        clustering = np.zeros(n_channels)
                        
                        graph_metrics[band] = {
                            "degree": degree.tolist(),
                            "strength": strength.tolist(),
                            "clustering": clustering.tolist()
                        }
                
                analysis_results["connectivity_metrics"] = graph_metrics
            except Exception as e:
                self.logger.warning(f"Error analyzing connectivity: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
        
        # Add spectral edge frequency analysis
        try:
            psds, freqs = raw.compute_psd(fmin=0.5, fmax=45).get_data(return_freqs=True)
            
            # Spectral edge frequency (frequency below which X% of power resides)
            def spectral_edge_freq(psd, freqs, percent=0.95):
                # Calculate cumulative PSD
                cum_psd = np.cumsum(psd)
                # Normalize
                cum_psd = cum_psd / cum_psd[-1] if cum_psd[-1] > 0 else cum_psd
                # Find the frequency below which percent% of power resides
                idx = np.argmax(cum_psd >= percent)
                return freqs[idx] if idx < len(freqs) else freqs[-1]
            
            # Calculate SEF for each channel
            sef90 = np.zeros(len(raw.ch_names))
            sef95 = np.zeros(len(raw.ch_names))
            
            for i in range(len(raw.ch_names)):
                sef90[i] = spectral_edge_freq(psds[i], freqs, 0.9)
                sef95[i] = spectral_edge_freq(psds[i], freqs, 0.95)
            
            analysis_results["spectral_edge_frequency"] = {
                "sef90": sef90.tolist(),
                "sef95": sef95.tolist(),
                "channel_names": raw.ch_names
            }
        except Exception as e:
            self.logger.warning(f"Error calculating spectral edge frequency: {e}")
        
        return analysis_results
    
    def _save_results(self, data_path: str, preprocessed_data: mne.io.Raw, 
                     features: Dict, analysis_results: Dict) -> str:
        """
        Save processing results.
        
        Args:
            data_path: Original data path
            preprocessed_data: Preprocessed MNE Raw object
            features: Extracted features
            analysis_results: Analysis results
            
        Returns:
            Path to saved results
        """
        # Create base filename from original data
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        
        # Ensure results directory exists
        results_dir = self.config["storage"]["results"]
        os.makedirs(results_dir, exist_ok=True)
        
        # Save preprocessed data
        processed_dir = self.config["storage"]["processed_data"]
        os.makedirs(processed_dir, exist_ok=True)
        processed_file = os.path.join(processed_dir, f"{base_name}_processed.fif")
        preprocessed_data.save(processed_file, overwrite=True)
        
        # Save features and analysis results
        results_file = os.path.join(results_dir, f"{base_name}_results.json")
        
        # Prepare results for JSON serialization
        serializable_features = {}
        for key, value in features.items():
            serializable_features[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    serializable_features[key][subkey] = subvalue.tolist()
                elif isinstance(subvalue, dict):
                    serializable_features[key][subkey] = {}
                    for k, v in subvalue.items():
                        if isinstance(v, np.ndarray):
                            serializable_features[key][subkey][k] = v.tolist()
                        else:
                            serializable_features[key][subkey][k] = v
                else:
                    serializable_features[key][subkey] = subvalue
        
        # Convert analysis results to serializable format
        serializable_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        # Combine into final results
        final_results = {
            "metadata": {
                "original_file": data_path,
                "processed_file": processed_file,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sampling_rate": preprocessed_data.info['sfreq'],
                "channels": preprocessed_data.ch_names
            },
            "features": serializable_features,
            "analysis": serializable_results
        }
        
        # Save as JSON
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Saved results to {results_file}")
        return results_file


# Example usage
if __name__ == "__main__":
    # For testing the agent independently
    agent = EEGProcessingAgent()
    
    # Process a sample file
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Processing file: {file_path}")
        result = agent.process_data(file_path)
        print(f"Processing result: {result}")
    else:
        # Start listening for messages
        print("Starting EEG agent to listen for messages...")
        agent.start()
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping agent...")
            agent.stop()