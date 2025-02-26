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
        
        if ext == '.edf':
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
        
        self.logger.info(f"Loaded data: {len(raw.ch_names)} channels, {raw.n_times} samples at {raw.info['sfreq']} Hz")
        return raw
    
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
        
        # Calculate power spectral density
        psds, freqs = mne.time_frequency.psd_welch(
            raw, 
            fmin=0.5, 
            fmax=100, 
            n_fft=int(raw.info['sfreq'] * 2),
            n_overlap=int(raw.info['sfreq']),
            n_per_seg=int(raw.info['sfreq'] * 4)
        )
        
        # Calculate band powers
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            # Find frequencies in band
            freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            # Calculate average power in band for each channel
            band_power = np.mean(psds[:, freq_idx], axis=1)
            band_powers[band_name] = band_power
        
        return {
            'band_powers': band_powers,
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
        # This is a simplified placeholder. In a real implementation,
        # you would use proper connectivity measures like coherence,
        # PLV, or more advanced metrics.
        
        # For demonstration, we'll just compute a simple correlation matrix
        data = raw.get_data()
        conn_matrix = np.corrcoef(data)
        
        return {
            'connectivity_matrix': conn_matrix,
            'measure': 'correlation',
            'channel_names': raw.ch_names
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
        # This is a placeholder. In a real implementation, you would
        # need event information to create meaningful epochs.
        
        # For demonstration, we'll create mock events at regular intervals
        sfreq = raw.info['sfreq']
        events = mne.make_fixed_length_events(raw, id=1, duration=1.0)
        
        # Create epochs
        tmin = epochs_config.get('tmin', -0.2)
        tmax = epochs_config.get('tmax', 1.0)
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=(tmin, 0), preload=True)
        
        # Average epochs to get ERP
        evoked = epochs.average()
        
        return {
            'times': evoked.times,
            'evoked_data': evoked.data,
            'channel_names': evoked.ch_names
        }
    
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
        config = self.config["analysis"]
        for key, value in parameters.get("analysis", {}).items():
            config[key] = value
        
        analysis_results = {}
        
        # Perform machine learning if enabled
        if config.get("machine_learning", {}).get("enabled", False):
            # This is a placeholder for ML analysis
            analysis_results["machine_learning"] = {
                "model": "placeholder",
                "performance": {
                    "accuracy": 0.85,
                    "f1_score": 0.84
                }
            }
        
        # Calculate statistical measures
        if "band_power" in features:
            # Simple statistical analysis of band powers
            band_powers = features["band_power"]["band_powers"]
            stats = {}
            for band, powers in band_powers.items():
                stats[band] = {
                    "mean": float(np.mean(powers)),
                    "std": float(np.std(powers)),
                    "median": float(np.median(powers))
                }
            analysis_results["band_power_statistics"] = stats
        
        # Analyze connectivity if available
        if "connectivity" in features:
            # For demonstration, we'll just compute some basic graph metrics
            conn_matrix = features["connectivity"]["connectivity_matrix"]
            
            # Calculate node degree (simplified)
            degree = np.sum(conn_matrix > 0.5, axis=0)
            
            analysis_results["connectivity_metrics"] = {
                "degree": degree.tolist(),
                "channel_names": features["connectivity"]["channel_names"]
            }
        
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