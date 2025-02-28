# eeg_preprocessing.py
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
from pathlib import Path
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler

@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing pipeline."""
    # Filtering
    lowpass_freq: float = 50.0
    highpass_freq: float = 1.0
    notch_freq: float = 60.0
    apply_notch: bool = True
    
    # Resampling
    resample: bool = True
    resample_freq: float = 250.0
    
    # Referencing
    reference: str = "average"  # "average", "mastoids", or List of channel names
    
    # Artifact detection
    reject_artifacts: bool = True
    amplitude_threshold_uv: float = 200.0
    flatline_duration_s: float = 0.5
    
    # Bad channel detection
    detect_bad_channels: bool = True
    bad_channel_criteria: str = "correlation"  # "correlation", "amplitude", "spectrum"
    bad_channel_threshold: float = 0.7
    
    # ICA
    apply_ica: bool = False
    n_ica_components: Optional[int] = None
    
    # Epoching
    create_epochs: bool = False
    epoch_duration_s: float = 1.0
    epoch_overlap_s: float = 0.0
    
    # Normalization
    normalize: bool = True
    normalization_method: str = "robust"  # "standard", "robust", "minmax"
    
    # Spatial filtering
    apply_car: bool = False  # Common Average Reference as spatial filter
    apply_laplacian: bool = False
    
    # Feature extraction
    extract_features: bool = True
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0)
    })
    feature_types: List[str] = field(default_factory=lambda: ["bandpower", "connectivity"])
    connectivity_method: str = "plv"  # "plv", "coherence", "wpli"
    
    # Output
    save_interim: bool = True
    interim_dir: str = "./data/interim/eeg"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "PreprocessingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, file_path: str) -> str:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return file_path
    
    @classmethod
    def load(cls, file_path: str) -> "PreprocessingConfig":
        """Load config from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing pipeline.
    
    This class implements a modular preprocessing pipeline for EEG data,
    with configurable steps including filtering, artifact removal,
    bad channel detection, ICA, and feature extraction.
    """
    
    def __init__(self, config: Optional[Union[Dict, PreprocessingConfig]] = None):
        """
        Initialize the EEG preprocessor.
        
        Args:
            config: Preprocessing configuration (dictionary or PreprocessingConfig)
        """
        self.logger = self._setup_logging()
        
        # Set configuration
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, dict):
            self.config = PreprocessingConfig.from_dict(config)
        else:
            self.config = config
        
        # Create interim directory if needed
        if self.config.save_interim:
            os.makedirs(self.config.interim_dir, exist_ok=True)
        
        self.logger.info("EEG Preprocessor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("EEGPreprocessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def preprocess(self, raw: mne.io.Raw, file_id: Optional[str] = None) -> Dict:
        """
        Apply the complete preprocessing pipeline to raw EEG data.
        
        Args:
            raw: MNE Raw object containing EEG data
            file_id: Optional identifier for the file (for saving interim results)
            
        Returns:
            Dictionary with preprocessed data and processing results
        """
        if file_id is None:
            file_id = f"eeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting preprocessing pipeline for {file_id}")
        
        # Create result dictionary
        result = {
            "file_id": file_id,
            "raw_info": {
                "n_channels": len(raw.ch_names),
                "ch_names": raw.ch_names,
                "sfreq": raw.info["sfreq"],
                "duration": raw.times[-1],
                "n_samples": len(raw.times)
            },
            "processing_steps": [],
            "artifacts_removed": 0,
            "bad_channels": [],
            "features": {}
        }
        
        # Save raw input if requested
        if self.config.save_interim:
            self._save_interim(raw, file_id, "00_raw")
        
        # 1. Filter data
        raw, filter_info = self._apply_filtering(raw)
        result["processing_steps"].append({
            "step": "filtering",
            "info": filter_info
        })
        
        if self.config.save_interim:
            self._save_interim(raw, file_id, "01_filtered")
        
        # 2. Resample if needed
        if self.config.resample and self.config.resample_freq != raw.info["sfreq"]:
            raw, resample_info = self._apply_resampling(raw)
            result["processing_steps"].append({
                "step": "resampling",
                "info": resample_info
            })
            
            if self.config.save_interim:
                self._save_interim(raw, file_id, "02_resampled")
        
        # 3. Detect and interpolate bad channels
        if self.config.detect_bad_channels:
            raw, bad_channels_info = self._detect_bad_channels(raw)
            result["processing_steps"].append({
                "step": "bad_channel_detection",
                "info": bad_channels_info
            })
            result["bad_channels"] = bad_channels_info["bad_channels"]
            
            if self.config.save_interim:
                self._save_interim(raw, file_id, "03_bad_channels_fixed")
        
        # 4. Apply referencing
        raw, reference_info = self._apply_reference(raw)
        result["processing_steps"].append({
            "step": "referencing",
            "info": reference_info
        })
        
        if self.config.save_interim:
            self._save_interim(raw, file_id, "04_referenced")
        
        # 5. Artifact detection and removal
        if self.config.reject_artifacts:
            raw, artifacts_info = self._detect_artifacts(raw)
            result["processing_steps"].append({
                "step": "artifact_detection",
                "info": artifacts_info
            })
            result["artifacts_removed"] = artifacts_info["n_segments_removed"]
            
            if self.config.save_interim:
                self._save_interim(raw, file_id, "05_artifacts_removed")
        
        # 6. Apply ICA if requested
        if self.config.apply_ica:
            raw, ica_info = self._apply_ica(raw)
            result["processing_steps"].append({
                "step": "ica",
                "info": ica_info
            })
            
            if self.config.save_interim:
                self._save_interim(raw, file_id, "06_ica_applied")
        
        # 7. Apply spatial filtering if requested
        if self.config.apply_car or self.config.apply_laplacian:
            raw, spatial_info = self._apply_spatial_filtering(raw)
            result["processing_steps"].append({
                "step": "spatial_filtering",
                "info": spatial_info
            })
            
            if self.config.save_interim:
                self._save_interim(raw, file_id, "07_spatial_filtered")
        
        # 8. Create epochs if requested
        epochs = None
        if self.config.create_epochs:
            epochs, epoch_info = self._create_epochs(raw)
            result["processing_steps"].append({
                "step": "epoching",
                "info": epoch_info
            })
            
            if self.config.save_interim:
                self._save_interim(epochs, file_id, "08_epoched", is_epochs=True)
        
        # 9. Normalize data if requested
        if self.config.normalize:
            if epochs is not None:
                data_obj = epochs
                is_epochs = True
            else:
                data_obj = raw
                is_epochs = False
            
            data_obj, norm_info = self._apply_normalization(data_obj, is_epochs=is_epochs)
            result["processing_steps"].append({
                "step": "normalization",
                "info": norm_info
            })
            
            if self.config.save_interim:
                self._save_interim(data_obj, file_id, "09_normalized", is_epochs=is_epochs)
            
            # Update objects
            if is_epochs:
                epochs = data_obj
            else:
                raw = data_obj
        
        # 10. Extract features if requested
        if self.config.extract_features:
            features = {}
            
            # Use epochs if available, otherwise use raw
            if epochs is not None:
                features, feature_info = self._extract_features(epochs, is_epochs=True)
            else:
                features, feature_info = self._extract_features(raw, is_epochs=False)
            
            result["processing_steps"].append({
                "step": "feature_extraction",
                "info": feature_info
            })
            result["features"] = features
            
            # Save features separately
            if self.config.save_interim:
                feature_file = os.path.join(self.config.interim_dir, f"{file_id}_10_features.pkl")
                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)
        
        # Save final result
        if self.config.save_interim:
            result_file = os.path.join(self.config.interim_dir, f"{file_id}_result.json")
            # Remove raw data from result to make it JSON serializable
            json_result = {k: v for k, v in result.items() if k != 'raw_data' and k != 'epoch_data'}
            
            # Convert NumPy arrays to lists for JSON serialization
            def convert_numpy_to_python(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_to_python(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_python(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            # Convert all NumPy types to Python native types
            json_result = convert_numpy_to_python(json_result)
    
            # Now save to JSON
            with open(result_file, 'w') as f:
                json.dump(json_result, f, indent=2)
            self._save_interim(raw, file_id, "00_raw")
        
        # Add data to result
        if epochs is not None:
            result["epoch_data"] = epochs
        result["raw_data"] = raw
        
        self.logger.info(f"Preprocessing completed for {file_id}")
        return result
    
    def _apply_filtering(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Apply frequency filters to the data."""
        self.logger.info("Applying frequency filters")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        # Apply bandpass filter
        if self.config.highpass_freq > 0 or self.config.lowpass_freq < raw.info['sfreq'] / 2:
            raw.filter(
                l_freq=self.config.highpass_freq if self.config.highpass_freq > 0 else None,
                h_freq=self.config.lowpass_freq if self.config.lowpass_freq > 0 else None,
                method='fir',
                phase='zero',
                verbose=False
            )
        
        # Apply notch filter if requested
        if self.config.apply_notch and self.config.notch_freq > 0:
            raw.notch_filter(
                freqs=self.config.notch_freq,
                method='fir',
                verbose=False
            )
        
        filter_info = {
            "highpass": self.config.highpass_freq,
            "lowpass": self.config.lowpass_freq,
            "notch": self.config.notch_freq if self.config.apply_notch else None
        }
        
        return raw, filter_info
    
    def _apply_resampling(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Resample the data to target frequency."""
        self.logger.info(f"Resampling data from {raw.info['sfreq']} Hz to {self.config.resample_freq} Hz")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        # Resample
        original_sfreq = raw.info['sfreq']
        raw.resample(self.config.resample_freq)
        
        resample_info = {
            "original_sfreq": original_sfreq,
            "new_sfreq": raw.info['sfreq']
        }
        
        return raw, resample_info
    
    def _detect_bad_channels(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Detect and interpolate bad channels."""
        self.logger.info("Detecting bad channels")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        # Extract data
        data = raw.get_data()
        ch_names = raw.ch_names
        
        bad_channels = []
        
        if self.config.bad_channel_criteria == "correlation":
            # Correlation-based detection
            corr_matrix = np.corrcoef(data)
            
            # Average correlation with other channels
            mean_corrs = np.mean(corr_matrix, axis=1)
            
            # Identify channels with mean correlation below threshold
            for i, mean_corr in enumerate(mean_corrs):
                if mean_corr < self.config.bad_channel_threshold:
                    bad_channels.append(ch_names[i])
            
            # Safety check: don't mark all channels as bad
            if len(bad_channels) >= len(ch_names) * 0.7:  # If more than 70% marked as bad
                self.logger.warning(f"Too many channels ({len(bad_channels)}/{len(ch_names)}) marked as bad. Using only the worst 30%.")
                
                # Sort channels by correlation and take only the worst 30%
                sorted_indices = np.argsort(mean_corrs)
                worst_indices = sorted_indices[:int(len(ch_names) * 0.3)]
                bad_channels = [ch_names[i] for i in worst_indices]
        
        elif self.config.bad_channel_criteria == "amplitude":
            # Amplitude-based detection
            channel_stds = np.std(data, axis=1)
            channel_ranges = np.ptp(data, axis=1)
            
            # Find outliers in standard deviation
            std_median = np.median(channel_stds)
            std_threshold = std_median * 5  # Channels with 5x median std
            
            # Find outliers in peak-to-peak amplitude
            range_median = np.median(channel_ranges)
            range_threshold = range_median * 5  # Channels with 5x median range
            
            for i in range(len(ch_names)):
                if (channel_stds[i] > std_threshold) or (channel_ranges[i] > range_threshold):
                    bad_channels.append(ch_names[i])
        
        elif self.config.bad_channel_criteria == "spectrum":
            # Spectrum-based detection
            psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=50, n_fft=2048)
            
            # Check power in frequency bands
            freq_mask = (freqs >= 1) & (freqs <= 50)
            mean_psds = np.mean(psds[:, freq_mask], axis=1)
            
            # Identify channels with abnormal power spectrum
            psd_median = np.median(mean_psds)
            psd_threshold_high = psd_median * 3  # 3x median power
            psd_threshold_low = psd_median * 0.1  # 0.1x median power
            
            for i, mean_psd in enumerate(mean_psds):
                if mean_psd > psd_threshold_high or mean_psd < psd_threshold_low:
                    bad_channels.append(ch_names[i])
        
        self.logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        
        # Mark as bad channels
        raw.info['bads'] = bad_channels
        
        # Interpolate bad channels if we have at least 4 good channels
        if len(bad_channels) > 0 and len(ch_names) - len(bad_channels) >= 4:
            self.logger.info(f"Interpolating {len(bad_channels)} bad channels")
            # Set montage if not set (required for interpolation)
            if raw.get_montage() is None:
                try:
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage)
                except Exception as e:
                    self.logger.warning(f"Could not set standard montage: {e}")
                    self.logger.warning("Skipping interpolation due to missing montage")
                    return raw, {"bad_channels": bad_channels, "interpolated": False}
            
            # Interpolate
            try:
                raw = raw.interpolate_bads()
                interpolated = True
            except Exception as e:
                self.logger.warning(f"Error interpolating channels: {e}")
                interpolated = False
        else:
            interpolated = False
        
        bad_channels_info = {
            "bad_channels": bad_channels,
            "interpolated": interpolated,
            "detection_method": self.config.bad_channel_criteria,
            "threshold": self.config.bad_channel_threshold
        }
        
        return raw, bad_channels_info
    
    def _apply_reference(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Apply EEG reference."""
        self.logger.info(f"Applying reference: {self.config.reference}")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        # First, verify there are EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        
        if len(eeg_picks) == 0:
            # No channels marked as EEG, try to mark them
            self.logger.warning("No channels marked as EEG, marking all channels as EEG")
            for ch_name in raw.ch_names:
                # Skip any channels that are clearly not EEG
                if ch_name.lower() in ['stim', 'ecg', 'eog']:
                    continue
                raw.set_channel_types({ch_name: 'eeg'})
            
            # Check again
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            
            if len(eeg_picks) == 0:
                # Still no EEG channels, can't proceed with referencing
                self.logger.error("Could not identify any EEG channels, skipping referencing")
                return raw, {"type": "none", "reason": "no EEG channels identified"}
        
        # Try-except block to catch any referencing errors
        try:
            if self.config.reference == "average":
                # Average reference
                raw.set_eeg_reference("average", verbose=False)
                ref_info = {"type": "average"}
            elif self.config.reference == "mastoids":
                # Mastoid reference (try various naming conventions)
                mastoid_candidates = ["M1", "M2", "A1", "A2", "TP9", "TP10", "P9", "P10"]
                available_mastoids = [ch for ch in mastoid_candidates if ch in raw.ch_names]
                
                if len(available_mastoids) >= 1:
                    ref_info = {"type": "mastoid", "channels": available_mastoids}
                    raw.set_eeg_reference(available_mastoids, verbose=False)
                else:
                    self.logger.warning("No mastoid channels found, falling back to average")
                    raw.set_eeg_reference("average", verbose=False)
                    ref_info = {"type": "average", "fallback": True}
            elif isinstance(self.config.reference, list):
                # Custom reference channels
                available_refs = [ch for ch in self.config.reference if ch in raw.ch_names]
                
                if len(available_refs) >= 1:
                    ref_info = {"type": "custom", "channels": available_refs}
                    raw.set_eeg_reference(available_refs, verbose=False)
                else:
                    self.logger.warning("No specified reference channels found, falling back to average")
                    raw.set_eeg_reference("average", verbose=False)
                    ref_info = {"type": "average", "fallback": True}
            else:
                # Invalid reference type, use average
                self.logger.warning(f"Invalid reference type: {self.config.reference}, using average")
                raw.set_eeg_reference("average", verbose=False)
                ref_info = {"type": "average", "fallback": True}
        except Exception as e:
            # If referencing fails, log it and continue without referencing
            self.logger.error(f"Error applying reference: {e}")
            self.logger.info("Continuing without reference")
            ref_info = {"type": "none", "error": str(e)}
        
        return raw, ref_info
    
    def _detect_artifacts(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Detect and remove artifacts from the data."""
        self.logger.info("Detecting and removing artifacts")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        # Get data and parameters
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Find segments with amplitudes exceeding threshold
        amplitude_threshold = self.config.amplitude_threshold_uv * 1e-6  # Convert to volts
        bad_segments = []
        
        # Flatline detection (consecutive identical values)
        flatline_samples = int(self.config.flatline_duration_s * sfreq)
        
        # Process each channel
        for ch_idx in range(data.shape[0]):
            # Amplitude threshold detection
            over_threshold = np.abs(data[ch_idx]) > amplitude_threshold

            # Extend segments to include nearby samples (100 ms before and after)
            extension = int(0.1 * sfreq)
            # Find segments where amplitude threshold is exceeded
            if np.any(over_threshold):
                extended = np.zeros_like(over_threshold)
                for i in range(len(over_threshold)):
                    if over_threshold[i]:
                        start = max(0, i - extension)
                        end = min(len(over_threshold), i + extension + 1)
                        extended[start:end] = True
                
                over_threshold = extended
            
            # Flatline detection
            diff = np.diff(data[ch_idx])
            zero_diff = (np.abs(diff) < 1e-10)
            
            # Find consecutive samples with zero difference
            for i in range(len(zero_diff) - flatline_samples):
                if np.all(zero_diff[i:i+flatline_samples]):
                    start = max(0, i - extension)
                    end = min(len(over_threshold), i + flatline_samples + extension)
                    over_threshold[start:end] = True
            
            # Add detected segments to the list
            if np.any(over_threshold):
                # Find contiguous segments
                from scipy import ndimage
                labeled, num_features = ndimage.label(over_threshold)
                
                for j in range(1, num_features + 1):
                    segment = np.where(labeled == j)[0]
                    start_sample = segment[0]
                    end_sample = segment[-1]
                    
                    # Convert to time
                    start_time = start_sample / sfreq
                    end_time = end_sample / sfreq
                    
                    bad_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "channel": raw.ch_names[ch_idx]
                    })
        
        # Merge overlapping segments across channels
        merged_segments = []
        if bad_segments:
            # Sort by start time
            bad_segments.sort(key=lambda x: x["start"])
            
            current = bad_segments[0].copy()
            for segment in bad_segments[1:]:
                if segment["start"] <= current["end"]:
                    # Overlapping segments, merge
                    current["end"] = max(current["end"], segment["end"])
                    current["duration"] = current["end"] - current["start"]
                    current["channel"] = f"{current['channel']}, {segment['channel']}"
                else:
                    # Non-overlapping, add current to merged and start new
                    merged_segments.append(current)
                    current = segment.copy()
            
            # Add the last segment
            merged_segments.append(current)
        
        # Create annotation for each bad segment
        for segment in merged_segments:
            raw.annotations.append(
                onset=segment["start"],
                duration=segment["duration"],
                description="BAD_artifact"
            )
        
        # Create a cropped version without the bad segments
        if merged_segments:
            self.logger.info(f"Removing {len(merged_segments)} artifact segments")
            try:
                # This removes segments marked as BAD
                raw = raw.copy().crop(tmin=0, tmax=raw.times[-1], include_tmax=True)
            except Exception as e:
                self.logger.warning(f"Error cropping artifacts: {e}")
        
        artifacts_info = {
            "n_segments": len(bad_segments),
            "n_merged_segments": len(merged_segments),
            "n_segments_removed": len(merged_segments),
            "amplitude_threshold_uv": self.config.amplitude_threshold_uv,
            "flatline_duration_s": self.config.flatline_duration_s
        }
        
        return raw, artifacts_info
    
    def _apply_ica(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Apply ICA for artifact removal."""
        self.logger.info("Applying ICA")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        # Determine number of components
        n_components = self.config.n_ica_components
        if n_components is None:
            # Use 95% of variance by default
            n_components = min(15, len(raw.ch_names) - 1)
        
        # Initialize and fit ICA
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        
        # Apply current amplitude threshold for ICA
        reject = dict(eeg=self.config.amplitude_threshold_uv * 1e-6)
        
        try:
            ica.fit(raw, reject=reject)
            
            # Detect EOG artifacts (if EOG channels exist)
            eog_channels = [ch for ch in raw.ch_names if "EOG" in ch or "eog" in ch.lower()]
            if eog_channels:
                eog_ch = eog_channels[0]
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_ch)
                ica.exclude = eog_indices
            else:
                # Try to find blinks based on frontal channels
                frontal_channels = [ch for ch in raw.ch_names if ch.startswith(('Fp', 'F')) and ch != 'Fz']
                if frontal_channels:
                    frontal_ch = frontal_channels[0]
                    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=frontal_ch)
                    ica.exclude = eog_indices
            
            # Optionally, add automatic component detection
            # For simplicity, we'll use a threshold on kurtosis to find components
            component_properties = ica.get_properties()
            if "kurtosis" in component_properties:
                kurt = component_properties["kurtosis"]
                # Components with very high kurtosis are likely artifacts
                kurt_threshold = np.percentile(kurt, 95)
                high_kurt_indices = np.where(kurt > kurt_threshold)[0]
                ica.exclude.extend(high_kurt_indices)
            
            # Remove duplicate indices
            ica.exclude = list(set(ica.exclude))
            
            self.logger.info(f"Excluding {len(ica.exclude)} ICA components: {ica.exclude}")
            
            # Apply ICA
            raw = ica.apply(raw.copy())
            
            ica_info = {
                "n_components": n_components,
                "n_excluded": len(ica.exclude),
                "excluded_components": ica.exclude
            }
        except Exception as e:
            self.logger.warning(f"Error applying ICA: {e}")
            ica_info = {
                "n_components": n_components,
                "error": str(e),
                "applied": False
            }
        
        return raw, ica_info
    
    def _apply_spatial_filtering(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """Apply spatial filtering techniques."""
        self.logger.info("Applying spatial filtering")
        
        # Copy to avoid modifying original
        raw = raw.copy()
        
        spatial_info = {}
        
        # Apply CAR if requested
        if self.config.apply_car:
            # Common Average Reference
            raw = raw.set_eeg_reference('average', projection=True)
            raw.apply_proj()
            spatial_info["car_applied"] = True
        
        # Apply Surface Laplacian if requested
        if self.config.apply_laplacian:
            try:
                # Check if montage is set
                if raw.get_montage() is None:
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage)
                
                # Apply Laplacian
                raw = mne.preprocessing.compute_current_source_density(raw)
                spatial_info["laplacian_applied"] = True
            except Exception as e:
                self.logger.warning(f"Error applying Laplacian: {e}")
                spatial_info["laplacian_applied"] = False
                spatial_info["laplacian_error"] = str(e)
        
        return raw, spatial_info
    
    def _create_epochs(self, raw: mne.io.Raw) -> Tuple[mne.Epochs, Dict]:
        """Create epochs from continuous data."""
        self.logger.info("Creating epochs")
        
        # Create fixed-length epochs
        epoch_duration = self.config.epoch_duration_s
        overlap = self.config.epoch_overlap_s
        
        # Calculate step between epochs
        step = epoch_duration - overlap
        
        # Create events at regular intervals
        events = mne.make_fixed_length_events(
            raw, 
            id=1, 
            start=0, 
            stop=None, 
            duration=step
        )
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=epoch_duration,
            baseline=None,
            preload=True,
            reject=None,  # We already handled artifact rejection
            proj=False
        )
        
        epoch_info = {
            "n_epochs": len(epochs),
            "duration": epoch_duration,
            "overlap": overlap,
            "step": step
        }
        
        return epochs, epoch_info
    
    def _apply_normalization(self, data_obj: Union[mne.io.Raw, mne.Epochs], 
                            is_epochs: bool = False) -> Tuple[Union[mne.io.Raw, mne.Epochs], Dict]:
        """Apply normalization to the data."""
        self.logger.info(f"Normalizing data using {self.config.normalization_method}")
        
        # Copy to avoid modifying original
        data_obj = data_obj.copy()
        
        # Get data array
        if is_epochs:
            # For epochs, shape is (n_epochs, n_channels, n_times)
            data = data_obj.get_data()
            # Reshape to (n_channels, n_epochs * n_times)
            data_reshaped = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
        else:
            # For raw, shape is (n_channels, n_times)
            data = data_obj.get_data()
            data_reshaped = data
        
        # Apply normalization
        if self.config.normalization_method == "standard":
            # Standardize each channel separately
            scaler = StandardScaler()
            normalized = np.zeros_like(data_reshaped)
            for i in range(data_reshaped.shape[0]):
                normalized[i] = scaler.fit_transform(data_reshaped[i].reshape(-1, 1)).ravel()
        
        elif self.config.normalization_method == "robust":
            # Robust scaling (using median and IQR)
            scaler = RobustScaler()
            normalized = np.zeros_like(data_reshaped)
            for i in range(data_reshaped.shape[0]):
                normalized[i] = scaler.fit_transform(data_reshaped[i].reshape(-1, 1)).ravel()
        
        elif self.config.normalization_method == "minmax":
            # Min-max scaling to [0, 1]
            normalized = np.zeros_like(data_reshaped)
            for i in range(data_reshaped.shape[0]):
                channel_min = data_reshaped[i].min()
                channel_max = data_reshaped[i].max()
                if channel_max > channel_min:
                    normalized[i] = (data_reshaped[i] - channel_min) / (channel_max - channel_min)
                else:
                    normalized[i] = data_reshaped[i]
        
        else:
            self.logger.warning(f"Unknown normalization method: {self.config.normalization_method}")
            return data_obj, {"applied": False, "method": self.config.normalization_method}
        
        # Reshape back to original shape
        if is_epochs:
            normalized = normalized.reshape(data.shape[1], data.shape[0], data.shape[2]).transpose(1, 0, 2)
            data_obj._data = normalized
        else:
            data_obj._data = normalized
        
        norm_info = {
            "applied": True,
            "method": self.config.normalization_method
        }
        
        return data_obj, norm_info
    
    def _extract_features(self, data_obj: Union[mne.io.Raw, mne.Epochs], 
                         is_epochs: bool = False) -> Tuple[Dict, Dict]:
        """Extract features from the preprocessed data."""
        self.logger.info(f"Extracting features: {self.config.feature_types}")
        
        features = {}
        feature_info = {
            "types": self.config.feature_types,
            "bands": self.config.bands
        }
        
        # Extract different feature types
        if "bandpower" in self.config.feature_types:
            band_powers = self._extract_band_powers(data_obj, is_epochs)
            features["bandpower"] = band_powers
        
        if "connectivity" in self.config.feature_types:
            connectivity = self._extract_connectivity(data_obj, is_epochs)
            features["connectivity"] = connectivity
        
        if "time_domain" in self.config.feature_types:
            time_features = self._extract_time_domain_features(data_obj, is_epochs)
            features["time_domain"] = time_features
        
        return features, feature_info
    
    def _extract_band_powers(self, data_obj: Union[mne.io.Raw, mne.Epochs], 
                            is_epochs: bool = False) -> Dict:
        """Extract band power features."""
        self.logger.info("Extracting band power features")
        
        # Get frequency bands
        bands = self.config.bands
        
        # Calculate appropriate FFT length
        if is_epochs:
            # For epochs, check the length of each epoch
            n_times = data_obj.get_data().shape[2]
        else:
            # For raw data, use a longer segment
            n_times = min(int(data_obj.info['sfreq'] * 4), data_obj.get_data().shape[1])
        
        # Make sure n_fft is not larger than n_times
        n_fft = min(int(data_obj.info['sfreq'] * 2), n_times)
        n_overlap = min(int(n_fft * 0.5), n_times - 1)
        
        self.logger.info(f"Using n_fft={n_fft}, n_times={n_times}, n_overlap={n_overlap}")
        
        # Calculate PSDs using newer MNE API
        try:
            if is_epochs:
                # For epochs
                psds, freqs = data_obj.compute_psd(
                    method='welch',
                    fmin=min([band[0] for band in bands.values()]),
                    fmax=max([band[1] for band in bands.values()]),
                    n_fft=n_fft,
                    n_overlap=n_overlap
                ).get_data(return_freqs=True)
            else:
                # For raw data
                psds, freqs = data_obj.compute_psd(
                    method='welch',
                    fmin=min([band[0] for band in bands.values()]),
                    fmax=max([band[1] for band in bands.values()]),
                    n_fft=n_fft,
                    n_overlap=n_overlap
                ).get_data(return_freqs=True)
        except Exception as e:
            self.logger.error(f"Error computing PSD: {e}")
            # Return empty results as fallback
            return {
                "error": str(e),
                "channel_names": data_obj.ch_names
            }
        
        # Calculate band powers
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            # Find frequencies in band
            freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            # Handle empty frequency range
            if not np.any(freq_idx):
                self.logger.warning(f"No frequencies found in band {band_name} ({fmin}-{fmax} Hz)")
                if is_epochs:
                    band_powers[band_name] = np.zeros((psds.shape[0], psds.shape[1]))
                else:
                    band_powers[band_name] = np.zeros(psds.shape[0])
                continue
            
            # Calculate average power in band for each channel/epoch
            if is_epochs:
                # For epochs, psds has shape (n_epochs, n_channels, n_freqs)
                band_power = np.mean(psds[:, :, freq_idx], axis=2)
                
                # Store as (n_epochs, n_channels)
                band_powers[band_name] = band_power
            else:
                # For raw, psds has shape (n_channels, n_freqs)
                band_power = np.mean(psds[:, freq_idx], axis=1)
                
                # Store as (n_channels,)
                band_powers[band_name] = band_power
        
        # Calculate relative band powers
        total_power = np.sum([powers for powers in band_powers.values()], axis=0)
        
        relative_band_powers = {}
        for band_name, powers in band_powers.items():
            if is_epochs:
                # Handle potential division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    relative_powers = powers / total_power
                    relative_powers[np.isinf(relative_powers)] = 0
                    relative_powers[np.isnan(relative_powers)] = 0
            else:
                # Handle potential division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    relative_powers = powers / total_power
                    relative_powers[np.isinf(relative_powers)] = 0
                    relative_powers[np.isnan(relative_powers)] = 0
            
            relative_band_powers[f"rel_{band_name}"] = relative_powers
        
        # Combine absolute and relative powers
        result = {}
        result.update(band_powers)
        result.update(relative_band_powers)
        
        # Add metadata
        result["channel_names"] = data_obj.ch_names
        
        return result
    
    def _extract_connectivity(self, data_obj: Union[mne.io.Raw, mne.Epochs], 
                             is_epochs: bool = False) -> Dict:
        """Extract connectivity features."""
        self.logger.info(f"Extracting connectivity using {self.config.connectivity_method}")
        
        # Get frequency bands
        bands = self.config.bands
        
        # Calculate connectivity based on method
        result = {}
        
        try:
            if self.config.connectivity_method == "plv":
                # Phase Locking Value
                if is_epochs:
                    # For epochs, we can use MNE's connectivity functions
                    for band_name, (fmin, fmax) in bands.items():
                        con = mne.connectivity.spectral_connectivity(
                            data_obj,
                            method='plv',
                            mode='multitaper',
                            sfreq=data_obj.info['sfreq'],
                            fmin=fmin,
                            fmax=fmax,
                            faverage=True,
                            verbose=False
                        )
                        
                        # Extract connectivity matrix
                        n_channels = len(data_obj.ch_names)
                        con_matrix = np.zeros((n_channels, n_channels))
                        
                        # Fill in the matrix (MNE returns lower triangle)
                        idx = 0
                        for i in range(n_channels):
                            for j in range(i):
                                con_matrix[i, j] = con[0][idx, 0]
                                con_matrix[j, i] = con[0][idx, 0]  # Symmetrical
                                idx += 1
                        
                        result[band_name] = con_matrix
                else:
                    # For raw data, we need to segment it first
                    # Create temporary epochs for connectivity calculation
                    events = mne.make_fixed_length_events(
                        data_obj, 
                        id=1, 
                        start=0, 
                        stop=None, 
                        duration=1.0
                    )
                    
                    temp_epochs = mne.Epochs(
                        data_obj,
                        events,
                        tmin=0,
                        tmax=1.0,
                        baseline=None,
                        preload=True
                    )
                    
                    # Calculate connectivity for each band
                    for band_name, (fmin, fmax) in bands.items():
                        con = mne.connectivity.spectral_connectivity(
                            temp_epochs,
                            method='plv',
                            mode='multitaper',
                            sfreq=data_obj.info['sfreq'],
                            fmin=fmin,
                            fmax=fmax,
                            faverage=True,
                            verbose=False
                        )
                        
                        # Extract connectivity matrix
                        n_channels = len(data_obj.ch_names)
                        con_matrix = np.zeros((n_channels, n_channels))
                        
                        # Fill in the matrix (MNE returns lower triangle)
                        idx = 0
                        for i in range(n_channels):
                            for j in range(i):
                                con_matrix[i, j] = con[0][idx, 0]
                                con_matrix[j, i] = con[0][idx, 0]  # Symmetrical
                                idx += 1
                        
                        result[band_name] = con_matrix
            
            elif self.config.connectivity_method == "coherence":
                # Coherence
                # Similar implementation as PLV but with method='coh'
                if is_epochs:
                    for band_name, (fmin, fmax) in bands.items():
                        con = mne.connectivity.spectral_connectivity(
                            data_obj,
                            method='coh',
                            mode='multitaper',
                            sfreq=data_obj.info['sfreq'],
                            fmin=fmin,
                            fmax=fmax,
                            faverage=True,
                            verbose=False
                        )
                        
                        n_channels = len(data_obj.ch_names)
                        con_matrix = np.zeros((n_channels, n_channels))
                        
                        idx = 0
                        for i in range(n_channels):
                            for j in range(i):
                                con_matrix[i, j] = con[0][idx, 0]
                                con_matrix[j, i] = con[0][idx, 0]
                                idx += 1
                        
                        result[band_name] = con_matrix
                else:
                    # For raw data, segment first
                    events = mne.make_fixed_length_events(
                        data_obj, 
                        id=1, 
                        start=0, 
                        stop=None, 
                        duration=1.0
                    )
                    
                    temp_epochs = mne.Epochs(
                        data_obj,
                        events,
                        tmin=0,
                        tmax=1.0,
                        baseline=None,
                        preload=True
                    )
                    
                    for band_name, (fmin, fmax) in bands.items():
                        con = mne.connectivity.spectral_connectivity(
                            temp_epochs,
                            method='coh',
                            mode='multitaper',
                            sfreq=data_obj.info['sfreq'],
                            fmin=fmin,
                            fmax=fmax,
                            faverage=True,
                            verbose=False
                        )
                        
                        n_channels = len(data_obj.ch_names)
                        con_matrix = np.zeros((n_channels, n_channels))
                        
                        idx = 0
                        for i in range(n_channels):
                            for j in range(i):
                                con_matrix[i, j] = con[0][idx, 0]
                                con_matrix[j, i] = con[0][idx, 0]
                                idx += 1
                        
                        result[band_name] = con_matrix
            
            elif self.config.connectivity_method == "wpli":
                # Weighted Phase Lag Index
                # Similar implementation but with method='wpli'
                if is_epochs:
                    for band_name, (fmin, fmax) in bands.items():
                        con = mne.connectivity.spectral_connectivity(
                            data_obj,
                            method='wpli',
                            mode='multitaper',
                            sfreq=data_obj.info['sfreq'],
                            fmin=fmin,
                            fmax=fmax,
                            faverage=True,
                            verbose=False
                        )
                        
                        n_channels = len(data_obj.ch_names)
                        con_matrix = np.zeros((n_channels, n_channels))
                        
                        idx = 0
                        for i in range(n_channels):
                            for j in range(i):
                                con_matrix[i, j] = con[0][idx, 0]
                                con_matrix[j, i] = con[0][idx, 0]
                                idx += 1
                        
                        result[band_name] = con_matrix
                else:
                    # For raw data, segment first
                    events = mne.make_fixed_length_events(
                        data_obj, 
                        id=1, 
                        start=0, 
                        stop=None, 
                        duration=1.0
                    )
                    
                    temp_epochs = mne.Epochs(
                        data_obj,
                        events,
                        tmin=0,
                        tmax=1.0,
                        baseline=None,
                        preload=True
                    )
                    
                    for band_name, (fmin, fmax) in bands.items():
                        con = mne.connectivity.spectral_connectivity(
                            temp_epochs,
                            method='wpli',
                            mode='multitaper',
                            sfreq=data_obj.info['sfreq'],
                            fmin=fmin,
                            fmax=fmax,
                            faverage=True,
                            verbose=False
                        )
                        
                        n_channels = len(data_obj.ch_names)
                        con_matrix = np.zeros((n_channels, n_channels))
                        
                        idx = 0
                        for i in range(n_channels):
                            for j in range(i):
                                con_matrix[i, j] = con[0][idx, 0]
                                con_matrix[j, i] = con[0][idx, 0]
                                idx += 1
                        
                        result[band_name] = con_matrix
            
            else:
                self.logger.warning(f"Unknown connectivity method: {self.config.connectivity_method}")
        
        except Exception as e:
            self.logger.warning(f"Error calculating connectivity: {e}")
            result["error"] = str(e)
        
        # Add metadata
        result["channel_names"] = data_obj.ch_names
        result["method"] = self.config.connectivity_method
        
        return result
    
    def _extract_time_domain_features(self, data_obj: Union[mne.io.Raw, mne.Epochs], 
                                     is_epochs: bool = False) -> Dict:
        """Extract time domain features."""
        self.logger.info("Extracting time domain features")
        
        # Get data
        if is_epochs:
            # Shape: (n_epochs, n_channels, n_times)
            data = data_obj.get_data()
        else:
            # Shape: (n_channels, n_times)
            data = data_obj.get_data()
        
        # Initialize result dictionary
        result = {}
        
        # Calculate features
        if is_epochs:
            # Statistical features for each channel and epoch
            n_epochs, n_channels, _ = data.shape
            
            # Mean
            result["mean"] = np.mean(data, axis=2)  # (n_epochs, n_channels)
            
            # Standard deviation
            result["std"] = np.std(data, axis=2)
            
            # Variance
            result["var"] = np.var(data, axis=2)
            
            # Kurtosis
            from scipy.stats import kurtosis
            kurt = np.zeros((n_epochs, n_channels))
            for i in range(n_epochs):
                for j in range(n_channels):
                    kurt[i, j] = kurtosis(data[i, j, :])
            result["kurtosis"] = kurt
            
            # Skewness
            from scipy.stats import skew
            skewness = np.zeros((n_epochs, n_channels))
            for i in range(n_epochs):
                for j in range(n_channels):
                    skewness[i, j] = skew(data[i, j, :])
            result["skewness"] = skewness
            
            # Hjorth parameters
            hjorth_activity = np.var(data, axis=2)  # Same as variance
            
            hjorth_mobility = np.zeros((n_epochs, n_channels))
            hjorth_complexity = np.zeros((n_epochs, n_channels))
            
            for i in range(n_epochs):
                for j in range(n_channels):
                    # First derivative
                    d1 = np.diff(data[i, j, :])
                    # Second derivative
                    d2 = np.diff(d1)
                    
                    # Mobility
                    if np.var(data[i, j, :]) > 0:
                        hjorth_mobility[i, j] = np.sqrt(np.var(d1) / np.var(data[i, j, :]))
                    else:
                        hjorth_mobility[i, j] = 0
                    
                    # Complexity
                    if np.var(d1) > 0:
                        hjorth_complexity[i, j] = np.sqrt(np.var(d2) / np.var(d1)) / hjorth_mobility[i, j]
                    else:
                        hjorth_complexity[i, j] = 0
            
            result["hjorth_activity"] = hjorth_activity
            result["hjorth_mobility"] = hjorth_mobility
            result["hjorth_complexity"] = hjorth_complexity
            
            # Line length (a measure of signal complexity)
            line_length = np.zeros((n_epochs, n_channels))
            for i in range(n_epochs):
                for j in range(n_channels):
                    line_length[i, j] = np.sum(np.abs(np.diff(data[i, j, :])))
            
            result["line_length"] = line_length
        
        else:
            # For raw data, calculate features across time
            n_channels, _ = data.shape
            
            # Mean
            result["mean"] = np.mean(data, axis=1)
            
            # Standard deviation
            result["std"] = np.std(data, axis=1)
            
            # Variance
            result["var"] = np.var(data, axis=1)
            
            # Kurtosis
            from scipy.stats import kurtosis
            result["kurtosis"] = kurtosis(data, axis=1)
            
            # Skewness
            from scipy.stats import skew
            result["skewness"] = skew(data, axis=1)
            
            # Hjorth parameters
            hjorth_activity = np.var(data, axis=1)  # Same as variance
            
            hjorth_mobility = np.zeros(n_channels)
            hjorth_complexity = np.zeros(n_channels)
            
            for j in range(n_channels):
                # First derivative
                d1 = np.diff(data[j, :])
                # Second derivative
                d2 = np.diff(d1)
                
                # Mobility
                if np.var(data[j, :]) > 0:
                    hjorth_mobility[j] = np.sqrt(np.var(d1) / np.var(data[j, :]))
                else:
                    hjorth_mobility[j] = 0
                
                # Complexity
                if np.var(d1) > 0:
                    hjorth_complexity[j] = np.sqrt(np.var(d2) / np.var(d1)) / hjorth_mobility[j]
                else:
                    hjorth_complexity[j] = 0
            
            result["hjorth_activity"] = hjorth_activity
            result["hjorth_mobility"] = hjorth_mobility
            result["hjorth_complexity"] = hjorth_complexity
            
            # Line length
            line_length = np.zeros(n_channels)
            for j in range(n_channels):
                line_length[j] = np.sum(np.abs(np.diff(data[j, :])))
            
            result["line_length"] = line_length
        
        # Add metadata
        result["channel_names"] = data_obj.ch_names
        
        return result
    
    def _save_interim(self, data_obj: Union[mne.io.Raw, mne.Epochs], 
                     file_id: str, step_name: str, is_epochs: bool = False) -> None:
        """Save interim results during preprocessing."""
        if not self.config.save_interim:
            return
        
        # Create directory if needed
        os.makedirs(self.config.interim_dir, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(self.config.interim_dir, f"{file_id}_{step_name}")
        
        # Save based on type
        if is_epochs:
            # Save epochs as -epo.fif
            epochs_path = file_path + "-epo.fif"
            data_obj.save(epochs_path, overwrite=True)
        else:
            # Save raw as -raw.fif
            raw_path = file_path + "-raw.fif"
            data_obj.save(raw_path, overwrite=True)


# Example usage
if __name__ == "__main__":
    import sys
    from neural_ai_network.eeg.data_loader import EEGDataLoader
    
    # Initialize
    loader = EEGDataLoader()
    
    # Load sample data
    try:
        # Try to load EEGLAB sample first
        raw = loader.load_eeglab_dataset("eeglab_sample")
    except Exception as e:
        print(f"Error loading EEGLAB sample: {e}")
        try:
            # Try Temple EEG sample as fallback
            raw = loader.load_temple_eeg_file()
        except Exception as e:
            print(f"Error loading Temple EEG sample: {e}")
            print("Using a simulated EEG sample instead")
            
            # Create simulated EEG data
            from mne.simulation import simulate_raw
            
            # Create info object
            info = mne.create_info(
                ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                          'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'],
                ch_types=['eeg'] * 19,
                sfreq=256
            )
            
            # Simulate 60 seconds of data
            raw = simulate_raw(info, duration=60)
    
    # Create a default preprocessing configuration
    config = PreprocessingConfig()
    
    # Customize configuration for demonstration
    config.lowpass_freq = 40.0
    config.highpass_freq = 1.0
    config.apply_notch = True
    config.notch_freq = 60.0
    config.resample = True
    config.resample_freq = 250.0
    config.detect_bad_channels = True
    config.reject_artifacts = True
    config.create_epochs = True
    config.epoch_duration_s = 2.0
    config.epoch_overlap_s = 1.0
    config.extract_features = True
    config.feature_types = ["bandpower", "connectivity", "time_domain"]
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(config)
    
    # Apply preprocessing
    result = preprocessor.preprocess(raw, file_id="example")
    
    # Print summary
    print("\nPreprocessing completed:")
    print(f"- Original data: {result['raw_info']['n_channels']} channels, "
          f"{result['raw_info']['duration']:.1f}s at {result['raw_info']['sfreq']} Hz")
    
    print("- Processing steps:")
    for step in result["processing_steps"]:
        print(f"  * {step['step']}")
    
    print(f"- Bad channels detected: {result['bad_channels']}")
    print(f"- Artifacts removed: {result['artifacts_removed']} segments")
    
    if "epoch_data" in result:
        print(f"- Epochs created: {len(result['epoch_data'])} epochs of {config.epoch_duration_s}s")
    
    print("- Features extracted:")
    for feature_type in result["features"]:
        print(f"  * {feature_type}")
        
    # Plot some results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        if "bandpower" in result["features"]:
            # Plot band powers
            band_powers = result["features"]["bandpower"]
            bands = [b for b in band_powers.keys() if b not in ["channel_names"] and not b.startswith("rel_")]
            
            plt.figure(figsize=(10, 6))
            
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
                plt.tight_layout()
            else:
                # For continuous data, plot band power per channel
                x = np.arange(len(bands))
                width = 0.8 / len(band_powers["channel_names"])
                
                for i, channel in enumerate(band_powers["channel_names"]):
                    channel_powers = [band_powers[band][i] for band in bands]
                    plt.bar(x + i*width, channel_powers, width, label=channel)
                
                plt.xlabel('Frequency Band')
                plt.ylabel('Power (µV²/Hz)')
                plt.title('Band Powers by Channel')
                plt.xticks(x + width * (len(band_powers["channel_names"]) - 1) / 2, bands)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
                plt.tight_layout()
            
            plt.savefig(os.path.join(config.interim_dir, "band_powers.png"))
            plt.close()
            
        if "connectivity" in result["features"] and "alpha" in result["features"]["connectivity"]:
            # Plot alpha band connectivity matrix
            plt.figure(figsize=(8, 6))
            
            con_matrix = result["features"]["connectivity"]["alpha"]
            channel_names = result["features"]["connectivity"]["channel_names"]
            
            plt.imshow(con_matrix, cmap='viridis', interpolation='none')
            plt.colorbar(label=result["features"]["connectivity"]["method"])
            plt.title('Alpha Band Connectivity')
            
            # Add channel labels if not too many
            if len(channel_names) <= 20:
                plt.xticks(np.arange(len(channel_names)), channel_names, rotation=45)
                plt.yticks(np.arange(len(channel_names)), channel_names)
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.interim_dir, "connectivity_matrix.png"))
            plt.close()
        
        print(f"\nPlots saved to {config.interim_dir}")
    
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    print("\nProcessed data and features are available in the 'result' dictionary.")