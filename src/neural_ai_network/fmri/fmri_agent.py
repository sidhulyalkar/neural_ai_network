# fmri_agent.py
import os
import json
import numpy as np
import logging
import nibabel as nib
from typing import Dict, List, Any, Optional, Tuple
import pika
import threading
import time
from nilearn import image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class FMRIProcessingAgent:
    """
    Specialized agent for processing fMRI data.
    
    This agent handles various fMRI data formats, performs preprocessing,
    feature extraction, and analysis using both traditional methods and
    advanced machine learning approaches.
    """
    
    def __init__(self, config_path: str = "fmri_agent_config.json"):
        """
        Initialize the fMRI processing agent.
        
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
        
        self.logger.info("fMRI Processing Agent initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger("FMRIAgent")
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
                    "queue": "fmri_processing"
                },
                "preprocessing": {
                    "slice_timing": True,
                    "motion_correction": True,
                    "spatial_smoothing": {
                        "enabled": True,
                        "fwhm_mm": 6.0
                    },
                    "temporal_filtering": {
                        "enabled": True,
                        "high_pass": 0.01,
                        "low_pass": None
                    },
                    "normalization": {
                        "enabled": True,
                        "template": "MNI152"
                    },
                    "skull_stripping": True
                },
                "analysis": {
                    "task_based": {
                        "enabled": True,
                        "model": "glm"
                    },
                    "resting_state": {
                        "enabled": True,
                        "methods": ["seed_based", "ica"]
                    },
                    "machine_learning": {
                        "enabled": True,
                        "models": ["svm", "random_forest"]
                    }
                },
                "storage": {
                    "processed_data": "./data/processed/fmri",
                    "results": "./data/results/fmri",
                    "models": "./models/fmri"
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
        self.logger.info("fMRI Processing Agent stopped")
    
    def process_data(self, data_path: str, parameters: Dict = None) -> Dict:
        """
        Process fMRI data.
        
        Args:
            data_path: Path to fMRI data file or directory
            parameters: Optional processing parameters to override defaults
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Ensure parameters is not None
            parameters = parameters or {}
            
            # Load the data
            data_info = self._load_data(data_path)
            
            # Preprocess the data
            preprocessed_data = self._preprocess_data(data_info, parameters)
            
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
                "features": list(features.keys()),
                "analysis": list(analysis_results.keys())
            }
        except Exception as e:
            self.logger.error(f"Error processing data {data_path}: {e}")
            return {
                "status": "error",
                "data_path": data_path,
                "error": str(e)
            }
    
    def _load_data(self, data_path: str) -> Dict:
        """
        Load fMRI data from file or directory.
        
        Args:
            data_path: Path to fMRI data (file or directory)
            
        Returns:
            Dictionary containing loaded data and metadata
        """
        self.logger.info(f"Loading data from {data_path}")
        
        result = {
            "data_path": data_path,
            "functional": None,
            "anatomical": None,
            "mask": None,
            "events": None,
            "confounds": None,
            "tr": None,
            "metadata": {}
        }
        
        # Check if it's a file or directory
        if os.path.isfile(data_path):
            # Single file
            if data_path.endswith(('.nii', '.nii.gz')):
                self.logger.info(f"Loading single NIfTI file: {data_path}")
                img = nib.load(data_path)
                if len(img.shape) > 3:  # 4D file (functional)
                    result["functional"] = img
                    result["tr"] = img.header.get_zooms()[3] if len(img.header.get_zooms()) > 3 else None
                else:  # 3D file (anatomical or mask)
                    result["anatomical"] = img
                
                # Extract metadata from header
                result["metadata"] = {
                    "dimensions": img.shape,
                    "voxel_size": img.header.get_zooms(),
                    "affine": img.affine.tolist()
                }
            elif data_path.endswith('.json'):
                # Try to load JSON metadata
                with open(data_path, 'r') as f:
                    result["metadata"] = json.load(f)
                
                # Check for TR in metadata
                if "RepetitionTime" in result["metadata"]:
                    result["tr"] = result["metadata"]["RepetitionTime"]
            elif data_path.endswith(('.tsv', '.csv')):
                # Try to load events or confounds
                try:
                    if "events" in data_path.lower():
                        result["events"] = pd.read_csv(data_path, sep='\t' if data_path.endswith('.tsv') else ',')
                    elif "confounds" in data_path.lower() or "regressors" in data_path.lower():
                        result["confounds"] = pd.read_csv(data_path, sep='\t' if data_path.endswith('.tsv') else ',')
                except Exception as e:
                    self.logger.warning(f"Error loading tabular data {data_path}: {e}")
        else:
            # Directory - look for files with specific patterns
            self.logger.info(f"Loading data from directory: {data_path}")
            
            # Look for functional images
            func_files = self._find_files(data_path, ['*bold*.nii*', '*func*.nii*', '*fmri*.nii*'])
            if func_files:
                result["functional"] = nib.load(func_files[0])
                # Extract TR from header
                result["tr"] = result["functional"].header.get_zooms()[3] if len(result["functional"].header.get_zooms()) > 3 else None
            
            # Look for anatomical images
            anat_files = self._find_files(data_path, ['*T1*.nii*', '*anat*.nii*', '*struct*.nii*'])
            if anat_files:
                result["anatomical"] = nib.load(anat_files[0])
            
            # Look for mask files
            mask_files = self._find_files(data_path, ['*mask*.nii*', '*brain*.nii*'])
            if mask_files:
                result["mask"] = nib.load(mask_files[0])
            
            # Look for events files
            event_files = self._find_files(data_path, ['*events*.tsv', '*events*.csv', '*task*.tsv'])
            if event_files:
                result["events"] = pd.read_csv(event_files[0], sep='\t' if event_files[0].endswith('.tsv') else ',')
            
            # Look for confounds/nuisance regressors
            confound_files = self._find_files(data_path, ['*confounds*.tsv', '*regressors*.tsv', '*motion*.txt'])
            if confound_files:
                result["confounds"] = pd.read_csv(confound_files[0], sep='\t' if confound_files[0].endswith('.tsv') else ',')
            
            # Look for JSON metadata
            json_files = self._find_files(data_path, ['*bold*.json', '*func*.json', '*dataset*.json'])
            if json_files:
                with open(json_files[0], 'r') as f:
                    result["metadata"] = json.load(f)
                
                # Check for TR in metadata
                if "RepetitionTime" in result["metadata"]:
                    result["tr"] = result["metadata"]["RepetitionTime"]
        
        # Basic validation
        if not result["functional"] and not result["anatomical"]:
            raise ValueError(f"No valid fMRI data found in {data_path}")
        
        if result["functional"]:
            self.logger.info(f"Loaded functional data: {result['functional'].shape}")
        if result["anatomical"]:
            self.logger.info(f"Loaded anatomical data: {result['anatomical'].shape}")
        
        return result
    
    def _find_files(self, directory: str, patterns: List[str]) -> List[str]:
        """Find files matching any of the patterns in the directory."""
        import glob
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(os.path.join(directory, pattern)))
            # Also check subdirectories
            matches.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
        return matches
    
    def _preprocess_data(self, data_info: Dict, parameters: Dict) -> Dict:
        """
        Preprocess fMRI data.
        
        Args:
            data_info: Dictionary with loaded data and metadata
            parameters: Processing parameters
            
        Returns:
            Dictionary with preprocessed data
        """
        self.logger.info("Preprocessing fMRI data")
        
        # Merge configuration with provided parameters
        config = self.config["preprocessing"]
        for key, value in parameters.get("preprocessing", {}).items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        
        # Initialize result with original data
        result = data_info.copy()
        result["preprocessing_steps"] = []
        
        # Ensure we have functional data to process
        if not result["functional"]:
            self.logger.warning("No functional data to preprocess")
            return result
        
        # Extract functional data for processing
        func_img = result["functional"]
        
        # Initialize NiLearn memory cache for efficiency
        from nilearn.cache import Memory
        cache_dir = os.path.join(self.config["storage"]["processed_data"], "cache")
        os.makedirs(cache_dir, exist_ok=True)
        memory = Memory(cache_dir)
        
        # 1. Slice timing correction
        if config.get("slice_timing", False) and result["tr"]:
            self.logger.info("Applying slice timing correction")
            try:
                from nilearn.image import slice_time_correction
                # Get number of slices from the third dimension
                n_slices = func_img.shape[2]
                
                # Default slice order is sequential (0, 1, 2, ...)
                slice_order = parameters.get("slice_order", list(range(n_slices)))
                
                # Apply slice timing correction
                func_img = slice_time_correction(
                    func_img,
                    result["tr"],
                    slice_order,
                    memory=memory
                )
                
                result["preprocessing_steps"].append("slice_timing_correction")
            except Exception as e:
                self.logger.warning(f"Error in slice timing correction: {e}")
        
        # 2. Motion correction
        if config.get("motion_correction", True):
            self.logger.info("Applying motion correction")
            try:
                from nilearn.image import resample_img
                from nipype.interfaces import fsl
                from nipype.interfaces.fsl import MCFLIRT
                import tempfile
                import shutil
                
                # Create temporary directory for MCFLIRT
                temp_dir = tempfile.mkdtemp()
                
                # Save input image to temp dir
                input_file = os.path.join(temp_dir, "input.nii.gz")
                nib.save(func_img, input_file)
                
                # Run MCFLIRT
                mcflirt = MCFLIRT()
                mcflirt.inputs.in_file = input_file
                mcflirt.inputs.cost = "mutualinfo"
                mcflirt.inputs.mean_vol = True
                mcflirt.inputs.output_type = "NIFTI_GZ"
                mcflirt.inputs.ref_vol = 0  # Use first volume as reference
                mcflirt.inputs.out_file = os.path.join(temp_dir, "mcf.nii.gz")
                mcflirt.run()
                
                # Load motion-corrected image
                func_img = nib.load(os.path.join(temp_dir, "mcf.nii.gz"))
                
                # Load motion parameters if available
                motion_params_file = os.path.join(temp_dir, "mcf.par")
                if os.path.exists(motion_params_file):
                    motion_params = np.loadtxt(motion_params_file)
                    result["motion_parameters"] = motion_params
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)
                
                result["preprocessing_steps"].append("motion_correction")
            except Exception as e:
                self.logger.warning(f"Error in motion correction: {e}")
                self.logger.warning("Continuing without motion correction")
        
        # 3. Skull stripping (brain extraction)
        if config.get("skull_stripping", True) and result["anatomical"]:
            self.logger.info("Applying skull stripping to anatomical image")
            try:
                from nilearn.masking import compute_brain_mask
                
                # Create brain mask
                brain_mask = compute_brain_mask(result["anatomical"])
                
                # Apply mask to anatomical image
                result["anatomical_brain"] = image.math_img("img1 * img2", 
                                                           img1=result["anatomical"], 
                                                           img2=brain_mask)
                
                result["brain_mask"] = brain_mask
                result["preprocessing_steps"].append("skull_stripping")
            except Exception as e:
                self.logger.warning(f"Error in skull stripping: {e}")
        
        # 4. Spatial normalization (registration to template)
        if config.get("normalization", {}).get("enabled", True) and result.get("anatomical", None):
            template_name = config.get("normalization", {}).get("template", "MNI152")
            self.logger.info(f"Applying spatial normalization to {template_name} template")
            
            try:
                from nilearn.image import resample_to_img
                
                # Get template image
                if template_name == "MNI152":
                    from nilearn.datasets import load_mni152_template
                    template = load_mni152_template()
                else:
                    # Use custom template if specified
                    template_path = config.get("normalization", {}).get("template_path")
                    if template_path and os.path.exists(template_path):
                        template = nib.load(template_path)
                    else:
                        raise ValueError(f"Unknown template: {template_name}")
                
                # Normalize anatomical to template
                if result.get("anatomical_brain"):
                    # Use skull-stripped image if available
                    result["anatomical_normalized"] = resample_to_img(
                        result["anatomical_brain"],
                        template,
                        interpolation="continuous"
                    )
                else:
                    result["anatomical_normalized"] = resample_to_img(
                        result["anatomical"],
                        template,
                        interpolation="continuous"
                    )
                
                # Apply same transformation to functional data
                target_shape = template.shape[:3]
                target_affine = template.affine
                
                result["functional_normalized"] = image.resample_img(
                    func_img,
                    target_affine=target_affine,
                    target_shape=target_shape,
                    interpolation="continuous"
                )
                
                # Update func_img to use normalized version
                func_img = result["functional_normalized"]
                
                result["preprocessing_steps"].append("spatial_normalization")
            except Exception as e:
                self.logger.warning(f"Error in spatial normalization: {e}")
        
        # 5. Spatial smoothing
        if config.get("spatial_smoothing", {}).get("enabled", True):
            fwhm = config.get("spatial_smoothing", {}).get("fwhm_mm", 6.0)
            self.logger.info(f"Applying spatial smoothing with FWHM={fwhm}mm")
            
            try:
                from nilearn.image import smooth_img
                
                # Apply smoothing
                func_img = smooth_img(func_img, fwhm=fwhm)
                
                result["preprocessing_steps"].append("spatial_smoothing")
            except Exception as e:
                self.logger.warning(f"Error in spatial smoothing: {e}")
        
        # 6. Temporal filtering
        if config.get("temporal_filtering", {}).get("enabled", True):
            high_pass = config.get("temporal_filtering", {}).get("high_pass", 0.01)
            low_pass = config.get("temporal_filtering", {}).get("low_pass", None)
            
            if high_pass or low_pass:
                self.logger.info(f"Applying temporal filtering (high_pass={high_pass}, low_pass={low_pass})")
                
                try:
                    from nilearn.image import clean_img
                    
                    # Apply temporal filtering
                    func_img = clean_img(
                        func_img,
                        high_pass=high_pass,
                        low_pass=low_pass,
                        t_r=result["tr"]
                    )
                    
                    result["preprocessing_steps"].append("temporal_filtering")
                except Exception as e:
                    self.logger.warning(f"Error in temporal filtering: {e}")
        
        # Store the preprocessed functional image
        result["preprocessed_functional"] = func_img
        
        self.logger.info(f"Preprocessing completed with steps: {result['preprocessing_steps']}")
        return result
    
    def _extract_features(self, preprocessed_data: Dict, parameters: Dict) -> Dict:
        """
        Extract features from preprocessed fMRI data.
        
        Args:
            preprocessed_data: Dictionary with preprocessed data
            parameters: Processing parameters
            
        Returns:
            Dictionary of extracted features
        """
        self.logger.info("Extracting features from fMRI data")
        
        features = {}
        
        # Get the preprocessed functional image
        func_img = preprocessed_data.get("preprocessed_functional", 
                                        preprocessed_data.get("functional_normalized", 
                                                             preprocessed_data.get("functional")))
        
        if func_img is None:
            self.logger.warning("No functional data for feature extraction")
            return features
        
        # 1. Extract ROI time series
        try:
            from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker
            
            # Use atlas-based ROIs
            from nilearn.datasets import (
                load_mni152_template,
                load_harvard_oxford_atlas,
                load_atlas_aal,
                load_atlas_basc_multiscale_2015
            )
            
            # Harvard-Oxford atlas
            ho_atlas = load_harvard_oxford_atlas('cort-maxprob-thr25-2mm')
            ho_masker = NiftiLabelsMasker(
                labels_img=ho_atlas['maps'],
                labels=ho_atlas['labels'],
                standardize=True,
                memory='nilearn_cache'
            )
            ho_time_series = ho_masker.fit_transform(func_img)
            
            # AAL atlas
            aal_atlas = load_atlas_aal()
            aal_masker = NiftiLabelsMasker(
                labels_img=aal_atlas['maps'],
                labels=aal_atlas['labels'],
                standardize=True,
                memory='nilearn_cache'
            )
            aal_time_series = aal_masker.fit_transform(func_img)
            
            # Store ROI time series
            features["roi_time_series"] = {
                "harvard_oxford": {
                    "data": ho_time_series,
                    "labels": ho_atlas['labels']
                },
                "aal": {
                    "data": aal_time_series,
                    "labels": aal_atlas['labels']
                }
            }
            
            self.logger.info(f"Extracted ROI time series from {len(ho_atlas['labels'])} Harvard-Oxford regions and {len(aal_atlas['labels'])} AAL regions")
        except Exception as e:
            self.logger.warning(f"Error extracting ROI time series: {e}")
        
        # 2. Calculate functional connectivity matrices
        try:
            from nilearn.connectome import ConnectivityMeasure
            
            # Calculate correlation matrices
            correlation_measure = ConnectivityMeasure(kind='correlation')
            
            # Harvard-Oxford connectivity
            ho_correlation = correlation_measure.fit_transform([features["roi_time_series"]["harvard_oxford"]["data"]])[0]
            
            # AAL connectivity
            aal_correlation = correlation_measure.fit_transform([features["roi_time_series"]["aal"]["data"]])[0]
            
            # Store connectivity matrices
            features["functional_connectivity"] = {
                "harvard_oxford": {
                    "correlation": ho_correlation,
                    "labels": features["roi_time_series"]["harvard_oxford"]["labels"]
                },
                "aal": {
                    "correlation": aal_correlation,
                    "labels": features["roi_time_series"]["aal"]["labels"]
                }
            }
            
            self.logger.info("Calculated functional connectivity matrices")
        except Exception as e:
            self.logger.warning(f"Error calculating functional connectivity: {e}")
        
        # 3. Extract graph theory metrics
        try:
            import networkx as nx
            
            # Calculate graph metrics for each atlas
            atlases = ["harvard_oxford", "aal"]
            graph_metrics = {}
            
            for atlas in atlases:
                if atlas in features.get("functional_connectivity", {}):
                    # Get correlation matrix
                    corr_matrix = features["functional_connectivity"][atlas]["correlation"]
                    labels = features["functional_connectivity"][atlas]["labels"]
                    
                    # Threshold matrix (keep only positive correlations)
                    threshold = 0.2  # Arbitrary threshold
                    thresholded_matrix = corr_matrix.copy()
                    thresholded_matrix[thresholded_matrix < threshold] = 0
                    
                    # Create graph
                    G = nx.from_numpy_array(thresholded_matrix)
                    
                    # Calculate metrics
                    metrics = {
                        "degree_centrality": nx.degree_centrality(G),
                        "betweenness_centrality": nx.betweenness_centrality(G),
                        "clustering_coefficient": nx.clustering(G),
                        "efficiency": nx.efficiency(G),
                        "node_labels": labels
                    }
                    
                    graph_metrics[atlas] = metrics
            
            features["graph_metrics"] = graph_metrics
            self.logger.info("Calculated graph theory metrics")
        except Exception as e:
            self.logger.warning(f"Error calculating graph metrics: {e}")
        
        # 4. Independent Component Analysis (ICA)
        try:
            from nilearn.decomposition import CanICA
            
            # Apply CanICA
            canica = CanICA(
                n_components=20,
                memory="nilearn_cache",
                memory_level=2,
                verbose=0,
                mask_strategy='template',
                random_state=42,
                standardize=True
            )
            
            canica.fit(func_img)
            components_img = canica.components_img_
            
            # Extract component time series
            component_masker = NiftiMapsMasker(
                components_img,
                standardize=True,
                memory='nilearn_cache'
            )
            component_time_series = component_masker.fit_transform(func_img)
            
            # Store ICA results
            features["ica"] = {
                "components_img": components_img,
                "component_time_series": component_time_series,
                "n_components": 20
            }
            
            self.logger.info("Performed Independent Component Analysis")
        except Exception as e:
            self.logger.warning(f"Error in ICA decomposition: {e}")
        
        return features
    
    def _analyze_data(self, preprocessed_data: Dict, features: Dict, parameters: Dict) -> Dict:
        """
        Analyze the preprocessed data and extracted features.
        
        Args:
            preprocessed_data: Dictionary with preprocessed data
            features: Dictionary of extracted features
            parameters: Processing parameters
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Analyzing fMRI data")
        
        analysis_results = {}
        
        # Determine analysis type based on available data
        task_based = "events" in preprocessed_data and preprocessed_data["events"] is not None
        
        # 1. Task-based analysis (GLM)
        if task_based and self.config["analysis"]["task_based"]["enabled"]:
            try:
                self.logger.info("Performing task-based GLM analysis")
                
                # Get required data
                func_img = preprocessed_data.get("preprocessed_functional", 
                                               preprocessed_data.get("functional"))
                events_df = preprocessed_data["events"]
                tr = preprocessed_data["tr"]
                
                # Set up first-level model
                from nilearn.glm.first_level import FirstLevelModel
                
                # Determine task conditions from events file
                conditions = events_df['trial_type'].unique() if 'trial_type' in events_df.columns else ['condition']
                
                # Extract events for each condition
                event_lists = {}
                for condition in conditions:
                    if 'trial_type' in events_df.columns:
                        condition_events = events_df[events_df['trial_type'] == condition]
                    else:
                        condition_events = events_df
                    
                    # Convert to required format (onset, duration, amplitude)
                    onsets = condition_events['onset'].values
                    durations = condition_events['duration'].values if 'duration' in condition_events.columns else np.ones_like(onsets)
                    
                    event_lists[condition] = np.column_stack((onsets, durations, np.ones_like(onsets)))
                
                # Set up first-level model
                model = FirstLevelModel(
                    t_r=tr,
                    noise_model='ar1',
                    standardize=True,
                    hrf_model='spm',
                    drift_model='cosine',
                    high_pass=0.01,
                    mask_img=preprocessed_data.get("brain_mask", None),
                    memory='nilearn_cache',
                    verbose=0
                )
                
                # Fit the model
                model.fit(func_img, events=event_lists)
                
                # Compute contrasts
                contrast_results = {}
                
                # Contrast for each condition vs. baseline
                for condition in conditions:
                    contrast_results[f"{condition}_vs_baseline"] = model.compute_contrast(
                        condition,
                        output_type='z_score'
                    )
                
                # If we have multiple conditions, compute between-condition contrasts
                if len(conditions) > 1:
                    for i, cond1 in enumerate(conditions):
                        for j, cond2 in enumerate(conditions):
                            if i < j:  # Avoid redundant contrasts
                                contrast_name = f"{cond1}_vs_{cond2}"
                                contrast_def = np.zeros(len(conditions))
                                contrast_def[i] = 1
                                contrast_def[j] = -1
                                
                                # Define contrast by assigning weights to conditions
                                explicit_contrast = {}
                                for k, cond in enumerate(conditions):
                                    explicit_contrast[cond] = contrast_def[k]
                                
                                contrast_results[contrast_name] = model.compute_contrast(
                                    explicit_contrast,
                                    output_type='z_score'
                                )
                
                # Store GLM results
                analysis_results["task_glm"] = {
                    "model": model,
                    "contrasts": contrast_results,
                    "conditions": list(conditions)
                }
                
                self.logger.info(f"Completed GLM analysis with {len(conditions)} conditions and {len(contrast_results)} contrasts")
            except Exception as e:
                self.logger.error(f"Error in task-based GLM analysis: {e}")
        
        # 2. Resting-state analysis
        if self.config["analysis"]["resting_state"]["enabled"]:
            try:
                self.logger.info("Performing resting-state analysis")
                
                # 2.1 Seed-based correlation analysis
                if "seed_based" in self.config["analysis"]["resting_state"]["methods"]:
                    # Define default seeds (key brain regions)
                    seeds = {
                        "PCC": [-6, -52, 28],  # Posterior Cingulate Cortex (DMN)
                        "mPFC": [0, 52, -6],   # Medial Prefrontal Cortex (DMN)
                        "rAI": [36, 16, 4],    # Right Anterior Insula (SN)
                        "dACC": [4, 16, 36],   # Dorsal Anterior Cingulate Cortex (SN)
                        "rDLPFC": [44, 36, 20] # Right Dorsolateral Prefrontal Cortex (CEN)
                    }
                    
                    # Get functional image
                    func_img = preprocessed_data.get("preprocessed_functional", 
                                                   preprocessed_data.get("functional"))
                    
                    # Calculate seed-based correlation maps
                    from nilearn.maskers import NiftiSpheresMasker
                    from nilearn.mass_univariate import permuted_ols
                    
                    seed_maps = {}
                    for seed_name, coordinates in seeds.items():
                        # Extract time series from seed region
                        seed_masker = NiftiSpheresMasker(
                            [coordinates],
                            radius=8,  # 8mm sphere
                            detrend=True,
                            standardize=True,
                            low_pass=0.1,
                            high_pass=0.01,
                            t_r=preprocessed_data["tr"],
                            memory='nilearn_cache',
                            verbose=0
                        )
                        
                        seed_time_series = seed_masker.fit_transform(func_img)
                        
                        # Calculate correlation map
                        from nilearn.masking import compute_brain_mask
                        from nilearn.image import clean_img
                        
                        # Create brain mask if not already available
                        if "brain_mask" not in preprocessed_data:
                            brain_mask = compute_brain_mask(func_img)
                        else:
                            brain_mask = preprocessed_data["brain_mask"]
                        
                        # Compute statistical map
                        from nilearn.glm import fmri_glm
                        
                        # Setup design matrix with seed time series
                        from nilearn.glm.first_level import make_first_level_design_matrix
                        import pandas as pd
                        
                        n_scans = func_img.shape[3]
                        frame_times = np.arange(n_scans) * preprocessed_data["tr"]
                        
                        # Create design matrix with seed time series as regressor
                        design_matrix = pd.DataFrame(
                            seed_time_series,
                            columns=['seed'],
                            index=frame_times
                        )
                        
                        # Add confounds if available
                        if "confounds" in preprocessed_data and preprocessed_data["confounds"] is not None:
                            confounds = preprocessed_data["confounds"]
                            # Select a subset of useful confounds
                            useful_confounds = [c for c in confounds.columns if 
                                               ('trans' in c or 'rot' in c or 'wm' in c or 'csf' in c)
                                               and 'derivative' not in c and 'power' not in c]
                            if useful_confounds:
                                confounds_data = confounds[useful_confounds].fillna(0)
                                for column in confounds_data.columns:
                                    design_matrix[column] = confounds_data[column].values
                        
                        # Add drift regressors
                        design_matrix = make_first_level_design_matrix(
                            frame_times,
                            add_regs=design_matrix.values,
                            add_reg_names=design_matrix.columns.tolist(),
                            high_pass=0.01,
                            drift_model='cosine'
                        )
                        
                        # Run GLM
                        fmri_glm = FirstLevelModel(
                            mask_img=brain_mask,
                            standardize=True,
                            noise_model='ar1',
                            t_r=preprocessed_data["tr"],
                            memory='nilearn_cache',
                            verbose=0
                        )
                        
                        fmri_glm.fit(func_img, design_matrices=design_matrix)
                        
                        # Compute contrast on the seed regressor
                        contrast = np.zeros(design_matrix.shape[1])
                        contrast[0] = 1  # The seed regressor is the first column
                        
                        seed_maps[seed_name] = fmri_glm.compute_contrast(
                            contrast,
                            output_type='z_score'
                        )
                    
                    # Store seed correlation results
                    analysis_results["seed_based_connectivity"] = {
                        "seed_coordinates": seeds,
                        "correlation_maps": seed_maps
                    }
                    
                    self.logger.info(f"Completed seed-based analysis for {len(seeds)} seed regions")
                
                # 2.2 ICA network analysis
                if "ica" in self.config["analysis"]["resting_state"]["methods"] and "ica" in features:
                    # Analyze ICA components
                    
                    # Correlate ICA components with known resting-state networks
                    # Here we define spatial templates for major networks
                    
                    # Load network templates
                    from nilearn.datasets import fetch_atlas_smith_2009
                    
                    # Smith 2009 atlas contains 10 canonical resting-state networks
                    rsn = fetch_atlas_smith_2009()
                    rsn_20 = rsn.maps  # 20 component version
                    
                    # Get our ICA components
                    components_img = features["ica"]["components_img"]
                    
                    # Calculate spatial correlation with templates
                    from nilearn.image import iter_img
                    from nilearn.maskers import NiftiMasker
                    
                    # Create masker
                    masker = NiftiMasker(mask_strategy='whole-brain-template')
                    masker.fit()
                    
                    # Unmask ICA components to get spatial maps
                    ica_components = list(iter_img(components_img))
                    ica_maps = [masker.transform(component) for component in ica_components]
                    
                    # Unmask RSN templates
                    rsn_maps = [masker.transform(template) for template in iter_img(rsn_20)]
                    
                    # Calculate spatial correlations
                    import numpy as np
                    correlation_matrix = np.zeros((len(ica_maps), len(rsn_maps)))
                    
                    for i, ica_map in enumerate(ica_maps):
                        for j, rsn_map in enumerate(rsn_maps):
                            correlation_matrix[i, j] = np.corrcoef(ica_map.ravel(), rsn_map.ravel())[0, 1]
                    
                    # Identify best-matching RSN for each ICA component
                    best_matches = {}
                    for i in range(len(ica_maps)):
                        best_match_idx = np.argmax(correlation_matrix[i, :])
                        correlation = correlation_matrix[i, best_match_idx]
                        
                        # Only consider strong correlations
                        if correlation > 0.3:
                            network_names = [
                                "Visual Primary",
                                "Visual Occipital",
                                "Visual Lateral",
                                "Default Mode",
                                "Cerebellum",
                                "Sensorimotor",
                                "Auditory",
                                "Executive Control",
                                "Frontoparietal Right",
                                "Frontoparietal Left"
                            ]
                            
                            # Using modulus to handle 20-component atlas
                            if best_match_idx < len(network_names):
                                network_name = network_names[best_match_idx]
                            else:
                                network_name = f"Network {best_match_idx}"
                            
                            best_matches[f"Component {i}"] = {
                                "network": network_name,
                                "correlation": correlation,
                                "rsn_index": int(best_match_idx)
                            }
                    
                    # Store ICA network analysis results
                    analysis_results["ica_network_analysis"] = {
                        "correlation_matrix": correlation_matrix,
                        "best_matches": best_matches
                    }
                    
                    self.logger.info(f"Identified {len(best_matches)} ICA components matching known resting-state networks")
            except Exception as e:
                self.logger.error(f"Error in resting-state analysis: {e}")
        
        # 3. Machine learning analysis
        if self.config["analysis"]["machine_learning"]["enabled"] and "functional_connectivity" in features:
            try:
                self.logger.info("Performing machine learning analysis on functional connectivity")
                
                # Use AAL atlas connectivity for this example
                if "aal" in features["functional_connectivity"]:
                    # Get connectivity matrix
                    connectivity_matrix = features["functional_connectivity"]["aal"]["correlation"]
                    
                    # Extract upper triangle (to avoid redundancy)
                    import numpy as np
                    mask = np.triu_indices_from(connectivity_matrix, k=1)
                    connectivity_features = connectivity_matrix[mask]
                    
                    # Calculate basic statistics of connectivity values
                    conn_stats = {
                        "mean": float(np.mean(connectivity_features)),
                        "std": float(np.std(connectivity_features)),
                        "min": float(np.min(connectivity_features)),
                        "max": float(np.max(connectivity_features)),
                        "median": float(np.median(connectivity_features)),
                        # Calculate percentile thresholds
                        "percentiles": {
                            "p25": float(np.percentile(connectivity_features, 25)),
                            "p50": float(np.percentile(connectivity_features, 50)),
                            "p75": float(np.percentile(connectivity_features, 75)),
                            "p90": float(np.percentile(connectivity_features, 90)),
                            "p95": float(np.percentile(connectivity_features, 95)),
                        }
                    }
                    
                    # Clustering analysis
                    from sklearn.cluster import KMeans
                    
                    # Prepare data (normalize features)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(connectivity_features.reshape(-1, 1)).ravel()
                    
                    # Try different numbers of clusters
                    k_values = [2, 3, 4, 5]
                    cluster_results = {}
                    
                    for k in k_values:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        cluster_labels = kmeans.fit_predict(scaled_features.reshape(-1, 1))
                        
                        # Calculate basic stats for each cluster
                        cluster_stats = []
                        for i in range(k):
                            cluster_values = connectivity_features[cluster_labels == i]
                            cluster_stats.append({
                                "size": int(len(cluster_values)),
                                "mean": float(np.mean(cluster_values)),
                                "std": float(np.std(cluster_values)),
                                "min": float(np.min(cluster_values)),
                                "max": float(np.max(cluster_values))
                            })
                        
                        cluster_results[f"k_{k}"] = {
                            "cluster_centers": kmeans.cluster_centers_.ravel().tolist(),
                            "cluster_stats": cluster_stats,
                            "inertia": float(kmeans.inertia_)
                        }
                    
                    # Store machine learning results
                    analysis_results["connectivity_ml_analysis"] = {
                        "connectivity_stats": conn_stats,
                        "clustering": cluster_results
                    }
                    
                    self.logger.info("Completed machine learning analysis on functional connectivity")
            except Exception as e:
                self.logger.error(f"Error in machine learning analysis: {e}")
        
        return analysis_results
    
    def _save_results(self, data_path: str, preprocessed_data: Dict, 
                     features: Dict, analysis_results: Dict) -> str:
        """
        Save processing results.
        
        Args:
            data_path: Original data path
            preprocessed_data: Preprocessed data dictionary
            features: Extracted features
            analysis_results: Analysis results
            
        Returns:
            Path to saved results
        """
        # Create base directory for results
        import os
        import json
        import numpy as np
        import nibabel as nib
        from datetime import datetime
        
        # Create a unique results directory
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.config["storage"]["results"], f"{base_name}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        self.logger.info(f"Saving results to {results_dir}")
        
        # Helper function to make data JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (datetime,)):
                return obj.isoformat()
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                return {key: make_serializable(value) for key, value in obj.__dict__.items()
                        if not key.startswith('_')}
            else:
                return str(obj)
        
        # 1. Save preprocessed NIfTI images
        images_dir = os.path.join(results_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Save key preprocessed images
        image_paths = {}
        for key, img in preprocessed_data.items():
            if isinstance(img, nib.Nifti1Image) or isinstance(img, nib.Nifti2Image):
                img_path = os.path.join(images_dir, f"{key}.nii.gz")
                nib.save(img, img_path)
                image_paths[key] = img_path
        
        # 2. Save feature data
        features_dir = os.path.join(results_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        
        # Extract serializable feature data
        serializable_features = {}
        
        # ROI time series
        if "roi_time_series" in features:
            roi_dir = os.path.join(features_dir, "roi_time_series")
            os.makedirs(roi_dir, exist_ok=True)
            
            for atlas, data in features["roi_time_series"].items():
                # Save time series as CSV
                import pandas as pd
                time_series = data["data"]
                labels = data["labels"]
                
                # Create DataFrame
                df = pd.DataFrame(time_series, columns=labels)
                csv_path = os.path.join(roi_dir, f"{atlas}_time_series.csv")
                df.to_csv(csv_path, index=False)
                
                # Store path in serializable features
                if "roi_time_series" not in serializable_features:
                    serializable_features["roi_time_series"] = {}
                
                serializable_features["roi_time_series"][atlas] = {
                    "path": csv_path,
                    "shape": time_series.shape,
                    "labels": labels
                }
        
        # Functional connectivity
        if "functional_connectivity" in features:
            conn_dir = os.path.join(features_dir, "connectivity")
            os.makedirs(conn_dir, exist_ok=True)
            
            for atlas, data in features["functional_connectivity"].items():
                # Save correlation matrix
                conn_matrix = data["correlation"]
                labels = data["labels"]
                
                # Save as CSV
                matrix_df = pd.DataFrame(conn_matrix, index=labels, columns=labels)
                csv_path = os.path.join(conn_dir, f"{atlas}_connectivity.csv")
                matrix_df.to_csv(csv_path)
                
                # Save as NumPy array
                npy_path = os.path.join(conn_dir, f"{atlas}_connectivity.npy")
                np.save(npy_path, conn_matrix)
                
                # Store path in serializable features
                if "functional_connectivity" not in serializable_features:
                    serializable_features["functional_connectivity"] = {}
                
                serializable_features["functional_connectivity"][atlas] = {
                    "csv_path": csv_path,
                    "npy_path": npy_path,
                    "shape": conn_matrix.shape,
                    "labels": labels
                }
        
        # ICA components
        if "ica" in features:
            ica_dir = os.path.join(features_dir, "ica")
            os.makedirs(ica_dir, exist_ok=True)
            
            # Save component maps
            comp_img = features["ica"]["components_img"]
            comp_path = os.path.join(ica_dir, "ica_components.nii.gz")
            nib.save(comp_img, comp_path)
            
            # Save component time series
            comp_ts = features["ica"]["component_time_series"]
            comp_ts_path = os.path.join(ica_dir, "component_time_series.npy")
            np.save(comp_ts_path, comp_ts)
            
            # Store in serializable features
            serializable_features["ica"] = {
                "components_path": comp_path,
                "time_series_path": comp_ts_path,
                "n_components": features["ica"]["n_components"]
            }
        
        # 3. Save analysis results
        analysis_dir = os.path.join(results_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Task GLM
        if "task_glm" in analysis_results:
            glm_dir = os.path.join(analysis_dir, "task_glm")
            os.makedirs(glm_dir, exist_ok=True)
            
            # Save contrast maps
            contrast_info = {}
            for contrast_name, contrast_map in analysis_results["task_glm"]["contrasts"].items():
                map_path = os.path.join(glm_dir, f"{contrast_name}_zmap.nii.gz")
                nib.save(contrast_map, map_path)
                contrast_info[contrast_name] = map_path
            
            # Create serializable GLM results
            glm_results = {
                "conditions": analysis_results["task_glm"]["conditions"],
                "contrast_maps": contrast_info
            }
            
            # Save GLM metadata
            glm_meta_path = os.path.join(glm_dir, "glm_metadata.json")
            with open(glm_meta_path, 'w') as f:
                json.dump(glm_results, f, indent=2)
        
        # Seed-based connectivity
        if "seed_based_connectivity" in analysis_results:
            seed_dir = os.path.join(analysis_dir, "seed_connectivity")
            os.makedirs(seed_dir, exist_ok=True)
            
            # Save connectivity maps
            seed_info = {
                "seed_coordinates": analysis_results["seed_based_connectivity"]["seed_coordinates"]
            }
            
            map_paths = {}
            for seed_name, conn_map in analysis_results["seed_based_connectivity"]["correlation_maps"].items():
                map_path = os.path.join(seed_dir, f"{seed_name}_connectivity.nii.gz")
                nib.save(conn_map, map_path)
                map_paths[seed_name] = map_path
            
            seed_info["map_paths"] = map_paths
            
            # Save seed metadata
            seed_meta_path = os.path.join(seed_dir, "seed_metadata.json")
            with open(seed_meta_path, 'w') as f:
                json.dump(seed_info, f, indent=2)
        
        # ICA network analysis
        if "ica_network_analysis" in analysis_results:
            ica_dir = os.path.join(analysis_dir, "ica_networks")
            os.makedirs(ica_dir, exist_ok=True)
            
            # Save correlation matrix
            corr_matrix = analysis_results["ica_network_analysis"]["correlation_matrix"]
            np.save(os.path.join(ica_dir, "rsn_correlation_matrix.npy"), corr_matrix)
            
            # Save best matches
            best_matches = analysis_results["ica_network_analysis"]["best_matches"]
            
            # Save as JSON
            matches_path = os.path.join(ica_dir, "network_matches.json")
            with open(matches_path, 'w') as f:
                json.dump(best_matches, f, indent=2)
        
        # Machine learning analysis
        if "connectivity_ml_analysis" in analysis_results:
            ml_dir = os.path.join(analysis_dir, "ml_analysis")
            os.makedirs(ml_dir, exist_ok=True)
            
            # Save as JSON
            ml_path = os.path.join(ml_dir, "connectivity_analysis.json")
            with open(ml_path, 'w') as f:
                json.dump(make_serializable(analysis_results["connectivity_ml_analysis"]), f, indent=2)
        
        # 4. Generate visualization HTML report
        try:
            html_path = os.path.join(results_dir, "report.html")
            
            # Create plots
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot connectivity matrix
            if "functional_connectivity" in features:
                for atlas, data in features["functional_connectivity"].items():
                    conn_matrix = data["correlation"]
                    labels = data["labels"]
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(conn_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.colorbar(label='Correlation')
                    plt.title(f'{atlas} Functional Connectivity Matrix')
                    plt.savefig(os.path.join(plots_dir, f"{atlas}_connectivity.png"), dpi=150, bbox_inches='tight')
                    plt.close()
            
            # Plot GLM contrast maps
            if "task_glm" in analysis_results:
                for contrast_name, contrast_map in analysis_results["task_glm"]["contrasts"].items():
                    # Create plot using nilearn
                    from nilearn import plotting
                    
                    # Plot glass brain view
                    glass_brain_path = os.path.join(plots_dir, f"{contrast_name}_glass_brain.png")
                    display = plotting.plot_glass_brain(
                        contrast_map,
                        colorbar=True,
                        threshold=3.1,
                        display_mode='ortho',
                        plot_abs=False,
                        title=contrast_name
                    )
                    display.savefig(glass_brain_path, dpi=150)
                    display.close()
                    
                    # Plot stat map with MNI template background
                    stat_map_path = os.path.join(plots_dir, f"{contrast_name}_stat_map.png")
                    display = plotting.plot_stat_map(
                        contrast_map,
                        colorbar=True,
                        threshold=3.1,
                        cut_coords=(0, 0, 0),
                        title=contrast_name
                    )
                    display.savefig(stat_map_path, dpi=150)
                    display.close()
            
            # Generate simple HTML report
            with open(html_path, 'w') as f:
                f.write(f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>fMRI Analysis Results: {base_name}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1, h2, h3 {{ color: #444; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                        .image-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                        .image-card {{ border: 1px solid #eee; padding: 10px; border-radius: 5px; }}
                        img {{ max-width: 100%; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>fMRI Analysis Results: {base_name}</h1>
                        <div class="section">
                            <h2>Analysis Information</h2>
                            <p>Data path: {data_path}</p>
                            <p>Processing date: {timestamp}</p>
                        </div>
                """)
                
                # Add preprocessing steps
                if "preprocessing_steps" in preprocessed_data:
                    f.write(f"""
                        <div class="section">
                            <h2>Preprocessing Steps</h2>
                            <ul>
                    """)
                    for step in preprocessed_data["preprocessing_steps"]:
                        f.write(f"<li>{step}</li>")
                    f.write("""
                            </ul>
                        </div>
                    """)
                
                # Add connectivity plots
                if os.path.exists(os.path.join(plots_dir, "harvard_oxford_connectivity.png")) or \
                os.path.exists(os.path.join(plots_dir, "aal_connectivity.png")):
                    f.write("""
                        <div class="section">
                            <h2>Functional Connectivity</h2>
                            <div class="image-container">
                    """)
                    
                    if os.path.exists(os.path.join(plots_dir, "harvard_oxford_connectivity.png")):
                        f.write(f"""
                            <div class="image-card">
                                <h3>Harvard-Oxford Atlas</h3>
                                <img src="plots/harvard_oxford_connectivity.png" alt="Harvard-Oxford Connectivity">
                            </div>
                        """)
                    
                    if os.path.exists(os.path.join(plots_dir, "aal_connectivity.png")):
                        f.write(f"""
                            <div class="image-card">
                                <h3>AAL Atlas</h3>
                                <img src="plots/aal_connectivity.png" alt="AAL Connectivity">
                            </div>
                        """)
                    
                    f.write("""
                            </div>
                        </div>
                    """)
                
                # Add task GLM plots
                if "task_glm" in analysis_results and os.path.exists(plots_dir):
                    f.write("""
                        <div class="section">
                            <h2>Task-based GLM Analysis</h2>
                            <div class="image-container">
                    """)
                    
                    # Find all contrast plots
                    contrast_plots = [f for f in os.listdir(plots_dir) if f.endswith('_glass_brain.png') or f.endswith('_stat_map.png')]
                    
                    # Group by contrast
                    import re
                    contrasts = set()
                    for plot in contrast_plots:
                        match = re.match(r'(.+?)_(glass_brain|stat_map)\.png', plot)
                        if match:
                            contrasts.add(match.group(1))
                    
                    # Add each contrast
                    for contrast in contrasts:
                        f.write(f"""
                            <div class="image-card">
                                <h3>Contrast: {contrast}</h3>
                        """)
                        
                        if os.path.exists(os.path.join(plots_dir, f"{contrast}_glass_brain.png")):
                            f.write(f"""
                                <h4>Glass Brain View</h4>
                                <img src="plots/{contrast}_glass_brain.png" alt="{contrast} Glass Brain">
                            """)
                        
                        if os.path.exists(os.path.join(plots_dir, f"{contrast}_stat_map.png")):
                            f.write(f"""
                                <h4>Statistical Map View</h4>
                                <img src="plots/{contrast}_stat_map.png" alt="{contrast} Stat Map">
                            """)
                        
                        f.write("""
                            </div>
                        """)
                    
                    f.write("""
                            </div>
                        </div>
                    """)
                
                f.write("""
                    </div>
                </body>
                </html>
                """)
        except Exception as e:
            self.logger.warning(f"Error generating HTML report: {e}")
        
        # 5. Save summary metadata
        summary = {
            "data_path": data_path,
            "results_directory": results_dir,
            "processing_timestamp": timestamp,
            "preprocessing_steps": preprocessed_data.get("preprocessing_steps", []),
            "features_extracted": list(features.keys()),
            "analyses_performed": list(analysis_results.keys()),
            "image_paths": image_paths
        }
        
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"All results saved to {results_dir}")
        return results_dir