# calcium_agent.py
import os
import json
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
import pika
from scipy import ndimage
import skimage.io
import skimage.restoration
import skimage.measure
import skimage.segmentation
from skimage.filters import threshold_otsu
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class CalciumProcessingConfig:
    """Configuration for calcium imaging processing."""
    # Preprocessing
    spatial_filter: str = "gaussian"  # gaussian, median, bilateral
    temporal_filter: str = "savgol"   # savgol, median, kalman
    motion_correction: str = "ecc"    # ecc, optical_flow, none
    background_removal: str = "percentile"  # percentile, rolling_ball, none
    
    # Spatial filter parameters
    spatial_filter_size: int = 3
    
    # Temporal filter parameters
    temporal_window: int = 7
    savgol_order: int = 3
    
    # Motion correction parameters
    max_shifts: Tuple[int, int] = (20, 20)
    
    # Background parameters
    background_percentile: float = 10.0
    rolling_ball_radius: int = 50
    
    # Cell detection
    cell_detection_method: str = "watershed"  # watershed, cnmf, suite2p
    min_cell_size: int = 30
    max_cell_size: int = 500
    cell_threshold: float = 1.5  # Threshold multiplier over background
    min_distance_between_cells: int = 10
    
    # Signal extraction
    roi_expansion: int = 2  # Pixels to expand ROI for neuropil
    neuropil_correction: bool = True
    baseline_percentile: float = 20.0
    
    # Event detection
    event_threshold_std: float = 2.5
    min_event_duration: int = 2  # frames
    
    # Storage
    save_interim: bool = True
    interim_dir: str = "./data/interim/calcium"
    processed_dir: str = "./data/processed/calcium"
    results_dir: str = "./data/results/calcium"


class CalciumProcessingAgent:
    """
    Specialized agent for processing calcium imaging data.
    
    This agent handles various calcium imaging data formats, performs preprocessing,
    cell detection, signal extraction, and event detection.
    """
    
    def __init__(self, config_path: str = "calcium_agent_config.json"):
        """
        Initialize the calcium processing agent.
        
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
        
        self.logger.info("Calcium Processing Agent initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger("CalciumAgent")
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
            default_config = CalciumProcessingConfig()
            return {
                "message_broker": {
                    "host": "localhost",
                    "port": 5672,
                    "username": "guest",
                    "password": "guest",
                    "queue": "calcium_processing"
                },
                "preprocessing": {
                    "spatial_filter": default_config.spatial_filter,
                    "temporal_filter": default_config.temporal_filter,
                    "motion_correction": default_config.motion_correction,
                    "background_removal": default_config.background_removal,
                    "spatial_filter_size": default_config.spatial_filter_size,
                    "temporal_window": default_config.temporal_window,
                    "savgol_order": default_config.savgol_order,
                    "max_shifts": default_config.max_shifts,
                    "background_percentile": default_config.background_percentile,
                    "rolling_ball_radius": default_config.rolling_ball_radius
                },
                "cell_detection": {
                    "method": default_config.cell_detection_method,
                    "min_cell_size": default_config.min_cell_size,
                    "max_cell_size": default_config.max_cell_size,
                    "threshold": default_config.cell_threshold,
                    "min_distance": default_config.min_distance_between_cells
                },
                "signal_extraction": {
                    "roi_expansion": default_config.roi_expansion,
                    "neuropil_correction": default_config.neuropil_correction,
                    "baseline_percentile": default_config.baseline_percentile
                },
                "event_detection": {
                    "threshold_std": default_config.event_threshold_std,
                    "min_duration": default_config.min_event_duration
                },
                "storage": {
                    "save_interim": default_config.save_interim,
                    "interim_dir": default_config.interim_dir,
                    "processed_dir": default_config.processed_dir,
                    "results_dir": default_config.results_dir
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
        self.logger.info("Calcium Processing Agent stopped")
    
    def process_data(self, data_path: str, parameters: Dict = None) -> Dict:
        """
        Process calcium imaging data.
        
        Args:
            data_path: Path to calcium imaging data file
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
            
            # Detect cells
            cells = self._detect_cells(preprocessed_data, parameters)
            
            # Extract signals
            signals = self._extract_signals(preprocessed_data, cells, parameters)
            
            # Detect events
            events = self._detect_events(signals, parameters)
            
            # Analyze data
            analysis_results = self._analyze_data(signals, events, parameters)
            
            # Save results
            results_path = self._save_results(
                data_path, 
                preprocessed_data, 
                cells, 
                signals, 
                events, 
                analysis_results
            )
            
            return {
                "status": "success",
                "data_path": data_path,
                "results_path": results_path,
                "cells_detected": len(cells),
                "events_detected": sum(len(e) for e in events.values())
            }
        except Exception as e:
            self.logger.error(f"Error processing data {data_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "data_path": data_path,
                "error": str(e)
            }
    
    def _load_data(self, data_path: str) -> np.ndarray:
        """
        Load calcium imaging data from file.
        
        Args:
            data_path: Path to calcium imaging data file
            
        Returns:
            Numpy array with dimensions [frames, height, width]
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Determine file type and use appropriate loader
        _, ext = os.path.splitext(data_path.lower())
        
        if ext in ['.tif', '.tiff']:
            # Load TIFF stack
            data = skimage.io.imread(data_path)
            
            # Ensure correct dimensions (frames, height, width)
            if data.ndim == 2:
                # Single frame
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[0] > data.shape[2]:
                # Likely [height, width, frames] format, transpose to [frames, height, width]
                if data.shape[2] < 10:  # Heuristic: if third dimension is small, it's likely RGB
                    pass  # Keep as is, it's probably a color image
                else:
                    data = np.transpose(data, (2, 0, 1))
            
        elif ext in ['.avi', '.mp4', '.mov']:
            # Load video file
            cap = cv2.VideoCapture(data_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale if colored
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frames.append(frame)
            
            cap.release()
            data = np.array(frames)
            
        elif ext == '.npy':
            # Load numpy array
            data = np.load(data_path)
            
            # Ensure correct dimensions
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        self.logger.info(f"Loaded data: {data.shape} (frames, height, width)")
        return data
    
    def _preprocess_data(self, data: np.ndarray, parameters: Dict = None) -> np.ndarray:
        """
        Preprocess calcium imaging data.
        
        Args:
            data: Raw calcium imaging data [frames, height, width]
            parameters: Processing parameters
            
        Returns:
            Preprocessed data [frames, height, width]
        """
        self.logger.info("Preprocessing data")
        
        # Extract preprocessing parameters
        params = parameters.get("preprocessing", {})
        config = self.config["preprocessing"]
        
        # Get parameters with defaults from config
        spatial_filter = params.get("spatial_filter", config["spatial_filter"])
        temporal_filter = params.get("temporal_filter", config["temporal_filter"])
        motion_correction = params.get("motion_correction", config["motion_correction"])
        background_removal = params.get("background_removal", config["background_removal"])
        
        # Make a copy of the data
        processed = data.copy().astype(np.float32)
        
        # 1. Apply motion correction if enabled
        if motion_correction != "none":
            processed = self._apply_motion_correction(
                processed,
                method=motion_correction,
                max_shifts=params.get("max_shifts", config["max_shifts"])
            )
        
        # 2. Apply spatial filtering
        if spatial_filter != "none":
            filter_size = params.get("spatial_filter_size", config["spatial_filter_size"])
            
            for i in range(processed.shape[0]):
                if spatial_filter == "gaussian":
                    processed[i] = ndimage.gaussian_filter(processed[i], sigma=filter_size/3)
                elif spatial_filter == "median":
                    processed[i] = ndimage.median_filter(processed[i], size=filter_size)
                elif spatial_filter == "bilateral":
                    # Bilateral filter preserves edges better
                    processed[i] = cv2.bilateralFilter(
                        processed[i].astype(np.float32), 
                        d=filter_size, 
                        sigmaColor=75, 
                        sigmaSpace=75
                    )
        
        # 3. Apply background removal
        if background_removal != "none":
            processed = self._remove_background(
                processed,
                method=background_removal,
                percentile=params.get("background_percentile", config["background_percentile"]),
                rolling_ball_radius=params.get("rolling_ball_radius", config["rolling_ball_radius"])
            )
        
        # 4. Apply temporal filtering
        if temporal_filter != "none":
            window = params.get("temporal_window", config["temporal_window"])
            
            if temporal_filter == "savgol":
                from scipy.signal import savgol_filter
                savgol_order = params.get("savgol_order", config["savgol_order"])
                # Apply Savitzky-Golay filter to each pixel's time series
                for y in range(processed.shape[1]):
                    for x in range(processed.shape[2]):
                        processed[:, y, x] = savgol_filter(
                            processed[:, y, x],
                            window_length=window,
                            polyorder=savgol_order
                        )
                        
            elif temporal_filter == "median":
                # Apply median filter to each pixel's time series
                for y in range(processed.shape[1]):
                    for x in range(processed.shape[2]):
                        processed[:, y, x] = ndimage.median_filter(
                            processed[:, y, x],
                            size=window
                        )
            
            elif temporal_filter == "kalman":
                # Simple Kalman filter implementation for time series
                processed = self._apply_kalman_filter(processed)
        
        self.logger.info("Preprocessing completed")
        return processed
    
    def _apply_motion_correction(self, data: np.ndarray, method: str, max_shifts: Tuple[int, int]) -> np.ndarray:
        """Apply motion correction to the data."""
        self.logger.info(f"Applying motion correction: {method}")
        
        if method == "none":
            return data
        
        corrected = np.zeros_like(data)
        template = data[0]  # Use first frame as template
        
        if method == "ecc":
            # Enhanced Correlation Coefficient alignment
            for i in range(data.shape[0]):
                if i == 0:
                    corrected[i] = data[i]
                    continue
                
                # Convert to 8-bit for ECC algorithm
                template_8bit = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frame_8bit = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Define warp matrix
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                
                # Define termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-8)
                
                # Run ECC algorithm
                try:
                    _, warp_matrix = cv2.findTransformECC(
                        template_8bit, 
                        frame_8bit, 
                        warp_matrix, 
                        cv2.MOTION_TRANSLATION, 
                        criteria
                    )
                    
                    # Apply warp to original frame
                    corrected[i] = cv2.warpAffine(
                        data[i], 
                        warp_matrix, 
                        (data.shape[2], data.shape[1]), 
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )
                except:
                    self.logger.warning(f"ECC alignment failed for frame {i}, using original frame")
                    corrected[i] = data[i]
        
        elif method == "optical_flow":
            # Optical flow based alignment
            for i in range(data.shape[0]):
                if i == 0:
                    corrected[i] = data[i]
                    continue
                
                # Convert to 8-bit for optical flow
                template_8bit = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frame_8bit = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    template_8bit, 
                    frame_8bit, 
                    None, 
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate median displacement
                dy = np.median(flow[:, :, 0])
                dx = np.median(flow[:, :, 1])
                
                # Limit shifts
                dy = max(min(dy, max_shifts[0]), -max_shifts[0])
                dx = max(min(dx, max_shifts[1]), -max_shifts[1])
                
                # Create transformation matrix
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # Apply transformation
                corrected[i] = cv2.warpAffine(data[i], M, (data.shape[2], data.shape[1]))
        
        return corrected
    
    def _remove_background(self, data: np.ndarray, method: str, percentile: float = 10.0, rolling_ball_radius: int = 50) -> np.ndarray:
        """Remove background from the data."""
        self.logger.info(f"Removing background: {method}")
        
        if method == "none":
            return data
        
        result = data.copy()
        
        if method == "percentile":
            # Calculate percentile value for each pixel over time
            background = np.percentile(data, percentile, axis=0)
            
            # Subtract background from each frame
            for i in range(data.shape[0]):
                result[i] = data[i] - background
                
        elif method == "rolling_ball":
            # Rolling ball algorithm (approximation using morphological operations)
            for i in range(data.shape[0]):
                # Create structural element
                selem = skimage.morphology.disk(rolling_ball_radius)
                
                # Apply morphological opening
                background = skimage.morphology.opening(data[i], selem)
                
                # Subtract background
                result[i] = data[i] - background
        
        # Ensure no negative values
        result[result < 0] = 0
        
        return result
    
    def _apply_kalman_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply Kalman filter to each pixel time series."""
        self.logger.info("Applying Kalman filter")
        
        # Simple Kalman filter parameters
        process_noise = 1e-5
        measurement_noise = 1e-3
        
        # Initialize result array
        result = np.zeros_like(data)
        
        # Apply filter to each pixel time series
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                # Get time series for this pixel
                measurements = data[:, y, x]
                
                # Initialize Kalman filter state
                x_hat = measurements[0]  # Initial state
                p = 1.0  # Initial uncertainty
                
                # Process time series
                filtered_values = np.zeros_like(measurements)
                filtered_values[0] = x_hat
                
                for i in range(1, len(measurements)):
                    # Prediction step
                    x_hat_minus = x_hat
                    p_minus = p + process_noise
                    
                    # Update step
                    k = p_minus / (p_minus + measurement_noise)
                    x_hat = x_hat_minus + k * (measurements[i] - x_hat_minus)
                    p = (1 - k) * p_minus
                    
                    filtered_values[i] = x_hat
                
                # Store filtered time series
                result[:, y, x] = filtered_values
        
        return result
    
    def _detect_cells(self, data: np.ndarray, parameters: Dict = None) -> List[Dict]:
        """
        Detect cells in preprocessed data.
        
        Args:
            data: Preprocessed calcium imaging data [frames, height, width]
            parameters: Processing parameters
            
        Returns:
            List of dictionaries with cell properties
        """
        self.logger.info("Detecting cells")
        
        # Extract cell detection parameters
        params = parameters.get("cell_detection", {})
        config = self.config["cell_detection"]
        
        # Get parameters with defaults from config
        method = params.get("method", config["method"])
        min_cell_size = params.get("min_cell_size", config["min_cell_size"])
        max_cell_size = params.get("max_cell_size", config["max_cell_size"])
        threshold_factor = params.get("threshold", config["threshold"])
        min_distance = params.get("min_distance", config["min_distance"])
        
        # Calculate standard deviation projection for cell detection
        std_projection = np.std(data, axis=0)
        
        # Detect cells based on method
        cells = []
        
        if method == "watershed":
            # Watershed-based cell detection
            
            # Apply Gaussian filter to smooth the image
            smoothed = ndimage.gaussian_filter(std_projection, sigma=1.0)
            
            # Calculate threshold
            threshold = threshold_otsu(smoothed) * threshold_factor
            
            # Create binary mask
            binary = smoothed > threshold
            
            # Remove small objects
            binary = skimage.morphology.remove_small_objects(binary, min_size=min_cell_size)
            
            # Apply distance transform
            distance = ndimage.distance_transform_edt(binary)
            
            # Find local maxima (cell centers)
            from skimage.feature import peak_local_max
            coordinates = peak_local_max(
                distance, 
                min_distance=min_distance,
                labels=binary
            )
            
            # Create markers for watershed
            markers = np.zeros_like(std_projection, dtype=np.int32)
            markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
            
            # Apply watershed
            segmented = skimage.segmentation.watershed(-smoothed, markers, mask=binary)
            
            # Extract properties for each cell
            properties = skimage.measure.regionprops(segmented)
            
            for i, prop in enumerate(properties):
                # Skip cells that are too small or too large
                if prop.area < min_cell_size or prop.area > max_cell_size:
                    continue
                
                # Create cell dictionary
                cell = {
                    "id": i,
                    "y": int(prop.centroid[0]),
                    "x": int(prop.centroid[1]),
                    "radius": int(np.sqrt(prop.area / np.pi)),
                    "mask": segmented == prop.label,
                    "intensity": np.mean(std_projection[prop.coords[:, 0], prop.coords[:, 1]])
                }
                
                cells.append(cell)
        
        elif method == "cnmf":
            # Constrained Non-negative Matrix Factorization
            # This is a placeholder - actual CNMF implementation would use CaImAn or similar
            self.logger.warning("CNMF method not fully implemented, using simple thresholding")
            
            # Simple threshold-based detection as fallback
            threshold = np.mean(std_projection) + threshold_factor * np.std(std_projection)
            binary = std_projection > threshold
            binary = skimage.morphology.remove_small_objects(binary, min_size=min_cell_size)
            
            # Find connected components
            labeled, num_cells = ndimage.label(binary)
            properties = skimage.measure.regionprops(labeled)
            
            for i, prop in enumerate(properties):
                if prop.area < min_cell_size or prop.area > max_cell_size:
                    continue
                
                cell = {
                    "id": i,
                    "y": int(prop.centroid[0]),
                    "x": int(prop.centroid[1]),
                    "radius": int(np.sqrt(prop.area / np.pi)),
                    "mask": labeled == prop.label,
                    "intensity": np.mean(std_projection[prop.coords[:, 0], prop.coords[:, 1]])
                }
                
                cells.append(cell)
        
        elif method == "suite2p":
            # Suite2p-like approach
            # This is a placeholder - actual Suite2p implementation would use the suite2p package
            self.logger.warning("Suite2p method not fully implemented, using simple local maxima")
            
            # Simple local maxima detection as fallback
            from skimage.feature import peak_local_max
            
            # Smooth the image
            smoothed = ndimage.gaussian_filter(std_projection, sigma=1.5)
            
            # Find local maxima
            coordinates = peak_local_max(
                smoothed, 
                min_distance=min_distance,
                threshold_abs=np.mean(smoothed) + threshold_factor * np.std(smoothed)
            )
            
            # Create circular ROIs around each local maximum
            for i, (y, x) in enumerate(coordinates):
                # Create circular mask
                y_grid, x_grid = np.ogrid[-y:std_projection.shape[0]-y, -x:std_projection.shape[1]-x]
                mask = x_grid*x_grid + y_grid*y_grid <= min_cell_size
                
                cell = {
                    "id": i,
                    "y": int(y),
                    "x": int(x),
                    "radius": int(np.sqrt(min_cell_size / np.pi)),
                    "mask": mask,
                    "intensity": np.mean(std_projection[mask])
                }
                
                cells.append(cell)
        
        self.logger.info(f"Detected {len(cells)} cells")
        return cells
    
    def _extract_signals(self, data: np.ndarray, cells: List[Dict], parameters: Dict = None) -> Dict:
        """
        Extract fluorescence signals from cells.
        
        Args:
            data: Preprocessed calcium imaging data [frames, height, width]
            cells: List of detected cells
            parameters: Processing parameters
            
        Returns:
            Dictionary with cell signals
        """
        self.logger.info("Extracting signals")
        
        # Extract signal extraction parameters
        params = parameters.get("signal_extraction", {})
        config = self.config["signal_extraction"]
        
        # Get parameters with defaults from config
        roi_expansion = params.get("roi_expansion", config["roi_expansion"])
        neuropil_correction = params.get("neuropil_correction", config["neuropil_correction"])
        baseline_percentile = params.get("baseline_percentile", config["baseline_percentile"])
        
        # Initialize signal arrays
        n_frames = data.shape[0]
        n_cells = len(cells)
        
        raw_signals = np.zeros((n_cells, n_frames))
        neuropil_signals = np.zeros((n_cells, n_frames))
        corrected_signals = np.zeros((n_cells, n_frames))
        df_f_signals = np.zeros((n_cells, n_frames))
        
        # Extract signals for each cell
        for i, cell in enumerate(cells):
            # Get cell mask
            cell_mask = cell["mask"]
            
            # Create expanded mask for neuropil
            y, x = cell["y"], cell["x"]
            radius = cell["radius"]
            expanded_radius = radius + roi_expansion
            
            y_grid, x_grid = np.ogrid[-y:data.shape[1]-y, -x:data.shape[2]-x]
            neuropil_mask = (x_grid*x_grid + y_grid*y_grid <= expanded_radius*expanded_radius) & ~cell_mask
            
            # Extract raw signal (mean fluorescence within cell mask)
            for f in range(n_frames):
                if np.any(cell_mask):
                    raw_signals[i, f] = np.mean(data[f][cell_mask])
                else:
                    raw_signals[i, f] = 0
                
                if np.any(neuropil_mask):
                    neuropil_signals[i, f] = np.mean(data[f][neuropil_mask])
                else:
                    neuropil_signals[i, f] = 0
            
            # Apply neuropil correction if enabled
            if neuropil_correction:
                # Simple subtraction with scaling factor
                r = 0.7  # Contamination ratio, typically 0.7
                corrected_signals[i] = raw_signals[i] - r * neuropil_signals[i]
            else:
                corrected_signals[i] = raw_signals[i]
            
            # Calculate baseline as percentile of signal
            baseline = np.percentile(corrected_signals[i], baseline_percentile)
            
            # Calculate ΔF/F
            df_f_signals[i] = (corrected_signals[i] - baseline) / baseline
            
            # Update cell with signal properties
            cells[i]["baseline"] = float(baseline)
            cells[i]["max_df_f"] = float(np.max(df_f_signals[i]))
        
        signals = {
            "raw": raw_signals,
            "neuropil": neuropil_signals,
            "corrected": corrected_signals,
            "df_f": df_f_signals,
            "cell_ids": [cell["id"] for cell in cells]
        }
        
        self.logger.info(f"Extracted signals for {n_cells} cells")
        return signals
    
    def _detect_events(self, signals: Dict, parameters: Dict = None) -> Dict:
        """
        Detect calcium events in the extracted signals.
        
        Args:
            signals: Dictionary with cell signals
            parameters: Processing parameters
            
        Returns:
            Dictionary with detected events for each cell
        """
        self.logger.info("Detecting events")
        
        # Extract event detection parameters
        params = parameters.get("event_detection", {})
        config = self.config["event_detection"]
        
        # Get parameters with defaults from config
        threshold_std = params.get("threshold_std", config["threshold_std"])
        min_duration = params.get("min_duration", config["min_duration"])
        
        # Get ΔF/F signals
        df_f = signals["df_f"]
        n_cells, n_frames = df_f.shape
        
        # Initialize events dictionary
        events = {}
        
        # Detect events for each cell
        for i in range(n_cells):
            cell_events = []
            signal = df_f[i]
            
            # Calculate threshold as mean + std * threshold_factor
            threshold = np.mean(signal) + np.std(signal) * threshold_std
            
            # Find regions above threshold
            above_threshold = signal > threshold
            
            # Label contiguous regions
            labeled, n_regions = ndimage.label(above_threshold)
            
            # Process each region
            for j in range(1, n_regions + 1):
                region = np.where(labeled == j)[0]
                
                # Skip if duration too short
                if len(region) < min_duration:
                    continue
                
                # Get event properties
                start_frame = int(region[0])
                end_frame = int(region[-1])
                peak_frame = int(start_frame + np.argmax(signal[region]))
                amplitude = float(np.max(signal[region]))
                duration = float(end_frame - start_frame + 1)
                
                # Add event
                event = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "peak_frame": peak_frame,
                    "amplitude": amplitude,
                    "duration": duration
                }
                
                cell_events.append(event)
            
            events[str(signals["cell_ids"][i])] = cell_events
        
        # Count total events
        total_events = sum(len(e) for e in events.values())
        self.logger.info(f"Detected {total_events} events across {n_cells} cells")
        
        return events
    
    def _analyze_data(self, signals: Dict, events: Dict, parameters: Dict = None) -> Dict:
        """
        Analyze calcium imaging data.
        
        Args:
            signals: Dictionary with cell signals
            events: Dictionary with detected events
            parameters: Processing parameters
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing data")
        
        # Get ΔF/F signals
        df_f = signals["df_f"]
        n_cells, n_frames = df_f.shape
        
        # Initialize results dictionary
        results = {
            "cell_statistics": {},
            "population_statistics": {},
            "temporal_statistics": {}
        }
        
        # Calculate cell statistics
        for i in range(n_cells):
            cell_id = str(signals["cell_ids"][i])
            signal = df_f[i]
            cell_events = events.get(cell_id, [])
            
            cell_stats = {
                "mean_df_f": float(np.mean(signal)),
                "max_df_f": float(np.max(signal)),
                "std_df_f": float(np.std(signal)),
                "event_count": len(cell_events),
                "event_frequency": len(cell_events) / (n_frames / 30),  # Assuming 30 Hz
                "mean_event_amplitude": float(np.mean([e["amplitude"] for e in cell_events])) if cell_events else 0,
                "mean_event_duration": float(np.mean([e["duration"] for e in cell_events])) if cell_events else 0
            }
            
            results["cell_statistics"][cell_id] = cell_stats
        
        # Calculate population statistics
        event_counts = [len(events.get(str(cid), [])) for cid in signals["cell_ids"]]
        active_cells = sum(count > 0 for count in event_counts)
        
        results["population_statistics"] = {
            "n_cells": n_cells,
            "active_cells": active_cells,
            "active_fraction": float(active_cells / n_cells) if n_cells > 0 else 0,
            "mean_events_per_cell": float(np.mean(event_counts)),
            "max_events_per_cell": float(np.max(event_counts)),
            "total_events": sum(event_counts)
        }
        
        # Calculate temporal statistics (activity over time)
        activity_profile = np.zeros(n_frames)
        for i in range(n_cells):
            # Binarize signal (1 when cell is active)
            cell_id = str(signals["cell_ids"][i])
            for event in events.get(cell_id, []):
                activity_profile[event["start_frame"]:event["end_frame"]+1] += 1
        
        # Normalize by number of cells
        activity_profile = activity_profile / n_cells if n_cells > 0 else activity_profile
        
        # Find peaks in population activity
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(activity_profile, height=0.1, distance=10)
        
        results["temporal_statistics"] = {
            "mean_activity": float(np.mean(activity_profile)),
            "peak_activity": float(np.max(activity_profile)),
            "activity_profile": activity_profile.tolist(),
            "n_population_events": len(peaks),
            "population_event_frames": peaks.tolist()
        }
        
        self.logger.info("Analysis completed")
        return results
    
    def _save_results(self, data_path: str, preprocessed_data: np.ndarray, 
                     cells: List[Dict], signals: Dict, events: Dict, 
                     analysis_results: Dict) -> str:
        """
        Save processing results.
        
        Args:
            data_path: Original data path
            preprocessed_data: Preprocessed data
            cells: Detected cells
            signals: Extracted signals
            events: Detected events
            analysis_results: Analysis results
            
        Returns:
            Path to saved results
        """
        # Create base filename from original data
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        
        # Ensure results directory exists
        results_dir = self.config["storage"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        
        # Create dataset-specific directory
        dataset_dir = os.path.join(results_dir, base_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save cell masks
        cell_masks = np.zeros((preprocessed_data.shape[1], preprocessed_data.shape[2]), dtype=np.int32)
        for cell in cells:
            cell_masks[cell["mask"]] = cell["id"] + 1
        
        np.save(os.path.join(dataset_dir, "cell_masks.npy"), cell_masks)
        
        # Save summary image
        std_projection = np.std(preprocessed_data, axis=0)
        max_projection = np.max(preprocessed_data, axis=0)
        
        np.save(os.path.join(dataset_dir, "std_projection.npy"), std_projection)
        np.save(os.path.join(dataset_dir, "max_projection.npy"), max_projection)
        
        # Save signals
        np.save(os.path.join(dataset_dir, "raw_signals.npy"), signals["raw"])
        np.save(os.path.join(dataset_dir, "df_f_signals.npy"), signals["df_f"])
        
        # Save events
        with open(os.path.join(dataset_dir, "events.json"), 'w') as f:
            json.dump(events, f, indent=2)
        
        # Save analysis results
        with open(os.path.join(dataset_dir, "analysis.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Save a summary with cell locations and basic stats
        cell_summary = []
        for i, cell in enumerate(cells):
            cell_id = str(cell["id"])
            stats = analysis_results["cell_statistics"].get(cell_id, {})
            
            summary = {
                "id": cell["id"],
                "x": cell["x"],
                "y": cell["y"],
                "radius": cell["radius"],
                "max_df_f": stats.get("max_df_f", 0),
                "event_count": stats.get("event_count", 0),
                "event_frequency": stats.get("event_frequency", 0)
            }
            
            cell_summary.append(summary)
        
        with open(os.path.join(dataset_dir, "cell_summary.json"), 'w') as f:
            json.dump(cell_summary, f, indent=2)
        
        # Create visualization figures
        self._create_visualizations(dataset_dir, preprocessed_data, cells, signals, events, analysis_results)
        
        self.logger.info(f"Saved results to {dataset_dir}")
        return dataset_dir
    
    def _create_visualizations(self, output_dir: str, data: np.ndarray, 
                              cells: List[Dict], signals: Dict, events: Dict, 
                              analysis_results: Dict):
        """Create visualization figures for results."""
        # 1. Cell map visualization
        plt.figure(figsize=(10, 8))
        
        # Show std projection
        std_projection = np.std(data, axis=0)
        plt.imshow(std_projection, cmap='gray')
        
        # Plot cell locations
        for cell in cells:
            y, x = cell["y"], cell["x"]
            radius = cell["radius"]
            
            circle = plt.Circle((x, y), radius, fill=False, edgecolor='r', linewidth=1)
            plt.gca().add_patch(circle)
            plt.text(x, y, str(cell["id"]), color='white', fontsize=8, 
                    ha='center', va='center')
        
        plt.title(f"Detected Cells (n={len(cells)})")
        plt.colorbar(label="Standard Deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cell_map.png"), dpi=150)
        plt.close()
        
        # 2. Example traces
        n_examples = min(10, len(cells))
        plt.figure(figsize=(12, 8))
        
        for i in range(n_examples):
            cell_id = signals["cell_ids"][i]
            trace = signals["df_f"][i]
            
            plt.subplot(n_examples, 1, i+1)
            plt.plot(trace)
            
            # Mark events
            for event in events.get(str(cell_id), []):
                start = event["start_frame"]
                end = event["end_frame"]
                plt.axvspan(start, end, color='r', alpha=0.3)
            
            plt.ylabel(f"Cell {cell_id}")
            if i == 0:
                plt.title("ΔF/F Traces with Detected Events")
            if i == n_examples - 1:
                plt.xlabel("Frame")
            else:
                plt.xticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "example_traces.png"), dpi=150)
        plt.close()
        
        # 3. Population activity
        if "temporal_statistics" in analysis_results and "activity_profile" in analysis_results["temporal_statistics"]:
            plt.figure(figsize=(12, 4))
            
            activity = analysis_results["temporal_statistics"]["activity_profile"]
            plt.plot(activity)
            
            # Mark population events
            for frame in analysis_results["temporal_statistics"].get("population_event_frames", []):
                plt.axvline(frame, color='r', alpha=0.5, linestyle='--')
            
            plt.xlabel("Frame")
            plt.ylabel("Active Fraction")
            plt.title("Population Activity Over Time")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "population_activity.png"), dpi=150)
            plt.close()
    
    # External analysis methods for test_calcium_agent.py
    def analyze_activity(self, signals):
        """
        Analyze calcium activity from extracted signals.
        
        Args:
            signals: Dictionary with cell signals from the preprocessing pipeline
            
        Returns:
            String with analysis and interpretation
        """
        if not signals or "df_f" not in signals:
            return "No valid signals provided for analysis"
        
        df_f = signals["df_f"]
        n_cells, n_frames = df_f.shape
        
        # Calculate basic statistics
        active_cells = sum(np.max(df_f, axis=1) > 0.5)
        max_activity = np.max(df_f)
        mean_activity = np.mean(df_f)
        
        # Calculate temporal correlation
        corr_matrix = np.corrcoef(df_f)
        mean_correlation = np.sum(np.triu(corr_matrix, k=1)) / (n_cells * (n_cells - 1) / 2) if n_cells > 1 else 0
        
        analysis = f"Calcium Activity Analysis:\n"
        analysis += f"- {n_cells} cells analyzed over {n_frames} frames\n"
        analysis += f"- {active_cells} cells ({active_cells/n_cells*100:.1f}%) showed significant activity (ΔF/F > 0.5)\n"
        analysis += f"- Maximum ΔF/F: {max_activity:.2f}\n"
        analysis += f"- Mean ΔF/F: {mean_activity:.2f}\n"
        analysis += f"- Mean pairwise correlation: {mean_correlation:.2f}\n"
        
        # Add interpretation
        if active_cells / n_cells > 0.5:
            analysis += "\nInterpretation: High network activity with many active cells."
        elif active_cells / n_cells > 0.2:
            analysis += "\nInterpretation: Moderate network activity with some silent cells."
        else:
            analysis += "\nInterpretation: Low network activity with mostly silent cells."
        
        if mean_correlation > 0.5:
            analysis += "\nCells show strong synchronization, suggesting coordinated network activity."
        elif mean_correlation > 0.2:
            analysis += "\nCells show moderate synchronization with some independent activity."
        else:
            analysis += "\nCells show weak synchronization, suggesting mostly independent activity."
        
        return analysis
    
    def analyze_cell_types(self, signals, events):
        """
        Analyze cell types based on activity patterns.
        
        Args:
            signals: Dictionary with cell signals
            events: Dictionary with detected events
            
        Returns:
            String with cell type analysis
        """
        if not signals or "df_f" not in signals or not events:
            return "No valid signals or events provided for analysis"
        
        df_f = signals["df_f"]
        n_cells = df_f.shape[0]
        
        # Calculate features for classification
        event_frequency = []
        event_amplitude = []
        event_duration = []
        
        for i, cell_id in enumerate(signals["cell_ids"]):
            cell_events = events.get(str(cell_id), [])
            
            # Event frequency (events per minute, assuming 30 Hz)
            freq = len(cell_events) / (df_f.shape[1] / 30 / 60)
            event_frequency.append(freq)
            
            # Mean event amplitude
            amp = np.mean([e["amplitude"] for e in cell_events]) if cell_events else 0
            event_amplitude.append(amp)
            
            # Mean event duration
            dur = np.mean([e["duration"] for e in cell_events]) if cell_events else 0
            event_duration.append(dur)
        
        # Simple classification based on features
        cell_types = []
        for i in range(n_cells):
            if event_frequency[i] < 0.1:
                cell_type = "Silent"
            elif event_amplitude[i] > 1.0 and event_frequency[i] > 1.0:
                cell_type = "Highly Active"
            elif event_duration[i] > 20:  # Long events
                cell_type = "Plateau/Sustained"
            else:
                cell_type = "Regular"
            
            cell_types.append(cell_type)
        
        # Count cell types
        type_counts = {}
        for t in cell_types:
            if t in type_counts:
                type_counts[t] += 1
            else:
                type_counts[t] = 1
        
        analysis = f"Cell Type Analysis:\n"
        for cell_type, count in type_counts.items():
            analysis += f"- {cell_type}: {count} cells ({count/n_cells*100:.1f}%)\n"
        
        # Add interpretation
        analysis += "\nInterpretation:\n"
        if "Silent" in type_counts and type_counts["Silent"] / n_cells > 0.5:
            analysis += "- Network dominated by silent cells, suggesting low overall activity.\n"
        
        if "Highly Active" in type_counts and type_counts["Highly Active"] > 0:
            analysis += f"- Found {type_counts.get('Highly Active', 0)} highly active cells that may be driving network activity.\n"
        
        if "Plateau/Sustained" in type_counts and type_counts["Plateau/Sustained"] > 0:
            analysis += f"- Found {type_counts.get('Plateau/Sustained', 0)} cells with sustained activity patterns.\n"
        
        return analysis


# Example usage
if __name__ == "__main__":
    agent = CalciumProcessingAgent()
    
    # Process a sample file
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Processing file: {file_path}")
        result = agent.process_data(file_path)
        print(f"Processing result: {result}")
    else:
        # Start listening for messages
        print("Starting Calcium agent to listen for messages...")
        agent.start()
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping agent...")
            agent.stop()