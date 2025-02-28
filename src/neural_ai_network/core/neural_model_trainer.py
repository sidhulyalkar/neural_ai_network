# neural_model_trainer.py
import os
import json
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Any, Tuple, Optional
import glob

class NeuralDataModelTrainer:
    """
    Trains AI models for neural data analysis.
    
    This module can train various types of models for different neural data modalities,
    including classification, regression, and representation learning models.
    """
    
    def __init__(self, config_path: str = "model_trainer_config.json"):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Ensure model directory exists
        os.makedirs(self.config["storage"]["models"], exist_ok=True)
        
        # Initialize TensorFlow
        self._setup_tensorflow()
        
        self.logger.info("Neural Data Model Trainer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("ModelTrainer")
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
                "training": {
                    "batch_size": 32,
                    "epochs": 100,
                    "early_stopping": {
                        "enabled": True,
                        "patience": 10,
                        "min_delta": 0.001
                    },
                    "validation_split": 0.2,
                    "test_split": 0.1
                },
                "models": {
                    "eeg": {
                        "cnn_lstm": {
                            "filters": [16, 32, 64],
                            "kernel_sizes": [3, 3, 3],
                            "lstm_units": 128,
                            "dense_units": [64, 32]
                        },
                        "transformer": {
                            "num_layers": 4,
                            "d_model": 128,
                            "num_heads": 8,
                            "dff": 512,
                            "dropout_rate": 0.1
                        }
                    },
                    "fmri": {
                        "3d_cnn": {
                            "filters": [16, 32, 64],
                            "kernel_sizes": [3, 3, 3],
                            "dense_units": [128, 64]
                        }
                    },
                    "calcium_imaging": {
                        "unet": {
                            "filters": [16, 32, 64, 128],
                            "kernel_size": 3
                        }
                    }
                },
                "storage": {
                    "data": "./data/processed",
                    "models": "./models",
                    "logs": "./logs"
                }
            }
    
    def _setup_tensorflow(self):
        """Set up TensorFlow environment."""
        # Try to use GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Using {len(gpus)} GPU(s) with memory growth enabled")
            except RuntimeError as e:
                self.logger.warning(f"Error setting GPU memory growth: {e}")
        else:
            self.logger.info("No GPUs found, using CPU")
    
    def train_model(self, modality: str, model_type: str, data_path: str, parameters: Dict = None) -> Dict:
        """
        Train a model for the specified neural data modality.
        
        Args:
            modality: Neural data modality (e.g., 'eeg', 'fmri')
            model_type: Type of model to train (e.g., 'cnn_lstm', '3d_cnn')
            data_path: Path to processed data directory or specific file
            parameters: Optional training parameters to override defaults
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Training {model_type} model for {modality} modality using data from {data_path}")
        
        # Merge configuration with provided parameters
        training_config = self.config["training"].copy()
        if parameters:
            for key, value in parameters.items():
                if key in training_config:
                    training_config[key] = value
        
        # Get model configuration
        model_config = self.config["models"].get(modality, {}).get(model_type, {})
        if not model_config:
            raise ValueError(f"No configuration found for {modality}/{model_type}")
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = self._prepare_data(
            modality, data_path, training_config
        )
        
        # Build and train model
        model, history = self._build_and_train_model(
            modality, model_type, model_config, training_config,
            X_train, y_train, X_val, y_val, metadata
        )
        
        # Evaluate model
        evaluation = self._evaluate_model(model, X_test, y_test, metadata)
        
        # Save model
        model_path = self._save_model(model, modality, model_type, metadata)
        
        # Prepare and return results
        results = {
            "model_path": model_path,
            "training": {
                "epochs_completed": len(history.history["loss"]),
                "final_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1])
            },
            "evaluation": evaluation,
            "metadata": metadata
        }
        
        # Add accuracy metrics if available
        if "accuracy" in history.history:
            results["training"]["final_accuracy"] = float(history.history["accuracy"][-1])
            results["training"]["final_val_accuracy"] = float(history.history["val_accuracy"][-1])
        
        return results
    
    def _prepare_data(self, modality: str, data_path: str, 
                     training_config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load and prepare data for model training.
        
        Args:
            modality: Neural data modality
            data_path: Path to data
            training_config: Training configuration
            
        Returns:
            Training, validation, and test data splits, plus metadata
        """
        self.logger.info(f"Preparing {modality} data from {data_path}")
        
        # Handle different modalities
        if modality == "eeg":
            return self._prepare_eeg_data(data_path, training_config)
        elif modality == "fmri":
            return self._prepare_fmri_data(data_path, training_config)
        elif modality == "calcium_imaging":
            return self._prepare_calcium_data(data_path, training_config)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _prepare_eeg_data(self, data_path: str, training_config: Dict) -> Tuple[np.ndarray, np.ndarray, 
                                                                             np.ndarray, np.ndarray, 
                                                                             np.ndarray, np.ndarray, Dict]:
        """
        Prepare EEG data for model training.
        
        Args:
            data_path: Path to EEG data
            training_config: Training configuration
            
        Returns:
            Training, validation, and test data splits, plus metadata
        """
        # This is a simplified implementation. In a real system, you would need
        # more sophisticated data loading and preprocessing.
        
        # Check if data_path is a directory or a specific file
        if os.path.isdir(data_path):
            # Find all result files in the directory
            files = glob.glob(os.path.join(data_path, "*_results.json"))
            if not files:
                raise ValueError(f"No result files found in {data_path}")
        else:
            # Single file
            files = [data_path]
        
        # Load all files
        all_features = []
        all_labels = []
        metadata = {"channels": None, "classes": None, "files": []}
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract features (assuming band powers are available)
                if "features" in data and "band_power" in data["features"]:
                    band_powers = data["features"]["band_power"]["band_powers"]
                    channels = data["features"]["band_power"]["channel_names"]
                    
                    # Combine band powers into features
                    features = []
                    for band, powers in band_powers.items():
                        features.extend(powers)
                    
                    # For demonstration, we'll create random labels
                    # In a real system, you would extract labels from the data
                    labels = np.random.randint(0, 2)  # Binary classification for demo
                    
                    all_features.append(features)
                    all_labels.append(labels)
                    
                    # Update metadata
                    if metadata["channels"] is None:
                        metadata["channels"] = channels
                    
                    metadata["files"].append(os.path.basename(file_path))
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")
        
        if not all_features:
            raise ValueError("No valid features found in the data")
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Determine unique classes
        metadata["classes"] = np.unique(y).tolist()
        metadata["num_features"] = X.shape[1]
        metadata["num_classes"] = len(metadata["classes"])
        
        # Split into training, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=training_config["test_split"], random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=training_config["validation_split"] / (1 - training_config["test_split"]),
            random_state=42
        )
        
        self.logger.info(f"Prepared data: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata
    
    def _prepare_fmri_data(self, data_path: str, training_config: Dict) -> Tuple[np.ndarray, np.ndarray, 
                                                                              np.ndarray, np.ndarray, 
                                                                              np.ndarray, np.ndarray, Dict]:
        """
        Prepare fMRI data for model training.
        
        This is a placeholder implementation. In a real system, you would need
        to handle actual fMRI data formats and preprocessing.
        """
        # Create dummy data for demonstration
        X = np.random.rand(100, 64, 64, 32, 1)  # 100 samples of 64x64x32 3D volumes, 1 channel
        y = np.random.randint(0, 2, size=(100,))  # Binary labels
        
        metadata = {
            "shape": X.shape[1:],
            "classes": np.unique(y).tolist(),
            "num_classes": len(np.unique(y)),
            "files": [os.path.basename(data_path)]
        }
        
        # Split into training, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=training_config["test_split"], random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=training_config["validation_split"] / (1 - training_config["test_split"]),
            random_state=42
        )
        
        self.logger.info(f"Prepared data: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata
    
    def _prepare_calcium_data(self, data_path: str, training_config: Dict) -> Tuple[np.ndarray, np.ndarray, 
                                                                                np.ndarray, np.ndarray, 
                                                                                np.ndarray, np.ndarray, Dict]:
        """
        Prepare calcium imaging data for model training.
        
        This is a placeholder implementation. In a real system, you would need
        to handle actual calcium imaging data formats and preprocessing.
        """
        # Create dummy data for demonstration
        X = np.random.rand(100, 512, 512, 1)  # 100 samples of 512x512 images, 1 channel
        y = np.random.rand(100, 512, 512, 1)  # Segmentation masks
        
        metadata = {
            "shape": X.shape[1:],
            "task": "segmentation",
            "files": [os.path.basename(data_path)]
        }
        
        # Split into training, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=training_config["test_split"], random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=training_config["validation_split"] / (1 - training_config["test_split"]),
            random_state=42
        )
        
        self.logger.info(f"Prepared data: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata
    
    def _build_and_train_model(self, modality: str, model_type: str, model_config: Dict, 
                              training_config: Dict, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray, 
                              metadata: Dict) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Build and train a neural network model.
        
        Args:
            modality: Neural data modality
            model_type: Type of model to build
            model_config: Model configuration
            training_config: Training configuration
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metadata: Data metadata
            
        Returns:
            Trained model and training history
        """
        self.logger.info(f"Building {model_type} model for {modality}")
        
        # Build model based on modality and type
        if modality == "eeg":
            model = self._build_eeg_model(model_type, model_config, metadata)
        elif modality == "fmri":
            model = self._build_fmri_model(model_type, model_config, metadata)
        elif modality == "calcium_imaging":
            model = self._build_calcium_model(model_type, model_config, metadata)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._get_loss_function(modality, metadata),
            metrics=self._get_metrics(modality, metadata)
        )
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        if training_config.get("early_stopping", {}).get("enabled", False):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=training_config["early_stopping"]["patience"],
                min_delta=training_config["early_stopping"]["min_delta"],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config["storage"]["models"],
            f"{modality}_{model_type}_checkpoint.h5"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)
        
        # TensorBoard logging
        log_dir = os.path.join(
            self.config["storage"]["logs"],
            f"{modality}_{model_type}_{int(time.time())}"
        )
        os.makedirs(log_dir, exist_ok=True)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard)
        
        # Train model
        self.logger.info(f"Training model with {X_train.shape[0]} samples for up to {training_config['epochs']} epochs")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info(f"Model training completed after {len(history.history['loss'])} epochs")
        return model, history
    
    def _build_eeg_model(self, model_type: str, model_config: Dict, metadata: Dict) -> tf.keras.Model:
        """
        Build an EEG model based on the specified type.
        
        Args:
            model_type: Type of model to build
            model_config: Model configuration
            metadata: Data metadata
            
        Returns:
            TensorFlow model
        """
        if model_type == "cnn_lstm":
            # Reshape input for CNN-LSTM (assuming time series data)
            input_shape = (metadata["num_features"], 1)  # Reshape features as time series
            
            # Input layer
            inputs = tf.keras.layers.Input(shape=input_shape)
            
            # CNN layers
            x = inputs
            for i, (filters, kernel_size) in enumerate(zip(
                model_config["filters"],
                model_config["kernel_sizes"]
            )):
                x = tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv_{i+1}'
                )(x)
                x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x)
            
            # LSTM layer
            x = tf.keras.layers.LSTM(
                units=model_config["lstm_units"],
                return_sequences=False,
                name='lstm'
            )(x)
            
            # Dense layers
            for i, units in enumerate(model_config["dense_units"]):
                x = tf.keras.layers.Dense(
                    units=units,
                    activation='relu',
                    name=f'dense_{i+1}'
                )(x)
                x = tf.keras.layers.Dropout(0.5, name=f'dropout_{i+1}')(x)
            
            # Output layer
            if metadata["num_classes"] == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = tf.keras.layers.Dense(metadata["num_classes"], activation='softmax', name='output')(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        elif model_type == "transformer":
            # Implement transformer model for EEG
            # This is a simplified implementation
            input_shape = (metadata["num_features"], 1)
            
            inputs = tf.keras.layers.Input(shape=input_shape)
            x = inputs
            
            # Transformer layers
            for i in range(model_config["num_layers"]):
                x = self._transformer_encoder_layer(
                    x,
                    d_model=model_config["d_model"],
                    num_heads=model_config["num_heads"],
                    dff=model_config["dff"],
                    dropout_rate=model_config["dropout_rate"],
                    name=f'transformer_{i+1}'
                )
            
            # Global pooling and output
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            for i, units in enumerate([64, 32]):
                x = tf.keras.layers.Dense(units, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.1)(x)
            
            if metadata["num_classes"] == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = tf.keras.layers.Dense(metadata["num_classes"], activation='softmax')(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        else:
            raise ValueError(f"Unsupported EEG model type: {model_type}")
    
    def _transformer_encoder_layer(self, inputs, d_model, num_heads, dff, dropout_rate, name):
        """Helper function to create a Transformer encoder layer."""
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name}_attention"
        )(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed-forward network
        outputs = tf.keras.layers.Dense(dff, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(d_model)(outputs)
        outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        
        return outputs
    
    def _build_fmri_model(self, model_type: str, model_config: Dict, metadata: Dict) -> tf.keras.Model:
        """
        Build an fMRI model based on the specified type.
        
        This is a placeholder implementation for a 3D CNN model.
        """
        if model_type == "3d_cnn":
            input_shape = metadata["shape"]
            
            inputs = tf.keras.layers.Input(shape=input_shape)
            x = inputs
            
            # 3D CNN layers
            for i, (filters, kernel_size) in enumerate(zip(
                model_config["filters"],
                model_config.get("kernel_sizes", [3, 3, 3])
            )):
                x = tf.keras.layers.Conv3D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same'
                )(x)
                x = tf.keras.layers.MaxPooling3D(pool_size=2)(x)
            
            # Flatten and dense layers
            x = tf.keras.layers.Flatten()(x)
            
            for units in model_config["dense_units"]:
                x = tf.keras.layers.Dense(units, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.5)(x)
            
            # Output layer
            if metadata["num_classes"] == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = tf.keras.layers.Dense(metadata["num_classes"], activation='softmax')(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        else:
            raise ValueError(f"Unsupported fMRI model type: {model_type}")
    
    def _build_calcium_model(self, model_type: str, model_config: Dict, metadata: Dict) -> tf.keras.Model:
        """
        Build a calcium imaging model based on the specified type.
        
        This is a placeholder implementation for a U-Net model.
        """
        if model_type == "unet":
            input_shape = metadata["shape"]
            
            # This is a simplified U-Net implementation
            inputs = tf.keras.layers.Input(shape=input_shape)
            
            # Encoder
            encoder_outputs = []
            x = inputs
            
            for i, filters in enumerate(model_config["filters"]):
                x = tf.keras.layers.Conv2D(filters, model_config["kernel_size"], activation='relu', padding='same')(x)
                x = tf.keras.layers.Conv2D(filters, model_config["kernel_size"], activation='relu', padding='same')(x)
                encoder_outputs.append(x)
                if i < len(model_config["filters"]) - 1:
                    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            
            # Decoder
            for i in range(len(model_config["filters"]) - 2, -1, -1):
                filters = model_config["filters"][i]
                x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
                x = tf.keras.layers.concatenate([x, encoder_outputs[i]])
                x = tf.keras.layers.Conv2D(filters, model_config["kernel_size"], activation='relu', padding='same')(x)
                x = tf.keras.layers.Conv2D(filters, model_config["kernel_size"], activation='relu', padding='same')(x)
            
            # Output layer
            outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        else:
            raise ValueError(f"Unsupported calcium imaging model type: {model_type}")
    
    def _get_loss_function(self, modality: str, metadata: Dict) -> tf.keras.losses.Loss:
        """
        Get appropriate loss function based on modality and task.
        
        Args:
            modality: Neural data modality
            metadata: Data metadata
            
        Returns:
            TensorFlow loss function
        """
        if modality == "calcium_imaging" and metadata.get("task") == "segmentation":
            # Dice loss or binary cross-entropy for segmentation
            return tf.keras.losses.BinaryCrossentropy()
        elif modality in ["eeg", "fmri"]:
            # Classification task
            if metadata.get("num_classes") == 2:
                return tf.keras.losses.BinaryCrossentropy()
            else:
                return tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            # Default to MSE for regression
            return tf.keras.losses.MeanSquaredError()
    
    def _get_metrics(self, modality: str, metadata: Dict) -> List:
        """
        Get appropriate metrics based on modality and task.
        
        Args:
            modality: Neural data modality
            metadata: Data metadata
            
        Returns:
            List of TensorFlow metrics
        """
        if modality == "calcium_imaging" and metadata.get("task") == "segmentation":
            # IoU and accuracy for segmentation
            return ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
        elif modality in ["eeg", "fmri"]:
            # Classification metrics
            if metadata.get("num_classes") == 2:
                return ['accuracy', tf.keras.metrics.AUC()]
            else:
                return ['accuracy']
        else:
            # Regression metrics
            return [tf.keras.metrics.MeanAbsoluteError()]
    
    def _evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, 
                       y_test: np.ndarray, metadata: Dict) -> Dict:
        """
        Evaluate trained model on test data.
        
        Args:
            model: Trained TensorFlow model
            X_test: Test features
            y_test: Test labels
            metadata: Data metadata
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating model on {X_test.shape[0]} test samples")
        
        # Get model predictions
        y_pred = model.predict(X_test)
        
        # Convert probabilities to class labels for classification
        if metadata.get("num_classes", 0) > 0:
            if metadata.get("num_classes") == 2:
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
            
            # Calculate classification metrics
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred_class)),
                "precision": float(precision_score(y_test, y_pred_class, average='macro')),
                "recall": float(recall_score(y_test, y_pred_class, average='macro')),
                "f1_score": float(f1_score(y_test, y_pred_class, average='macro'))
            }
        else:
            # Regression metrics
            metrics = {
                "mse": float(np.mean((y_test - y_pred) ** 2)),
                "mae": float(np.mean(np.abs(y_test - y_pred)))
            }
        
        return metrics
    
    def _save_model(self, model: tf.keras.Model, modality: str, 
                   model_type: str, metadata: Dict) -> str:
        """
        Save trained model and metadata.
        
        Args:
            model: Trained TensorFlow model
            modality: Neural data modality
            model_type: Type of model
            metadata: Data metadata
            
        Returns:
            Path to saved model
        """
        # Create model directory
        model_dir = os.path.join(
            self.config["storage"]["models"],
            f"{modality}_{model_type}_{int(time.time())}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.h5")
        model.save(model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "modality": modality,
                "model_type": model_type,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": metadata
            }, f, indent=2)
        
        self.logger.info(f"Saved model to {model_path}")
        return model_path


# Example usage
if __name__ == "__main__":
    import time
    trainer = NeuralDataModelTrainer()
    
    # Example: Train an EEG model using processed data
    result = trainer.train_model(
        modality="eeg",
        model_type="cnn_lstm",
        data_path="./data/processed/eeg",
        parameters={
            "epochs": 10,  # Reduced for demonstration
            "batch_size": 16
        }
    )
    
    print(f"Training result: {json.dumps(result, indent=2)}")