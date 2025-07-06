"""
Neural network model for the chatbot
Handles model creation, training, prediction, and persistence
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)
from tensorflow.keras.regularizers import l2
import joblib

from ..config.settings import Settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatbotModel:
    """Neural network model for intent classification"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[Sequential] = None
        self.training_history: Optional[tf.keras.callbacks.History] = None
        
        # Configure TensorFlow
        self._configure_tensorflow()
    
    def _configure_tensorflow(self):
        """Configure TensorFlow settings for optimal performance"""
        try:
            # Set memory growth for GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
            
            # Set mixed precision for better performance
            # tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
        except Exception as e:
            logger.warning(f"Error configuring TensorFlow: {e}")
    
    def create_model(self, input_shape: int, num_classes: int) -> Sequential:
        """Create the neural network model architecture"""
        model = Sequential(name='neural_chatbot')
        
        # Input layer
        model.add(Input(shape=(input_shape,), name='input_layer'))
        
        # Hidden layers based on configuration
        for i, units in enumerate(self.settings.model.hidden_layers):
            model.add(Dense(
                units=units,
                activation=self.settings.model.activation,
                kernel_regularizer=l2(0.001),
                name=f'hidden_layer_{i+1}'
            ))
            
            # Batch normalization for training stability
            model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Dropout for regularization
            model.add(Dropout(
                rate=self.settings.model.dropout_rate,
                name=f'dropout_{i+1}'
            ))
        
        # Output layer
        model.add(Dense(
            units=num_classes,
            activation='softmax',
            name='output_layer'
        ))
        
        # Compile model
        optimizer = self._get_optimizer()
        
        model.compile(
            optimizer=optimizer,
            loss=self.settings.model.loss_function,
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Print model summary
        logger.info("Model architecture created:")
        model.summary(print_fn=logger.info)
        
        self.model = model
        return model
    
    def _get_optimizer(self):
        """Get the configured optimizer"""
        lr = self.settings.model.learning_rate
        
        if self.settings.model.optimizer.lower() == 'adam':
            return Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        elif self.settings.model.optimizer.lower() == 'sgd':
            return SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        elif self.settings.model.optimizer.lower() == 'rmsprop':
            return RMSprop(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer: {self.settings.model.optimizer}, using Adam")
            return Adam(learning_rate=lr)
    
    def _get_callbacks(self, model_filepath: str) -> List:
        """Get training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.settings.training.monitor_metric,
            patience=self.settings.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='max' if 'accuracy' in self.settings.training.monitor_metric else 'min'
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=self.settings.training.monitor_metric,
            factor=self.settings.training.reduce_lr_factor,
            patience=self.settings.training.reduce_lr_patience,
            min_lr=self.settings.training.min_learning_rate,
            verbose=1,
            mode='max' if 'accuracy' in self.settings.training.monitor_metric else 'min'
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if self.settings.training.save_best_only:
            checkpoint = ModelCheckpoint(
                filepath=model_filepath,
                monitor=self.settings.training.monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max' if 'accuracy' in self.settings.training.monitor_metric else 'min'
            )
            callbacks.append(checkpoint)
        
        # TensorBoard logging
        log_dir = "logs/tensorboard"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # CSV logging
        csv_logger = CSVLogger(
            filename='logs/training_log.csv',
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> tf.keras.callbacks.History:
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Ensure models directory exists
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # Create model filepath with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filepath = model_dir / f"best_model_{timestamp}.h5"
        
        # Get callbacks
        callbacks = self._get_callbacks(str(model_filepath))
        
        logger.info("Starting model training...")
        logger.info(f"Training samples: {len(train_x)}")
        logger.info(f"Features: {train_x.shape[1]}")
        logger.info(f"Classes: {train_y.shape[1]}")
        
        # Train the model
        self.training_history = self.model.fit(
            train_x, train_y,
            validation_data=validation_data,
            epochs=self.settings.model.epochs,
            batch_size=self.settings.model.batch_size,
            verbose=1,
            callbacks=callbacks,
            shuffle=True
        )
        
        # Log training completion
        final_train_acc = self.training_history.history['accuracy'][-1]
        final_val_acc = self.training_history.history.get('val_accuracy', [0])[-1]
        
        logger.info("Training completed!")
        logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        if validation_data is not None:
            logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        
        return self.training_history
    
    def predict(self, input_data: np.ndarray, return_probabilities: bool = True) -> np.ndarray:
        """Make predictions with the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train a model first.")
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        predictions = self.model.predict(input_data, verbose=0)
        
        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_class_with_confidence(self, input_data: np.ndarray) -> Tuple[int, float]:
        """Predict class and return confidence score"""
        probabilities = self.predict(input_data, return_probabilities=True)
        
        if probabilities.ndim > 1:
            probabilities = probabilities[0]
        
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            filepath = f"models/{self.settings.data.model_file}"
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            
            # Also save the model configuration
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                import json
                config = {
                    'model_config': self.model.get_config(),
                    'training_config': self.settings.to_dict(),
                }
                json.dump(config, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: Optional[str] = None):
        """Load a trained model"""
        if filepath is None:
            # Try multiple locations
            possible_paths = [
                f"models/{self.settings.data.model_file}",
                self.settings.data.model_file,
                f"data/{self.settings.data.model_file}",
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    filepath = path
                    break
            else:
                raise FileNotFoundError(f"Model file not found: {self.settings.data.model_file}")
        
        try:
            self.model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
            # Print model summary
            logger.info("Loaded model architecture:")
            self.model.summary(print_fn=logger.info)
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            raise
    
    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        logger.info("Evaluating model...")
        results = self.model.evaluate(test_x, test_y, verbose=0)
        
        # Create results dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(results[i])
        
        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "total_params": self.model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
            "layers": len(self.model.layers),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
        }
    
    def export_model(self, export_path: str, format: str = 'tensorflow'):
        """Export model in different formats"""
        if self.model is None:
            raise ValueError("No model to export")
        
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'tensorflow':
            # Export as TensorFlow SavedModel
            tf.saved_model.save(self.model, export_path)
            logger.info(f"Model exported as TensorFlow SavedModel to {export_path}")
            
        elif format.lower() == 'tflite':
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"Model exported as TensorFlow Lite to {export_path}")
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Get training history"""
        if self.training_history is None:
            return None
        
        return self.training_history.history