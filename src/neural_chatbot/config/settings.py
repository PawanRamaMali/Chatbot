"""
Configuration settings for Neural Chatbot
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration settings"""
    learning_rate: float = 0.001
    epochs: int = 200
    batch_size: int = 8
    dropout_rate: float = 0.5
    confidence_threshold: float = 0.25
    hidden_layers: list = field(default_factory=lambda: [128, 64, 32])
    activation: str = 'relu'
    optimizer: str = 'adam'
    loss_function: str = 'categorical_crossentropy'


@dataclass
class DataConfig:
    """Data configuration settings"""
    intents_file: str = 'intents.json'
    words_file: str = 'words.pkl'
    classes_file: str = 'classes.pkl'
    model_file: str = 'chatbot_model.h5'
    max_response_length: int = 500
    min_word_frequency: int = 1
    max_vocab_size: int = 10000


@dataclass
class TrainingConfig:
    """Training configuration settings"""
    validation_split: float = 0.1
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.2
    min_learning_rate: float = 0.0001
    save_best_only: bool = True
    monitor_metric: str = 'val_accuracy'


@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    secret_key: str = 'change-me-in-production'
    cors_enabled: bool = True
    rate_limit_per_minute: int = 60
    max_content_length: int = 1024 * 1024  # 1MB


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file: Optional[str] = 'logs/chatbot.log'
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_message_length: int = 500
    session_timeout_hours: int = 24
    csrf_enabled: bool = True
    sanitize_input: bool = True
    blacklisted_words: list = field(default_factory=list)


class Settings:
    """Main settings class that loads and manages all configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        
        # Initialize with defaults
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load from file and environment
        self._load_from_file()
        self._load_from_env()
        
        # Validate settings
        self._validate()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations"""
        possible_paths = [
            'config.yaml',
            'config/config.yaml',
            'src/neural_chatbot/config/config.yaml',
            os.path.expanduser('~/.neural_chatbot/config.yaml'),
            '/etc/neural_chatbot/config.yaml',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        if not self.config_file or not os.path.exists(self.config_file):
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return
            
            # Update configurations
            if 'model' in config_data:
                self._update_dataclass(self.model, config_data['model'])
            if 'data' in config_data:
                self._update_dataclass(self.data, config_data['data'])
            if 'training' in config_data:
                self._update_dataclass(self.training, config_data['training'])
            if 'api' in config_data:
                self._update_dataclass(self.api, config_data['api'])
            if 'logging' in config_data:
                self._update_dataclass(self.logging, config_data['logging'])
            if 'security' in config_data:
                self._update_dataclass(self.security, config_data['security'])
                
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Model config
            'NEURAL_CHATBOT_LEARNING_RATE': ('model', 'learning_rate', float),
            'NEURAL_CHATBOT_EPOCHS': ('model', 'epochs', int),
            'NEURAL_CHATBOT_BATCH_SIZE': ('model', 'batch_size', int),
            'NEURAL_CHATBOT_DROPOUT_RATE': ('model', 'dropout_rate', float),
            'NEURAL_CHATBOT_CONFIDENCE_THRESHOLD': ('model', 'confidence_threshold', float),
            
            # Data config
            'NEURAL_CHATBOT_INTENTS_FILE': ('data', 'intents_file', str),
            'NEURAL_CHATBOT_MODEL_FILE': ('data', 'model_file', str),
            
            # API config
            'NEURAL_CHATBOT_HOST': ('api', 'host', str),
            'NEURAL_CHATBOT_PORT': ('api', 'port', int),
            'NEURAL_CHATBOT_DEBUG': ('api', 'debug', bool),
            'NEURAL_CHATBOT_SECRET_KEY': ('api', 'secret_key', str),
            
            # Logging config
            'NEURAL_CHATBOT_LOG_LEVEL': ('logging', 'level', str),
            'NEURAL_CHATBOT_LOG_FILE': ('logging', 'file', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if type_func == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = type_func(value)
                    
                    config_obj = getattr(self, section)
                    setattr(config_obj, key, value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _validate(self):
        """Validate configuration settings"""
        # Validate model config
        if self.model.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.model.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.model.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= self.model.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if not 0 <= self.model.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Validate training config
        if not 0 <= self.training.validation_split <= 1:
            raise ValueError("Validation split must be between 0 and 1")
        
        # Validate API config
        if not 1 <= self.api.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            'models',
            'logs',
            'data',
            os.path.dirname(self.logging.file) if self.logging.file else 'logs',
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__,
            'security': self.security.__dict__,
        }
    
    def save_to_file(self, filepath: str):
        """Save current settings to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create Settings from dictionary"""
        settings = cls()
        
        if 'model' in data:
            settings._update_dataclass(settings.model, data['model'])
        if 'data' in data:
            settings._update_dataclass(settings.data, data['data'])
        if 'training' in data:
            settings._update_dataclass(settings.training, data['training'])
        if 'api' in data:
            settings._update_dataclass(settings.api, data['api'])
        if 'logging' in data:
            settings._update_dataclass(settings.logging, data['logging'])
        if 'security' in data:
            settings._update_dataclass(settings.security, data['security'])
        
        settings._validate()
        return settings