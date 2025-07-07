"""
Data loading utilities for Neural Chatbot
"""

import json
import csv
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

# Fixed import - use absolute import or create local logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading various data formats"""
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON from {filepath}")
            return data
        except FileNotFoundError:
            logger.error(f"JSON file not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Union[str, Path], 
                  indent: int = 2, ensure_ascii: bool = False) -> None:
        """Save data to JSON file"""
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logger.info(f"Successfully saved JSON to {filepath}")
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load data from YAML file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.info(f"Successfully loaded YAML from {filepath}")
            return data or {}
        except FileNotFoundError:
            logger.error(f"YAML file not found: {filepath}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading YAML from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save data to YAML file"""
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            logger.info(f"Successfully saved YAML to {filepath}")
        except Exception as e:
            logger.error(f"Error saving YAML to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Any:
        """Load data from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Successfully loaded pickle from {filepath}")
            return data
        except FileNotFoundError:
            logger.error(f"Pickle file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading pickle from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
        """Save data to pickle file"""
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Successfully saved pickle to {filepath}")
        except Exception as e:
            logger.error(f"Error saving pickle to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_csv(filepath: Union[str, Path], 
                 delimiter: str = ',', 
                 has_header: bool = True) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        try:
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                if has_header:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    data = list(reader)
                else:
                    reader = csv.reader(f, delimiter=delimiter)
                    data = list(reader)
            
            logger.info(f"Successfully loaded CSV from {filepath} ({len(data)} rows)")
            return data
        except FileNotFoundError:
            logger.error(f"CSV file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], filepath: Union[str, Path],
                 delimiter: str = ',') -> None:
        """Save data to CSV file"""
        try:
            if not data:
                logger.warning(f"No data to save to {filepath}")
                return
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = data[0].keys() if isinstance(data[0], dict) else None
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if fieldnames:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(data)
                else:
                    writer = csv.writer(f, delimiter=delimiter)
                    writer.writerows(data)
            
            logger.info(f"Successfully saved CSV to {filepath} ({len(data)} rows)")
        except Exception as e:
            logger.error(f"Error saving CSV to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_text(filepath: Union[str, Path]) -> str:
        """Load text from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Successfully loaded text from {filepath} ({len(text)} chars)")
            return text
        except FileNotFoundError:
            logger.error(f"Text file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading text from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_text(text: str, filepath: Union[str, Path]) -> None:
        """Save text to file"""
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Successfully saved text to {filepath} ({len(text)} chars)")
        except Exception as e:
            logger.error(f"Error saving text to {filepath}: {e}")
            raise
    
    @classmethod
    def load_intents(cls, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load intents data with validation"""
        try:
            intents_data = cls.load_json(filepath)
            
            # Basic validation
            if 'intents' not in intents_data:
                raise ValueError("Invalid intents format: missing 'intents' key")
            
            if not isinstance(intents_data['intents'], list):
                raise ValueError("Invalid intents format: 'intents' must be a list")
            
            # Validate each intent
            for i, intent in enumerate(intents_data['intents']):
                required_fields = ['tag', 'patterns', 'responses']
                for field in required_fields:
                    if field not in intent:
                        raise ValueError(f"Intent {i} missing required field: {field}")
                
                if not intent['patterns']:
                    raise ValueError(f"Intent {i} has empty patterns")
                
                if not intent['responses']:
                    raise ValueError(f"Intent {i} has empty responses")
            
            logger.info(f"Successfully validated {len(intents_data['intents'])} intents")
            return intents_data
            
        except Exception as e:
            logger.error(f"Error loading intents from {filepath}: {e}")
            raise
    
    @classmethod
    def backup_file(cls, filepath: Union[str, Path], 
                    backup_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create a backup of a file"""
        try:
            original_path = Path(filepath)
            if not original_path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Determine backup directory
            if backup_dir is None:
                backup_dir = original_path.parent / 'backups'
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{original_path.stem}_backup_{timestamp}{original_path.suffix}"
            backup_path = backup_dir / backup_filename
            
            # Copy file
            import shutil
            shutil.copy2(original_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup of {filepath}: {e}")
            raise
    
    @classmethod
    def find_data_files(cls, directory: Union[str, Path], 
                       pattern: str = "*.json") -> List[Path]:
        """Find data files matching a pattern in directory"""
        try:
            directory = Path(directory)
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                return []
            
            files = list(directory.glob(pattern))
            logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
            return files
            
        except Exception as e:
            logger.error(f"Error finding files in {directory}: {e}")
            return []
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"


# Convenience functions
def load_intents(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load intents"""
    return DataLoader.load_intents(filepath)


def save_intents(intents_data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Convenience function to save intents"""
    DataLoader.save_json(intents_data, filepath)


def backup_intents(filepath: Union[str, Path]) -> Path:
    """Convenience function to backup intents file"""
    return DataLoader.backup_file(filepath)