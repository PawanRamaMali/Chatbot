"""
Data processing module for Neural Chatbot
Handles text preprocessing, tokenization, and data preparation
"""

import json
import pickle
import random
import re
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import logging

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

from ..config.settings import Settings
from ..utils.logger import get_logger
from ..data.loader import DataLoader

logger = get_logger(__name__)


class DataProcessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoder = LabelEncoder()
        self.data_loader = DataLoader()  
        
        # Text processing settings
        self.ignore_chars = ['?', '!', '.', ',', ';', ':', '"', "'", '(', ')', '[', ']']
        self.stop_words = set()
        
        # Data storage
        self.words: List[str] = []
        self.classes: List[str] = []
        self.documents: List[Tuple[List[str], str]] = []
        self.intents_data: Dict[str, Any] = {}
        
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Download and initialize NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            downloads = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
            for download in downloads:
                try:
                    nltk.download(download, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download {download}: {e}")
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not load stopwords: {e}")
            self.stop_words = set()
    
    def load_intents(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced intents loading with DataLoader"""
        if filepath is None:
            possible_paths = [
                self.settings.data.intents_file,
                f"data/{self.settings.data.intents_file}",
                f"src/neural_chatbot/data/{self.settings.data.intents_file}",
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    filepath = path
                    break
            else:
                raise FileNotFoundError(f"Could not find intents file")
        
        # Use DataLoader instead of manual JSON loading
        self.intents_data = self.data_loader.load_intents(filepath)
        return self.intents_data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove ignore characters
        for char in self.ignore_chars:
            text = text.replace(char, '')
        
        # Remove digits if needed (optional)
        # text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """Tokenize and lemmatize text"""
        try:
            # Clean text first
            text = self.clean_text(text)
            
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords if requested
            if remove_stopwords and self.stop_words:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Lemmatize
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Filter out empty tokens
            return [token for token in lemmatized if token.strip()]
            
        except Exception as e:
            logger.warning(f"Error processing text '{text}': {e}")
            return []
    
    def augment_patterns(self, patterns: List[str]) -> List[str]:
        """Apply data augmentation to patterns"""
        augmented = patterns.copy()
        
        for pattern in patterns:
            # Add variations without punctuation
            clean_pattern = self.clean_text(pattern)
            if clean_pattern and clean_pattern not in augmented:
                augmented.append(clean_pattern)
            
            # Add question variations
            if not pattern.endswith('?') and any(word in pattern.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who']):
                question_pattern = pattern + '?'
                if question_pattern not in augmented:
                    augmented.append(question_pattern)
            
            # Add casual variations
            casual_replacements = {
                'hello': ['hi', 'hey'],
                'goodbye': ['bye', 'see ya'],
                'thank you': ['thanks', 'thx'],
                'you are': ["you're"],
                'i am': ["i'm"],
            }
            
            lower_pattern = pattern.lower()
            for formal, casual_list in casual_replacements.items():
                if formal in lower_pattern:
                    for casual in casual_list:
                        new_pattern = lower_pattern.replace(formal, casual)
                        if new_pattern not in [p.lower() for p in augmented]:
                            augmented.append(new_pattern)
        
        return augmented
    
    def preprocess_intents(self, intents_data: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str]]:
        """Preprocess intents data and create vocabulary"""
        if intents_data is None:
            intents_data = self.intents_data
        
        if not intents_data or 'intents' not in intents_data:
            raise ValueError("Invalid intents data format")
        
        # Reset collections
        self.words = []
        self.classes = []
        self.documents = []
        
        # Process each intent
        for intent in intents_data['intents']:
            tag = intent['tag']
            patterns = intent.get('patterns', [])
            
            # Augment patterns
            augmented_patterns = self.augment_patterns(patterns)
            
            # Add to classes if not present
            if tag not in self.classes:
                self.classes.append(tag)
            
            # Process each pattern
            for pattern in augmented_patterns:
                # Tokenize and lemmatize
                tokens = self.tokenize_and_lemmatize(pattern)
                
                if tokens:  # Only add non-empty token lists
                    self.words.extend(tokens)
                    self.documents.append((tokens, tag))
        
        # Remove duplicates and sort
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        
        # Apply vocabulary constraints
        if len(self.words) > self.settings.data.max_vocab_size:
            # Keep most frequent words (this is a simple approach)
            from collections import Counter
            word_freq = Counter(word for doc, _ in self.documents for word in doc)
            most_frequent = [word for word, count in word_freq.most_common(self.settings.data.max_vocab_size)
                           if count >= self.settings.data.min_word_frequency]
            self.words = sorted(most_frequent)
        
        logger.info(f"Processed {len(self.documents)} documents, {len(self.words)} unique words, {len(self.classes)} classes")
        return self.words, self.classes
    
    def create_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data arrays from processed documents"""
        if not self.documents or not self.words or not self.classes:
            raise ValueError("No processed data available. Run preprocess_intents() first.")
        
        training_data = []
        output_empty = [0] * len(self.classes)
        
        for document, tag in self.documents:
            # Create bag of words
            bag = [1 if word in document else 0 for word in self.words]
            
            # Create output row
            output_row = output_empty.copy()
            output_row[self.classes.index(tag)] = 1
            
            training_data.append([bag, output_row])
        
        # Shuffle training data
        random.shuffle(training_data)
        
        # Convert to numpy arrays
        training_data = np.array(training_data, dtype=object)
        train_x = np.array(list(training_data[:, 0]))
        train_y = np.array(list(training_data[:, 1]))
        
        logger.info(f"Created training data: {train_x.shape[0]} samples, {train_x.shape[1]} features")
        return train_x, train_y
    
    def text_to_bag_of_words(self, text: str) -> np.ndarray:
        """Convert text to bag of words representation"""
        if not self.words:
            raise ValueError("Vocabulary not initialized. Run preprocess_intents() first.")
        
        # Tokenize and lemmatize input text
        tokens = self.tokenize_and_lemmatize(text)
        
        # Create bag of words
        bag = [1 if word in tokens else 0 for word in self.words]
        
        return np.array(bag)
    
    def save_processed_data(self, words_file: Optional[str] = None, classes_file: Optional[str] = None):
        """Save processed words and classes to pickle files"""
        words_file = words_file or f"data/{self.settings.data.words_file}"
        classes_file = classes_file or f"data/{self.settings.data.classes_file}"
        
        # Ensure directory exists
        Path(words_file).parent.mkdir(parents=True, exist_ok=True)
        Path(classes_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(words_file, 'wb') as f:
                pickle.dump(self.words, f)
            
            with open(classes_file, 'wb') as f:
                pickle.dump(self.classes, f)
            
            logger.info(f"Saved processed data to {words_file} and {classes_file}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, words_file: Optional[str] = None, classes_file: Optional[str] = None):
        """Load processed words and classes from pickle files"""
        words_file = words_file or f"data/{self.settings.data.words_file}"
        classes_file = classes_file or f"data/{self.settings.data.classes_file}"
        
        try:
            with open(words_file, 'rb') as f:
                self.words = pickle.load(f)
            
            with open(classes_file, 'rb') as f:
                self.classes = pickle.load(f)
            
            logger.info(f"Loaded processed data from {words_file} and {classes_file}")
            
        except FileNotFoundError as e:
            logger.error(f"Processed data files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def get_response_for_tag(self, tag: str) -> str:
        """Get a random response for a given intent tag"""
        if not self.intents_data or 'intents' not in self.intents_data:
            return "I'm not sure how to respond to that."
        
        for intent in self.intents_data['intents']:
            if intent['tag'] == tag:
                responses = intent.get('responses', [])
                if responses:
                    return random.choice(responses)
        
        return "I'm not sure how to respond to that."
    
    def validate_intents_data(self, intents_data: Dict[str, Any]) -> bool:
        """Validate intents data structure"""
        try:
            if not isinstance(intents_data, dict):
                logger.error("Intents data must be a dictionary")
                return False
            
            if 'intents' not in intents_data:
                logger.error("Intents data must contain 'intents' key")
                return False
            
            intents = intents_data['intents']
            if not isinstance(intents, list):
                logger.error("'intents' must be a list")
                return False
            
            for i, intent in enumerate(intents):
                if not isinstance(intent, dict):
                    logger.error(f"Intent {i} must be a dictionary")
                    return False
                
                required_fields = ['tag', 'patterns', 'responses']
                for field in required_fields:
                    if field not in intent:
                        logger.error(f"Intent {i} missing required field: {field}")
                        return False
                
                if not isinstance(intent['patterns'], list) or not intent['patterns']:
                    logger.error(f"Intent {i} must have non-empty patterns list")
                    return False
                
                if not isinstance(intent['responses'], list) or not intent['responses']:
                    logger.error(f"Intent {i} must have non-empty responses list")
                    return False
            
            logger.info("Intents data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating intents data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        return {
            'total_words': len(self.words),
            'total_classes': len(self.classes),
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.words),
            'classes': self.classes,
            'average_tokens_per_document': np.mean([len(doc) for doc, _ in self.documents]) if self.documents else 0,
            'min_tokens_per_document': min([len(doc) for doc, _ in self.documents]) if self.documents else 0,
            'max_tokens_per_document': max([len(doc) for doc, _ in self.documents]) if self.documents else 0,
        }