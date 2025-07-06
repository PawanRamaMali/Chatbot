"""
Main chatbot class that orchestrates all components
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

import numpy as np

from ..config.settings import Settings
from .processor import DataProcessor
from .model import ChatbotModel
from ..utils.logger import get_logger
from ..utils.helpers import sanitize_input, validate_message_length

logger = get_logger(__name__)


class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_interaction(self, user_id: str, message: str, response: str, 
                       intent: str, confidence: float):
        """Add an interaction to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'id': str(uuid.uuid4())
        }
        
        self.conversations[user_id].append(interaction)
        
        # Maintain history limit
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
    
    def get_conversation(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return self.conversations.get(user_id, [])
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def get_all_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all conversations"""
        return self.conversations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        total_users = len(self.conversations)
        total_interactions = sum(len(conv) for conv in self.conversations.values())
        
        # Calculate average interactions per user
        avg_interactions = total_interactions / total_users if total_users > 0 else 0
        
        # Find most common intents
        intent_counts = {}
        for conv in self.conversations.values():
            for interaction in conv:
                intent = interaction.get('intent', 'unknown')
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        most_common_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'average_interactions_per_user': avg_interactions,
            'most_common_intents': most_common_intents,
            'unique_intents': len(intent_counts)
        }


class NeuralChatbot:
    """Main chatbot class that coordinates all components"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        
        # Initialize components
        self.processor = DataProcessor(self.settings)
        self.model = ChatbotModel(self.settings)
        self.conversation_manager = ConversationManager()
        
        # State variables
        self.is_initialized = False
        self.is_trained = False
        
        logger.info("Neural Chatbot initialized")
    
    def initialize(self):
        """Initialize the chatbot by loading or training the model"""
        try:
            # Try to load existing model and data
            self.load_trained_model()
            self.is_initialized = True
            logger.info("Chatbot initialized with existing model")
            
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load existing model: {e}")
            logger.info("Chatbot needs to be trained first")
            self.is_initialized = False
    
    def train(self, intents_file: Optional[str] = None, save_model: bool = True) -> Dict[str, Any]:
        """Train the chatbot model"""
        logger.info("Starting chatbot training...")
        
        # Load and preprocess data
        intents_data = self.processor.load_intents(intents_file)
        
        # Validate intents data
        if not self.processor.validate_intents_data(intents_data):
            raise ValueError("Invalid intents data")
        
        # Preprocess intents
        words, classes = self.processor.preprocess_intents(intents_data)
        
        # Create training data
        train_x, train_y = self.processor.create_training_data()
        
        # Split data for validation
        if self.settings.training.validation_split > 0:
            split_idx = int(len(train_x) * (1 - self.settings.training.validation_split))
            x_train, x_val = train_x[:split_idx], train_x[split_idx:]
            y_train, y_val = train_y[:split_idx], train_y[split_idx:]
            validation_data = (x_val, y_val)
        else:
            x_train, y_train = train_x, train_y
            validation_data = None
        
        # Create and train model
        self.model.create_model(len(words), len(classes))
        history = self.model.train(x_train, y_train, validation_data)
        
        # Save processed data and model
        if save_model:
            self.processor.save_processed_data()
            self.model.save_model()
        
        # Update state
        self.is_trained = True
        self.is_initialized = True
        
        # Return training results
        final_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history.get('val_accuracy', [0])[-1]
        
        results = {
            'final_accuracy': final_accuracy,
            'validation_accuracy': val_accuracy,
            'epochs_completed': len(history.history['accuracy']),
            'total_samples': len(train_x),
            'vocabulary_size': len(words),
            'num_classes': len(classes),
            'training_time': 'completed'
        }
        
        logger.info(f"Training completed with accuracy: {final_accuracy:.4f}")
        return results
    
    def load_trained_model(self):
        """Load a pre-trained model and processed data"""
        # Load processed data
        self.processor.load_processed_data()
        
        # Load intents data
        self.processor.load_intents()
        
        # Load model
        self.model.load_model()
        
        self.is_trained = True
        self.is_initialized = True
        
        logger.info("Loaded trained model and data")
    
    def predict_intent(self, message: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predict intent for a message"""
        if not self.is_initialized:
            raise ValueError("Chatbot not initialized. Call initialize() or train() first.")
        
        # Convert message to bag of words
        bag_of_words = self.processor.text_to_bag_of_words(message)
        
        # Get predictions
        predictions = self.model.predict(bag_of_words, return_probabilities=True)
        
        if predictions.ndim > 1:
            predictions = predictions[0]
        
        # Get all predictions above threshold
        results = []
        for i, probability in enumerate(predictions):
            if probability > self.settings.model.confidence_threshold:
                class_name = self.processor.classes[i]
                results.append((class_name, float(probability)))
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        if results:
            best_intent, best_confidence = results[0]
            return best_intent, best_confidence, results
        else:
            return 'unknown', 0.0, []
    
    def get_response(self, message: str, user_id: str = 'default') -> str:
        """Get chatbot response for a message"""
        try:
            # Validate and sanitize input
            if not validate_message_length(message, self.settings.security.max_message_length):
                return "Your message is too long. Please keep it under 500 characters."
            
            if self.settings.security.sanitize_input:
                message = sanitize_input(message, self.settings.security.blacklisted_words)
            
            if not message.strip():
                return "I didn't receive any message. Please try again."
            
            # Predict intent
            intent, confidence, all_predictions = self.predict_intent(message)
            
            # Get response based on intent
            if confidence < self.settings.model.confidence_threshold:
                response = self._get_fallback_response(message, all_predictions)
            else:
                response = self.processor.get_response_for_tag(intent)
            
            # Add to conversation history
            self.conversation_manager.add_interaction(
                user_id=user_id,
                message=message,
                response=response,
                intent=intent,
                confidence=confidence
            )
            
            logger.info(f"User {user_id}: '{message}' -> Intent: {intent} (confidence: {confidence:.3f})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm experiencing some technical difficulties. Please try again in a moment."
    
    def _get_fallback_response(self, message: str, predictions: List[Tuple[str, float]]) -> str:
        """Get fallback response when confidence is low"""
        fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "I didn't quite catch that. Can you try asking in a different way?",
            "I'm still learning! Could you be more specific?",
            "Hmm, I'm not sure about that. Can you try rephrasing your question?",
            "I want to help, but I need you to be more clear. Can you try again?"
        ]
        
        # If we have some predictions but they're all low confidence
        if predictions:
            best_intent, best_confidence = predictions[0]
            if best_confidence > 0.1:  # Very low but not zero
                fallback_responses.extend([
                    f"Did you mean something about {best_intent}? Please clarify.",
                    f"I think you might be asking about {best_intent}, but I'm not certain. Can you be more specific?"
                ])
        
        import random
        return random.choice(fallback_responses)
    
    def chat_interactive(self):
        """Start an interactive chat session"""
        if not self.is_initialized:
            print("❌ Chatbot not initialized. Please train the model first.")
            return
        
        print("🤖 Neural Chatbot is ready!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 50)
        
        user_id = str(uuid.uuid4())
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("🤖 Neural: Goodbye! Thanks for chatting with me! 👋")
                    break
                
                if not user_input:
                    continue
                
                response = self.get_response(user_input, user_id)
                print(f"🤖 Neural: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Neural: Goodbye! Thanks for chatting with me! 👋")
                break
            except Exception as e:
                logger.error(f"Error in interactive chat: {e}")
                print("🤖 Neural: Sorry, I encountered an error. Please try again.")
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return self.conversation_manager.get_conversation(user_id)
    
    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user"""
        self.conversation_manager.clear_conversation(user_id)
    
    def export_conversation_history(self, filepath: str, user_id: Optional[str] = None):
        """Export conversation history to file"""
        if user_id:
            data = {user_id: self.conversation_manager.get_conversation(user_id)}
        else:
            data = self.conversation_manager.get_all_conversations()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation history exported to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive chatbot statistics"""
        model_stats = self.model.get_model_summary() if self.model.model else {}
        data_stats = self.processor.get_statistics()
        conversation_stats = self.conversation_manager.get_statistics()
        
        return {
            'status': {
                'initialized': self.is_initialized,
                'trained': self.is_trained,
                'model_loaded': self.model.model is not None
            },
            'model': model_stats,
            'data': data_stats,
            'conversations': conversation_stats,
            'settings': {
                'confidence_threshold': self.settings.model.confidence_threshold,
                'max_message_length': self.settings.security.max_message_length,
                'vocabulary_size': len(self.processor.words) if self.processor.words else 0,
                'num_intents': len(self.processor.classes) if self.processor.classes else 0
            }
        }
    
    def evaluate_model(self, test_data: Optional[tuple] = None) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_initialized:
            raise ValueError("Chatbot not initialized")
        
        if test_data is None:
            # Use a portion of training data as test data
            intents_data = self.processor.load_intents()
            self.processor.preprocess_intents(intents_data)
            train_x, train_y = self.processor.create_training_data()
            
            # Use last 20% as test data
            test_split = int(len(train_x) * 0.8)
            test_x, test_y = train_x[test_split:], train_y[test_split:]
        else:
            test_x, test_y = test_data
        
        # Evaluate model
        results = self.model.evaluate(test_x, test_y)
        
        # Add classification metrics
        predictions = self.model.predict(test_x, return_probabilities=False)
        actual = np.argmax(test_y, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Generate classification report
        class_report = classification_report(
            actual, predictions,
            target_names=self.processor.classes,
            output_dict=True,
            zero_division=0
        )
        
        results['classification_report'] = class_report
        results['confusion_matrix'] = confusion_matrix(actual, predictions).tolist()
        
        return results
    
    def update_intents(self, new_intents: Dict[str, Any], retrain: bool = True):
        """Update intents and optionally retrain the model"""
        # Validate new intents
        if not self.processor.validate_intents_data(new_intents):
            raise ValueError("Invalid intents data")
        
        # Save new intents
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(new_intents, f, indent=2)
            temp_file = f.name
        
        try:
            if retrain:
                # Retrain with new intents
                results = self.train(temp_file)
                logger.info("Model retrained with updated intents")
                return results
            else:
                # Just update the processor's intents data
                self.processor.intents_data = new_intents
                logger.info("Intents updated (model not retrained)")
                return {'status': 'updated', 'retrained': False}
        finally:
            # Clean up temp file
            import os
            os.unlink(temp_file)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Check processor
            health['components']['processor'] = {
                'status': 'ok' if self.processor.words and self.processor.classes else 'not_loaded',
                'words_loaded': len(self.processor.words) if self.processor.words else 0,
                'classes_loaded': len(self.processor.classes) if self.processor.classes else 0
            }
            
            # Check model
            health['components']['model'] = {
                'status': 'ok' if self.model.model else 'not_loaded',
                'model_loaded': self.model.model is not None
            }
            
            # Check conversation manager
            health['components']['conversation_manager'] = {
                'status': 'ok',
                'active_users': len(self.conversation_manager.conversations)
            }
            
            # Overall status
            if not self.is_initialized:
                health['status'] = 'not_initialized'
            elif not self.is_trained:
                health['status'] = 'not_trained'
            
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health