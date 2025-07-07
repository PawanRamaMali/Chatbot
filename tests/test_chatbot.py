# tests/test_chatbot.py

import pytest
import json
import tempfile
from pathlib import Path

from neural_chatbot.core.chatbot import NeuralChatbot
from neural_chatbot.config.settings import Settings


@pytest.fixture
def sample_intents():
    """Sample intents data for testing"""
    return {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["hi", "hello", "hey"],
                "responses": ["Hello!", "Hi there!"]
            },
            {
                "tag": "goodbye",
                "patterns": ["bye", "goodbye", "see you"],
                "responses": ["Goodbye!", "See you later!"]
            }
        ]
    }


@pytest.fixture
def temp_intents_file(sample_intents):
    """Create temporary intents file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_intents, f)
        return f.name


@pytest.fixture
def chatbot_settings():
    """Test settings for chatbot"""
    settings = Settings()
    settings.model.epochs = 5  # Faster training for tests
    settings.model.batch_size = 4
    return settings


@pytest.fixture
def chatbot(chatbot_settings):
    """Create chatbot instance for testing"""
    return NeuralChatbot(chatbot_settings)


class TestNeuralChatbot:
    """Test suite for NeuralChatbot"""
    
    def test_chatbot_initialization(self, chatbot):
        """Test chatbot initialization"""
        assert chatbot is not None
        assert not chatbot.is_initialized
        assert not chatbot.is_trained
    
    def test_chatbot_training(self, chatbot, temp_intents_file):
        """Test chatbot training process"""
        # Train the chatbot
        results = chatbot.train(temp_intents_file)
        
        # Verify training results
        assert 'final_accuracy' in results
        assert 'validation_accuracy' in results
        assert 'vocabulary_size' in results
        assert 'num_classes' in results
        
        # Verify chatbot state
        assert chatbot.is_trained
        assert chatbot.is_initialized
        assert len(chatbot.processor.words) > 0
        assert len(chatbot.processor.classes) > 0
    
    def test_intent_prediction(self, chatbot, temp_intents_file):
        """Test intent prediction"""
        # Train first
        chatbot.train(temp_intents_file)
        
        # Test predictions
        intent, confidence, all_predictions = chatbot.predict_intent("hello")
        assert intent in ['greeting', 'unknown']
        assert 0.0 <= confidence <= 1.0
        assert isinstance(all_predictions, list)
        
        intent, confidence, _ = chatbot.predict_intent("goodbye")
        assert intent in ['goodbye', 'unknown']
    
    def test_response_generation(self, chatbot, temp_intents_file):
        """Test response generation"""
        # Train first
        chatbot.train(temp_intents_file)
        
        # Test response generation
        response = chatbot.get_response("hello", "test_user")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check conversation history
        history = chatbot.get_conversation_history("test_user")
        assert len(history) == 1
        assert history[0]['message'] == "hello"
        assert history[0]['response'] == response
    
    def test_conversation_management(self, chatbot, temp_intents_file):
        """Test conversation history management"""
        chatbot.train(temp_intents_file)
        
        # Add multiple conversations
        chatbot.get_response("hello", "user1")
        chatbot.get_response("hi", "user1")
        chatbot.get_response("hello", "user2")
        
        # Check histories
        user1_history = chatbot.get_conversation_history("user1")
        user2_history = chatbot.get_conversation_history("user2")
        
        assert len(user1_history) == 2
        assert len(user2_history) == 1
        
        # Clear history
        chatbot.clear_conversation_history("user1")
        user1_history = chatbot.get_conversation_history("user1")
        assert len(user1_history) == 0
    
    def test_health_check(self, chatbot, temp_intents_file):
        """Test health check functionality"""
        # Before training
        health = chatbot.health_check()
        assert health['status'] == 'not_initialized'
        
        # After training
        chatbot.train(temp_intents_file)
        health = chatbot.health_check()
        assert health['status'] == 'healthy'
        assert 'components' in health
    
    def test_statistics(self, chatbot, temp_intents_file):
        """Test statistics generation"""
        chatbot.train(temp_intents_file)
        
        stats = chatbot.get_statistics()
        assert 'status' in stats
        assert 'model' in stats
        assert 'data' in stats
        assert 'conversations' in stats
        assert 'settings' in stats
    
    def test_input_validation(self, chatbot, temp_intents_file):
        """Test input validation and sanitization"""
        chatbot.train(temp_intents_file)
        
        # Test empty message
        response = chatbot.get_response("", "test_user")
        assert "didn't receive any message" in response.lower()
        
        # Test very long message
        long_message = "x" * 1000
        response = chatbot.get_response(long_message, "test_user")
        assert "too long" in response.lower()
    
    def test_model_evaluation(self, chatbot, temp_intents_file):
        """Test model evaluation"""
        chatbot.train(temp_intents_file)
        
        evaluation = chatbot.evaluate_model()
        assert 'accuracy' in evaluation or 'loss' in evaluation
        assert 'classification_report' in evaluation
        assert 'confusion_matrix' in evaluation
    
    def test_intents_update(self, chatbot, temp_intents_file, sample_intents):
        """Test updating intents"""
        chatbot.train(temp_intents_file)
        
        # Add new intent
        new_intents = sample_intents.copy()
        new_intents['intents'].append({
            "tag": "help",
            "patterns": ["help", "assist"],
            "responses": ["How can I help?"]
        })
        
        # Update without retraining
        result = chatbot.update_intents(new_intents, retrain=False)
        assert result['retrained'] == False
        
        # Update with retraining
        result = chatbot.update_intents(new_intents, retrain=True)
        assert 'final_accuracy' in result


class TestConversationManager:
    """Test suite for ConversationManager"""
    
    def test_conversation_creation(self):
        """Test conversation creation and retrieval"""
        from neural_chatbot.core.chatbot import ConversationManager
        
        manager = ConversationManager()
        
        # Add interaction
        manager.add_interaction(
            user_id="test_user",
            message="hello",
            response="hi there",
            intent="greeting",
            confidence=0.95
        )
        
        # Retrieve conversation
        conversation = manager.get_conversation("test_user")
        assert len(conversation) == 1
        assert conversation[0]['message'] == "hello"
        assert conversation[0]['response'] == "hi there"
        assert conversation[0]['intent'] == "greeting"
        assert conversation[0]['confidence'] == 0.95
    
    def test_conversation_statistics(self):
        """Test conversation statistics"""
        from neural_chatbot.core.chatbot import ConversationManager
        
        manager = ConversationManager()
        
        # Add multiple interactions
        manager.add_interaction("user1", "hello", "hi", "greeting", 0.9)
        manager.add_interaction("user1", "bye", "goodbye", "goodbye", 0.8)
        manager.add_interaction("user2", "hello", "hi", "greeting", 0.95)
        
        stats = manager.get_statistics()
        assert stats['total_users'] == 2
        assert stats['total_interactions'] == 3
        assert stats['average_interactions_per_user'] == 1.5
        assert len(stats['most_common_intents']) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])