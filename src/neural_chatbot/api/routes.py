"""
API routes for Neural Chatbot
"""

import uuid
from datetime import datetime
from typing import Dict, Any

from flask import Blueprint, request, jsonify, session, current_app
from marshmallow import Schema, fields, ValidationError

from ..utils.logger import get_logger
from ..utils.helpers import validate_message_length, sanitize_input
from .middleware import (
    validate_json_request, require_initialization, 
    log_request_response, monitor_performance
)

logger = get_logger(__name__)


# Request/Response Schemas
class ChatRequestSchema(Schema):
    message = fields.Str(required=True, validate=lambda x: len(x.strip()) > 0)
    user_id = fields.Str(missing=None)
    context = fields.Dict(missing={})


class ChatResponseSchema(Schema):
    response = fields.Str()
    intent = fields.Str()
    confidence = fields.Float()
    timestamp = fields.DateTime()
    user_id = fields.Str()


class TrainingRequestSchema(Schema):
    intents_data = fields.Dict(required=True)
    retrain = fields.Bool(missing=True)


class ConversationResponseSchema(Schema):
    conversation = fields.List(fields.Dict())
    total_messages = fields.Int()
    user_id = fields.Str()


# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')


def get_or_create_user_id() -> str:
    """Get user ID from session or create new one"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


@api_bp.route('/chat', methods=['POST'])
@monitor_performance
@require_initialization
@validate_json_request(['message'])
def chat():
    """Chat endpoint for sending messages to the chatbot"""
    try:
        # Validate request
        schema = ChatRequestSchema()
        data = schema.load(request.get_json() or {})
        
        message = data['message'].strip()
        user_id = data['user_id'] or get_or_create_user_id()
        
        # Validate message
        if not validate_message_length(message, current_app.settings.security.max_message_length):
            return jsonify({
                'error': f'Message too long. Maximum length is {current_app.settings.security.max_message_length} characters.'
            }), 400
        
        # Get chatbot response
        response = current_app.chatbot.get_response(message, user_id)
        intent, confidence, _ = current_app.chatbot.predict_intent(message)
        
        # Prepare response
        result = {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        
        logger.info(f"Chat request processed for user {user_id}: {intent} ({confidence:.3f})")
        return jsonify(result)
        
    except ValidationError as e:
        return jsonify({'error': 'Invalid request', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/conversation', methods=['GET'])
@monitor_performance
@require_initialization
def get_conversation():
    """Get conversation history for current user"""
    try:
        user_id = get_or_create_user_id()
        conversation = current_app.chatbot.get_conversation_history(user_id)
        
        result = {
            'conversation': conversation,
            'total_messages': len(conversation),
            'user_id': user_id
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/conversation', methods=['DELETE'])
@monitor_performance
@require_initialization
def clear_conversation():
    """Clear conversation history for current user"""
    try:
        user_id = get_or_create_user_id()
        current_app.chatbot.clear_conversation_history(user_id)
        
        logger.info(f"Cleared conversation history for user {user_id}")
        return jsonify({'message': 'Conversation history cleared', 'user_id': user_id})
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/conversation/export', methods=['GET'])
@monitor_performance
@require_initialization
def export_conversation():
    """Export conversation history"""
    try:
        user_id = get_or_create_user_id()
        conversation = current_app.chatbot.get_conversation_history(user_id)
        
        # Create export data
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'conversation_count': len(conversation),
            'conversation': conversation
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/status', methods=['GET'])
@monitor_performance
def get_status():
    """Get chatbot status"""
    try:
        health = current_app.chatbot.health_check()
        statistics = current_app.chatbot.get_statistics()
        
        status = {
            'health': health,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/statistics', methods=['GET'])
@monitor_performance
@require_initialization
def get_statistics():
    """Get detailed chatbot statistics"""
    try:
        stats = current_app.chatbot.get_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/training', methods=['POST'])
@monitor_performance
@validate_json_request(['intents_data'])
def train_model():
    """Train or retrain the chatbot model"""
    try:
        # Validate request
        schema = TrainingRequestSchema()
        data = schema.load(request.get_json() or {})
        
        intents_data = data['intents_data']
        retrain = data['retrain']
        
        # Validate intents data structure
        if not current_app.chatbot.processor.validate_intents_data(intents_data):
            return jsonify({'error': 'Invalid intents data format'}), 400
        
        # Update intents and optionally retrain
        results = current_app.chatbot.update_intents(intents_data, retrain)
        
        logger.info(f"Model training completed: {results}")
        return jsonify({
            'message': 'Training completed successfully',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValidationError as e:
        return jsonify({'error': 'Invalid request', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Error in training endpoint: {e}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


@api_bp.route('/training/status', methods=['GET'])
@monitor_performance
def get_training_status():
    """Get training status"""
    try:
        status = {
            'is_trained': current_app.chatbot.is_trained,
            'is_initialized': current_app.chatbot.is_initialized,
            'model_loaded': current_app.chatbot.model.model is not None,
            'vocabulary_size': len(current_app.chatbot.processor.words) if current_app.chatbot.processor.words else 0,
            'num_classes': len(current_app.chatbot.processor.classes) if current_app.chatbot.processor.classes else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/intents', methods=['GET'])
@monitor_performance
@require_initialization
def get_intents():
    """Get current intents data"""
    try:
        intents = current_app.chatbot.processor.intents_data
        return jsonify(intents)
        
    except Exception as e:
        logger.error(f"Error getting intents: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/intents', methods=['PUT'])
@monitor_performance
@require_initialization
@validate_json_request()
def update_intents():
    """Update intents without retraining"""
    try:
        intents_data = request.get_json()
        
        if not intents_data:
            return jsonify({'error': 'No intents data provided'}), 400
        
        # Validate intents data
        if not current_app.chatbot.processor.validate_intents_data(intents_data):
            return jsonify({'error': 'Invalid intents data format'}), 400
        
        # Update intents without retraining
        result = current_app.chatbot.update_intents(intents_data, retrain=False)
        
        return jsonify({
            'message': 'Intents updated successfully',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating intents: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/predict', methods=['POST'])
@monitor_performance
@require_initialization
@validate_json_request(['message'])
def predict_intent():
    """Predict intent for a message without storing conversation"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Validate message length
        if not validate_message_length(message, current_app.settings.security.max_message_length):
            return jsonify({'error': 'Message too long'}), 400
        
        # Predict intent
        intent, confidence, all_predictions = current_app.chatbot.predict_intent(message)
        
        result = {
            'message': message,
            'predicted_intent': intent,
            'confidence': confidence,
            'all_predictions': [
                {'intent': pred[0], 'confidence': pred[1]} 
                for pred in all_predictions
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/evaluate', methods=['POST'])
@monitor_performance
@require_initialization
def evaluate_model():
    """Evaluate model performance"""
    try:
        # Evaluate model
        evaluation_results = current_app.chatbot.evaluate_model()
        
        return jsonify({
            'evaluation_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/config', methods=['GET'])
@monitor_performance
def get_config():
    """Get current configuration (sanitized)"""
    try:
        config = current_app.settings.to_dict()
        
        # Remove sensitive information
        config['api']['secret_key'] = '***'
        
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/users/stats', methods=['GET'])
@monitor_performance
@require_initialization
def get_user_stats():
    """Get user statistics"""
    try:
        conversation_stats = current_app.chatbot.conversation_manager.get_statistics()
        
        return jsonify({
            'user_statistics': conversation_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/docs', methods=['GET'])
@monitor_performance
def api_documentation():
    """API documentation endpoint"""
    docs = {
        'title': 'Neural Chatbot API Documentation',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/chat': {
                'description': 'Send a message to the chatbot',
                'parameters': {
                    'message': 'string (required) - The message to send',
                    'user_id': 'string (optional) - User identifier',
                    'context': 'object (optional) - Additional context'
                },
                'response': {
                    'response': 'string - Chatbot response',
                    'intent': 'string - Predicted intent',
                    'confidence': 'float - Confidence score',
                    'timestamp': 'string - Response timestamp',
                    'user_id': 'string - User identifier'
                }
            },
            'GET /api/conversation': {
                'description': 'Get conversation history for current user',
                'response': {
                    'conversation': 'array - List of conversation messages',
                    'total_messages': 'integer - Total message count',
                    'user_id': 'string - User identifier'
                }
            },
            'DELETE /api/conversation': {
                'description': 'Clear conversation history for current user',
                'response': {
                    'message': 'string - Success message',
                    'user_id': 'string - User identifier'
                }
            },
            'GET /api/status': {
                'description': 'Get chatbot status and health information',
                'response': {
                    'health': 'object - Health check results',
                    'statistics': 'object - Usage statistics',
                    'timestamp': 'string - Status timestamp'
                }
            },
            'POST /api/training': {
                'description': 'Train or retrain the chatbot model',
                'parameters': {
                    'intents_data': 'object (required) - Intent patterns and responses',
                    'retrain': 'boolean (optional) - Whether to retrain model'
                },
                'response': {
                    'message': 'string - Success message',
                    'results': 'object - Training results',
                    'timestamp': 'string - Training timestamp'
                }
            },
            'POST /api/predict': {
                'description': 'Predict intent for a message',
                'parameters': {
                    'message': 'string (required) - The message to analyze'
                },
                'response': {
                    'predicted_intent': 'string - Predicted intent',
                    'confidence': 'float - Confidence score',
                    'all_predictions': 'array - All predictions with scores'
                }
            }
        },
        'error_codes': {
            '400': 'Bad Request - Invalid parameters',
            '404': 'Not Found - Endpoint not found',
            '405': 'Method Not Allowed - HTTP method not supported',
            '413': 'Request Entity Too Large - Request body too large',
            '429': 'Too Many Requests - Rate limit exceeded',
            '500': 'Internal Server Error - Server error',
            '503': 'Service Unavailable - Chatbot not initialized'
        }
    }
    
    return jsonify(docs)


def register_routes(app):
    """Register all API routes"""
    app.register_blueprint(api_bp)
    logger.info("API routes registered successfully")