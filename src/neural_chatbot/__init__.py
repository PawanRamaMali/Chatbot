"""
Neural Chatbot - An intelligent AI chatbot using neural networks
"""

__version__ = '1.0.0'
__author__ = 'Neural Chatbot Team'
__email__ = 'team@neural-chatbot.com'
__license__ = 'MIT'
__description__ = 'An intelligent AI chatbot using neural networks and natural language processing'

from .core.chatbot import NeuralChatbot
from .core.model import ChatbotModel
from .core.processor import DataProcessor
from .core.trainer import ModelTrainer
from .config.settings import Settings

__all__ = [
    'NeuralChatbot',
    'ChatbotModel', 
    'DataProcessor',
    'ModelTrainer',
    'Settings',
]