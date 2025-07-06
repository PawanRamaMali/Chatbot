"""
API module for Neural Chatbot
"""

from .app import create_app
from .routes import api_bp, register_routes
from .middleware import setup_middleware

__all__ = [
    'create_app',
    'api_bp', 
    'register_routes',
    'setup_middleware'
]