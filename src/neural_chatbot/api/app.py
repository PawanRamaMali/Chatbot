"""
Flask API application for Neural Chatbot
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, request, session
from flask_cors import CORS
import click

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_chatbot.config.settings import Settings
from neural_chatbot.core.chatbot import NeuralChatbot
from neural_chatbot.utils.logger import setup_logging, get_logger
from neural_chatbot.api.routes import register_routes
from neural_chatbot.api.middleware import setup_middleware

logger = get_logger(__name__)


def create_app(config_file: str = None) -> Flask:
    """Create and configure Flask application"""
    
    # Load settings
    settings = Settings(config_file)
    
    # Setup logging
    setup_logging(
        level=settings.logging.level,
        log_file=settings.logging.file,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count
    )
    
    logger.info("Creating Flask application...")
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure Flask
    app.config.update({
        'SECRET_KEY': settings.api.secret_key,
        'MAX_CONTENT_LENGTH': settings.api.max_content_length,
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True,
    })
    
    # Setup CORS
    if settings.api.cors_enabled:
        CORS(app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
    
    # Initialize chatbot
    chatbot = NeuralChatbot(settings)
    
    try:
        chatbot.initialize()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize chatbot: {e}")
        logger.info("Chatbot will need to be trained before use")
    
    # Store chatbot and settings in app context
    app.chatbot = chatbot
    app.settings = settings
    
    # Setup middleware
    setup_middleware(app)
    
    # Register routes
    register_routes(app)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return {'error': 'Method not allowed'}, 405
    
    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return {'error': 'Request entity too large'}, 413
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        try:
            health = app.chatbot.health_check()
            return health, 200 if health['status'] == 'healthy' else 503
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'status': 'error', 'error': str(e)}, 503
    
    # Root endpoint
    @app.route('/')
    def index():
        """Root endpoint with API information"""
        return {
            'name': 'Neural Chatbot API',
            'version': '1.0.0',
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'endpoints': {
                'chat': '/api/chat',
                'health': '/health',
                'status': '/api/status',
                'statistics': '/api/statistics',
                'conversation': '/api/conversation',
                'training': '/api/training'
            },
            'documentation': '/api/docs'
        }
    
    logger.info("Flask application created successfully")
    return app


def main():
    """Main function to run the Flask application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Chatbot API Server')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--threaded', action='store_true', help='Enable threaded mode')
    
    args = parser.parse_args()
    
    try:
        # Create application
        app = create_app(args.config)
        
        # Override settings with command line arguments
        host = args.host or app.settings.api.host
        port = args.port or app.settings.api.port
        debug = args.debug or app.settings.api.debug
        
        logger.info(f"Starting Neural Chatbot API server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        # Run the application
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=args.threaded,
            use_reloader=debug
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


@click.command()
@click.option('--config', type=str, help='Path to config file')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=5000, type=int, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
def serve(config, host, port, debug, workers):
    """Serve the Neural Chatbot API using Gunicorn (production)"""
    import subprocess
    
    # Build gunicorn command
    gunicorn_cmd = [
        'gunicorn',
        '--bind', f'{host}:{port}',
        '--workers', str(workers),
        '--worker-class', 'gevent',
        '--timeout', '30',
        '--keep-alive', '2',
        '--max-requests', '1000',
        '--max-requests-jitter', '100',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info',
    ]
    
    if debug:
        gunicorn_cmd.extend(['--reload', '--log-level', 'debug'])
    
    # Set environment variables
    env = os.environ.copy()
    if config:
        env['NEURAL_CHATBOT_CONFIG'] = config
    
    # Add app module
    gunicorn_cmd.append('neural_chatbot.api.app:create_app()')
    
    try:
        logger.info(f"Starting Gunicorn server on {host}:{port} with {workers} workers")
        subprocess.run(gunicorn_cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Gunicorn failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == '__main__':
    main()