"""
API middleware for Neural Chatbot
Handles rate limiting, security, logging, and request processing
"""

import time
import hashlib
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
import json

from flask import request, jsonify, g, current_app, session
from werkzeug.exceptions import TooManyRequests

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.blocked_ips = {}
    
    def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        
        # Clean old requests
        self.requests[key] = [
            timestamp for timestamp in self.requests[key]
            if now - timestamp < window
        ]
        
        # Check if blocked
        if key in self.blocked_ips:
            if now < self.blocked_ips[key]:
                return False
            else:
                del self.blocked_ips[key]
        
        # Check rate limit
        if len(self.requests[key]) >= limit:
            # Block for 5 minutes if rate limit exceeded
            self.blocked_ips[key] = now + 300
            logger.warning(f"Rate limit exceeded for {key}, blocking for 5 minutes")
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True
    
    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limit stats for a key"""
        now = time.time()
        recent_requests = [
            timestamp for timestamp in self.requests.get(key, [])
            if now - timestamp < 60
        ]
        
        return {
            'requests_last_minute': len(recent_requests),
            'is_blocked': key in self.blocked_ips,
            'block_expires': self.blocked_ips.get(key, 0)
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


def setup_middleware(app):
    """Setup all middleware for the Flask app"""
    
    @app.before_request
    def before_request():
        """Process request before it reaches the route handler"""
        g.start_time = time.time()
        
        # Log request
        logger.info(f"{request.method} {request.path} from {request.remote_addr}")
        
        # Rate limiting
        if hasattr(current_app, 'settings') and current_app.settings.api.rate_limit_per_minute > 0:
            client_ip = get_client_ip()
            rate_limit = current_app.settings.api.rate_limit_per_minute
            
            if not rate_limiter.is_allowed(client_ip, rate_limit):
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': 'Too many requests. Please try again later.',
                    'retry_after': 300
                }), 429
        
        # Request size validation
        if request.content_length and request.content_length > current_app.config.get('MAX_CONTENT_LENGTH', 1024 * 1024):
            return jsonify({'error': 'Request entity too large'}), 413
        
        # CORS preflight
        if request.method == 'OPTIONS':
            return '', 200
    
    @app.after_request
    def after_request(response):
        """Process response after route handler"""
        # Calculate response time
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Log response
        logger.info(f"Response: {response.status_code} in {response.headers.get('X-Response-Time', 'unknown')}")
        
        return response
    
    @app.errorhandler(404)
    def not_found_handler(error):
        """Handle 404 errors"""
        return jsonify({
            'error': 'Not found',
            'message': 'The requested endpoint does not exist',
            'status': 404
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed_handler(error):
        """Handle 405 errors"""
        return jsonify({
            'error': 'Method not allowed',
            'message': 'The method is not allowed for this endpoint',
            'status': 405
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large_handler(error):
        """Handle 413 errors"""
        return jsonify({
            'error': 'Request entity too large',
            'message': 'The request is too large',
            'status': 413
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_handler(error):
        """Handle 429 errors"""
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.',
            'status': 429
        }), 429
    
    @app.errorhandler(500)
    def internal_error_handler(error):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred',
            'status': 500
        }), 500


def get_client_ip() -> str:
    """Get the real client IP address"""
    # Check for X-Forwarded-For header (when behind a proxy)
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    
    # Check for X-Real-IP header
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    
    # Fall back to remote_addr
    return request.remote_addr or 'unknown'


def validate_json_request(required_fields: list = None):
    """Decorator to validate JSON request data"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'error': 'Invalid content type',
                    'message': 'Request must be JSON'
                }), 400
            
            try:
                data = request.get_json()
                if data is None:
                    return jsonify({
                        'error': 'Invalid JSON',
                        'message': 'Request body contains invalid JSON'
                    }), 400
            except Exception as e:
                return jsonify({
                    'error': 'JSON parsing error',
                    'message': str(e)
                }), 400
            
            # Validate required fields
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        'error': 'Missing required fields',
                        'message': f"Required fields: {', '.join(missing_fields)}"
                    }), 400
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_initialization(func: Callable) -> Callable:
    """Decorator to ensure chatbot is initialized"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(current_app, 'chatbot') or not current_app.chatbot.is_initialized:
            return jsonify({
                'error': 'Service unavailable',
                'message': 'Chatbot is not initialized. Please train the model first.',
                'status': 503
            }), 503
        return func(*args, **kwargs)
    return wrapper


def log_request_response(func: Callable) -> Callable:
    """Decorator to log detailed request/response information"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log request details
        request_data = {
            'method': request.method,
            'path': request.path,
            'client_ip': get_client_ip(),
            'user_agent': request.headers.get('User-Agent', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        if request.is_json:
            try:
                request_data['body'] = request.get_json()
            except:
                request_data['body'] = 'invalid_json'
        
        logger.debug(f"Request: {json.dumps(request_data, default=str)}")
        
        # Execute function
        response = func(*args, **kwargs)
        
        # Log response
        response_data = {
            'status_code': response[1] if isinstance(response, tuple) else 200,
            'response_time': f"{time.time() - g.get('start_time', time.time()):.3f}s"
        }
        
        logger.debug(f"Response: {json.dumps(response_data)}")
        
        return response
    return wrapper


def cache_response(ttl: int = 300):
    """Simple in-memory response caching decorator"""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from request
            cache_key = hashlib.md5(
                f"{request.method}:{request.path}:{request.query_string.decode()}".encode()
            ).hexdigest()
            
            now = time.time()
            
            # Check cache
            if cache_key in cache:
                cached_response, timestamp = cache[cache_key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_response
                else:
                    del cache[cache_key]
            
            # Execute function and cache result
            response = func(*args, **kwargs)
            cache[cache_key] = (response, now)
            
            # Clean old cache entries (simple cleanup)
            if len(cache) > 1000:  # Prevent memory issues
                expired_keys = [
                    key for key, (_, timestamp) in cache.items()
                    if now - timestamp >= ttl
                ]
                for key in expired_keys:
                    del cache[key]
            
            logger.debug(f"Cache miss for {cache_key}")
            return response
        return wrapper
    return decorator


def sanitize_response_headers(func: Callable) -> Callable:
    """Decorator to add security headers to responses"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        
        # If response is a tuple (data, status_code), convert to Response object
        if isinstance(response, tuple):
            data, status_code = response
            response = jsonify(data)
            response.status_code = status_code
        
        # Add security headers
        if hasattr(response, 'headers'):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response
    return wrapper


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor endpoint performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            response = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Endpoint error in {func.__name__}: {e}")
            success = False
            raise
        finally:
            execution_time = time.time() - start_time
            
            # Log performance metrics
            metrics = {
                'endpoint': func.__name__,
                'execution_time': execution_time,
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'client_ip': get_client_ip()
            }
            
            logger.info(f"Performance: {json.dumps(metrics)}")
            
            # Alert on slow responses
            if execution_time > 5.0:  # 5 seconds threshold
                logger.warning(f"Slow response detected: {func.__name__} took {execution_time:.2f}s")
        
        return response
    return wrapper


def get_rate_limit_status():
    """Get current rate limit status for monitoring"""
    client_ip = get_client_ip()
    return rate_limiter.get_stats(client_ip)


def reset_rate_limit(key: str = None):
    """Reset rate limit for a specific key or all keys"""
    if key:
        if key in rate_limiter.requests:
            del rate_limiter.requests[key]
        if key in rate_limiter.blocked_ips:
            del rate_limiter.blocked_ips[key]
        logger.info(f"Rate limit reset for {key}")
    else:
        rate_limiter.requests.clear()
        rate_limiter.blocked_ips.clear()
        logger.info("All rate limits reset")


def get_middleware_stats():
    """Get middleware statistics for monitoring"""
    return {
        'rate_limiter': {
            'active_clients': len(rate_limiter.requests),
            'blocked_clients': len(rate_limiter.blocked_ips),
            'total_requests': sum(len(requests) for requests in rate_limiter.requests.values())
        },
        'timestamp': datetime.now().isoformat()
    }
