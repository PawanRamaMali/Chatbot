"""
Common decorators for Neural Chatbot
"""

import time
import functools
from datetime import datetime
from typing import Callable, Any, Optional, Dict
import logging
import traceback

from .logger import get_logger

logger = get_logger(__name__)


def timing(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
        finally:
            # Store timing info in function attribute
            wrapper.last_execution_time = time.time() - start_time
    
    wrapper.last_execution_time = 0.0
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)):
    """Decorator to retry function execution on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {wait_time:.2f} seconds..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise last_exception
            
            return None  # Should never reach here
        return wrapper
    return decorator


def cache(ttl: Optional[float] = None, maxsize: int = 128):
    """Simple caching decorator with TTL support"""
    def decorator(func: Callable) -> Callable:
        cache_dict = {}
        access_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if key in cache_dict:
                if ttl is None or (current_time - access_times[key]) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache_dict[key]
                else:
                    # Remove expired entry
                    del cache_dict[key]
                    del access_times[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Implement LRU eviction if cache is full
            if len(cache_dict) >= maxsize:
                # Remove oldest entry
                oldest_key = min(access_times.keys(), key=access_times.get)
                del cache_dict[oldest_key]
                del access_times[oldest_key]
            
            cache_dict[key] = result
            access_times[key] = current_time
            
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        
        # Add cache management methods
        def clear_cache():
            cache_dict.clear()
            access_times.clear()
            logger.info(f"Cache cleared for {func.__name__}")
        
        def get_cache_info():
            return {
                'cache_size': len(cache_dict),
                'max_size': maxsize,
                'ttl': ttl,
                'hit_ratio': getattr(wrapper, '_hit_count', 0) / max(getattr(wrapper, '_call_count', 1), 1)
            }
        
        wrapper.clear_cache = clear_cache
        wrapper.cache_info = get_cache_info
        wrapper._hit_count = 0
        wrapper._call_count = 0
        
        return wrapper
    return decorator


def validate_types(**type_specs):
    """Decorator to validate function argument types"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_specs.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"{func.__name__}: parameter '{param_name}' must be of type "
                            f"{expected_type.__name__}, got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_calls(level: int = logging.INFO, include_args: bool = False, 
              include_result: bool = False):
    """Decorator to log function calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(func.__module__)
            
            # Prepare log message
            msg_parts = [f"Calling {func.__name__}"]
            
            if include_args:
                if args:
                    msg_parts.append(f"args={args}")
                if kwargs:
                    msg_parts.append(f"kwargs={kwargs}")
            
            func_logger.log(level, " with ".join(msg_parts))
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    func_logger.log(level, f"{func.__name__} returned: {result}")
                else:
                    func_logger.log(level, f"{func.__name__} completed successfully")
                
                return result
                
            except Exception as e:
                func_logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def deprecated(reason: str = ""):
    """Decorator to mark functions as deprecated"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            
            import warnings
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(message)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def singleton(cls):
    """Decorator to make a class a singleton"""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
            logger.debug(f"Created singleton instance of {cls.__name__}")
        return instances[cls]
    
    return get_instance


def thread_safe(func: Callable) -> Callable:
    """Decorator to make function thread-safe using a lock"""
    import threading
    lock = threading.RLock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return wrapper


def rate_limit(calls_per_second: float = 1.0):
    """Decorator to rate limit function calls"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator


def monitor_memory(threshold_mb: float = 100.0):
    """Decorator to monitor memory usage of function"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                
                # Get memory before execution
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                # Get memory after execution
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_diff = mem_after - mem_before
                
                if abs(mem_diff) > threshold_mb:
                    logger.warning(
                        f"{func.__name__} memory change: {mem_diff:+.2f} MB "
                        f"(before: {mem_before:.2f} MB, after: {mem_after:.2f} MB)"
                    )
                else:
                    logger.debug(
                        f"{func.__name__} memory change: {mem_diff:+.2f} MB"
                    )
                
                return result
                
            except ImportError:
                logger.warning("psutil not available, memory monitoring disabled")
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Memory monitoring failed: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def handle_exceptions(default_return=None, log_traceback: bool = True):
    """Decorator to handle exceptions and return default value"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_traceback:
                    logger.error(f"{func.__name__} failed: {e}\n{traceback.format_exc()}")
                else:
                    logger.error(f"{func.__name__} failed: {e}")
                
                return default_return
        return wrapper
    return decorator


def profile(sort_by: str = 'cumulative', lines_to_print: int = 10):
    """Decorator to profile function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import cProfile
                import pstats
                import io
                
                profiler = cProfile.Profile()
                profiler.enable()
                
                result = func(*args, **kwargs)
                
                profiler.disable()
                
                # Generate profile report
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                ps.print_stats(lines_to_print)
                
                logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")
                
                return result
                
            except ImportError:
                logger.warning("cProfile not available, profiling disabled")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def ensure_dir_exists(path_param: str = 'filepath'):
    """Decorator to ensure directory exists for file operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            
            # Get the parameter value
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            if path_param in bound.arguments:
                from pathlib import Path
                filepath = Path(bound.arguments[path_param])
                
                # Create directory if it doesn't exist
                if not filepath.parent.exists():
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {filepath.parent}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceTracker:
    """Context manager and decorator for performance tracking"""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.log(self.log_level, f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            logger.log(self.log_level, f"Completed {self.name} in {duration:.4f}s")
        else:
            logger.error(f"Failed {self.name} after {duration:.4f}s: {exc_val}")
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTracker(func.__name__, self.log_level):
                return func(*args, **kwargs)
        return wrapper
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration if tracking is complete"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Convenience aliases
perf_track = PerformanceTracker
measure_time = timing
safe_call = handle_exceptions