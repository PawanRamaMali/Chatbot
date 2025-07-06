"""
Helper utilities for Neural Chatbot
"""

import re
import html
import unicodedata
from typing import List, Dict, Any, Optional
import string
import hashlib
import json
from datetime import datetime, timedelta


def sanitize_input(text: str, blacklisted_words: Optional[List[str]] = None) -> str:
    """Sanitize user input to prevent injection attacks and clean text"""
    if not text:
        return ""
    
    # HTML escape
    text = html.escape(text)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '{', '}', '[', ']', '\\', '|', '`']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check for blacklisted words
    if blacklisted_words:
        words = text.lower().split()
        filtered_words = [word for word in words if word not in blacklisted_words]
        text = ' '.join(filtered_words)
    
    return text


def validate_message_length(message: str, max_length: int = 500) -> bool:
    """Validate message length"""
    return len(message) <= max_length


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Split into words
    words = text.split()
    
    # Filter words by length and common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    return list(set(keywords))  # Remove duplicates


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using Jaccard similarity"""
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords from both texts
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 and not keywords2:
        return 1.0 if text1.lower() == text2.lower() else 0.0
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union) if union else 0.0


def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())


def hash_text(text: str) -> str:
    """Generate hash for text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string"""
    try:
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        # Try alternative formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def validate_json(json_str: str) -> bool:
    """Validate JSON string"""
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON with default value"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def deep_merge_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    try:
        from pathlib import Path
        return Path(filepath).stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def create_backup_filename(original_filename: str) -> str:
    """Create backup filename with timestamp"""
    from pathlib import Path
    
    path = Path(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return str(path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}")


def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on exception"""
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise last_exception
            
            return None  # This should never be reached
        return wrapper
    return decorator


def measure_execution_time(func):
    """Decorator to measure function execution time"""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Store execution time as function attribute
        wrapper.last_execution_time = execution_time
        
        return result
    
    wrapper.last_execution_time = 0.0
    return wrapper


def validate_email(email: str) -> bool:
    """Validate email address format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None


def generate_random_string(length: int = 8, include_digits: bool = True, 
                          include_special: bool = False) -> str:
    """Generate random string"""
    import random
    
    chars = string.ascii_letters
    if include_digits:
        chars += string.digits
    if include_special:
        chars += "!@#$%^&*"
    
    return ''.join(random.choice(chars) for _ in range(length))


def is_valid_url(url: str) -> bool:
    """Validate URL format"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def convert_bytes_to_human_readable(bytes_value: int) -> str:
    """Convert bytes to human readable format"""
    return format_file_size(bytes_value)


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'free': psutil.disk_usage('/').free,
        }
    }


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a text progress bar"""
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    progress = current / total
    filled_width = int(width * progress)
    
    bar = "=" * filled_width + "-" * (width - filled_width)
    percentage = progress * 100
    
    return f"[{bar}] {percentage:.1f}%"