# Create: src/neural_chatbot/utils/session_manager.py

import redis
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class SessionManager:
    """Manages user sessions with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 session_timeout_hours: int = 24):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()  # Test connection
            self.use_redis = True
            logger.info("Connected to Redis for session management")
        except:
            logger.warning("Redis not available, using in-memory sessions")
            self.use_redis = False
            self.memory_sessions = {}
        
        self.session_timeout = timedelta(hours=session_timeout_hours)
    
    def create_session(self, user_data: Dict[str, Any] = None) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'user_data': user_data or {}
        }
        
        if self.use_redis:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
        else:
            self.memory_sessions[session_id] = session_data
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if self.use_redis:
            data = self.redis_client.get(f"session:{session_id}")
            return json.loads(data) if data else None
        else:
            return self.memory_sessions.get(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        
        session_data.update(data)
        session_data['last_activity'] = datetime.now().isoformat()
        
        if self.use_redis:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
        else:
            self.memory_sessions[session_id] = session_data
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if self.use_redis:
            return bool(self.redis_client.delete(f"session:{session_id}"))
        else:
            return self.memory_sessions.pop(session_id, None) is not None