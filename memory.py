"""Memory management system for maintaining conversation history and context."""

from typing import List, Dict, Any
from datetime import datetime
from collections import deque


class ConversationMemory:
    """Manages conversation history and retrieval."""
    
    def __init__(self, max_history: int = 10):
        """Initialize memory with a maximum history size."""
        self.history: deque = deque(maxlen=max_history)
        self.max_history = max_history
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to memory."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(message)
    
    def get_history(self, num_messages: int = None) -> List[Dict[str, Any]]:
        """Retrieve conversation history."""
        if num_messages is None:
            return list(self.history)
        return list(self.history)[-num_messages:]
    
    def get_context(self) -> str:
        """Format history as context for the model."""
        context = []
        for msg in self.history:
            context.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(context)
    
    def clear(self):
        """Clear all history."""
        self.history.clear()
    
    def summary_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        return {
            "total_messages": len(self.history),
            "total_turns": len([m for m in self.history if m["role"] == "user"]),
            "max_capacity": self.max_history
        }
