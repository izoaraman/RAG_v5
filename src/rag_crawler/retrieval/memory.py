"""
Conversation Memory Module for RAG_v5.

Manages conversation history for context-aware responses.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


class ConversationMemory:
    """
    Manages conversation history with a sliding window.

    Keeps the most recent N exchanges (question + answer pairs) for context.
    """

    def __init__(
        self,
        window_size: int = 10,
        max_tokens_estimate: int = 4000,
    ):
        """
        Initialize conversation memory.

        Args:
            window_size: Number of message pairs to keep.
            max_tokens_estimate: Rough estimate of max tokens to include in context.
        """
        self.window_size = window_size
        self.max_tokens_estimate = max_tokens_estimate
        self._messages: deque[Message] = deque(maxlen=window_size * 2)  # pairs of messages
        self._session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Set session ID."""
        self._session_id = value

    def add_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a user message to history."""
        message = Message(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(message)
        logger.debug(f"Added user message: {content[:50]}...")

    def add_assistant_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an assistant message to history."""
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(message)
        logger.debug(f"Added assistant message: {content[:50]}...")

    def add_system_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a system message (usually at the start)."""
        message = Message(
            role="system",
            content=content,
            metadata=metadata or {},
        )
        # System messages go at the front
        self._messages.appendleft(message)

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """
        Get messages from history.

        Args:
            limit: Maximum number of messages to return (most recent).

        Returns:
            List of messages, oldest first.
        """
        messages = list(self._messages)
        if limit:
            return messages[-limit:]
        return messages

    def get_context_string(self, limit: int | None = None) -> str:
        """
        Get conversation history as a formatted string.

        Args:
            limit: Maximum number of exchanges to include.

        Returns:
            Formatted conversation history string.
        """
        messages = self.get_messages(limit=limit * 2 if limit else None)

        if not messages:
            return ""

        context_parts = []
        for msg in messages:
            role_label = msg.role.capitalize()
            context_parts.append(f"{role_label}: {msg.content}")

        return "\n".join(context_parts)

    def get_recent_exchanges(self, n: int = 2) -> list[tuple[str, str]]:
        """
        Get the most recent N question-answer exchanges.

        Args:
            n: Number of exchanges to return.

        Returns:
            List of (question, answer) tuples.
        """
        messages = list(self._messages)
        exchanges = []

        i = len(messages) - 1
        while i >= 0 and len(exchanges) < n:
            # Find assistant message
            if i >= 0 and messages[i].role == "assistant":
                answer = messages[i].content
                # Find preceding user message
                if i > 0 and messages[i - 1].role == "user":
                    question = messages[i - 1].content
                    exchanges.append((question, answer))
                    i -= 2
                else:
                    i -= 1
            else:
                i -= 1

        return list(reversed(exchanges))

    def get_langchain_messages(self) -> list[dict[str, str]]:
        """
        Get messages in LangChain format.

        Returns:
            List of {"role": str, "content": str} dicts.
        """
        return [{"role": msg.role, "content": msg.content} for msg in self._messages]

    def estimate_token_count(self) -> int:
        """
        Rough estimate of token count in current memory.

        Uses ~4 characters per token as approximation.
        """
        total_chars = sum(len(msg.content) for msg in self._messages)
        return total_chars // 4

    def trim_to_token_limit(self) -> None:
        """Remove oldest messages if over token limit."""
        while self.estimate_token_count() > self.max_tokens_estimate and len(self._messages) > 2:
            self._messages.popleft()
            logger.debug("Trimmed oldest message to stay within token limit")

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        logger.info("Conversation memory cleared")

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "session_id": self._session_id,
            "window_size": self.window_size,
            "max_tokens_estimate": self.max_tokens_estimate,
            "messages": [msg.to_dict() for msg in self._messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMemory":
        """Deserialize memory from dictionary."""
        memory = cls(
            window_size=data.get("window_size", 10),
            max_tokens_estimate=data.get("max_tokens_estimate", 4000),
        )
        memory._session_id = data.get("session_id")

        for msg_data in data.get("messages", []):
            memory._messages.append(Message.from_dict(msg_data))

        return memory

    def __len__(self) -> int:
        """Return number of messages in memory."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Return True if there are messages in memory."""
        return len(self._messages) > 0


class SessionMemoryManager:
    """
    Manages multiple conversation sessions.

    Useful for multi-user or multi-session applications.
    """

    def __init__(
        self,
        default_window_size: int = 10,
        max_sessions: int = 100,
    ):
        """
        Initialize session manager.

        Args:
            default_window_size: Default window size for new sessions.
            max_sessions: Maximum number of sessions to keep.
        """
        self.default_window_size = default_window_size
        self.max_sessions = max_sessions
        self._sessions: dict[str, ConversationMemory] = {}
        self._access_order: list[str] = []

    def get_session(self, session_id: str) -> ConversationMemory:
        """
        Get or create a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            ConversationMemory for the session.
        """
        if session_id not in self._sessions:
            # Enforce max sessions limit
            if len(self._sessions) >= self.max_sessions:
                oldest = self._access_order.pop(0)
                del self._sessions[oldest]
                logger.info(f"Evicted oldest session: {oldest}")

            self._sessions[session_id] = ConversationMemory(window_size=self.default_window_size)
            self._sessions[session_id].session_id = session_id
            logger.info(f"Created new session: {session_id}")

        # Update access order
        if session_id in self._access_order:
            self._access_order.remove(session_id)
        self._access_order.append(session_id)

        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session to delete.

        Returns:
            True if session was deleted, False if not found.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            if session_id in self._access_order:
                self._access_order.remove(session_id)
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """Get list of all session IDs."""
        return list(self._sessions.keys())

    def clear_all(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()
        self._access_order.clear()
        logger.info("Cleared all sessions")
