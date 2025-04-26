import logging
import uuid
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Service for managing user chat sessions
    Handles session creation, retrieval, and history management
    """

    def __init__(self):
        self.default_session_timeout = 60 * 60 * 24  # 24 hours in seconds

    async def create_session(
        self,
        db: AsyncIOMotorDatabase,
        preferred_language: str,
        user_id: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat session

        Args:
            db: MongoDB database instance
            preferred_language: User's preferred language
            user_id: Optional user identifier
            device_info: Optional device metadata

        Returns:
            Session data including the new session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "preferred_language": preferred_language,
            "created_at": now,
            "last_active": now,
            "device_info": device_info or {},
            "is_active": True,
            "message_count": 0
        }

        await db.sessions.insert_one(session_data)
        logger.info(f"Created new session {session_id} with language {preferred_language}")

        return session_data

    async def get_session(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID

        Args:
            db: MongoDB database instance
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        session = await db.sessions.find_one({"session_id": session_id})

        if not session:
            logger.warning(f"Session {session_id} not found")
            return None

        # Update last active timestamp
        await self.update_last_active(db, session_id)

        return session

    async def update_last_active(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str
    ) -> bool:
        """
        Update the last active timestamp for a session

        Args:
            db: MongoDB database instance
            session_id: Session identifier

        Returns:
            True if updated successfully, False otherwise
        """
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"last_active": datetime.utcnow()}}
        )

        return result.modified_count > 0

    async def update_preferred_language(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str,
        language: str
    ) -> bool:
        """
        Update the preferred language for a session

        Args:
            db: MongoDB database instance
            session_id: Session identifier
            language: New preferred language code

        Returns:
            True if updated successfully, False otherwise
        """
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "preferred_language": language,
                "last_active": datetime.utcnow()
            }}
        )

        if result.modified_count > 0:
            logger.info(f"Updated language preference for session {session_id} to {language}")
            return True

        logger.warning(f"Failed to update language for session {session_id}")
        return False

    async def add_message_to_history(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str,
        message: str,
        is_user: bool,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a message to the session history

        Args:
            db: MongoDB database instance
            session_id: Session identifier
            message: Message content
            is_user: True if message is from user, False if from bot
            language: Language of the message
            metadata: Additional message metadata

        Returns:
            Message ID if added successfully, None otherwise
        """
        # Check if session exists
        session = await self.get_session(db, session_id)
        if not session:
            logger.error(f"Cannot add message to non-existent session {session_id}")
            return None

        # Create message document
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()

        message_data = {
            "message_id": message_id,
            "session_id": session_id,
            "content": message,
            "is_user": is_user,
            "role": "user" if is_user else "assistant",
            "language": language or session["preferred_language"],
            "timestamp": now,
            "metadata": metadata or {}
        }

        # Insert the message
        await db.messages.insert_one(message_data)

        # Update message count in session
        await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$inc": {"message_count": 1},
                "$set": {"last_active": now}
            }
        )

        logger.debug(f"Added message to session {session_id}: {message_id}")
        return message_id

    async def get_session_history(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve message history for a session

        Args:
            db: MongoDB database instance
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            skip: Number of messages to skip (for pagination)

        Returns:
            List of message objects in chronological order
        """
        cursor = db.messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).skip(skip).limit(limit)

        messages = await cursor.to_list(length=limit)

        # Convert MongoDB ObjectId to string for serialization
        for message in messages:
            if "_id" in message:
                message["_id"] = str(message["_id"])

        return messages

    async def end_session(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str
    ) -> bool:
        """
        End a session (mark as inactive)

        Args:
            db: MongoDB database instance
            session_id: Session identifier

        Returns:
            True if ended successfully, False otherwise
        """
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False}}
        )

        if result.modified_count > 0:
            logger.info(f"Ended session {session_id}")
            return True

        logger.warning(f"Failed to end session {session_id}")
        return False

    async def get_active_sessions(
        self,
        db: AsyncIOMotorDatabase,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all active sessions, optionally filtered by user

        Args:
            db: MongoDB database instance
            user_id: Optional user identifier to filter by
            limit: Maximum number of sessions to retrieve

        Returns:
            List of active session objects
        """
        filter_query = {"is_active": True}
        if user_id:
            filter_query["user_id"] = user_id

        cursor = db.sessions.find(filter_query).sort("last_active", -1).limit(limit)
        sessions = await cursor.to_list(length=limit)

        # Convert MongoDB ObjectId to string for serialization
        for session in sessions:
            if "_id" in session:
                session["_id"] = str(session["_id"])

        return sessions

# Create a singleton instance
session_manager = SessionManager()