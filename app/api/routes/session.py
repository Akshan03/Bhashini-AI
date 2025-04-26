from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.api.dependencies import get_database
from app.services.session import session_manager
from datetime import datetime

router = APIRouter()

class SessionCreate(BaseModel):
    user_id: Optional[str] = None
    preferred_language: str
    device_info: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_active: datetime
    preferred_language: str

class UpdateLanguageRequest(BaseModel):
    language: str

@router.post("/create", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Create a new chat session"""
    session = await session_manager.create_session(
        db=db,
        preferred_language=session_data.preferred_language,
        user_id=session_data.user_id,
        device_info=session_data.device_info
    )

    return {
        "session_id": session["session_id"],
        "created_at": session["created_at"],
        "last_active": session["last_active"],
        "preferred_language": session["preferred_language"]
    }

@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get session details by ID"""
    session = await session_manager.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return session

@router.put("/{session_id}/update-language")
async def update_preferred_language(
    session_id: str,
    request: UpdateLanguageRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Update the preferred language for a session"""
    success = await session_manager.update_preferred_language(
        db=db,
        session_id=session_id,
        language=request.language
    )

    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"message": f"Language updated to {request.language}", "session_id": session_id}

@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = 50,
    skip: int = 0,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get message history for a session"""
    # First check if session exists
    session = await session_manager.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    messages = await session_manager.get_session_history(
        db=db,
        session_id=session_id,
        limit=limit,
        skip=skip
    )

    return {
        "session_id": session_id,
        "message_count": session.get("message_count", 0),
        "messages": messages
    }

@router.delete("/{session_id}")
async def end_session(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """End a chat session"""
    success = await session_manager.end_session(db, session_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"message": f"Session {session_id} ended"}

@router.get("/active")
async def get_active_sessions(
    user_id: Optional[str] = None,
    limit: int = 20,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get all active sessions, optionally filtered by user"""
    sessions = await session_manager.get_active_sessions(
        db=db,
        user_id=user_id,
        limit=limit
    )

    return {"sessions": sessions, "count": len(sessions)}
