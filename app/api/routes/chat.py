from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from api.dependencies import get_database
from datetime import datetime
import uuid

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    input_language: Optional[str] = None
    output_language: str
    input_type: str = "text"  # text, audio, image
    output_type: str = "text"  # text, audio, image

class ChatResponse(BaseModel):
    session_id: str
    response: str
    response_type: str
    media_url: Optional[str] = None
    detected_language: Optional[str] = None

@router.post("/", response_model=ChatResponse)
async def process_chat(
    chat_request: ChatMessage,
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    # Create new session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Save the incoming message
    message_data = {
        "session_id": session_id,
        "message": chat_request.message,
        "input_language": chat_request.input_language,
        "output_language": chat_request.output_language,
        "input_type": chat_request.input_type,
        "output_type": chat_request.output_type,
        "timestamp": datetime.utcnow(),
        "direction": "incoming"
    }
    
    await db.messages.insert_one(message_data)
    
    # Placeholder for actual processing
    # This would involve language detection, LLM processing, etc.
    response = {
        "session_id": session_id,
        "response": f"This is a placeholder response for: {chat_request.message}",
        "response_type": chat_request.output_type,
        "detected_language": chat_request.input_language or "hindi",
        "media_url": None
    }
    
    # Save the outgoing message
    response_data = {
        "session_id": session_id,
        "message": response["response"],
        "language": chat_request.output_language,
        "type": chat_request.output_type,
        "timestamp": datetime.utcnow(),
        "direction": "outgoing"
    }
    
    # Use background task to save the response
    background_tasks.add_task(save_response, db, response_data)
    
    return response

async def save_response(db, response_data):
    """Save response to database in the background"""
    await db.messages.insert_one(response_data)

@router.get("/history/{session_id}", response_model=List[Dict[str, Any]])
async def get_chat_history(
    session_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Retrieve chat history for a given session"""
    cursor = db.messages.find({"session_id": session_id}).sort("timestamp", 1)
    messages = await cursor.to_list(length=100)
    
    # Convert ObjectId to string for JSON serialization
    for message in messages:
        message["_id"] = str(message["_id"])
    
    return messages
