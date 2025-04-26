from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.api.dependencies import get_database
from app.services.llm.client import llm_client
from datetime import datetime, timezone
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
        "timestamp": datetime.now(timezone.utc),
        "direction": "incoming"
    }

    await db.messages.insert_one(message_data)

    # Generate response using Groq LLM
    system_prompt = f"""You are a helpful, multilingual assistant for users in rural India.
    Respond in the {chat_request.output_language} language when requested.
    Be respectful, clear, and concise in your responses.
    Provide practical information that is relevant to rural Indian contexts."""

    llm_result = await llm_client.generate_response(
        prompt=chat_request.message,
        system_prompt=system_prompt,
        language=chat_request.output_language
    )

    response = {
        "session_id": session_id,
        "response": llm_result["text"],
        "response_type": chat_request.output_type,
        "detected_language": chat_request.input_language or "en",
        "media_url": None
    }

    # Save the outgoing message
    response_data = {
        "session_id": session_id,
        "message": response["response"],
        "language": chat_request.output_language,
        "type": chat_request.output_type,
        "timestamp": datetime.now(timezone.utc),
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
