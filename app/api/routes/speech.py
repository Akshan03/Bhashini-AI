from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.api.dependencies import get_database
from pydantic import BaseModel
from typing import Optional
import io
import base64
from app.services.speech import speech_to_text_service, text_to_speech_service

router = APIRouter()

class TextToSpeechRequest(BaseModel):
    text: str
    language: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0

@router.post("/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Convert speech audio to text"""
    # Read the audio content
    audio_content = await audio_file.read()

    # Use our speech-to-text service
    result = await speech_to_text_service.transcribe_audio(
        audio_data=audio_content,
        language=language
    )

    # Log the request
    await db.speech_requests.insert_one({
        "operation": "speech_to_text",
        "language": language,
        "file_size": len(audio_content),
        "transcription": result.get("transcription", ""),
        "detected_language": result.get("detected_language", "unknown"),
        "service": result.get("service", "unknown")
    })

    return result

@router.post("/text-to-speech")
async def text_to_speech(
    request: TextToSpeechRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Convert text to speech audio"""
    # Use our text-to-speech service
    result = await text_to_speech_service.generate_speech(
        text=request.text,
        language=request.language,
        voice=request.voice,
        speed=request.speed or 1.0
    )

    # Log the request
    await db.speech_requests.insert_one({
        "operation": "text_to_speech",
        "text": request.text,
        "language": request.language,
        "voice": request.voice,
        "service": result.get("service", "unknown")
    })

    # Check if we have audio data
    if not result.get("audio_data"):
        raise HTTPException(status_code=500, detail="Failed to generate speech")

    # Create a BytesIO object from the audio data
    audio_content = io.BytesIO(result["audio_data"])

    return StreamingResponse(
        audio_content,
        media_type=f"audio/{result.get('format', 'wav')}",
        headers={
            "Content-Disposition": f"attachment; filename=speech.{result.get('format', 'wav')}"
        }
    )

@router.post("/base64-speech-to-text")
async def base64_speech_to_text(
    audio_base64: str,
    language: Optional[str] = None,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Convert base64-encoded speech audio to text"""
    # Use our speech-to-text service
    result = await speech_to_text_service.transcribe_audio(
        audio_data=audio_base64,
        language=language
    )

    # Log the request
    await db.speech_requests.insert_one({
        "operation": "base64_speech_to_text",
        "language": language,
        "transcription": result.get("transcription", ""),
        "detected_language": result.get("detected_language", "unknown"),
        "service": result.get("service", "unknown")
    })

    return result
