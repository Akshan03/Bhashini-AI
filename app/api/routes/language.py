from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorDatabase
from api.dependencies import get_database
from typing import List, Optional
import httpx

router = APIRouter()

class DetectLanguageRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    source_language: Optional[str] = None  # If None, auto-detect
    target_language: str

class SupportedLanguage(BaseModel):
    code: str
    name: str
    native_name: str

@router.post("/detect")
async def detect_language(
    request: DetectLanguageRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Detect the language of the provided text"""
    # This would call LibreTranslate or similar free service
    # For now, return a placeholder response
    
    # Log the request
    await db.language_requests.insert_one({
        "operation": "detect",
        "text": request.text,
        "result": "hindi"  # Placeholder
    })
    
    return {"detected_language": "hindi", "confidence": 0.95}

@router.post("/translate")
async def translate_text(
    request: TranslateRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Translate text between languages"""
    # This would call LibreTranslate or similar free service
    # For now, return a placeholder response
    
    translated_text = f"Translated: {request.text}"
    
    # Log the request
    await db.language_requests.insert_one({
        "operation": "translate",
        "source_text": request.text,
        "source_language": request.source_language,
        "target_language": request.target_language,
        "translated_text": translated_text
    })
    
    return {
        "translated_text": translated_text,
        "source_language": request.source_language or "auto-detected",
        "target_language": request.target_language
    }

@router.get("/supported", response_model=List[SupportedLanguage])
async def get_supported_languages():
    """Get list of supported languages"""
    # This would ideally come from your database or a configuration file
    supported_languages = [
        {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"},
        {"code": "bn", "name": "Bengali", "native_name": "বাংলা"},
        {"code": "te", "name": "Telugu", "native_name": "తెలుగు"},
        {"code": "mr", "name": "Marathi", "native_name": "मराठी"},
        {"code": "ta", "name": "Tamil", "native_name": "தமிழ்"},
        {"code": "ur", "name": "Urdu", "native_name": "اردو"},
        {"code": "gu", "name": "Gujarati", "native_name": "ગુજરાતી"},
        {"code": "kn", "name": "Kannada", "native_name": "ಕನ್ನಡ"},
        {"code": "or", "name": "Odia", "native_name": "ଓଡ଼ିଆ"},
        {"code": "pa", "name": "Punjabi", "native_name": "ਪੰਜਾਬੀ"},
        {"code": "en", "name": "English", "native_name": "English"}
    ]
    return supported_languages
