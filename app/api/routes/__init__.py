# This file makes the routes directory a Python package
from fastapi import APIRouter
from .chat import router as chat_router
from .language import router as language_router
from .speech import router as speech_router
from .image import router as image_router
from .session import router as session_router

api_router = APIRouter()

api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(language_router, prefix="/language", tags=["language"])
api_router.include_router(speech_router, prefix="/speech", tags=["speech"])
api_router.include_router(image_router, prefix="/image", tags=["image"])
api_router.include_router(session_router, prefix="/session", tags=["session"])
