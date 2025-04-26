import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from app.api.middleware import LoggingMiddleware
from app.api.dependencies import get_database
from app.api.routes import chat, language, speech, image, session
from app.core.config import settings
import os

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multilingual and Multimodal Chatbot API",
    description="API for a chatbot that supports multiple Indian languages and modalities",
    version="1.0.0"
)

# Set up database connection
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(settings.MONGO_DETAILS)
    app.mongodb = app.mongodb_client[settings.DATABASE_NAME]
    logger.info("Connected to MongoDB")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    logger.info("Disconnected from MongoDB")

# Set up middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(language.router, prefix="/api", tags=["language"])
app.include_router(speech.router, prefix="/api", tags=["speech"])
app.include_router(image.router, prefix="/api", tags=["image"])
app.include_router(session.router, prefix="/api/session", tags=["session"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Multilingual Multimodal Chatbot API",
        "version": "1.0.0",
        "status": "active"
    }

# Health check endpoint
@app.get("/health")
async def health_check(db=Depends(get_database)):
    try:
        # Verify database connection
        await db.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "api": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )