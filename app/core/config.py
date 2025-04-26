import os
from typing import List, Optional

class Settings:
    # MongoDB settings
    MONGO_DETAILS = os.getenv('MONGO_DETAILS', 'mongodb://localhost:27017')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'chatbot_db')

    # API Keys
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    WHISPER_API_KEY = os.getenv('WHISPER_API_KEY', '')
    CAMB_API_KEY = os.getenv('CAMB_API_KEY', '')
    RESTACKIO_API_KEY = os.getenv('RESTACKIO_API_KEY', '')

    # Service Configurations
    USE_LOCAL_LLM = os.getenv('USE_LOCAL_LLM', 'true').lower() == 'true'
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')

    # Application Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    PORT = int(os.getenv('PORT', '8000'))

settings = Settings()