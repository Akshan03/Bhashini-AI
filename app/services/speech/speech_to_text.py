import os
import logging
import tempfile
import httpx
import base64
from typing import Dict, Any, Optional, BinaryIO, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class SpeechToTextService:
    """
    Service for converting speech to text using free/open-source APIs
    Primary: WhisperAPI (free tier)
    Fallback: Local Whisper model via Hugging Face
    """
    def __init__(self):
        # API keys and endpoints
        self.whisper_api_key = os.environ.get("WHISPER_API_KEY", "")
        self.whisper_api_url = "https://whisperapi.com/v1/transcribe"

        # Hugging Face integration for fallback
        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
        self.hf_model_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

        # Language mapping for better accuracy
        self.language_codes = {
            "hindi": "hi",
            "bengali": "bn",
            "telugu": "te",
            "marathi": "mr",
            "tamil": "ta",
            "urdu": "ur",
            "gujarati": "gu",
            "kannada": "kn",
            "odia": "or",
            "punjabi": "pa",
            "english": "en"
        }

        # Reverse mapping
        self.code_to_language = {v: k for k, v in self.language_codes.items()}

    async def transcribe_audio(
        self,
        audio_data: Union[bytes, BinaryIO, str],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text

        Args:
            audio_data: Audio content (bytes, file object, or path)
            language: Language code or name (optional, will auto-detect if not provided)

        Returns:
            Dictionary with transcription results
        """
        # Normalize language code if provided
        lang_code = None
        if language:
            if language in self.language_codes:
                lang_code = self.language_codes[language]
            elif language in self.code_to_language:
                lang_code = language

        # Ensure we have audio data as bytes
        audio_bytes = await self._get_audio_bytes(audio_data)

        # Try primary service first
        if self.whisper_api_key:
            try:
                return await self._transcribe_with_whisperapi(audio_bytes, lang_code)
            except Exception as e:
                logger.error(f"Error with WhisperAPI: {str(e)}")
                # Fall through to next method

        # Try Hugging Face as fallback
        if self.hf_api_key:
            try:
                return await self._transcribe_with_huggingface(audio_bytes, lang_code)
            except Exception as e:
                logger.error(f"Error with Hugging Face: {str(e)}")
                # Fall through to local fallback

        # Local fallback using a simple pattern matching approach
        return self._fallback_transcription(audio_bytes, lang_code)

    async def _transcribe_with_whisperapi(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe using WhisperAPI (free tier)"""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)

        try:
            # Prepare payload
            files = {"file": open(temp_path, "rb")}
            data = {}

            if language_code:
                data["language"] = language_code

            headers = {"Authorization": f"Bearer {self.whisper_api_key}"}

            # Make API request
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.whisper_api_url,
                    files=files,
                    data=data,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "transcription": result.get("text", ""),
                        "detected_language": result.get("language", language_code or "unknown"),
                        "confidence": result.get("confidence", 0.0),
                        "segments": result.get("segments", []),
                        "service": "whisperapi"
                    }
                else:
                    logger.error(f"WhisperAPI error: {response.status_code} - {response.text}")
                    raise Exception(f"WhisperAPI transcription failed: {response.text}")

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _transcribe_with_huggingface(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe using Hugging Face"""
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}

        # Prepare payload
        payload = {}
        if language_code:
            payload = {"language": language_code}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.hf_model_url,
                    headers=headers,
                    data=audio_bytes,
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "transcription": result.get("text", ""),
                        "detected_language": language_code or "unknown",
                        "confidence": 0.8,  # Hugging Face doesn't return confidence
                        "service": "huggingface"
                    }
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                    raise Exception(f"Hugging Face transcription failed: {response.text}")

        except Exception as e:
            logger.error(f"Error with Hugging Face transcription: {str(e)}")
            raise

    def _fallback_transcription(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Local fallback for when all services fail"""
        # In a real implementation, you might use a local Vosk model or similar
        # For now, this is just a placeholder
        return {
            "transcription": "[Transcription unavailable - services offline]",
            "detected_language": language_code or "unknown",
            "confidence": 0.0,
            "error": "All transcription services failed",
            "service": "fallback"
        }

    async def _get_audio_bytes(self, audio_data: Union[bytes, BinaryIO, str]) -> bytes:
        """Convert audio_data to bytes regardless of input format"""
        if isinstance(audio_data, bytes):
            return audio_data

        elif isinstance(audio_data, str):
            # Check if it's a base64 string
            if audio_data.startswith('data:audio/'):
                # Handle data URL format
                base64_data = audio_data.split(',')[1]
                return base64.b64decode(base64_data)

            # Check if it's a file path
            elif os.path.isfile(audio_data):
                with open(audio_data, 'rb') as f:
                    return f.read()

            # Try to decode as base64
            try:
                return base64.b64decode(audio_data)
            except:
                raise ValueError("Invalid audio data string")

        elif hasattr(audio_data, 'read'):
            # It's a file-like object
            return audio_data.read()

        else:
            raise ValueError("Unsupported audio data format")

# Create a singleton instance
speech_to_text_service = SpeechToTextService()