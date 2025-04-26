import os
import logging
import httpx
import base64
import io
from typing import Dict, Any, Optional, Union, BinaryIO
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class TextToSpeechService:
    """
    Service for converting text to speech using Bhashini TTS
    With fallbacks to CAMB.AI and Hugging Face TTS models
    """
    def __init__(self):
        # Bhashini credentials
        self.user_id = os.environ.get("BHASHINI_USER_ID", "")
        self.ulca_api_key = os.environ.get("BHASHINI_ULCA_API_KEY", "")
        self.base_url = os.environ.get("BHASHINI_BASE_URL", "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/")
        self.tts_pipeline_id = os.environ.get("BHASHINI_TTS_PIPELINE_ID", "")  # Will be determined dynamically

        # Fallback services
        # CAMB.AI integration
        self.camb_api_key = os.environ.get("CAMB_API_KEY", "")
        self.camb_api_url = "https://api.camb.ai/v1/speech/generate"

        # Hugging Face integration
        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY", "")

        # Model endpoints for different languages
        self.hf_models = {
            "default": "fishaudio/fish-speech-1",  # Multilingual model (EN, ZH, JA)
            "hi": "Harveenchadha/indic-tts-hindi-female",
            "bn": "Harveenchadha/indic-tts-bengali-female",
            "te": "Harveenchadha/indic-tts-telugu-female",
            "mr": "Harveenchadha/indic-tts-marathi-female",
            "ta": "Harveenchadha/indic-tts-tamil-female",
            "gu": "Harveenchadha/indic-tts-gujarati-female",
            "kn": "Harveenchadha/indic-tts-kannada-female",
            "or": "Harveenchadha/indic-tts-odia-female",
            "pa": "Harveenchadha/indic-tts-punjabi-female",
            "en": "facebook/seamless-m4t-v2-large"  # Good for Indian English
        }

        # Voice mapping for CAMB.AI
        self.camb_voices = {
            "hi": "hindi-female",
            "bn": "bengali-female",
            "te": "telugu-female",
            "mr": "marathi-female",
            "ta": "tamil-female",
            "gu": "gujarati-female",
            "kn": "kannada-female",
            "pa": "punjabi-female",
            "en": "en-female-1"
        }

    async def generate_speech(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate speech from text

        Args:
            text: Text to convert to speech
            language: Language code (hi, bn, etc.)
            voice: Voice ID (if None, will use default for language)
            speed: Speech rate multiplier

        Returns:
            Dictionary with audio data and metadata
        """
        if not text:
            return {"error": "Empty text provided", "audio_data": None}

        # Normalize language code
        language = language.lower()

        # Try Bhashini first if credentials are available
        if self.user_id and self.ulca_api_key:
            try:
                return await self._generate_with_bhashini(text, language, voice)
            except Exception as e:
                logger.error(f"Error with Bhashini TTS: {str(e)}")
                # Fall through to next method

        # Try CAMB.AI as first fallback
        if self.camb_api_key:
            try:
                return await self._generate_with_camb(text, language, voice, speed)
            except Exception as e:
                logger.error(f"Error with CAMB.AI: {str(e)}")
                # Fall through to next method

        # Try Hugging Face as second fallback
        if self.hf_api_key:
            try:
                return await self._generate_with_huggingface(text, language, voice, speed)
            except Exception as e:
                logger.error(f"Error with Hugging Face: {str(e)}")
                # Fall through to local fallback

        # Local fallback - generate silent audio with metadata
        return self._fallback_speech_generation(text, language)

    async def _generate_with_bhashini(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate speech using Bhashini TTS"""
        # Get TTS pipeline ID if not already set
        if not self.tts_pipeline_id:
            self.tts_pipeline_id = await self._get_tts_pipeline_id(language)
            if not self.tts_pipeline_id:
                raise Exception(f"No TTS pipeline found for language {language}")

        # Prepare payload for TTS
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": language
                        },
                        "gender": voice or "female"  # Default to female voice if not specified
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": text
                    }
                ]
            }
        }

        headers = {
            "userID": self.user_id,
            "ulcaApiKey": self.ulca_api_key,
            "Content-Type": "application/json"
        }

        if self.tts_pipeline_id:
            headers["pipelineId"] = self.tts_pipeline_id

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}compute",
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    # Extract audio data from response
                    audio_base64 = result.get("pipelineResponse", [])[0].get("audio", [])[0].get("audioContent", "")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        return {
                            "audio_data": audio_bytes,
                            "format": "wav",
                            "language": language,
                            "voice": voice or "female",
                            "service": "bhashini"
                        }
                    else:
                        logger.error("Bhashini TTS returned no audio content")
                        raise Exception("No audio content in response")
                else:
                    logger.error(f"Bhashini TTS failed: {response.text}")
                    raise Exception(f"Bhashini TTS failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error in Bhashini TTS: {str(e)}")
            raise

    async def _get_tts_pipeline_id(self, language: str) -> Optional[str]:
        """Get the appropriate TTS pipeline ID for a language"""
        # Prepare search API payload
        payload = {
            "task": {
                "type": "tts"
            },
            "languages": {
                "sourceLanguage": language
            }
        }

        headers = {
            "userID": self.user_id,
            "ulcaApiKey": self.ulca_api_key,
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}search",
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    pipelines = result.get("pipelines", [])
                    if pipelines:
                        # Return the first pipeline ID
                        return pipelines[0].get("pipelineId")
                    else:
                        logger.warning(f"No TTS pipeline found for {language}")
                else:
                    logger.error(f"Failed to get TTS pipeline ID: {response.text}")

        except Exception as e:
            logger.error(f"Error getting TTS pipeline ID: {str(e)}")

        return None

    async def _generate_with_camb(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """Generate speech using CAMB.AI"""
        # Map language to voice if not provided
        if not voice:
            voice = self.camb_voices.get(language, self.camb_voices.get("en", "en-female-1"))

        # Prepare payload
        payload = {
            "text": text,
            "voice": voice,
            "speed": speed
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.camb_api_key}"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.camb_api_url,
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    audio_data = result.get("audio_data")

                    if audio_data:
                        # Audio is usually returned as base64
                        audio_bytes = base64.b64decode(audio_data)

                        return {
                            "audio_data": audio_bytes,
                            "format": "wav",
                            "language": language,
                            "voice": voice,
                            "service": "camb.ai"
                        }
                    else:
                        logger.error("CAMB.AI returned no audio data")
                        raise Exception("No audio data in response")
                else:
                    logger.error(f"CAMB.AI API error: {response.status_code} - {response.text}")
                    raise Exception(f"CAMB.AI speech generation failed: {response.text}")

        except Exception as e:
            logger.error(f"Error with CAMB.AI generation: {str(e)}")
            raise

    async def _generate_with_huggingface(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """Generate speech using Hugging Face"""
        # Select appropriate model for language
        model_id = self.hf_models.get(language, self.hf_models["default"])
        model_url = f"https://api-inference.huggingface.co/models/{model_id}"

        headers = {"Authorization": f"Bearer {self.hf_api_key}"}

        # Prepare payload - format depends on the model
        if "indic-tts" in model_id:
            payload = {"inputs": text}
        elif "fish-speech" in model_id:
            payload = {
                "inputs": {
                    "text": text,
                    "language": language if language in ["en", "zh", "ja"] else "en"
                }
            }
        elif "seamless" in model_id:
            payload = {
                "inputs": text,
                "parameters": {
                    "task": "text_to_speech",
                    "source_lang": language
                }
            }
        else:
            # Generic format for other models
            payload = {"inputs": text}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    model_url,
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    # Response is typically the raw audio bytes
                    audio_bytes = response.content

                    return {
                        "audio_data": audio_bytes,
                        "format": "wav", # Most HF models return wav
                        "language": language,
                        "model": model_id,
                        "service": "huggingface"
                    }
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                    raise Exception(f"Hugging Face speech generation failed: {response.text}")

        except Exception as e:
            logger.error(f"Error with Hugging Face generation: {str(e)}")
            raise

    def _fallback_speech_generation(
        self,
        text: str,
        language: str
    ) -> Dict[str, Any]:
        """Local fallback for when all services fail"""
        # In a real implementation, you might use a local TTS engine
        # For now, this creates a simple audio file with metadata

        logger.warning("Using fallback speech generation")

        # Create a silent WAV file (1 second)
        from array import array
        buffer = io.BytesIO()

        # WAV header for 1 second of silence
        sample_rate = 16000
        channels = 1
        bits_per_sample = 16

        # Write WAV header
        buffer.write(b'RIFF')
        buffer.write((36 + sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write((16).to_bytes(4, 'little'))
        buffer.write((1).to_bytes(2, 'little'))  # PCM format
        buffer.write((channels).to_bytes(2, 'little'))
        buffer.write((sample_rate).to_bytes(4, 'little'))
        buffer.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
        buffer.write((channels * bits_per_sample // 8).to_bytes(2, 'little'))
        buffer.write((bits_per_sample).to_bytes(2, 'little'))
        buffer.write(b'data')
        buffer.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))

        # Write 1 second of silence
        silence = array('h', [0] * sample_rate)
        buffer.write(silence.tobytes())

        # Get the WAV data
        audio_data = buffer.getvalue()

        return {
            "audio_data": audio_data,
            "format": "wav",
            "language": language,
            "is_fallback": True,
            "text_content": text,  # Include original text for display
            "service": "fallback"
        }

# Create a singleton instance
text_to_speech_service = TextToSpeechService()