import httpx
import os
from typing import Dict, Any, Optional
import logging
from .supported import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

class Translator:
    """
    Service for translating between languages using Bhashini API
    """
    def __init__(self):
        # Bhashini credentials
        self.user_id = os.environ.get("BHASHINI_USER_ID", "")
        self.ulca_api_key = os.environ.get("BHASHINI_ULCA_API_KEY", "")
        self.base_url = os.environ.get("BHASHINI_BASE_URL", "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/")
        self.pipeline_id = None  # Will be determined based on language pair

        # Fallback to LibreTranslate if Bhashini credentials are not available
        self.libre_translate_url = "https://libretranslate.com"
        self.libre_translate_endpoint = f"{self.libre_translate_url}/translate"
        self.libre_translate_languages_endpoint = f"{self.libre_translate_url}/languages"

    async def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate text from one language to another

        Args:
            text: The text to translate
            source_language: The source language code (auto-detect if None)
            target_language: The target language code

        Returns:
            A dictionary containing the translated text and metadata
        """
        if not text:
            return {"translated_text": "", "source_language": source_language or "auto", "target_language": target_language}

        # Verify target language is supported
        if target_language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Target language {target_language} not in supported languages, defaulting to English")
            target_language = "en"

        # Use Bhashini if credentials are available
        if self.user_id and self.ulca_api_key:
            try:
                return await self._translate_with_bhashini(text, source_language or "en", target_language)
            except Exception as e:
                logger.error(f"Error translating with Bhashini: {str(e)}")
                # Fall through to LibreTranslate

        # Fallback to LibreTranslate
        try:
            return await self._translate_with_libretranslate(text, source_language, target_language)
        except Exception as e:
            logger.error(f"Error translating with LibreTranslate: {str(e)}")

        # Final fallback
        return {
            "translated_text": f"[Translation Failed: {text}]",
            "source_language": source_language or "auto",
            "target_language": target_language,
            "error": True
        }

    async def _translate_with_bhashini(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """Translate text using Bhashini API"""
        # First, ensure we have the correct pipeline ID for this language pair
        if not self.pipeline_id:
            self.pipeline_id = await self._get_translation_pipeline_id(source_language, target_language)

        # Prepare compute API payload
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language
                        }
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

        if self.pipeline_id:
            headers["pipelineId"] = self.pipeline_id

        # Make the API call
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}compute",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                # Extract the translated text from the response
                translated_text = result.get("pipelineResponse", [])[0].get("output", [])[0].get("target", "")
                return {
                    "translated_text": translated_text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "service": "bhashini"
                }
            else:
                logger.error(f"Bhashini translation failed: {response.text}")
                raise Exception(f"Bhashini translation failed: {response.status_code}")

    async def _get_translation_pipeline_id(
        self,
        source_language: str,
        target_language: str
    ) -> Optional[str]:
        """Get the appropriate pipeline ID for a language pair"""
        # Prepare search API payload
        payload = {
            "task": {
                "type": "translation"
            },
            "languages": {
                "sourceLanguage": source_language,
                "targetLanguage": target_language
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
                        logger.warning(f"No translation pipeline found for {source_language} to {target_language}")
                else:
                    logger.error(f"Failed to get pipeline ID: {response.text}")

        except Exception as e:
            logger.error(f"Error getting pipeline ID: {str(e)}")

        return None

    async def _translate_with_libretranslate(
        self,
        text: str,
        source_language: Optional[str],
        target_language: str
    ) -> Dict[str, Any]:
        """Fallback translation using LibreTranslate"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "q": text,
                "source": source_language or "auto",
                "target": target_language
            }

            response = await client.post(
                self.libre_translate_endpoint,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "translated_text": result.get("translatedText", ""),
                    "source_language": source_language or "auto",
                    "target_language": target_language,
                    "service": "libretranslate"
                }
            else:
                logger.error(f"LibreTranslate failed: {response.text}")
                raise Exception(f"LibreTranslate failed: {response.status_code}")

    async def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of languages supported by the translation service

        Returns:
            Dictionary mapping language codes to their details
        """
        # For Bhashini, we'll use our predefined list since it's focused on Indian languages
        if self.user_id and self.ulca_api_key:
            try:
                # We could query Bhashini for supported languages, but for now we'll use our predefined list
                return {code: {"name": details["name"]} for code, details in SUPPORTED_LANGUAGES.items()}
            except Exception as e:
                logger.error(f"Error fetching Bhashini supported languages: {str(e)}")

        # Fallback to LibreTranslate
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.libre_translate_languages_endpoint)

                if response.status_code == 200:
                    languages = response.json()
                    return {lang["code"]: {"name": lang["name"]} for lang in languages}
                else:
                    logger.error(f"Failed to fetch languages: {response.text}")

        except Exception as e:
            logger.error(f"Error fetching supported languages: {str(e)}")

        # Return predefined supported languages as final fallback
        return {code: {"name": details["name"]} for code, details in SUPPORTED_LANGUAGES.items()}

# Create a singleton instance
translator = Translator()