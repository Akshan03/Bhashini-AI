import httpx
from typing import Dict, Any, Optional
import logging
from .supported import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

class Translator:
    """
    Service for translating between languages using free APIs
    Uses LibreTranslate for translation
    """
    def __init__(self, base_url: str = "https://libretranslate.com/"):
        self.base_url = base_url.rstrip('/')
        self.translate_endpoint = f"{self.base_url}/translate"
        self.languages_endpoint = f"{self.base_url}/languages"

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

        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "q": text,
                    "source": source_language or "auto",
                    "target": target_language
                }

                response = await client.post(
                    self.translate_endpoint,
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "translated_text": result.get("translatedText", ""),
                        "source_language": source_language or "auto",
                        "target_language": target_language
                    }
                else:
                    logger.error(f"Translation failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")

        # Fallback for error cases
        return {
            "translated_text": f"[Translation Failed: {text}]",
            "source_language": source_language or "auto",
            "target_language": target_language,
            "error": True
        }

    async def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of languages supported by the translation service

        Returns:
            Dictionary mapping language codes to their details
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.languages_endpoint)

                if response.status_code == 200:
                    languages = response.json()
                    return {lang["code"]: {"name": lang["name"]} for lang in languages}
                else:
                    logger.error(f"Failed to fetch languages with status code {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Error fetching supported languages: {str(e)}")

        # Return predefined supported languages as fallback
        return {code: {"name": details["name"]} for code, details in SUPPORTED_LANGUAGES.items()}

# Create a singleton instance
translator = Translator()