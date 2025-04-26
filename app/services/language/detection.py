import httpx
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    Service for detecting languages using free APIs
    Uses LibreTranslate for language detection
    """
    def __init__(self, base_url: str = "https://libretranslate.com/"):
        self.base_url = base_url.rstrip('/')
        self.detect_endpoint = f"{self.base_url}/detect"

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a given text

        Args:
            text: The text to detect language for

        Returns:
            A dictionary containing the detected language and confidence
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.detect_endpoint,
                    data={"q": text}
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return {
                            "detected_language": result[0]["language"],
                            "confidence": result[0]["confidence"]
                        }
                    logger.warning(f"Unexpected detection response format: {result}")
                else:
                    logger.error(f"Language detection failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")

        # Fallback detection - very simple, just for demonstration
        fallback_result = self._fallback_detect(text)
        logger.info(f"Using fallback language detection: {fallback_result}")
        return fallback_result

    def _fallback_detect(self, text: str) -> Dict[str, Any]:
        """
        Simple fallback detection based on character sets
        This is a very naive implementation and should only be used as a last resort

        Args:
            text: The text to detect language for

        Returns:
            A dictionary with detected language and confidence
        """
        # Check for common Indic scripts
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        bengali = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        telugu = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        tamil = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')

        total_chars = len(text)
        if total_chars == 0:
            return {"detected_language": "en", "confidence": 0.5}

        if devanagari / total_chars > 0.5:
            return {"detected_language": "hi", "confidence": 0.8}
        elif bengali / total_chars > 0.5:
            return {"detected_language": "bn", "confidence": 0.8}
        elif telugu / total_chars > 0.5:
            return {"detected_language": "te", "confidence": 0.8}
        elif tamil / total_chars > 0.5:
            return {"detected_language": "ta", "confidence": 0.8}

        # Default to English
        return {"detected_language": "en", "confidence": 0.5}

# Create a singleton instance
language_detector = LanguageDetector()