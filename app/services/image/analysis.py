import httpx
import base64
import logging
import os
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Service for analyzing images to extract content and meaning
    Uses HuggingFace models for image captioning and OCR
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "")

        # Model endpoints
        self.caption_model_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        self.ocr_model_url = "https://api-inference.huggingface.co/models/microsoft/trocr-large-printed"
        self.vqa_model_url = "https://api-inference.huggingface.co/models/dandelin/vilt-b32-finetuned-vqa"

    async def analyze_image(self, image_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Analyze an image to extract its content and meaning

        Args:
            image_data: Raw image bytes or base64-encoded string

        Returns:
            Dictionary with image analysis results including caption and OCR text
        """
        # Ensure we have image as bytes
        if isinstance(image_data, str) and image_data.startswith("data:image"):
            # Handle data URLs
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, str):
            # Assume base64
            image_bytes = base64.b64decode(image_data)
        else:
            # Already bytes
            image_bytes = image_data

        try:
            # Run these tasks concurrently for better performance
            caption_task = self._generate_caption(image_bytes)
            ocr_task = self._extract_text(image_bytes)

            # Get results
            caption_result = await caption_task
            ocr_result = await ocr_task

            return {
                "success": True,
                "description": caption_result.get("caption", ""),
                "text_content": ocr_result.get("text", ""),
                "confidence": caption_result.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "success": False,
                "error": "Failed to analyze image",
                "message": str(e)
            }

    async def _generate_caption(self, image_bytes: bytes) -> Dict[str, Any]:
        """Generate a descriptive caption for the image"""
        if not self.api_key:
            return {"caption": "Image caption not available", "confidence": 0.0}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.caption_model_url,
                    content=image_bytes,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return {
                            "caption": result[0].get("generated_text", ""),
                            "confidence": 0.9  # Most models don't return confidence
                        }
                    return {"caption": "Unknown image content", "confidence": 0.0}
                else:
                    logger.error(f"Caption API error: {response.status_code} - {response.text}")
                    return {"caption": "Image description unavailable", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return {"caption": "Failed to describe image", "confidence": 0.0}

    async def _extract_text(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract text from the image using OCR"""
        if not self.api_key:
            return {"text": "", "confidence": 0.0}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.ocr_model_url,
                    content=image_bytes,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return {
                            "text": result[0].get("generated_text", ""),
                            "confidence": 0.85  # Most models don't return confidence
                        }
                    return {"text": "", "confidence": 0.0}
                else:
                    logger.error(f"OCR API error: {response.status_code} - {response.text}")
                    return {"text": "", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {"text": "", "confidence": 0.0}

    async def ask_question_about_image(self, image_bytes: bytes, question: str) -> Dict[str, Any]:
        """
        Ask a specific question about the image content

        Args:
            image_bytes: Raw image data
            question: Question to ask about the image

        Returns:
            Dictionary with the answer
        """
        if not self.api_key:
            return {"answer": "Visual question answering not available", "confidence": 0.0}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            # Encode image to base64
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "inputs": {
                    "image": encoded_image,
                    "question": question
                }
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.vqa_model_url,
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "answer": result.get("answer", "I don't know"),
                        "confidence": result.get("score", 0.0)
                    }
                else:
                    logger.error(f"VQA API error: {response.status_code} - {response.text}")
                    return {"answer": "Cannot answer questions about this image", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error with visual QA: {str(e)}")
            return {"answer": "Failed to process question", "confidence": 0.0}

# Create a singleton instance
image_analyzer = ImageAnalyzer()