import httpx
import base64
import io
import logging
import os
import json
import urllib.parse
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    Service for generating images from text prompts using Pollinations.ai
    No API key required - completely free and open-source
    """
    def __init__(self):
        # Base URL for Pollinations.ai
        self.base_url = "https://image.pollinations.ai/prompt"

        # Set a custom user agent to be a good API citizen
        self.user_agent = "MultiliingualChatbot/1.0"

        # Default parameters
        self.default_width = 512
        self.default_height = 512

        # Fallback services if needed
        self.restackio_api_key = os.environ.get("RESTACKIO_API_KEY", "")
        self.restackio_api_url = "https://api.restack.io/v1/vision/text-to-image"

        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
        self.hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

    async def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image based on a text prompt using Pollinations.ai

        Args:
            prompt: Text description of the image to generate
            width: Desired image width
            height: Desired image height
            negative_prompt: Things to avoid in the image (unused in Pollinations.ai but kept for API compatibility)

        Returns:
            Dictionary containing image data or URL
        """
        try:
            # URL encode the prompt
            encoded_prompt = urllib.parse.quote(prompt)

            # Create the full URL with parameters
            # Pollinations.ai format: https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}
            image_url = f"{self.base_url}/{encoded_prompt}?width={width}&height={height}"

            # Fetch the image
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    image_url,
                    headers={"User-Agent": self.user_agent}
                )

                if response.status_code == 200:
                    # Successfully retrieved the image
                    image_bytes = response.content

                    # Convert to base64 for consistent API response
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")

                    return {
                        "success": True,
                        "image_data": base64_image,
                        "image_url": image_url,  # Also provide direct URL
                        "format": "base64",
                        "service": "pollinations.ai"
                    }
                else:
                    logger.error(f"Pollinations.ai error: {response.status_code} - {response.text}")
                    # Try fallback services
                    return await self._try_fallback_services(prompt, width, height, negative_prompt)

        except Exception as e:
            logger.error(f"Error generating image with Pollinations.ai: {str(e)}")
            # Try fallback services
            return await self._try_fallback_services(prompt, width, height, negative_prompt)

    async def _try_fallback_services(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Try fallback services when Pollinations.ai fails"""
        # Try Restackio as first fallback
        if self.restackio_api_key:
            try:
                return await self._generate_with_restackio(
                    prompt, width, height, negative_prompt
                )
            except Exception as e:
                logger.error(f"Error generating image with Restackio: {str(e)}")
                # Fall through to next fallback

        # Try HuggingFace as second fallback
        if self.hf_api_key:
            try:
                return await self._generate_with_huggingface(
                    prompt, width, height, negative_prompt
                )
            except Exception as e:
                logger.error(f"Error generating image with HuggingFace: {str(e)}")

        # Final fallback - generate a placeholder image with text
        return self._generate_placeholder_image(prompt, width, height)

    async def _generate_with_restackio(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use Restackio API to generate an image"""
        # Ensure dimensions are valid (multiples of 32)
        width = (width // 32) * 32
        height = (height // 32) * 32

        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.restackio_api_key}"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.restackio_api_url,
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "image_data": result.get("image_base64"),
                    "format": "base64",
                    "service": "restackio"
                }
            else:
                logger.error(f"Restackio API error: {response.status_code} - {response.text}")
                raise Exception(f"Image generation failed: {response.text}")

    async def _generate_with_huggingface(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use HuggingFace API to generate an image"""
        # Ensure dimensions are valid (multiples of 8)
        width = (width // 8) * 8
        height = (height // 8) * 8

        payload = {
            "inputs": prompt,
            "parameters": {
                "width": width,
                "height": height
            }
        }

        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        headers = {"Authorization": f"Bearer {self.hf_api_key}"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.hf_url,
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                # HuggingFace returns the image directly
                image_bytes = response.content
                base64_image = base64.b64encode(image_bytes).decode("utf-8")

                return {
                    "success": True,
                    "image_data": base64_image,
                    "format": "base64",
                    "service": "huggingface"
                }
            else:
                logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                raise Exception(f"Image generation failed: {response.text}")

    def _generate_placeholder_image(
        self,
        prompt: str,
        width: int,
        height: int
    ) -> Dict[str, Any]:
        """Generate a simple placeholder image with prompt text when API fails"""
        try:
            # Create a simple image with text
            from PIL import Image, ImageDraw, ImageFont

            # Create a blank image with a gradient background
            image = Image.new("RGB", (width, height), color=(240, 240, 240))
            draw = ImageDraw.Draw(image)

            # Create a subtle gradient background
            for y in range(height):
                color_val = int(240 - (y / height) * 40)
                for x in range(width):
                    draw.point((x, y), fill=(color_val, color_val, color_val))

            # Draw a border
            draw.rectangle([(5, 5), (width-6, height-6)], outline=(200, 200, 200))

            # Add text "Image would be generated from:"
            font_size = max(12, min(24, width // 20))

            # Try to load a font, fall back to default if needed
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            # Draw header text
            header_text = "Image would be generated from:"
            draw.text((width//2, height//4), header_text, fill=(50, 50, 50), font=font, anchor="mm")

            # Draw the prompt text, wrapped if necessary
            lines = []
            words = prompt.split()
            current_line = ""

            for word in words:
                test_line = current_line + " " + word if current_line else word
                if draw.textlength(test_line, font=font) < width - 40:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            # Draw each line of the wrapped prompt
            for i, line in enumerate(lines):
                y_position = height//2 + i * (font_size + 5)
                draw.text((width//2, y_position), line, fill=(0, 0, 0), font=font, anchor="mm")

            # Save to a bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            return {
                "success": True,
                "image_data": base64_image,
                "format": "base64",
                "is_placeholder": True,
                "service": "placeholder"
            }

        except Exception as e:
            logger.error(f"Error generating placeholder image: {str(e)}")
            return {
                "success": False,
                "error": "Failed to generate image",
                "message": str(e)
            }

# Create a singleton instance
image_generator = ImageGenerator()