import httpx
import os
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class LLMClient:
    """Service for interacting with LLMs using Groq API"""

    def __init__(self):
        # Groq API configuration
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.api_url = "https://api.groq.com/v1/chat/completions"

        # Llama 3 model for conversations - choose based on your needs
        self.model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # or use the Maverick model for multimodal

        # Fallback configurations
        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
        self.hf_base_url = "https://api-inference.huggingface.co/models/"
        self.hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"

        # Multilingual support - special models for Indian languages
        self.indic_model = "Sarvam/Sarvam-1"  # Model trained on Indian languages

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response from Groq API"""

        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Try Groq API first if API key is available
        if self.groq_api_key:
            try:
                return await self._generate_with_groq(messages, max_tokens, temperature, language)
            except Exception as e:
                logger.error(f"Error with Groq API: {str(e)}. Falling back to Hugging Face.")
                # Fall through to try Hugging Face

        # Try Hugging Face as fallback
        if self.hf_api_key:
            try:
                return await self._generate_with_huggingface(prompt, system_prompt, max_tokens, temperature, language)
            except Exception as e:
                logger.error(f"Error with Hugging Face: {str(e)}")
                # Fall through to fallback

        # Final fallback
        return self._generate_fallback_response(prompt)

    async def _generate_with_groq(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response using Groq API"""
        # Determine which model to use based on language
        model = self.model

        # Prepare payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Set headers
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result["choices"][0]["message"]["content"],
                        "model": model,
                        "usage": result.get("usage", {}),
                        "service": "groq"
                    }
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    raise Exception(f"Groq API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error with Groq API: {str(e)}")
            raise

    async def _generate_with_huggingface(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Generate text using Hugging Face models"""
        # Format the prompt for the model
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        # Determine which model to use based on language
        model_name = self.hf_model

        # For Indian languages, prefer the Sarvam model
        if language and language in ["hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "or", "pa"]:
            model_name = self.indic_model

        # Prepare headers and payload
        headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                model_url = f"{self.hf_base_url}{model_name}"
                response = await client.post(
                    model_url,
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()

                    # Extract the generated text from the response
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = str(result)

                    return {
                        "text": generated_text,
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": len(formatted_prompt.split()),  # Rough estimate
                            "completion_tokens": len(generated_text.split()),  # Rough estimate
                            "total_tokens": len(formatted_prompt.split()) + len(generated_text.split())  # Rough estimate
                        },
                        "service": "huggingface"
                    }
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                    raise Exception(f"Hugging Face API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error with Hugging Face API: {str(e)}")
            raise

    def _generate_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a fallback response when API fails"""
        # Create a simple rule-based response as fallback
        words = prompt.lower().split()

        if any(greeting in words for greeting in ["hello", "hi", "hey", "greetings", "namaste", "namaskar"]):
            response = "Hello! I'm your assistant. How can I help you today?"
        elif any(question in words for question in ["how", "what", "when", "where", "why", "who"]):
            response = "I understand you have a question, but I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
        else:
            response = "I apologize, but I'm experiencing some technical difficulties at the moment. Please try again later."

        return {
            "text": response,
            "model": "fallback",
            "is_fallback": True,
            "service": "fallback"
        }

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format the prompt according to the model's expected format"""
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."

        # Format for Llama models (works for both Ollama and direct API)
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

        return formatted_prompt

# Create a singleton instance
llm_client = LLMClient()
