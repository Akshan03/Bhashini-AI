import httpx
import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Union
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Service for interacting with LLMs
    Supports both local models via Ollama and remote models via Hugging Face
    """
    def __init__(self, use_local: bool = True):
        # Configuration for Hugging Face API
        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
        self.hf_base_url = "https://api-inference.huggingface.co/models/"
        
        # Models to use
        self.hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"  # Fallback to this if no API key
        self.local_model = "llama3"  # Default model for Ollama
        
        # Determine primary approach based on environment
        self.use_local = use_local
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Multilingual support - special models for Indian languages
        self.indic_model = "Sarvam/Sarvam-1"  # Model trained on Indian languages
        
        # Check if Ollama is available when using local mode
        if self.use_local:
            self._check_ollama_installation()
    
    def _check_ollama_installation(self):
        """Verify Ollama is installed and download required model if needed"""
        try:
            # Check if Ollama is installed
            result = subprocess.run(["ollama", "list"], 
                                   capture_output=True, 
                                   text=True, 
                                   check=False)
            
            # If successful, check if our model is available
            if result.returncode == 0:
                if self.local_model not in result.stdout:
                    logger.info(f"Downloading model {self.local_model} with Ollama")
                    subprocess.run(["ollama", "pull", self.local_model], 
                                   check=False)
                else:
                    logger.info(f"Model {self.local_model} is already available in Ollama")
            else:
                logger.warning("Ollama command failed. Is Ollama installed?")
                self.use_local = False
        except FileNotFoundError:
            logger.warning("Ollama not found. Falling back to Hugging Face.")
            self.use_local = False
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM based on the prompt
        
        Args:
            prompt: User message or query
            system_prompt: System instructions for the LLM
            max_tokens: Maximum length of the generated response
            temperature: Controls randomness (0-1, higher = more random)
            language: Target language code (for multilingual support)
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        # Format the prompt for the chosen model
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        # First try using local model with Ollama if enabled
        if self.use_local:
            try:
                return await self._generate_with_ollama(formatted_prompt, max_tokens, temperature, language)
            except Exception as e:
                logger.error(f"Error with Ollama: {str(e)}. Falling back to Hugging Face.")
                # Fall through to try Hugging Face
        
        # Try Hugging Face if Ollama fails or is disabled
        try:
            return await self._generate_with_huggingface(formatted_prompt, max_tokens, temperature, language)
        except Exception as e:
            logger.error(f"Error with Hugging Face: {str(e)}")
            return self._generate_fallback_response(prompt)
    
    async def _generate_with_ollama(
        self, 
        prompt: str, 
        max_tokens: int,
        temperature: float, 
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Generate text using local Ollama instance"""
        # Determine which model to use based on language
        model_name = self.local_model
        
        # For Indian languages, we might have specialized models
        if language and language in ["hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "or", "pa"]:
            # Use a specialized model if available, otherwise stick with default
            if "indic" in subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout:
                model_name = "indic"
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.ollama_url,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result.get("response", ""),
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                        }
                    }
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error with Ollama API: {str(e)}")
            raise
    
    async def _generate_with_huggingface(
        self, 
        prompt: str, 
        max_tokens: int,
        temperature: float, 
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Generate text using Hugging Face models"""
        # Determine which model to use based on language
        model_name = self.hf_model
        
        # For Indian languages, prefer the Sarvam model
        if language and language in ["hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "or", "pa"]:
            model_name = self.indic_model
        
        # Prepare headers and payload
        headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}
        
        payload = {
            "inputs": prompt,
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
                            "prompt_tokens": len(prompt.split()),  # Rough estimate
                            "completion_tokens": len(generated_text.split()),  # Rough estimate
                            "total_tokens": len(prompt.split()) + len(generated_text.split())  # Rough estimate
                        }
                    }
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                    raise Exception(f"Hugging Face API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error with Hugging Face API: {str(e)}")
            raise
    
    def _generate_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a fallback response when all LLM services fail"""
        # Create a simple rule-based response as a last resort
        words = prompt.lower().split()
        
        if any(greeting in words for greeting in ["hello", "hi", "hey", "greetings", "namaste", "namaskar"]):
            response = "Hello! I'm your assistant. How can I help you today?"
        elif any(question in words for question in ["how", "what", "when", "where", "why", "who"]):
            response = "I understand you have a question, but I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
        elif len(words) < 5:
            response = "I see your message, but I'm having difficulty processing it. Could you provide more details or try again later?"
        else:
            response = "I apologize, but I'm experiencing some technical difficulties at the moment. Please try again later."
        
        return {
            "text": response,
            "model": "fallback",
            "is_fallback": True,
            "usage": {
                "prompt_tokens": len(words),
                "completion_tokens": len(response.split()),
                "total_tokens": len(words) + len(response.split())
            }
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
