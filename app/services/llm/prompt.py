import re
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Service for managing and optimizing prompts for LLM interactions
    Handles templating, context management, and prompt engineering
    """
    
    def __init__(self):
        # System prompt templates for different scenarios
        self.system_prompts = {
            "default": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible while being safe.",
            "chat": "You are a helpful assistant designed to provide information and assistance to users in rural areas of India. Be respectful, patient, and explain concepts simply.",
            "image_description": "You are an assistant that describes images accurately and concisely. Focus on the main elements in the image.",
            "multilingual": "You are a multilingual assistant that can understand and respond in multiple Indian languages. Be respectful and culturally sensitive.",
            "educational": "You are an educational assistant designed to explain concepts simply and clearly. Use examples that are relevant to rural Indian contexts.",
            "technical_support": "You are a technical support assistant. Explain solutions step-by-step in simple language without technical jargon."
        }
        
        # Token limits to manage context window
        self.max_prompt_tokens = 4000
        self.max_context_messages = 10
    
    def create_system_prompt(
        self, 
        scenario: str = "default", 
        language: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a system prompt for the given scenario and language
        
        Args:
            scenario: The use case scenario ("chat", "image_description", etc.)
            language: Target language code
            user_preferences: Additional user preferences to customize the prompt
            
        Returns:
            Formatted system prompt
        """
        # Get the base system prompt for the scenario
        base_prompt = self.system_prompts.get(scenario, self.system_prompts["default"])
        
        # Add language-specific instructions if needed
        language_instructions = ""
        if language:
            language_name = self._get_language_name(language)
            language_instructions = f"\nThe user prefers to communicate in {language_name}. When appropriate, respond in {language_name}."
        
        # Add user preferences if provided
        preference_instructions = ""
        if user_preferences:
            if user_preferences.get("formal_tone", False):
                preference_instructions += "\nUse a formal and respectful tone."
            if user_preferences.get("simple_language", False):
                preference_instructions += "\nUse simple language and avoid technical terms."
            if user_preferences.get("examples", False):
                preference_instructions += "\nInclude practical examples when explaining concepts."
        
        # Combine all parts
        system_prompt = f"{base_prompt}{language_instructions}{preference_instructions}"
        
        return system_prompt
    
    def prepare_chat_prompt(
        self,
        current_message: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare a complete prompt for a chat interaction
        
        Args:
            current_message: The current user message
            conversation_history: List of previous messages
            system_prompt: Custom system prompt to use
            language: Target language code
            
        Returns:
            Dictionary with formatted prompt and metadata
        """
        # Create system prompt if not provided
        if not system_prompt:
            system_prompt = self.create_system_prompt("chat", language)
        
        # Truncate conversation history if needed
        truncated_history = self._truncate_conversation_history(conversation_history)
        
        # Format the conversation history as context
        context = self._format_conversation_history(truncated_history)
        
        # Combine everything into the final prompt
        final_prompt = current_message
        if context:
            final_prompt = f"Previous conversation:\n{context}\n\nCurrent message: {current_message}"
        
        return {
            "prompt": final_prompt,
            "system_prompt": system_prompt,
            "language": language,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def prepare_image_prompt(
        self,
        user_query: str,
        image_description: str,
        extracted_text: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare a prompt for an image-related query
        
        Args:
            user_query: User's question about the image
            image_description: Description of the image content
            extracted_text: Text extracted from the image (OCR)
            language: Target language code
            
        Returns:
            Dictionary with formatted prompt and metadata
        """
        # Create appropriate system prompt
        system_prompt = self.create_system_prompt("image_description", language)
        
        # Build context from image analysis
        context_parts = [f"Image description: {image_description}"]
        if extracted_text:
            context_parts.append(f"Text found in the image: {extracted_text}")
        
        context = "\n\n".join(context_parts)
        
        # Combine into final prompt
        final_prompt = f"User has uploaded an image and asks: \"{user_query}\"\n\nImage context:\n{context}\n\nPlease respond to the user's query about this image."
        
        return {
            "prompt": final_prompt,
            "system_prompt": system_prompt,
            "language": language,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _truncate_conversation_history(
        self, 
        conversation_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Truncate conversation history to stay within token limits"""
        # If history is already short enough, return as is
        if len(conversation_history) <= self.max_context_messages:
            return conversation_history
        
        # Keep only the most recent messages
        return conversation_history[-self.max_context_messages:]
    
    def _format_conversation_history(
        self, 
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Format conversation history into a string"""
        formatted_messages = []
        for message in conversation_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_messages.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _get_language_name(self, language_code: str) -> str:
        """Convert language code to language name"""
        language_names = {
            "hi": "Hindi",
            "bn": "Bengali",
            "te": "Telugu",
            "mr": "Marathi",
            "ta": "Tamil",
            "ur": "Urdu",
            "gu": "Gujarati",
            "kn": "Kannada",
            "or": "Odia",
            "pa": "Punjabi",
            "en": "English"
        }
        
        return language_names.get(language_code, "the preferred language")

# Create a singleton instance
prompt_manager = PromptManager()

