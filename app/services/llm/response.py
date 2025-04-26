import re
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseProcessor:
    """
    Service for processing and formatting LLM responses
    Handles post-processing, formatting, and validation of responses
    """
    
    def __init__(self):
        # Regex patterns for filtering/cleaning
        self.code_block_regex = r"``````"
        self.url_regex = r"https?://\S+"
        self.unsafe_content_markers = [
            "I'm sorry, I cannot",
            "I apologize, but I cannot",
            "I'm not able to",
            "As an AI, I'm not comfortable"
        ]
    
    async def process_response(
        self, 
        llm_response: Dict[str, Any],
        language: Optional[str] = None,
        output_format: str = "text",
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and format the LLM response
        
        Args:
            llm_response: Raw response from the LLM
            language: Target language code
            output_format: Desired output format (text, json, markdown)
            user_query: Original user query for context
            
        Returns:
            Processed and formatted response
        """
        # Extract the raw text from the LLM response
        raw_text = llm_response.get("text", "")
        
        # Clean up the text
        cleaned_text = self._clean_response(raw_text)
        
        # Check for potential unsafe content
        is_safe, safety_issues = self._check_safety(cleaned_text)
        
        # Format according to requested output format
        formatted_text = self._format_output(cleaned_text, output_format)
        
        # Prepare the final response object
        processed_response = {
            "content": formatted_text,
            "original_content": raw_text,
            "format": output_format,
            "language": language,
            "model": llm_response.get("model", "unknown"),
            "is_fallback": llm_response.get("is_fallback", False),
            "is_safe": is_safe,
            "safety_issues": safety_issues if not is_safe else [],
            "metadata": {
                "processed_at": datetime.utcnow().isoformat(),
                "tokens": llm_response.get("usage", {}).get("completion_tokens", 0)
            }
        }
        
        # Log the response processing
        logger.info(f"Processed LLM response: format={output_format}, language={language}, is_safe={is_safe}")
        
        return processed_response
    
    def _clean_response(self, text: str) -> str:
        """Clean up the response text"""
        # Remove any system artifacts that might have leaked into the response
        cleaned = re.sub(r"<\/\w+>", "", text)  # Remove closing HTML/XML tags
        cleaned = re.sub(r"^\s*\[.*?\]\s*", "", cleaned)  # Remove markdown-style annotations
        
        # Remove trailing whitespace and normalize line breaks
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # Replace multiple line breaks with double breaks
        
        return cleaned
    
    def _check_safety(self, text: str) -> tuple:
        """Check for potentially unsafe or inappropriate content"""
        # Initialize safety flags
        is_safe = True
        safety_issues = []
        
        # Check for markers of refusal or safety concerns
        for marker in self.unsafe_content_markers:
            if marker.lower() in text.lower():
                is_safe = False
                safety_issues.append("Response contains content refusal markers")
                break
        
        # Additional safety checks can be added here
        
        return is_safe, safety_issues
    
    def _format_output(self, text: str, output_format: str) -> Union[str, Dict]:
        """Format the output according to the requested format"""
        if output_format == "json":
            # Try to parse or convert to JSON
            try:
                # Check if the response is already in JSON format
                if text.strip().startswith("{") and text.strip().endswith("}"):
                    return json.loads(text)
                
                # Extract JSON from code blocks if present
                json_match = re.search(r"``````", text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # If not, wrap the text in a simple JSON structure
                return {"response": text}
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON, returning as text")
                return {"response": text, "format_error": "Failed to parse as JSON"}
        
        elif output_format == "markdown":
            # Return the text as is, assuming it already has markdown formatting
            return text
        
        else:  # Default to plain text
            # Remove markdown code blocks, replacing with plain text
            text = re.sub(self.code_block_regex, r"\1", text)
            
            # Remove markdown formatting symbols
            text = re.sub(r"[*_~`#]+", "", text)
            
            return text
    
    def extract_code_from_response(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from the response"""
        code_blocks = []
        
        # Find all code blocks with language specification
        pattern = r"``````"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            language = match.group(1)
            code = match.group(2).strip()
            code_blocks.append({
                "language": language,
                "code": code
            })
        
        # Find code blocks without language specification
        pattern = r"``````"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            code = match.group(1).strip()
            # Skip if this block was already captured in the previous pattern
            if not any(block["code"] == code for block in code_blocks):
                code_blocks.append({
                    "language": "text",
                    "code": code
                })
        
        return code_blocks
    
    def extract_links_from_response(self, text: str) -> List[str]:
        """Extract URLs from the response"""
        return re.findall(self.url_regex, text)

# Create a singleton instance
response_processor = ResponseProcessor()

