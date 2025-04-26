"""
Define the languages supported by the chatbot
This includes all the languages we want to support for the rural Indian audience
"""

# Dictionary mapping language codes to language details
SUPPORTED_LANGUAGES = {
    "hi": {
        "name": "Hindi",
        "native_name": "हिन्दी",
        "script": "Devanagari",
        "regions": ["Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan", "Delhi", "Haryana"]
    },
    "bn": {
        "name": "Bengali",
        "native_name": "বাংলা",
        "script": "Bengali",
        "regions": ["West Bengal", "Tripura", "Assam"]
    },
    "te": {
        "name": "Telugu",
        "native_name": "తెలుగు",
        "script": "Telugu",
        "regions": ["Andhra Pradesh", "Telangana"]
    },
    "mr": {
        "name": "Marathi",
        "native_name": "मराठी",
        "script": "Devanagari",
        "regions": ["Maharashtra"]
    },
    "ta": {
        "name": "Tamil",
        "native_name": "தமிழ்",
        "script": "Tamil",
        "regions": ["Tamil Nadu", "Puducherry"]
    },
    "ur": {
        "name": "Urdu",
        "native_name": "اردو",
        "script": "Persian-Arabic",
        "regions": ["Jammu and Kashmir", "Uttar Pradesh", "Bihar"]
    },
    "gu": {
        "name": "Gujarati",
        "native_name": "ગુજરાતી",
        "script": "Gujarati",
        "regions": ["Gujarat", "Dadra and Nagar Haveli", "Daman and Diu"]
    },
    "kn": {
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "script": "Kannada",
        "regions": ["Karnataka"]
    },
    "or": {
        "name": "Odia",
        "native_name": "ଓଡ଼ିଆ",
        "script": "Odia",
        "regions": ["Odisha"]
    },
    "pa": {
        "name": "Punjabi",
        "native_name": "ਪੰਜਾਬੀ",
        "script": "Gurmukhi",
        "regions": ["Punjab", "Haryana"]
    },
    "en": {
        "name": "English",
        "native_name": "English",
        "script": "Latin",
        "regions": ["All India"]
    }
}

# List of language codes
LANGUAGE_CODES = list(SUPPORTED_LANGUAGES.keys())

# Function to get language details
def get_language_details(language_code: str):
    """
    Get details for a specific language

    Args:
        language_code: ISO language code

    Returns:
        Dictionary with language details or None if not supported
    """
    return SUPPORTED_LANGUAGES.get(language_code)

# Function to check if a language is supported
def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported by the chatbot

    Args:
        language_code: ISO language code

    Returns:
        True if supported, False otherwise
    """
    return language_code in SUPPORTED_LANGUAGES

# Function to get language name
def get_language_name(language_code: str, native: bool = False) -> str:
    """
    Get the name of a language

    Args:
        language_code: ISO language code
        native: If True, return the native name instead of English name

    Returns:
        Language name string or language code if not found
    """
    if language_code in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language_code]["native_name" if native else "name"]
    return language_code