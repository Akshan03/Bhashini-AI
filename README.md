# Multilingual Multimodal Chatbot for Rural India

A comprehensive chatbot solution that breaks language and technology barriers by supporting multiple Indian languages and various interaction modes.

## Features

- **Multilingual Support**: Communicates in 10+ Indian languages including Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Odia, and Punjabi
- **Multimodal Interaction**: Accepts and generates text, speech, and images
- **Free and Open Source**: Uses freely available APIs and models
- **Resilient Design**: Includes fallback mechanisms for all components

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB
- [Optional] Ollama for local LLM support

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Akshan03/Bhashini-AI.git
   cd Bhashini-AI
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Start MongoDB:
   ```
   # Using Docker
   docker run -d -p 27017:27017 --name mongodb mongo:latest

   # Or use your existing MongoDB installation
   ```

6. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

The API will be available at http://localhost:8000

## API Endpoints

### Chat Endpoints

- `POST /api/chat/` - Send a message to the chatbot
- `GET /api/chat/history/{session_id}` - Get chat history for a session

### Language Processing

- `POST /api/language/detect` - Detect language of text
- `POST /api/language/translate` - Translate text between languages
- `GET /api/language/supported` - Get supported languages

### Speech Processing

- `POST /api/speech/speech-to-text` - Convert speech to text
- `POST /api/speech/text-to-speech` - Convert text to speech

### Image Processing

- `POST /api/image/generate` - Generate an image from text
- `POST /api/image/analyze` - Analyze an image

### Session Management

- `POST /api/session/create` - Create a new chat session
- `GET /api/session/{session_id}` - Get session details
- `PUT /api/session/{session_id}/update-language` - Update preferred language
- `DELETE /api/session/{session_id}` - End a chat session

## Architecture

The project follows a modular architecture with the following components:

- **API Layer**: FastAPI endpoints for handling requests
- **Services Layer**: Business logic for language, speech, image processing, and LLM
- **Database Layer**: MongoDB for persistence

## Technology Stack

- **Backend Framework**: FastAPI
- **LLM Integration**: Hugging Face models and APIs
- **Database**: MongoDB for session management and user preferences
- **Speech Processing**: WhisperAPI for STT, CAMB.AI for TTS
- **Image Processing**: Restackio/HuggingFace for generation and analysis
- **Language Processing**: LibreTranslate for translation

## Project Structure

```
/app
  /api
    /routes
      chat.py
      language.py
      speech.py
      image.py
      session.py
    dependencies.py
    middleware.py
  /core
    config.py
    security.py
    logging.py
  /services
    /language
      detection.py
      translation.py
      supported.py
    /speech
      speech_to_text.py
      text_to_speech.py
    /image
      generation.py
      analysis.py
    /llm
      client.py
      prompt.py
      response.py
    session.py
  main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for open-source models
- LibreTranslate for translation capabilities
- WhisperAPI for speech-to-text
- CAMB.AI for text-to-speech