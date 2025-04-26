from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.api.dependencies import get_database
from pydantic import BaseModel
from app.services.image import image_generator, image_analyzer
import base64
from typing import Optional

router = APIRouter()

class ImageGenerationRequest(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512
    negative_prompt: Optional[str] = None
    language: Optional[str] = "en"  # Language of the prompt

class ImageAnalysisResponse(BaseModel):
    description: str
    text_content: Optional[str] = None
    confidence: Optional[float] = None
    success: bool = True

class ImageQuestionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    question: str

@router.post("/generate")
async def generate_image(
    request: ImageGenerationRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Generate an image based on a text prompt"""
    # Log the request
    await db.image_requests.insert_one({
        "operation": "generate",
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "language": request.language
    })

    # Generate image using Pollinations.ai
    result = await image_generator.generate_image(
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        negative_prompt=request.negative_prompt
    )

    if not result.get("success", False):
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate image", "details": result.get("message", "")}
        )

    # For Pollinations.ai, we can also return a direct URL
    return {
        "success": result.get("success", False),
        "image_url": result.get("image_url", ""),
        "image_data": result.get("image_data", ""),
        "format": result.get("format", "base64"),
        "is_placeholder": result.get("is_placeholder", False),
        "service": result.get("service", ""),
        "prompt": request.prompt
    }

@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Analyze an image to extract text and content"""
    image_content = await image.read()

    # Use our image analysis service
    analysis = await image_analyzer.analyze_image(image_content)

    # Log the request
    await db.image_requests.insert_one({
        "operation": "analyze",
        "file_size": len(image_content),
        "analysis_result": analysis
    })

    return analysis

@router.post("/ask-question")
async def ask_question_about_image(
    request: ImageQuestionRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Ask a specific question about an image"""
    try:
        # Decode the base64 image
        if request.image_data.startswith("data:image"):
            # Handle data URLs
            image_data = request.image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            # Assume base64
            image_bytes = base64.b64decode(request.image_data)

        # Use our image analysis service to ask a question
        result = await image_analyzer.ask_question_about_image(
            image_bytes=image_bytes,
            question=request.question
        )

        # Log the request
        await db.image_requests.insert_one({
            "operation": "ask-question",
            "question": request.question,
            "answer": result.get("answer", "")
        })

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process question", "details": str(e)}
        )
