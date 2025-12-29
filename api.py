from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import the translation function from main.py
from main import translate, TranslationOutput

app = FastAPI(
    title="Translation API",
    description="API for translating text between languages using SD-15 MT Model",
    version="1.0.0"
)

# Configure CORS - Allow all origins for public API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for public deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

@app.get("/")
async def root():
    return {
        "message": "Translation API is running",
        "endpoints": {
            "translate": "/translate (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text from source language to target language.

    Args:
        text: The text to translate
        source_language: Source language (e.g., "English", "Bangla")
        target_language: Target language (e.g., "English", "Bangla")

    Returns:
        TranslationResponse with original and translated text
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Call the translation function
        result = translate(
            text=request.text,
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

        return TranslationResponse(
            original_text=result.original_text,
            translated_text=result.translated_text,
            source_language=request.source_language,
            target_language=request.target_language
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
