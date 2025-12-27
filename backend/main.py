"""
Korean Calligraphy API Server
- Various handwriting fonts support
- AI texture enhancement (optional, GPU/CPU support)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import io
import random

from calligraphy import (
    generate_calligraphy, 
    get_available_fonts, 
    FONTS, 
    CATEGORIES,
    is_ai_available,
    is_ai_loaded,
    get_device_info,
    load_ai_model
)

app = FastAPI(
    title="Korean Calligraphy API",
    description="Generate high-quality Korean calligraphy with various handwriting fonts (AI texture enhancement: GPU/CPU support)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalligraphyRequest(BaseModel):
    text: str = Field(..., description="Text to generate", min_length=1, max_length=50)
    font_type: str = Field(default="gaegu", description="Font ID")
    artistic_level: str = Field(default="high", description="Artistic level: low, medium, high")
    brush_intensity: str = Field(default="medium", description="Brush intensity: light, medium, strong")
    use_ai: bool = Field(default=False, description="Use AI texture enhancement (GPU/CPU)")
    ai_strength: float = Field(default=0.38, ge=0.20, le=0.50, description="AI strength (0.35-0.40 recommended)")
    seed: Optional[int] = Field(default=None, description="Random seed (reproducibility)")
    width: int = Field(default=1200, ge=400, le=2000, description="Image width")
    height: int = Field(default=400, ge=200, le=800, description="Image height")


@app.get("/")
async def root():
    device_info = get_device_info()
    return {
        "message": "Korean Calligraphy API",
        "version": "2.1.0",
        "ai_available": is_ai_available(),
        "ai_device": device_info.get("device"),
        "endpoints": {
            "GET /fonts": "Available fonts list",
            "GET /ai/status": "AI availability status",
            "POST /ai/load": "Load AI model",
            "POST /generate": "Generate calligraphy image",
            "GET /health": "Server health check"
        }
    }


@app.get("/health")
async def health_check():
    device_info = get_device_info()
    return {
        "status": "healthy",
        "ai_available": is_ai_available(),
        "ai_device": device_info.get("device")
    }


@app.get("/ai/status")
async def ai_status():
    device_info = get_device_info()
    
    if not is_ai_available():
        return {
            "available": False,
            "model_loaded": False,
            "device": None,
            "is_gpu": False,
            "message": "PyTorch/Diffusers not installed"
        }
    
    return {
        "available": True,
        "model_loaded": is_ai_loaded(),
        "device": device_info.get("device"),
        "is_gpu": device_info.get("is_gpu", False),
        "message": device_info.get("message")
    }


@app.post("/ai/load")
async def load_ai():
    if not is_ai_available():
        raise HTTPException(
            status_code=400, 
            detail="AI unavailable: PyTorch/Diffusers required"
        )
    
    device_info = get_device_info()
    
    success = load_ai_model()
    if success:
        return {
            "status": "loaded", 
            "device": device_info.get("device"),
            "message": f"AI model loaded ({device_info.get('message')})"
        }
    else:
        raise HTTPException(status_code=500, detail="AI model load failed")


@app.get("/fonts")
async def get_fonts():
    available = get_available_fonts()
    
    return {
        "total": available["total"],
        "fonts": available["fonts"],
        "categories": available["by_category"],
        "category_info": CATEGORIES
    }


@app.post("/generate")
async def generate(request: CalligraphyRequest):
    if request.artistic_level not in ["low", "medium", "high"]:
        raise HTTPException(status_code=400, detail="artistic_level must be 'low', 'medium', or 'high'")
    
    if request.brush_intensity not in ["light", "medium", "strong"]:
        raise HTTPException(status_code=400, detail="brush_intensity must be 'light', 'medium', or 'strong'")
    
    if request.use_ai and not is_ai_available():
        request.use_ai = False
    
    try:
        seed = request.seed if request.seed is not None else random.randint(0, 2**31)
        
        base, with_brush, final = generate_calligraphy(
            text=request.text,
            font_type=request.font_type,
            artistic_level=request.artistic_level,
            brush_intensity=request.brush_intensity,
            use_ai=request.use_ai,
            ai_strength=request.ai_strength,
            seed=seed,
            width=request.width,
            height=request.height
        )
        
        img_byte_arr = io.BytesIO()
        final.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)
        
        device_info = get_device_info()
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Font": request.font_type,
                "X-AI-Used": str(request.use_ai).lower(),
                "X-AI-Device": device_info.get("device", "none") if request.use_ai else "none",
                "Content-Disposition": f'inline; filename="calligraphy_{seed}.png"'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
