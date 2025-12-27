"""
Korean Calligraphy Generator
- Various Korean handwriting fonts support
- High-quality Korean brush calligraphy generation (font + OpenCV)
- Brush effect simulation
- AI texture enhancement (optional, GPU/CPU support)
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# AI module (optional)
AI_AVAILABLE = False
AI_LOADED = False
pipe = None
device = "cpu"
torch_dtype = None

try:
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
    AI_AVAILABLE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[AI] PyTorch detected, Device: {device}")
except ImportError:
    print("[AI] PyTorch/Diffusers not installed - AI features disabled")

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

FONTS = {
    "eastseadokdo": {
        "file": "EastSeaDokdo.ttf",
        "name": "East Sea Dokdo",
        "category": "brush",
        "style": "Rough brush strokes",
        "size_factor": 1.2
    },
    "nanumbrush": {
        "file": "NanumBrushScript.ttf",
        "name": "Nanum Brush",
        "category": "brush",
        "style": "Smooth flowing brush",
        "size_factor": 1.1
    },
    "songmyung": {
        "file": "SongMyung.ttf",
        "name": "Song Myung",
        "category": "brush",
        "style": "Traditional style",
        "size_factor": 1.0
    },
    "gowunbatang": {
        "file": "GowunBatang.ttf",
        "name": "Gowun Batang",
        "category": "brush",
        "style": "Elegant classical",
        "size_factor": 1.0
    },
    "gowunbatangbold": {
        "file": "GowunBatangBold.ttf",
        "name": "Gowun Batang Bold",
        "category": "brush",
        "style": "Bold classical",
        "size_factor": 1.0
    },
    "hahmlet": {
        "file": "Hahmlet.ttf",
        "name": "Hahmlet",
        "category": "brush",
        "style": "Unique serif calligraphy",
        "size_factor": 1.0
    },
    "gaegu": {
        "file": "Gaegu.ttf",
        "name": "Gaegu",
        "category": "handwriting",
        "style": "Cute handwriting",
        "size_factor": 1.0
    },
    "himelody": {
        "file": "HiMelody.ttf",
        "name": "Hi Melody",
        "category": "handwriting",
        "style": "Bright cheerful",
        "size_factor": 1.0
    },
    "singleday": {
        "file": "SingleDay.ttf",
        "name": "Single Day",
        "category": "handwriting",
        "style": "Clean handwriting",
        "size_factor": 1.0
    },
    "nanumpen": {
        "file": "NanumPenScript.ttf",
        "name": "Nanum Pen",
        "category": "handwriting",
        "style": "Pen-like handwriting",
        "size_factor": 1.1
    },
    "cutefont": {
        "file": "CuteFont.ttf",
        "name": "Cute Font",
        "category": "handwriting",
        "style": "Adorable handwriting",
        "size_factor": 1.0
    },
    "gamjaflower": {
        "file": "GamjaFlower.ttf",
        "name": "Gamja Flower",
        "category": "cute",
        "style": "Chubby cute",
        "size_factor": 1.0
    },
    "jua": {
        "file": "Jua.ttf",
        "name": "Jua",
        "category": "cute",
        "style": "Round cute",
        "size_factor": 1.0
    },
    "dohyeon": {
        "file": "DoHyeon.ttf",
        "name": "Do Hyeon",
        "category": "cute",
        "style": "Round bold",
        "size_factor": 0.9
    },
    "gothica1": {
        "file": "GothicA1.ttf",
        "name": "Gothic A1",
        "category": "cute",
        "style": "Clean round gothic",
        "size_factor": 1.0
    },
    "stylish": {
        "file": "Stylish.ttf",
        "name": "Stylish",
        "category": "modern",
        "style": "Modern stylish",
        "size_factor": 1.0
    },
    "notosanskr": {
        "file": "NotoSansKR.ttf",
        "name": "Noto Sans KR",
        "category": "modern",
        "style": "Clean sans",
        "size_factor": 1.0
    },
    "blackhansans": {
        "file": "BlackHanSans.ttf",
        "name": "Black Han Sans",
        "category": "title",
        "style": "Bold title",
        "size_factor": 0.85
    },
}

CATEGORIES = {
    "brush": {"name": "Brush", "icon": "brush", "description": "Traditional brush feel"},
    "handwriting": {"name": "Handwriting", "icon": "edit", "description": "Natural handwriting"},
    "cute": {"name": "Cute", "icon": "favorite", "description": "Round cute feel"},
    "modern": {"name": "Modern", "icon": "auto_awesome", "description": "Clean modern"},
    "title": {"name": "Title", "icon": "local_fire_department", "description": "Bold for titles"},
}

FALLBACK_FONTS = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "C:/Windows/Fonts/batang.ttc",
]


def is_ai_available():
    return AI_AVAILABLE


def is_ai_loaded():
    return AI_LOADED and pipe is not None


def get_device_info():
    if not AI_AVAILABLE:
        return {"available": False, "device": None, "message": "PyTorch/Diffusers not installed"}
    
    return {
        "available": True,
        "device": device,
        "is_gpu": device == "cuda",
        "message": f"{'GPU (CUDA)' if device == 'cuda' else 'CPU'} mode"
    }


def load_ai_model():
    global pipe, AI_LOADED
    
    if not AI_AVAILABLE:
        print("[AI] PyTorch/Diffusers not installed")
        return False
    
    if pipe is not None:
        return True
    
    try:
        print(f"[AI] Loading SDXL model... (Device: {device})")
        
        if device == "cpu":
            print("[AI] [WARNING] CPU mode: Generation may be slow (1-5 min)")
        
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch_dtype
        )
        
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True
        )
        
        pipe = pipe.to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing()
        else:
            pipe.enable_attention_slicing(1)
        
        AI_LOADED = True
        print(f"[AI] Model loaded! (Device: {device})")
        return True
        
    except Exception as e:
        print(f"[AI] Model load failed: {e}")
        pipe = None
        AI_LOADED = False
        return False


def get_available_fonts():
    available = []
    
    for font_id, font_info in FONTS.items():
        font_path = os.path.join(FONTS_DIR, font_info["file"])
        if os.path.exists(font_path):
            available.append({
                "id": font_id,
                "name": font_info["name"],
                "category": font_info["category"],
                "style": font_info["style"]
            })
    
    by_category = {}
    for font in available:
        cat = font["category"]
        if cat not in by_category:
            by_category[cat] = {
                "info": CATEGORIES.get(cat, {"name": cat, "icon": "text_fields"}),
                "fonts": []
            }
        by_category[cat]["fonts"].append(font)
    
    return {
        "fonts": available,
        "by_category": by_category,
        "total": len(available)
    }


def get_font_path(font_type: str) -> tuple:
    font_info = FONTS.get(font_type.lower())
    
    if font_info:
        font_path = os.path.join(FONTS_DIR, font_info["file"])
        if os.path.exists(font_path):
            return font_path, font_info.get("size_factor", 1.0)
    
    for fid, finfo in FONTS.items():
        fp = os.path.join(FONTS_DIR, finfo["file"])
        if os.path.exists(fp):
            return fp, finfo.get("size_factor", 1.0)
    
    for fallback in FALLBACK_FONTS:
        if os.path.exists(fallback):
            return fallback, 1.0
    
    return None, 1.0


def create_high_quality_brush_text(
    text: str,
    size: tuple = (1024, 512),
    font_type: str = 'gaegu',
    artistic_level: str = 'high'
) -> Image.Image:
    
    font_path, size_factor = get_font_path(font_type)
    
    levels = {
        'low': {'size_var': 0.03, 'rotation': 2, 'position_var': 0.03, 'spacing_var': 0.05},
        'medium': {'size_var': 0.08, 'rotation': 5, 'position_var': 0.08, 'spacing_var': 0.12},
        'high': {'size_var': 0.12, 'rotation': 8, 'position_var': 0.12, 'spacing_var': 0.18}
    }
    p = levels.get(artistic_level, levels['high'])
    
    text_length = len(text)
    if text_length <= 3:
        base_size = int(size[1] * 0.65)
    elif text_length <= 6:
        base_size = int(size[1] * 0.55)
    elif text_length <= 10:
        base_size = int(size[1] * 0.45)
    else:
        base_size = int(size[1] * 0.38)
    
    base_size = int(base_size * size_factor)
    base_size = max(base_size, 120)
    
    bg = np.ones((size[1], size[0], 3), dtype=np.uint8) * 248
    noise = np.random.randint(-12, 12, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (3, 3), 0.5)
    
    image = Image.fromarray(bg)
    
    temp_img = Image.new('RGB', size, (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)
    
    try:
        temp_font = ImageFont.truetype(font_path, base_size)
    except:
        try:
            temp_font = ImageFont.truetype("arial.ttf", base_size)
        except:
            temp_font = ImageFont.load_default()
    
    bbox = temp_draw.textbbox((0, 0), text, font=temp_font)
    total_width = bbox[2] - bbox[0]
    
    current_x = (size[0] - total_width) // 2
    base_y = size[1] // 2
    
    for char in text:
        size_factor_rand = 1.0 + random.uniform(-p['size_var'], p['size_var'])
        char_size = int(base_size * size_factor_rand)
        
        try:
            char_font = ImageFont.truetype(font_path, char_size)
        except:
            char_font = temp_font
        
        canvas_size = char_size * 3
        char_img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        
        char_draw.text(
            (canvas_size // 2, canvas_size // 2),
            char,
            font=char_font,
            fill=(0, 0, 0, 255),
            anchor='mm'
        )
        
        rotation = random.uniform(-p['rotation'], p['rotation'])
        char_img = char_img.rotate(rotation, expand=True, fillcolor=(255, 255, 255, 0))
        
        y_offset = int(random.uniform(-p['position_var'] * base_size, p['position_var'] * base_size))
        
        paste_x = current_x - char_img.width // 2
        paste_y = base_y + y_offset - char_img.height // 2
        
        image.paste(char_img, (paste_x, paste_y), char_img)
        
        char_bbox = char_draw.textbbox((0, 0), char, font=char_font)
        char_width = char_bbox[2] - char_bbox[0]
        
        spacing = random.uniform(0.92, 1.08 + p['spacing_var'])
        current_x += int(char_width * spacing)
    
    return image


def apply_brush_effects(image: Image.Image, intensity: str = 'medium') -> Image.Image:
    
    intensity_levels = {
        'light': {'blur': 1.0, 'noise': 8, 'edge': 0.3, 'contrast': 1.10},
        'medium': {'blur': 1.5, 'noise': 12, 'edge': 0.5, 'contrast': 1.15},
        'strong': {'blur': 2.0, 'noise': 18, 'edge': 0.7, 'contrast': 1.20}
    }
    
    params = intensity_levels.get(intensity, intensity_levels['medium'])
    img_array = np.array(image).astype(float)
    
    for i in range(3):
        img_array[:,:,i] = gaussian_filter(img_array[:,:,i], sigma=params['blur'])
    
    mask = img_array < 200
    noise = np.random.normal(0, params['noise'], img_array.shape)
    img_array = img_array + noise * mask
    
    img_gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    for i in range(3):
        img_array[:,:,i] = img_array[:,:,i] - edges_dilated * params['edge']
    
    img_array = np.clip(img_array, 0, 255)
    result = Image.fromarray(img_array.astype(np.uint8))
    
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(params['contrast'])
    
    return result


def apply_ai_texture(
    image: Image.Image,
    ai_strength: float = 0.38,
    seed: int = None
) -> Image.Image:
    global pipe
    
    if not AI_AVAILABLE or pipe is None:
        return image
    
    prompt = """
    High quality Korean calligraphy on traditional paper,
    natural paper texture and ink absorption,
    subtle variations in ink tone,
    professional traditional calligraphy,
    preserve all text exactly as is
    """.strip()
    
    negative_prompt = """
    text distortion, character deformation,
    blurry text, illegible, abstract,
    changed text, modified characters,
    low quality
    """.strip()
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    try:
        num_steps = 25 if device == "cuda" else 15
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=ai_strength,
            num_inference_steps=num_steps,
            guidance_scale=7.0,
            generator=generator
        ).images[0]
        
        return result
    except Exception as e:
        print(f"[AI] Texture enhancement failed: {e}")
        return image


def generate_calligraphy(
    text: str,
    font_type: str = 'gaegu',
    artistic_level: str = 'high',
    brush_intensity: str = 'medium',
    use_ai: bool = False,
    ai_strength: float = 0.38,
    seed: int = None,
    width: int = 1200,
    height: int = 400
) -> tuple:
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    text_len = len(text)
    if text_len <= 4:
        size = (max(width, 900), height)
    elif text_len <= 8:
        size = (max(width, 1300), height)
    else:
        size = (max(width, 1600), height)
    
    base = create_high_quality_brush_text(
        text, size=size, font_type=font_type, artistic_level=artistic_level
    )
    
    with_brush = apply_brush_effects(base, intensity=brush_intensity)
    
    if use_ai and AI_AVAILABLE and pipe is not None:
        final = apply_ai_texture(with_brush, ai_strength=ai_strength, seed=seed)
    else:
        final = with_brush
    
    return base, with_brush, final
