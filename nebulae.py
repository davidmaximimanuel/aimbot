"""
NEBAE — The Miracle Worker (Zero-Cost Edition)
Handles Vision, Image Generation, Audio, PDFs, and Video (safely).
"""

import os
import logging
import requests
import asyncio
from typing import Optional
from urllib.parse import quote
from google import genai
from google.genai import types
import edge_tts
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

logger = logging.getLogger("nebulae")

# Safe OpenCV import for video analysis
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV loaded successfully")
except ImportError as e:
    CV2_AVAILABLE = False
    logger.warning(f"⚠️ OpenCV not available: {e}. Video analysis disabled.")

# ═══════════════════════════════════════
# CONFIG
# ════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ════════════════════════════════════════
# 1. VISION (Analyze Images)
# ════════════════════════════════════════
async def analyze_image(image_bytes: bytes, prompt: str = "Describe this image in detail.") -> str:
    if not gemini_client:
        return "❌ Vision is currently offline."
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt)
            ],
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=1024)
        )
        return response.text
    except Exception as e:
        logger.error(f"Nebulae Vision Error: {e}")
        return "I couldn't analyze this image properly."

# ════════════════════════════════════════
# 2. IMAGE GENERATION
# ════════════════════════════════════════
async def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[bytes]:
    try:
        safe_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={width}&height={height}&model=flux&nologo=true&seed={os.urandom(4).hex()}"
        logger.info(f"Nebulae: Generating image for prompt: {prompt[:50]}...")
        response = requests.get(url, timeout=60)
        if response.status_code == 200 and len(response.content) > 1000:
            return response.content
        return None
    except Exception as e:
        logger.error(f"Nebulae Image Gen Error: {e}")
        return None

# ════════════════════════════════════════
# 3. AUDIO
# ════════════════════════════════════════
async def generate_audio(text: str, voice: str = "en-US-GuyNeural") -> Optional[bytes]:
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_buffer = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()
    except Exception as e:
        logger.error(f"Nebulae Audio Error: {e}")
        return None

# ════════════════════════════════════════
# 4. PDF GENERATION
# ════════════════════════════════════════
def generate_pdf(title: str, content: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, title)
    c.setFont("Helvetica", 12)
    y = height - 100
    for line in content.split('\n'):
        if y < 72:
            c.showPage()
            y = height - 72
        c.drawString(72, y, line[:90])
        y -= 20
    c.save()
    return buffer.getvalue()

# ════════════════════════════════════════
# 5. DOCUMENT ANALYSIS
# ════════════════════════════════════════
async def analyze_document(file_bytes: bytes, mime_type: str, filename: str, prompt: str = "Analyze this document and provide a detailed summary.") -> str:
    if not gemini_client:
        return "❌ Nebulae's document reader is offline."
    try:
        if mime_type == "application/pdf":
            parts = [
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=prompt)
            ]
        else:
            try:
                text_content = file_bytes.decode('utf-8')[:15000]
                parts = [types.Part.from_text(text=f"Document content ({filename}):\n{text_content}\n\n{prompt}")]
            except UnicodeDecodeError:
                return "❌ I can currently read PDFs and text-based files."
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048)
        )
        return response.text
    except Exception as e:
        logger.error(f"Nebulae Document Error: {e}")
        return "I couldn't read this document properly."

# ════════════════════════════════════════
# 6. VIDEO ANALYSIS (Safe)
# ════════════════════════════════════════
async def analyze_video(video_bytes: bytes, prompt: str = "Describe what is happening in this video in detail.") -> str:
    if not CV2_AVAILABLE:
        return "❌ Video analysis is currently unavailable (system libraries missing). Try photos or documents."
    if not gemini_client:
        return "❌ Nebulae's video analyzer is offline."
    
    temp_video_path = f"temp_video_{os.urandom(4).hex()}.mp4"
    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)
            
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return "❌ Could not read video frames."
        
        num_frames = min(6, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        image_parts = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                image_parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
        
        cap.release()
        if not image_parts:
            return "❌ No frames could be extracted."
        
        image_parts.append(types.Part.from_text(text=prompt))
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=image_parts,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=1024)
        )
        return response.text
    except Exception as e:
        logger.error(f"Nebulae Video Error: {e}")
        return "I couldn't analyze this video properly."
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# Intent detectors and logo recognition (unchanged)
def is_image_gen_request(text: str) -> bool:
    t = text.lower()
    keywords = ["generate image", "create image", "make a picture", "draw me", "generate a photo"]
    return any(kw in t for kw in keywords)

def is_audio_request(text: str) -> bool:
    t = text.lower()
    keywords = ["read this out loud", "say this", "convert to audio", "make an audio", "text to speech"]
    return any(kw in t for kw in keywords)

AIM_LOGO_DESCRIPTION = """
AIM's logo features:
- A bright vertical beam of light/cyan/white energy going upward
- Glowing at the top like a star or bright light source
- Wavy, fluid tail at the bottom with organic flowing patterns
- Set against a dark/black background with tiny stars
- Represents intelligence, aspiration, and African innovation
- Minimalist, cosmic, ethereal design
"""

async def is_aim_logo(image_bytes: bytes) -> bool:
    if not gemini_client:
        return False
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=f"Analyze this image carefully. Does it match this description?\n\n{AIM_LOGO_DESCRIPTION}\n\nAnswer ONLY with YES or NO.")
            ],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=10)
        )
        answer = response.text.strip().upper()
        return "YES" in answer
    except Exception as e:
        logger.error(f"Logo recognition error: {e}")
        return False