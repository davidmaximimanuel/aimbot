"""
NEBAE — The Miracle Worker (Zero-Cost Edition)
Handles Vision, Image Generation, Audio, and PDFs.
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

# ═══════════════════════════════════════
# CONFIG
# ════════════════════════════════════════
# Reuse existing Gemini key for Vision (It's free and powerful)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ════════════════════════════════════════
# 1. VISION (Analyze Images)
# ════════════════════════════════════════
async def analyze_image(image_bytes: bytes, prompt: str = "Describe this image in detail.") -> str:
    """Analyzes an image using Gemini 2.5 Flash Vision."""
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
# 2. IMAGE GENERATION (Free via Pollinations)
# ════════════════════════════════════════
async def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[bytes]:
    """Generates an image using the free Pollinations API (Flux model)."""
    try:
        # Pollinations URL format
        safe_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={width}&height={height}&model=flux&nologo=true&seed={os.urandom(4).hex()}"
        
        logger.info(f"Nebulae: Generating image for prompt: {prompt[:50]}...")
        response = requests.get(url, timeout=60) # 60s timeout as image gen takes time
        
        if response.status_code == 200 and len(response.content) > 1000:
            return response.content
        else:
            logger.error(f"Nebulae Image Gen failed: Status {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Nebulae Image Gen Error: {e}")
        return None


# ════════════════════════════════════════
# 3. AUDIO / TEXT-TO-SPEECH (Free via Edge-TTS)
# ════════════════════════════════════════
async def generate_audio(text: str, voice: str = "en-US-GuyNeural") -> Optional[bytes]:
    """Converts text to speech using Microsoft Edge TTS (Free & High Quality)."""
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
# 4. PDF GENERATION (Free via ReportLab)
# ════════════════════════════════════════
def generate_pdf(title: str, content: str) -> bytes:
    """Generates a simple PDF document."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, title)
    
    # Content (Simple wrapping)
    c.setFont("Helvetica", 12)
    y = height - 100
    lines = content.split('\n')
    for line in lines:
        if y < 72: # Bottom of page
            c.showPage()
            y = height - 72
        c.drawString(72, y, line[:90]) # Limit chars per line
        y -= 20
        
    c.save()
    return buffer.getvalue()


# ════════════════════════════════════════
# INTENT DETECTORS (For AIM to use)
# ════════════════════════════════════════
def is_image_gen_request(text: str) -> bool:
    t = text.lower()
    keywords = ["generate image", "create image", "make a picture", "draw me", "generate a photo"]
    return any(kw in t for kw in keywords)

def is_audio_request(text: str) -> bool:
    t = text.lower()
    keywords = ["read this out loud", "say this", "convert to audio", "make an audio", "text to speech"]
    return any(kw in t for kw in keywords)