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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# OpenAI client (for TTS + Image Gen)
_openai_client = None
if OPENAI_API_KEY:
    from openai import AsyncOpenAI as _AsyncOpenAI
    _openai_client = _AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ Nebulae: OpenAI client ready (TTS + Image Gen)")
else:
    logger.warning("⚠️ Nebulae: OPENAI_API_KEY not set — TTS uses Edge-TTS, Image Gen uses Pollinations")

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
    """Generates an image. Uses OpenAI DALL-E 3 if key is set, else Pollinations (free)."""
    # ── OpenAI DALL-E 3 (paid, higher quality) ──────────────
    if _openai_client:
        try:
            logger.info("Nebulae: Generating image via DALL-E 3: %s", prompt[:50])
            response = await _openai_client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )
            image_url = response.data[0].url
            img_resp  = requests.get(image_url, timeout=30)
            if img_resp.status_code == 200:
                return img_resp.content
            logger.error("Nebulae DALL-E 3: failed to download image")
        except Exception as e:
            logger.error("Nebulae DALL-E 3 error: %s — falling back to Pollinations", e)

    # ── Pollinations fallback (free) ─────────────────────────
    try:
        safe_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={width}&height={height}&model=flux&nologo=true&seed={os.urandom(4).hex()}"
        logger.info("Nebulae: Generating image via Pollinations")
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        logger.error("Nebulae Pollinations failed: status %s", resp.status_code)
        return None
    except Exception as e:
        logger.error("Nebulae Image Gen Error: %s", e)
        return None


# ════════════════════════════════════════
# 3. AUDIO / TEXT-TO-SPEECH (Free via Edge-TTS)
# ════════════════════════════════════════
async def generate_audio(text: str, voice: str = "alloy") -> Optional[bytes]:
    """Converts text to speech. Uses OpenAI TTS if key is set, else Edge-TTS (free).
    OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
    """
    text = text[:4096].strip()
    if not text:
        return None

    # ── OpenAI TTS (paid, natural quality) ──────────────────
    if _openai_client:
        try:
            logger.info("Nebulae: TTS via OpenAI (voice=%s), %d chars", voice, len(text))
            response = await _openai_client.audio.speech.create(
                model="gpt-4o-mini-tts",  # Best quality/price balance
                voice=voice,
                input=text,
                response_format="mp3",
            )
            audio_bytes = response.content
            if audio_bytes:
                logger.info("Nebulae TTS: generated %d bytes", len(audio_bytes))
                return audio_bytes
        except Exception as e:
            logger.error("Nebulae OpenAI TTS error: %s — falling back to Edge-TTS", e)

    # ── Edge-TTS fallback (free) ─────────────────────────────
    try:
        logger.info("Nebulae: TTS via Edge-TTS fallback")
        communicate  = edge_tts.Communicate(text[:3000], "en-US-GuyNeural")
        audio_buffer = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            logger.error("Nebulae Edge-TTS: returned empty bytes")
            return None
        return audio_bytes
    except Exception as e:
        logger.error("Nebulae Edge-TTS Error: %s", e)
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

# ════════════════════════════════════
# 5. DOCUMENT READING (Native Gemini PDF + Text Fallback)
# ════════════════════════════════════
async def analyze_document(file_bytes: bytes, mime_type: str, filename: str, prompt: str = "Analyze this document and provide a detailed summary.") -> str:
    """Analyzes a document (PDF, TXT, etc.) using Gemini."""
    if not gemini_client:
        return "❌ Nebulae's document reader is offline."
    try:
        # Gemini natively supports PDFs! We pass the raw bytes.
        if mime_type == "application/pdf":
            parts = [
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=prompt)
            ]
        else:
            # For other files (like .txt, .csv), try to decode as text
            try:
                text_content = file_bytes.decode('utf-8')
                text_content = text_content[:15000] # Truncate to avoid token limits
                parts = [
                    types.Part.from_text(text=f"Document content ({filename}):\n{text_content}\n\n{prompt}")
                ]
            except UnicodeDecodeError:
                return "❌ I can currently read PDFs and text-based files (TXT, CSV). Please send a PDF or text file."

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=parts,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048)
        )
        return response.text
    except Exception as e:
        logger.error(f"Nebulae Document Error: {e}")
        return "I couldn't read this document properly."

# ════════════════════════════════════
# 6. VIDEO ANALYSIS (Frame Extraction)
# ════════════════════════════════════
async def analyze_video(video_bytes: bytes, prompt: str = "Describe what is happening in this video in detail.") -> str:
    """Analyzes a video by extracting keyframes and sending them to Gemini Vision."""
    if not gemini_client:
        return "❌ Nebulae's video analyzer is offline."

    try:
        import cv2
        import numpy as np
    except ImportError:
        return "❌ Video analysis is not available on this server (missing system libraries)."

    temp_video_path = f"temp_video_{os.urandom(4).hex()}.mp4"
    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return "❌ Could not read video frames."
            
        # Extract up to 6 evenly spaced frames to keep token usage low but accurate
        num_frames = min(6, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        image_parts = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and compress to JPEG
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                image_parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
        
        cap.release()
        
        if not image_parts:
            return "❌ No frames could be extracted from the video."
            
        # Add the text prompt at the end
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

# ═══════════════════════════════════════════════════════════
# LOGO RECOGNITION
# ═══════════════════════════════════════════════════════════

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
    """
    Analyzes if the image is AIM's logo.
    Returns True if confidence is high, False otherwise.
    """
    if not gemini_client:
        return False
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=f"""
Analyze this image carefully. Does it match this description?

{AIM_LOGO_DESCRIPTION}

Answer ONLY with "YES" if it matches the logo, or "NO" if it doesn't.
Be strict - only say YES if it clearly matches the beam of light design.
""")
            ],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=10)
        )
        
        answer = response.text.strip().upper()
        return "YES" in answer
        
    except Exception as e:
        logger.error(f"Logo recognition error: {e}")
        return False