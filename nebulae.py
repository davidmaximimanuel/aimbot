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
    """Converts text to speech using Microsoft Edge TTS and returns MP3 bytes."""
    try:
        # Truncate text to avoid edge-tts limits
        text = text[:3000].strip()
        if not text:
            return None

        communicate   = edge_tts.Communicate(text, voice)
        audio_buffer  = BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            logger.error("Nebulae Audio: edge-tts returned empty bytes")
            return None

        logger.info("Nebulae Audio: generated %d bytes", len(audio_bytes))
        return audio_bytes
    except Exception as e:
        logger.error("Nebulae Audio Error: %s", e)
        return None


# ════════════════════════════════════════
# 4. PDF GENERATION (Free via ReportLab)
# ════════════════════════════════════════
def generate_pdf(title: str, content: str) -> bytes:
    """Generates a well-formatted PDF with proper text wrapping."""
    import textwrap
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 72
    max_width = width - 2 * margin  # usable width in points

    def new_page():
        c.showPage()
        c.setFont("Helvetica", 11)
        return height - margin

    # ── Title ──
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, height - 60, title[:80])
    c.setLineWidth(0.5)
    c.line(margin, height - 68, width - margin, height - 68)

    # ── Body ──
    c.setFont("Helvetica", 11)
    y = height - 90
    line_height = 16

    for paragraph in content.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            y -= line_height // 2  # blank line gap
            if y < margin:
                y = new_page()
            continue

        # Wrap long paragraphs at ~95 chars to fit the page width
        wrapped = textwrap.wrap(paragraph, width=95)
        for line in wrapped:
            if y < margin + line_height:
                y = new_page()
            c.drawString(margin, y, line)
            y -= line_height

        y -= 4  # small gap between paragraphs

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
        # Save video bytes to a temp file so OpenCV can read it
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