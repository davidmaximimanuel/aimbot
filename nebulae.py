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

# Safe OpenCV import
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV loaded successfully")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ OpenCV not available. Video analysis disabled.")

# Try to use PyMuPDF for better PDF text extraction (fallback)
try:
    import fitz  # PyMuPDF
    PDF_TEXT_AVAILABLE = True
    logger.info("✅ PyMuPDF loaded for document fallback")
except ImportError:
    PDF_TEXT_AVAILABLE = False
    logger.warning("⚠️ PyMuPDF not available. Using basic fallback.")

# Config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# =============================================
# 1. IMAGE ANALYSIS
# =============================================
async def analyze_image(image_bytes: bytes, prompt: str = "Describe this image in detail.") -> str:
    if not gemini_client:
        return "❌ Vision is currently offline."
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=1024)
        )
        return response.text or "No description available."
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        return "I couldn't analyze this image properly."

# =============================================
# 2. IMAGE GENERATION
# =============================================
async def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[bytes]:
    try:
        safe_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={width}&height={height}&model=flux&nologo=true&seed={os.urandom(4).hex()}"
        response = requests.get(url, timeout=60)
        if response.status_code == 200 and len(response.content) > 1000:
            return response.content
        return None
    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        return None

# =============================================
# 3. AUDIO
# =============================================
async def generate_audio(text: str, voice: str = "en-US-GuyNeural") -> Optional[bytes]:
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_buffer = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()
    except Exception as e:
        logger.error(f"Audio Error: {e}")
        return None

# =============================================
# 4. PDF GENERATION
# =============================================
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

# =============================================
# 5. DOCUMENT ANALYSIS (Improved)
# =============================================
async def analyze_document(file_bytes: bytes, mime_type: str, filename: str, prompt: str = "Summarize this document in detail.") -> str:
    if not gemini_client:
        return "❌ Document reader is offline."

    try:
        # Try Gemini native PDF support first
        if mime_type == "application/pdf":
            parts = [
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=prompt)
            ]
        else:
            # Text fallback
            text_content = file_bytes.decode('utf-8', errors='replace')[:20000]
            parts = [types.Part.from_text(text=f"Document: {filename}\n\n{text_content}\n\n{prompt}")]

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048)
        )
        return response.text or "No content extracted."

    except Exception as e:
        logger.warning(f"Gemini document failed: {e}. Trying manual fallback.")

        # Manual fallback for PDFs
        if PDF_TEXT_AVAILABLE and mime_type == "application/pdf":
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                if text.strip():
                    return f"**Extracted Text from {filename}**\n\n{text[:15000]}\n\n{prompt}"
            except Exception as fallback_e:
                logger.error(f"Manual PDF extraction failed: {fallback_e}")

        return "I couldn't read this document. Try a smaller PDF or text file."

# =============================================
# 6. VIDEO ANALYSIS
# =============================================
async def analyze_video(video_bytes: bytes, prompt: str = "Describe what is happening in this video.") -> str:
    if not CV2_AVAILABLE:
        return "❌ Video analysis unavailable (system libraries missing)."
    if not gemini_client:
        return "❌ Video analyzer offline."

    temp_path = f"temp_video_{os.urandom(4).hex()}.mp4"
    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return "Could not read video."

        num_frames = min(6, total_frames)
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

        image_parts = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                image_parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))

        cap.release()
        image_parts.append(types.Part.from_text(text=prompt))

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=image_parts,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=1024)
        )
        return response.text
    except Exception as e:
        logger.error(f"Video Error: {e}")
        return "I couldn't analyze this video."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# =============================================
# Helper Functions
# =============================================
def is_image_gen_request(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ["generate image", "create image", "make a picture", "draw me"])

def is_audio_request(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ["read this out loud", "say this", "text to speech", "make audio"])

# Logo recognition (unchanged)
AIM_LOGO_DESCRIPTION = """..."""  # Keep your original description

async def is_aim_logo(image_bytes: bytes) -> bool:
    if not gemini_client:
        return False
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), types.Part.from_text(text=f"Does this match AIM logo? {AIM_LOGO_DESCRIPTION} Answer YES or NO only.")],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=10)
        )
        return "YES" in response.text.strip().upper()
    except:
        return False