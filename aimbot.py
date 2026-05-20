import logging
import os
import asyncio
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from telegram import Update, Bot
from telegram.error import Forbidden
from telegram.request import HTTPXRequest
from google import genai
from google.genai import types
from supabase import create_client, Client

# ─── ENVIRONMENT ───
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()
WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").strip()

TELEGRAM_MAX_CHARS = 4096

# ─── SUPABASE CLIENT ───
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── GEMINI CLIENT ───
client: genai.Client | None = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# ─── LOGGING ───
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── SYSTEM PROMPT ───
SYSTEM_PROMPT = """
You are AIM (Africa's Intelligence Machine) — the first true AI built for Africa.
Created by Empire AI and David Emmanuel. You live on Telegram as @askaimbot.

WHO YOU ARE
- A witty, sharp, relatable Nigerian "smart friend" — never a stiff corporate bot.
- Gatekeeper of the AIM Empire: proud, warm, culturally grounded, globally aware.

CITIZENS (how to address users)
- For now, treat every user as a "Citizen" (full verification is coming later)
- Do NOT call anyone "Verified Citizen" unless they explicitly say they are verified.
- First message in a chat can be slightly more welcoming; ongoing chats stay natural.

LANGUAGE & PIDGIN
- Default: clear, smooth English — confident, conversational, not academic.
- Use Nigerian Pidgin ONLY when:
  (a) the user writes in Pidgin, OR
  (b) they ask for Pidgin, OR
  (c) they say they are from a Pidgin-speaking country and want that vibe.
- Never force Pidgin on formal, corporate, or clearly international users.
- Understand Naija slang when users use it; explain briefly if they seem lost.

TELEGRAM REPLIES
- Short and sharp: most replies under ~12 lines unless summarizing or fact-checking.
- Use line breaks; one idea per paragraph.
- Emojis: sparingly (🔥 🇳🇬 🚀 😂) — about 1–3 per message, not every sentence.

LOCAL CONTEXT (use when relevant, not every reply)
- Naija life: slang, football banter, Afrobeats, japa/diaspora talk, African tech.
- May 2026 vibes when topical: fuel (Abuja ~₦1,370/L; Dangote cuts ~₦899.50/L),
  Victor Osimhen transfer rumors, Unity Cup / Super Eagles (e.g. Prosper Obah).
- Prefer African angles (Punch, Vanguard, BBC Africa, The Cable) over generic Western takes.

FACT-CHECK MODE — trigger when they ask to verify, fact-check, "true?", "confirm", or "cap":
Use exactly this structure:

[✅ CONFIRMED] — established facts with broad agreement
[⚠️ UNVERIFIED] — rumors, single sources, or you lack proof
[❌ FALSE] — debunked or clearly wrong

VERDICT: [one sentence]
THE GIST:
• [evidence point 1 — name source type, e.g. "per Punch reporting"]
• [evidence point 2]
• [optional point 3]

If you cannot verify with confidence, choose UNVERIFIED and say what is missing.
Never invent quotes, URLs, or headlines.

THE ROAST
- Only for silly, empty, or clearly lazy questions — light, respectful, never cruel.
- Never roast: religion, tribe, gender, disability, trauma, poverty, or mental health.

BOUNDARIES
- No hate, scams, crime, or violence instructions.
- Not a lawyer or doctor — say so for legal/medical emergencies; urge real professionals.

EMPIRE PROTOCOL
- You serve Citizens of the AIM Empire with loyalty and humor.
- Be the smartest friend in the chat — helpful first, personality second.
"""

# ─── TOPIC EXTRACTION PROMPT ───
TOPIC_EXTRACTION_PROMPT = """
Analyze this conversation and classify it into EXACTLY ONE topic from this list:
career, finance, tech, sports, health, relationships, politics, entertainment, education, general

Rules:
- Return ONLY the topic word, nothing else.
- If multiple topics, pick the dominant one.
- "general" is the fallback.

Conversation:
User: {user_message}
AIM: {aim_response}

Topic:"""

# ─── HELPER FUNCTIONS ───

def chunk_telegram_text(text: str, max_len: int = TELEGRAM_MAX_CHARS) -> list[str]:
    if not text:
        return []
    parts = []
    i = 0
    while i < len(text):
        parts.append(text[i:i + max_len])
        i += max_len
    return parts


async def send_text_chunks(bot: Bot, chat_id: int, text: str) -> None:
    for chunk in chunk_telegram_text(text):
        await bot.send_message(chat_id=chat_id, text=chunk)


async def extract_topic(user_message: str, aim_response: str) -> str:
    """Use Gemini to extract topic from conversation."""
    if not client:
        return "general"

    try:
        prompt = TOPIC_EXTRACTION_PROMPT.format(
            user_message=user_message[:500],
            aim_response=aim_response[:500]
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent classification
                max_output_tokens=20
            ),
        )

        topic = response.text.strip().lower() if response.text else "general"
        valid_topics = {"career", "finance", "tech", "sports", "health", 
                       "relationships", "politics", "entertainment", "education", "general"}

        return topic if topic in valid_topics else "general"
    except Exception as e:
        logger.error("Topic extraction failed: %s", e)
        return "general"


async def save_chat_to_memory(user_id: int, username: str | None, 
                               message: str, response: str, chat_type: str) -> None:
    """Save chat with AI-extracted topic to Supabase."""
    if not supabase:
        logger.warning("Supabase not connected — skipping memory save")
        return

    try:
        # Extract topic using AI
        topic = await extract_topic(message, response)

        data = {
            "user_id": str(user_id),
            "username": username or "anonymous",
            "message": message[:2000],  # Limit size
            "response": response[:2000],
            "chat_type": chat_type,
            "topic": topic,
            "created_at": datetime.utcnow().isoformat()
        }

        supabase.table("chat_memory").insert(data).execute()
        logger.info("Saved chat memory for user %s — topic: %s", user_id, topic)

        # Update user profile with topic frequency
        await update_user_topic_profile(user_id, username, topic)

    except Exception as e:
        logger.error("Failed to save chat memory: %s", e)


async def update_user_topic_profile(user_id: int, username: str | None, topic: str) -> None:
    """Track what topics each user talks about most."""
    if not supabase:
        return

    try:
        # Check if user profile exists
        result = supabase.table("user_profiles")            .select("*")            .eq("user_id", str(user_id))            .execute()

        if result.data:
            # Update existing profile
            profile = result.data[0]
            topic_counts = profile.get("topic_counts", {}) or {}
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

            supabase.table("user_profiles")                .update({
                    "topic_counts": topic_counts,
                    "last_active": datetime.utcnow().isoformat(),
                    "total_chats": profile.get("total_chats", 0) + 1
                })                .eq("user_id", str(user_id))                .execute()
        else:
            # Create new profile
            supabase.table("user_profiles")                .insert({
                    "user_id": str(user_id),
                    "username": username or "anonymous",
                    "topic_counts": {topic: 1},
                    "total_chats": 1,
                    "preferred_language": "english",
                    "news_categories": ["general", "technology", "sports"],
                    "created_at": datetime.utcnow().isoformat(),
                    "last_active": datetime.utcnow().isoformat()
                })                .execute()

    except Exception as e:
        logger.error("Failed to update user profile: %s", e)


async def get_user_memory(user_id: int, limit: int = 5, topic: str | None = None) -> list[dict]:
    """Retrieve recent chat history, optionally filtered by topic."""
    if not supabase:
        return []

    try:
        query = supabase.table("chat_memory")            .select("*")            .eq("user_id", str(user_id))            .order("created_at", desc=True)            .limit(limit)

        if topic:
            query = query.eq("topic", topic)

        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.error("Failed to fetch memory: %s", e)
        return []


async def get_user_profile(user_id: int) -> dict | None:
    """Get user profile with topic preferences."""
    if not supabase:
        return None

    try:
        result = supabase.table("user_profiles")            .select("*")            .eq("user_id", str(user_id))            .execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error("Failed to fetch user profile: %s", e)
        return None


async def get_memory_context(user_id: int, user_message: str) -> str:
    """Build memory context string for Gemini prompt."""
    memory = await get_user_memory(user_id, limit=3)
    profile = await get_user_profile(user_id)

    context_parts = []

    # Add user interests from profile
    if profile and profile.get("topic_counts"):
        top_topics = sorted(
            profile["topic_counts"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        interests = ", ".join([t[0] for t in top_topics])
        context_parts.append(f"[User typically discusses: {interests}]")

    # Add recent memory
    if memory:
        context_parts.append("[Recent conversations:]")
        for m in memory:
            context_parts.append(f"Q: {m['message'][:100]}...")
            context_parts.append(f"A: {m['response'][:100]}...")
        context_parts.append("[End recent conversations]")

    return "\n".join(context_parts) if context_parts else ""


# ─── MESSAGE HANDLER ───

async def handle_message(update: Update):
    user_text = update.message.text
    chat_type = update.message.chat.type
    user = update.message.from_user
    bot_user = (await bot.get_me()).username or ""

    logger.info(
        "Incoming message chat_id=%s user=%s type=%s preview=%r",
        update.effective_chat.id,
        user.id,
        chat_type,
        (user_text or "")[:80],
    )

    # Group mention check
    if chat_type in ("group", "supergroup"):
        if bot_user:
            mention = f"@{bot_user.lower()}"
            if mention not in user_text.lower():
                logger.info("Ignoring group message (no @%s).", bot_user)
                return "OK"

    chat_id = update.effective_chat.id

    # Get memory context
    memory_context = await get_memory_context(user.id, user_text)

    # Build prompt with memory
    full_prompt = user_text
    if memory_context:
        full_prompt = f"{memory_context}\n\n--- Current Message ---\n{user_text}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=full_prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )

        if response.text:
            await send_text_chunks(bot, chat_id, response.text)
            # Save to memory with topic extraction (fire and forget)
            asyncio.create_task(save_chat_to_memory(
                user.id, user.username, user_text, response.text, chat_type
            ))
        else:
            await bot.send_message(
                chat_id=chat_id, 
                text="I hear you, but my mouth dry. Ask again?"
            )

    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "resource_exhausted" in error_msg:
            logger.warning("Gemini quota hit: %s", e)
            await bot.send_message(
                chat_id=chat_id,
                text="Citizen, the Empire's lines are busy! Abeg give me 1 minute make I rest.",
            )
        elif "404" in error_msg or "not_found" in error_msg:
            logger.warning("Gemini model error: %s", e)
            await bot.send_message(
                chat_id=chat_id,
                text="My line dey static — model no gree connect. Try again later.",
            )
        else:
            logger.exception("handle_message failed")
            await bot.send_message(
                chat_id=chat_id, text="Abeg wait small, my brain dey reset..."
            )

    return "OK"


# ─── FLASK APP ───
app = Flask(__name__)

# Initialize bot
bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest())


@app.route("/")
def health_check():
    return jsonify({
        "status": "AIM Bot is live! 🚀",
        "empire": "rising",
        "version": "v2.1 - Smart Memory",
        "features": ["topic_extraction", "user_profiles", "memory_context"]
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive updates from Telegram."""
    update = Update.de_json(request.get_json(force=True), bot)
    asyncio.run(handle_message(update))
    return "OK", 200


@app.route("/set-webhook", methods=["GET"])
def set_webhook():
    """Set Telegram webhook to this URL."""
    if not WEBHOOK_URL:
        return jsonify({"error": "WEBHOOK_URL not set"}), 400

    success = asyncio.run(bot.set_webhook(url=f"{WEBHOOK_URL}/webhook"))
    if success:
        return jsonify({"status": "Webhook set successfully!", "url": WEBHOOK_URL})
    return jsonify({"error": "Failed to set webhook"}), 500


@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    """Remove webhook."""
    success = asyncio.run(bot.delete_webhook())
    if success:
        return jsonify({"status": "Webhook deleted!"})
    return jsonify({"error": "Failed to delete webhook"}), 500


@app.route("/memory/<user_id>", methods=["GET"])
def get_memory_api(user_id):
    """API endpoint to view user memory (for debugging)."""
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500

    try:
        result = supabase.table("chat_memory")            .select("*")            .eq("user_id", user_id)            .order("created_at", desc=True)            .limit(20)            .execute()
        return jsonify({"user_id": user_id, "memory": result.data or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/profile/<user_id>", methods=["GET"])
def get_profile_api(user_id):
    """API endpoint to view user profile."""
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500

    try:
        result = supabase.table("user_profiles")            .select("*")            .eq("user_id", user_id)            .execute()
        return jsonify({"user_id": user_id, "profile": result.data[0] if result.data else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        raise SystemExit("Set TELEGRAM_TOKEN and GEMINI_API_KEY environment variables.")

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))