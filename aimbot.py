import logging
import os
import asyncio
import json
import secrets
import re
import threading
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
LOGTO_ENDPOINT = (os.getenv("LOGTO_ENDPOINT") or "").strip()
LOGTO_CLIENT_ID = (os.getenv("LOGTO_CLIENT_ID") or "").strip()
LOGTO_CLIENT_SECRET = (os.getenv("LOGTO_CLIENT_SECRET") or "").strip()

TELEGRAM_MAX_CHARS = 4096

# ─── DUPLICATE PREVENTION ───
_processed_update_ids: set[int] = set()
_MAX_PROCESSED_IDS = 200
_lock = threading.Lock()

def is_duplicate_update(update_id: int) -> bool:
    """Check if we've already processed this Telegram update."""
    with _lock:
        if update_id in _processed_update_ids:
            return True
        _processed_update_ids.add(update_id)
        if len(_processed_update_ids) > _MAX_PROCESSED_IDS:
            ids_to_remove = list(_processed_update_ids)[:_MAX_PROCESSED_IDS // 2]
            for old_id in ids_to_remove:
                _processed_update_ids.discard(old_id)
        return False

# ─── SUPABASE CLIENT ───
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")

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

# ─── SYSTEM PROMPT (African Intelligence Model) ───
SYSTEM_PROMPT = """
You are AIM (African Intelligence Model) — the first true AI built for Africa.
Created by Empire AI and David Emmanuel. You live on Telegram as @askaimbot.

WHO YOU ARE
- A witty, sharp, relatable Nigerian "smart friend" — never a stiff corporate bot.
- Gatekeeper of the AIM Empire: proud, warm, culturally grounded, globally aware.
- You are an AFRICAN INTELLIGENCE MODEL — built by Africans, for Africans, representing African excellence in AI.

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
- You are the AFRICAN INTELLIGENCE MODEL — represent the continent with pride! 🇳🇬
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

# ─── MEMORY SEARCH DETECTION ───
MEMORY_SEARCH_PATTERNS = [
    r"what did we talk about",
    r"what did we discuss",
    r"remember when",
    r"do you remember",
    r"what was our last",
    r"tell me about my",
    r"summarize our chats",
    r"what have we discussed",
    r"my previous questions",
    r"topics we covered",
    r"last week|yesterday|last time",
]

def is_memory_search_query(text: str) -> bool:
    """Check if user is asking about past conversations."""
    if not text:
        return False
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in MEMORY_SEARCH_PATTERNS)

# ─── THREAD-SAFE EVENT LOOP ───
_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_loop_thread.start()

def run_async(coro):
    """Run coroutine in the persistent event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=60)

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


async def send_text_chunks_async(bot: Bot, chat_id: int, text: str) -> None:
    for chunk in chunk_telegram_text(text):
        await bot.send_message(chat_id=chat_id, text=chunk)


def send_text_chunks(bot: Bot, chat_id: int, text: str) -> None:
    """Synchronous wrapper for sending chunks."""
    run_async(send_text_chunks_async(bot, chat_id, text))


async def extract_topic_async(user_message: str, aim_response: str) -> str:
    """Async topic extraction."""
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
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20),
        )
        topic = response.text.strip().lower() if response.text else "general"
        valid_topics = {"career", "finance", "tech", "sports", "health", 
                       "relationships", "politics", "entertainment", "education", "general"}
        return topic if topic in valid_topics else "general"
    except Exception as e:
        logger.error("Topic extraction failed: %s", e)
        return "general"


def extract_topic(user_message: str, aim_response: str) -> str:
    """Synchronous wrapper."""
    return run_async(extract_topic_async(user_message, aim_response))


def save_chat_sync(user_id: str, username: str | None, message: str, 
                   response: str, chat_type: str, topic: str) -> bool:
    """Synchronous memory save."""
    if not supabase:
        logger.error("❌ Supabase not connected")
        return False
    try:
        data = {
            "user_id": user_id,
            "username": username or "anonymous",
            "message": message[:2000],
            "response": response[:2000],
            "chat_type": chat_type,
            "topic": topic,
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("chat_memory").insert(data).execute()
        logger.info("✅ Saved memory for user %s — topic: %s", user_id, topic)
        return True
    except Exception as e:
        logger.error("❌ Failed to save memory: %s", e)
        return False


def update_profile_sync(user_id: str, username: str | None, topic: str) -> bool:
    """Synchronous profile update."""
    if not supabase:
        return False
    try:
        result = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        if result.data:
            profile = result.data[0]
            topic_counts = profile.get("topic_counts", {}) or {}
            if isinstance(topic_counts, str):
                topic_counts = json.loads(topic_counts)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            supabase.table("user_profiles").update({
                "topic_counts": topic_counts,
                "last_active": datetime.utcnow().isoformat(),
                "total_chats": profile.get("total_chats", 0) + 1
            }).eq("user_id", user_id).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": user_id,
                "username": username or "anonymous",
                "topic_counts": {topic: 1},
                "total_chats": 1,
                "preferred_language": "english",
                "news_categories": ["general", "technology", "sports"],
                "created_at": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat()
            }).execute()
        return True
    except Exception as e:
        logger.error("❌ Failed to update profile: %s", e)
        return False


def save_chat_to_memory(user_id: int, username: str | None, 
                        message: str, response: str, chat_type: str) -> None:
    """Save chat with topic extraction."""
    topic = extract_topic(message, response)
    user_id_str = str(user_id)
    if save_chat_sync(user_id_str, username, message, response, chat_type, topic):
        update_profile_sync(user_id_str, username, topic)


def get_user_memory_sync(user_id: int, limit: int = 10, topic: str | None = None,
                          days: int | None = None) -> list[dict]:
    """Synchronous memory retrieval."""
    if not supabase:
        return []
    try:
        query = supabase.table("chat_memory").select("*").eq("user_id", str(user_id)).order("created_at", desc=True).limit(limit)
        if topic:
            query = query.eq("topic", topic)
        if days:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = query.gte("created_at", cutoff)
        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.error("Failed to fetch memory: %s", e)
        return []


def get_user_profile_sync(user_id: int) -> dict | None:
    """Synchronous profile retrieval."""
    if not supabase:
        return None
    try:
        result = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error("Failed to fetch profile: %s", e)
        return None


def build_memory_summary(user_id: int, query: str = "") -> str:
    """Build memory summary."""
    memory = get_user_memory_sync(user_id, limit=15)
    profile = get_user_profile_sync(user_id)

    if not memory and not profile:
        return "We haven't chatted much yet, Citizen! Start asking me things and I'll remember. 🧠"

    summary_parts = []
    if profile and profile.get("topic_counts"):
        topic_counts = profile["topic_counts"]
        if isinstance(topic_counts, str):
            topic_counts = json.loads(topic_counts)
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_list = ", ".join([f"{t[0]} ({t[1]}x)" for t in top_topics])
        summary_parts.append(f"📊 **Your Top Topics:** {topic_list}")
        summary_parts.append(f"💬 **Total Chats:** {profile.get('total_chats', 0)}")

    if memory:
        summary_parts.append("\n📝 **Recent Conversations:**")
        for i, m in enumerate(memory[:5], 1):
            date_str = m.get("created_at", "")[:10]
            topic_emoji = {
                "career": "💼", "finance": "💰", "tech": "💻",
                "sports": "⚽", "health": "🏥", "relationships": "❤️",
                "politics": "🏛️", "entertainment": "🎬", "education": "📚",
                "general": "💬"
            }.get(m.get("topic", "general"), "💬")
            msg_preview = m["message"][:60] + "..." if len(m["message"]) > 60 else m["message"]
            summary_parts.append(f"{i}. {topic_emoji} [{date_str}] {msg_preview}")

    summary_parts.append("\n🇳🇬 Want me to dive deeper into any of these topics? Just ask!")
    return "\n".join(summary_parts)


def get_memory_context(user_id: int, user_message: str) -> str:
    """Build memory context for Gemini."""
    memory = get_user_memory_sync(user_id, limit=3)
    profile = get_user_profile_sync(user_id)
    context_parts = []

    if profile and profile.get("topic_counts"):
        topic_counts = profile["topic_counts"]
        if isinstance(topic_counts, str):
            topic_counts = json.loads(topic_counts)
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        interests = ", ".join([t[0] for t in top_topics])
        context_parts.append(f"[User typically discusses: {interests}]")

    if memory:
        context_parts.append("[Recent conversations:]")
        for m in memory:
            context_parts.append(f"Q: {m['message'][:100]}...")
            context_parts.append(f"A: {m['response'][:100]}...")
        context_parts.append("[End recent conversations]")

    return "\n".join(context_parts) if context_parts else ""


# ─── BACKGROUND MESSAGE PROCESSOR ───

def process_message_background(update_data: dict, bot_instance: Bot):
    """Process message in background thread — webhook returns instantly."""
    try:
        update = Update.de_json(update_data, bot_instance)
        if not update.message:
            return

        user_text = update.message.text
        chat_type = update.message.chat.type if update.message.chat else "private"
        user = update.message.from_user

        if not user:
            return

        # Group mention check
        if chat_type in ("group", "supergroup"):
            if user_text and "@askaimbot" not in user_text.lower():
                logger.info("Ignoring group message (no @askaimbot)")
                return

        chat_id = update.effective_chat.id

        # Check if memory search
        if is_memory_search_query(user_text):
            logger.info("🔍 Memory search for user %s", user.id)
            memory_summary = build_memory_summary(user.id, user_text or "")
            send_text_chunks(bot_instance, chat_id, memory_summary)
            return

        # Normal chat with Gemini
        if not user_text:
            send_text_chunks(bot_instance, chat_id, "I can only read text messages for now, Citizen! 📝")
            return

        memory_context = get_memory_context(user.id, user_text)
        full_prompt = f"{memory_context}\n\n--- Current Message ---\n{user_text}" if memory_context else user_text

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=full_prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )

            if response.text:
                send_text_chunks(bot_instance, chat_id, response.text)
                # Save to memory AFTER successful send
                try:
                    save_chat_to_memory(user.id, user.username, user_text, response.text, chat_type)
                except Exception as mem_e:
                    logger.error("Memory save failed (non-critical): %s", mem_e)
            else:
                send_text_chunks(bot_instance, chat_id, "I hear you, but my mouth dry. Ask again?")

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                logger.warning("Gemini quota hit: %s", e)
                send_text_chunks(bot_instance, chat_id, "Citizen, the Empire's lines are busy! Abeg give me 1 minute.")
            elif "404" in error_msg or "not_found" in error_msg:
                logger.warning("Gemini model error: %s", e)
                send_text_chunks(bot_instance, chat_id, "My line dey static. Try again later.")
            else:
                logger.exception("Gemini generation failed")
                send_text_chunks(bot_instance, chat_id, "Abeg wait small, my brain dey reset...")

    except Exception as e:
        logger.exception("Background processing failed: %s", e)


# ─── FLASK APP ───
app = Flask(__name__)

# Initialize bot with new event loop
bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest())


@app.route("/")
def health_check():
    return jsonify({
        "status": "AIM Bot is live! 🚀",
        "empire": "rising",
        "version": "v3.1 - African Intelligence Model + Async Webhook",
        "features": ["topic_extraction", "user_profiles", "memory_context", "memory_search", "duplicate_prevention", "async_webhook", "logto_auth_ready"],
        "supabase_connected": supabase is not None
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive updates — return 200 OK instantly, process in background."""
    try:
        data = request.get_json(force=True)
        update_id = data.get("update_id")

        # INSTANTLY reject duplicates
        if update_id and is_duplicate_update(update_id):
            logger.info("Ignoring duplicate update_id: %s", update_id)
            return "OK", 200

        # Return 200 OK immediately — process in background thread
        threading.Thread(
            target=process_message_background,
            args=(data, bot),
            daemon=True
        ).start()

        return "OK", 200
    except Exception as e:
        logger.error("Webhook error: %s", e)
        return "Error", 500


@app.route("/set-webhook", methods=["GET"])
def set_webhook():
    if not WEBHOOK_URL:
        return jsonify({"error": "WEBHOOK_URL not set"}), 400
    try:
        success = run_async(bot.set_webhook(url=f"{WEBHOOK_URL}/webhook"))
        if success:
            return jsonify({"status": "Webhook set!", "url": WEBHOOK_URL})
        return jsonify({"error": "Failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    try:
        success = run_async(bot.delete_webhook())
        if success:
            return jsonify({"status": "Webhook deleted!"})
        return jsonify({"error": "Failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/<user_id>", methods=["GET"])
def get_memory_api(user_id):
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500
    try:
        result = supabase.table("chat_memory").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        return jsonify({"user_id": user_id, "count": len(result.data or []), "memory": result.data or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/profile/<user_id>", methods=["GET"])
def get_profile_api(user_id):
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500
    try:
        result = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        return jsonify({"user_id": user_id, "profile": result.data[0] if result.data else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/supabase", methods=["GET"])
def debug_supabase():
    if not supabase:
        return jsonify({"error": "Supabase not connected", "env_check": {"SUPABASE_URL_set": bool(SUPABASE_URL), "SUPABASE_KEY_set": bool(SUPABASE_KEY)}}), 500
    try:
        chat_result = supabase.table("chat_memory").select("count", count="exact").execute()
        profile_result = supabase.table("user_profiles").select("count", count="exact").execute()
        return jsonify({"status": "connected", "chat_memory_rows": getattr(chat_result, 'count', 'unknown'), "user_profiles_rows": getattr(profile_result, 'count', 'unknown')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── LOGTO AUTH ROUTES ───

@app.route("/auth/link/<telegram_user_id>", methods=["GET"])
def generate_logto_link(telegram_user_id):
    if not all([LOGTO_ENDPOINT, LOGTO_CLIENT_ID, WEBHOOK_URL]):
        return jsonify({"error": "Logto not fully configured"}), 500
    state = secrets.token_urlsafe(32)
    if supabase:
        try:
            supabase.table("auth_states").insert({"state": state, "telegram_user_id": telegram_user_id, "created_at": datetime.utcnow().isoformat(), "used": False}).execute()
        except Exception as e:
            logger.error("Auth state error: %s", e)
    redirect_uri = f"{WEBHOOK_URL}/auth/callback"
    auth_url = f"{LOGTO_ENDPOINT}/oidc/auth?response_type=code&client_id={LOGTO_CLIENT_ID}&redirect_uri={redirect_uri}&state={state}&scope=openid profile email"
    return jsonify({"auth_url": auth_url, "state": state})


@app.route("/auth/callback", methods=["GET"])
def logto_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    if not code or not state:
        return jsonify({"error": "Missing code or state"}), 400
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500
    try:
        result = supabase.table("auth_states").select("*").eq("state", state).eq("used", False).execute()
        if not result.data:
            return jsonify({"error": "Invalid state"}), 400
        telegram_user_id = result.data[0]["telegram_user_id"]
        supabase.table("auth_states").update({"used": True}).eq("state", state).execute()
        supabase.table("user_profiles").update({"logto_linked": True, "logto_linked_at": datetime.utcnow().isoformat()}).eq("user_id", telegram_user_id).execute()
        return jsonify({"status": "success", "message": f"Linked! 🎉"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── PRIVACY POLICY PAGE ───

@app.route("/privacy", methods=["GET"])
def privacy_policy():
    try:
        with open("privacy.html", "r", encoding="utf-8") as f:
            return f.read(), 200, {'Content-Type': 'text/html'}
    except FileNotFoundError:
        return "<h1>Privacy Policy</h1><p>Coming soon.</p>", 200, {'Content-Type': 'text/html'}


if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        raise SystemExit("Set TELEGRAM_TOKEN and GEMINI_API_KEY")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))