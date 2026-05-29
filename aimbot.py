"""
AIM Bot v4.0 — African Intelligence Model
Ultra-simple inline placeholder detection + professional tone
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, request, jsonify
from telegram import (
    Update, Bot, InlineQueryResultArticle, InputTextMessageContent
)
from telegram.constants import ParseMode
from supabase import create_client, Client
from google import genai
from google.genai import types

# ─── LOGGING ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aimbot")

# ─── CONFIG ───
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")

TELEGRAM_MAX_CHARS = 4096

# ─── DUPLICATE PREVENTION ───
_processed_update_ids: set[int] = set()
_MAX_PROCESSED_IDS = 200
_lock = threading.Lock()

def is_duplicate_update(update_id: int) -> bool:
    with _lock:
        if update_id in _processed_update_ids:
            return True
        _processed_update_ids.add(update_id)
        if len(_processed_update_ids) > _MAX_PROCESSED_IDS:
            ids_to_remove = list(_processed_update_ids)[:_MAX_PROCESSED_IDS // 2]
            for old_id in ids_to_remove:
                _processed_update_ids.discard(old_id)
        return False

# ─── INIT CLIENTS ───
app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase connected")
    except Exception as e:
        logger.error("❌ Supabase connection failed: %s", e)
else:
    logger.warning("⚠️ Supabase not configured")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ─── ASYNCIO EVENT LOOP (daemon thread) ───
_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()

def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

threading.Thread(target=_run_loop, daemon=True, name="async-loop").start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop)

# ─── SYSTEM PROMPT (PROFESSIONAL VERSION) ───
SYSTEM_PROMPT = """You are AIM — African Intelligence Model. You are a professional AI assistant built for Africans, by Africans.

Personality:
- Warm, respectful, and culturally aware
- Reference African culture and context when relevant
- Be helpful, patient, and empowering
- Use standard English only — Pidgin, slang, or informal dialects only wen initiated by the user
- Never use phrases like "The Empire is rising" or similar taglines

Rules:
- Keep responses concise but informative
- If you don't know something, say so honestly
- Never make up facts about Africa or Nigeria — use your knowledge or admit uncertainty
- Respect all users regardless of background
- Use emojis naturally but not excessively

Current date: {date}
""".format(date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

# ─── GEMINI API ───
async def get_gemini_response(user_text: str, user_id: str, chat_type: str, context: str = "") -> Optional[types.GenerateContentResponse]:
    if not gemini_client:
        return None

    try:
        prompt = user_text
        if context:
            prompt = f"Context from previous chats:\n{context}\n\nUser question: {user_text}"

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
        )
        return response
    except Exception as e:
        logger.error("Gemini error: %s", e)
        return None

# ─── TOPIC EXTRACTION ───
async def extract_topic(user_text: str, bot_response: str) -> str:
    if not gemini_client:
        return "general"

    topics = ["career", "finance", "tech", "sports", "health", "relationships", "politics", "entertainment", "education", "general"]
    prompt = f"Classify this conversation into ONE topic from this list: {', '.join(topics)}.\n\nUser: {user_text[:200]}\nAIM: {bot_response[:200]}\n\nReturn ONLY the topic word, nothing else."

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20)
        )
        topic = response.text.strip().lower() if response and response.text else "general"
        return topic if topic in topics else "general"
    except Exception:
        return "general"

# ─── MEMORY FUNCTIONS ───
async def save_chat_memory(user_id: str, username: str, message: str, response: str, chat_type: str, topic: str = "general"):
    if not supabase:
        return
    try:
        supabase.table("chat_memory").insert({
            "user_id": str(user_id),
            "username": username or "",
            "message": message[:2000],
            "response": response[:2000],
            "chat_type": chat_type,
            "topic": topic
        }).execute()
        logger.info("✅ Saved memory for user %s — topic: %s", user_id, topic)
    except Exception as e:
        logger.error("❌ Memory save failed: %s", e)

async def update_user_profile(user_id: str, username: str, topic: str):
    if not supabase:
        return
    try:
        existing = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if existing.data:
            profile = existing.data[0]
            topic_counts = profile.get("topic_counts", {})
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            supabase.table("user_profiles").update({
                "topic_counts": topic_counts,
                "total_chats": profile.get("total_chats", 0) + 1,
                "last_active": datetime.now(timezone.utc).isoformat()
            }).eq("user_id", str(user_id)).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": str(user_id),
                "username": username or "",
                "topic_counts": {topic: 1},
                "total_chats": 1
            }).execute()
    except Exception as e:
        logger.error("❌ Profile update failed: %s", e)

async def get_user_context(user_id: str, limit: int = 5) -> str:
    if not supabase:
        return ""
    try:
        rows = supabase.table("chat_memory").select("message, response, topic, created_at")\
            .eq("user_id", str(user_id)).order("created_at", desc=True).limit(limit).execute()
        if not rows.data:
            return ""
        context_parts = []
        for row in rows.data:
            context_parts.append(f"[{row.get('topic', 'general')}] User: {row['message']} | AIM: {row['response']}")
        return "\n".join(reversed(context_parts))
    except Exception:
        return ""

# ─── MEMORY SEARCH ───
async def search_memory(user_id: str) -> str:
    if not supabase:
        return "My memory is currently offline. Please try again later."
    try:
        profile_res = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if not profile_res.data:
            return "We haven't chatted before! Start a conversation so I can remember you."

        profile = profile_res.data[0]
        topic_counts = profile.get("topic_counts", {})
        total_chats = profile.get("total_chats", 0)

        memory_res = supabase.table("chat_memory").select("message, response, topic, created_at")\
            .eq("user_id", str(user_id)).order("created_at", desc=True).limit(10).execute()

        lines = [f"📊 Your Top Topics: {', '.join(f'{k} ({v}x)' for k, v in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3])}",
                 f"💬 Total Chats: {total_chats}", "", "📝 Recent Conversations:"]

        for i, row in enumerate(memory_res.data[:5], 1):
            emoji = {"career": "💼", "finance": "💰", "tech": "💻", "sports": "⚽", 
                     "health": "🏥", "relationships": "❤️", "politics": "🏛️", 
                     "entertainment": "🎬", "education": "📚"}.get(row.get("topic"), "💬")
            date = row.get("created_at", "")[:10] if row.get("created_at") else ""
            lines.append(f"{i}. {emoji} [{date}] {row['message'][:60]}...")

        lines.append("\nWant me to dive deeper? Just ask!")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Memory search error: %s", e)
        return "Memory search is having issues right now. Please try again later."

# ─── SEND MESSAGE ───
async def send_text_chunks(chat_id: int, text: str, reply_to: Optional[int] = None, message_id: Optional[int] = None):
    """Send or edit message."""
    if not bot:
        return

    try:
        if message_id:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text[:TELEGRAM_MAX_CHARS],
                parse_mode=ParseMode.HTML
            )
            logger.info("✅ Edited message %s in chat %s", message_id, chat_id)
        else:
            kwargs = {"chat_id": chat_id, "text": text[:TELEGRAM_MAX_CHARS], "parse_mode": ParseMode.HTML}
            if reply_to:
                kwargs["reply_to_message_id"] = reply_to
            await bot.send_message(**kwargs)
    except Exception as e:
        logger.error("Send/edit error: %s", e)
        try:
            if message_id:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS])
            else:
                await bot.send_message(chat_id=chat_id, text=text[:TELEGRAM_MAX_CHARS])
        except Exception as e2:
            logger.error("Fallback send failed: %s", e2)

# ─── INLINE QUERY HANDLER ───
async def handle_inline_query_async(inline_query):
    """Handle inline queries using native method."""
    query_id = inline_query.id
    query_text = inline_query.query.strip()
    user_id = inline_query.from_user.id if inline_query.from_user else ""

    logger.info("📨 Inline query from %s: '%s'", user_id, query_text)

    if not query_text or len(query_text) < 2:
        await bot.answer_inline_query(
            inline_query_id=query_id,
            results=[],
            cache_time=1
        )
        return

    # Try fast path: get Gemini answer within 8 seconds
    answer_text = None
    try:
        response = await asyncio.wait_for(
            get_gemini_response(query_text, user_id, "private"),
            timeout=8.0
        )
        if response and response.text:
            answer_text = response.text.strip()[:300]
    except asyncio.TimeoutError:
        logger.info("⏱️ Gemini timeout for inline query, using placeholder")
    except Exception as e:
        logger.error("Inline Gemini error: %s", e)

    if answer_text:
        # Return actual answer immediately
        result = InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title=f"AIM: {query_text[:30]}",
            description=answer_text[:100],
            input_message_content=InputTextMessageContent(
                message_text=f"🤖 <b>AIM says:</b>\n\n{answer_text}\n\n<i>Asked via @askaimbot</i>",
                parse_mode=ParseMode.HTML
            )
        )
    else:
        # Fallback: placeholder that triggers reply
        result = InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title=f"Ask AIM: {query_text[:40]}",
            description="Click to get AIM's answer",
            input_message_content=InputTextMessageContent(
                message_text=f"🤖 Asking AIM: {query_text}\n⏳ Processing...",
                parse_mode=ParseMode.HTML
            )
        )

    try:
        await bot.answer_inline_query(
            inline_query_id=query_id,
            results=[result],
            cache_time=0,
            is_personal=True
        )
        logger.info("✅ Inline query answered for %s", user_id)
    except Exception as e:
        logger.error("❌ Inline query answer failed: %s", e)

# ─── PROCESS INLINE ANSWER (BACKGROUND) ───
async def process_inline_answer(chat_id: int, message_id: int, query_text: str, user_id: str):
    """Process the inline query answer and send as REPLY."""
    logger.info("🔄 Processing inline answer for msg %s: '%s'", message_id, query_text)

    try:
        context = await get_user_context(user_id)
        response = await get_gemini_response(query_text, user_id, "private", context)

        if response and response.text:
            answer = response.text.strip()
            topic = await extract_topic(query_text, answer)
            await save_chat_memory(user_id, "", query_text, answer, "inline", topic)
            await update_user_profile(user_id, "", topic)

            final_text = f"🤖 <b>AIM says:</b>\n\n{answer}"
            await send_text_chunks(chat_id, final_text, reply_to=message_id)
            logger.info("✅ Inline answer sent as reply for msg %s", message_id)
        else:
            error_text = "🔥 AIM is experiencing high demand right now. Please try again in 30 seconds."
            await send_text_chunks(chat_id, error_text, reply_to=message_id)
    except Exception as e:
        logger.error("Inline answer processing error: %s", e)
        error_text = "🛠️ AIM's engine is warming up. Please try again in a few seconds."
        await send_text_chunks(chat_id, error_text, reply_to=message_id)

# ─── CHECK IF MESSAGE IS INLINE PLACEHOLDER ───
def is_inline_placeholder(text: str) -> tuple[bool, str]:
    """Ultra-simple check: does this look like an inline placeholder?

    Expected format from Telegram:
    🤖 Asking AIM: [question]
    ⏳ Processing...
    """
    if not text:
        return False, ""

    text_clean = text.strip()
    text_lower = text_clean.lower()

    # Log what we received for debugging
    logger.info("🔍 Checking if placeholder: '%s'", text_clean[:200].replace('\n', ' | '))

    # MUST contain "asking aim" (case insensitive)
    if "asking aim" not in text_lower:
        return False, ""

    # MUST contain "processing" or "thinking" (case insensitive)
    has_processing = "processing" in text_lower or "thinking" in text_lower
    if not has_processing:
        return False, ""

    # Extract the question: everything between "Asking AIM:" and the newline before "Processing"
    # Split by "Asking AIM:" (case insensitive)
    parts = text_clean.split(":", 1)
    if len(parts) < 2:
        return False, ""

    # parts[0] = "🤖 Asking AIM"
    # parts[1] = " tell him I never said...\n⏳ Processing..."
    after_prefix = parts[1].strip()

    # Remove everything from the first newline onward
    if "\n" in after_prefix:
        query = after_prefix.split("\n", 1)[0].strip()
    else:
        query = after_prefix.strip()

    # Clean up any remaining processing/thinking text
    query = query.replace("⏳", "").replace("Processing...", "").replace("processing...", "").replace("Thinking...", "").replace("thinking...", "").strip()

    if query:
        logger.info("🎯 PLACEHOLDER DETECTED! Query: '%s'", query)
        return True, query

    logger.info("⚠️ Found 'Asking AIM' + 'Processing' but no query extracted")
    return False, ""

# ─── MESSAGE PROCESSOR ───
async def handle_message_async(update: Update):
    """Process incoming messages."""
    if not update.message:
        return

    user = update.message.from_user
    chat = update.message.chat
    user_text = update.message.text or ""
    chat_type = chat.type if chat else "private"
    message_id = update.message.message_id

    if not user_text:
        await send_text_chunks(chat.id, "I can only read text messages for now.")
        return

    user_id = str(user.id)
    username = user.username or user.first_name or "User"

    logger.info("📩 Message from %s in %s: '%s'", user_id, chat_type, user_text[:100].replace('\n', ' | '))

    # Check if this is an inline placeholder that needs answering
    is_placeholder, query_text = is_inline_placeholder(user_text)
    if is_placeholder and query_text:
        logger.info("🔄 PROCESSING INLINE PLACEHOLDER for query: '%s'", query_text)
        await process_inline_answer(chat.id, message_id, query_text, user_id)
        return

    # Check for memory search keywords
    memory_keywords = ["what did we talk about", "what have we discussed", "remember our chats", 
                        "my memory", "our conversations", "what did i ask you"]
    if any(kw in user_text.lower() for kw in memory_keywords):
        memory_summary = await search_memory(user_id)
        await send_text_chunks(chat.id, memory_summary, reply_to=message_id)
        return

    # Group mention check
    if chat_type in ("group", "supergroup"):
        mention_found = "@askaimbot" in user_text.lower() or "askaimbot" in user_text.lower()
        reply_to_bot = False
        if update.message.reply_to_message and update.message.reply_to_message.from_user:
            reply_to_bot = update.message.reply_to_message.from_user.is_bot and update.message.reply_to_message.from_user.username == "askaimbot"

        if not mention_found and not reply_to_bot:
            logger.info("Ignoring group message (no @askaimbot)")
            return

        if "@askaimbot" in user_text.lower():
            user_text = user_text.lower().replace("@askaimbot", "").strip()
        elif "askaimbot" in user_text.lower():
            user_text = user_text.lower().replace("askaimbot", "").strip()

    # Normal message processing
    context = await get_user_context(user_id)
    response = await get_gemini_response(user_text, user_id, chat_type, context)

    if response and response.text:
        answer = response.text.strip()
        await send_text_chunks(chat.id, answer, reply_to=message_id)

        topic = await extract_topic(user_text, answer)
        await save_chat_memory(user_id, username, user_text, answer, chat_type, topic)
        await update_user_profile(user_id, username, topic)
    else:
        error_msg = "🔥 AIM is experiencing high demand right now. Please try again in 30 seconds."
        await send_text_chunks(chat.id, error_msg, reply_to=message_id)

# ─── WEBHOOK ROUTES ───
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "AIM Bot is live!",
        "version": "v4.0",
        "model": "African Intelligence Model",
        "features": ["memory", "topics", "inline_mode"]
    })

@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive updates from Telegram."""
    try:
        data = request.get_json(force=True)
        update_id = data.get("update_id")

        if update_id and is_duplicate_update(update_id):
            logger.info("Ignoring duplicate update_id: %s", update_id)
            return "OK", 200

        update = Update.de_json(data, bot)

        if update.inline_query:
            run_async(handle_inline_query_async(update.inline_query))
            return "OK", 200

        if update.callback_query:
            return "OK", 200

        if update.message:
            run_async(handle_message_async(update))

        return "OK", 200
    except Exception as e:
        logger.error("Webhook error: %s", e)
        return "Error", 500

@app.route("/set-webhook", methods=["GET"])
def set_webhook():
    if not bot or not WEBHOOK_URL:
        return jsonify({"error": "Bot or webhook URL not configured"}), 500
    try:
        url = f"{WEBHOOK_URL}/webhook"
        bot.set_webhook(url=url)
        return jsonify({"status": "Webhook set successfully!", "url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    if not bot:
        return jsonify({"error": "Bot not configured"}), 500
    try:
        bot.delete_webhook()
        return jsonify({"status": "Webhook deleted!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── DEBUG ENDPOINTS ───
@app.route("/debug/supabase", methods=["GET"])
def debug_supabase():
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500
    try:
        chat_rows = supabase.table("chat_memory").select("*", count="exact").execute()
        profile_rows = supabase.table("user_profiles").select("*", count="exact").execute()
        return jsonify({
            "status": "connected",
            "chat_memory_rows": chat_rows.count if hasattr(chat_rows, 'count') else len(chat_rows.data),
            "user_profiles_rows": profile_rows.count if hasattr(profile_rows, 'count') else len(profile_rows.data),
            "tables": ["chat_memory", "user_profiles", "auth_states"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memory/<user_id>", methods=["GET"])
def get_memory(user_id: str):
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500
    try:
        rows = supabase.table("chat_memory").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        return jsonify({"user_id": user_id, "chats": rows.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/profile/<user_id>", methods=["GET"])
def get_profile(user_id: str):
    if not supabase:
        return jsonify({"error": "Supabase not connected"}), 500
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        return jsonify({"user_id": user_id, "profile": rows.data[0] if rows.data else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── PRIVACY POLICY (uses external file) ───
@app.route("/privacy", methods=["GET"])
def privacy_policy():
    try:
        with open("privacy.html", "r", encoding="utf-8") as f:
            return f.read(), 200, {'Content-Type': 'text/html'}
    except FileNotFoundError:
        return "<h1>Privacy Policy</h1><p>Coming soon.</p>", 200, {'Content-Type': 'text/html'}

# ─── MAIN ───
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)