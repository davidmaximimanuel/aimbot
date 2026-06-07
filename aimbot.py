"""
AIM Bot v5.0 — African Intelligence Model
Smart Memory + User Preferences + Professional Tone
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import threading
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

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

# ─── BASE SYSTEM PROMPT ───# ─── BASE SYSTEM PROMPT ───
BASE_SYSTEM_PROMPT = """You are AIM — African Intelligence Model. You are a professional AI assistant built for Africans, by Africans.

Personality:
- Warm, respectful, and culturally aware
- Reference African culture and context when relevant
- Be helpful, patient, and empowering
- Use standard English only — Pidgin, slang, or informal dialects only when initiated by the user
- Never use phrases like "The Empire is rising" or similar taglines

Rules:
- Keep responses concise but informative
- If you don't know something, say so honestly
- Never make up facts about Africa or Nigeria — use your knowledge or admit uncertainty
- Respect all users regardless of background
- Use emojis naturally but not excessively

TIME AWARENESS:
- Current time and date: {datetime_info}
- Use this time context naturally in your responses
- If it's late night (10 PM - 6 AM), you may gently suggest rest when appropriate
- If it's morning, you can say "Good morning" naturally
- Reference the time only when relevant to the conversation
"""

# ─── USER PREFERENCE INJECTION ───
async def get_user_profile_data(user_id: str) -> dict:
    """Fetch user profile from Supabase."""
    if not supabase:
        return {}
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if rows.data:
            return rows.data[0]
        return {}
    except Exception as e:
        logger.error("Profile fetch error: %s", e)
        return {}

def build_enhanced_prompt(user_text: str, user_id: str, profile: dict, context: str = "") -> str:
    """Build full prompt with user preferences and memory context."""

    # Get current time in Lagos (WAT = UTC+1)
    wat_offset = timedelta(hours=1)
    wat_timezone = timezone(wat_offset)
    now_wat = datetime.now(wat_timezone)
    
    # Format datetime info
    date_str = now_wat.strftime("%A, %B %d, %Y")  # "Sunday, June 07, 2026"
    time_str = now_wat.strftime("%I:%M %p")  # "02:47 AM"
    
    # Determine time of day
    hour = now_wat.hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    
    # Build datetime info string
    datetime_info = f"{time_str} WAT, {date_str} ({time_of_day})"

    # Start with base system prompt
    prompt_parts = [BASE_SYSTEM_PROMPT.format(datetime_info=datetime_info)]

    # Start with base system prompt
    prompt_parts = [BASE_SYSTEM_PROMPT.format(date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]

    # Add user preferences section
    pref_language = profile.get("preferred_language", "english")
    timezone_str = profile.get("timezone", "Africa/Lagos")
    topic_counts = profile.get("topic_counts", {})
    total_chats = profile.get("total_chats", 0)

    pref_lines = ["\n--- USER PREFERENCES ---"]
    pref_lines.append(f"- User ID: {user_id}")
    pref_lines.append(f"- Preferred language: {pref_language}")
    pref_lines.append(f"- Timezone: {timezone_str}")
    pref_lines.append(f"- Total chats together: {total_chats}")

    if topic_counts:
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        pref_lines.append(f"- Top interests: {', '.join(f'{k}({v})' for k, v in top_topics)}")

    # CRITICAL: Language instruction based on preference
    if pref_language.lower() == "english":
        pref_lines.append("- LANGUAGE RULE: Respond in standard English ONLY. Do NOT use Pidgin, Nigerian slang, or informal dialects unless the user explicitly asks you to.")

    pref_lines.append("--- END PREFERENCES ---\n")
    prompt_parts.append("\n".join(pref_lines))

    # Add memory context
    if context:
        prompt_parts.append(f"\n--- RELEVANT MEMORY ---\n{context}\n--- END MEMORY ---\n")

    # Add current question
    prompt_parts.append(f"\nUSER QUESTION: {user_text}")

    return "\n".join(prompt_parts)

# ─── GEMINI API ───
async def get_gemini_response(user_text: str, user_id: str, chat_type: str, profile: dict = None, context: str = "") -> Optional[types.GenerateContentResponse]:
    if not gemini_client:
        return None

    try:
        if profile is None:
            profile = await get_user_profile_data(user_id)

        prompt = build_enhanced_prompt(user_text, user_id, profile, context)

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
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

async def get_user_context(user_id: str, limit: int = 15) -> str:
    """Get last N chats for context injection."""
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

async def get_relevant_context(user_id: str, query_text: str, limit: int = 15) -> str:
    """Get context sorted by relevance to query (topic match + recency)."""
    if not supabase:
        return ""
    try:
        # Get last 30 chats
        rows = supabase.table("chat_memory").select("message, response, topic, created_at")\
            .eq("user_id", str(user_id)).order("created_at", desc=True).limit(30).execute()
        if not rows.data:
            return ""

        # Score each chat by relevance
        query_lower = query_text.lower()
        keyword_topics = {
            "space": ["tech", "science"],
            "nigeria": ["general", "politics", "entertainment"],
            "money": ["finance"],
            "job": ["career"],
            "health": ["health"],
            "love": ["relationships"],
            "sport": ["sports"],
            "music": ["entertainment"],
            "school": ["education"],
            "code": ["tech"],
            "programming": ["tech"],
            "ai": ["tech"],
            "goat": ["general", "education"],
            "animal": ["general", "education"]
        }

        # Find matching topics for query
        matched_topics = set()
        for keyword, topics in keyword_topics.items():
            if keyword in query_lower:
                matched_topics.update(topics)

        scored = []
        for row in rows.data:
            score = 0
            # Recency score (newer = higher)
            age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))).days
            score += max(0, 30 - age_days)

            # Topic match score
            if row.get("topic") in matched_topics:
                score += 50

            # Keyword match in message/response
            msg_resp = f"{row['message']} {row['response']}".lower()
            for word in query_lower.split():
                if len(word) > 3 and word in msg_resp:
                    score += 10

            scored.append((score, row))

        # Sort by score, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top_rows = scored[:limit]

        context_parts = []
        for _, row in top_rows:
            context_parts.append(f"[{row.get('topic', 'general')}] User: {row['message']} | AIM: {row['response']}")

        return "\n".join(context_parts)
    except Exception as e:
        logger.error("Relevant context error: %s", e)
        return await get_user_context(user_id, limit)

# ─── SMART MEMORY SEARCH ───
def is_memory_search_query(user_text: str) -> bool:
    """Detect if user is asking about past conversations."""
    if not user_text:
        return False

    memory_keywords = [
        "what did we talk about",
        "what have we discussed",
        "remember our chats",
        "my memory",
        "our conversations",
        "what did i ask you",
        "what were we talking about",
        "we were talking about",
        "we discussed",
        "tell me about",
        "what did we say about",
        "remember when",
        "do you remember",
        "what about that time",
        "didn't we talk about"
    ]

    text_lower = user_text.lower()
    return any(kw in text_lower for kw in memory_keywords)

def extract_search_keywords(user_text: str) -> list:
    """Extract keywords from memory search query."""
    # Remove common phrases
    clean = user_text.lower()
    for phrase in ["what did we talk about", "what were we talking about", "we were talking about",
                   "tell me about", "what did we say about", "do you remember", "remember when",
                   "what about", "didn't we talk about", "what have we discussed"]:
        clean = clean.replace(phrase, "")

    # Remove punctuation and split
    clean = re.sub(r'[^\w\s]', ' ', clean)
    words = [w.strip() for w in clean.split() if len(w.strip()) > 2]

    # Filter out stop words
    stop_words = {"the", "and", "about", "were", "did", "have", "what", "when", "that", "this", "with", "for", "from", "you", "are", "was", "is", "it", "we", "our", "me", "my", "i", "a", "an", "to", "of", "in", "on", "at", "be", "been", "being", "do", "does", "did", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "shall", "say", "said", "get", "got", "go", "went", "come", "came", "know", "knew", "think", "thought", "take", "took", "see", "saw", "want", "wanted", "use", "used", "find", "found", "give", "gave", "tell", "told", "ask", "asked", "work", "worked", "seem", "seemed", "feel", "felt", "try", "tried", "leave", "left", "call", "called", "good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right", "big", "high", "different", "small", "large", "next", "early", "young", "important", "few", "public", "bad", "same", "able"}

    keywords = [w for w in words if w not in stop_words]
    return keywords

async def search_memory_by_keyword(user_id: str, query_text: str) -> str:
    """Search memory for specific topics/keywords."""
    if not supabase:
        return "My memory is currently offline. Please try again later."

    try:
        keywords = extract_search_keywords(query_text)
        logger.info("🔍 Memory search keywords: %s", keywords)

        if not keywords:
            # Fallback to general memory summary
            return await search_memory(user_id)

        # Try keyword search across messages and responses
        all_results = []

        for keyword in keywords[:3]:  # Check top 3 keywords
            # Search in messages
            msg_results = supabase.table("chat_memory").select("*")\
                .eq("user_id", str(user_id))\
                .ilike("message", f"%{keyword}%")\
                .order("created_at", desc=True).limit(5).execute()

            # Search in responses
            resp_results = supabase.table("chat_memory").select("*")\
                .eq("user_id", str(user_id))\
                .ilike("response", f"%{keyword}%")\
                .order("created_at", desc=True).limit(5).execute()

            all_results.extend(msg_results.data)
            all_results.extend(resp_results.data)

        # Also search by topic inference
        topic_map = {
            "space": "tech", "nigeria": "general", "money": "finance", "job": "career",
            "health": "health", "love": "relationships", "sport": "sports", "music": "entertainment",
            "school": "education", "code": "tech", "programming": "tech", "ai": "tech",
            "goat": "general", "animal": "general", "bleat": "general"
        }

        for keyword in keywords:
            if keyword in topic_map:
                topic_results = supabase.table("chat_memory").select("*")\
                    .eq("user_id", str(user_id))\
                    .eq("topic", topic_map[keyword])\
                    .order("created_at", desc=True).limit(5).execute()
                all_results.extend(topic_results.data)

        # Deduplicate by ID
        seen_ids = set()
        unique_results = []
        for row in all_results:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                unique_results.append(row)

        # Sort by date
        unique_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        if not unique_results:
            return f"I don't recall us discussing {' '.join(keywords)}. Would you like to start a new conversation about it?"

        # Build response
        lines = [f"🔍 I found {len(unique_results)} conversation(s) about that:"]

        for i, row in enumerate(unique_results[:5], 1):
            emoji = {"career": "💼", "finance": "💰", "tech": "💻", "sports": "⚽",
                     "health": "🏥", "relationships": "❤️", "politics": "🏛️",
                     "entertainment": "🎬", "education": "📚"}.get(row.get("topic"), "💬")
            date = row.get("created_at", "")[:10] if row.get("created_at") else ""
            msg_preview = row["message"][:80] if row["message"] else ""
            resp_preview = row["response"][:120] if row["response"] else ""
            lines.append(f'\n{i}. {emoji} [{date}] You: "{msg_preview}..."')
            lines.append(f'   AIM: "{resp_preview}..."')

        return "\n".join(lines)
    except Exception as e:
        logger.error("Keyword memory search error: %s", e)
        return "I found something in my memory, but I'm having trouble organizing it. Please try again."

async def search_memory(user_id: str) -> str:
    """General memory summary."""
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
        profile = await get_user_profile_data(user_id)
        response = await asyncio.wait_for(
            get_gemini_response(query_text, user_id, "private", profile),
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
        profile = await get_user_profile_data(user_id)
        context = await get_relevant_context(user_id, query_text)
        response = await get_gemini_response(query_text, user_id, "private", profile, context)

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
def is_inline_placeholder(text: str) -> Tuple[bool, str]:
    """Ultra-simple check: does this look like an inline placeholder?"""
    if not text:
        return False, ""

    text_clean = text.strip()
    text_lower = text_clean.lower()

    logger.info("🔍 Checking if placeholder: '%s'", text_clean[:200].replace('\n', ' | '))

    if "asking aim" not in text_lower:
        return False, ""

    has_processing = "processing" in text_lower or "thinking" in text_lower
    if not has_processing:
        return False, ""

    parts = text_clean.split(":", 1)
    if len(parts) < 2:
        return False, ""

    after_prefix = parts[1].strip()

    if "\n" in after_prefix:
        query = after_prefix.split("\n", 1)[0].strip()
    else:
        query = after_prefix.strip()

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

    # Fetch user profile for preferences
    profile = await get_user_profile_data(user_id)

    # Check if this is an inline placeholder that needs answering
    is_placeholder, query_text = is_inline_placeholder(user_text)
    if is_placeholder and query_text:
        logger.info("🔄 PROCESSING INLINE PLACEHOLDER for query: '%s'", query_text)
        await process_inline_answer(chat.id, message_id, query_text, user_id)
        return

    # Check for memory search keywords (BROADER)
    if is_memory_search_query(user_text):
        # Check if it's a specific topic search or general summary
        keywords = extract_search_keywords(user_text)
        if keywords and len(keywords) > 0:
            memory_result = await search_memory_by_keyword(user_id, user_text)
        else:
            memory_result = await search_memory(user_id)
        await send_text_chunks(chat.id, memory_result, reply_to=message_id)
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

    # Normal message processing — use relevant context, not just last 5
    context = await get_relevant_context(user_id, user_text)
    response = await get_gemini_response(user_text, user_id, chat_type, profile, context)

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
        "version": "v5.0",
        "model": "African Intelligence Model",
        "features": ["smart_memory", "user_preferences", "topic_search", "inline_mode"]
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