"""
AIM Bot v9.4 — African Intelligence Model (DeepSeek V4 Pro + API Integration)

CHANGES FROM v9.3:

  FIX 1 — MISSING `import json`
  The task parser used json.loads() but json was never imported.
  Every task parse silently crashed with NameError.

  FIX 2 — TASK DETECTION INDENTATION BUG
  handle_task_message() was indented inside the group-chat block,
  so it only ran for group messages. Private chat tasks were never handled.
  Moved to the correct place — runs for ALL chat types.

  FIX 3 — RECURRING TASK NEXT_RUN BUG
  The weekly recurrence calculation used `days_ahead = [d for d in days_ahead if d > 0] or [7]`
  which meant if today matched a target day, it would wait 7 days.
  Fixed to check if next_run is in the past and add exactly 1 day,
  then scan forward day-by-day to find the next matching weekday.

  FIX 4 — MISSING SUPABASE COLUMNS HANDLED GRACEFULLY
  `completed_at` and `last_run` are now only written if they exist;
  added try/except fallbacks so missing columns don't crash the worker.
  Added a /debug/tasks endpoint to help diagnose task issues.

  FIX 5 — TIMER ≠ TASK (separation of concerns)
  Timer keywords (set a timer, stopwatch) are now explicitly excluded from
  task detection so "set a 5 minute timer" doesn't accidentally create a task.

  IMPROVEMENT — TASK NOTIFICATION CONTENT
  Recurring category tasks (news, verse, word) now fetch actual content
  when they fire, not just a reminder string.
"""

import os
import json
import uuid
import asyncio
import logging
import threading
import re
import time
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

from flask import Flask, request, jsonify
from telegram import (
    Update, Bot, InlineQueryResultArticle, InputTextMessageContent
)
from telegram.constants import ParseMode
from supabase import create_client, Client
from google import genai
from google.genai import types

# ─── LOGGING ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aimbot")

# ─── CONFIG ───
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
USE_DEEPSEEK     = os.environ.get("USE_DEEPSEEK", "false").lower() == "true"
SUPABASE_URL     = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY     = os.environ.get("SUPABASE_KEY", "")
WEBHOOK_URL      = os.environ.get("WEBHOOK_URL", "")
BRAVE_API_KEY    = os.environ.get("BRAVE_API_KEY", "")
GNEWS_API_KEY    = os.environ.get("GNEWS_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")

TELEGRAM_MAX_CHARS = 4096
WAT = timezone(timedelta(hours=1))

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
deepseek_client: Optional[AsyncOpenAI] = None

if USE_DEEPSEEK and DEEPSEEK_API_KEY:
    deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    logger.info("✅ Using DeepSeek V4 Pro API")
elif GEMINI_API_KEY:
    logger.info("✅ Using Gemini API")
else:
    logger.warning("⚠️ No AI API configured!")

groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
if GROQ_API_KEY: logger.info("✅ Groq API (Voice STT enabled)")
if BRAVE_API_KEY: logger.info("✅ Brave Search API")
if GNEWS_API_KEY: logger.info("✅ GNews API")


# ═══════════════════════════════════════════════════════════
# VOICE TRANSCRIPTION
# ═══════════════════════════════════════════════════════════

async def transcribe_voice(file_id: str) -> Optional[str]:
    if not groq_client: return None
    temp_path = f"voice_{file_id}.ogg"
    try:
        file = await bot.get_file(file_id)
        await file.download_to_drive(custom_path=temp_path)
        with open(temp_path, "rb") as audio_file:
            transcription = await groq_client.audio.transcriptions.create(
                model="whisper-large-v3", file=audio_file, response_format="text"
            )
        return transcription.strip()
    except Exception as e:
        logger.error("Voice transcription error: %s", e)
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ═══════════════════════════════════════════════════════════
# SEMANTIC ROUTER
# ═══════════════════════════════════════════════════════════

logger.info("🧠 Loading semantic router model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

SEARCH_TRIGGER_PHRASES = [
    "who won the match", "what is the score", "latest news about", "current events",
    "what happened today", "search for information", "look up", "find out about",
    "who knocked out", "eliminated from", "when did they win", "what is the price",
    "exchange rate", "weather forecast", "stock price", "bitcoin price",
    "currency conversion", "flight status", "traffic update", "road conditions",
    "event schedule", "concert tickets", "movie release date", "album release",
    "who is the president", "who is the governor", "latest update on",
    "recent developments", "breaking news", "current situation", "what is happening now",
    "live update", "real-time information", "who won the election", "match result",
    "game outcome", "tournament winner", "championship result", "final score",
    "standings table", "league table", "fixture list", "upcoming matches",
    "next game", "who is playing", "schedule for", "when is the match",
    "kickoff time", "venue information", "ticket prices", "how to watch",
    "broadcast information", "streaming options", "what did this celebrity do",
    "Politics", "Sports", "Entertainment",
    "formula 1 result", "f1 race winner", "grand prix results",
    "nba score", "basketball result", "tennis result", "wimbledon winner",
    "boxing match result", "ufc fight night", "mma result",
    "rugby result", "cricket score", "ipl result", "who won the super bowl",
    "who stopped them from qualifying", "who stopped nigeria", "who knocked nigeria out",
    "why did nigeria not qualify", "who beat nigeria", "did nigeria qualify",
    "nigeria world cup", "super eagles result", "super eagles match", "afcon result",
    "african cup of nations", "world cup qualification africa",
    "who invented", "what caused", "why did", "how did", "when did", "what year did",
    "tell me about", "give me information on", "what do you know about",
    "news about", "update on", "facts about", "history of", "background on",
    "what is going on with", "recent news", "what happened with", "explain what happened",
    "naira exchange rate", "dollar to naira", "fuel price nigeria",
    "nigerian government", "nigerian politics", "tinubu", "lagos news", "abuja news",
    "nigeria economy", "nigeria inflation", "nigeria election", "nigeria insecurity",
]

logger.info("🔢 Computing trigger embeddings...")
trigger_embeddings = semantic_model.encode(SEARCH_TRIGGER_PHRASES)
logger.info("✅ Semantic router ready with %d trigger phrases!", len(SEARCH_TRIGGER_PHRASES))

# ─── ASYNCIO EVENT LOOP ───
_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()
threading.Thread(target=_run_loop, daemon=True, name="async-loop").start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop)


# ═══════════════════════════════════════════════════════════
# BACKGROUND WORKERS
# ═══════════════════════════════════════════════════════════

def check_timers_background():
    logger.info("⏲️ Timer worker started.")
    while True:
        time.sleep(30)
        if not supabase or not bot: continue
        try:
            now_utc = datetime.now(timezone.utc).isoformat()
            res = (supabase.table("user_tools").select("*")
                   .eq("tool_type", "timer").eq("is_active", True)
                   .lte("target_time", now_utc).execute())
            for row in res.data or []:
                user_id  = row["user_id"]
                duration = row.get("duration_seconds", 0)
                supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                mins, secs = divmod(duration, 60)
                t = ""
                if mins: t += f"{mins} minute{'s' if mins != 1 else ''}"
                if secs:
                    if t: t += f" and {secs} second{'s' if secs != 1 else ''}"
                    else: t = f"{secs} second{'s' if secs != 1 else ''}"
                run_async(bot.send_message(chat_id=int(user_id), text=f"⏰ Time's up! Your {t.strip()} timer is over."))
        except Exception as e:
            logger.error("Timer worker error: %s", e)

threading.Thread(target=check_timers_background, daemon=True, name="timer-worker").start()


def _calc_next_run(task: dict, from_time: datetime) -> datetime:
    """
    Calculate the next run time for a recurring task.
    from_time should be WAT.
    """
    pattern = task.get("recurrence_pattern", "daily")
    rec_time = task.get("recurrence_time")        # "HH:MM"
    days_list = task.get("recurrence_days") or [] # ["monday", "wednesday"]

    DAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
               "friday":4,"saturday":5,"sunday":6}

    # Start from from_time, apply the recurrence time
    base = from_time
    if rec_time:
        try:
            h, m = map(int, rec_time.split(":"))
            base = base.replace(hour=h, minute=m, second=0, microsecond=0)
        except Exception:
            pass

    # If that slot is already in the past, push it forward 1 day to start searching
    if base <= from_time:
        base += timedelta(days=1)
        if rec_time:
            try:
                h, m = map(int, rec_time.split(":"))
                base = base.replace(hour=h, minute=m, second=0, microsecond=0)
            except Exception:
                pass

    if pattern == "daily":
        return base

    elif pattern == "weekly" and days_list:
        target_days = [DAY_MAP[d.lower()] for d in days_list if d.lower() in DAY_MAP]
        if not target_days:
            return base
        # Scan forward up to 7 days to find the next matching weekday
        for i in range(8):
            candidate = base + timedelta(days=i)
            if candidate.weekday() in target_days:
                return candidate
        return base + timedelta(days=7)

    elif pattern == "monthly":
        # Same day of month, next month
        month = base.month + 1
        year  = base.year + (1 if month > 12 else 0)
        month = 1 if month > 12 else month
        try:
            return base.replace(year=year, month=month)
        except ValueError:
            # e.g. Jan 31 → Feb 28
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            return base.replace(year=year, month=month, day=last_day)

    return base


def check_tasks_background():
    logger.info("📋 Task worker started.")
    while True:
        time.sleep(30)
        if not supabase or not bot: continue
        try:
            now_utc = datetime.now(timezone.utc).isoformat()
            res = (supabase.table("user_tasks").select("*")
                   .eq("is_active", True)
                   .lte("next_run", now_utc)
                   .order("next_run", desc=False)
                   .execute())

            for task in res.data or []:
                user_id     = task["user_id"]
                task_id     = task["id"]
                description = task["task_description"]
                task_type   = task["task_type"]
                category    = task.get("task_category", "reminder")

                # ── Build notification message
                if category == "news":
                    content = get_latest_news("Nigeria latest news today", 3)
                    msg = f"📰 <b>Your Daily News Update:</b>\n\n{content[:3000]}"
                elif category == "verse":
                    content = search_web("daily bible verse today", 1)
                    msg = f"📖 <b>Daily Bible Verse:</b>\n\n{content[:1000]}"
                elif category == "word":
                    content = search_web("word of the day meaning", 1)
                    msg = f"📚 <b>Word of the Day:</b>\n\n{content[:500]}"
                else:
                    msg = f"⏰ <b>Reminder:</b>\n\n{description}"

                try:
                    run_async(bot.send_message(chat_id=int(user_id), text=msg, parse_mode=ParseMode.HTML))
                    logger.info("📋 Task fired for user %s: %s", user_id, description)
                except Exception as send_err:
                    logger.error("Task send error for user %s: %s", user_id, send_err)

                if task_type == "one_time":
                    # Mark inactive — use update without completed_at in case column doesn't exist
                    try:
                        supabase.table("user_tasks").update({
                            "is_active": False,
                            "completed_at": datetime.now(timezone.utc).isoformat()
                        }).eq("id", task_id).execute()
                    except Exception:
                        supabase.table("user_tasks").update({"is_active": False}).eq("id", task_id).execute()

                elif task_type == "recurring":
                    now_wat  = datetime.now(WAT)
                    next_run = _calc_next_run(task, now_wat)
                    try:
                        supabase.table("user_tasks").update({
                            "next_run": next_run.isoformat(),
                            "last_run": datetime.now(timezone.utc).isoformat()
                        }).eq("id", task_id).execute()
                    except Exception:
                        supabase.table("user_tasks").update({
                            "next_run": next_run.isoformat()
                        }).eq("id", task_id).execute()
                    logger.info("📋 Recurring task rescheduled → %s", next_run.strftime("%Y-%m-%d %H:%M WAT"))

        except Exception as e:
            logger.error("Task worker error: %s", e)

threading.Thread(target=check_tasks_background, daemon=True, name="task-worker").start()


# ═══════════════════════════════════════════════════════════
# TASK SYSTEM
# ═══════════════════════════════════════════════════════════

# Keywords that indicate a TASK/REMINDER (not a timer)
TASK_KEYWORDS = [
    "remind me", "set a reminder", "reminder", "remind", "notify me",
    "don't let me forget", "alert me", "schedule", "every day", "every week",
    "every monday", "every tuesday", "every wednesday", "every thursday",
    "every friday", "every saturday", "every sunday", "daily reminder",
    "always remind",
]

# Keywords that are TIMERS only — exclude from task detection
TIMER_ONLY_KEYWORDS = [
    "set a timer", "start a timer", "set timer", "start timer",
    "set a stopwatch", "start stopwatch", "stopwatch",
]


async def parse_task_with_ai(user_text: str, user_id: str) -> dict:
    """Parse natural language into structured task data using AI."""
    now_wat = datetime.now(WAT)
    current_time_str = now_wat.strftime("%A, %B %d, %Y at %I:%M %p WAT")

    prompt = f"""Parse this user message into a task. Return ONLY valid JSON — no markdown, no extra text, no backticks.

JSON fields:
- description (string): clear task description
- type ("one_time"|"recurring")
- scheduled_time (ISO 8601 string or null): for one_time tasks
- recurrence_pattern ("daily"|"weekly"|"monthly"|null)
- recurrence_time ("HH:MM" 24-hour string or null)
- recurrence_days (array of lowercase day names or null)
- category ("reminder"|"news"|"verse"|"word"|"custom")
- needs_clarification (boolean)
- clarification_question (string, empty string if not needed)

Examples:
- "remind me at 6pm to cook" → one_time, scheduled_time=today 18:00 WAT
- "remind me every day at 6pm to cook" → recurring, pattern=daily, time=18:00
- "always remind me at 7am" → recurring, pattern=daily, time=07:00
- "every Monday at 9am send me news" → recurring, pattern=weekly, days=["monday"], time=09:00, category=news
- "remind me in 2 hours" → one_time, scheduled_time=now+2h
- "remind me tomorrow at 3pm" → one_time, scheduled_time=tomorrow 15:00 WAT

Current time: {current_time_str}
User message: "{user_text}"

Return ONLY the JSON object:"""

    try:
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=400
            )
            raw = r.choices[0].message.content.strip()
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=400),
            )
            raw = r.text.strip() if r and r.text else ""
        else:
            return {"needs_clarification": True, "clarification_question": "Could you clarify when you want this reminder?"}

        # Strip any accidental markdown fences
        raw = re.sub(r'^```json\s*', '', raw.strip())
        raw = re.sub(r'^```\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw.strip())
        raw = raw.strip()

        parsed = json.loads(raw)
        logger.info("✅ Task parsed: %s", parsed)
        return parsed

    except json.JSONDecodeError as e:
        logger.error("Task JSON parse error: %s | Raw: %s", e, raw if 'raw' in dir() else "N/A")
        return {"needs_clarification": True, "clarification_question": "Could you clarify when and how often you want this reminder?"}
    except Exception as e:
        logger.error("Task parse error: %s", e)
        return {"needs_clarification": True, "clarification_question": "Could you clarify when you want this reminder?"}


async def create_task_in_db(user_id: str, task_data: dict) -> str:
    """Save parsed task to Supabase. Returns user-facing confirmation string."""
    if not supabase:
        return "❌ Memory is offline. Can't save the task right now."
    try:
        now_wat = datetime.now(WAT)

        next_run = None
        if task_data.get("type") == "one_time" and task_data.get("scheduled_time"):
            try:
                st = task_data["scheduled_time"]
                # Handle both naive and tz-aware ISO strings
                if "+" in st or st.endswith("Z"):
                    next_run = datetime.fromisoformat(st.replace("Z", "+00:00"))
                else:
                    # Assume WAT
                    next_run = datetime.fromisoformat(st).replace(tzinfo=WAT)
                if next_run < datetime.now(timezone.utc):
                    next_run += timedelta(days=1)
            except Exception as e:
                logger.error("scheduled_time parse error: %s", e)
                next_run = now_wat + timedelta(hours=1)

        elif task_data.get("type") == "recurring":
            next_run = _calc_next_run(task_data, now_wat)

        row = {
            "user_id":              str(user_id),
            "task_description":     task_data.get("description", "Reminder"),
            "task_type":            task_data.get("type", "one_time"),
            "scheduled_time":       task_data.get("scheduled_time"),
            "recurrence_pattern":   task_data.get("recurrence_pattern"),
            "recurrence_time":      task_data.get("recurrence_time"),
            "recurrence_days":      task_data.get("recurrence_days") or [],
            "task_category":        task_data.get("category", "reminder"),
            "is_active":            True,
            "next_run":             next_run.isoformat() if next_run else None,
        }

        supabase.table("user_tasks").insert(row).execute()

        # Build user-facing confirmation
        desc     = task_data.get("description", "your reminder")
        t_type   = task_data.get("type", "one_time")
        if t_type == "recurring":
            pattern = task_data.get("recurrence_pattern", "daily")
            r_time  = task_data.get("recurrence_time", "")
            r_days  = task_data.get("recurrence_days") or []
            if pattern == "weekly" and r_days:
                schedule_str = f"every {', '.join(r_days)}"
            else:
                schedule_str = pattern
            if r_time:
                try:
                    h, m  = map(int, r_time.split(":"))
                    ampm  = "AM" if h < 12 else "PM"
                    h12   = h % 12 or 12
                    schedule_str += f" at {h12}:{m:02d} {ampm}"
                except Exception:
                    schedule_str += f" at {r_time}"
            return f"✅ Got it! I'll remind you {schedule_str}: \"{desc}\""
        else:
            time_str = next_run.astimezone(WAT).strftime("%A, %b %d at %I:%M %p WAT") if next_run else "soon"
            return f"✅ Reminder set for {time_str}: \"{desc}\""

    except Exception as e:
        logger.error("Task creation error: %s", e)
        return "❌ Couldn't save the task. Please try again."


async def handle_task_message(user_text: str, user_id: str, chat_id: int, message_id: int) -> bool:
    """
    Detect task/reminder intent and handle it.
    Returns True if this message was handled as a task.

    FIX: This is now called BEFORE the group check so it works in all chat types.
    """
    text_lower = user_text.lower()

    # Bail out early if this is clearly a timer/stopwatch request
    if any(kw in text_lower for kw in TIMER_ONLY_KEYWORDS):
        return False

    # Check for task keywords
    if not any(kw in text_lower for kw in TASK_KEYWORDS):
        return False

    logger.info("📋 Task intent detected: '%s'", user_text[:60])

    # Check if user is answering a clarification we asked earlier
    context_prefix = ""
    if supabase:
        try:
            recent = (supabase.table("chat_memory").select("message, response")
                      .eq("user_id", str(user_id))
                      .order("created_at", desc=True).limit(1).execute())
            if recent.data:
                last_response = recent.data[0].get("response", "")
                # If our last reply was a clarification question, include context
                if "clarify" in last_response.lower() or "when" in last_response.lower():
                    original_msg = recent.data[0].get("message", "")
                    context_prefix = f"Original request: {original_msg}\nUser's answer: "
        except Exception:
            pass

    task_data = await parse_task_with_ai(context_prefix + user_text, user_id)

    if task_data.get("needs_clarification"):
        question = task_data.get("clarification_question", "Could you clarify when you want this reminder?")
        await send_text_chunks(chat_id, f"🤔 {question}", reply_to=message_id)
        return True

    result = await create_task_in_db(user_id, task_data)
    await send_text_chunks(chat_id, result, reply_to=message_id)
    return True


# ═══════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════

def get_latest_news(query: str, max_results: int = 5) -> str:
    if not GNEWS_API_KEY:
        return search_web(query, max_results)
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "apikey": GNEWS_API_KEY, "lang": "en", "country": "ng", "max": max_results},
            timeout=10,
        )
        if resp.status_code != 200:
            return search_web(query, max_results)
        articles = resp.json().get("articles", [])
        if not articles:
            return search_web(query, max_results)
        lines = []
        for i, a in enumerate(articles[:max_results], 1):
            pub = a.get("publishedAt","")[:16].replace("T"," ")
            lines.append(f"{i}. {a.get('title','')}\n   Source: {a.get('source',{}).get('name','')} | {pub}\n   {a.get('description','')}\n   {a.get('url','')}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("GNews error: %s", e)
        return search_web(query, max_results)


def get_sports_data(query: str) -> str:
    if not GNEWS_API_KEY:
        return search_web(query, 5)
    try:
        q_lower = query.lower()
        sport_q = "Nigeria football"
        if "premier league" in q_lower or "epl" in q_lower: sport_q = "Premier League"
        elif "champions league" in q_lower: sport_q = "Champions League"
        elif "afcon" in q_lower or "african cup" in q_lower: sport_q = "AFCON"
        elif "world cup" in q_lower: sport_q = "World Cup"
        elif "f1" in q_lower or "formula" in q_lower: sport_q = "Formula 1"
        elif "nba" in q_lower or "basketball" in q_lower: sport_q = "NBA basketball"
        elif "tennis" in q_lower: sport_q = "tennis"
        elif "boxing" in q_lower or "ufc" in q_lower: sport_q = "boxing MMA UFC"
        elif "cricket" in q_lower: sport_q = "cricket"
        elif "rugby" in q_lower: sport_q = "rugby"
        else: sport_q = query

        resp = requests.get(
            "https://gnews.io/api/v4/top-headlines",
            params={"category": "sports", "q": sport_q, "apikey": GNEWS_API_KEY, "lang": "en", "country": "ng", "max": 5},
            timeout=10,
        )
        if resp.status_code != 200:
            return search_web(query, 5)
        articles = resp.json().get("articles", [])
        if not articles:
            return search_web(query, 5)
        lines = ["🏆 LATEST SPORTS UPDATES:\n"]
        for i, a in enumerate(articles[:5], 1):
            pub = a.get("publishedAt","")[:16].replace("T"," ")
            lines.append(f"{i}. {a.get('title','')}\n   Source: {a.get('source',{}).get('name','')} | {pub}\n   {a.get('description','')}\n   {a.get('url','')}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("GNews Sports error: %s", e)
        return search_web(query, 5)


def _search_brave_scrape(query: str, max_results: int = 5) -> Optional[list]:
    from urllib.parse import quote, unquote
    try:
        resp = requests.get(
            f"https://search.brave.com/search?q={quote(query)}&source=web",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            timeout=15,
        )
        if resp.status_code != 200: return None
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        snippets = soup.find_all("div", class_="snippet") or soup.find_all("div", class_="result")
        for s in snippets[:max_results]:
            a = s.find("a", href=True)
            if not a: continue
            title = a.get_text(strip=True)
            url   = a["href"]
            if url and not url.startswith("http"):
                url = "https://search.brave.com" + url
            if "url=" in url:
                m = re.search(r'url=([^&]+)', url)
                if m: url = unquote(m.group(1))
            desc_el = s.find("p") or s.find("div", class_="snippet-description")
            desc = desc_el.get_text(strip=True)[:300] if desc_el else ""
            if title and len(title) > 3:
                results.append({"title": title, "description": desc, "url": url})
        return results[:max_results] if results else None
    except Exception as e:
        logger.error("Brave scrape error: %s", e)
        return None


def _search_brave_api(query: str, max_results: int = 5) -> Optional[list]:
    if not BRAVE_API_KEY: return None
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": max_results},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("web", {}).get("results", [])
            if items:
                return [{"title": i.get("title",""), "description": i.get("description",""), "url": i.get("url","")} for i in items[:max_results]]
        return None
    except Exception as e:
        logger.error("Brave API error: %s", e)
        return None


def _search_duckduckgo_lite(query: str, max_results: int = 5) -> Optional[list]:
    try:
        resp = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept-Language": "en-US,en;q=0.9"},
            timeout=10,
        )
        if resp.status_code != 200: return None
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for link in soup.find_all("a", class_="result-link")[:max_results]:
            title = link.get_text(strip=True)
            href  = link.get("href", "")
            desc  = ""
            row   = link.find_parent("tr")
            if row:
                nr = row.find_next_sibling("tr")
                if nr:
                    td = nr.find("td", class_="result-snippet")
                    if td: desc = td.get_text(strip=True)
            if title and href:
                results.append({"title": title, "description": desc, "url": href})
        return results if results else None
    except Exception as e:
        logger.error("DuckDuckGo Lite error: %s", e)
        return None


def search_web(query: str, max_results: int = 5) -> str:
    results  = _search_brave_scrape(query, max_results) or \
               _search_brave_api(query, max_results)    or \
               _search_duckduckgo_lite(query, max_results)
    if not results:
        return "No search results found."
    lines = [f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}"
             for i, r in enumerate(results, 1)]
    return "\n\n".join(lines)


def deep_research(query: str) -> str:
    seen_urls, all_results = set(), []
    for q in [query, f"{query} latest news", f"{query} results details"]:
        rs = _search_brave_scrape(q, 4) or _search_brave_api(q, 4) or _search_duckduckgo_lite(q, 4)
        if rs:
            for r in rs:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
    if not all_results:
        return "Deep research found no results."
    lines = ["=== DEEP RESEARCH RESULTS ===\n"]
    for i, r in enumerate(all_results[:12], 1):
        lines.append(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}")
    return "\n\n".join(lines)


def fetch_url_content(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]): tag.extract()
        return " ".join(soup.get_text(separator=" ", strip=True).split())[:3000]
    except Exception as e:
        logger.error("URL fetch error: %s", e)
        return "Failed to read link."


def detect_urls(text: str) -> list:
    return re.findall(r'https?://\S+', text)


def is_search_query_semantic(text: str, threshold: float = 0.45) -> bool:
    try:
        sims    = np.dot(trigger_embeddings, semantic_model.encode([text]).T).flatten()
        max_sim = float(np.max(sims))
        best    = SEARCH_TRIGGER_PHRASES[int(np.argmax(sims))]
        result  = max_sim >= threshold
        logger.info("🔍 Semantic: '%.60s' → %.3f (best: '%s') → %s", text, max_sim, best, "SEARCH" if result else "skip")
        return result
    except Exception as e:
        logger.error("Semantic routing error: %s", e)
        return False


def is_search_query(text: str) -> bool:
    tl = text.lower().strip()
    if any(t in tl for t in ["search for","google","look up","find out","search the web","browse","search"]):
        return True
    return is_search_query_semantic(text)


# ═══════════════════════════════════════════════════════════
# PROMPTING
# ═══════════════════════════════════════════════════════════

BASE_SYSTEM_PROMPT = """You are AIM — African Intelligence Model. A professional, highly intelligent AI assistant built for Africans, by Africans.

PERSONALITY & TONE:
- Warm, respectful, and culturally aware.
- Adapt to the user's vibe. Formal ↔ Casual. Match their energy.
- Be helpful, patient, and empowering.

LANGUAGE RULE (CRITICAL):
- Respond in the EXACT language or dialect the user is using (Pidgin, Yoruba, Hausa, Igbo, English, etc.).
- If the user mixes languages, you can mix too.
- EXCEPTION: Never use rude or hateful language.

RULES:
- Keep responses concise but informative.
- If you don't know something, use the SEARCH TRIGGER.
- Never make up facts.
- Use emojis naturally but not excessively.

CAPABILITIES:
- Memory: Rolling session summary + last 5 conversations with timestamps.
- Tasks & Reminders: Users can set one-time or recurring reminders.
- Time Tools: Timers and stopwatches.
- Web Search: Real-time results.
- Sports: All sports via GNews (Football, F1, NBA, Tennis, Boxing, UFC, Cricket, Rugby).
- News: Real-time news from Nigeria and worldwide.

CONVERSATION CONTINUITY:
- Read SESSION SUMMARY and RECENT HISTORY before responding.
- Short follow-ups ("yes", "ok", "go on") → continue previous topic.
- Pronouns (he, she, it) → resolve from previous message.
- DO NOT start every message with "Hello there! It's [date/time]".
- Only mention time if the user explicitly asks, or if it's been a long gap.

─────────────────────────────────────────
SPECIAL INSTRUCTIONS:

1. TIMERS/STOPWATCHES:
   Append machine code at END of response (never for reminders/tasks):
   [TIMER:Xs] [TIMER:Xm] [TIMER:Xh] [STOPWATCH:START] [STOPWATCH:STOP]

2. SEARCH TRIGGER:
   SEARCH_TRIGGER: <your search query>
   Use for current events, scores, prices, weather, recent news, or anything uncertain.

3. WEB CONTEXT PROVIDED:
   Synthesize results. Do NOT output SEARCH_TRIGGER again.

4. GENERAL KNOWLEDGE:
   Answer directly if confident. No SEARCH_TRIGGER needed.
─────────────────────────────────────────
"""


async def get_user_profile_data(user_id: str) -> dict:
    if not supabase: return {}
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        return rows.data[0] if rows.data else {}
    except Exception as e:
        logger.error("Profile fetch error: %s", e)
        return {}


async def get_session_summary(user_id: str) -> str:
    if not supabase: return ""
    try:
        res = supabase.table("user_profiles").select("session_summary, last_active").eq("user_id", str(user_id)).execute()
        if not res.data: return ""
        p = res.data[0]
        summary = p.get("session_summary", "") or ""
        last_active_str = p.get("last_active", "")
        if not last_active_str: return summary
        last_active = datetime.fromisoformat(last_active_str.replace("Z", "+00:00"))
        if (datetime.now(timezone.utc) - last_active).total_seconds() > 10800:
            supabase.table("user_profiles").update({"session_summary": ""}).eq("user_id", str(user_id)).execute()
            return ""
        return summary
    except Exception as e:
        logger.error("Session summary error: %s", e)
        return ""


async def update_session_summary(user_id: str, recent_messages: list, current_summary: str):
    if not supabase: return
    try:
        msg_text = "\n".join([f"User: {m['message']}\nAIM: {m['response']}" for m in recent_messages])
        prompt = f"""Current Summary: {current_summary or 'None yet'}
New Messages:
{msg_text}

Create a concise updated summary including key facts, ongoing topics, and context. Max 150 words."""
        new_summary = None
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash", messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=200)
            new_summary = r.choices[0].message.content.strip() if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=200))
            new_summary = r.text.strip() if r and r.text else None
        if new_summary:
            supabase.table("user_profiles").update({"session_summary": new_summary}).eq("user_id", str(user_id)).execute()
    except Exception as e:
        logger.error("Summarization error: %s", e)


def _fmt_wat(dt: datetime) -> str:
    return dt.astimezone(WAT).strftime("%a %b %d, %Y · %I:%M %p WAT")


def _gap_instruction(seconds: float) -> str:
    if seconds < 1800:
        return "NO_GAP_ACK: User just chatted. Respond normally, no welcome-back."
    elif seconds < 10800:
        return "LIGHT_ACK: Moderate gap. Optional brief acknowledgment."
    elif seconds < 86400:
        return "GAP_ACK: Several hours. You can say 'It's been a while!' naturally."
    else:
        return "LONG_GAP_ACK: Over 24h. Warm welcome back: 'Long time no see!', 'Welcome back!' etc."


async def get_conversation_context(user_id: str, query_text: str) -> tuple[str, str, float]:
    if not supabase: return "", "", 0.0
    try:
        rows = (supabase.table("chat_memory").select("message, response, topic, created_at")
                .eq("user_id", str(user_id)).order("created_at", desc=True).limit(40).execute())
        if not rows.data: return "", "", 0.0

        gap_seconds = 0.0
        try:
            last_ts = datetime.fromisoformat(rows.data[0]["created_at"].replace("Z", "+00:00"))
            gap_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
        except Exception: pass

        # Recent: last 5 in chronological order with WAT timestamps
        recent_rows = list(reversed(rows.data[:5]))
        recent_lines = []
        for row in recent_rows:
            try:
                ts   = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                tstr = _fmt_wat(ts)
            except Exception:
                tstr = "unknown time"
            recent_lines += [f"  [{tstr}]", f"  User : {row['message']}", f"  AIM  : {row['response']}", ""]
        recent_history = "\n".join(recent_lines).strip()

        # Older: relevance-scored
        older_rows = rows.data[5:]
        if not older_rows: return recent_history, "", gap_seconds

        query_lower = query_text.lower()
        keyword_topics = {
            "space":["tech"], "nigeria":["general","politics"], "money":["finance"],
            "job":["career"], "health":["health"], "love":["relationships"],
            "sport":["sports"], "music":["entertainment"], "school":["education"],
            "code":["tech"], "ai":["tech"],
        }
        matched_topics: set = set()
        for kw, tps in keyword_topics.items():
            if kw in query_lower: matched_topics.update(tps)

        scored = []
        for row in older_rows:
            score = 0
            try:
                age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"].replace("Z","+00:00"))).days
                score += max(0, 30 - age_days)
            except Exception: pass
            if row.get("topic") in matched_topics: score += 50
            blob = f"{row['message']} {row['response']}".lower()
            for word in query_lower.split():
                if len(word) > 3 and word in blob: score += 10
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        older_lines = [f"[{r.get('topic','general')}] User: {r['message']} | AIM: {r['response']}" for _, r in scored[:10]]
        return recent_history, "\n".join(older_lines), gap_seconds
    except Exception as e:
        logger.error("Context retrieval error: %s", e)
        return "", "", 0.0


def build_enhanced_prompt(
    user_text: str, user_id: str, profile: dict,
    session_summary: str = "", recent_history: str = "", older_context: str = "",
    web_context: str = "", tool_status: str = "", gap_seconds: float = 0.0,
) -> str:
    now_wat = datetime.now(WAT)
    parts   = [BASE_SYSTEM_PROMPT]

    pref_language = profile.get("preferred_language", "english")
    topic_counts  = profile.get("topic_counts", {})
    total_chats   = profile.get("total_chats", 0)
    pref_lines = [
        "\n--- USER PREFERENCES ---",
        f"  User ID: {user_id}  |  Language: {pref_language}  |  Total chats: {total_chats}",
    ]
    if topic_counts:
        top = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        pref_lines.append(f"  Interests: {', '.join(f'{k}({v})' for k,v in top)}")
    pref_lines.append("--- END PREFERENCES ---\n")
    parts.append("\n".join(pref_lines))

    # TIME & CONTEXT BLOCK
    parts.append(
        "\n┌──────────────────────────────────────────┐\n"
        "│  TIME & CONTEXT                          │\n"
        "└──────────────────────────────────────────┘\n"
        f"  Current time (WAT)  : {now_wat.strftime('%A, %B %d, %Y · %I:%M %p')}\n"
        f"  User's last message : {_gap_label(gap_seconds)}\n"
        f"  Greeting guidance   : {_gap_instruction(gap_seconds)}\n"
        "\n  ❌ DO NOT start reply with the date/time.\n"
        "  ✅ Only mention time if user explicitly asks.\n"
        "─────────────────────────────────────────────\n"
    )

    if session_summary:
        parts.append(
            "\n╔══════════════════════════════════════╗\n"
            "║       SESSION SUMMARY                ║\n"
            "╚══════════════════════════════════════╝\n"
            + session_summary +
            "\n════════════════════════════════════════\n"
        )

    if recent_history:
        parts.append(
            "\n╔══════════════════════════════════════════════╗\n"
            "║  CONVERSATION HISTORY — LAST 5 MESSAGES    ║\n"
            "║  (READ FIRST. Use timestamps for time Q's) ║\n"
            "╚══════════════════════════════════════════════╝\n\n"
            + recent_history +
            "\n\n══════════════════════════════════════════════\n"
        )

    if web_context:
        parts.append("\n--- WEB SEARCH RESULTS ---\n" + web_context + "\n--- END WEB RESULTS ---\n")

    if older_context:
        parts.append("\n--- OLDER MEMORY (background) ---\n" + older_context + "\n--- END OLDER MEMORY ---\n")

    if tool_status:
        parts.append(f"\n--- TOOL STATUS ---\n{tool_status}\n---\n")

    parts.append(f"\nUSER MESSAGE: {user_text}")
    return "\n".join(parts)


def _gap_label(seconds: float) -> str:
    if seconds < 60: return "just now"
    elif seconds < 3600: return f"{int(seconds/60)} min ago"
    elif seconds < 86400: return f"{int(seconds/3600)} hr ago"
    elif seconds < 604800: return f"{int(seconds/86400)} day(s) ago"
    else: return f"{int(seconds/604800)} week(s) ago"


async def get_ai_response(
    user_text: str, user_id: str, chat_type: str, profile: dict = None,
    session_summary: str = "", recent_history: str = "", older_context: str = "",
    web_context: str = "", tool_status: str = "", gap_seconds: float = 0.0,
) -> Optional[str]:
    try:
        if profile is None: profile = await get_user_profile_data(user_id)
        prompt = build_enhanced_prompt(
            user_text, user_id, profile,
            session_summary, recent_history, older_context, web_context, tool_status, gap_seconds
        )
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role":"system","content":BASE_SYSTEM_PROMPT},{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=1024)
            return r.choices[0].message.content if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024))
            return r.text if r and r.text else None
        return None
    except Exception as e:
        logger.error("AI response error: %s", e)
        return None


async def extract_topic(user_text: str, bot_response: str) -> str:
    topics = ["career","finance","tech","sports","health","relationships","politics","entertainment","education","general"]
    prompt = f"Classify into ONE topic from {topics}.\nUser: {user_text[:200]}\nAIM: {bot_response[:200]}\nReturn ONLY the topic word."
    try:
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash", messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=20)
            t = r.choices[0].message.content.strip().lower() if r.choices else "general"
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20))
            t = r.text.strip().lower() if r and r.text else "general"
        else: return "general"
        return t if t in topics else "general"
    except Exception: return "general"


async def save_chat_memory(user_id: str, username: str, message: str, response: str, chat_type: str, topic: str = "general"):
    if not supabase: return
    try:
        supabase.table("chat_memory").insert({
            "user_id": str(user_id), "username": username or "",
            "message": message[:2000], "response": response[:2000],
            "chat_type": chat_type, "topic": topic,
        }).execute()
    except Exception as e: logger.error("Memory save failed: %s", e)


async def update_user_profile(user_id: str, username: str, topic: str):
    if not supabase: return
    try:
        ex = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if ex.data:
            p = ex.data[0]
            tc = p.get("topic_counts", {})
            tc[topic] = tc.get(topic, 0) + 1
            supabase.table("user_profiles").update({
                "topic_counts": tc, "total_chats": p.get("total_chats",0)+1,
                "last_active": datetime.now(timezone.utc).isoformat(),
            }).eq("user_id", str(user_id)).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": str(user_id), "username": username or "",
                "topic_counts": {topic: 1}, "total_chats": 1,
                "last_active": datetime.now(timezone.utc).isoformat(),
            }).execute()
    except Exception as e: logger.error("Profile update failed: %s", e)


def is_memory_search_query(user_text: str) -> bool:
    keywords = ["what did we talk about","what have we discussed","remember our chats",
                "our conversations","what did i ask you","what were we talking about",
                "we discussed","what did we say about","remember when","do you remember"]
    return any(kw in user_text.lower() for kw in keywords)


def extract_search_keywords(user_text: str) -> list:
    clean = user_text.lower()
    for phrase in ["what did we talk about","what were we talking about","tell me about",
                   "what did we say about","do you remember","remember when","what about",
                   "didn't we talk about","what have we discussed"]:
        clean = clean.replace(phrase, "")
    clean = re.sub(r'[^\w\s]', ' ', clean)
    stop  = {"the","and","about","were","did","have","what","when","that","this","with","for",
             "from","you","are","was","is","it","we","our","me","my","i","a","an","to","of",
             "in","on","at","be","been","do","does","say","get","go","know","think","take",
             "see","want","use","find","give","tell","ask","work"}
    return [w for w in clean.split() if len(w) > 2 and w not in stop]


async def search_memory_by_keyword(user_id: str, query_text: str) -> str:
    if not supabase: return "Memory is offline."
    try:
        keywords = extract_search_keywords(query_text)
        if not keywords: return await search_memory(user_id)
        all_results = []
        for kw in keywords[:3]:
            for field in ["message","response"]:
                r = supabase.table("chat_memory").select("*").eq("user_id",str(user_id)).ilike(field,f"%{kw}%").order("created_at",desc=True).limit(5).execute()
                all_results.extend(r.data)
        seen, unique = set(), []
        for row in all_results:
            if row["id"] not in seen:
                seen.add(row["id"])
                unique.append(row)
        unique.sort(key=lambda x: x.get("created_at",""), reverse=True)
        if not unique: return f"I don't recall discussing {' '.join(keywords)}. Want to talk about it?"
        emoji_map = {"career":"💼","finance":"💰","tech":"💻","sports":"⚽","health":"🏥","relationships":"❤️","politics":"🏛️","entertainment":"🎬","education":"📚"}
        lines = [f"🔍 Found {len(unique)} conversation(s):"]
        for i, row in enumerate(unique[:5], 1):
            em   = emoji_map.get(row.get("topic"),"💬")
            date = row.get("created_at","")[:10]
            lines.append(f'\n{i}. {em} [{date}] You: "{row["message"][:80]}..."')
            lines.append(f'   AIM: "{row["response"][:120]}..."')
        return "\n".join(lines)
    except Exception as e:
        logger.error("Keyword memory search error: %s", e)
        return "Having trouble searching memory right now."


async def search_memory(user_id: str) -> str:
    if not supabase: return "Memory is offline."
    try:
        pr = supabase.table("user_profiles").select("*").eq("user_id",str(user_id)).execute()
        if not pr.data: return "We haven't chatted before!"
        p  = pr.data[0]
        tc = p.get("topic_counts",{})
        mr = supabase.table("chat_memory").select("message,response,topic,created_at").eq("user_id",str(user_id)).order("created_at",desc=True).limit(10).execute()
        emoji_map = {"career":"💼","finance":"💰","tech":"💻","sports":"⚽","health":"🏥","relationships":"❤️","politics":"🏛️","entertainment":"🎬","education":"📚"}
        lines = [
            f"📊 Top Topics: {', '.join(f'{k}({v}x)' for k,v in sorted(tc.items(),key=lambda x:x[1],reverse=True)[:3])}",
            f"💬 Total Chats: {p.get('total_chats',0)}", "", "📝 Recent conversations:",
        ]
        for i, row in enumerate(mr.data[:5], 1):
            em   = emoji_map.get(row.get("topic"),"💬")
            date = row.get("created_at","")[:10]
            lines.append(f"{i}. {em} [{date}] {row['message'][:60]}...")
        lines.append("\nWant to dive deeper? Just ask!")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Memory search error: %s", e)
        return "Memory search is having issues."


# ═══════════════════════════════════════════════════════════
# BOT COMMANDS
# ═══════════════════════════════════════════════════════════

async def handle_bot_command(user_id: str, chat_id: int, message_id: int, user_text: str) -> bool:
    tl = user_text.lower().strip()

    if tl.startswith("/help"):
        await send_text_chunks(chat_id, """🤖 <b>AIM Bot Commands</b>

<b>General:</b>
/help — Show this message

<b>Web Search:</b>
/search [query] — Quick web search
/deep [query] — Deep multi-angle research

<b>Time Tools:</b>
/timer [time] — Set a timer (e.g. /timer 5m, /timer 30s, /timer 1h)
/stopwatch — Start or stop stopwatch

<b>Tasks & Reminders:</b>
/tasks — View your active tasks
/tasks delete [id] — Delete a task

<b>Quick Recurring Tasks:</b>
/news daily 8am — Daily news at 8am
/verse daily — Daily bible verse at 8am
/word daily — Word of the day at 9am

<b>Natural Language works too:</b>
- "Remind me at 6pm to cook dinner"
- "Remind me every Monday at 9am"
- "Set a 5 minute timer"
""", reply_to=message_id)
        return True

    elif tl.startswith("/search "):
        query = user_text[8:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔍 Searching...", reply_to=message_id)
        if "news" in query.lower() or "latest" in query.lower():
            results = get_latest_news(query)
        elif any(s in query.lower() for s in ["football","match","score","f1","nba","tennis","boxing"]):
            results = get_sports_data(query)
        else:
            results = search_web(query)
        if results == "No search results found.":
            await send_text_chunks(chat_id, "Couldn't find results. Try rephrasing.", reply_to=message_id)
            return True
        prompt = f"User asked: {query}\n\nSearch Results:\n{results}\n\nAnswer using ONLY these results. Do NOT output SEARCH_TRIGGER."
        try:
            txt = await get_ai_response(prompt, user_id, "private")
            if not txt or "SEARCH_TRIGGER:" in txt: txt = results
            await send_text_chunks(chat_id, txt.strip(), reply_to=message_id)
        except Exception:
            await send_text_chunks(chat_id, results, reply_to=message_id)
        return True

    elif tl.startswith("/deep "):
        query = user_text[6:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔬 Researching from multiple angles...", reply_to=message_id)
        deep = deep_research(query)
        txt  = await get_ai_response(query, user_id, "private", web_context=deep)
        await send_text_chunks(chat_id, txt.strip() if txt else deep, reply_to=message_id)
        return True

    elif tl.startswith("/timer "):
        ts = user_text[7:].strip()
        m  = re.match(r'(\d+)(s|m|h)', ts.lower())
        if m:
            amt, unit = int(m.group(1)), m.group(2)
            dur    = amt * (1 if unit=="s" else 60 if unit=="m" else 3600)
            target = datetime.now(timezone.utc) + timedelta(seconds=dur)
            supabase.table("user_tools").insert({
                "user_id": str(user_id), "tool_type": "timer",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": dur, "target_time": target.isoformat(), "is_active": True,
            }).execute()
            await send_text_chunks(chat_id, f"⏲️ Timer set for {ts}!", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "❌ Use: /timer 5m  /timer 30s  /timer 1h", reply_to=message_id)
        return True

    elif tl == "/stopwatch":
        res = supabase.table("user_tools").select("*").eq("user_id",str(user_id)).eq("tool_type","stopwatch").eq("is_active",True).order("created_at",desc=True).limit(1).execute()
        if res.data:
            row     = res.data[0]
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row["start_time"].replace("Z","+00:00"))
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
            await send_text_chunks(chat_id, f"⏱️ Stopped! Elapsed: {mins}m {secs}s" if mins else f"⏱️ Stopped! Elapsed: {secs}s", reply_to=message_id)
        else:
            supabase.table("user_tools").insert({
                "user_id": str(user_id), "tool_type": "stopwatch",
                "start_time": datetime.now(timezone.utc).isoformat(), "is_active": True,
            }).execute()
            await send_text_chunks(chat_id, "⏱️ Stopwatch started! Use /stopwatch again to stop.", reply_to=message_id)
        return True

    elif tl == "/tasks":
        res = supabase.table("user_tasks").select("id,task_description,task_type,next_run,task_category").eq("user_id",str(user_id)).eq("is_active",True).order("next_run",desc=False).execute()
        if not res.data:
            await send_text_chunks(chat_id, "📋 You have no active tasks.", reply_to=message_id)
            return True
        lines = ["📋 <b>Your Active Tasks:</b>\n"]
        for t in res.data:
            next_time = datetime.fromisoformat(t["next_run"].replace("Z","+00:00")) if t.get("next_run") else None
            time_str  = next_time.astimezone(WAT).strftime("%b %d, %I:%M %p WAT") if next_time else "Soon"
            icon      = "🔁" if t["task_type"] == "recurring" else "1️⃣"
            lines.append(f"{icon} <code>{t['id'][:8]}</code> | {t['task_description']}\n   Next: {time_str}")
        lines.append("\n<i>Use /tasks delete &lt;id&gt; to remove</i>")
        await send_text_chunks(chat_id, "\n".join(lines), reply_to=message_id)
        return True

    elif tl.startswith("/tasks delete"):
        parts = tl.split()
        if len(parts) < 3:
            await send_text_chunks(chat_id, "❌ Use: /tasks delete <task_id>", reply_to=message_id)
            return True
        partial_id = parts[2]
        res = supabase.table("user_tasks").select("id").eq("user_id",str(user_id)).eq("is_active",True).execute()
        full_id = next((t["id"] for t in res.data if t["id"].startswith(partial_id)), None)
        if not full_id:
            await send_text_chunks(chat_id, "❌ Task not found. Check /tasks for the ID.", reply_to=message_id)
            return True
        supabase.table("user_tasks").update({"is_active": False}).eq("id", full_id).eq("user_id", str(user_id)).execute()
        await send_text_chunks(chat_id, "✅ Task deleted!", reply_to=message_id)
        return True

    elif tl.startswith("/news "):
        parts = user_text.split()
        if len(parts) < 3:
            await send_text_chunks(chat_id, "❌ Use: /news daily 8am", reply_to=message_id)
            return True
        pattern   = parts[1].lower()
        time_part = parts[2].lower().replace("am","").replace("pm","")
        # Parse hour
        try:
            hour = int(time_part)
            if "pm" in parts[2].lower() and hour != 12: hour += 12
            rec_time = f"{hour:02d}:00"
        except ValueError:
            rec_time = "08:00"
        task_data = {"description": "Send me the latest news", "type": "recurring",
                     "recurrence_pattern": pattern, "recurrence_time": rec_time,
                     "recurrence_days": [], "category": "news", "needs_clarification": False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/verse daily":
        task_data = {"description": "Daily bible verse", "type": "recurring", "recurrence_pattern": "daily",
                     "recurrence_time": "08:00", "category": "verse", "needs_clarification": False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/word daily":
        task_data = {"description": "Word of the day", "type": "recurring", "recurrence_pattern": "daily",
                     "recurrence_time": "09:00", "category": "word", "needs_clarification": False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    return False


# ─── SEND MESSAGE ───
async def send_text_chunks(chat_id: int, text: str, reply_to: Optional[int] = None, message_id: Optional[int] = None):
    if not bot: return
    try:
        if message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS], parse_mode=ParseMode.HTML)
        else:
            kw = {"chat_id": chat_id, "text": text[:TELEGRAM_MAX_CHARS], "parse_mode": ParseMode.HTML}
            if reply_to: kw["reply_to_message_id"] = reply_to
            await bot.send_message(**kw)
    except Exception as e:
        logger.error("Send error: %s", e)
        try:
            if message_id: await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS])
            else: await bot.send_message(chat_id=chat_id, text=text[:TELEGRAM_MAX_CHARS])
        except Exception as e2: logger.error("Fallback send failed: %s", e2)


# ─── INLINE QUERY ───
async def handle_inline_query_async(inline_query):
    qid, qtext = inline_query.id, inline_query.query.strip()
    uid = str(inline_query.from_user.id) if inline_query.from_user else ""
    if not qtext or len(qtext) < 2:
        await bot.answer_inline_query(inline_query_id=qid, results=[], cache_time=1)
        return
    web_ctx = ""
    for url in detect_urls(qtext):
        c = fetch_url_content(url)
        if c and "Failed" not in c: web_ctx += f"Content from {url}:\n{c}\n"
    if is_search_query(qtext) and not web_ctx:
        if "news" in qtext.lower() or "latest" in qtext.lower(): sr = get_latest_news(qtext)
        elif any(s in qtext.lower() for s in ["football","match","score","f1","nba","tennis","boxing"]): sr = get_sports_data(qtext)
        else: sr = search_web(qtext)
        if "No search results" not in sr: web_ctx = f"Web Search Results:\n{sr}"
    answer_text = None
    try:
        profile = await get_user_profile_data(uid)
        r_text = await asyncio.wait_for(get_ai_response(qtext, uid, "private", profile, web_context=web_ctx), timeout=15.0)
        if r_text: answer_text = r_text.strip()[:300]
    except asyncio.TimeoutError: pass
    except Exception as e: logger.error("Inline AI error: %s", e)
    result = InlineQueryResultArticle(
        id=str(uuid.uuid4()), title=f"AIM: {qtext[:30]}",
        description=(answer_text or "Click to get AIM's answer")[:100],
        input_message_content=InputTextMessageContent(
            message_text=f"🤖 <b>AIM says:</b>\n\n{answer_text}\n\n<i>via @askaimbot</i>" if answer_text else f"🤖 Asking AIM: {qtext}\n⏳ Processing...",
            parse_mode=ParseMode.HTML))
    try:
        await bot.answer_inline_query(inline_query_id=qid, results=[result], cache_time=0, is_personal=True)
    except Exception as e: logger.error("Inline answer failed: %s", e)


async def process_inline_answer(chat_id: int, message_id: int, query_text: str, user_id: str):
    try:
        profile = await get_user_profile_data(user_id)
        recent, older, gap = await get_conversation_context(user_id, query_text)
        r_text = await get_ai_response(query_text, user_id, "private", profile, recent_history=recent, older_context=older, gap_seconds=gap)
        if r_text:
            answer = r_text.strip()
            topic  = await extract_topic(query_text, answer)
            await save_chat_memory(user_id, "", query_text, answer, "inline", topic)
            await update_user_profile(user_id, "", topic)
            await send_text_chunks(chat_id, f"🤖 <b>AIM says:</b>\n\n{answer}", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "🔥 High demand — try again in 30 seconds.", reply_to=message_id)
    except Exception as e:
        logger.error("Inline answer error: %s", e)
        await send_text_chunks(chat_id, "🛠️ Something went wrong. Try again shortly.", reply_to=message_id)


def is_inline_placeholder(text: str) -> Tuple[bool, str]:
    if not text: return False, ""
    tl = text.strip().lower()
    if "asking aim" not in tl: return False, ""
    if "processing" not in tl and "thinking" not in tl: return False, ""
    parts = text.strip().split(":", 1)
    if len(parts) < 2: return False, ""
    q = parts[1].strip().split("\n")[0].replace("⏳","").replace("Processing...","").replace("Thinking...","").strip()
    if q:
        logger.info("🎯 Inline placeholder → '%s'", q)
        return True, q
    return False, ""


# ═══════════════════════════════════════════════════════════
# MAIN MESSAGE HANDLER
# ═══════════════════════════════════════════════════════════

async def handle_message_async(update: Update):
    if not update.message: return

    user       = update.message.from_user
    chat       = update.message.chat
    user_text  = update.message.text or ""
    chat_type  = chat.type if chat else "private"
    message_id = update.message.message_id

    # Voice transcription
    if update.message.voice or update.message.audio:
        file_obj = update.message.voice or update.message.audio
        await send_text_chunks(chat.id, "🎙️ Listening...", reply_to=message_id)
        transcribed = await transcribe_voice(file_obj.file_id)
        if transcribed:
            user_text = transcribed
            await send_text_chunks(chat.id, f"📝 You said: \"{user_text}\"", reply_to=message_id)
        else:
            await send_text_chunks(chat.id, "🎤 Sorry, couldn't understand the voice note.", reply_to=message_id)
            return

    if not user_text:
        await send_text_chunks(chat.id, "I can only read text and voice messages for now.")
        return

    user_id  = str(user.id)
    username = user.username or user.first_name or "User"
    logger.info("📩 [%s/%s] '%s'", user_id, chat_type, user_text[:80])

    # /commands
    if user_text.startswith("/"):
        if await handle_bot_command(user_id, chat.id, message_id, user_text):
            return

    profile = await get_user_profile_data(user_id)

    # Inline placeholder
    is_ph, ph_query = is_inline_placeholder(user_text)
    if is_ph and ph_query:
        await process_inline_answer(chat.id, message_id, ph_query, user_id)
        return

    # Memory search
    if is_memory_search_query(user_text):
        kws    = extract_search_keywords(user_text)
        result = await search_memory_by_keyword(user_id, user_text) if kws else await search_memory(user_id)
        await send_text_chunks(chat.id, result, reply_to=message_id)
        return

    # ── TASK DETECTION — runs for ALL chat types (FIX: was inside group block before)
        # ── TASK DETECTION — runs for ALL chat types
    if await handle_task_message(user_text, user_id, chat.id, message_id):
        # FIX: Save task interaction to memory so AIM knows what happened
        try:
            # Get AIM's last response (the task confirmation)
            last_response = supabase.table("chat_memory").select("response").eq("user_id", str(user_id)).order("created_at", desc=True).limit(1).execute()
            ai_response = last_response.data[0]["response"] if last_response.data else "Task created"
            
            # Save to memory
            await save_chat_memory(user_id, username, user_text, ai_response, chat_type, "reminder")
            await update_user_profile(user_id, username, "reminder")
        except Exception as e:
            logger.error("Failed to save task interaction to memory: %s", e)
        return

    # Group mention filter (AFTER task detection)
    if chat_type in ("group", "supergroup"):
        mentioned     = "@askaimbot" in user_text.lower()
        replied_to_bot = (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user and
            update.message.reply_to_message.from_user.is_bot and
            update.message.reply_to_message.from_user.username == "askaimbot"
        )
        if not mentioned and not replied_to_bot:
            return
        user_text = re.sub(r'@askaimbot', '', user_text, flags=re.IGNORECASE).strip()

    try:
        session_summary              = await get_session_summary(user_id)
        recent_history, older_context, gap_seconds = await get_conversation_context(user_id, user_text)

        web_context = ""
        for url in detect_urls(user_text):
            c = fetch_url_content(url)
            if c and "Failed" not in c: web_context += f"Content from {url}:\n{c}\n"

        if is_search_query(user_text) and not web_context:
            if "news" in user_text.lower() or "latest" in user_text.lower() or "today" in user_text.lower():
                sr = get_latest_news(user_text)
            elif any(s in user_text.lower() for s in ["football","match","score","team","player","league","f1","nba","tennis","boxing","ufc","cricket","rugby"]):
                sr = get_sports_data(user_text)
            else:
                sr = search_web(user_text)
            if "No search results" not in sr:
                web_context = f"Web Search Results for '{user_text}':\n{sr}"

        max_iter, iteration, final_answer, tool_status = 3, 0, None, ""

        while iteration < max_iter:
            iteration += 1
            logger.info("🔄 Agentic iteration %d", iteration)

            answer = await get_ai_response(
                user_text, user_id, chat_type, profile,
                session_summary, recent_history, older_context, web_context, tool_status, gap_seconds,
            )

            if not answer:
                final_answer = "🔥 High demand right now — please try again in 30 seconds."
                break

            answer = answer.strip()

            if "SEARCH_TRIGGER:" in answer:
                m = re.search(r'SEARCH_TRIGGER:\s*(.+)', answer, re.IGNORECASE)
                if m:
                    sq = m.group(1).strip()
                    logger.info("🔍 SEARCH_TRIGGER: '%s'", sq)
                    if "news" in sq.lower() or "latest" in sq.lower(): sr = get_latest_news(sq)
                    elif any(s in sq.lower() for s in ["football","match","score","f1","nba","tennis","boxing"]): sr = get_sports_data(sq)
                    else: sr = search_web(sq)
                    if sr == "No search results found.":
                        web_context += f"\n\nSearch for '{sq}': No results. Tell user you couldn't find current info."
                    else:
                        web_context += f"\n\nWeb Search Results for '{sq}':\n{sr}"
                    continue
                else:
                    final_answer = answer; break

            # Timer
            tm = re.search(r'\[TIMER:(\d+)(s|m|h)\]', answer, re.IGNORECASE)
            if tm:
                amt, unit = int(tm.group(1)), tm.group(2).lower()
                dur    = amt * (1 if unit=="s" else 60 if unit=="m" else 3600)
                target = datetime.now(timezone.utc) + timedelta(seconds=dur)
                supabase.table("user_tools").insert({
                    "user_id": user_id, "tool_type": "timer",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": dur, "target_time": target.isoformat(), "is_active": True,
                }).execute()
                answer      = re.sub(r'\[TIMER:\d+[smh]\]', '', answer, flags=re.IGNORECASE).strip()
                tool_status = f"✅ Timer set for {amt}{unit}"
                answer     += f"\n\n_{tool_status}_"

            # Stopwatch
            sm = re.search(r'\[STOPWATCH:(START|STOP)\]', answer, re.IGNORECASE)
            if sm:
                action = sm.group(1).upper()
                if action == "START":
                    supabase.table("user_tools").insert({
                        "user_id": user_id, "tool_type": "stopwatch",
                        "start_time": datetime.now(timezone.utc).isoformat(), "is_active": True,
                    }).execute()
                    answer = re.sub(r'\[STOPWATCH:START\]','',answer,flags=re.IGNORECASE).strip()
                    answer += "\n\n_⏱️ Stopwatch started!_"
                elif action == "STOP":
                    res = supabase.table("user_tools").select("*").eq("user_id",user_id).eq("tool_type","stopwatch").eq("is_active",True).order("created_at",desc=True).limit(1).execute()
                    if res.data:
                        row     = res.data[0]
                        elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row["start_time"].replace("Z","+00:00"))
                        mins, secs = divmod(int(elapsed.total_seconds()), 60)
                        supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                        ts     = f"{mins}m {secs}s" if mins else f"{secs}s"
                        answer = re.sub(r'\[STOPWATCH:STOP\]','',answer,flags=re.IGNORECASE).strip()
                        answer += f"\n\n_⏱️ Stopped! Time: {ts}_"

            final_answer = answer
            break

        if final_answer is None:
            final_answer = "I tried searching but couldn't find results right now. Please try again."

        await send_text_chunks(chat.id, final_answer, reply_to=message_id)

        topic = await extract_topic(user_text, final_answer)
        await save_chat_memory(user_id, username, user_text, final_answer, chat_type, topic)
        await update_user_profile(user_id, username, topic)

        # Rolling summary every 4 messages
        if profile.get("total_chats", 0) % 4 == 0:
            recent_msgs = (supabase.table("chat_memory").select("message,response")
                          .eq("user_id",str(user_id)).order("created_at",desc=True).limit(4).execute())
            if recent_msgs.data:
                recent_msgs.data.reverse()
                run_async(update_session_summary(user_id, recent_msgs.data, session_summary))

    except Exception as e:
        logger.error("Critical error in message handler: %s", e)
        await send_text_chunks(chat.id, "🛠️ Something went wrong. Try again shortly.", reply_to=message_id)


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "AIM Bot is live!",
        "version": "v9.4",
        "ai": "DeepSeek V4 Pro" if USE_DEEPSEEK else "Gemini",
        "search": "Brave Scrape → Brave API → DDG Lite",
        "apis": {"gnews": "✅" if GNEWS_API_KEY else "❌", "brave": "✅" if BRAVE_API_KEY else "❌"},
        "features": ["task_reminders_fixed","rolling_memory","time_gap_awareness","voice_stt",
                     "multi_language","all_sports","news_api","agentic_loop","deep_research",
                     "bot_commands","inline_mode","switchable_ai"],
    })

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        uid  = data.get("update_id")
        if uid and is_duplicate_update(uid): return "OK", 200
        upd = Update.de_json(data, bot)
        if upd.inline_query: run_async(handle_inline_query_async(upd.inline_query))
        elif upd.message: run_async(handle_message_async(upd))
        return "OK", 200
    except Exception as e:
        logger.error("Webhook error: %s", e)
        return "Error", 500

@app.route("/set-webhook", methods=["GET"])
def set_webhook():
    if not bot or not WEBHOOK_URL: return jsonify({"error": "Not configured"}), 500
    try:
        bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")
        return jsonify({"status": "Webhook set!", "url": f"{WEBHOOK_URL}/webhook"})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    if not bot: return jsonify({"error": "Bot not configured"}), 500
    try:
        bot.delete_webhook()
        return jsonify({"status": "Webhook deleted!"})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/debug/supabase", methods=["GET"])
def debug_supabase():
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        cr = supabase.table("chat_memory").select("*", count="exact").execute()
        pr = supabase.table("user_profiles").select("*", count="exact").execute()
        return jsonify({
            "status": "connected",
            "chat_memory_rows":   getattr(cr, 'count', len(cr.data)),
            "user_profiles_rows": getattr(pr, 'count', len(pr.data)),
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/debug/search", methods=["GET"])
def debug_search():
    q = request.args.get("q","").strip()
    if not q: return jsonify({"error": "Provide ?q=your+query"}), 400
    try:
        if "news" in q.lower(): results = get_latest_news(q)
        elif any(s in q.lower() for s in ["football","match","score","f1","nba"]): results = get_sports_data(q)
        else: results = search_web(q)
        return jsonify({"query": q, "results": results})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/debug/tasks/<user_id>", methods=["GET"])
def debug_tasks(user_id: str):
    """NEW: Inspect all tasks for a user — helpful for diagnosing task issues."""
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        tasks = supabase.table("user_tasks").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        now_utc = datetime.now(timezone.utc).isoformat()
        return jsonify({
            "user_id": user_id,
            "server_time_utc": now_utc,
            "task_count": len(tasks.data),
            "tasks": tasks.data,
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/memory/<user_id>", methods=["GET"])
def get_memory(user_id: str):
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        rows = supabase.table("chat_memory").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        return jsonify({"user_id": user_id, "chats": rows.data})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/profile/<user_id>", methods=["GET"])
def get_profile(user_id: str):
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        return jsonify({"user_id": user_id, "profile": rows.data[0] if rows.data else None})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/privacy", methods=["GET"])
def privacy_policy():
    try:
        with open("privacy.html", "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    except FileNotFoundError:
        return "<h1>Privacy Policy</h1><p>Coming soon.</p>", 200, {"Content-Type": "text/html"}

# ─── MAIN ───
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)