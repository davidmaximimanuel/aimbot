"""
AIM Bot v9.3 — African Intelligence Model (DeepSeek V4 Pro + API Integration)
Supports both Gemini and DeepSeek V4 Pro - switch via USE_DEEPSEEK variable
Sports search uses GNews Sports category (legal & reliable)
"""

import os
import tempfile
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
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
USE_DEEPSEEK = os.environ.get("USE_DEEPSEEK", "false").lower() == "true"  # Switch

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
WEBHOOK_URL     = os.environ.get("WEBHOOK_URL", "")
BRAVE_API_KEY   = os.environ.get("BRAVE_API_KEY", "")
GNEWS_API_KEY   = os.environ.get("GNEWS_API_KEY", "")
SPORTAPI_KEY    = os.environ.get("SPORTAPI_KEY", "")  # Kept for backward compatibility but no longer used

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

# AI Client Setup - Switchable
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
deepseek_client: Optional[AsyncOpenAI] = None

if USE_DEEPSEEK and DEEPSEEK_API_KEY:
    deepseek_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    logger.info("✅ Using DeepSeek V4 Pro API")
elif GEMINI_API_KEY:
    logger.info("✅ Using Gemini API")
else:
    logger.warning("⚠️ No AI API configured!")

# Groq Client for Voice Transcription (STT)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
if GROQ_API_KEY: logger.info("✅ Groq API key found (Voice STT enabled)")

async def transcribe_voice(file_id: str) -> Optional[str]:
    """Transcribes a voice note using Groq Whisper."""
    if not groq_client: return None
    temp_path = f"voice_{file_id}.ogg"
    try:
        file = await bot.get_file(file_id)
        await file.download_to_drive(custom_path=temp_path)
        
        with open(temp_path, "rb") as audio_file:
            transcription = await groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
        return transcription.strip()
    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ═══════════════════════════════════════════════════════════
# TASKS SYSTEM (Natural Language + Recurring + One-Time)
# ═══════════════════════════════════════════════════════════

TASK_KEYWORDS = ["remind me", "set a reminder", "remind", "task", "todo", "set an alarm", "notify me"]

async def parse_task_with_ai(user_text: str, user_id: str) -> dict:
    """Uses DeepSeek to parse natural language into structured task data."""
    now_wat = datetime.now(timezone(timedelta(hours=1)))
    current_time_str = now_wat.strftime("%A, %B %d, %Y at %I:%M %p WAT")
    
    prompt = f"""Parse this user message into a task. Return ONLY valid JSON. No markdown, no extra text.
Fields:
- description (string): clear task description
- type ("one_time"|"recurring")
- scheduled_time (ISO 8601 string or null): for one_time only
- recurrence_pattern ("daily"|"weekly"|"monthly"|null)
- recurrence_time ("HH:MM" string or null)
- recurrence_days (array of lowercase day names ["monday","wednesday"] or null)
- category ("reminder"|"news"|"verse"|"word"|"custom")
- needs_clarification (boolean)
- clarification_question (string, empty if not needed)

Rules:
- "remind me at 6pm to cook" -> one_time, scheduled_time=today 18:00 WAT
- "always remind me at 6pm to cook" -> recurring, pattern=daily, time=18:00
- "every monday at 9am send me news" -> recurring, pattern=weekly, days=["monday"], time=09:00, category=news
- "remind me in 2 hours" -> one_time, scheduled_time=now+2h
- If unclear, set needs_clarification=true and ask a short question
Current time: {current_time_str}
User message: "{user_text}"
"""
    
    try:
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=300
            )
            raw = r.choices[0].message.content.strip()
            # Clean markdown if DeepSeek wraps it
            if raw.startswith("```json"):
                raw = raw.replace("```json", "").replace("```", "").strip()
            elif raw.startswith("```"):
                raw = raw.replace("```", "").strip()
            return json.loads(raw)
    except Exception as e:
        logger.error(f"Task parsing error: {e}")
    return {"needs_clarification": True, "clarification_question": "Could you clarify when and how often you want this reminder?"}


async def create_task_in_db(user_id: str, task_data: dict) -> str:
    """Saves parsed task to Supabase and returns success/failure message."""
    try:
        now_wat = datetime.now(timezone(timedelta(hours=1)))
        
        # Calculate next_run based on type
        next_run = None
        if task_data["type"] == "one_time" and task_data.get("scheduled_time"):
            next_run = datetime.fromisoformat(task_data["scheduled_time"].replace("Z", "+00:00"))
            # If scheduled time is in past, assume tomorrow
            if next_run < now_wat:
                next_run += timedelta(days=1)
        elif task_data["type"] == "recurring":
            # Calculate first occurrence
            base_time = now_wat
            if task_data.get("recurrence_time"):
                h, m = map(int, task_data["recurrence_time"].split(":"))
                base_time = base_time.replace(hour=h, minute=m, second=0, microsecond=0)
                if base_time < now_wat:
                    base_time += timedelta(days=1)
            next_run = base_time

        # Insert into Supabase
        supabase.table("user_tasks").insert({
            "user_id": str(user_id),
            "task_description": task_data["description"],
            "task_type": task_data["type"],
            "scheduled_time": task_data.get("scheduled_time"),
            "recurrence_pattern": task_data.get("recurrence_pattern"),
            "recurrence_time": task_data.get("recurrence_time"),
            "recurrence_days": task_data.get("recurrence_days", []),
            "task_category": task_data.get("category", "reminder"),
            "is_active": True,
            "next_run": next_run.isoformat() if next_run else None
        }).execute()
        
        type_text = "daily" if task_data["type"] == "recurring" else "once"
        return f"✅ Task saved! I'll remind you {type_text}: \"{task_data['description']}\""
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        return "❌ Couldn't save the task. Please try again."


async def handle_task_message(user_text: str, user_id: str, chat_id: int, message_id: int) -> bool:
    """Detects task-like messages and handles them. Returns True if handled."""
    if not any(kw in user_text.lower() for kw in TASK_KEYWORDS):
        return False
    
    # Check if user is responding to a clarification
    recent = supabase.table("chat_memory").select("message, response").eq("user_id", str(user_id)).order("created_at", desc=True).limit(1).execute()
    if recent.data and "clarification_question" in recent.data[0].get("response", ""):
        # User is answering clarification, re-parse with context
        combined = f"Original: {recent.data[0]['message']}\nClarification: {user_text}"
        task_data = await parse_task_with_ai(combined, user_id)
    else:
        task_data = await parse_task_with_ai(user_text, user_id)
    
    if task_data.get("needs_clarification"):
        await send_text_chunks(chat_id, f"🤔 {task_data['clarification_question']}", reply_to=message_id)
        return True
    
    result = await create_task_in_db(user_id, task_data)
    await send_text_chunks(chat_id, result, reply_to=message_id)
    return True

if BRAVE_API_KEY:
    logger.info("✅ Brave Search API key found")
else:
    logger.warning("⚠️ BRAVE_API_KEY not set — using DuckDuckGo Lite")

if GNEWS_API_KEY:
    logger.info("✅ GNews API key found")
else:
    logger.warning("⚠️ GNEWS_API_KEY not set")

if SPORTAPI_KEY:
    logger.info("ℹ️ SPORTAPI_KEY found (deprecated - sports now uses GNews)")

# ─── SEMANTIC ROUTER ───
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
    "formula 1 result", "f1 race winner", "grand prix results", "max verstappen", "lewis hamilton",
    "nba score", "basketball result", "lebron james", "steph curry",
    "tennis result", "wimbledon winner", "us open tennis", "rafael nadal",
    "boxing match result", "ufc fight night", "mma result",
    "rugby result", "cricket score", "ipl result", "who won the super bowl", "nfl score",
    "who stopped them from qualifying", "who stopped nigeria", "who knocked nigeria out",
    "why did nigeria not qualify", "who beat nigeria", "did nigeria qualify",
    "nigeria world cup", "super eagles result", "super eagles match", "afcon result",
    "african cup of nations", "world cup qualification africa", "who qualified for the world cup",
    "who failed to qualify", "nigeria football history", "nigeria vs",
    "what happened to nigeria in", "past world cup results", "previous tournament result",
    "historical match result", "sports history africa", "who beat who in football",
    "football result nigeria", "african football news", "premier league result",
    "champions league result", "world cup result", "who won the world cup",
    "who won afcon", "which team won", "which country qualified",
    "who invented", "what caused", "why did", "how did", "when did", "what year did",
    "tell me about", "give me information on", "what do you know about",
    "news about", "update on", "facts about", "history of", "background on",
    "what is going on with", "recent news", "what happened with", "explain what happened",
    "naira exchange rate", "dollar to naira", "fuel price nigeria",
    "nigerian government", "nigerian politics", "tinubu", "lagos news", "abuja news",
    "nigeria economy", "cbdc nigeria", "enaira", "nigeria inflation", "nigeria election",
    "nass", "nigerian army", "borno attack", "bandits", "kidnapping nigeria",
    "nigeria insecurity",
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

# ─── BACKGROUND TIMER WORKER ───
def check_timers_background():
    logger.info("⏲️ Timer background worker started.")
    while True:
        time.sleep(30)
        if not supabase or not bot:
            continue
        try:
            now_utc = datetime.now(timezone.utc).isoformat()
            res = (supabase.table("user_tools")
                   .select("*").eq("tool_type", "timer")
                   .eq("is_active", True).lte("target_time", now_utc).execute())
            if res.data:
                for row in res.data:
                    user_id = row["user_id"]
                    duration = row.get("duration_seconds", 0)
                    supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                    mins, secs = divmod(duration, 60)
                    time_str = ""
                    if mins: time_str += f"{mins} minute{'s' if mins != 1 else ''}"
                    if secs:
                        if time_str: time_str += f" and {secs} second{'s' if secs != 1 else ''}"
                        else: time_str = f"{secs} second{'s' if secs != 1 else ''}"
                    run_async(bot.send_message(chat_id=int(user_id),
                                               text=f"⏰ Time's up! Your {time_str.strip()} timer is over."))
                    logger.info("⏰ Timer fired for user %s", user_id)
        except Exception as e:
            logger.error("Timer background check error: %s", e)

# ─── BACKGROUND TASK WORKER ───
def check_tasks_background():
    logger.info(" Task background worker started.")
    while True:
        time.sleep(30)  # Check every 30 seconds
        if not supabase or not bot:
            continue
        try:
            now_utc = datetime.now(timezone.utc).isoformat()
            # Get all active tasks that are due
            res = (supabase.table("user_tasks")
                   .select("*")
                   .eq("is_active", True)
                   .lte("next_run", now_utc)
                   .order("next_run", desc=False)
                   .execute())
            
            if not res.data:
                continue
                
            for task in res.data:
                user_id = task["user_id"]
                task_id = task["id"]
                description = task["task_description"]
                task_type = task["task_type"]
                
                # Send notification
                msg = f"⏰ Task Reminder:\n\n{description}"
                run_async(bot.send_message(chat_id=int(user_id), text=msg))
                logger.info(f"📋 Task fired for user {user_id}: {description}")
                
                if task_type == "one_time":
                    # Mark as completed/inactive
                    supabase.table("user_tasks").update({"is_active": False, "completed_at": datetime.now(timezone.utc).isoformat()}).eq("id", task_id).execute()
                elif task_type == "recurring":
                    # Calculate next occurrence
                    now_wat = datetime.now(timezone(timedelta(hours=1)))
                    pattern = task.get("recurrence_pattern")
                    days = task.get("recurrence_days", [])
                    rec_time = task.get("recurrence_time")
                    
                    next_run = now_wat
                    if rec_time:
                        h, m = map(int, rec_time.split(":"))
                        next_run = next_run.replace(hour=h, minute=m, second=0, microsecond=0)
                        if next_run <= now_wat:
                            next_run += timedelta(days=1)
                    
                    if pattern == "weekly" and days:
                        # Find next matching day
                        day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
                        current_day = next_run.weekday()
                        target_days = [day_map.get(d.lower(), 0) for d in days]
                        days_ahead = [(d - current_day) % 7 for d in target_days]
                        days_ahead = [d for d in days_ahead if d > 0] or [7]
                        next_run += timedelta(days=min(days_ahead))
                    elif pattern == "monthly":
                        next_run = (next_run.replace(day=1) + timedelta(days=32)).replace(day=1)
                    else:  # daily
                        next_run += timedelta(days=1)
                    
                    supabase.table("user_tasks").update({
                        "last_run": datetime.now(timezone.utc).isoformat(),
                        "next_run": next_run.isoformat()
                    }).eq("id", task_id).execute()
                    
        except Exception as e:
            logger.error("Task background check error: %s", e)

threading.Thread(target=check_tasks_background, daemon=True, name="task-worker").start()

threading.Thread(target=check_timers_background, daemon=True, name="timer-worker").start()

# ═══════════════════════════════════════════════════════════
# APIs (GNews for News + Sports, Brave/DDG for general search)
# ═══════════════════════════════════════════════════════════

def get_latest_news(query: str, max_results: int = 5) -> str:
    """Fetch latest news from GNews API."""
    if not GNEWS_API_KEY:
        return search_web(query, max_results)
    
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "apikey": GNEWS_API_KEY,
            "lang": "en",
            "country": "ng",
            "max": max_results,
        }
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code != 200:
            logger.warning("⚠️ GNews API status %s for '%s'", resp.status_code, query)
            return search_web(query, max_results)
        
        data = resp.json()
        articles = data.get("articles", [])
        
        if not articles:
            logger.info("ℹ️ GNews: 0 results for '%s'", query)
            return search_web(query, max_results)
        
        logger.info("✅ GNews: %d results for '%s'", len(articles), query)
        
        lines = []
        for i, article in enumerate(articles[:max_results], 1):
            title = article.get("title", "")
            desc = article.get("description", "")
            source = article.get("source", {}).get("name", "")
            url = article.get("url", "")
            published = article.get("publishedAt", "")[:16].replace("T", " ") if article.get("publishedAt") else ""
            
            lines.append(f"{i}. {title}\n   Source: {source} | {published}\n   Summary: {desc}\n   Link: {url}")
        
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("GNews API error: %s", e)
        return search_web(query, max_results)


def get_sports_data(query: str) -> str:
    """Fetch sports data using GNews Sports Category (Legal & Reliable)."""
    if not GNEWS_API_KEY:
        return search_web(query, 5)
    
    try:
        url = "https://gnews.io/api/v4/top-headlines"
        params = {
            "category": "sports",
            "apikey": GNEWS_API_KEY,
            "lang": "en",
            "country": "ng",
            "max": 5,
        }
        
        query_lower = query.lower()
        if "nigeria" in query_lower or "super eagles" in query_lower:
            params["q"] = "Nigeria football"
        elif "premier league" in query_lower or "epl" in query_lower:
            params["q"] = "Premier League"
        elif "champions league" in query_lower:
            params["q"] = "Champions League"
        elif "la liga" in query_lower:
            params["q"] = "La Liga"
        elif "serie a" in query_lower:
            params["q"] = "Serie A"
        elif "bundesliga" in query_lower:
            params["q"] = "Bundesliga"
        elif "afcon" in query_lower or "african cup" in query_lower:
            params["q"] = "AFCON"
        elif "world cup" in query_lower:
            params["q"] = "World Cup"
        elif "f1" in query_lower or "formula 1" in query_lower:
            params["q"] = "Formula 1"
        elif "nba" in query_lower or "basketball" in query_lower:
            params["q"] = "NBA basketball"
        elif "tennis" in query_lower:
            params["q"] = "tennis"
        elif "boxing" in query_lower or "ufc" in query_lower or "mma" in query_lower:
            params["q"] = "boxing MMA UFC"
        elif "cricket" in query_lower:
            params["q"] = "cricket"
        elif "rugby" in query_lower:
            params["q"] = "rugby"
        else:
            params["q"] = query

        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code != 200:
            logger.warning("⚠️ GNews Sports status %s for '%s'", resp.status_code, query)
            return search_web(query, 5)
        
        data = resp.json()
        articles = data.get("articles", [])
        
        if not articles:
            logger.info("ℹ️ GNews Sports: 0 results for '%s'", query)
            return search_web(query, 5)
        
        logger.info("✅ GNews Sports: %d results for '%s'", len(articles), query)
        
        lines = ["🏆 LATEST SPORTS UPDATES:\n"]
        for i, article in enumerate(articles[:5], 1):
            title = article.get("title", "")
            desc = article.get("description", "")
            source = article.get("source", {}).get("name", "")
            published = article.get("publishedAt", "")[:16].replace("T", " ") if article.get("publishedAt") else ""
            url = article.get("url", "")
            
            lines.append(f"{i}. {title}\n   Source: {source} | {published}\n   Summary: {desc}\n   Link: {url}")
        
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("GNews Sports error: %s", e)
        return search_web(query, 5)


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search pipeline:
    1. Brave Scraping (PRIMARY - best quality)
    2. Brave API (if key available)
    3. DuckDuckGo Lite (ultimate fallback)
    """
    results = _search_brave_scrape(query, max_results)
    provider = "Brave (Scraping)"
    
    if results is None:
        results = _search_brave_api(query, max_results)
        provider = "Brave (API)"
    
    if results is None:
        results = _search_duckduckgo_lite(query, max_results)
        provider = "DuckDuckGo Lite"
    
    if not results:
        logger.warning("❌ All search providers failed for '%s'", query)
        return "No search results found."

    logger.info("🔍 '%s' → %s (%d results)", query, provider, len(results))
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}")
    return "\n\n".join(lines)


def _search_brave_scrape(query: str, max_results: int = 5) -> Optional[list]:
    """Scrape Brave Search directly."""
    try:
        from urllib.parse import quote, unquote
        search_url = f"https://search.brave.com/search?q={quote(query)}&source=web"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        logger.info("🔍 Brave scraping: %s", search_url)
        resp = requests.get(search_url, headers=headers, timeout=15)
        
        if resp.status_code != 200:
            logger.warning("⚠️ Brave scrape status %s", resp.status_code)
            return None
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        
        snippets = soup.find_all('div', class_='snippet') or soup.find_all('div', {'data-pos': True}) or soup.find_all('div', class_='result')
        
        for snippet in snippets[:max_results]:
            title_elem = snippet.find('a', class_='result-title') or snippet.find('a', class_='title') or snippet.find('a', href=True)
            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')
                if url and not url.startswith('http'):
                    url = 'https://search.brave.com' + url
                
                desc_elem = snippet.find('p', class_='snippet-description') or snippet.find('div', class_='snippet-description')
                description = desc_elem.get_text(strip=True) if desc_elem else snippet.get_text(separator=' ', strip=True)[:300]
                
                if url and 'search.brave.com' in url and 'url=' in url:
                    url_match = re.search(r'url=([^&]+)', url)
                    if url_match:
                        url = unquote(url_match.group(1))
                
                if title and len(title) > 3:
                    results.append({"title": title, "description": description[:500], "url": url})
        
        if results:
            logger.info("✅ Brave scrape: %d results", len(results))
            return results[:max_results]
        
        logger.warning("⚠️ Brave scrape: 0 results parsed")
        return None
    except Exception as e:
        logger.error("Brave scrape error: %s", e)
        return None


def _search_brave_api(query: str, max_results: int = 5) -> Optional[list]:
    """Brave Search API (if key available)."""
    if not BRAVE_API_KEY:
        return None
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
    """DuckDuckGo Lite scraping (ultimate fallback)."""
    try:
        resp = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept-Language": "en-US,en;q=0.9"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for link in soup.find_all("a", class_="result-link")[:max_results]:
            title = link.get_text(strip=True)
            href = link.get("href", "")
            description = ""
            row = link.find_parent("tr")
            if row:
                next_row = row.find_next_sibling("tr")
                if next_row:
                    td = next_row.find("td", class_="result-snippet")
                    if td:
                        description = td.get_text(strip=True)
            if title and href:
                results.append({"title": title, "description": description, "url": href})
        return results if results else None
    except Exception as e:
        logger.error("DuckDuckGo Lite error: %s", e)
        return None


def deep_research(query: str) -> str:
    angle_queries = [query, f"{query} latest news", f"{query} results details"]
    all_results = []
    seen_urls = set()
    for q in angle_queries:
        results = _search_brave_scrape(q, 4) or _search_brave_api(q, 4) or _search_duckduckgo_lite(q, 4)
        if results:
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
    if not all_results:
        return "Deep research could not retrieve any results."
    lines = ["=== DEEP RESEARCH RESULTS ===\n"]
    for i, r in enumerate(all_results[:12], 1):
        lines.append(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}")
    return "\n\n".join(lines)


def fetch_url_content(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
        text = " ".join(soup.get_text(separator=" ", strip=True).split())
        return text[:3000]
    except Exception as e:
        logger.error("URL fetch error for %s: %s", url, e)
        return "Failed to read the link content."


def detect_urls(text: str) -> list:
    return re.findall(r'https?://\S+', text)


def is_search_query_semantic(text: str, threshold: float = 0.45) -> bool:
    try:
        q_emb = semantic_model.encode([text])
        sims = np.dot(trigger_embeddings, q_emb.T).flatten()
        max_sim = float(np.max(sims))
        best = SEARCH_TRIGGER_PHRASES[int(np.argmax(sims))]
        result = max_sim >= threshold
        logger.info("🔍 Semantic: '%.60s' → %.3f (best: '%s') → %s", text, max_sim, best, "SEARCH" if result else "skip")
        return result
    except Exception as e:
        logger.error("Semantic routing error: %s", e)
        return False


def is_search_query(text: str) -> bool:
    text_lower = text.lower().strip()
    explicit = ["search for", "google", "look up", "find out", "search the web", "browse", "search"]
    if any(t in text_lower for t in explicit):
        logger.info("🔍 Explicit search trigger")
        return True
    return is_search_query_semantic(text)


# ═══════════════════════════════════════════════════════════
# PROMPTING (v9.3 — DeepSeek V4 Pro)
# ═══════════════════════════════════════════════════════════

BASE_SYSTEM_PROMPT = """You are AIM — African Intelligence Model. You are a professional, highly intelligent AI assistant built for Africans, by Africans.

PERSONALITY & TONE:
- Warm, respectful, and culturally aware.
- Adapt to the user's vibe. If they are formal, be formal. If they are casual, be casual.
- Be helpful, patient, and empowering.

LANGUAGE RULE (CRITICAL):
- DO NOT restrict yourself to English. 
- Respond in the EXACT language or dialect the user is using (Pidgin, Yoruba, Hausa, Igbo, English, etc.).
- If the user mixes languages, you can mix them too.
- EXCEPTION: Never use rude, insulting, or hateful language.

RULES:
- Keep responses concise but informative.
- If you don't know something, use the SEARCH TRIGGER (see below).
- Never make up facts.
- Use emojis naturally but not excessively.

CAPABILITIES:
- Memory: You have a "Session Summary" of our recent chat. Use it for context.
- Time Tools: Timers and stopwatches.
- Time Awareness: You know the current time and when we last spoke.
- Web Search: Real-time web results available.
- Sports: You cover ALL sports (Football, F1, Basketball, Tennis, Boxing, UFC, Rugby, Cricket, etc.) via GNews Sports.
- News: Real-time news from Nigeria and worldwide via GNews.
- Entertainment: Movies, TV shows, reality TV, music, celebrities.

CONVERSATION CONTINUITY:
- Read the "SESSION SUMMARY" and "RECENT HISTORY" before responding.
- If the user uses pronouns (he, she, it), look at the immediate previous message to resolve them.
- If it has been a long time since the last message (check TIME CONTEXT), acknowledge it naturally (e.g., "Welcome back!", "Long time no see!").
- DO NOT start every message with "Hello there! It's [date]". Only mention time if relevant.

─────────────────────────────────────────
SPECIAL INSTRUCTIONS:

1. TIMERS/STOPWATCHES:
   Append machine code at the END of your response:
   - Timer: [TIMER:Xs] [TIMER:Xm] [TIMER:Xh]
   - Stopwatch: [STOPWATCH:START] [STOPWATCH:STOP]

2. SEARCH TRIGGER:
   Use for current events, live scores, prices, weather, or ANYTHING you aren't 100% sure of.
   Output EXACTLY: SEARCH_TRIGGER: <your search query>

3. WEB CONTEXT:
   If you see search results below, synthesize them. Do NOT output SEARCH_TRIGGER again.

4. GENERAL KNOWLEDGE:
   Answer directly if you are confident.
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


# ─── ROLLING MEMORY ENGINE ───
async def get_session_summary(user_id: str) -> str:
    """Fetches the session summary. Resets it if > 3 hours have passed."""
    if not supabase: return ""
    try:
        res = supabase.table("user_profiles").select("session_summary, last_active").eq("user_id", str(user_id)).execute()
        if not res.data: return ""
        
        profile = res.data[0]
        summary = profile.get("session_summary", "") or ""
        last_active_str = profile.get("last_active", "")
        
        if not last_active_str: return summary
        
        try:
            last_active = datetime.fromisoformat(last_active_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_since = (now - last_active).total_seconds() / 3600
            
            if hours_since > 3:
                logger.info("🕒 Time decay triggered (>3h). Clearing session summary for user %s", user_id)
                supabase.table("user_profiles").update({"session_summary": ""}).eq("user_id", str(user_id)).execute()
                return ""
            return summary
        except Exception:
            return summary
    except Exception as e:
        logger.error("Session summary fetch error: %s", e)
        return ""


async def update_session_summary(user_id: str, recent_messages: list, current_summary: str):
    """Uses AI to create a rolling summary of the conversation."""
    if not supabase: return
    
    try:
        msg_text = "\n".join([f"User: {m['message']}\nAIM: {m['response']}" for m in recent_messages])
        
        prompt = f"""Current Summary: {current_summary if current_summary else 'None yet'}
New Messages:
{msg_text}

Task: Create a concise, updated summary of the conversation. Include key facts about the user, ongoing topics, and important context. Keep it under 150 words."""

        if USE_DEEPSEEK and deepseek_client:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            new_summary = response.choices[0].message.content.strip() if response.choices else None
        elif gemini_client:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=200),
            )
            new_summary = response.text.strip() if response and response.text else None
        else:
            return
        
        if new_summary:
            supabase.table("user_profiles").update({"session_summary": new_summary}).eq("user_id", str(user_id)).execute()
            logger.info("📝 Session summary updated for user %s", user_id)
    except Exception as e:
        logger.error("Summarization error: %s", e)


def build_enhanced_prompt(
    user_text: str,
    user_id: str,
    profile: dict,
    session_summary: str = "",
    recent_history: str = "",
    older_context: str = "",
    web_context: str = "",
    tool_status: str = "",
) -> str:
    now_wat = datetime.now(timezone(timedelta(hours=1)))
    datetime_info = (
        f"{now_wat.strftime('%I:%M %p')} WAT, "
        f"{now_wat.strftime('%A, %B %d, %Y')} "
        f"({'morning' if 5 <= now_wat.hour < 12 else 'afternoon' if now_wat.hour < 17 else 'evening' if now_wat.hour < 21 else 'night'})"
    )

    parts = [BASE_SYSTEM_PROMPT.format(datetime_info=datetime_info)]

    pref_language = profile.get("preferred_language", "english")
    topic_counts  = profile.get("topic_counts", {})
    total_chats   = profile.get("total_chats", 0)
    pref_lines = [
        "\n--- USER PREFERENCES ---",
        f"- User ID: {user_id}",
        f"- Preferred language: {pref_language}",
        f"- Timezone: {profile.get('timezone', 'Africa/Lagos')}",
        f"- Total chats: {total_chats}",
    ]
    if topic_counts:
        top = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        pref_lines.append(f"- Top interests: {', '.join(f'{k}({v})' for k, v in top)}")
    pref_lines.append("--- END PREFERENCES ---\n")
    parts.append("\n".join(pref_lines))

    time_gap_info = ""
    try:
        last_msg = (supabase.table("chat_memory")
                   .select("created_at")
                   .eq("user_id", str(user_id))
                   .order("created_at", desc=True)
                   .limit(1)
                   .execute())
        
        if last_msg.data:
            last_time = datetime.fromisoformat(last_msg.data[0]["created_at"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            time_diff = now - last_time
            hours = time_diff.total_seconds() / 3600
            days = hours / 24
            
            if hours < 1:
                gap_text = f"{int(time_diff.total_seconds() / 60)} minutes ago"
            elif hours < 24:
                gap_text = f"{int(hours)} hours ago"
            elif days < 7:
                gap_text = f"{int(days)} days ago"
            else:
                gap_text = f"{int(days / 7)} weeks ago"
            
            time_gap_info = f"""
--- TIME CONTEXT ---
Current time: {now_wat.strftime('%A, %B %d, %Y at %I:%M %p')} WAT
User's last message: {gap_text}
If it's been more than 3 hours, acknowledge the time gap naturally.
DO NOT start every message with the date/time.
--- END TIME CONTEXT ---
"""
    except Exception as e:
        logger.error("Time gap calculation error: %s", e)
    
    if time_gap_info:
        parts.append(time_gap_info)

    if session_summary:
        parts.append(
            "\n╔══════════════════════════════════════╗\n"
            "║       SESSION SUMMARY                ║\n"
            "║  (Key context from our recent chat)  ║\n"
            "╚══════════════════════════════════════╝\n"
            + session_summary +
            "\n════════════════════════════════════════\n"
        )

    if recent_history:
        parts.append(
            "\n╔══════════════════════════════════════╗\n"
            "║   RECENT CONVERSATION HISTORY        ║\n"
            "║  (Read this first — immediate context)║\n"
            "╚══════════════════════════════════════╝\n"
            + recent_history +
            "\n════════════════════════════════════════\n"
        )

    if web_context:
        parts.append(
            "\n--- WEB SEARCH RESULTS (real-time data) ---\n"
            + web_context +
            "\n--- END WEB SEARCH RESULTS ---\n"
        )

    if older_context:
        parts.append(
            "\n--- OLDER RELEVANT MEMORY (background context) ---\n"
            + older_context +
            "\n--- END OLDER MEMORY ---\n"
        )

    if tool_status:
        parts.append(f"\n--- TOOL STATUS ---\n{tool_status}\n--- END TOOL STATUS ---\n")

    parts.append(f"\nUSER MESSAGE: {user_text}")
    return "\n".join(parts)


# ─── SWITCHABLE AI RESPONSE FUNCTION ───
async def get_ai_response(
    user_text: str,
    user_id: str,
    chat_type: str,
    profile: dict = None,
    session_summary: str = "",
    recent_history: str = "",
    older_context: str = "",
    web_context: str = "",
    tool_status: str = "",
) -> Optional[str]:
    """Calls either DeepSeek V4 Pro or Gemini based on USE_DEEPSEEK flag."""
    try:
        if profile is None:
            profile = await get_user_profile_data(user_id)
        prompt = build_enhanced_prompt(
            user_text, user_id, profile,
            session_summary, recent_history, older_context, web_context, tool_status
        )
        
        if USE_DEEPSEEK and deepseek_client:
            logger.info("🤖 Using DeepSeek V4 Pro for response")
            response = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content if response.choices else None
        elif gemini_client:
            logger.info("🤖 Using Gemini for response")
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
            )
            return response.text if response and response.text else None
        else:
            logger.error("❌ No AI client available!")
            return None
    except Exception as e:
        logger.error("AI response error: %s", e)
        return None


# ─── TOPIC EXTRACTION (Switchable) ───
async def extract_topic(user_text: str, bot_response: str) -> str:
    topics = ["career", "finance", "tech", "sports", "health", "relationships", 
              "politics", "entertainment", "education", "general"]
    
    prompt = f"""Classify this conversation into ONE topic from: {', '.join(topics)}.

User: {user_text[:200]}
AIM: {bot_response[:200]}

Examples:
- "I love my girlfriend" → relationships
- "My heart is troubled" → health
- "I'm stressed about work" → career
- "How to invest money" → finance
- "Who won the match" → sports
- "Python code error" → tech
- "Math homework help" → education
- "New movie release" → entertainment
- "Election results" → politics

Return ONLY the topic word, nothing else."""

    try:
        if USE_DEEPSEEK and deepseek_client:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            t = response.choices[0].message.content.strip().lower() if response.choices else "general"
        elif gemini_client:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20),
            )
            t = response.text.strip().lower() if response and response.text else "general"
        else:
            return "general"
        
        return t if t in topics else "general"
    except Exception:
        return "general"


# ─── MEMORY ───
async def save_chat_memory(user_id: str, username: str, message: str,
                           response: str, chat_type: str, topic: str = "general"):
    if not supabase: return
    try:
        supabase.table("chat_memory").insert({
            "user_id": str(user_id), "username": username or "",
            "message": message[:2000], "response": response[:2000],
            "chat_type": chat_type, "topic": topic,
        }).execute()
    except Exception as e:
        logger.error("❌ Memory save failed: %s", e)


async def update_user_profile(user_id: str, username: str, topic: str):
    if not supabase: return
    try:
        ex = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if ex.data:
            p = ex.data[0]
            tc = p.get("topic_counts", {})
            tc[topic] = tc.get(topic, 0) + 1
            supabase.table("user_profiles").update({
                "topic_counts": tc,
                "total_chats": p.get("total_chats", 0) + 1,
                "last_active": datetime.now(timezone.utc).isoformat(),
            }).eq("user_id", str(user_id)).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": str(user_id), "username": username or "",
                "topic_counts": {topic: 1}, "total_chats": 1,
                "last_active": datetime.now(timezone.utc).isoformat(),
            }).execute()
    except Exception as e:
        logger.error("❌ Profile update failed: %s", e)


async def get_conversation_context(user_id: str, query_text: str) -> tuple[str, str]:
    if not supabase:
        return "", ""
    try:
        rows = (supabase.table("chat_memory")
                .select("message, response, topic, created_at")
                .eq("user_id", str(user_id))
                .order("created_at", desc=True)
                .limit(30)
                .execute())
        if not rows.data:
            return "", ""

        recent_rows = list(reversed(rows.data[:5]))
        recent_lines = []
        for row in recent_rows:
            try:
                msg_time = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                wat_time = msg_time + timedelta(hours=1)
                time_str = wat_time.strftime("%a, %b %d at %I:%M %p")
            except:
                time_str = "unknown time"
            
            recent_lines.append(f"[{time_str}]")
            recent_lines.append(f"User: {row['message']}")
            recent_lines.append(f"AIM:  {row['response']}")
            recent_lines.append("")
        recent_history = "\n".join(recent_lines).strip()

        older_rows = rows.data[5:]
        if not older_rows:
            return recent_history, ""

        query_lower = query_text.lower()
        keyword_topics = {
            "space": ["tech"], "nigeria": ["general", "politics"],
            "money": ["finance"], "job": ["career"], "health": ["health"],
            "love": ["relationships"], "sport": ["sports"], "music": ["entertainment"],
            "school": ["education"], "code": ["tech"], "ai": ["tech"],
        }
        matched_topics = set()
        for kw, tps in keyword_topics.items():
            if kw in query_lower:
                matched_topics.update(tps)

        scored = []
        for row in older_rows:
            score = 0
            try:
                age_days = (datetime.now(timezone.utc) -
                            datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))).days
                score += max(0, 30 - age_days)
            except Exception:
                pass
            if row.get("topic") in matched_topics:
                score += 50
            blob = f"{row['message']} {row['response']}".lower()
            for word in query_lower.split():
                if len(word) > 3 and word in blob:
                    score += 10
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        older_lines = []
        for _, row in scored[:10]:
            older_lines.append(f"[{row.get('topic','general')}] User: {row['message']} | AIM: {row['response']}")
        older_context = "\n".join(older_lines)

        return recent_history, older_context

    except Exception as e:
        logger.error("Context retrieval error: %s", e)
        return "", ""


# ─── MEMORY SEARCH ───
def is_memory_search_query(user_text: str) -> bool:
    keywords = [
        "what did we talk about", "what have we discussed", "remember our chats",
        "our conversations", "what did i ask you", "what were we talking about",
        "we were talking about", "we discussed", "what did we say about",
        "remember when", "do you remember", "what about that time", "didn't we talk about",
    ]
    return any(kw in user_text.lower() for kw in keywords)


def extract_search_keywords(user_text: str) -> list:
    clean = user_text.lower()
    for phrase in ["what did we talk about", "what were we talking about", "we were talking about",
                   "tell me about", "what did we say about", "do you remember", "remember when",
                   "what about", "didn't we talk about", "what have we discussed"]:
        clean = clean.replace(phrase, "")
    clean = re.sub(r'[^\w\s]', ' ', clean)
    stop = {"the","and","about","were","did","have","what","when","that","this","with","for",
            "from","you","are","was","is","it","we","our","me","my","i","a","an","to","of",
            "in","on","at","be","been","being","do","does","say","said","get","got","go",
            "know","think","take","see","want","use","find","give","tell","ask","work"}
    return [w for w in clean.split() if len(w) > 2 and w not in stop]


async def search_memory_by_keyword(user_id: str, query_text: str) -> str:
    if not supabase: return "My memory is currently offline."
    try:
        keywords = extract_search_keywords(query_text)
        if not keywords: return await search_memory(user_id)
        all_results = []
        for kw in keywords[:3]:
            for field in ["message", "response"]:
                r = (supabase.table("chat_memory").select("*")
                     .eq("user_id", str(user_id))
                     .ilike(field, f"%{kw}%")
                     .order("created_at", desc=True).limit(5).execute())
                all_results.extend(r.data)
        topic_map = {"space":"tech","nigeria":"general","money":"finance","job":"career",
                     "health":"health","love":"relationships","sport":"sports",
                     "music":"entertainment","school":"education","code":"tech","ai":"tech"}
        for kw in keywords:
            if kw in topic_map:
                r = (supabase.table("chat_memory").select("*")
                     .eq("user_id", str(user_id)).eq("topic", topic_map[kw])
                     .order("created_at", desc=True).limit(5).execute())
                all_results.extend(r.data)
        seen, unique = set(), []
        for row in all_results:
            if row["id"] not in seen:
                seen.add(row["id"])
                unique.append(row)
        unique.sort(key=lambda x: x.get("created_at",""), reverse=True)
        if not unique:
            return f"I don't recall us discussing {' '.join(keywords)}. Want to start a conversation about it?"
        emoji_map = {"career":"💼","finance":"💰","tech":"💻","sports":"⚽","health":"🏥",
                     "relationships":"❤️","politics":"🏛️","entertainment":"🎬","education":"📚"}
        lines = [f"🔍 Found {len(unique)} conversation(s):"]
        for i, row in enumerate(unique[:5], 1):
            em = emoji_map.get(row.get("topic"), "💬")
            date = row.get("created_at","")[:10]
            lines.append(f'\n{i}. {em} [{date}] You: "{row["message"][:80]}..."')
            lines.append(f'   AIM: "{row["response"][:120]}..."')
        return "\n".join(lines)
    except Exception as e:
        logger.error("Keyword memory search error: %s", e)
        return "Having trouble searching memory right now."


async def search_memory(user_id: str) -> str:
    if not supabase: return "My memory is currently offline."
    try:
        pr = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if not pr.data: return "We haven't chatted before! Start a conversation so I can remember you."
        p = pr.data[0]
        tc = p.get("topic_counts", {})
        mr = (supabase.table("chat_memory")
              .select("message, response, topic, created_at")
              .eq("user_id", str(user_id)).order("created_at", desc=True).limit(10).execute())
        emoji_map = {"career":"💼","finance":"💰","tech":"💻","sports":"⚽","health":"🏥",
                     "relationships":"❤️","politics":"🏛️","entertainment":"🎬","education":"📚"}
        lines = [
            f"📊 Top Topics: {', '.join(f'{k} ({v}x)' for k,v in sorted(tc.items(), key=lambda x:x[1], reverse=True)[:3])}",
            f"💬 Total Chats: {p.get('total_chats',0)}", "", "📝 Recent:",
        ]
        for i, row in enumerate(mr.data[:5], 1):
            em = emoji_map.get(row.get("topic"), "💬")
            date = row.get("created_at","")[:10]
            lines.append(f"{i}. {em} [{date}] {row['message'][:60]}...")
        lines.append("\nWant me to dive deeper? Just ask!")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Memory search error: %s", e)
        return "Memory search is having issues right now."


# ─── BOT COMMANDS ───
async def handle_bot_command(user_id: str, chat_id: int, message_id: int, user_text: str) -> bool:
    tl = user_text.lower().strip()

    if tl.startswith("/help"):
        await send_text_chunks(chat_id, """🤖 <b>AIM Bot Commands</b>

<b>General:</b>
/help — Show this message
/tasks — View your tasks
/tasks delete <id> — Delete a task

<b>Web Search:</b>
/search [query] — Quick web search
/deep [query] — Deep multi-angle research

<b>Time Tools:</b>
/timer [time] — Set a timer (e.g. /timer 5m, /timer 30s, /timer 1h)
/stopwatch — Start or stop stopwatch

<b>Quick Tasks:</b>
/news daily 8am — Daily news at 8am
/verse daily — Daily bible verse
/word daily — Word of the day

<b>Natural Language also works:</b>
- "Remind me at 6pm to cook"
- "Always remind me to pray"
- "Every Monday send me news"
""", reply_to=message_id)
        return True

    elif tl.startswith("/search "):
        query = user_text[8:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔍 Searching...", reply_to=message_id)
        
        if "news" in query.lower() or "latest" in query.lower():
            results = get_latest_news(query)
        elif any(sport in query.lower() for sport in ["football", "match", "score", "team", "player", "f1", "nba", "tennis", "boxing"]):
            results = get_sports_data(query)
        else:
            results = search_web(query)
        
        if results == "No search results found.":
            await send_text_chunks(chat_id, "Couldn't find results for that. Try rephrasing.", reply_to=message_id)
            return True
        prompt = (f"User asked: {query}\n\nSearch Results:\n{results}\n\n"
                  "Answer using ONLY these results. Be concise. Do NOT output SEARCH_TRIGGER.")
        try:
            response_text = await get_ai_response(prompt, user_id, "private")
            txt = response_text.strip() if response_text else results
            if "SEARCH_TRIGGER:" in txt:
                txt = results
            await send_text_chunks(chat_id, txt, reply_to=message_id)
        except Exception as e:
            logger.error("Search command error: %s", e)
            await send_text_chunks(chat_id, results, reply_to=message_id)
        return True

    elif tl.startswith("/deep "):
        query = user_text[6:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔬 Researching from multiple angles...", reply_to=message_id)
        deep = deep_research(query)
        profile = await get_user_profile_data(user_id)
        r_text = await get_ai_response(query, user_id, "private", profile, "", "", "", deep)
        await send_text_chunks(chat_id, r_text if r_text else deep, reply_to=message_id)
        return True

    elif tl.startswith("/timer "):
        ts = user_text[7:].strip()
        m = re.match(r'(\d+)(s|m|h)', ts.lower())
        if m:
            amt, unit = int(m.group(1)), m.group(2)
            dur = amt * (1 if unit=="s" else 60 if unit=="m" else 3600)
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
        res = (supabase.table("user_tools").select("*")
               .eq("user_id", str(user_id)).eq("tool_type", "stopwatch")
               .eq("is_active", True).order("created_at", desc=True).limit(1).execute())
        if res.data:
            row = res.data[0]
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row["start_time"].replace("Z","+00:00"))
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
            ts = f"{mins}m {secs}s" if mins else f"{secs}s"
            await send_text_chunks(chat_id, f"⏱️ Stopped! Elapsed: {ts}", reply_to=message_id)
        else:
            supabase.table("user_tools").insert({
                "user_id": str(user_id), "tool_type": "stopwatch",
                "start_time": datetime.now(timezone.utc).isoformat(), "is_active": True,
            }).execute()
            await send_text_chunks(chat_id, "⏱️ Stopwatch started! Use /stopwatch again to stop.", reply_to=message_id)
        return True

    # ─── TASKS COMMANDS (NEW) ───
    elif tl == "/tasks":
        res = supabase.table("user_tasks").select("id, task_description, task_type, next_run, task_category").eq("user_id", str(user_id)).eq("is_active", True).order("next_run", desc=False).execute()
        if not res.data:
            await send_text_chunks(chat_id, "📋 You have no active tasks.", reply_to=message_id)
            return True
        lines = ["📋 <b>Your Tasks:</b>\n"]
        for t in res.data:
            next_time = datetime.fromisoformat(t["next_run"].replace("Z", "+00:00")) if t["next_run"] else None
            time_str = next_time.strftime("%b %d, %I:%M %p") if next_time else "Soon"
            type_icon = "🔁" if t["task_type"] == "recurring" else "1️⃣"
            lines.append(f"{type_icon} <code>{t['id'][:8]}</code> | {t['task_description']}\n   Next: {time_str}")
        lines.append("\n<i>Use /tasks delete &lt;id&gt; to remove a task</i>")
        await send_text_chunks(chat_id, "\n".join(lines), reply_to=message_id)
        return True

    elif tl.startswith("/tasks delete"):
        parts = tl.split()
        if len(parts) < 3:
            await send_text_chunks(chat_id, "❌ Use: /tasks delete <task_id>\n<i>Get the ID from /tasks</i>", reply_to=message_id)
            return True
        task_id = parts[2]
        # Get full ID from partial match
        res = supabase.table("user_tasks").select("id").eq("user_id", str(user_id)).eq("is_active", True).execute()
        full_id = None
        for t in res.data:
            if t["id"].startswith(task_id):
                full_id = t["id"]
                break
        if not full_id:
            await send_text_chunks(chat_id, "❌ Task not found. Check the ID from /tasks.", reply_to=message_id)
            return True
        supabase.table("user_tasks").update({"is_active": False}).eq("id", full_id).eq("user_id", str(user_id)).execute()
        await send_text_chunks(chat_id, "✅ Task deleted!", reply_to=message_id)
        return True

    elif tl.startswith("/news "):
        parts = user_text.split()
        if len(parts) < 3:
            await send_text_chunks(chat_id, "❌ Use: /news daily 8am  or  /news weekly monday 9am", reply_to=message_id)
            return True
        pattern = parts[1].lower()
        time_str = parts[2] if len(parts) > 2 else "08:00"
        days = parts[2:-1] if pattern == "weekly" else []
        task_data = {
            "description": "Send me news",
            "type": "recurring",
            "recurrence_pattern": pattern,
            "recurrence_time": time_str if ":" in time_str else f"{time_str}:00",
            "recurrence_days": [d.lower() for d in days],
            "category": "news",
            "needs_clarification": False
        }
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/verse daily":
        task_data = {"description": "Send me a daily bible verse", "type": "recurring", "recurrence_pattern": "daily", "recurrence_time": "08:00", "category": "verse", "needs_clarification": False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/word daily":
        task_data = {"description": "Send me word of the day", "type": "recurring", "recurrence_pattern": "daily", "recurrence_time": "09:00", "category": "word", "needs_clarification": False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    return False


# ─── SEND MESSAGE ───
async def send_text_chunks(chat_id: int, text: str,
                           reply_to: Optional[int] = None,
                           message_id: Optional[int] = None):
    if not bot: return
    try:
        if message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id,
                                        text=text[:TELEGRAM_MAX_CHARS], parse_mode=ParseMode.HTML)
        else:
            kw = {"chat_id": chat_id, "text": text[:TELEGRAM_MAX_CHARS], "parse_mode": ParseMode.HTML}
            if reply_to: kw["reply_to_message_id"] = reply_to
            await bot.send_message(**kw)
    except Exception as e:
        logger.error("Send/edit error: %s", e)
        try:
            if message_id:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS])
            else:
                await bot.send_message(chat_id=chat_id, text=text[:TELEGRAM_MAX_CHARS])
        except Exception as e2:
            logger.error("Fallback send failed: %s", e2)


# ─── INLINE QUERY ───
async def handle_inline_query_async(inline_query):
    qid   = inline_query.id
    qtext = inline_query.query.strip()
    uid   = str(inline_query.from_user.id) if inline_query.from_user else ""

    if not qtext or len(qtext) < 2:
        await bot.answer_inline_query(inline_query_id=qid, results=[], cache_time=1)
        return

    web_ctx = ""
    for url in detect_urls(qtext):
        c = fetch_url_content(url)
        if c and "Failed" not in c:
            web_ctx += f"Content from {url}:\n{c}\n"

    if is_search_query(qtext) and not web_ctx:
        if "news" in qtext.lower() or "latest" in qtext.lower():
            sr = get_latest_news(qtext)
        elif any(sport in qtext.lower() for sport in ["football", "match", "score", "team", "f1", "nba", "tennis", "boxing"]):
            sr = get_sports_data(qtext)
        else:
            sr = search_web(qtext)
        if "No search results" not in sr:
            web_ctx = f"Web Search Results for '{qtext}':\n{sr}"

    answer_text = None
    try:
        profile = await get_user_profile_data(uid)
        r_text = await asyncio.wait_for(
            get_ai_response(qtext, uid, "private", profile, "", "", "", web_ctx),
            timeout=15.0,
        )
        if r_text:
            answer_text = r_text.strip()[:300]
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        logger.error("Inline AI error: %s", e)

    result = InlineQueryResultArticle(
        id=str(uuid.uuid4()),
        title=f"AIM: {qtext[:30]}",
        description=(answer_text or "Click to get AIM's answer")[:100],
        input_message_content=InputTextMessageContent(
            message_text=(
                f"🤖 <b>AIM says:</b>\n\n{answer_text}\n\n<i>via @askaimbot</i>"
                if answer_text else
                f"🤖 Asking AIM: {qtext}\n⏳ Processing..."
            ),
            parse_mode=ParseMode.HTML,
        ),
    )
    try:
        await bot.answer_inline_query(inline_query_id=qid, results=[result], cache_time=0, is_personal=True)
    except Exception as e:
        logger.error("Inline answer failed: %s", e)


async def process_inline_answer(chat_id: int, message_id: int, query_text: str, user_id: str):
    try:
        profile = await get_user_profile_data(user_id)
        recent, older = await get_conversation_context(user_id, query_text)
        r_text = await get_ai_response(query_text, user_id, "private", profile, "", recent, older)
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
    q = parts[1].strip().split("\n")[0]
    q = q.replace("⏳","").replace("Processing...","").replace("Thinking...","").strip()
    if q:
        logger.info("🎯 Inline placeholder → query: '%s'", q)
        return True, q
    return False, ""


# ─── MAIN MESSAGE HANDLER ───
async def handle_message_async(update: Update):
    if not update.message: return

    user      = update.message.from_user
    chat      = update.message.chat
    user_text = update.message.text or ""
    chat_type = chat.type if chat else "private"
    message_id = update.message.message_id

    # --- VOICE MESSAGE HANDLING (STT) ---
    if update.message.voice or update.message.audio:
        file_obj = update.message.voice or update.message.audio
        await send_text_chunks(chat.id, "🎙️ Listening...", reply_to=message_id)
        transcribed = await transcribe_voice(file_obj.file_id)
        if transcribed:
            user_text = transcribed
            await send_text_chunks(chat.id, f"📝 You said: \"{user_text}\"", reply_to=message_id)
        else:
            await send_text_chunks(chat.id, "🎤 Sorry, I couldn't understand the voice note.", reply_to=message_id)
            return

    if not user_text:
        await send_text_chunks(chat.id, "I can only read text and voice messages for now.")
        return

    user_id  = str(user.id)
    username = user.username or user.first_name or "User"
    logger.info("📩 [%s/%s] '%s'", user_id, chat_type, user_text[:80])

    # --- COMMAND HANDLING ---
    if user_text.startswith("/"):
        if await handle_bot_command(user_id, chat.id, message_id, user_text):
            return

    profile = await get_user_profile_data(user_id)

    is_ph, ph_query = is_inline_placeholder(user_text)
    if is_ph and ph_query:
        await process_inline_answer(chat.id, message_id, ph_query, user_id)
        return

    if is_memory_search_query(user_text):
        kws = extract_search_keywords(user_text)
        result = (await search_memory_by_keyword(user_id, user_text)
                  if kws else await search_memory(user_id))
        await send_text_chunks(chat.id, result, reply_to=message_id)
        return

    if chat_type in ("group", "supergroup"):
        mentioned = "@askaimbot" in user_text.lower()
        replied_to_bot = (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user and
            update.message.reply_to_message.from_user.is_bot and
            update.message.reply_to_message.from_user.username == "askaimbot"
        )
        if not mentioned and not replied_to_bot:
            return
        user_text = re.sub(r'@askaimbot', '', user_text, flags=re.IGNORECASE).strip()

    # --- TASK DETECTION (MOVED TO CORRECT PLACE) ---
    # Only detect tasks if message contains specific task keywords
    task_keywords_strict = ["remind me to", "remind me at", "set a reminder", "set an alarm", "create a task"]
    if any(kw in user_text.lower() for kw in task_keywords_strict):
        logger.info("📋 Task keyword detected in message")
        if await handle_task_message(user_text, user_id, chat.id, message_id):
            return
    # -----------------------------------------------

    try:
        session_summary = await get_session_summary(user_id)
        recent_history, older_context = await get_conversation_context(user_id, user_text)

        web_context = ""
        for url in detect_urls(user_text):
            c = fetch_url_content(url)
            if c and "Failed" not in c:
                web_context += f"Content from {url}:\n{c}\n"

        if is_search_query(user_text) and not web_context:
            if "news" in user_text.lower() or "latest" in user_text.lower() or "today" in user_text.lower():
                logger.info("📰 Routing to GNews API")
                sr = get_latest_news(user_text)
            elif any(sport in user_text.lower() for sport in ["football", "match", "score", "team", "player", "league", "f1", "nba", "tennis", "boxing", "ufc", "cricket", "rugby"]):
                logger.info("⚽ Routing to GNews Sports")
                sr = get_sports_data(user_text)
            else:
                logger.info("🔍 Routing to web search")
                sr = search_web(user_text)
            
            if "No search results" not in sr:
                web_context = f"Web Search Results for '{user_text}':\n{sr}"

        max_iter = 3
        iteration = 0
        final_answer = None
        tool_status = ""

        while iteration < max_iter:
            iteration += 1
            logger.info("🔄 Agentic iteration %d", iteration)

            answer = await get_ai_response(
                user_text, user_id, chat_type, profile,
                session_summary, recent_history, older_context, web_context, tool_status,
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
                    if "news" in sq.lower() or "latest" in sq.lower():
                        sr = get_latest_news(sq)
                    elif any(sport in sq.lower() for sport in ["football", "match", "score", "team", "f1", "nba", "tennis", "boxing"]):
                        sr = get_sports_data(sq)
                    else:
                        sr = search_web(sq)
                    if sr == "No search results found.":
                        web_context += (f"\n\nSearch for '{sq}': No results found. "
                                        "Tell the user you couldn't find current info and offer to help otherwise.")
                    else:
                        web_context += f"\n\nWeb Search Results for '{sq}':\n{sr}"
                    continue
                else:
                    final_answer = answer
                    break

            tm = re.search(r'\[TIMER:(\d+)(s|m|h)\]', answer, re.IGNORECASE)
            if tm:
                amt, unit = int(tm.group(1)), tm.group(2).lower()
                dur = amt * (1 if unit=="s" else 60 if unit=="m" else 3600)
                target = datetime.now(timezone.utc) + timedelta(seconds=dur)
                supabase.table("user_tools").insert({
                    "user_id": user_id, "tool_type": "timer",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": dur, "target_time": target.isoformat(), "is_active": True,
                }).execute()
                answer = re.sub(r'\[TIMER:\d+[smh]\]', '', answer, flags=re.IGNORECASE).strip()
                tool_status = f"✅ Timer set for {amt}{unit}"
                answer += f"\n\n_{tool_status}_"

            sm = re.search(r'\[STOPWATCH:(START|STOP)\]', answer, re.IGNORECASE)
            if sm:
                action = sm.group(1).upper()
                if action == "START":
                    supabase.table("user_tools").insert({
                        "user_id": user_id, "tool_type": "stopwatch",
                        "start_time": datetime.now(timezone.utc).isoformat(), "is_active": True,
                    }).execute()
                    answer = re.sub(r'\[STOPWATCH:START\]', '', answer, flags=re.IGNORECASE).strip()
                    answer += "\n\n_⏱️ Stopwatch started!_"
                elif action == "STOP":
                    res = (supabase.table("user_tools").select("*")
                           .eq("user_id", user_id).eq("tool_type", "stopwatch")
                           .eq("is_active", True).order("created_at", desc=True).limit(1).execute())
                    if res.data:
                        row = res.data[0]
                        elapsed = (datetime.now(timezone.utc) -
                                   datetime.fromisoformat(row["start_time"].replace("Z","+00:00")))
                        mins, secs = divmod(int(elapsed.total_seconds()), 60)
                        supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                        ts = f"{mins}m {secs}s" if mins else f"{secs}s"
                        answer = re.sub(r'\[STOPWATCH:STOP\]', '', answer, flags=re.IGNORECASE).strip()
                        answer += f"\n\n_⏱️ Stopped! Time: {ts}_"

            final_answer = answer
            break

        if final_answer is None:
            final_answer = "I tried searching but couldn't find results right now. Please try again."

        await send_text_chunks(chat.id, final_answer, reply_to=message_id)
        
        topic = await extract_topic(user_text, final_answer)
        await save_chat_memory(user_id, username, user_text, final_answer, chat_type, topic)
        await update_user_profile(user_id, username, topic)
        
        if profile.get("total_chats", 0) % 4 == 0:
            recent_msgs = (supabase.table("chat_memory")
                          .select("message, response")
                          .eq("user_id", str(user_id))
                          .order("created_at", desc=True)
                          .limit(4)
                          .execute())
            if recent_msgs.data:
                recent_msgs.data.reverse()
                run_async(update_session_summary(user_id, recent_msgs.data, session_summary))

    except Exception as e:
        logger.error("Critical error in message handler: %s", e)
        await send_text_chunks(chat.id, "🛠️ Something went wrong. Try again shortly.", reply_to=message_id)


# ─── ROUTES ───
@app.route("/", methods=["GET"])
def health():
    ai_provider = "DeepSeek V4 Pro" if USE_DEEPSEEK else "Gemini"
    return jsonify({
        "status": "AIM Bot is live!",
        "version": "v9.3",
        "model": f"African Intelligence Model ({ai_provider})",
        "search_provider": "Brave Scraping → Brave API → DuckDuckGo Lite",
        "apis": {
            "gnews": "✅" if GNEWS_API_KEY else "❌",
            "brave": "✅" if BRAVE_API_KEY else "❌",
        },
        "features": ["rolling_memory", "time_decay", "multi_language", "all_sports_gnews",
                     "news_api", "smart_routing", "reliable_search",
                     "agentic_loop", "deep_research", "bot_commands", "inline_mode", "switchable_ai"],
    })

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        uid  = data.get("update_id")
        if uid and is_duplicate_update(uid):
            return "OK", 200
        upd = Update.de_json(data, bot)
        if upd.inline_query:
            run_async(handle_inline_query_async(upd.inline_query))
        elif upd.message:
            run_async(handle_message_async(upd))
        return "OK", 200
    except Exception as e:
        logger.error("Webhook error: %s", e)
        return "Error", 500

@app.route("/set-webhook", methods=["GET"])
def set_webhook():
    if not bot or not WEBHOOK_URL:
        return jsonify({"error": "Bot or webhook URL not configured"}), 500
    try:
        bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")
        return jsonify({"status": "Webhook set!", "url": f"{WEBHOOK_URL}/webhook"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    if not bot: return jsonify({"error": "Bot not configured"}), 500
    try:
        bot.delete_webhook()
        return jsonify({"status": "Webhook deleted!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/supabase", methods=["GET"])
def debug_supabase():
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        cr = supabase.table("chat_memory").select("*", count="exact").execute()
        pr = supabase.table("user_profiles").select("*", count="exact").execute()
        return jsonify({
            "status": "connected",
            "chat_memory_rows": getattr(cr, 'count', len(cr.data)),
            "user_profiles_rows": getattr(pr, 'count', len(pr.data)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/search", methods=["GET"])
def debug_search():
    q = request.args.get("q","").strip()
    if not q: return jsonify({"error": "Provide ?q=your+query"}), 400
    try:
        if "news" in q.lower():
            results = get_latest_news(q)
        elif any(sport in q.lower() for sport in ["football", "match", "score", "f1", "nba"]):
            results = get_sports_data(q)
        else:
            results = search_web(q)
        return jsonify({
            "query": q,
            "results": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memory/<user_id>", methods=["GET"])
def get_memory(user_id: str):
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        rows = (supabase.table("chat_memory").select("*")
                .eq("user_id", user_id).order("created_at", desc=True).limit(20).execute())
        return jsonify({"user_id": user_id, "chats": rows.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/profile/<user_id>", methods=["GET"])
def get_profile(user_id: str):
    if not supabase: return jsonify({"error": "Supabase not connected"}), 500
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        return jsonify({"user_id": user_id, "profile": rows.data[0] if rows.data else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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