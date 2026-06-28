"""
AIM Bot v9.7 — African Intelligence Model
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
import random
import string
import base64
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import nebulae
from telegram import InputMediaPhoto, InputMediaAudio, InputMediaDocument
from flask import Flask, request, jsonify, redirect
from telegram import (
    Update, Bot, InlineQueryResultArticle, InputTextMessageContent
)
from telegram.constants import ParseMode
from supabase import create_client, Client
from google import genai
from google.genai import types

# ── Module imports (deduplicated) ──────────────────────────
from core import BASE_SYSTEM_PROMPT, build_enhanced_prompt, WAT
from capabilities import is_search_query, SEARCH_TRIGGER_PHRASES, trigger_embeddings, semantic_model
# Import START_TIME from admin so both files share the SAME start reference.
# This is what fixes the stale uptime bug — admin.py sets it at import time
# on fresh deploy, and both files read from it.
from admin import load_admins, is_admin, handle_admin_command, ADMIN_IDS, START_TIME

# ─── LOGGING ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aimbot")

# ─── CONFIG ───
TELEGRAM_TOKEN      = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY    = os.environ.get("DEEPSEEK_API_KEY", "")
USE_DEEPSEEK        = os.environ.get("USE_DEEPSEEK", "false").lower() == "true"
SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY        = os.environ.get("SUPABASE_KEY", "")
WEBHOOK_URL         = os.environ.get("WEBHOOK_URL", "")
BRAVE_API_KEY       = os.environ.get("BRAVE_API_KEY", "")
GNEWS_API_KEY       = os.environ.get("GNEWS_API_KEY", "")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")

# ─── LOGTO CONFIG ───
LOGTO_ENDPOINT      = os.environ.get("LOGTO_ENDPOINT", "").rstrip("/")
LOGTO_CLIENT_ID     = os.environ.get("LOGTO_CLIENT_ID", "")
LOGTO_CLIENT_SECRET = os.environ.get("LOGTO_CLIENT_SECRET", "")

def get_redirect_uri() -> str:
    """Built lazily so Railway env vars are always resolved at call time."""
    base = os.environ.get("WEBHOOK_URL", "").rstrip("/")
    return f"{base}/auth/callback" if base else ""

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

# ─── IN-MEMORY OAUTH STATE STORE ───
_oauth_states: dict[str, dict] = {}
_oauth_states_lock = threading.Lock()

def _create_oauth_state(telegram_user_id: str, chat_id: int) -> str:
    state = secrets.token_urlsafe(32)
    with _oauth_states_lock:
        now = time.time()
        stale = [k for k, v in _oauth_states.items() if now - v["created_at"] > 600]
        for k in stale:
            del _oauth_states[k]
        _oauth_states[state] = {"telegram_user_id": telegram_user_id, "chat_id": chat_id, "created_at": now}
    return state

def _consume_oauth_state(state: str) -> Optional[dict]:
    with _oauth_states_lock:
        ctx = _oauth_states.pop(state, None)
    if ctx is None: return None
    if time.time() - ctx["created_at"] > 600: return None
    return ctx

def build_logto_auth_url(state: str) -> str:
    from urllib.parse import urlencode
    redirect_uri = get_redirect_uri()
    params = {
        "client_id":     LOGTO_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid profile email",
        "state":         state,
    }
    endpoint = os.environ.get("LOGTO_ENDPOINT", "").rstrip("/")
    return f"{endpoint}/oidc/auth?{urlencode(params)}"

def exchange_logto_code(code: str) -> Optional[dict]:
    try:
        endpoint   = os.environ.get("LOGTO_ENDPOINT", "").rstrip("/")
        client_id  = os.environ.get("LOGTO_CLIENT_ID", "")
        client_sec = os.environ.get("LOGTO_CLIENT_SECRET", "")
        token_url  = f"{endpoint}/oidc/token"
        resp = requests.post(token_url, data={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  get_redirect_uri(),
            "client_id":     client_id,
            "client_secret": client_sec,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=10)
        resp.raise_for_status()
        tokens = resp.json()
        id_token = tokens.get("id_token", "")
        if not id_token:
            logger.error("Logto: no id_token in response")
            return None
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception as e:
        logger.error("Logto token exchange error: %s", e)
        return None

# ─── EMPIRE ID ───
def generate_empire_id() -> str:
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"EMP-{random_str}"

# ─── FILE UTILS (used by admin commands, also available here) ───
def read_file_safely(filepath: str) -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    requested_path = os.path.abspath(os.path.join(base_dir, filepath))
    if not requested_path.startswith(base_dir):
        return "❌ Access Denied: Path traversal detected."
    if not os.path.exists(requested_path):
        return f"❌ File not found: {filepath}"
    try:
        with open(requested_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"❌ Error reading file: {e}"

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
    logger.info("✅ Using DeepSeek V4 API")
elif GEMINI_API_KEY:
    logger.info("✅ Using Gemini API")
else:
    logger.warning("⚠️ No AI API configured!")

groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
if GROQ_API_KEY:     logger.info("✅ Groq API (Voice STT enabled)")
if BRAVE_API_KEY:    logger.info("✅ Brave Search API")
if GNEWS_API_KEY:    logger.info("✅ GNews API")
_logto_ok = bool(os.environ.get("LOGTO_ENDPOINT")) and bool(os.environ.get("LOGTO_CLIENT_ID"))
if _logto_ok:
    logger.info("✅ Logto OAuth configured → %s", get_redirect_uri())
else:
    logger.warning("⚠️ Logto not configured — /link will not work")

# Load admins at startup
load_admins(supabase)

# ─── ASYNCIO EVENT LOOP ───
_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()
threading.Thread(target=_run_loop, daemon=True, name="async-loop").start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop)

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
    pattern   = task.get("recurrence_pattern", "daily")
    rec_time  = task.get("recurrence_time")
    days_list = task.get("recurrence_days") or []
    DAY_MAP   = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
    base = from_time
    if rec_time:
        try:
            h, m = map(int, rec_time.split(":"))
            base = base.replace(hour=h, minute=m, second=0, microsecond=0)
        except Exception: pass
    if base <= from_time:
        base += timedelta(days=1)
        if rec_time:
            try:
                h, m = map(int, rec_time.split(":"))
                base = base.replace(hour=h, minute=m, second=0, microsecond=0)
            except Exception: pass
    if pattern == "daily": return base
    elif pattern == "weekly" and days_list:
        target_days = [DAY_MAP[d.lower()] for d in days_list if d.lower() in DAY_MAP]
        if not target_days: return base
        for i in range(8):
            candidate = base + timedelta(days=i)
            if candidate.weekday() in target_days: return candidate
        return base + timedelta(days=7)
    elif pattern == "monthly":
        month = base.month + 1
        year  = base.year + (1 if month > 12 else 0)
        month = 1 if month > 12 else month
        try: return base.replace(year=year, month=month)
        except ValueError:
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
                   .eq("is_active", True).lte("next_run", now_utc)
                   .order("next_run", desc=False).execute())
            for task in res.data or []:
                user_id, task_id = task["user_id"], task["id"]
                description, task_type = task["task_description"], task["task_type"]
                category = task.get("task_category", "reminder")
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
                except Exception as send_err:
                    logger.error("Task send error: %s", send_err)
                if task_type == "one_time":
                    try:
                        supabase.table("user_tasks").update({"is_active": False, "completed_at": datetime.now(timezone.utc).isoformat()}).eq("id", task_id).execute()
                    except Exception:
                        supabase.table("user_tasks").update({"is_active": False}).eq("id", task_id).execute()
                elif task_type == "recurring":
                    now_wat  = datetime.now(WAT)
                    next_run = _calc_next_run(task, now_wat)
                    try:
                        supabase.table("user_tasks").update({"next_run": next_run.isoformat(), "last_run": datetime.now(timezone.utc).isoformat()}).eq("id", task_id).execute()
                    except Exception:
                        supabase.table("user_tasks").update({"next_run": next_run.isoformat()}).eq("id", task_id).execute()
        except Exception as e:
            logger.error("Task worker error: %s", e)

threading.Thread(target=check_tasks_background, daemon=True, name="task-worker").start()

# ═══════════════════════════════════════════════════════════
# TASK SYSTEM
# ═══════════════════════════════════════════════════════════
TASK_KEYWORDS      = ["remind me","set a reminder","reminder","remind","notify me","don't let me forget","alert me","schedule","every day","every week","every monday","every tuesday","every wednesday","every thursday","every friday","every saturday","every sunday","daily reminder","always remind"]
TIMER_ONLY_KEYWORDS = ["set a timer","start a timer","set timer","start timer","set a stopwatch","start stopwatch","stopwatch"]

async def parse_task_with_ai(user_text: str, user_id: str) -> dict:
    now_wat = datetime.now(WAT)
    current_time_str = now_wat.strftime("%A, %B %d, %Y at %I:%M %p WAT")
    prompt = f"""Parse this user message into a task. Return ONLY valid JSON.
Fields: description, type ("one_time"|"recurring"), scheduled_time (ISO or null), recurrence_pattern, recurrence_time, recurrence_days, category, needs_clarification, clarification_question.
Current time: {current_time_str}
User message: "{user_text}"
Return ONLY JSON:"""
    try:
        if USE_DEEPSEEK and deepseek_client:
            r   = await deepseek_client.chat.completions.create(model="deepseek-v4-flash", messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=400)
            raw = r.choices[0].message.content.strip()
        elif gemini_client:
            r   = gemini_client.models.generate_content(model="gemini-2.5-flash-lite", contents=[types.Content(role="user", parts=[types.Part(text=prompt)])], config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=400))
            raw = r.text.strip() if r and r.text else ""
        else:
            return {"needs_clarification": True, "clarification_question": "Could you clarify when you want this reminder?"}
        raw = re.sub(r'^```json\s*', '', raw.strip())
        raw = re.sub(r'^```\s*',     '', raw.strip())
        raw = re.sub(r'\s*```$',     '', raw.strip())
        return json.loads(raw.strip())
    except Exception as e:
        logger.error("Task parse error: %s", e)
        return {"needs_clarification": True, "clarification_question": "Could you clarify when you want this reminder?"}

async def create_task_in_db(user_id: str, task_data: dict) -> str:
    if not supabase: return "❌ Memory is offline."
    try:
        now_wat  = datetime.now(WAT)
        next_run = None
        if task_data.get("type") == "one_time" and task_data.get("scheduled_time"):
            try:
                st       = task_data["scheduled_time"]
                next_run = datetime.fromisoformat(st.replace("Z", "+00:00")) if "+" in st or st.endswith("Z") else datetime.fromisoformat(st).replace(tzinfo=WAT)
                if next_run < datetime.now(timezone.utc): next_run += timedelta(days=1)
            except Exception: next_run = now_wat + timedelta(hours=1)
        elif task_data.get("type") == "recurring":
            next_run = _calc_next_run(task_data, now_wat)
        row = {
            "user_id":            str(user_id),
            "task_description":   task_data.get("description", "Reminder"),
            "task_type":          task_data.get("type", "one_time"),
            "scheduled_time":     task_data.get("scheduled_time"),
            "recurrence_pattern": task_data.get("recurrence_pattern"),
            "recurrence_time":    task_data.get("recurrence_time"),
            "recurrence_days":    task_data.get("recurrence_days") or [],
            "task_category":      task_data.get("category", "reminder"),
            "is_active":          True,
            "next_run":           next_run.isoformat() if next_run else None,
        }
        supabase.table("user_tasks").insert(row).execute()
        desc, t_type = task_data.get("description", "your reminder"), task_data.get("type", "one_time")
        if t_type == "recurring":
            pattern, r_time, r_days = task_data.get("recurrence_pattern","daily"), task_data.get("recurrence_time",""), task_data.get("recurrence_days") or []
            schedule_str = f"every {', '.join(r_days)}" if pattern == "weekly" and r_days else pattern
            if r_time:
                try:
                    h, m = map(int, r_time.split(":"))
                    ampm, h12 = "AM" if h < 12 else "PM", h % 12 or 12
                    schedule_str += f" at {h12}:{m:02d} {ampm}"
                except Exception: schedule_str += f" at {r_time}"
            return f"✅ Got it! I'll remind you {schedule_str}: \"{desc}\""
        else:
            time_str = next_run.astimezone(WAT).strftime("%A, %b %d at %I:%M %p WAT") if next_run else "soon"
            return f"✅ Reminder set for {time_str}: \"{desc}\""
    except Exception as e:
        logger.error("Task creation error: %s", e)
        return "❌ Couldn't save the task."

async def handle_task_message(user_text: str, user_id: str, chat_id: int, message_id: int) -> bool:
    text_lower = user_text.lower()
    if any(kw in text_lower for kw in TIMER_ONLY_KEYWORDS): return False
    if not any(kw in text_lower for kw in TASK_KEYWORDS):   return False
    logger.info("📋 Task intent detected: '%s'", user_text[:60])
    context_prefix = ""
    if supabase:
        try:
            recent = supabase.table("chat_memory").select("message,response").eq("user_id",str(user_id)).order("created_at",desc=True).limit(1).execute()
            if recent.data:
                last_response = recent.data[0].get("response","")
                if "clarify" in last_response.lower() or "when" in last_response.lower():
                    context_prefix = f"Original request: {recent.data[0].get('message','')}\nUser's answer: "
        except Exception: pass
    task_data = await parse_task_with_ai(context_prefix + user_text, user_id)
    if task_data.get("needs_clarification"):
        await send_text_chunks(chat_id, f"🤔 {task_data.get('clarification_question','Could you clarify?')}", reply_to=message_id)
        return True
    result = await create_task_in_db(user_id, task_data)
    await send_text_chunks(chat_id, result, reply_to=message_id)
    return True

# ═══════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════
def get_latest_news(query: str, max_results: int = 5) -> str:
    if not GNEWS_API_KEY: return search_web(query, max_results)
    try:
        resp = requests.get("https://gnews.io/api/v4/search", params={"q":query,"apikey":GNEWS_API_KEY,"lang":"en","country":"ng","max":max_results}, timeout=10)
        if resp.status_code != 200: return search_web(query, max_results)
        articles = resp.json().get("articles",[])
        if not articles: return search_web(query, max_results)
        lines = []
        for i, a in enumerate(articles[:max_results], 1):
            pub = a.get("publishedAt","")[:16].replace("T"," ")
            lines.append(f"{i}. {a.get('title','')}\n   Source: {a.get('source',{}).get('name','')} | {pub}\n   {a.get('description','')}\n   {a.get('url','')}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("GNews error: %s", e)
        return search_web(query, max_results)

def get_sports_data(query: str) -> str:
    if not GNEWS_API_KEY: return search_web(query, 5)
    try:
        q_lower  = query.lower()
        sport_q  = "Nigeria football"
        if "premier league" in q_lower or "epl" in q_lower:      sport_q = "Premier League"
        elif "champions league" in q_lower:                        sport_q = "Champions League"
        elif "afcon" in q_lower or "african cup" in q_lower:      sport_q = "AFCON"
        elif "world cup" in q_lower:                               sport_q = "World Cup"
        elif "f1" in q_lower or "formula" in q_lower:             sport_q = "Formula 1"
        elif "nba" in q_lower or "basketball" in q_lower:         sport_q = "NBA basketball"
        elif "tennis" in q_lower:                                  sport_q = "tennis"
        elif "boxing" in q_lower or "ufc" in q_lower:             sport_q = "boxing MMA UFC"
        elif "cricket" in q_lower:                                 sport_q = "cricket"
        elif "rugby" in q_lower:                                   sport_q = "rugby"
        else:                                                      sport_q = query
        resp = requests.get("https://gnews.io/api/v4/top-headlines", params={"category":"sports","q":sport_q,"apikey":GNEWS_API_KEY,"lang":"en","country":"ng","max":5}, timeout=10)
        if resp.status_code != 200: return search_web(query, 5)
        articles = resp.json().get("articles",[])
        if not articles: return search_web(query, 5)
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
        resp = requests.get(f"https://search.brave.com/search?q={quote(query)}&source=web", headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200: return None
        soup    = BeautifulSoup(resp.text, "html.parser")
        results = []
        snippets = soup.find_all("div", class_="snippet") or soup.find_all("div", class_="result")
        for s in snippets[:max_results]:
            a = s.find("a", href=True)
            if not a: continue
            title = a.get_text(strip=True)
            url   = a["href"]
            if url and not url.startswith("http"): url = "https://search.brave.com" + url
            if "url=" in url:
                m = re.search(r'url=([^&]+)', url)
                if m: url = unquote(m.group(1))
            desc_el = s.find("p") or s.find("div", class_="snippet-description")
            desc    = desc_el.get_text(strip=True)[:300] if desc_el else ""
            if title and len(title) > 3: results.append({"title": title, "description": desc, "url": url})
        return results[:max_results] if results else None
    except Exception as e:
        logger.error("Brave scrape error: %s", e)
        return None

def _search_brave_api(query: str, max_results: int = 5) -> Optional[list]:
    if not BRAVE_API_KEY: return None
    try:
        resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers={"Accept":"application/json","X-Subscription-Token":BRAVE_API_KEY}, params={"q":query,"count":max_results}, timeout=10)
        if resp.status_code == 200:
            items = resp.json().get("web",{}).get("results",[])
            if items: return [{"title":i.get("title",""),"description":i.get("description",""),"url":i.get("url","")} for i in items[:max_results]]
        return None
    except Exception as e:
        logger.error("Brave API error: %s", e)
        return None

def _search_duckduckgo_lite(query: str, max_results: int = 5) -> Optional[list]:
    try:
        resp = requests.post("https://lite.duckduckgo.com/lite/", data={"q":query}, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200: return None
        soup    = BeautifulSoup(resp.text, "html.parser")
        results = []
        for link in soup.find_all("a", class_="result-link")[:max_results]:
            title = link.get_text(strip=True)
            href  = link.get("href","")
            desc  = ""
            row   = link.find_parent("tr")
            if row:
                nr = row.find_next_sibling("tr")
                if nr:
                    td = nr.find("td", class_="result-snippet")
                    if td: desc = td.get_text(strip=True)
            if title and href: results.append({"title":title,"description":desc,"url":href})
        return results if results else None
    except Exception as e:
        logger.error("DuckDuckGo Lite error: %s", e)
        return None

def search_web(query: str, max_results: int = 5) -> str:
    results = _search_brave_scrape(query, max_results) or _search_brave_api(query, max_results) or _search_duckduckgo_lite(query, max_results)
    if not results: return "No search results found."
    return "\n\n".join(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}" for i, r in enumerate(results, 1))

def deep_research(query: str) -> str:
    seen_urls, all_results = set(), []
    for q in [query, f"{query} latest news", f"{query} results details"]:
        rs = _search_brave_scrape(q, 4) or _search_brave_api(q, 4) or _search_duckduckgo_lite(q, 4)
        if rs:
            for r in rs:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
    if not all_results: return "Deep research found no results."
    lines = ["=== DEEP RESEARCH RESULTS ===\n"]
    for i, r in enumerate(all_results[:12], 1):
        lines.append(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}")
    return "\n\n".join(lines)

def fetch_url_content(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]): tag.extract()
        return " ".join(soup.get_text(separator=" ", strip=True).split())[:3000]
    except Exception as e:
        logger.error("URL fetch error: %s", e)
        return "Failed to read link."

def detect_urls(text: str) -> list:
    return re.findall(r'https?://\S+', text)

# ═══════════════════════════════════════════════════════════
# AI & MEMORY
# ═══════════════════════════════════════════════════════════
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
        res = supabase.table("user_profiles").select("session_summary,last_active").eq("user_id",str(user_id)).execute()
        if not res.data: return ""
        p = res.data[0]
        summary        = p.get("session_summary","") or ""
        last_active_str = p.get("last_active","")
        if not last_active_str: return summary
        last_active = datetime.fromisoformat(last_active_str.replace("Z","+00:00"))
        if (datetime.now(timezone.utc) - last_active).total_seconds() > 10800:
            supabase.table("user_profiles").update({"session_summary":""}).eq("user_id",str(user_id)).execute()
            return ""
        return summary
    except Exception as e:
        logger.error("Session summary error: %s", e)
        return ""

async def update_session_summary(user_id: str, recent_messages: list, current_summary: str):
    if not supabase: return
    try:
        msg_text = "\n".join([f"User: {m['message']}\nAIM: {m['response']}" for m in recent_messages])
        prompt   = f"Current Summary: {current_summary or 'None yet'}\nNew Messages:\n{msg_text}\nCreate a concise updated summary. Max 150 words."
        new_summary = None
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(model="deepseek-v4-flash", messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=200)
            new_summary = r.choices[0].message.content.strip() if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(model="gemini-2.5-flash-lite", contents=[types.Content(role="user",parts=[types.Part(text=prompt)])], config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=200))
            new_summary = r.text.strip() if r and r.text else None
        if new_summary:
            supabase.table("user_profiles").update({"session_summary":new_summary}).eq("user_id",str(user_id)).execute()
    except Exception as e:
        logger.error("Summarization error: %s", e)

def _fmt_wat(dt: datetime) -> str:
    return dt.astimezone(WAT).strftime("%a %b %d, %Y · %I:%M %p WAT")

async def get_conversation_context(user_id: str, query_text: str) -> tuple[str, str, float]:
    if not supabase: return "", "", 0.0
    try:
        rows = supabase.table("chat_memory").select("message,response,topic,created_at").eq("user_id",str(user_id)).order("created_at",desc=True).limit(40).execute()
        if not rows.data: return "", "", 0.0
        gap_seconds = 0.0
        try:
            last_ts     = datetime.fromisoformat(rows.data[0]["created_at"].replace("Z","+00:00"))
            gap_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
        except Exception: pass
        recent_rows  = list(reversed(rows.data[:5]))
        recent_lines = []
        for row in recent_rows:
            try:    tstr = _fmt_wat(datetime.fromisoformat(row["created_at"].replace("Z","+00:00")))
            except: tstr = "unknown time"
            recent_lines += [f"  [{tstr}]", f"  User : {row['message']}", f"  AIM  : {row['response']}", ""]
        recent_history = "\n".join(recent_lines).strip()
        older_rows     = rows.data[5:]
        if not older_rows: return recent_history, "", gap_seconds
        query_lower    = query_text.lower()
        keyword_topics = {"space":["tech"],"nigeria":["general","politics"],"money":["finance"],"job":["career"],"health":["health"],"love":["relationships"],"sport":["sports"],"music":["entertainment"],"school":["education"],"code":["tech"],"ai":["tech"]}
        matched_topics = set()
        for kw, tps in keyword_topics.items():
            if kw in query_lower: matched_topics.update(tps)
        scored = []
        for row in older_rows:
            score = 0
            try: score += max(0, 30 - (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"].replace("Z","+00:00"))).days)
            except: pass
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

async def get_ai_response(
    user_text: str, user_id: str, chat_type: str, profile: dict = None,
    session_summary: str = "", recent_history: str = "", older_context: str = "",
    web_context: str = "", tool_status: str = "", gap_seconds: float = 0.0
) -> Optional[str]:
    try:
        if profile is None: profile = await get_user_profile_data(user_id)
        # build_enhanced_prompt is in core.py; it does NOT take is_admin_func —
        # admin detection is injected via the prompt text built inside core.py
        # using the is_admin import we pass via the is_admin parameter.
        prompt = build_enhanced_prompt(
            user_text, user_id, profile, is_admin,
            session_summary, recent_history, older_context,
            web_context, tool_status, gap_seconds
        )
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(model="deepseek-v4-flash", messages=[{"role":"system","content":BASE_SYSTEM_PROMPT},{"role":"user","content":prompt}], temperature=0.7, max_tokens=1024)
            return r.choices[0].message.content if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(model="gemini-2.5-flash-lite", contents=[types.Content(role="user",parts=[types.Part(text=prompt)])], config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024))
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
            r = await deepseek_client.chat.completions.create(model="deepseek-v4-flash", messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=20)
            t = r.choices[0].message.content.strip().lower() if r.choices else "general"
        elif gemini_client:
            r = gemini_client.models.generate_content(model="gemini-2.5-flash-lite", contents=[types.Content(role="user",parts=[types.Part(text=prompt)])], config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20))
            t = r.text.strip().lower() if r and r.text else "general"
        else: return "general"
        return t if t in topics else "general"
    except Exception: return "general"

async def save_chat_memory(user_id: str, username: str, message: str, response: str, chat_type: str, topic: str = "general"):
    if not supabase: return
    try: supabase.table("chat_memory").insert({"user_id":str(user_id),"username":username or "","message":message[:2000],"response":response[:2000],"chat_type":chat_type,"topic":topic}).execute()
    except Exception as e: logger.error("Memory save failed: %s", e)

async def update_user_profile(user_id: str, username: str, topic: str):
    if not supabase: return
    try:
        ex = supabase.table("user_profiles").select("*").eq("user_id",str(user_id)).execute()
        if ex.data:
            p  = ex.data[0]
            tc = p.get("topic_counts",{})
            tc[topic] = tc.get(topic,0) + 1
            supabase.table("user_profiles").update({"topic_counts":tc,"total_chats":p.get("total_chats",0)+1,"last_active":datetime.now(timezone.utc).isoformat()}).eq("user_id",str(user_id)).execute()
        else:
            supabase.table("user_profiles").insert({"user_id":str(user_id),"username":username or "","topic_counts":{topic:1},"total_chats":1,"last_active":datetime.now(timezone.utc).isoformat()}).execute()
    except Exception as e: logger.error("Profile update failed: %s", e)

def is_memory_search_query(user_text: str) -> bool:
    keywords = ["what did we talk about","what have we discussed","remember our chats","our conversations","what did i ask you","what were we talking about","we discussed","what did we say about","remember when","do you remember"]
    return any(kw in user_text.lower() for kw in keywords)

def extract_search_keywords(user_text: str) -> list:
    clean = user_text.lower()
    for phrase in ["what did we talk about","what were we talking about","tell me about","what did we say about","do you remember","remember when","what about","didn't we talk about","what have we discussed"]:
        clean = clean.replace(phrase, "")
    clean = re.sub(r'[^\w\s]', ' ', clean)
    stop  = {"the","and","about","were","did","have","what","when","that","this","with","for","from","you","are","was","is","it","we","our","me","my","i","a","an","to","of","in","on","at","be","been","do","does","say","get","go","know","think","take","see","want","use","find","give","tell","ask","work"}
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
        lines = [f"📊 Top Topics: {', '.join(f'{k}({v}x)' for k,v in sorted(tc.items(),key=lambda x:x[1],reverse=True)[:3])}", f"💬 Total Chats: {p.get('total_chats',0)}", "", "📝 Recent conversations:"]
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
# SEND MESSAGE
# ═══════════════════════════════════════════════════════════
async def send_text_chunks(chat_id: int, text: str, reply_to: Optional[int] = None, message_id: Optional[int] = None):
    if not bot: return
    try:
        if message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS], parse_mode=ParseMode.HTML)
        else:
            kw = {"chat_id":chat_id, "text":text[:TELEGRAM_MAX_CHARS], "parse_mode":ParseMode.HTML}
            if reply_to: kw["reply_to_message_id"] = reply_to
            await bot.send_message(**kw)
    except Exception as e:
        logger.error("Send error: %s", e)
        try:
            if message_id: await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS])
            else:           await bot.send_message(chat_id=chat_id, text=text[:TELEGRAM_MAX_CHARS])
        except Exception as e2: logger.error("Fallback send failed: %s", e2)

# ═══════════════════════════════════════════════════════════
# BOT COMMANDS
# ═══════════════════════════════════════════════════════════
async def handle_bot_command(user_id: str, chat_id: int, message_id: int, user_text: str) -> bool:
    tl = user_text.lower().strip()

    if tl.startswith("/start") or tl.startswith("/help"):
        welcome_msg = """🌟 <b>Welcome to AIM — African Intelligence Model!</b>

I'm your personal AI assistant built for Africans, by Africans. Here's what I can do:

<b>🧠 Smart Conversations</b>
• Chat in English, Pidgin, Yoruba, Hausa, Igbo & more
• I remember our conversations and learn your preferences

<b>🔍 Real-Time Search</b>
• Web search, news, sports updates
• Deep research on any topic

<b>⏰ Tasks & Reminders</b>
• "Remind me at 6pm to cook"
• "Every Monday send me news at 8am"
• "Daily bible verse" or "Word of the day"

<b>🎨 Creative Tools</b>
• Generate images, audio, PDFs
• Analyze photos and documents

<b>🎙️ Voice Support</b>
• Send voice notes — I'll transcribe and respond

<b>Quick Commands:</b>
/help — See all commands
/tasks — View your reminders
/search [query] — Quick search
/deep [query] — Deep research
/claim — Get your Empire ID
/link — Connect to Empire AI Web App

Just talk to me naturally — I'll understand! 🇳🇬✨"""
        await send_text_chunks(chat_id, welcome_msg, reply_to=message_id)
        return True

    elif tl.startswith("/link") or tl.startswith("/account"):
        await _handle_link_command(user_id, chat_id, message_id)
        return True

    elif tl.startswith("/search "):
        query = user_text[8:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔍 Searching...", reply_to=message_id)
        if "news" in query.lower() or "latest" in query.lower(): results = get_latest_news(query)
        elif any(s in query.lower() for s in ["football","match","score","f1","nba","tennis","boxing"]): results = get_sports_data(query)
        else: results = search_web(query)
        if results == "No search results found.":
            await send_text_chunks(chat_id, "Couldn't find results.", reply_to=message_id)
            return True
        try:
            txt = await get_ai_response(f"User asked: {query}\n\nSearch Results:\n{results}\n\nAnswer using ONLY these results.", user_id, "private")
            if not txt or "SEARCH_TRIGGER:" in txt: txt = results
            await send_text_chunks(chat_id, txt.strip(), reply_to=message_id)
        except Exception:
            await send_text_chunks(chat_id, results, reply_to=message_id)
        return True

    elif tl.startswith("/deep "):
        query = user_text[6:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔬 Researching...", reply_to=message_id)
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
            supabase.table("user_tools").insert({"user_id":str(user_id),"tool_type":"timer","start_time":datetime.now(timezone.utc).isoformat(),"duration_seconds":dur,"target_time":target.isoformat(),"is_active":True}).execute()
            await send_text_chunks(chat_id, f"⏲️ Timer set for {ts}!", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "❌ Use: /timer 5m", reply_to=message_id)
        return True

    elif tl == "/stopwatch":
        res = supabase.table("user_tools").select("*").eq("user_id",str(user_id)).eq("tool_type","stopwatch").eq("is_active",True).order("created_at",desc=True).limit(1).execute()
        if res.data:
            row     = res.data[0]
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row["start_time"].replace("Z","+00:00"))
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            supabase.table("user_tools").update({"is_active":False}).eq("id",row["id"]).execute()
            await send_text_chunks(chat_id, f"⏱️ Stopped! {mins}m {secs}s" if mins else f"⏱️ Stopped! {secs}s", reply_to=message_id)
        else:
            supabase.table("user_tools").insert({"user_id":str(user_id),"tool_type":"stopwatch","start_time":datetime.now(timezone.utc).isoformat(),"is_active":True}).execute()
            await send_text_chunks(chat_id, "⏱️ Stopwatch started!", reply_to=message_id)
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
        res        = supabase.table("user_tasks").select("id").eq("user_id",str(user_id)).eq("is_active",True).execute()
        full_id    = next((t["id"] for t in res.data if t["id"].startswith(partial_id)), None)
        if not full_id:
            await send_text_chunks(chat_id, "❌ Task not found.", reply_to=message_id)
            return True
        supabase.table("user_tasks").update({"is_active":False}).eq("id",full_id).eq("user_id",str(user_id)).execute()
        await send_text_chunks(chat_id, "✅ Task deleted!", reply_to=message_id)
        return True

    elif tl.startswith("/news "):
        parts = user_text.split()
        if len(parts) < 3:
            await send_text_chunks(chat_id, "❌ Use: /news daily 8am", reply_to=message_id)
            return True
        pattern   = parts[1].lower()
        time_part = parts[2].lower().replace("am","").replace("pm","")
        try:
            hour = int(time_part)
            if "pm" in parts[2].lower() and hour != 12: hour += 12
            rec_time = f"{hour:02d}:00"
        except ValueError: rec_time = "08:00"
        task_data = {"description":"Send me the latest news","type":"recurring","recurrence_pattern":pattern,"recurrence_time":rec_time,"recurrence_days":[],"category":"news","needs_clarification":False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/verse daily":
        task_data = {"description":"Daily bible verse","type":"recurring","recurrence_pattern":"daily","recurrence_time":"08:00","category":"verse","needs_clarification":False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/word daily":
        task_data = {"description":"Word of the day","type":"recurring","recurrence_pattern":"daily","recurrence_time":"09:00","category":"word","needs_clarification":False}
        await send_text_chunks(chat_id, await create_task_in_db(user_id, task_data), reply_to=message_id)
        return True

    elif tl == "/claim":
        profile     = await get_user_profile_data(user_id)
        existing_id = profile.get("empire_id")
        if existing_id:
            await send_text_chunks(chat_id, f"👑 Your Empire ID: <b>{existing_id}</b>\n\nSave it! You'll use it to log into the Web App.", reply_to=message_id)
        else:
            new_id = generate_empire_id()
            try:
                supabase.table("user_profiles").update({"empire_id":new_id}).eq("user_id",str(user_id)).execute()
                await send_text_chunks(chat_id, f"🎉 Empire ID created: <b>{new_id}</b>\n\n⚠️ <b>Save this.</b> You'll use it to sync Telegram memories to the web app.\n\n💡 Type /link to connect your account now!", reply_to=message_id)
            except Exception as e:
                logger.error("Failed to save Empire ID: %s", e)
                await send_text_chunks(chat_id, "❌ Failed to generate Empire ID. Please try again.", reply_to=message_id)
        return True

    # ── ADMIN COMMANDS — delegated entirely to admin.py ──────
    elif tl.startswith("/admin"):
        return await handle_admin_command(
            user_id, chat_id, message_id, user_text,
            supabase, get_ai_response, send_text_chunks, USE_DEEPSEEK
        )

    return False

# ═══════════════════════════════════════════════════════════
# INLINE QUERY
# ═══════════════════════════════════════════════════════════
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
        r_text  = await asyncio.wait_for(get_ai_response(qtext, uid, "private", profile, web_context=web_ctx), timeout=15.0)
        if r_text: answer_text = r_text.strip()[:300]
    except asyncio.TimeoutError: pass
    except Exception as e: logger.error("Inline AI error: %s", e)
    result = InlineQueryResultArticle(
        id=str(uuid.uuid4()), title=f"AIM: {qtext[:30]}",
        description=(answer_text or "Click to get AIM's answer")[:100],
        input_message_content=InputTextMessageContent(
            message_text=f"🤖 <b>AIM says:</b>\n\n{answer_text}\n\n<i>via @askaimbot</i>" if answer_text else f"🤖 Asking AIM: {qtext}\n⏳ Processing...",
            parse_mode=ParseMode.HTML
        )
    )
    try: await bot.answer_inline_query(inline_query_id=qid, results=[result], cache_time=0, is_personal=True)
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
            await send_text_chunks(chat_id, "🔥 High demand — try again.", reply_to=message_id)
    except Exception as e:
        logger.error("Inline answer error: %s", e)
        await send_text_chunks(chat_id, "🛠️ Something went wrong.", reply_to=message_id)

def is_inline_placeholder(text: str) -> Tuple[bool, str]:
    if not text: return False, ""
    tl = text.strip().lower()
    if "asking aim" not in tl: return False, ""
    if "processing" not in tl and "thinking" not in tl: return False, ""
    parts = text.strip().split(":", 1)
    if len(parts) < 2: return False, ""
    q = parts[1].strip().split("\n")[0].replace("⏳","").replace("Processing...","").replace("Thinking...","").strip()
    if q: return True, q
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

    # Voice
    if update.message.voice or update.message.audio:
        file_obj = update.message.voice or update.message.audio
        await send_text_chunks(chat.id, "🎙️ Listening...", reply_to=message_id)
        transcribed = await transcribe_voice(file_obj.file_id)
        if transcribed:
            user_text = transcribed
            await send_text_chunks(chat.id, f"📝 You said: \"{user_text}\"", reply_to=message_id)
        else:
            await send_text_chunks(chat.id, "🎤 Sorry, couldn't understand.", reply_to=message_id)
            return

    # Photo
    if update.message.photo:
        photo     = update.message.photo[-1]
        temp_path = f"photo_{photo.file_id}.jpg"
        await send_text_chunks(chat.id, "👁️ Nebulae is looking...", reply_to=message_id)
        try:
            file = await bot.get_file(photo.file_id)
            await file.download_to_drive(custom_path=temp_path)
            with open(temp_path, "rb") as img_file:
                img_bytes = img_file.read()
            is_logo = await nebulae.is_aim_logo(img_bytes)
            if is_logo:
                description = "This is AIM's logo — a beam of light representing African intelligence!"
                await send_text_chunks(chat.id, "✨ I recognize myself! This is my logo! 🌟🇳🇬", reply_to=message_id)
            else:
                description = await nebulae.analyze_image(img_bytes, "Describe this image in detail.")
                await send_text_chunks(chat.id, f"📝 Nebulae sees: {description[:300]}", reply_to=message_id)
            user_text = f"[User sent a photo. {description}]"
        except Exception as e:
            logger.error("Vision error: %s", e)
            await send_text_chunks(chat.id, "👁️ Couldn't process the photo.", reply_to=message_id)
            return
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

    if not user_text:
        await send_text_chunks(chat.id, "I can only read text, voice, and photo messages.")
        return

    user_id  = str(user.id)
    username = user.username or user.first_name or "User"
    logger.info("📩 [%s/%s] '%s'", user_id, chat_type, user_text[:80])

    if user_text.startswith("/"):
        if await handle_bot_command(user_id, chat.id, message_id, user_text): return

    profile = await get_user_profile_data(user_id)

    is_ph, ph_query = is_inline_placeholder(user_text)
    if is_ph and ph_query:
        await process_inline_answer(chat.id, message_id, ph_query, user_id)
        return

    if is_memory_search_query(user_text):
        kws    = extract_search_keywords(user_text)
        result = await search_memory_by_keyword(user_id, user_text) if kws else await search_memory(user_id)
        await send_text_chunks(chat.id, result, reply_to=message_id)
        return

    if await handle_task_message(user_text, user_id, chat.id, message_id):
        try:
            last_response = supabase.table("chat_memory").select("response").eq("user_id",str(user_id)).order("created_at",desc=True).limit(1).execute()
            ai_response   = last_response.data[0]["response"] if last_response.data else "Task created"
            await save_chat_memory(user_id, username, user_text, ai_response, chat_type, "reminder")
            await update_user_profile(user_id, username, "reminder")
        except Exception as e:
            logger.error("Failed to save task interaction: %s", e)
        return

    if chat_type in ("group", "supergroup"):
        mentioned      = "@askaimbot" in user_text.lower()
        replied_to_bot = (update.message.reply_to_message and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.is_bot and update.message.reply_to_message.from_user.username == "askaimbot")
        if not mentioned and not replied_to_bot: return
        user_text = re.sub(r'@askaimbot', '', user_text, flags=re.IGNORECASE).strip()

    try:
        session_summary                           = await get_session_summary(user_id)
        recent_history, older_context, gap_seconds = await get_conversation_context(user_id, user_text)
        web_context = ""
        for url in detect_urls(user_text):
            c = fetch_url_content(url)
            if c and "Failed" not in c: web_context += f"Content from {url}:\n{c}\n"
        if is_search_query(user_text) and not web_context:
            if "news" in user_text.lower() or "latest" in user_text.lower() or "today" in user_text.lower(): sr = get_latest_news(user_text)
            elif any(s in user_text.lower() for s in ["football","match","score","team","player","league","f1","nba","tennis","boxing","ufc","cricket","rugby"]): sr = get_sports_data(user_text)
            else: sr = search_web(user_text)
            if "No search results" not in sr: web_context = f"Web Search Results for '{user_text}':\n{sr}"

        max_iter, iteration, final_answer, tool_status = 3, 0, None, ""
        while iteration < max_iter:
            iteration += 1
            answer = await get_ai_response(user_text, user_id, chat_type, profile, session_summary, recent_history, older_context, web_context, tool_status, gap_seconds)
            if not answer:
                final_answer = "🔥 High demand right now — please try again."
                break
            answer = answer.strip()

            if "SEARCH_TRIGGER:" in answer:
                m = re.search(r'SEARCH_TRIGGER:\s*(.+)', answer, re.IGNORECASE)
                if m:
                    sq = m.group(1).strip()
                    if "news" in sq.lower() or "latest" in sq.lower(): sr = get_latest_news(sq)
                    elif any(s in sq.lower() for s in ["football","match","score","f1","nba","tennis","boxing"]): sr = get_sports_data(sq)
                    else: sr = search_web(sq)
                    if sr == "No search results found.": web_context += f"\n\nSearch for '{sq}': No results."
                    else: web_context += f"\n\nWeb Search Results for '{sq}':\n{sr}"
                    continue
                else: final_answer = answer; break

            # Timer
            tm = re.search(r'\[TIMER:(\d+)(s|m|h)\]', answer, re.IGNORECASE)
            if tm:
                amt, unit = int(tm.group(1)), tm.group(2).lower()
                dur    = amt * (1 if unit=="s" else 60 if unit=="m" else 3600)
                target = datetime.now(timezone.utc) + timedelta(seconds=dur)
                supabase.table("user_tools").insert({"user_id":user_id,"tool_type":"timer","start_time":datetime.now(timezone.utc).isoformat(),"duration_seconds":dur,"target_time":target.isoformat(),"is_active":True}).execute()
                answer      = re.sub(r'\[TIMER:\d+[smh]\]','',answer,flags=re.IGNORECASE).strip()
                tool_status = f"✅ Timer set for {amt}{unit}"
                answer     += f"\n\n_{tool_status}_"

            # Stopwatch
            sm = re.search(r'\[STOPWATCH:(START|STOP)\]', answer, re.IGNORECASE)
            if sm:
                action = sm.group(1).upper()
                if action == "START":
                    supabase.table("user_tools").insert({"user_id":user_id,"tool_type":"stopwatch","start_time":datetime.now(timezone.utc).isoformat(),"is_active":True}).execute()
                    answer = re.sub(r'\[STOPWATCH:START\]','',answer,flags=re.IGNORECASE).strip()
                    answer += "\n\n_⏱️ Stopwatch started!_"
                elif action == "STOP":
                    res = supabase.table("user_tools").select("*").eq("user_id",user_id).eq("tool_type","stopwatch").eq("is_active",True).order("created_at",desc=True).limit(1).execute()
                    if res.data:
                        row     = res.data[0]
                        elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row["start_time"].replace("Z","+00:00"))
                        mins, secs = divmod(int(elapsed.total_seconds()), 60)
                        supabase.table("user_tools").update({"is_active":False}).eq("id",row["id"]).execute()
                        ts     = f"{mins}m {secs}s" if mins else f"{secs}s"
                        answer = re.sub(r'\[STOPWATCH:STOP\]','',answer,flags=re.IGNORECASE).strip()
                        answer += f"\n\n_⏱️ Stopped! Time: {ts}_"

            # Nebulae image
            img_match = re.search(r'\[NEBULAE_IMAGE:\s*(.+?)\]', answer, re.IGNORECASE)
            if img_match:
                img_prompt = img_match.group(1).strip()
                await send_text_chunks(chat.id, "🎨 Nebulae is painting...", reply_to=message_id)
                img_bytes  = await nebulae.generate_image(img_prompt)
                if img_bytes:
                    try:
                        await bot.send_photo(chat_id=chat.id, photo=img_bytes, caption="✨ Generated by Nebulae", reply_to_message_id=message_id)
                        answer = "✅ Here is your image!"
                    except Exception: answer = "❌ Image failed to send."
                else: answer = "❌ Nebulae couldn't generate the image."
                answer = re.sub(r'\[NEBULAE_IMAGE:.*?\]','',answer,flags=re.IGNORECASE).strip()

            # Nebulae audio
            audio_match = re.search(r'\[NEBULAE_AUDIO:\s*(.+?)\]', answer, re.IGNORECASE)
            if audio_match:
                audio_text  = audio_match.group(1).strip()
                await send_text_chunks(chat.id, "🔊 Nebulae is speaking...", reply_to=message_id)
                audio_bytes = await nebulae.generate_audio(audio_text)
                if audio_bytes:
                    try:
                        await bot.send_audio(chat_id=chat.id, audio=audio_bytes, caption="🔊 Audio by Nebulae", reply_to_message_id=message_id)
                        answer = "✅ Here is your audio!"
                    except Exception: answer = "❌ Audio failed to send."
                else: answer = "❌ Nebulae couldn't generate the audio."
                answer = re.sub(r'\[NEBULAE_AUDIO:.*?\]','',answer,flags=re.IGNORECASE).strip()

            # Nebulae PDF
            pdf_match = re.search(r'\[NEBULAE_PDF:\s*([^\|]+)\|(.*?)\]', answer, re.IGNORECASE | re.DOTALL)
            if pdf_match:
                pdf_title   = pdf_match.group(1).strip()
                pdf_content = pdf_match.group(2).strip()
                await send_text_chunks(chat.id, "📄 Nebulae is generating your document...", reply_to=message_id)
                pdf_bytes = nebulae.generate_pdf(pdf_title, pdf_content)
                if pdf_bytes:
                    try:
                        await bot.send_document(chat_id=chat.id, document=pdf_bytes, filename=f"{pdf_title.replace(' ','_')}.pdf", caption=f"📄 {pdf_title}", reply_to_message_id=message_id)
                        answer = "✅ Here is your PDF!"
                    except Exception: answer = "❌ PDF failed to send."
                else: answer = "❌ Nebulae couldn't generate the PDF."
                answer = re.sub(r'\[NEBULAE_PDF:.*?\]','',answer,flags=re.IGNORECASE|re.DOTALL).strip()

            final_answer = answer
            break

        if final_answer is None: final_answer = "I tried searching but couldn't find results."
        await send_text_chunks(chat.id, final_answer, reply_to=message_id)
        topic = await extract_topic(user_text, final_answer)
        await save_chat_memory(user_id, username, user_text, final_answer, chat_type, topic)
        await update_user_profile(user_id, username, topic)
        if profile.get("total_chats",0) % 4 == 0:
            recent_msgs = supabase.table("chat_memory").select("message,response").eq("user_id",str(user_id)).order("created_at",desc=True).limit(4).execute()
            if recent_msgs.data:
                recent_msgs.data.reverse()
                run_async(update_session_summary(user_id, recent_msgs.data, session_summary))
    except Exception as e:
        logger.error("Critical error: %s", e)
        await send_text_chunks(chat.id, "🛠️ Something went wrong.", reply_to=message_id)

# ═══════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"AIM Bot is live!","version":"v9.7","ai":"DeepSeek V4" if USE_DEEPSEEK else "Gemini"})

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        uid  = data.get("update_id")
        if uid and is_duplicate_update(uid): return "OK", 200
        upd = Update.de_json(data, bot)
        if upd.inline_query: run_async(handle_inline_query_async(upd.inline_query))
        elif upd.message:    run_async(handle_message_async(upd))
        return "OK", 200
    except Exception as e:
        logger.error("Webhook error: %s", e)
        return "Error", 500

@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    """Logto OAuth callback — exchanges code for tokens, links accounts."""
    code  = request.args.get("code","")
    state = request.args.get("state","")
    error = request.args.get("error","")

    if error:
        logger.warning("Logto error: %s", error)
        return _html_page("❌ Login Failed", f"<p>Error: {error}</p>", success=False), 400

    ctx = _consume_oauth_state(state)
    if not ctx:
        return _html_page("❌ Invalid Session", "<p>This link has expired or is invalid. Please use /link again.</p>", success=False), 400

    telegram_user_id = ctx["telegram_user_id"]
    chat_id          = ctx["chat_id"]

    claims = exchange_logto_code(code)
    if not claims:
        run_async(bot.send_message(chat_id=chat_id, text="❌ Something went wrong during login. Please try /link again."))
        return _html_page("❌ Token Error", "<p>Could not verify your login. Please try again.</p>", success=False), 500

    logto_sub   = claims.get("sub","")
    logto_email = claims.get("email","")
    logto_name  = claims.get("name","") or claims.get("username","")

    if not logto_sub:
        return _html_page("❌ No User ID", "<p>Logto did not return a user ID.</p>", success=False), 500

    try:
        existing = supabase.table("user_profiles").select("*").eq("user_id", telegram_user_id).execute()
        if existing.data:
            profile   = existing.data[0]
            empire_id = profile.get("empire_id") or generate_empire_id()
            supabase.table("user_profiles").update({"logto_id":logto_sub,"logto_email":logto_email,"logto_name":logto_name,"empire_id":empire_id,"last_active":datetime.now(timezone.utc).isoformat()}).eq("user_id", telegram_user_id).execute()
        else:
            empire_id = generate_empire_id()
            supabase.table("user_profiles").insert({"user_id":telegram_user_id,"username":logto_name or "","logto_id":logto_sub,"logto_email":logto_email,"logto_name":logto_name,"empire_id":empire_id,"topic_counts":{},"total_chats":0,"last_active":datetime.now(timezone.utc).isoformat()}).execute()
        logger.info("✅ Linked Telegram %s → Logto %s (Empire ID: %s)", telegram_user_id, logto_sub, empire_id)
    except Exception as e:
        logger.error("Supabase link error: %s", e)
        run_async(bot.send_message(chat_id=chat_id, text="⚠️ Login verified but we couldn't save your link. Please try /link again."))
        return _html_page("❌ Database Error", "<p>We couldn't save your account link. Please try again.</p>", success=False), 500

    display = logto_name or logto_email or "there"
    run_async(bot.send_message(
        chat_id=chat_id,
        text=(f"🎉 <b>Account linked, {display}!</b>\n\n🆔 Empire ID: <b>{empire_id}</b>\n📧 Email: {logto_email or 'N/A'}\n\n✅ Your Telegram memory is now connected to the Empire AI web app. 🌍🇳🇬"),
        parse_mode=ParseMode.HTML,
    ))
    return _html_page("✅ Account Linked!", f"<p>Welcome, <b>{display}</b>! 🎉</p><p>Your Empire ID: <b>{empire_id}</b></p><p>You can close this tab and return to Telegram.</p>", success=True), 200

def _html_page(title: str, body: str, success: bool = True) -> str:
    color = "#22c55e" if success else "#ef4444"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Empire AI — {title}</title><style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#f0f0f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}}.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:16px;padding:40px 32px;max-width:420px;width:100%;text-align:center}}.icon{{font-size:48px;margin-bottom:16px}}h1{{font-size:22px;font-weight:700;color:{color};margin-bottom:16px}}p{{font-size:15px;color:#aaa;line-height:1.6;margin-bottom:10px}}b{{color:#f0f0f0}}.brand{{margin-top:32px;font-size:13px;color:#555}}</style></head><body><div class="card"><div class="icon">{'🌍' if success else '⚠️'}</div><h1>{title}</h1>{body}<div class="brand">Empire AI · African Intelligence Model</div></div></body></html>"""

@app.route("/debug/tasks/<user_id>", methods=["GET"])
def debug_tasks(user_id: str):
    if not supabase: return jsonify({"error":"Supabase not connected"}), 500
    try:
        now_utc = datetime.now(timezone.utc)
        tasks   = supabase.table("user_tasks").select("*").eq("user_id",user_id).order("created_at",desc=True).limit(20).execute()
        enriched = []
        for t in tasks.data:
            nr  = t.get("next_run")
            sfd = None
            sec = None
            if nr:
                try:
                    nrdt = datetime.fromisoformat(nr.replace("Z","+00:00"))
                    sfd  = nrdt <= now_utc
                    sec  = (nrdt - now_utc).total_seconds()
                except Exception: pass
            enriched.append({**t, "should_have_fired": sfd, "seconds_until_fire": sec})
        return jsonify({"user_id":user_id,"server_time_utc":now_utc.isoformat(),"tasks":enriched})
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/debug/worker-status", methods=["GET"])
def debug_worker_status():
    alive = [t.name for t in threading.enumerate()]
    return jsonify({
        "all_threads":        alive,
        "timer_worker_alive": "timer-worker" in alive,
        "task_worker_alive":  "task-worker" in alive,
        "async_loop_alive":   "async-loop" in alive,
        "server_time_utc":    datetime.now(timezone.utc).isoformat(),
    })

@app.route("/set-webhook", methods=["GET"])
def set_webhook():
    if not bot or not WEBHOOK_URL: return jsonify({"error":"Not configured"}), 500
    try:
        bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")
        return jsonify({"status":"Webhook set!"})
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    if not bot: return jsonify({"error":"Bot not configured"}), 500
    try:
        bot.delete_webhook()
        return jsonify({"status":"Webhook deleted!"})
    except Exception as e: return jsonify({"error":str(e)}), 500


@app.route("/debug/logto", methods=["GET"])
def debug_logto():
    """Check that Logto env vars are actually visible at runtime."""
    return jsonify({
        "LOGTO_ENDPOINT":     bool(os.environ.get("LOGTO_ENDPOINT")),
        "LOGTO_CLIENT_ID":    bool(os.environ.get("LOGTO_CLIENT_ID")),
        "LOGTO_CLIENT_SECRET":bool(os.environ.get("LOGTO_CLIENT_SECRET")),
        "WEBHOOK_URL":        bool(os.environ.get("WEBHOOK_URL")),
        "redirect_uri":       get_redirect_uri(),
        "auth_url_sample":    build_logto_auth_url("test_state_123") if os.environ.get("LOGTO_ENDPOINT") else "NOT CONFIGURED",
    })
@app.route("/privacy", methods=["GET"])
def privacy_policy():
    return "<h1>Privacy Policy</h1><p>Coming soon.</p>", 200, {"Content-Type":"text/html"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)