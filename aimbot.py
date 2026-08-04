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
    Update, Bot, InlineQueryResultArticle, InputTextMessageContent,
    InlineKeyboardMarkup, InlineKeyboardButton
)
from telegram.constants import ParseMode
from supabase import create_client, Client

# NOTE: empire_id_generator.py is no longer used — Empire ID lookups/creation
# now go through the standalone Empire ID service over HTTP (see eid_create /
# eid_get_by_logto below), so every app (AIM, ai.empireunion.xyz,
# empireunion.xyz) shares one implementation instead of each having its own.
EMPIRE_ID_SERVICE_URL = os.environ.get("EMPIRE_ID_SERVICE_URL", "")
EMPIRE_ID_SERVICE_API_KEY = os.environ.get("EMPIRE_ID_SERVICE_API_KEY", "")

def _eid_headers():
    return {"X-API-Key": EMPIRE_ID_SERVICE_API_KEY, "Content-Type": "application/json"}

def eid_get_by_logto(logto_id: str) -> Optional[dict]:
    """Looks up a user's Empire ID record by their Logto id via the
    Empire ID service. Returns the record dict, or None if not found
    or the service is unreachable."""
    if not EMPIRE_ID_SERVICE_URL:
        logger.error("EMPIRE_ID_SERVICE_URL not configured")
        return None
    try:
        r = requests.get(
            f"{EMPIRE_ID_SERVICE_URL}/v1/empire-id/by-logto/{logto_id}",
            headers=_eid_headers(), timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("record")
        return None  # 404 = not found, anything else also treated as "not found" here
    except Exception as e:
        logger.error("eid_get_by_logto error: %s", e)
        return None

def eid_create(logto_id: str, username: str, email: str, source: str = "telegram_bot"):
    """Get-or-create via the Empire ID service. Returns (ok: bool, result: str)
    to match the old empire_id_generator.py signature — result is the empire_id
    on success, or an error message on failure."""
    if not EMPIRE_ID_SERVICE_URL:
        return False, "❌ Empire ID service not configured"
    try:
        r = requests.post(
            f"{EMPIRE_ID_SERVICE_URL}/v1/empire-id",
            headers=_eid_headers(),
            json={"logto_id": logto_id, "username": username, "email": email, "source": source},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            return True, data.get("empire_id")
        return False, f"❌ Empire ID service error ({r.status_code}): {r.text}"
    except Exception as e:
        logger.error("eid_create error: %s", e)
        return False, f"❌ Empire ID service unreachable: {e}"
from google import genai
import stripe
from google.genai import types

# ── Module imports (deduplicated) ──────────────────────────
from core import BASE_SYSTEM_PROMPT, build_enhanced_prompt, WAT
from modes import apply_mode, get_generation_overrides, disables_web_search, requires_deepsearch, forces_search
from capabilities import is_search_query, SEARCH_TRIGGER_PHRASES, trigger_embeddings, semantic_model
# Import START_TIME from admin so both files share the SAME start reference.
# This is what fixes the stale uptime bug — admin.py sets it at import time
# on fresh deploy, and both files read from it.
from admin import load_admins, is_admin, handle_admin_command, ADMIN_IDS, START_TIME
from chess_engine import register_chess_routes
from language_engine import register_language_routes
from last_activity import register_last_activity_route

# ─── LOGGING ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aimbot")

# ─── CONFIG ───
TELEGRAM_TOKEN      = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY    = os.environ.get("DEEPSEEK_API_KEY", "")
USE_DEEPSEEK        = os.environ.get("USE_DEEPSEEK", "false").lower() == "true"
SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_SECRET_KEY      = os.environ.get("SUPABASE_SECRET_KEY", "") or os.environ.get("SUPABASE_KEY", "")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
WEBHOOK_URL         = os.environ.get("WEBHOOK_URL", "")
BRAVE_API_KEY       = os.environ.get("BRAVE_API_KEY", "")
GNEWS_API_KEY       = os.environ.get("GNEWS_API_KEY", "")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
SPORTAPI_KEY        = os.environ.get("SPORTAPI_KEY", "")

STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_BASIC    = os.environ.get("STRIPE_PRICE_BASIC", "")
STRIPE_PRICE_PRO      = os.environ.get("STRIPE_PRICE_PRO", "")

LOGTO_ENDPOINT      = os.environ.get("LOGTO_ENDPOINT", "").rstrip("/")
LOGTO_CLIENT_ID     = os.environ.get("LOGTO_CLIENT_ID", "")
LOGTO_CLIENT_SECRET = os.environ.get("LOGTO_CLIENT_SECRET", "")

def get_redirect_uri() -> str:
    base = os.environ.get("WEBHOOK_URL", "").rstrip("/")
    return f"{base}/auth/callback" if base else ""

TELEGRAM_MAX_CHARS = 4096

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

_oauth_states: dict[str, dict] = {}
_oauth_states_lock = threading.Lock()

# ─── MESSAGE COALESCING ("blend a fast follow-up into one answer") ───
# If a user sends a second message within COALESCE_WINDOW_SECONDS of the
# first, while AIM is still generating a reply to the first one, we cancel
# the in-flight generation, merge both messages into one combined prompt,
# and generate a single answer covering both — instead of answering the
# first question, then separately answering the second. If no follow-up
# arrives in time, nothing changes: the first message is answered normally
# at full speed (no artificial delay is ever added up front).
_pending_replies: dict[str, dict] = {}
_pending_replies_lock = threading.Lock()
COALESCE_WINDOW_SECONDS = 2.5

def _create_oauth_state(telegram_user_id: str, chat_id: int) -> str:
    state = secrets.token_urlsafe(32)
    with _oauth_states_lock:
        now = time.time()
        stale = [k for k, v in _oauth_states.items() if now - v["created_at"] > 600]
        for k in stale:
            del _oauth_states[k]
        _oauth_states[state] = {"telegram_user_id": telegram_user_id, "chat_id": chat_id, "created_at": now}
    return state

def _create_web_oauth_state(web_user_id: str, return_url: str) -> str:
    """Same idea as _create_oauth_state, but for the web app's 'Link account'
    button — no Telegram chat_id to notify, and it redirects back into the
    web app instead of posting a Telegram message when it's done."""
    state = secrets.token_urlsafe(32)
    with _oauth_states_lock:
        now = time.time()
        stale = [k for k, v in _oauth_states.items() if now - v["created_at"] > 600]
        for k in stale:
            del _oauth_states[k]
        _oauth_states[state] = {"web_user_id": web_user_id, "return_url": return_url, "created_at": now}
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

app = Flask(__name__)

# CORS: needed so browser-based apps on other subdomains (AIM Learn, and
# any future mini-apps) can call this API directly from JavaScript.
# Without this, the browser silently blocks the response and fetch()
# just looks like it hung — no error, no data, nothing.
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": [
    "https://learn.empireunion.xyz",
    "https://ai.empireunion.xyz",
    "https://empireunion.xyz",
    "http://localhost:3000",  # Next.js dev server for the aim-web app
]}})
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SECRET_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        logger.info("✅ Supabase connected")
    except Exception as e:
        logger.error("❌ Supabase connection failed: %s", e)
else:
    logger.warning("⚠️ Supabase not configured")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
deepseek_client: Optional[AsyncOpenAI] = None

# NOTE: deepseek_client is now created whenever a key exists, independent of
# USE_DEEPSEEK. Model *routing* (which model answers a given message) is
# decided per-user by plan tier in get_ai_response's TIER_MODEL_CHAIN below,
# not by this global flag. USE_DEEPSEEK is kept only as a legacy default for
# the handful of internal helper calls that don't carry a user plan.
if DEEPSEEK_API_KEY:
    deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    logger.info("✅ DeepSeek API available (used for Pro-tier routing)")
if GEMINI_API_KEY:
    logger.info("✅ Gemini API available (used for Free/Basic-tier routing)")
if not DEEPSEEK_API_KEY and not GEMINI_API_KEY:
    logger.warning("⚠️ No AI API configured!")

groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
if GROQ_API_KEY:     logger.info("✅ Groq API (Voice STT enabled)")
if BRAVE_API_KEY:    logger.info("✅ Brave Search API")
if GNEWS_API_KEY:    logger.info("✅ GNews API")
if SPORTAPI_KEY:     logger.info("✅ API-FOOTBALL (fixture lookups enabled)")
else:                logger.warning("⚠️ SPORTAPI_KEY not set — reminders tied to match kickoff times will fall back to news search")
_logto_ok = bool(os.environ.get("LOGTO_ENDPOINT")) and bool(os.environ.get("LOGTO_CLIENT_ID"))
if _logto_ok:
    logger.info("✅ Logto OAuth configured → %s", get_redirect_uri())
else:
    logger.warning("⚠️ Logto not configured — /link will not work")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe billing configured")
else:
    logger.warning("⚠️ Stripe not configured — billing endpoints will be unavailable")

load_admins(supabase)

_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()
threading.Thread(target=_run_loop, daemon=True, name="async-loop").start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop)

async def dynamic_status_updater(bot_instance, chat_id, message_id, phrases, stop_event):
    idx = 0
    try:
        while not stop_event.is_set():
            text = phrases[idx % len(phrases)]
            try:
                await bot_instance.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
            except Exception:
                pass
            idx += 1
            await asyncio.sleep(1.5)
    except asyncio.CancelledError:
        pass

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

async def _get_recent_words(user_id: str, limit: int = 30) -> list:
    """Fetch the words already sent to this user recently so we never repeat one.
    Requires a `word_of_day_history` table: user_id text, word text, sent_at timestamptz."""
    if not supabase:
        return []
    try:
        res = (supabase.table("word_of_day_history").select("word")
               .eq("user_id", str(user_id)).order("sent_at", desc=True).limit(limit).execute())
        return [r["word"].strip().lower() for r in (res.data or []) if r.get("word")]
    except Exception as e:
        logger.error("word history fetch error: %s", e)
        return []

async def _save_sent_word(user_id: str, word: str):
    if not supabase or not word:
        return
    try:
        supabase.table("word_of_day_history").insert({
            "user_id": str(user_id), "word": word.strip(),
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error("word history save error: %s", e)

def _extract_headword(text: str) -> str:
    """Best-effort pull of just the headword from a formatted word-of-the-day
    message, so we can dedupe on the word itself rather than the whole blob."""
    if not text:
        return ""
    first_line = text.strip().split("\n")[0]
    first_line = re.sub(r"[*_`#>\-•]+", "", first_line).strip()
    word = re.split(r"[\s,:(\-–—]", first_line)[0] if first_line else ""
    return word.strip().lower()

def _synthesize_daily_content(category: str, raw_search: str, avoid_words: list = None) -> str:
    """Raw search scrapes (title/snippet/url) aren't fit to send straight to
    a user. This runs the scrape through the AI once to produce an actual
    usable message. For 'word', we also tell the model which words were
    already sent recently so it picks something new."""
    avoid_clause = ""
    if category == "word" and avoid_words:
        avoid_clause = (
            "\n\nDo NOT pick any of these words — they were already sent to this user "
            f"recently: {', '.join(avoid_words)}. Pick a genuinely different word."
        )
    prompts = {
        "news": (
            "Using ONLY these search results, write a short, well-formatted "
            "'Daily News Update' with the top headlines for a Nigerian audience. "
            "If the results are empty or unusable, say so plainly instead of "
            "inventing news.\n\nSearch results:\n" + raw_search
        ),
        "verse": (
            "These search results may contain a Bible verse for today. If you "
            "can clearly identify one, present it cleanly with its reference "
            "(e.g. John 3:16) and one short line of reflection. If the results "
            "are empty or not actually a verse, just choose an uplifting Bible "
            "verse yourself and present it the same way — don't mention that "
            "the search failed.\n\nSearch results:\n" + raw_search
        ),
        "word": (
            "These search results may contain today's dictionary word of the "
            "day. If you can clearly identify an actual word from them (not "
            "just a page title), present it as: the WORD, its part of speech, "
            "a one-line definition, and one example sentence. If the results "
            "are empty or don't contain a real word (e.g. just a page title "
            "with no definition), pick an interesting, useful English word "
            "yourself and present it the same way — don't mention that the "
            "search failed." + avoid_clause + "\n\nSearch results:\n" + raw_search
        ),
    }
    prompt = prompts.get(category)
    if not prompt:
        return raw_search
    try:
        future = run_async(get_ai_response(prompt, "system", "private"))
        result = future.result(timeout=20)
        return result.strip() if result else raw_search
    except Exception as e:
        logger.error("Daily content synthesis error (%s): %s", category, e)
        return raw_search

async def _get_deduped_word_of_day(user_id: str) -> str:
    """Generate a word-of-the-day message that hasn't been sent to this user
    before, retrying a couple of times if the model repeats itself."""
    raw = search_web("word of the day meaning", 1)
    avoid = await _get_recent_words(user_id)
    for attempt in range(3):
        content = _synthesize_daily_content("word", raw, avoid_words=avoid)
        headword = _extract_headword(content)
        if not headword or headword not in avoid:
            if headword:
                await _save_sent_word(user_id, headword)
            return content
        # Model repeated a used word — add it to the avoid list and try again
        avoid.append(headword)
    return content  # give up after 3 tries, send it anyway rather than fail silently

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
                msg = None
                # CRITICAL: each task gets its own try/except. If content
                # generation fails (AI timeout, rate limit, etc.) we still
                # fall through to advancing next_run / deactivating below —
                # otherwise a single failure leaves next_run in the past and
                # the task fires again every 30s forever (infinite retry loop
                # that was silently burning the entire AI quota).
                try:
                    if category == "news":
                        raw = get_latest_news("Nigeria latest news today", 3)
                        content = _synthesize_daily_content("news", raw)
                        msg = f"📰 <b>Your Daily News Update:</b>\n\n{content[:3000]}"
                    elif category == "verse":
                        raw = search_web("daily bible verse today", 1)
                        content = _synthesize_daily_content("verse", raw)
                        msg = f"📖 <b>Daily Bible Verse:</b>\n\n{content[:1000]}"
                    elif category == "word":
                        future = run_async(_get_deduped_word_of_day(user_id))
                        content = future.result(timeout=25)
                        msg = f"📚 <b>Word of the Day:</b>\n\n{content[:500]}"
                    else:
                        msg = f"⏰ <b>Reminder:</b>\n\n{description}"
                except Exception as gen_err:
                    logger.error("Task content generation error (%s): %s", category, gen_err)
                    msg = None  # skip sending this cycle, but still advance/deactivate below

                if msg:
                    try:
                        run_async(send_text_chunks(int(user_id), msg))
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
    return False

def _api_football_get(path: str, params: dict) -> Optional[dict]:
    if not SPORTAPI_KEY: return None
    try:
        resp = requests.get(
            f"https://v3.football.api-sports.io/{path}",
            headers={"x-apisports-key": SPORTAPI_KEY},
            params=params, timeout=10
        )
        if resp.status_code != 200:
            logger.warning("API-FOOTBALL %s returned %s", path, resp.status_code)
            return None
        return resp.json()
    except Exception as e:
        logger.error("API-FOOTBALL error: %s", e)
        return None

def _find_team_id(name: str) -> Optional[int]:
    data = _api_football_get("teams", {"search": name})
    if not data or not data.get("response"):
        return None
    return data["response"][0]["team"]["id"]

_TEAM_CONNECTOR_RE = re.compile(
    r'\b([A-Za-z][A-Za-z\.]*(?:\s+[A-Za-z][A-Za-z\.]*){0,2})'
    r'\s+(?:vs\.?|versus|v\.?|and|against)\s+'
    r'([A-Za-z][A-Za-z\.]*(?:\s+[A-Za-z][A-Za-z\.]*){0,2})\b',
    re.IGNORECASE
)
_TEAM_NOCONNECTOR_RE = re.compile(
    r'\b([A-Za-z]+)\s+([A-Za-z]+)\s+(?:match|game|fixture|kickoff|kick off)\b',
    re.IGNORECASE
)
_SKIP_WORDS = {"the","a","an","this","that","next","upcoming","today","tomorrow","when","does","is","remind","me"}

def get_fixture_datetime(query_text: str) -> Optional[str]:
    if not SPORTAPI_KEY:
        return None

    candidates = []
    m = _TEAM_CONNECTOR_RE.search(query_text)
    if m:
        candidates.append((m.group(1).strip(), m.group(2).strip()))
    m2 = _TEAM_NOCONNECTOR_RE.search(query_text)
    if m2:
        a, b = m2.group(1).strip(), m2.group(2).strip()
        if a.lower() not in _SKIP_WORDS and b.lower() not in _SKIP_WORDS:
            candidates.append((a, b))
    if not candidates:
        return None

    id_a = id_b = None
    for team_a_name, team_b_name in candidates:
        id_a, id_b = _find_team_id(team_a_name), _find_team_id(team_b_name)
        if id_a and id_b:
            break
    if not id_a or not id_b:
        return None
    h2h = _api_football_get("fixtures/headtohead", {"h2h": f"{id_a}-{id_b}"})
    if not h2h or not h2h.get("response"):
        return None
    now_utc = datetime.now(timezone.utc)
    upcoming = []
    for fx in h2h["response"]:
        try:
            fx_date = datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
            if fx_date >= now_utc:
                upcoming.append((fx_date, fx))
        except Exception:
            continue
    if not upcoming:
        return None
    upcoming.sort(key=lambda x: x[0])
    fx_date, fx = upcoming[0]
    home = fx["teams"]["home"]["name"]
    away = fx["teams"]["away"]["name"]
    league = fx.get("league", {}).get("name", "")
    venue = fx.get("fixture", {}).get("venue", {}).get("name", "")
    return (
        f"Fixture found via API-FOOTBALL: {home} vs {away}"
        f"{' (' + league + ')' if league else ''} kicks off at "
        f"{fx_date.isoformat()} (UTC).{' Venue: ' + venue + '.' if venue else ''}"
    )

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

def aim_deepsearch(query: str, max_links: int = 3) -> str:
    """DeepSearch mode: unlike a normal search (snippets only), this fetches
    the top few results AND actually visits each page to pull real content,
    not just the search engine's short description. Slower and more
    expensive than search_web() — only use when DeepSearch mode is active."""
    results = _search_brave_scrape(query, max_links) or _search_brave_api(query, max_links) or _search_duckduckgo_lite(query, max_links)
    if not results:
        return "DeepSearch found no results to visit."
    lines = ["=== DEEPSEARCH RESULTS (full page content) ===\n"]
    for i, r in enumerate(results[:max_links], 1):
        page_content = fetch_url_content(r["url"])
        if not page_content or "Failed" in page_content:
            page_content = r.get("description", "(could not read full page)")
        lines.append(f"{i}. {r['title']}\n   Source: {r['url']}\n   Content: {page_content[:1500]}")
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

async def get_user_profile_data(user_id: str) -> dict:
    if not supabase: return {}
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        return rows.data[0] if rows.data else {}
    except Exception as e:
        logger.error("Profile fetch error: %s", e)
        return {}

async def set_pending_action(user_id: str, action_type: str, payload: dict = None):
    if not supabase: return
    try:
        supabase.table("user_pending_actions").upsert({
            "user_id": str(user_id),
            "action_type": action_type,
            "payload": payload or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error("set_pending_action error: %s", e)

async def get_pending_action(user_id: str) -> Optional[dict]:
    if not supabase: return None
    try:
        res = supabase.table("user_pending_actions").select("*").eq("user_id", str(user_id)).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error("get_pending_action error: %s", e)
        return None

async def clear_pending_action(user_id: str):
    if not supabase: return
    try:
        supabase.table("user_pending_actions").delete().eq("user_id", str(user_id)).execute()
    except Exception as e:
        logger.error("clear_pending_action error: %s", e)

DELETE_TABLE_MAP = {
    "profile": ["user_profiles"],
    "memory":  ["chat_memory"],
    "tasks":   ["user_tasks", "user_tools"],
    "all":     ["chat_memory", "user_tasks", "user_tools", "user_profiles"],
}
DELETE_LABELS = {
    "profile": "your profile, preferences & Empire ID link",
    "memory":  "your chat memory",
    "tasks":   "your tasks & reminders",
    "all":     "ALL your data",
}

async def classify_delete_intent(text: str) -> str:
    sys_prompt = (
        "You classify what a user wants deleted from their bot account. "
        "Reply with ONE WORD ONLY, no punctuation, no explanation: "
        "PROFILE, MEMORY, TASKS, ALL, or CANCEL.\n"
        "PROFILE = profile/preferences/account link. MEMORY = chat history/memory. "
        "TASKS = reminders/timers/scheduled tasks. ALL = everything. "
        "CANCEL = user does not want to delete anything / changed their mind."
    )
    try:
        word = ""
        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
                temperature=0.0, max_tokens=10,
            )
            word = (r.choices[0].message.content or "").strip().upper() if r.choices else ""
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=f"{sys_prompt}\n\nUser: {text}")])],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
            )
            word = (r.text or "").strip().upper() if r and r.text else ""
        for key in ("PROFILE", "MEMORY", "TASKS", "ALL", "CANCEL"):
            if key in word:
                return key.lower()
    except Exception as e:
        logger.error("classify_delete_intent error: %s", e)
    return "unknown"

async def execute_delete(user_id: str, choice: str) -> str:
    if choice not in DELETE_TABLE_MAP:
        return "❌ I didn't catch which one you meant — try /delete again."
    if not supabase:
        return "❌ Cannot delete — database offline."
    try:
        uid_str = str(user_id)

        # Profile/ALL deletion also removes the shared Empire ID record
        # (via the Empire ID service, same as the websites use) and
        # queues their Logto login for manual removal — see that
        # service's /v1/empire-id/by-logto/<id> DELETE route for why
        # login deletion is queued rather than instant right now.
        extra_note = ""
        if choice in ("profile", "all"):
            try:
                profile = await get_user_profile_data(user_id)
                logto_id = profile.get("logto_id")
                if logto_id and EMPIRE_ID_SERVICE_URL:
                    r = requests.delete(
                        f"{EMPIRE_ID_SERVICE_URL}/v1/empire-id/by-logto/{logto_id}",
                        params={"source": "telegram_bot"},
                        headers=_eid_headers(), timeout=10,
                    )
                    if r.ok:
                        extra_note = " Your Empire ID login will be fully removed within 48 hours."
                    else:
                        logger.error("Empire ID service delete failed: %s", r.text)
            except Exception as e:
                logger.error("Empire ID cleanup during /delete failed: %s", e)

        for table in DELETE_TABLE_MAP[choice]:
            supabase.table(table).delete().eq("user_id", uid_str).execute()
        logger.info("🗑️ User %s deleted: %s", uid_str, choice)
        return f"✅ Deleted {DELETE_LABELS[choice]}.{extra_note}"
    except Exception as e:
        logger.error("Delete error (%s): %s", choice, e)
        return "❌ Something went wrong during deletion. Please try again."

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

async def get_conversation_context(user_id: str, query_text: str, plan: str = "free") -> tuple[str, str, float]:
    if not supabase: return "", "", 0.0
    # Tier-aware depth: free users get a short recent-only window (cheapest
    # on tokens), basic gets recent + light older-context, pro gets the full
    # window. This is the main lever for keeping "unlimited" tiers affordable.
    tier_limits = {
        "free":  {"recent": 5,  "older": 0},
        "basic": {"recent": 15, "older": 10},
        "pro":   {"recent": 40, "older": 10},
    }
    limits = tier_limits.get((plan or "free").lower(), tier_limits["free"])
    fetch_count = max(limits["recent"] + limits["older"], limits["recent"])
    try:
        rows = supabase.table("chat_memory").select("message,response,topic,created_at").eq("user_id",str(user_id)).order("created_at",desc=True).limit(max(fetch_count, 40)).execute()
        if not rows.data: return "", "", 0.0
        gap_seconds = 0.0
        try:
            last_ts     = datetime.fromisoformat(rows.data[0]["created_at"].replace("Z","+00:00"))
            gap_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
        except Exception: pass
        recent_rows  = list(reversed(rows.data[:limits["recent"]]))
        recent_lines = []
        for row in recent_rows:
            try:    tstr = _fmt_wat(datetime.fromisoformat(row["created_at"].replace("Z","+00:00")))
            except: tstr = "unknown time"
            recent_lines += [f"  [{tstr}]", f"  User : {row['message']}", f"  AIM  : {row['response']}", ""]
        recent_history = "\n".join(recent_lines).strip()
        if limits["older"] <= 0:
            return recent_history, "", gap_seconds
        older_rows     = rows.data[limits["recent"]:]
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
        older_lines = [f"[{r.get('topic','general')}] User: {r['message']} | AIM: {r['response']}" for _, r in scored[:limits["older"]]]
        return recent_history, "\n".join(older_lines), gap_seconds
    except Exception as e:
        logger.error("Context retrieval error: %s", e)
        return "", "", 0.0

# ─── PER-TIER MODEL ROUTING ───
# AIM's own naming vs the actual underlying model, per the plan doc:
#   AIM Alpha 1 Mini → gemini-2.5-flash-lite   (free tier default)
#   AIM Alpha 1      → gemini-2.5-flash        (basic tier default)
#   AIM Alpha 1 Pro  → deepseek-v4 / deepseek-v4-flash when Pro is busy
# Each tier is a FALLBACK CHAIN, not a single model: if the first choice
# fails (rate limit, timeout, outage) we automatically drop to the next
# entry rather than returning nothing. Free/Basic always keep a Gemini
# fallback since that's the cheapest, most available option.
TIER_MODEL_CHAIN = {
    # NOTE (Aug 2026): DeepSeek retired the V3/V3.2/R1 lines on 2026-07-24 —
    # "deepseek-chat" / "deepseek-reasoner" aliases now error out. Only two
    # live models exist: deepseek-v4-flash (cheap) and deepseek-v4-pro
    # (stronger, ~3x the cost). DeepSeek is primary here since Gemini
    # billing is currently broken on this account; Gemini is kept as a
    # last-resort fallback so it silently starts working again on its own
    # once billing is fixed, with no code change needed.
    "free":  [("deepseek", "deepseek-v4-flash"), ("gemini", "gemini-2.5-flash-lite")],
    "basic": [("deepseek", "deepseek-v4-flash"), ("gemini", "gemini-2.5-flash")],
    "pro":   [("deepseek", "deepseek-v4-pro"), ("deepseek", "deepseek-v4-flash"), ("gemini", "gemini-2.5-flash")],
}

async def _call_gemini(model_name: str, system_prompt: str, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
    if not gemini_client:
        return None
    full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    r = gemini_client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part(text=full_text)])],
        config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens),
    )
    return r.text if r and r.text else None

async def _call_deepseek(model_name: str, system_prompt: str, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
    if not deepseek_client:
        return None
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
    r = await deepseek_client.chat.completions.create(model=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content if r.choices else None

async def _route_to_model(plan: str, system_prompt: str, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> Optional[str]:
    """Try each model in the plan's fallback chain in order; return the
    first successful response. Logs which link in the chain actually
    answered, and why earlier links were skipped, for debugging."""
    chain = TIER_MODEL_CHAIN.get((plan or "free").lower(), TIER_MODEL_CHAIN["free"])
    for provider, model_name in chain:
        try:
            if provider == "gemini":
                result = await _call_gemini(model_name, system_prompt, prompt, temperature, max_tokens)
            elif provider == "deepseek":
                result = await _call_deepseek(model_name, system_prompt, prompt, temperature, max_tokens)
            else:
                continue
            if result:
                return result
        except Exception as e:
            logger.warning("Model routing: %s/%s failed (%s), trying next in chain", provider, model_name, e)
            continue
    logger.error("Model routing: entire fallback chain exhausted for plan=%s", plan)
    return None

async def get_ai_response(
    user_text: str, user_id: str, chat_type: str, profile: dict = None,
    session_summary: str = "", recent_history: str = "", older_context: str = "",
    web_context: str = "", tool_status: str = "", gap_seconds: float = 0.0,
    plan: str = None, mode: str = None
) -> Optional[str]:
    try:
        if profile is None: profile = await get_user_profile_data(user_id)
        if plan is None: plan = (profile.get("plan") or "free")
        prompt = build_enhanced_prompt(
            user_text, user_id, profile, is_admin,
            session_summary, recent_history, older_context,
            web_context, tool_status, gap_seconds
        )
        system_prompt = apply_mode(BASE_SYSTEM_PROMPT, mode)
        gen_overrides = get_generation_overrides(mode)
        temperature = gen_overrides.get("temperature", 0.7)
        max_tokens   = gen_overrides.get("max_tokens", 1024)
        return await _route_to_model(plan, system_prompt, prompt, temperature=temperature, max_tokens=max_tokens)
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

async def save_chat_memory(user_id: str, username: str, message: str, response: str, chat_type: str, topic: str = "general", conversation_id: str = None):
    if not supabase: return
    try:
        row = {"user_id":str(user_id),"username":username or "","message":message[:2000],"response":response[:2000],"chat_type":chat_type,"topic":topic}
        if conversation_id:
            row["conversation_id"] = conversation_id
        supabase.table("chat_memory").insert(row).execute()
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

SAFE_CHUNK_LIMIT = 3800  # stay comfortably under Telegram's 4096 hard limit

def _split_into_chunks(text: str, limit: int = SAFE_CHUNK_LIMIT) -> list:
    """Split long text into at most a few human-readable chunks, breaking on
    paragraph boundaries first, then sentence boundaries, then hard-wrapping
    as a last resort. Keeps it to a handful of messages, not dozens."""
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        # Prefer breaking on a paragraph break, then a sentence end, then a space
        split_at = window.rfind("\n\n")
        if split_at < limit * 0.4:
            split_at = window.rfind(". ")
            if split_at >= 0:
                split_at += 1  # keep the period with the preceding chunk
        if split_at < limit * 0.4:
            split_at = window.rfind("\n")
        if split_at < limit * 0.4:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = limit  # hard cut, no good break point found

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks

async def send_text_chunks(
    chat_id: int,
    text: str,
    reply_to: Optional[int] = None,
    message_id: Optional[int] = None,
    parse_mode=ParseMode.HTML,
    platform: str = "telegram",
):
    """
    Send a response to the user.
    platform="telegram"  -> split into chunks, respect Telegram char limit
    platform="web"/"mobile" -> send full text unsplit
    """
    if not bot: return

    # Editing: always truncate — Telegram cannot edit into multiple messages
    if message_id:
        safe = text[:TELEGRAM_MAX_CHARS] if platform == "telegram" else text
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=safe, parse_mode=parse_mode)
        except Exception as e:
            logger.error("Edit error: %s", e)
            try:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=safe)
            except Exception as e2:
                logger.error("Fallback edit failed: %s", e2)
        return

    # Non-Telegram platforms: send once with no char limit
    if platform != "telegram":
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            logger.error("Non-telegram send error: %s", e)
        return

    # Telegram: split into chunks and send each part
    chunks = _split_into_chunks(text)
    for i, chunk in enumerate(chunks):
        try:
            kw = {"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode}
            if reply_to and i == 0:
                kw["reply_to_message_id"] = reply_to
            await bot.send_message(**kw)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.3)
        except Exception as e:
            logger.error("Send error chunk %d: %s", i, e)
            try:
                kw2 = {"chat_id": chat_id, "text": chunk}
                if reply_to and i == 0:
                    kw2["reply_to_message_id"] = reply_to
                await bot.send_message(**kw2)
            except Exception as e2:
                logger.error("Fallback send failed: %s", e2)

_EMPIRE_INTENT_PHRASES = [
    "empire id", "create my id", "get my id", "empire account",
    "link my account", "link telegram", "connect my account",
    "create account", "sign up", "make an account", "web app account",
    "link to web", "connect to web", "i want an account", "how do i sign up",
    "register", "web version", "use on web", "access on web",
]

async def _handle_learning_query(user_id: str, chat_id: int, message_id: int, user_text: str):
    """Handle chess/learning queries — reads from user_learning_profiles and chess tables."""
    if not supabase:
        await send_text_chunks(chat_id, "❌ Learning data is offline. Try again later.", reply_to=message_id)
        return

    profile = await get_user_profile_data(user_id)
    empire_id = profile.get("empire_id")

    if not empire_id:
        await send_text_chunks(chat_id, (
            "🎓 You need an Empire ID to track your learning progress!\n\n"
            "Tap /link to connect your account, then I can see everything you learn. 🇳🇬"
        ), reply_to=message_id)
        return

    tl = user_text.lower()

    if any(kw in tl for kw in ["chess", "elo", "games", "checkmate", "pawn", "knight", "bishop", "rook", "queen", "king"]):
        try:
            lp_res = supabase.table("user_learning_profiles").select("*").eq("user_id", user_id).single().execute()

            lp = lp_res.data if lp_res.data else None

            games_res = supabase.table("chess_games").select("result, difficulty, created_at, moves").eq("user_id", user_id).order("created_at", desc=True).limit(5).execute()

            games = games_res.data or []

            if not lp and not games:
                await send_text_chunks(chat_id, (
                    f"🎓 You haven't played any chess games yet!\n\n"
                    f"Start learning here:\n"
                    f'👉 <a href="https://learn.empireunion.xyz/learn/chess?empire_id={empire_id}">Play Chess</a>\n\n'
                    f"I'll track your progress and coach you as you improve! ♟️"
                ), reply_to=message_id)
                return

            games_played = lp.get("chess_games_played", 0) if lp else len(games)
            wins = lp.get("chess_wins", 0) if lp else sum(1 for g in games if g.get("result") == "win")
            losses = lp.get("chess_losses", 0) if lp else sum(1 for g in games if g.get("result") == "loss")
            draws = lp.get("chess_draws", 0) if lp else sum(1 for g in games if g.get("result") == "draw")
            elo = lp.get("chess_elo", 400) if lp else 400
            weaknesses = lp.get("chess_weaknesses", []) if lp else []

            win_rate = round((wins / games_played * 100), 1) if games_played > 0 else 0

            msg_lines = [
                f"♟️ <b>Your Chess Stats</b>",
                f"",
                f"📊 Games: <b>{games_played}</b> (W: {wins} | L: {losses} | D: {draws})",
                f"🏆 Win Rate: <b>{win_rate}%</b>",
                f"⭐ ELO Rating: <b>{elo}</b>",
            ]

            if weaknesses:
                msg_lines.append(f"")
                msg_lines.append(f"💡 Areas to improve: {', '.join(weaknesses[:3])}")

            if games:
                msg_lines.append(f"")
                msg_lines.append(f"📜 Recent games:")
                for i, g in enumerate(games[:3], 1):
                    result_emoji = {"win": "🏆", "loss": "💔", "draw": "🤝"}.get(g.get("result"), "❓")
                    date = g.get("created_at", "")[:10]
                    msg_lines.append(f"   {i}. {result_emoji} {g.get('result', 'unknown').title()} — {date}")

            msg_lines.append(f"")
            msg_lines.append(f'🎯 <a href="https://learn.empireunion.xyz/learn/chess?empire_id={empire_id}&skip_intro=true">Continue Playing</a>')
            msg_lines.append(f'🎓 <a href="https://learn.empireunion.xyz/learn">Explore Other Topics</a>')

            await send_text_chunks(chat_id, "\n".join(msg_lines), reply_to=message_id)

        except Exception as e:
            logger.error("Chess query error: %s", e)
            await send_text_chunks(chat_id, (
                f"🎓 Start your chess journey here:\n"
                f'👉 <a href="https://learn.empireunion.xyz/learn/chess?empire_id={empire_id}">Play Chess</a>'
            ), reply_to=message_id)
        return

    await send_text_chunks(chat_id, (
        f"🎓 <b>Empire Learn</b> — What would you like to master?\n\n"
        f'♟️ <a href="https://learn.empireunion.xyz/learn/chess?empire_id={empire_id}">Chess</a>\n'
        f'📐 <a href="https://learn.empireunion.xyz/learn/math?empire_id={empire_id}">Math</a>\n'
        f'🌍 <a href="https://learn.empireunion.xyz/learn/language?empire_id={empire_id}">Language</a>\n\n'
        f'Or just tell me: "Teach me [topic]" and I\'ll guide you! 🇳🇬'
    ), reply_to=message_id)

async def _handle_link_command(user_id: str, chat_id: int, message_id: int):
    ep  = os.environ.get("LOGTO_ENDPOINT", "").rstrip("/")
    cid = os.environ.get("LOGTO_CLIENT_ID", "")
    if not ep or not cid:
        await send_text_chunks(chat_id, "⚠️ Web login isn't set up yet — check back soon!", reply_to=message_id)
        return
    profile = await get_user_profile_data(user_id)
    if profile.get("logto_id"):
        empire_id = profile.get("empire_id", "N/A")
        await send_text_chunks(chat_id, (
            f"✅ <b>Your account is already linked!</b>\n\n"
            f"🆔 Empire ID: <b>{empire_id}</b>\n"
            f"📧 Email: {profile.get('logto_email') or 'N/A'}\n\n"
            f"When the web app launches, sign in with your Empire ID and AIM will remember everything. 🌍🇳🇬"
        ), reply_to=message_id)
        return
    state    = _create_oauth_state(user_id, chat_id)
    auth_url = build_logto_auth_url(state)
    await send_text_chunks(chat_id, (
        "🔗 <b>Link your Telegram to Empire AI</b>\n\n"
        "Tap below to sign up or log in. Once done, your Telegram memory "
        "will be available on the Empire AI web app when it launches.\n\n"
        f'👉 <a href="{auth_url}">Create / Sign into your Empire Account</a>\n\n'
        "⏳ Link expires in <b>10 minutes</b>."
    ), reply_to=message_id)

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
• Generate images, audio, PDFs, and code files
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
/delete — Delete all your data

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

    elif tl == "/news":
        await set_pending_action(user_id, "awaiting_recurring_time", {"category": "news"})
        await send_text_chunks(chat_id, "🕒 What time should I send your daily news? (e.g. 8am, 6pm)", reply_to=message_id)
        return True

    elif tl == "/verse":
        await set_pending_action(user_id, "awaiting_recurring_time", {"category": "verse"})
        await send_text_chunks(chat_id, "🕒 What time should I send your daily verse? (e.g. 8am — default is 8am if you skip this)", reply_to=message_id)
        return True

    elif tl == "/word":
        await set_pending_action(user_id, "awaiting_recurring_time", {"category": "word"})
        await send_text_chunks(chat_id, "🕒 What time should I send your word of the day? (e.g. 9am — default is 9am if you skip this)", reply_to=message_id)
        return True

    elif tl == "/claim":
        profile     = await get_user_profile_data(user_id)
        existing_id = profile.get("empire_id")
        if existing_id:
            await send_text_chunks(chat_id, f"👑 Your Empire ID: <b>{existing_id}</b>\n\nSave it! You'll use it to log into the Web App.", reply_to=message_id)
        elif profile.get("logto_id"):
            eid_record = eid_get_by_logto(profile["logto_id"])
            if eid_record and eid_record.get("empire_id"):
                empire_id = eid_record["empire_id"]
                try:
                    supabase.table("user_profiles").update({"empire_id": empire_id}).eq("user_id", str(user_id)).execute()
                    await send_text_chunks(chat_id, f"👑 Your Empire ID: <b>{empire_id}</b>\n\nSave it! You'll use it to log into the Web App.", reply_to=message_id)
                except Exception as e:
                    logger.error("Failed to save re-imported Empire ID: %s", e)
                    await send_text_chunks(chat_id, "❌ Found your Empire ID but couldn't save it — please try /claim again.", reply_to=message_id)
            else:
                await send_text_chunks(chat_id, "❓ Couldn't find an Empire ID for your linked account. Try /link again to reconnect.", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "🆔 You don't have an Empire ID yet — it's created automatically when you sign in. Type /link to connect your account and get one!", reply_to=message_id)
        return True

    elif tl == "/delete":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("👤 Profile & preferences", callback_data="del:profile")],
            [InlineKeyboardButton("🧠 Chat memory", callback_data="del:memory")],
            [InlineKeyboardButton("📋 Tasks & reminders", callback_data="del:tasks")],
            [InlineKeyboardButton("🗑️ Everything (ALL)", callback_data="del:all")],
        ])
        await set_pending_action(user_id, "delete_choice")
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ <b>Delete Your Data</b>\n\n"
                "What would you like to delete?\n"
                "• <b>Profile & preferences</b> — saved preferences, language, Empire ID link\n"
                "• <b>Chat memory</b> — everything AIM remembers from your conversations\n"
                "• <b>Tasks & reminders</b> — timers, stopwatches, scheduled tasks\n"
                "• <b>Everything</b> — all of the above\n\n"
                "Tap a button below, or just type what you want deleted — any language works."
            ),
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
            reply_to_message_id=message_id,
        )
        return True

    elif tl.startswith("/learn") or tl.startswith("/chess"):
        profile = await get_user_profile_data(user_id)
        empire_id = profile.get("empire_id")

        if not empire_id:
            await send_text_chunks(chat_id, (
                "🎓 You need an Empire ID to access Empire Learn!\n\n"
                "Tap /link to connect your account first. 🇳🇬"
            ), reply_to=message_id)
            return True

        topic = "chess"
        if tl.startswith("/learn "):
            topic_arg = tl[7:].strip().lower()
            if topic_arg in ["chess", "math", "language"]:
                topic = topic_arg

        await send_text_chunks(chat_id, (
            f"🎓 Opening <b>{topic.title()}</b>...\n\n"
            f'👉 <a href="https://learn.empireunion.xyz/learn/{topic}?empire_id={empire_id}&skip_intro=true">'
            f"Click here to start learning</a>\n\n"
            f"Your progress is saved automatically. Come back anytime and I'll remember where you left off! 🇳🇬"
        ), reply_to=message_id)
        return True

    elif tl.startswith("/admin"):
        return await handle_admin_command(
            user_id, chat_id, message_id, user_text,
            supabase, get_ai_response, send_text_chunks, USE_DEEPSEEK,
            bot=bot, save_chat_memory=save_chat_memory,
        )

    return False

async def handle_callback_query_async(cb):
    data    = cb.data or ""
    user_id = str(cb.from_user.id) if cb.from_user else ""
    chat_id = cb.message.chat.id if cb.message else None
    try:
        await bot.answer_callback_query(cb.id)
    except Exception as e:
        logger.error("answer_callback_query error: %s", e)

    if data.startswith("del:") and user_id and chat_id:
        choice = data.split(":", 1)[1]
        result = await execute_delete(user_id, choice)
        await clear_pending_action(user_id)
        await send_text_chunks(chat_id, result)
        try:
            if cb.message:
                await bot.edit_message_reply_markup(chat_id=chat_id, message_id=cb.message.message_id, reply_markup=None)
        except Exception:
            pass

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
        recent, older, gap = await get_conversation_context(user_id, query_text, plan=(profile.get("plan") or "free"))
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

async def handle_message_async(update: Update):
    if not update.message: 
        logger.info("No message in update")
        return

    media_processed = False

    user       = update.message.from_user
    chat       = update.message.chat
    user_text  = update.message.text or ""
    chat_type  = chat.type if chat else "private"
    message_id = update.message.message_id

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
        media_processed = True

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
        media_processed = True

    if update.message.document:
        media_processed = True
        doc      = update.message.document
        doc_name = doc.file_name or "unknown_document"
        doc_mime = doc.mime_type or "application/octet-stream"
        temp_path = f"doc_{doc.file_id}_{doc_name}"
        msg = await bot.send_message(chat.id, "📄 Nebulae is opening the document...", reply_to_message_id=message_id)
        status_msg_id = msg.message_id
        stop_event = asyncio.Event()
        doc_phrases = ["📄 Opening document...", "📄 Reading pages...", "📄 Analyzing content..."]
        updater_task = asyncio.create_task(dynamic_status_updater(bot, chat.id, status_msg_id, doc_phrases, stop_event))
        try:
            file = await bot.get_file(doc.file_id)
            await file.download_to_drive(custom_path=temp_path)
            with open(temp_path, "rb") as f:
                doc_bytes = f.read()
            stop_event.set()
            await updater_task
            description = await nebulae.analyze_document(doc_bytes, doc_mime, doc_name, "Summarize and explain this document in detail.")
            final_text  = f"📝 <b>Document Analysis: {doc_name}</b>\n\n{description[:2800]}"
            try:
                await bot.edit_message_text(chat_id=chat.id, message_id=status_msg_id, text=final_text, parse_mode=ParseMode.HTML)
            except Exception:
                await send_text_chunks(chat.id, final_text, reply_to=message_id)
            user_text = f"[User sent document {doc_name}. Analysis: {description[:500]}]"
        except Exception as e:
            logger.error("Document error: %s", e)
            stop_event.set()
            await send_text_chunks(chat.id, "❌ Could not process this document.", reply_to=status_msg_id)
            return
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

    if update.message.video or update.message.animation:
        media_processed = True
        video_obj = update.message.video or update.message.animation
        temp_path = f"video_{video_obj.file_id}.mp4"
        msg = await bot.send_message(chat.id, "🎬 Nebulae is loading the video...", reply_to_message_id=message_id)
        status_msg_id = msg.message_id
        stop_event = asyncio.Event()
        vid_phrases = ["🎬 Loading video...", "🎬 Extracting frames...", "🎬 Analyzing..."]
        updater_task = asyncio.create_task(dynamic_status_updater(bot, chat.id, status_msg_id, vid_phrases, stop_event))
        try:
            file = await bot.get_file(video_obj.file_id)
            await file.download_to_drive(custom_path=temp_path)
            with open(temp_path, "rb") as vid_file:
                vid_bytes = vid_file.read()
            stop_event.set()
            await updater_task
            description = await nebulae.analyze_video(vid_bytes, "Describe what is happening in this video in detail.")
            final_text  = f"🎥 <b>Video Analysis:</b>\n{description[:2800]}"
            try:
                await bot.edit_message_text(chat_id=chat.id, message_id=status_msg_id, text=final_text, parse_mode=ParseMode.HTML)
            except Exception:
                await send_text_chunks(chat.id, final_text, reply_to=message_id)
            user_text = f"[User sent a video. Nebulae analysis: {description}]"
        except Exception as e:
            logger.error("Video error: %s", e)
            stop_event.set()
            await send_text_chunks(chat.id, "❌ Could not process this video.", reply_to=status_msg_id)
            return
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

    if not media_processed and not user_text:
        await send_text_chunks(chat.id, "I can only read text, voice, photos, documents, and videos.")
        return

    user_id  = str(user.id)
    username = user.username or user.first_name or "User"
    logger.info("📩 [%s/%s] '%s'", user_id, chat_type, user_text[:80])

    if not user_text.startswith("/"):
        pending = await get_pending_action(user_id)
        if pending:
            action_type = pending.get("action_type")
            payload     = pending.get("payload") or {}

            if action_type == "delete_choice":
                choice = await classify_delete_intent(user_text)
                await clear_pending_action(user_id)
                if choice == "cancel":
                    await send_text_chunks(chat.id, "👍 Okay, nothing was deleted.", reply_to=message_id)
                elif choice in ("profile", "memory", "tasks", "all"):
                    await send_text_chunks(chat.id, await execute_delete(user_id, choice), reply_to=message_id)
                else:
                    await set_pending_action(user_id, "delete_choice")
                    await send_text_chunks(chat.id, "❓ Sorry, I couldn't tell what you want deleted — try tapping a button above, or say it plainly (e.g. \"delete my memory\").", reply_to=message_id)
                return

            elif action_type == "awaiting_recurring_time":
                category = payload.get("category", "news")
                m = re.search(r'(\d{1,2})\s*(am|pm)?', user_text.lower())
                if m:
                    hour = int(m.group(1))
                    if m.group(2) == "pm" and hour != 12: hour += 12
                    if m.group(2) == "am" and hour == 12: hour = 0
                    rec_time = f"{hour:02d}:00"
                else:
                    rec_time = {"news": "08:00", "verse": "08:00", "word": "09:00"}.get(category, "08:00")
                desc_map = {"news": "Send me the latest news", "verse": "Daily bible verse", "word": "Word of the day"}
                task_data = {
                    "description": desc_map.get(category, "Daily update"), "type": "recurring",
                    "recurrence_pattern": "daily", "recurrence_time": rec_time,
                    "recurrence_days": [], "category": category, "needs_clarification": False,
                }
                await clear_pending_action(user_id)
                await send_text_chunks(chat.id, await create_task_in_db(user_id, task_data), reply_to=message_id)
                return

            else:
                await clear_pending_action(user_id)

    if user_text.startswith("/"):
        await clear_pending_action(user_id)
        cmd_handled = await handle_bot_command(user_id, chat.id, message_id, user_text)
        if cmd_handled:
            if not user_text.lower().strip().startswith("/admin"):
                try:
                    cmd_response = f"[Command handled: {user_text.split()[0]}]"
                    await save_chat_memory(user_id, username, user_text, cmd_response, chat_type, "general")
                    await update_user_profile(user_id, username, "general")
                except Exception as _me:
                    logger.error("Failed to save command to memory: %s", _me)
            return

    profile = await get_user_profile_data(user_id)

    is_ph, ph_query = is_inline_placeholder(user_text)
    if is_ph and ph_query:
        await process_inline_answer(chat.id, message_id, ph_query, user_id)
        return

    if chat_type in ("group", "supergroup"):
        mentioned      = "@askaimbot" in user_text.lower()
        replied_to_bot = (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and update.message.reply_to_message.from_user.is_bot
            and update.message.reply_to_message.from_user.username == "askaimbot"
        )
        if not mentioned and not replied_to_bot:
            return
        user_text = re.sub(r'@askaimbot', '', user_text, flags=re.IGNORECASE).strip()

    # Dispatch generation — this may cancel-and-merge with an in-flight
    # reply if the user sent a fast follow-up (see _pending_replies above).
    # Deliberately not awaited: handle_message_async returns immediately so
    # the webhook responds fast, while the actual generation continues as
    # its own background task.
    _dispatch_reply_with_coalescing(user_id, username, chat, message_id, user_text, chat_type, profile)


async def _generate_and_send_reply(user_id: str, username: str, chat, message_id: int, user_text: str, chat_type: str, profile: dict):
    """The full AI-generation-and-send pipeline, pulled out into its own
    function so it can run as a cancellable asyncio.Task — this is what
    lets a fast follow-up message cancel and merge into an in-flight reply
    instead of always waiting for the first answer to finish."""
    try:
        user_plan = (profile.get("plan") or "free")
        session_summary                            = await get_session_summary(user_id)
        recent_history, older_context, gap_seconds = await get_conversation_context(user_id, user_text, plan=user_plan)
        web_context = ""

        # ── Fetch any URLs the user pasted ──
        for url in detect_urls(user_text):
            c = fetch_url_content(url)
            if c and "Failed" not in c:
                web_context += f"URL content from {url}:\n{c[:2000]}\n"

        # ── AI decides if memory search is needed ──
        try:
            _mem_prompt = (
                f"User message: \"{user_text}\"\n"
                f"Does this require searching past conversation history? "
                f"(e.g. asking about previous topics, referencing earlier chats, asking if you remember something)\n"
                f"Reply ONLY: YES or NO"
            )
            _mem_dec = "NO"
            if USE_DEEPSEEK and deepseek_client:
                _mr = await deepseek_client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role":"user","content":_mem_prompt}],
                    temperature=0.1, max_tokens=5,
                )
                _mem_dec = _mr.choices[0].message.content.strip().upper() if _mr.choices else "NO"
            elif gemini_client:
                _mr = gemini_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[types.Content(role="user",parts=[types.Part(text=_mem_prompt)])],
                    config=types.GenerateContentConfig(temperature=0.1,max_output_tokens=5),
                )
                _mem_dec = _mr.text.strip().upper() if _mr and _mr.text else "NO"
            if "YES" in _mem_dec:
                kws = extract_search_keywords(user_text)
                _mem_res = await search_memory_by_keyword(user_id, user_text) if kws else await search_memory(user_id)
                if _mem_res and "haven't chatted" not in _mem_res:
                    web_context = f"--- RELEVANT PAST CONVERSATIONS ---\n{_mem_res}\n--- END ---\n" + web_context
        except Exception as _me:
            logger.error("AI memory probe error: %s", _me)

        # ── AI also decides if it needs to search the web ──
        if is_search_query(user_text) and not web_context:
            tl = user_text.lower()
            if "news" in tl or "latest" in tl or "today" in tl:
                sr = get_latest_news(user_text)
            elif any(s in tl for s in ["football","match","score","team","player","league","f1","nba","tennis","boxing","ufc","cricket","rugby"]):
                sr = get_sports_data(user_text)
            else:
                sr = search_web(user_text)
            if "No search results" not in sr:
                web_context = f"Web Search Results for '{user_text}':\n{sr}"

        max_iter, iteration, final_answer, tool_status = 4, 0, None, ""
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
                    sq_lower = sq.lower()
                    if any(w in sq_lower for w in ["match","kickoff","kick off","fixture","vs","versus"]):
                        sr = get_fixture_datetime(sq) or get_sports_data(sq)
                    elif "news" in sq_lower or "latest" in sq_lower: sr = get_latest_news(sq)
                    elif any(s in sq_lower for s in ["football","score","f1","nba","tennis","boxing"]): sr = get_sports_data(sq)
                    else: sr = search_web(sq)
                    web_context += f"\n\nWeb Search Results for '{sq}':\n{sr}" if sr and "No search results" not in sr else f"\n\nSearch for '{sq}': No results."
                    continue
                else:
                    final_answer = answer
                    break

            elm = re.search(r'\[OPEN_LEARNING:\s*(\w+)\]', answer, re.IGNORECASE)
            if elm:
                topic = elm.group(1).strip().lower()
                answer = re.sub(r'\[OPEN_LEARNING:\s*\w+\]', '', answer, flags=re.IGNORECASE).strip()
                if answer:
                    await send_text_chunks(chat.id, answer, reply_to=message_id)
                await _handle_learning_query(user_id, chat.id, message_id, f"teach me {topic}")
                final_answer = None
                break

            elk = re.search(r'\[EMPIRE_LINK\]', answer, re.IGNORECASE)
            if elk:
                answer = re.sub(r'\[EMPIRE_LINK\]', '', answer, flags=re.IGNORECASE).strip()
                if answer:
                    await send_text_chunks(chat.id, answer, reply_to=message_id)
                await _handle_link_command(user_id, chat.id, message_id)
                final_answer = None
                break

            ctm = re.search(r'\[CREATE_TASK:(\{.*?\})\]', answer, re.IGNORECASE | re.DOTALL)
            if ctm:
                answer = re.sub(r'\[CREATE_TASK:\{.*?\}\]', '', answer, flags=re.IGNORECASE | re.DOTALL).strip()
                try:
                    task_data = json.loads(ctm.group(1))
                    task_result = await create_task_in_db(user_id, task_data)
                    answer = f"{answer}\n\n{task_result}" if answer else task_result
                except Exception as e:
                    logger.error("CREATE_TASK parse error: %s", e)
                    answer = f"{answer}\n\n❌ Couldn't save that reminder — mind trying again?" if answer else "❌ Couldn't save that reminder — mind trying again?"

            tm = re.search(r'\[TIMER:(\d+)(s|m|h)\]', answer, re.IGNORECASE)
            if tm:
                amt, unit = int(tm.group(1)), tm.group(2).lower()
                dur    = amt * (1 if unit=="s" else 60 if unit=="m" else 3600)
                target = datetime.now(timezone.utc) + timedelta(seconds=dur)
                supabase.table("user_tools").insert({"user_id":user_id,"tool_type":"timer","start_time":datetime.now(timezone.utc).isoformat(),"duration_seconds":dur,"target_time":target.isoformat(),"is_active":True}).execute()
                answer = re.sub(r'\[TIMER:\d+[smh]\]', '', answer, flags=re.IGNORECASE).strip()
                answer += f"\n\n_✅ Timer set for {amt}{unit}_"

            sm = re.search(r'\[STOPWATCH:(START|STOP)\]', answer, re.IGNORECASE)
            if sm:
                action_sw = sm.group(1).upper()
                if action_sw == "START":
                    supabase.table("user_tools").insert({"user_id":user_id,"tool_type":"stopwatch","start_time":datetime.now(timezone.utc).isoformat(),"is_active":True}).execute()
                    answer = re.sub(r'\[STOPWATCH:START\]','',answer,flags=re.IGNORECASE).strip() + "\n\n_⏱️ Stopwatch started!_"
                elif action_sw == "STOP":
                    res = supabase.table("user_tools").select("*").eq("user_id",user_id).eq("tool_type","stopwatch").eq("is_active",True).order("created_at",desc=True).limit(1).execute()
                    if res.data:
                        row     = res.data[0]
                        elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row["start_time"].replace("Z","+00:00"))
                        mins, secs = divmod(int(elapsed.total_seconds()), 60)
                        supabase.table("user_tools").update({"is_active":False}).eq("id",row["id"]).execute()
                        ts = f"{mins}m {secs}s" if mins else f"{secs}s"
                        answer = re.sub(r'\[STOPWATCH:STOP\]','',answer,flags=re.IGNORECASE).strip() + f"\n\n_⏱️ Stopped! Time: {ts}_"

            code_match = re.search(
                r'\[CODE_FILE:(\w+)\|(.*?)\]\[/CODE_FILE\]',
                answer, re.IGNORECASE | re.DOTALL
            )
            if code_match:
                file_ext     = code_match.group(1).strip().lower()
                code_content = code_match.group(2).strip()
                ext_map = {"py":"python","js":"javascript","ts":"typescript","jsx":"javascript","tsx":"typescript","html":"html","css":"css","json":"json","sh":"bash","sql":"sql","java":"java","cpp":"c++","c":"c","rb":"ruby","go":"go","rs":"rust","swift":"swift","kt":"kotlin","php":"php","r":"r","md":"markdown","yml":"yaml","yaml":"yaml","xml":"xml","txt":"text"}
                if file_ext in ext_map:
                    await send_text_chunks(chat.id, f"💾 Generating {ext_map[file_ext]} file...", reply_to=message_id)
                    try:
                        from io import BytesIO as _BIO
                        code_file = _BIO(code_content.encode("utf-8"))
                        code_file.name = f"aim_code.{file_ext}"
                        await bot.send_document(chat_id=chat.id, document=code_file, filename=code_file.name, caption=f"💾 {ext_map[file_ext].title()} file by AIM", reply_to_message_id=message_id)
                        answer = f"✅ Here is your {ext_map[file_ext]} file!"
                    except Exception as _ce:
                        logger.error("Code file send error: %s", _ce)
                        answer = "❌ Couldn't send the code file — here it is inline instead:\n\n" + code_content[:3500]
                answer = re.sub(r'\[CODE_FILE:\w+\|.*?\]\[/CODE_FILE\]', '', answer, flags=re.IGNORECASE | re.DOTALL).strip()

            img_match = re.search(r'\[NEBULAE_IMAGE:\s*(.+?)\]', answer, re.IGNORECASE)
            if img_match:
                img_prompt = img_match.group(1).strip()
                await send_text_chunks(chat.id, "🎨 Nebulae is painting...", reply_to=message_id)
                img_bytes = await nebulae.generate_image(img_prompt)
                if img_bytes:
                    try:
                        await bot.send_photo(chat_id=chat.id, photo=img_bytes, caption="✨ Generated by Nebulae", reply_to_message_id=message_id)
                        answer = "✅ Here is your image!"
                    except Exception: answer = "❌ Image failed to send."
                else: answer = "❌ Nebulae couldn\'t generate the image."
                answer = re.sub(r'\[NEBULAE_IMAGE:.*?\]', '', answer, flags=re.IGNORECASE).strip()

            audio_match = re.search(r'\[NEBULAE_AUDIO:\s*(.+?)\]', answer, re.IGNORECASE)
            if audio_match:
                audio_text = audio_match.group(1).strip()
                await send_text_chunks(chat.id, "🔊 Nebulae is speaking...", reply_to=message_id)
                audio_bytes = await nebulae.generate_audio(audio_text)
                if audio_bytes:
                    try:
                        from io import BytesIO as _BIO
                        audio_file = _BIO(audio_bytes)
                        audio_file.name = "aim_audio.mp3"
                        await bot.send_audio(chat_id=chat.id, audio=audio_file, caption="🔊 Audio by Nebulae", reply_to_message_id=message_id)
                        answer = "✅ Here is your audio!"
                    except Exception as _ae:
                        logger.error("Audio send error: %s", _ae)
                        answer = "❌ Audio failed to send."
                else: answer = "❌ Nebulae couldn\'t generate the audio."
                answer = re.sub(r'\[NEBULAE_AUDIO:.*?\]', '', answer, flags=re.IGNORECASE).strip()

            pdf_match = re.search(r'\[NEBULAE_PDF:\s*([^\|]+)\|(.*?)\]', answer, re.IGNORECASE | re.DOTALL)
            if pdf_match:
                pdf_title   = pdf_match.group(1).strip()
                pdf_content = pdf_match.group(2).strip()
                await send_text_chunks(chat.id, "📄 Nebulae is generating your document...", reply_to=message_id)
                pdf_bytes = nebulae.generate_pdf(pdf_title, pdf_content)
                if pdf_bytes:
                    try:
                        from io import BytesIO as _BIO
                        pdf_file = _BIO(pdf_bytes)
                        pdf_file.name = f"{pdf_title.replace(' ','_')}.pdf"
                        await bot.send_document(chat_id=chat.id, document=pdf_file, filename=pdf_file.name, caption=f"📄 {pdf_title}", reply_to_message_id=message_id)
                        answer = "✅ Here is your PDF!"
                    except Exception as _pe:
                        logger.error("PDF send error: %s", _pe)
                        answer = "❌ PDF failed to send."
                else: answer = "❌ Nebulae couldn\'t generate the PDF."
                answer = re.sub(r'\[NEBULAE_PDF:.*?\]', '', answer, flags=re.IGNORECASE | re.DOTALL).strip()

            final_answer = answer
            break

        if final_answer is None and iteration >= max_iter:
            final_answer = "I tried searching but couldn\'t find results."

        if final_answer is not None:
            await send_text_chunks(chat.id, final_answer, reply_to=message_id)
            topic = await extract_topic(user_text, final_answer)
            await save_chat_memory(user_id, username, user_text, final_answer, chat_type, topic)
            await update_user_profile(user_id, username, topic)
        else:
            await save_chat_memory(user_id, username, user_text, "[Handled via action tag]", chat_type, "education")
            await update_user_profile(user_id, username, "education")

        if profile.get("total_chats", 0) % 4 == 0:
            recent_msgs = supabase.table("chat_memory").select("message,response").eq("user_id",str(user_id)).order("created_at",desc=True).limit(4).execute()
            if recent_msgs.data:
                recent_msgs.data.reverse()
                run_async(update_session_summary(user_id, recent_msgs.data, session_summary))

    except asyncio.CancelledError:
        # Expected when a fast follow-up message supersedes this generation
        # (see _pending_replies coalescing logic in handle_message_async).
        # Not an error — just let it end quietly.
        logger.info("Reply generation for %s cancelled — merging into follow-up", user_id)
        raise
    except Exception as e:
        logger.error("Critical error in _generate_and_send_reply: %s", e)
        await send_text_chunks(chat.id, "🛠️ Something went wrong.", reply_to=message_id)
    finally:
        # Clean up the pending-reply slot once this task is done, but only
        # if it's still ours (a newer merged task may have already replaced it).
        with _pending_replies_lock:
            entry = _pending_replies.get(user_id)
            if entry and entry.get("current_text") == user_text:
                _pending_replies.pop(user_id, None)


def _dispatch_reply_with_coalescing(user_id: str, username: str, chat, message_id: int, user_text: str, chat_type: str, profile: dict):
    """Decides whether this message should start a fresh reply, or cancel
    and merge into one that's still in-flight for this same user. Never
    awaits the generation itself — it fires the task and returns instantly,
    so normal (non-follow-up) messages are answered at full speed with no
    added delay."""
    now = time.time()
    with _pending_replies_lock:
        existing = _pending_replies.get(user_id)
        if existing and not existing["task"].done() and (now - existing["started_at"]) <= COALESCE_WINDOW_SECONDS:
            existing["task"].cancel()
            combined_text = f"{existing['current_text']} {user_text}".strip()
        else:
            combined_text = user_text

        new_task = asyncio.create_task(
            _generate_and_send_reply(user_id, username, chat, message_id, combined_text, chat_type, profile)
        )
        _pending_replies[user_id] = {"task": new_task, "started_at": now, "current_text": combined_text}

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"AIM Bot is live!","version":"v9.8","ai":"DeepSeek V4" if USE_DEEPSEEK else "Gemini"})

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        uid = data.get("update_id")
        if uid and is_duplicate_update(uid): return "OK", 200
        upd = Update.de_json(data, bot)
        if upd.inline_query:      run_async(handle_inline_query_async(upd.inline_query))
        elif upd.callback_query:  run_async(handle_callback_query_async(upd.callback_query))
        elif upd.message:         run_async(handle_message_async(upd))
        return "OK", 200
    except Exception as e:
        logger.error("Webhook error: %s", e)
        return "Error", 500

@app.route("/api/web-link/start", methods=["GET"])
def web_link_start():
    """Kicks off the same Logto login as Telegram's /link, but for the
    web app. The frontend just sets window.location to this URL; we
    redirect straight into Logto, and /auth/callback below sends the
    browser back to `return_url` once linking is done."""
    web_user_id = request.args.get("user_id", "")
    return_url = request.args.get("return_url", "")
    if not web_user_id or not return_url:
        return jsonify({"error": "user_id and return_url are required"}), 400
    if not _logto_ok:
        return jsonify({"error": "Logto is not configured on the server"}), 503
    state = _create_web_oauth_state(web_user_id, return_url)
    return redirect(build_logto_auth_url(state))

@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    code  = request.args.get("code","")
    state = request.args.get("state","")
    error = request.args.get("error","")

    ctx = None if error else _consume_oauth_state(state)
    is_web = bool(ctx and "web_user_id" in ctx)

    def _fail_redirect_or_page(title, body, query):
        if is_web and ctx and ctx.get("return_url"):
            sep = "&" if "?" in ctx["return_url"] else "?"
            return redirect(f"{ctx['return_url']}{sep}{query}")
        return _html_page(title, body, success=False)

    if error:
        logger.warning("Logto error: %s", error)
        return _fail_redirect_or_page("❌ Login Failed", f"<p>Error: {error}</p>", "linked=error"), (302 if is_web else 400)

    if not ctx:
        return _fail_redirect_or_page("❌ Invalid Session", "<p>This link has expired or is invalid. Please try again.</p>", "linked=expired"), (302 if is_web else 400)

    claims = exchange_logto_code(code)
    if not claims:
        if not is_web:
            run_async(bot.send_message(chat_id=ctx["chat_id"], text="❌ Something went wrong during login. Please try /link again."))
        return _fail_redirect_or_page("❌ Token Error", "<p>Could not verify your login. Please try again.</p>", "linked=error"), (302 if is_web else 500)

    logto_sub   = claims.get("sub","")
    logto_email = claims.get("email","")
    logto_name  = claims.get("name","") or claims.get("username","")

    if not logto_sub:
        return _fail_redirect_or_page("❌ No User ID", "<p>Logto did not return a user ID.</p>", "linked=error"), (302 if is_web else 500)

    eid_record = eid_get_by_logto(logto_sub)
    if eid_record:
        empire_id = eid_record.get("empire_id")
    else:
        ok, result = eid_create(logto_id=logto_sub, username=logto_name, email=logto_email, source="web_app" if is_web else "telegram_bot")
        if ok:
            empire_id = result
        else:
            logger.error("Empire ID creation failed: %s", result)
            if not is_web:
                run_async(bot.send_message(chat_id=ctx["chat_id"], text="⚠️ Login verified but we couldn't set up your Empire ID. Please try /link again."))
            return _fail_redirect_or_page("❌ Empire ID Error", f"<p>{result}</p>", "linked=error"), (302 if is_web else 500)

    target_user_id = ctx["web_user_id"] if is_web else ctx["telegram_user_id"]

    try:
        existing = supabase.table("user_profiles").select("*").eq("user_id", target_user_id).execute()
        if existing.data:
            supabase.table("user_profiles").update({"logto_id":logto_sub,"logto_email":logto_email,"logto_name":logto_name,"empire_id":empire_id,"last_active":datetime.now(timezone.utc).isoformat()}).eq("user_id", target_user_id).execute()
        else:
            supabase.table("user_profiles").insert({"user_id":target_user_id,"username":logto_name or "","logto_id":logto_sub,"logto_email":logto_email,"logto_name":logto_name,"empire_id":empire_id,"topic_counts":{},"total_chats":0,"last_active":datetime.now(timezone.utc).isoformat()}).execute()
        logger.info("✅ Linked %s %s → Logto %s (Empire ID: %s)", "web" if is_web else "Telegram", target_user_id, logto_sub, empire_id)
    except Exception as e:
        logger.error("Supabase link error: %s", e)
        if not is_web:
            run_async(bot.send_message(chat_id=ctx["chat_id"], text="⚠️ Login verified but we couldn't save your link. Please try /link again."))
        return _fail_redirect_or_page("❌ Database Error", "<p>We couldn't save your account link. Please try again.</p>", "linked=error"), (302 if is_web else 500)

    display = logto_name or logto_email or "there"

    if is_web:
        sep = "&" if "?" in ctx["return_url"] else "?"
        return redirect(f"{ctx['return_url']}{sep}linked=success&empire_id={empire_id}")

    run_async(bot.send_message(
        chat_id=ctx["chat_id"],
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
    if not bot or not WEBHOOK_URL:
        return jsonify({"error": "Bot or WEBHOOK_URL not configured"}), 500
    try:
        future = run_async(bot.set_webhook(url=f"{WEBHOOK_URL}/webhook"))
        future.result(timeout=10)
        return jsonify({"status": "Webhook set successfully!"})
    except Exception as e:
        logger.error("Set webhook error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/delete-webhook", methods=["GET"])
def delete_webhook():
    if not bot:
        return jsonify({"error": "Bot not configured"}), 500
    try:
        future = run_async(bot.delete_webhook())
        future.result(timeout=10)
        return jsonify({"status": "Webhook deleted!"})
    except Exception as e:
        logger.error("Delete webhook error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/debug/logto", methods=["GET"])
def debug_logto():
    return jsonify({
        "LOGTO_ENDPOINT":     bool(os.environ.get("LOGTO_ENDPOINT")),
        "LOGTO_CLIENT_ID":    bool(os.environ.get("LOGTO_CLIENT_ID")),
        "LOGTO_CLIENT_SECRET":bool(os.environ.get("LOGTO_CLIENT_SECRET")),
        "WEBHOOK_URL":        bool(os.environ.get("WEBHOOK_URL")),
        "redirect_uri":       get_redirect_uri(),
        "auth_url_sample":    build_logto_auth_url("test_state_123") if os.environ.get("LOGTO_ENDPOINT") else "NOT CONFIGURED",
    })

# ═══════════════════════════════════════════════════════════════════════════════
#  EMPIRE ID LOOKUP API (for chess app and other mini-apps)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/empire-id-by-logto", methods=["POST"])
def get_empire_id_by_logto():
    """Lookup Empire ID by Logto user ID — used by mini-apps for auth."""
    try:
        data = request.get_json(silent=True) or {}
        logto_id = data.get("logto_id")

        if not logto_id:
            return jsonify({"error": "No logto_id provided"}), 400

        user = eid_get_by_logto(logto_id)

        if user:
            return jsonify({
                "empire_id": user.get("empire_id"),
                "username": user.get("username"),
                "email": user.get("email")
            })

        return jsonify({"error": "No Empire ID found for this Logto user"}), 404

    except Exception as e:
        logger.error(f"[EmpireID Lookup] Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ═══════════════════════════════════════════════════════════════════════════════
#  WEB APP DATA APIS — Projects, Conversations (sidebar), Memories
# ═══════════════════════════════════════════════════════════════════════════════

def _require_user_id(data_or_args) -> Optional[str]:
    uid = str(data_or_args.get("user_id") or "").strip()
    return uid or None

# ── Projects ─────────────────────────────────────────────────────────

@app.route("/api/projects", methods=["GET"])
def list_projects():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"projects": []})
    try:
        rows = supabase.table("projects").select("*").eq("user_id", user_id).order("pinned", desc=True).order("updated_at", desc=True).execute()
        projects = rows.data or []
        # attach a live chat count per project
        for p in projects:
            c = supabase.table("conversations").select("id", count="exact").eq("project_id", p["id"]).execute()
            p["chat_count"] = c.count or 0
        return jsonify({"projects": projects})
    except Exception as e:
        logger.error("list_projects error: %s", e)
        return jsonify({"error": "Could not load projects"}), 500

@app.route("/api/projects", methods=["POST"])
def create_project():
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    name = (data.get("name") or "").strip()
    if not user_id or not name:
        return jsonify({"error": "user_id and name are required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        row = {
            "user_id": user_id,
            "name": name,
            "description": (data.get("description") or "").strip(),
            "emoji": data.get("emoji") or "🚀",
            "color": data.get("color") or "rgba(74,144,217,0.12)",
        }
        res = supabase.table("projects").insert(row).execute()
        project = res.data[0] if res.data else row
        project["chat_count"] = 0
        return jsonify({"project": project}), 201
    except Exception as e:
        logger.error("create_project error: %s", e)
        return jsonify({"error": "Could not create project"}), 500

@app.route("/api/projects/<project_id>", methods=["PATCH"])
def update_project(project_id):
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        allowed = {k: v for k, v in data.items() if k in ("name", "description", "emoji", "color", "pinned")}
        if not allowed:
            return jsonify({"error": "Nothing to update"}), 400
        allowed["updated_at"] = datetime.now(timezone.utc).isoformat()
        res = supabase.table("projects").update(allowed).eq("id", project_id).eq("user_id", user_id).execute()
        if not res.data:
            return jsonify({"error": "Project not found"}), 404
        return jsonify({"project": res.data[0]})
    except Exception as e:
        logger.error("update_project error: %s", e)
        return jsonify({"error": "Could not update project"}), 500

@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        supabase.table("projects").delete().eq("id", project_id).eq("user_id", user_id).execute()
        return jsonify({"deleted": True})
    except Exception as e:
        logger.error("delete_project error: %s", e)
        return jsonify({"error": "Could not delete project"}), 500

# ── Conversations (sidebar "Recent" list) ───────────────────────────

def generate_conversation_title(text: str, max_words: int = 6) -> str:
    """Cheap, instant auto-naming — no extra model call. Takes the
    first few words of the opening message, same way most chat apps
    title a new conversation."""
    clean = re.sub(r'\s+', ' ', text or "").strip()
    if not clean:
        return "New chat"
    words = clean.split(" ")[:max_words]
    title = " ".join(words)
    if len(words) == max_words and len(clean.split(" ")) > max_words:
        title += "…"
    return title[:60].strip().capitalize() if title else "New chat"

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"conversations": []})
    try:
        rows = (
            supabase.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("pinned", desc=True)
            .order("updated_at", desc=True)
            .limit(30)
            .execute()
        )
        return jsonify({"conversations": rows.data or []})
    except Exception as e:
        logger.error("list_conversations error: %s", e)
        return jsonify({"error": "Could not load conversations"}), 500

@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        row = {"user_id": user_id, "title": "New chat"}
        if data.get("project_id"):
            row["project_id"] = data["project_id"]
        res = supabase.table("conversations").insert(row).execute()
        return jsonify({"conversation": res.data[0] if res.data else row}), 201
    except Exception as e:
        logger.error("create_conversation error: %s", e)
        return jsonify({"error": "Could not create conversation"}), 500

@app.route("/api/conversations/<conversation_id>", methods=["PATCH"])
def update_conversation(conversation_id):
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        allowed = {k: v for k, v in data.items() if k in ("title", "pinned", "project_id")}
        if not allowed:
            return jsonify({"error": "Nothing to update"}), 400
        allowed["updated_at"] = datetime.now(timezone.utc).isoformat()
        res = supabase.table("conversations").update(allowed).eq("id", conversation_id).eq("user_id", user_id).execute()
        if not res.data:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"conversation": res.data[0]})
    except Exception as e:
        logger.error("update_conversation error: %s", e)
        return jsonify({"error": "Could not update conversation"}), 500

@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        supabase.table("conversations").delete().eq("id", conversation_id).eq("user_id", user_id).execute()
        return jsonify({"deleted": True})
    except Exception as e:
        logger.error("delete_conversation error: %s", e)
        return jsonify({"error": "Could not delete conversation"}), 500

@app.route("/api/conversations/<conversation_id>/messages", methods=["GET"])
def get_conversation_messages(conversation_id):
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"messages": []})
    try:
        rows = (
            supabase.table("chat_memory")
            .select("message,response,created_at")
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .order("created_at")
            .execute()
        )
        messages = []
        for r in rows.data or []:
            messages.append({"role": "user", "text": r["message"]})
            messages.append({"role": "aim", "text": r["response"]})
        return jsonify({"messages": messages})
    except Exception as e:
        logger.error("get_conversation_messages error: %s", e)
        return jsonify({"error": "Could not load conversation"}), 500

# ── Memories (Memory page) ──────────────────────────────────────────

@app.route("/api/memories", methods=["GET"])
def list_memories():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"memories": [], "memory_enabled": True})
    try:
        q = supabase.table("user_memories").select("*").eq("user_id", user_id)
        category = request.args.get("category")
        if category and category != "all":
            q = q.eq("category", category)
        search = request.args.get("search")
        if search:
            q = q.ilike("text", f"%{search}%")
        rows = q.order("created_at", desc=True).execute()

        prof = supabase.table("user_profiles").select("memory_enabled").eq("user_id", user_id).execute()
        memory_enabled = True
        if prof.data and prof.data[0].get("memory_enabled") is not None:
            memory_enabled = bool(prof.data[0]["memory_enabled"])

        return jsonify({"memories": rows.data or [], "memory_enabled": memory_enabled})
    except Exception as e:
        logger.error("list_memories error: %s", e)
        return jsonify({"error": "Could not load memories"}), 500

@app.route("/api/memories", methods=["POST"])
def create_memory():
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    text = (data.get("text") or "").strip()
    if not user_id or not text:
        return jsonify({"error": "user_id and text are required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        row = {"user_id": user_id, "text": text, "category": data.get("category") or "personal"}
        res = supabase.table("user_memories").insert(row).execute()
        return jsonify({"memory": res.data[0] if res.data else row}), 201
    except Exception as e:
        logger.error("create_memory error: %s", e)
        return jsonify({"error": "Could not save memory"}), 500

@app.route("/api/memories/<memory_id>", methods=["DELETE"])
def delete_memory(memory_id):
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        supabase.table("user_memories").delete().eq("id", memory_id).eq("user_id", user_id).execute()
        return jsonify({"deleted": True})
    except Exception as e:
        logger.error("delete_memory error: %s", e)
        return jsonify({"error": "Could not delete memory"}), 500

@app.route("/api/memories", methods=["DELETE"])
def clear_memories():
    """Clears BOTH the discrete memory facts and the raw chat_memory
    log, so 'Clear all memory' actually means AIM forgets everything —
    not just the visible list."""
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        supabase.table("user_memories").delete().eq("user_id", user_id).execute()
        supabase.table("chat_memory").delete().eq("user_id", user_id).execute()
        return jsonify({"cleared": True})
    except Exception as e:
        logger.error("clear_memories error: %s", e)
        return jsonify({"error": "Could not clear memory"}), 500

@app.route("/api/memory-setting", methods=["GET"])
def get_memory_setting():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"memory_enabled": True})
    try:
        res = supabase.table("user_profiles").select("memory_enabled").eq("user_id", user_id).execute()
        enabled = True
        if res.data and res.data[0].get("memory_enabled") is not None:
            enabled = bool(res.data[0]["memory_enabled"])
        return jsonify({"memory_enabled": enabled})
    except Exception as e:
        logger.error("get_memory_setting error: %s", e)
        return jsonify({"error": "Could not load setting"}), 500

@app.route("/api/memory-setting", methods=["PATCH"])
def set_memory_setting():
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if "memory_enabled" not in data:
        return jsonify({"error": "memory_enabled is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        enabled = bool(data["memory_enabled"])
        ex = supabase.table("user_profiles").select("user_id").eq("user_id", user_id).execute()
        if ex.data:
            supabase.table("user_profiles").update({"memory_enabled": enabled}).eq("user_id", user_id).execute()
        else:
            supabase.table("user_profiles").insert({"user_id": user_id, "memory_enabled": enabled}).execute()
        return jsonify({"memory_enabled": enabled})
    except Exception as e:
        logger.error("set_memory_setting error: %s", e)
        return jsonify({"error": "Could not update setting"}), 500

# ── Profile (Settings > Profile tab) ────────────────────────────────

@app.route("/api/profile", methods=["GET"])
def get_profile():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"profile": {}})
    try:
        rows = supabase.table("user_profiles").select(
            "display_name,bio,avatar_url,logto_email,logto_name,empire_id,logto_id"
        ).eq("user_id", user_id).execute()
        row = rows.data[0] if rows.data else {}
        return jsonify({"profile": {
            "display_name": row.get("display_name") or "",
            "bio": row.get("bio") or "",
            "avatar_url": row.get("avatar_url") or "",
            "email": row.get("logto_email") or "",
            "empire_id": row.get("empire_id") or "",
            "linked": bool(row.get("logto_id")),
        }})
    except Exception as e:
        logger.error("get_profile error: %s", e)
        return jsonify({"error": "Could not load profile"}), 500

@app.route("/api/profile", methods=["PATCH"])
def update_profile():
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        allowed = {k: v for k, v in data.items() if k in ("display_name", "bio", "avatar_url")}
        if not allowed:
            return jsonify({"error": "Nothing to update"}), 400
        # Cheap guardrails so one bad request can't blow up storage.
        if "display_name" in allowed:
            allowed["display_name"] = str(allowed["display_name"])[:80]
        if "bio" in allowed:
            allowed["bio"] = str(allowed["bio"])[:500]
        if "avatar_url" in allowed and allowed["avatar_url"] and len(allowed["avatar_url"]) > 1_500_000:
            return jsonify({"error": "Image is too large — please use something smaller"}), 400

        ex = supabase.table("user_profiles").select("user_id").eq("user_id", user_id).execute()
        if ex.data:
            supabase.table("user_profiles").update(allowed).eq("user_id", user_id).execute()
        else:
            supabase.table("user_profiles").insert({"user_id": user_id, **allowed}).execute()
        return jsonify({"ok": True})
    except Exception as e:
        logger.error("update_profile error: %s", e)
        return jsonify({"error": "Could not update profile"}), 500

@app.route("/api/account", methods=["DELETE"])
def delete_account():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not supabase:
        return jsonify({"error": "Database offline"}), 503
    try:
        for table in ("chat_memory", "conversations", "projects", "user_memories", "user_profiles"):
            supabase.table(table).delete().eq("user_id", user_id).execute()
        return jsonify({"deleted": True})
    except Exception as e:
        logger.error("delete_account error: %s", e)
        return jsonify({"error": "Could not delete account"}), 500

# ═══════════════════════════════════════════════════════════════════════════════
#  BILLING (Settings > Billing tab) — Stripe
#
#  Card details are never handled or stored by us — Stripe's Checkout and
#  Billing Portal collect/display/update the actual card. We only store a
#  stripe_customer_id and read back plan/card-summary/renewal info to show
#  on the Billing tab, which is what "truly dynamic" card details means
#  here: real data pulled live from Stripe, not something we invented.
# ═══════════════════════════════════════════════════════════════════════════════

_PLAN_PRICE_IDS = {"basic": STRIPE_PRICE_BASIC, "pro": STRIPE_PRICE_PRO}
_PRICE_ID_TO_PLAN = {v: k for k, v in _PLAN_PRICE_IDS.items() if v}

def _get_or_create_stripe_customer(user_id: str, email: str = "") -> Optional[str]:
    if not supabase:
        return None
    try:
        row = supabase.table("user_profiles").select("stripe_customer_id,logto_email").eq("user_id", user_id).execute()
        existing_id = row.data[0].get("stripe_customer_id") if row.data else None
        if existing_id:
            return existing_id
        customer = stripe.Customer.create(email=email or (row.data[0].get("logto_email") if row.data else "") or None, metadata={"user_id": user_id})
        if row.data:
            supabase.table("user_profiles").update({"stripe_customer_id": customer.id}).eq("user_id", user_id).execute()
        else:
            supabase.table("user_profiles").insert({"user_id": user_id, "stripe_customer_id": customer.id}).execute()
        return customer.id
    except Exception as e:
        logger.error("_get_or_create_stripe_customer error: %s", e)
        return None

@app.route("/api/billing/checkout", methods=["POST"])
def billing_checkout():
    if not STRIPE_SECRET_KEY:
        return jsonify({"error": "Billing is not configured on the server"}), 503
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    plan = (data.get("plan") or "").strip().lower()
    return_url = (data.get("return_url") or "").strip()
    if not user_id or plan not in _PLAN_PRICE_IDS or not return_url:
        return jsonify({"error": "user_id, a valid plan, and return_url are required"}), 400
    price_id = _PLAN_PRICE_IDS[plan]
    if not price_id:
        return jsonify({"error": f"No Stripe price configured for the {plan} plan yet"}), 503
    customer_id = _get_or_create_stripe_customer(user_id)
    if not customer_id:
        return jsonify({"error": "Could not set up billing customer"}), 500
    try:
        sep = "&" if "?" in return_url else "?"
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{return_url}{sep}billing=success",
            cancel_url=f"{return_url}{sep}billing=cancelled",
        )
        return jsonify({"url": session.url})
    except Exception as e:
        logger.error("billing_checkout error: %s", e)
        return jsonify({"error": "Could not start checkout"}), 500

@app.route("/api/billing/portal", methods=["POST"])
def billing_portal():
    if not STRIPE_SECRET_KEY:
        return jsonify({"error": "Billing is not configured on the server"}), 503
    data = request.get_json(silent=True) or {}
    user_id = _require_user_id(data)
    return_url = (data.get("return_url") or "").strip()
    if not user_id or not return_url:
        return jsonify({"error": "user_id and return_url are required"}), 400
    customer_id = _get_or_create_stripe_customer(user_id)
    if not customer_id:
        return jsonify({"error": "No billing account found yet — subscribe to a plan first"}), 400
    try:
        session = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
        return jsonify({"url": session.url})
    except Exception as e:
        logger.error("billing_portal error: %s", e)
        return jsonify({"error": "Could not open billing portal"}), 500

@app.route("/api/billing", methods=["GET"])
def get_billing():
    user_id = _require_user_id(request.args)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    result = {
        "plan": "free", "status": "none", "card": None,
        "renews_at": None, "cancel_at_period_end": False,
    }
    if not supabase:
        return jsonify(result)
    try:
        row = supabase.table("user_profiles").select("stripe_customer_id,plan").eq("user_id", user_id).execute()
        customer_id = row.data[0].get("stripe_customer_id") if row.data else None
        result["plan"] = (row.data[0].get("plan") if row.data else None) or "free"
        if not customer_id or not STRIPE_SECRET_KEY:
            return jsonify(result)

        subs = stripe.Subscription.list(customer=customer_id, status="all", limit=1)
        if subs.data:
            sub = subs.data[0]
            result["status"] = sub.status
            result["renews_at"] = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc).isoformat()
            result["cancel_at_period_end"] = bool(sub.cancel_at_period_end)
            price_id = sub["items"]["data"][0]["price"]["id"] if sub["items"]["data"] else None
            if price_id in _PRICE_ID_TO_PLAN:
                result["plan"] = _PRICE_ID_TO_PLAN[price_id]

        customer = stripe.Customer.retrieve(customer_id, expand=["invoice_settings.default_payment_method"])
        pm = customer.get("invoice_settings", {}).get("default_payment_method")
        if pm and pm.get("type") == "card":
            card = pm["card"]
            result["card"] = {"brand": card["brand"], "last4": card["last4"], "exp_month": card["exp_month"], "exp_year": card["exp_year"]}

        return jsonify(result)
    except Exception as e:
        logger.error("get_billing error: %s", e)
        return jsonify(result)

@app.route("/api/billing/webhook", methods=["POST"])
def billing_webhook():
    """Keeps user_profiles.plan in sync with Stripe so the rest of the
    app (model routing, feature limits) can gate on a plain DB field
    instead of calling Stripe on every request."""
    if not STRIPE_WEBHOOK_SECRET:
        return jsonify({"error": "Webhook not configured"}), 503
    payload = request.get_data()
    sig = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.error("Stripe webhook signature error: %s", e)
        return jsonify({"error": "Invalid signature"}), 400

    obj = event["data"]["object"]
    if event["type"] in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        customer_id = obj.get("customer")
        status = obj.get("status")
        price_id = obj["items"]["data"][0]["price"]["id"] if obj.get("items", {}).get("data") else None
        plan = _PRICE_ID_TO_PLAN.get(price_id, "free")
        if event["type"] == "customer.subscription.deleted" or status in ("canceled", "incomplete_expired"):
            plan = "free"
        if supabase and customer_id:
            try:
                supabase.table("user_profiles").update({"plan": plan}).eq("stripe_customer_id", customer_id).execute()
            except Exception as e:
                logger.error("billing_webhook plan update error: %s", e)

    return jsonify({"received": True})

# ═══════════════════════════════════════════════════════════════════════════════
#  WEB CHAT API (for the Next.js web app — same brain as Telegram, different door)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
async def web_chat():
    """Lets the Next.js web app talk to AIM directly over HTTP.

    Mirrors the Telegram message pipeline (profile, session summary,
    recent/older memory, web search, the SEARCH_TRIGGER loop) so the
    web app gets the same brain as Telegram. Telegram-only action tags
    (timers, tasks, Nebulae image/audio/pdf, empire linking, learning
    mode) aren't wired up for web yet — they're stripped out of the
    reply instead of silently leaking into the chat as raw text.
    CODE_FILE is the one exception: it's returned as structured JSON
    so the frontend can offer it as a download.

    Also creates/auto-titles a `conversations` row (for the sidebar)
    and respects the memory_enabled setting from the Memory page.
    """
    try:
        data = request.get_json(silent=True) or {}
        user_text = (data.get("message") or "").strip()
        user_id = str(data.get("user_id") or "").strip()
        conversation_id = (data.get("conversation_id") or "").strip() or None

        # Which mode is active (Attach/Search/DeepThink/Medical/DeepSearch/
        # Sandbox from the Plus-button menu), and — for Sandbox — the
        # document text the frontend should send alongside the message.
        # Unrecognized mode values are treated as "no mode" rather than
        # erroring, so an outdated frontend never breaks the endpoint.
        from modes import MODES as _KNOWN_MODES
        mode = (data.get("mode") or "").strip().lower() or None
        if mode not in _KNOWN_MODES:
            mode = None
        document_content = (data.get("document_content") or data.get("document_text") or "").strip()

        if not user_text:
            return jsonify({"error": "message is required"}), 400
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # Sandbox mode has nothing to answer from without a document — ask
        # for one instead of silently calling the AI with empty context.
        if mode == "sandbox" and not document_content:
            return jsonify({
                "reply": "📄 Sandbox Mode needs a document to work from — attach one and I'll answer strictly from its content.",
                "conversation_id": conversation_id,
            })

        memory_enabled = True
        if supabase:
            try:
                prof_row = supabase.table("user_profiles").select("memory_enabled").eq("user_id", user_id).execute()
                if prof_row.data and prof_row.data[0].get("memory_enabled") is not None:
                    memory_enabled = bool(prof_row.data[0]["memory_enabled"])
            except Exception as e:
                logger.error("memory_enabled read error: %s", e)

        # Ensure a conversation row exists for the sidebar, before we
        # need its id below.
        is_new_conversation = False
        if supabase:
            try:
                if conversation_id:
                    existing = supabase.table("conversations").select("id,title").eq("id", conversation_id).eq("user_id", user_id).execute()
                    if not existing.data:
                        conversation_id = None
                if not conversation_id:
                    created = supabase.table("conversations").insert({"user_id": user_id, "title": "New chat"}).execute()
                    conversation_id = created.data[0]["id"] if created.data else None
                    is_new_conversation = True
            except Exception as e:
                logger.error("conversation ensure error: %s", e)

        profile = await get_user_profile_data(user_id)
        session_summary = await get_session_summary(user_id) if memory_enabled else ""
        recent_history, older_context, gap_seconds = (
            await get_conversation_context(user_id, user_text, plan=(profile.get("plan") or "free")) if memory_enabled else ("", "", 0.0)
        )

        web_context = ""
        if mode == "sandbox":
            # Strictly document-only — never touch the web, per modes.py's
            # disables_web_search(). The document becomes the entire
            # "web_context" the AI is given.
            web_context = f"--- PROVIDED DOCUMENT (Sandbox Mode — answer ONLY from this) ---\n{document_content[:12000]}\n--- END DOCUMENT ---"
        elif mode == "deepsearch":
            # User explicitly asked for a deeper look — always fetch and
            # visit pages, not just when is_search_query() would trigger.
            sr = aim_deepsearch(user_text)
            if sr and "found no results" not in sr:
                web_context = sr
        elif is_search_query(user_text) or forces_search(mode):
            tl = user_text.lower()
            if "news" in tl or "latest" in tl or "today" in tl:
                sr = get_latest_news(user_text)
            elif any(s in tl for s in ["football", "match", "score", "team", "player", "league", "f1", "nba", "tennis", "boxing", "ufc", "cricket", "rugby"]):
                sr = get_sports_data(user_text)
            else:
                sr = search_web(user_text)
            if sr and "No search results" not in sr:
                web_context = f"Web Search Results for '{user_text}':\n{sr}"

        max_iter, iteration, final_answer, tool_status, code_file = 4, 0, None, "", None
        while iteration < max_iter:
            iteration += 1
            answer = await get_ai_response(
                user_text, user_id, "web", profile, session_summary,
                recent_history, older_context, web_context, tool_status, gap_seconds,
                mode=mode
            )
            if not answer:
                final_answer = "🔥 High demand right now — please try again."
                break
            answer = answer.strip()

            if "SEARCH_TRIGGER:" in answer:
                if disables_web_search(mode):
                    # Sandbox mode: never actually search — strip the tag
                    # and use whatever text remains as the final answer.
                    answer = re.sub(r'SEARCH_TRIGGER:\s*.+', '', answer, flags=re.IGNORECASE).strip()
                    final_answer = answer or "I can't find that in the document you provided."
                    break
                m = re.search(r'SEARCH_TRIGGER:\s*(.+)', answer, re.IGNORECASE)
                if m:
                    sq = m.group(1).strip()
                    sq_lower = sq.lower()
                    if any(w in sq_lower for w in ["match", "kickoff", "kick off", "fixture", "vs", "versus"]):
                        sr = get_fixture_datetime(sq) or get_sports_data(sq)
                    elif "news" in sq_lower or "latest" in sq_lower:
                        sr = get_latest_news(sq)
                    elif any(s in sq_lower for s in ["football", "score", "f1", "nba", "tennis", "boxing"]):
                        sr = get_sports_data(sq)
                    else:
                        sr = search_web(sq)
                    web_context += f"\n\nWeb Search Results for '{sq}':\n{sr}" if sr and "No search results" not in sr else f"\n\nSearch for '{sq}': No results."
                    continue
                else:
                    final_answer = answer
                    break

            # Strip Telegram-only action tags — not wired up for web yet.
            answer = re.sub(r'\[OPEN_LEARNING:\s*\w+\]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\[EMPIRE_LINK\]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\[CREATE_TASK:\{.*?\}\]', '', answer, flags=re.IGNORECASE | re.DOTALL)
            answer = re.sub(r'\[TIMER:\d+[smh]\]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\[STOPWATCH:(START|STOP)\]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\[NEBULAE_IMAGE:.*?\]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\[NEBULAE_AUDIO:.*?\]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\[NEBULAE_PDF:.*?\]', '', answer, flags=re.IGNORECASE | re.DOTALL)

            code_match = re.search(r'\[CODE_FILE:(\w+)\|(.*?)\]\[/CODE_FILE\]', answer, re.IGNORECASE | re.DOTALL)
            if code_match:
                code_file = {"extension": code_match.group(1).strip().lower(), "content": code_match.group(2).strip()}
                answer = re.sub(r'\[CODE_FILE:\w+\|.*?\]\[/CODE_FILE\]', '', answer, flags=re.IGNORECASE | re.DOTALL)

            final_answer = answer.strip()
            break

        if not final_answer:
            final_answer = "Sorry, something went wrong on my end — mind trying that again?"

        topic = await extract_topic(user_text, final_answer)
        if memory_enabled:
            await save_chat_memory(user_id, "", user_text, final_answer, "web", topic, conversation_id=conversation_id)
            await update_user_profile(user_id, "", topic)
            await update_session_summary(
                user_id,
                [{"message": user_text, "response": final_answer}],
                session_summary,
            )

        conversation_title = None
        if supabase and conversation_id:
            try:
                update_fields = {"updated_at": datetime.now(timezone.utc).isoformat()}
                if is_new_conversation:
                    conversation_title = generate_conversation_title(user_text)
                    update_fields["title"] = conversation_title
                supabase.table("conversations").update(update_fields).eq("id", conversation_id).execute()
            except Exception as e:
                logger.error("conversation title/touch error: %s", e)

        resp = {"reply": final_answer, "conversation_id": conversation_id}
        if conversation_title:
            resp["conversation_title"] = conversation_title
        if code_file:
            resp["code_file"] = code_file
        return jsonify(resp)

    except Exception as e:
        logger.error("Web chat error: %s", e)
        return jsonify({"error": "Something went wrong. Please try again."}), 500

@app.route("/privacy", methods=["GET"])
def privacy_policy():
    return "<h1>Privacy Policy</h1><p>Coming soon.</p>", 200, {"Content-Type":"text/html"}

# Register chess API routes
register_chess_routes(app, supabase)
register_language_routes(app, supabase, gemini_client)
register_last_activity_route(app, supabase)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)