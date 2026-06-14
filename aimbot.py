"""
AIM Bot v7.3 — African Intelligence Model (Agentic AI + Deep Research)

CHANGES FROM v7.2.1:

  FIX 1 — REPETITIVE DATE/TIME GREETING:
  - Root cause: datetime_info was injected into BASE_SYSTEM_PROMPT at the top of
    every single prompt, so Gemini would parrot it back as a greeting every time.
  - Fix: datetime is NO LONGER in BASE_SYSTEM_PROMPT at all.
  - Instead, a TIME & CONTEXT block is built dynamically and injected only when
    relevant — and it explicitly tells Gemini not to announce the time unless asked.

  FIX 2 — LAST 10 MESSAGES WITH PRIORITY:
  - Was only passing 5 recent messages.
  - Now passes the last 10 exchanges in chronological order with full timestamps
    (WAT timezone), injected in a clearly-marked PRIORITY block at the very top
    of the prompt so Gemini reads it first.
  - Each message shows: timestamp, user message, AIM response.
  - If user asks "when was my last message?" AIM can answer from this block.

  FIX 3 — TIME GAP AWARENESS:
  - Time gap between now and last message is calculated in Python (reliable),
    not left to Gemini to figure out.
  - Gap is passed as a natural-language string to Gemini with clear instructions:
    * < 30 min: no acknowledgment needed
    * 30 min – 3 hr: light acknowledgment optional
    * 3 hr – 24 hr: "it's been a while" is appropriate
    * > 24 hr: "welcome back / long time" is appropriate
  - Gemini is explicitly told NOT to announce date/time unless the user asks.
"""

import os
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
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
WEBHOOK_URL     = os.environ.get("WEBHOOK_URL", "")
BRAVE_API_KEY   = os.environ.get("BRAVE_API_KEY", "")

TELEGRAM_MAX_CHARS = 4096
WAT = timezone(timedelta(hours=1))   # West Africa Time = UTC+1

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

if BRAVE_API_KEY:
    logger.info("✅ Brave Search API key found — primary: Brave API")
else:
    logger.warning("⚠️ BRAVE_API_KEY not set — fallback: DuckDuckGo Lite")

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
                    user_id  = row["user_id"]
                    duration = row.get("duration_seconds", 0)
                    supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                    mins, secs = divmod(duration, 60)
                    t = ""
                    if mins: t += f"{mins} minute{'s' if mins != 1 else ''}"
                    if secs:
                        if t: t += f" and {secs} second{'s' if secs != 1 else ''}"
                        else: t = f"{secs} second{'s' if secs != 1 else ''}"
                    run_async(bot.send_message(chat_id=int(user_id),
                                               text=f"⏰ Time's up! Your {t.strip()} timer is over."))
                    logger.info("⏰ Timer fired for user %s", user_id)
        except Exception as e:
            logger.error("Timer background check error: %s", e)

threading.Thread(target=check_timers_background, daemon=True, name="timer-worker").start()


# ═══════════════════════════════════════════════════════════
# HELPERS — TIME
# ═══════════════════════════════════════════════════════════

def fmt_wat(dt: datetime) -> str:
    """Format a UTC datetime as a human-readable WAT string."""
    return (dt.astimezone(WAT)).strftime("%a %b %d, %Y · %I:%M %p WAT")


def time_gap_label(seconds: float) -> str:
    """Return a human-readable gap label."""
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif seconds < 86400:
        hrs = int(seconds / 3600)
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def gap_greeting_instruction(seconds: float) -> str:
    """Tell Gemini how to handle the gap in its greeting."""
    if seconds < 1800:          # < 30 min — fresh conversation
        return "NO_GAP_ACK: The user just chatted recently. Do NOT say 'welcome back' or 'long time'. Just respond normally."
    elif seconds < 10800:       # 30 min – 3 hr
        return "LIGHT_ACK: A moderate gap. You may optionally acknowledge it but keep it brief and natural. Do not make it a big deal."
    elif seconds < 86400:       # 3 hr – 24 hr
        return "GAP_ACK: It's been several hours. You can naturally say something like 'It's been a while!' or 'Welcome back!' at the start of your reply, then move on."
    else:                       # > 24 hr
        return "LONG_GAP_ACK: It's been a long time! Start your reply with a warm welcome-back phrase like 'Long time no see!', 'Welcome back!', or 'It's been a minute!' Then dive into helping them."


# ═══════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════

def _search_brave_api(query: str, max_results: int = 5) -> Optional[list]:
    if not BRAVE_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": query, "count": max_results},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("⚠️ Brave API status %s for '%s'", resp.status_code, query)
            return None
        items = resp.json().get("web", {}).get("results", [])
        if not items:
            return None
        results = [{"title": i.get("title",""), "description": i.get("description",""), "url": i.get("url","")}
                   for i in items[:max_results]]
        logger.info("✅ Brave API: %d results for '%s'", len(results), query)
        return results
    except Exception as e:
        logger.error("Brave API error for '%s': %s", query, e)
        return None


def _search_duckduckgo_lite(query: str, max_results: int = 5) -> Optional[list]:
    try:
        resp = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
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
        if not results:
            return None
        logger.info("✅ DuckDuckGo Lite: %d results for '%s'", len(results), query)
        return results
    except Exception as e:
        logger.error("DuckDuckGo Lite error for '%s': %s", query, e)
        return None


def search_web(query: str, max_results: int = 5) -> str:
    results  = _search_brave_api(query, max_results)
    provider = "Brave API"
    if results is None:
        results  = _search_duckduckgo_lite(query, max_results)
        provider = "DuckDuckGo Lite"
    if not results:
        logger.warning("❌ All search providers failed for '%s'", query)
        return "No search results found."
    logger.info("🔍 '%s' → %s (%d results)", query, provider, len(results))
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}")
    return "\n\n".join(lines)


def search_brave(query: str, max_results: int = 5) -> str:
    return search_web(query, max_results)


def deep_research(query: str) -> str:
    angle_queries = [query, f"{query} latest news", f"{query} results details"]
    all_results, seen_urls = [], set()
    for q in angle_queries:
        rs = _search_brave_api(q, 4) or _search_duckduckgo_lite(q, 4)
        if rs:
            for r in rs:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
    if not all_results:
        return "Deep research could not retrieve any results. Try rephrasing the query."
    logger.info("🔬 Deep research: %d unique results for '%s'", len(all_results), query)
    lines = ["=== DEEP RESEARCH RESULTS ===\n"]
    for i, r in enumerate(all_results[:12], 1):
        lines.append(f"{i}. {r['title']}\n   Summary: {r['description']}\n   Source: {r['url']}")
    return "\n\n".join(lines)


def fetch_url_content(url: str) -> str:
    """Only called when the user explicitly pastes a URL — not used on search results."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
        return " ".join(soup.get_text(separator=" ", strip=True).split())[:3000]
    except Exception as e:
        logger.error("URL fetch error for %s: %s", url, e)
        return "Failed to read the link content."


def detect_urls(text: str) -> list:
    return re.findall(r'https?://\S+', text)


def is_search_query_semantic(text: str, threshold: float = 0.45) -> bool:
    try:
        q_emb   = semantic_model.encode([text])
        sims    = np.dot(trigger_embeddings, q_emb.T).flatten()
        max_sim = float(np.max(sims))
        best    = SEARCH_TRIGGER_PHRASES[int(np.argmax(sims))]
        result  = max_sim >= threshold
        logger.info("🔍 Semantic: '%.60s' → %.3f (best: '%s') → %s",
                    text, max_sim, best, "SEARCH" if result else "skip")
        return result
    except Exception as e:
        logger.error("Semantic routing error: %s", e)
        return False


def is_search_query(text: str) -> bool:
    tl = text.lower().strip()
    explicit = ["search for", "google", "look up", "find out", "search the web", "browse", "search"]
    if any(t in tl for t in explicit):
        return True
    return is_search_query_semantic(text)


# ═══════════════════════════════════════════════════════════
# CONTEXT MEMORY  (v7.3 — last 10 messages with timestamps)
# ═══════════════════════════════════════════════════════════

async def get_conversation_context(user_id: str, query_text: str) -> tuple[str, str, float]:
    """
    Returns (recent_history, older_context, gap_seconds).

    recent_history  → last 10 exchanges in chronological order with WAT timestamps
    older_context   → relevance-scored older messages (background)
    gap_seconds     → seconds since the user's last message (for greeting logic)
    """
    if not supabase:
        return "", "", 0.0

    try:
        rows = (supabase.table("chat_memory")
                .select("message, response, topic, created_at")
                .eq("user_id", str(user_id))
                .order("created_at", desc=True)
                .limit(40)
                .execute())

        if not rows.data:
            return "", "", 0.0

        # ── gap seconds (from the most recent saved message)
        gap_seconds = 0.0
        try:
            last_ts    = datetime.fromisoformat(rows.data[0]["created_at"].replace("Z", "+00:00"))
            gap_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()
        except Exception:
            pass

        # ── RECENT: last 10 in chronological order (oldest → newest so it reads naturally)
        recent_rows = list(reversed(rows.data[:10]))
        recent_lines = []
        for row in recent_rows:
            try:
                ts   = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                tstr = fmt_wat(ts)
            except Exception:
                tstr = "unknown time"
            recent_lines.append(f"  [{tstr}]")
            recent_lines.append(f"  User : {row['message']}")
            recent_lines.append(f"  AIM  : {row['response']}")
            recent_lines.append("")
        recent_history = "\n".join(recent_lines).strip()

        # ── OLDER: relevance-scored messages beyond the last 10
        older_rows = rows.data[10:]
        if not older_rows:
            return recent_history, "", gap_seconds

        query_lower = query_text.lower()
        keyword_topics = {
            "space": ["tech"], "nigeria": ["general", "politics"],
            "money": ["finance"], "job": ["career"], "health": ["health"],
            "love": ["relationships"], "sport": ["sports"], "music": ["entertainment"],
            "school": ["education"], "code": ["tech"], "ai": ["tech"],
        }
        matched_topics: set = set()
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
        older_lines = [
            f"[{r.get('topic','general')}] User: {r['message']} | AIM: {r['response']}"
            for _, r in scored[:10]
        ]
        return recent_history, "\n".join(older_lines), gap_seconds

    except Exception as e:
        logger.error("Context retrieval error: %s", e)
        return "", "", 0.0


# ═══════════════════════════════════════════════════════════
# PROMPTING  (v7.3 — no datetime in system prompt)
# ═══════════════════════════════════════════════════════════

# NOTE: {datetime_info} is intentionally GONE from this prompt.
# Datetime context is injected separately and only when needed.
BASE_SYSTEM_PROMPT = """You are AIM — African Intelligence Model. A professional AI assistant built for Africans, by Africans.

PERSONALITY & TONE:
- Warm, respectful, and culturally aware.
- Reference African culture and context when relevant.
- Be helpful, patient, and empowering.
- Use standard English only.
- NEVER use Nigerian Pidgin, slang, or informal dialects unless the user explicitly asks you to.
- NEVER use phrases like "The Empire is rising" or similar roleplay taglines.

RULES:
- Keep responses concise but informative.
- If you don't know something, DO NOT say "I don't know". Use the SEARCH TRIGGER (see below).
- Never make up facts about Africa or Nigeria.
- Use emojis naturally but not excessively.

CAPABILITIES:
- Memory: You recall the last 10 conversations, with exact timestamps.
- Time Tools: Timers and stopwatches.
- Web Search: Real-time web results available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GREETING & TIME RULES — READ CAREFULLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ DO NOT start your reply with "Hello there! It's [date] at [time]..." — EVER.
❌ DO NOT announce the current date and time at the start of messages.
❌ DO NOT say the date/time unless the user explicitly ASKS for it (e.g. "what time is it?").

✅ DO greet naturally based on the TIME GAP CONTEXT provided below.
✅ DO mention time when the user directly asks (e.g. "when did I last message you?").
✅ DO use WAT timestamps from the conversation history to answer time-related questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION CONTINUITY (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The CONVERSATION HISTORY below shows the last 10 exchanges WITH timestamps.
- ALWAYS read it before responding.
- Short follow-ups like "yes", "ok", "tell me more", "go on", "and?" → continue the previous topic.
- If the user uses pronouns (he, she, it, they) → resolve them from the previous message.
- If the user asks "when did I last message you?" → read the most recent timestamp from history.
- If the user asks "how long have we been talking?" → calculate from the first timestamp shown.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TIMERS / STOPWATCHES:
   Append a machine code at the END of your response (after your text):
   - Timer:     [TIMER:Xs]  [TIMER:Xm]  [TIMER:Xh]
   - Stopwatch: [STOPWATCH:START]  [STOPWATCH:STOP]

2. SEARCH TRIGGER — when to use:
   Use for: current events, live scores, prices, weather, recent news, exchange rates,
   elections, or ANYTHING you are not 100% certain about.

   Output EXACTLY:
   SEARCH_TRIGGER: <your search query>

   Examples:
   - "Who won the Champions League?" → SEARCH_TRIGGER: Champions League final 2026 winner
   - "Weather in Lagos?" → SEARCH_TRIGGER: Lagos Nigeria weather today
   - "Naira exchange rate?" → SEARCH_TRIGGER: Naira to Dollar exchange rate today

3. WHEN WEB CONTEXT IS PROVIDED:
   Synthesize the results into a clear answer.
   DO NOT output SEARCH_TRIGGER again — infinite loop.
   DO NOT say you don't have access.

4. IF SEARCH CAME BACK EMPTY:
   Tell the user honestly, offer to help with what you know. No more searches.

5. GENERAL KNOWLEDGE:
   Answer directly for things you're confident about. No SEARCH_TRIGGER needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


async def get_user_profile_data(user_id: str) -> dict:
    if not supabase: return {}
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        return rows.data[0] if rows.data else {}
    except Exception as e:
        logger.error("Profile fetch error: %s", e)
        return {}


def build_enhanced_prompt(
    user_text: str,
    user_id: str,
    profile: dict,
    recent_history: str = "",
    older_context: str = "",
    web_context: str = "",
    tool_status: str = "",
    gap_seconds: float = 0.0,
) -> str:

    now_wat  = datetime.now(WAT)
    now_str  = now_wat.strftime("%A, %B %d, %Y · %I:%M %p WAT")
    gap_lbl  = time_gap_label(gap_seconds)
    greeting_instruction = gap_greeting_instruction(gap_seconds)

    parts = [BASE_SYSTEM_PROMPT]

    # ── user prefs
    pref_language = profile.get("preferred_language", "english")
    topic_counts  = profile.get("topic_counts", {})
    total_chats   = profile.get("total_chats", 0)
    pref_lines = [
        "\n--- USER PREFERENCES ---",
        f"  User ID          : {user_id}",
        f"  Preferred lang   : {pref_language}",
        f"  Timezone         : {profile.get('timezone', 'Africa/Lagos')}",
        f"  Total chats      : {total_chats}",
    ]
    if topic_counts:
        top = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        pref_lines.append(f"  Top interests    : {', '.join(f'{k}({v})' for k, v in top)}")
    if pref_language.lower() == "english":
        pref_lines.append("  LANGUAGE RULE    : Standard English ONLY unless user explicitly asks otherwise.")
    pref_lines.append("--- END PREFERENCES ---\n")
    parts.append("\n".join(pref_lines))

    # ── TIME & CONTEXT BLOCK  (replaces the old {datetime_info} in BASE_SYSTEM_PROMPT)
    time_block = (
        "\n┌─────────────────────────────────────────┐\n"
        "│  TIME & CONTEXT                         │\n"
        "└─────────────────────────────────────────┘\n"
        f"  Current time (WAT) : {now_str}\n"
        f"  User's last message: {gap_lbl}\n"
        f"  Greeting guidance  : {greeting_instruction}\n"
        "\n"
        "  IMPORTANT — DO NOT start your reply by announcing the date/time.\n"
        "  Only mention time if the user explicitly asks about it.\n"
        "────────────────────────────────────────────\n"
    )
    parts.append(time_block)

    # ── PRIORITY BLOCK: last 10 messages — injected prominently
    if recent_history:
        parts.append(
            "\n╔══════════════════════════════════════════════╗\n"
            "║  CONVERSATION HISTORY — LAST 10 MESSAGES    ║\n"
            "║  (READ THIS FIRST before composing reply)   ║\n"
            "║  Timestamps are in WAT. Use them to answer  ║\n"
            "║  any time-related questions from the user.  ║\n"
            "╚══════════════════════════════════════════════╝\n\n"
            + recent_history +
            "\n\n══════════════════════════════════════════════\n"
        )

    # ── web context
    if web_context:
        parts.append(
            "\n--- WEB SEARCH RESULTS (real-time data) ---\n"
            + web_context +
            "\n--- END WEB SEARCH RESULTS ---\n"
        )

    # ── older background memory
    if older_context:
        parts.append(
            "\n--- OLDER RELEVANT MEMORY (background, lower priority) ---\n"
            + older_context +
            "\n--- END OLDER MEMORY ---\n"
        )

    if tool_status:
        parts.append(f"\n--- TOOL STATUS ---\n{tool_status}\n--- END TOOL STATUS ---\n")

    parts.append(f"\nUSER MESSAGE: {user_text}")
    return "\n".join(parts)


async def get_gemini_response(
    user_text: str,
    user_id: str,
    chat_type: str,
    profile: dict = None,
    recent_history: str = "",
    older_context: str = "",
    web_context: str = "",
    tool_status: str = "",
    gap_seconds: float = 0.0,
) -> Optional[types.GenerateContentResponse]:
    if not gemini_client: return None
    try:
        if profile is None:
            profile = await get_user_profile_data(user_id)
        prompt = build_enhanced_prompt(
            user_text, user_id, profile,
            recent_history, older_context, web_context, tool_status, gap_seconds
        )
        return gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
        )
    except Exception as e:
        logger.error("Gemini error: %s", e)
        return None


# ─── TOPIC EXTRACTION ───
async def extract_topic(user_text: str, bot_response: str) -> str:
    if not gemini_client: return "general"
    topics = ["career", "finance", "tech", "sports", "health", "relationships",
              "politics", "entertainment", "education", "general"]
    prompt = (f"Classify this conversation into ONE topic from: {', '.join(topics)}.\n"
              f"User: {user_text[:200]}\nAIM: {bot_response[:200]}\n"
              "Return ONLY the topic word, nothing else.")
    try:
        r = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20),
        )
        t = r.text.strip().lower() if r and r.text else "general"
        return t if t in topics else "general"
    except Exception:
        return "general"


# ─── MEMORY SAVE / UPDATE ───
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
            p  = ex.data[0]
            tc = p.get("topic_counts", {})
            tc[topic] = tc.get(topic, 0) + 1
            supabase.table("user_profiles").update({
                "topic_counts": tc,
                "total_chats":  p.get("total_chats", 0) + 1,
                "last_active":  datetime.now(timezone.utc).isoformat(),
            }).eq("user_id", str(user_id)).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": str(user_id), "username": username or "",
                "topic_counts": {topic: 1}, "total_chats": 1,
            }).execute()
    except Exception as e:
        logger.error("❌ Profile update failed: %s", e)


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
    stop  = {"the","and","about","were","did","have","what","when","that","this","with","for",
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
                     .eq("user_id", str(user_id)).ilike(field, f"%{kw}%")
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
            em   = emoji_map.get(row.get("topic"), "💬")
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
        p  = pr.data[0]
        tc = p.get("topic_counts", {})
        mr = (supabase.table("chat_memory")
              .select("message, response, topic, created_at")
              .eq("user_id", str(user_id)).order("created_at", desc=True).limit(10).execute())
        emoji_map = {"career":"💼","finance":"💰","tech":"💻","sports":"⚽","health":"🏥",
                     "relationships":"❤️","politics":"🏛️","entertainment":"🎬","education":"📚"}
        lines = [
            f"📊 Top Topics : {', '.join(f'{k} ({v}x)' for k,v in sorted(tc.items(), key=lambda x:x[1], reverse=True)[:3])}",
            f"💬 Total Chats: {p.get('total_chats',0)}", "", "📝 Recent conversations:",
        ]
        for i, row in enumerate(mr.data[:5], 1):
            em   = emoji_map.get(row.get("topic"), "💬")
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

<b>Web Search:</b>
/search [query] — Quick web search
/deep [query] — Deep multi-angle research

<b>Time Tools:</b>
/timer [time] — Set a timer (e.g. /timer 5m, /timer 30s, /timer 1h)
/stopwatch — Start or stop stopwatch

<b>Natural Language also works:</b>
- "What's the score?"
- "Set a 30 second timer"
- "Search for who won AFCON"
- "When did I last message you?"
""", reply_to=message_id)
        return True

    elif tl.startswith("/search "):
        query = user_text[8:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔍 Searching...", reply_to=message_id)
        results = search_web(query)
        if results == "No search results found.":
            await send_text_chunks(chat_id, "Couldn't find results. Try rephrasing.", reply_to=message_id)
            return True
        prompt = (f"User asked: {query}\n\nSearch Results:\n{results}\n\n"
                  "Answer using ONLY these results. Be concise. Do NOT output SEARCH_TRIGGER.")
        try:
            r   = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
            )
            txt = r.text.strip() if r and r.text else results
            if "SEARCH_TRIGGER:" in txt: txt = results
            await send_text_chunks(chat_id, txt, reply_to=message_id)
        except Exception as e:
            logger.error("Search command error: %s", e)
            await send_text_chunks(chat_id, results, reply_to=message_id)
        return True

    elif tl.startswith("/deep "):
        query = user_text[6:].strip()
        if not query: return True
        await send_text_chunks(chat_id, "🔬 Researching from multiple angles...", reply_to=message_id)
        deep    = deep_research(query)
        profile = await get_user_profile_data(user_id)
        r = await get_gemini_response(query, user_id, "private", profile, "", "", deep)
        await send_text_chunks(chat_id, r.text.strip() if r and r.text else deep, reply_to=message_id)
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
        res = (supabase.table("user_tools").select("*")
               .eq("user_id", str(user_id)).eq("tool_type", "stopwatch")
               .eq("is_active", True).order("created_at", desc=True).limit(1).execute())
        if res.data:
            row     = res.data[0]
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
        sr = search_web(qtext)
        if "No search results" not in sr:
            web_ctx = f"Web Search Results for '{qtext}':\n{sr}"

    answer_text = None
    try:
        profile = await get_user_profile_data(uid)
        r = await asyncio.wait_for(
            get_gemini_response(qtext, uid, "private", profile, "", "", web_ctx),
            timeout=8.0,
        )
        if r and r.text:
            answer_text = r.text.strip()[:300]
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        logger.error("Inline Gemini error: %s", e)

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
        recent, older, gap = await get_conversation_context(user_id, query_text)
        r = await get_gemini_response(query_text, user_id, "private", profile, recent, older, gap_seconds=gap)
        if r and r.text:
            answer = r.text.strip()
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
        logger.info("🎯 Inline placeholder → '%s'", q)
        return True, q
    return False, ""


# ─── MAIN MESSAGE HANDLER ───
async def handle_message_async(update: Update):
    if not update.message: return

    user       = update.message.from_user
    chat       = update.message.chat
    user_text  = update.message.text or ""
    chat_type  = chat.type if chat else "private"
    message_id = update.message.message_id

    if not user_text:
        await send_text_chunks(chat.id, "I can only read text messages for now.")
        return

    user_id  = str(user.id)
    username = user.username or user.first_name or "User"
    logger.info("📩 [%s/%s] '%s'", user_id, chat_type, user_text[:80])

    # /commands
    if user_text.startswith("/"):
        if await handle_bot_command(user_id, chat.id, message_id, user_text):
            return

    profile = await get_user_profile_data(user_id)

    # inline placeholder
    is_ph, ph_query = is_inline_placeholder(user_text)
    if is_ph and ph_query:
        await process_inline_answer(chat.id, message_id, ph_query, user_id)
        return

    # memory search
    if is_memory_search_query(user_text):
        kws    = extract_search_keywords(user_text)
        result = (await search_memory_by_keyword(user_id, user_text)
                  if kws else await search_memory(user_id))
        await send_text_chunks(chat.id, result, reply_to=message_id)
        return

    # group mention filter
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
        # ── context: last 10 messages + older + gap
        recent_history, older_context, gap_seconds = await get_conversation_context(user_id, user_text)

        # ── web context
        web_context = ""
        for url in detect_urls(user_text):
            c = fetch_url_content(url)
            if c and "Failed" not in c:
                web_context += f"Content from {url}:\n{c}\n"

        if is_search_query(user_text) and not web_context:
            sr = search_web(user_text)
            if "No search results" not in sr:
                web_context = f"Web Search Results for '{user_text}':\n{sr}"

        # ── AGENTIC LOOP (max 3 iterations)
        max_iter    = 3
        iteration   = 0
        final_answer = None
        tool_status  = ""

        while iteration < max_iter:
            iteration += 1
            logger.info("🔄 Agentic iteration %d", iteration)

            r = await get_gemini_response(
                user_text, user_id, chat_type, profile,
                recent_history, older_context, web_context, tool_status, gap_seconds,
            )

            if not r or not r.text:
                final_answer = "🔥 High demand right now — please try again in 30 seconds."
                break

            answer = r.text.strip()

            # SEARCH_TRIGGER
            if "SEARCH_TRIGGER:" in answer:
                m = re.search(r'SEARCH_TRIGGER:\s*(.+)', answer, re.IGNORECASE)
                if m:
                    sq = m.group(1).strip()
                    logger.info("🔍 SEARCH_TRIGGER: '%s'", sq)
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

            # TIMER
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

            # STOPWATCH
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
                        row     = res.data[0]
                        elapsed = (datetime.now(timezone.utc) -
                                   datetime.fromisoformat(row["start_time"].replace("Z","+00:00")))
                        mins, secs = divmod(int(elapsed.total_seconds()), 60)
                        supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                        ts     = f"{mins}m {secs}s" if mins else f"{secs}s"
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

    except Exception as e:
        logger.error("Critical error in message handler: %s", e)
        await send_text_chunks(chat.id, "🛠️ Something went wrong. Try again shortly.", reply_to=message_id)


# ─── ROUTES ───
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "AIM Bot is live!",
        "version": "v7.3",
        "model": "African Intelligence Model (Agentic AI)",
        "search_provider": "Brave API" if BRAVE_API_KEY else "DuckDuckGo Lite",
        "features": [
            "last_10_messages_with_timestamps",
            "time_gap_awareness",
            "no_repetitive_greeting",
            "reliable_search",
            "agentic_loop",
            "deep_research",
            "bot_commands",
            "inline_mode",
        ],
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
            "chat_memory_rows":    getattr(cr, 'count', len(cr.data)),
            "user_profiles_rows":  getattr(pr, 'count', len(pr.data)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/search", methods=["GET"])
def debug_search():
    """Test search: /debug/search?q=Nigeria news"""
    q = request.args.get("q","").strip()
    if not q: return jsonify({"error": "Provide ?q=your+query"}), 400
    try:
        return jsonify({
            "query":    q,
            "provider": "Brave API" if BRAVE_API_KEY else "DuckDuckGo Lite",
            "results":  search_web(q),
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