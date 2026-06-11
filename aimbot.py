"""
AIM Bot v7.0 — African Intelligence Model (Agentic AI + Deep Research)
Smart Memory + User Preferences + Professional Tone + Time Awareness + Tools + Web Search + Bot Commands
"""

import os
import sys
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

# ─── SEMANTIC ROUTER INITIALIZATION ───
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
    "broadcast information", "streaming options"
]

logger.info("🔢 Computing trigger embeddings...")
trigger_embeddings = semantic_model.encode(SEARCH_TRIGGER_PHRASES)
logger.info("✅ Semantic router ready!")

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
            res = supabase.table("user_tools").select("*").eq("tool_type", "timer").eq("is_active", True).lte("target_time", now_utc).execute()
            
            if res.data:
                for row in res.data:
                    user_id = row["user_id"]
                    duration = row.get("duration_seconds", 0)
                    supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                    
                    mins = duration // 60
                    secs = duration % 60
                    time_str = ""
                    if mins > 0: time_str += f"{mins} minute{'s' if mins != 1 else ''}"
                    if secs > 0:
                        if time_str: time_str += f" and {secs} second{'s' if secs != 1 else ''}"
                        else: time_str = f"{secs} second{'s' if secs != 1 else ''}"
                        
                    msg = f"⏰ Time's up! Your {time_str.strip()} timer is over."
                    run_async(bot.send_message(chat_id=int(user_id), text=msg))
                    logger.info("⏰ Timer fired for user %s", user_id)
        except Exception as e:
            logger.error("Timer background check error: %s", e)

threading.Thread(target=check_timers_background, daemon=True, name="timer-worker").start()

# ─── WEB SEARCH & DEEP RESEARCH ───
def search_brave(query: str, max_results: int = 3) -> str:
    """Search Brave and scrape results."""
    try:
        search_url = f"https://search.brave.com/search?q={requests.utils.quote(query)}&lang=en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        result_divs = soup.find_all('div', class_='result') or soup.find_all('div', {'data-tid': 'result'})
        
        for div in result_divs[:max_results]:
            title_elem = div.find('div', class_='title') or div.find('a')
            desc_elem = div.find('div', class_='description') or div.find('span')
            url_elem = div.find('a', href=True)
            
            if title_elem and desc_elem:
                title = title_elem.get_text(strip=True)
                desc = desc_elem.get_text(strip=True)
                url = url_elem['href'] if url_elem else ""
                results.append(f"- Title: {title}\n  Snippet: {desc}\n  Link: {url}")
        
        if not results:
            return "No search results found."
        
        return "\n\n".join(results)
    except Exception as e:
        logger.error("Brave search error: %s", e)
        return "Web search is currently unavailable."

def deep_research(query: str, max_links: int = 3) -> str:
    """Deep research: search and visit top links for full content."""
    try:
        # First, get search results
        search_url = f"https://search.brave.com/search?q={requests.utils.quote(query)}&lang=en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract URLs
        urls = []
        result_divs = soup.find_all('div', class_='result') or soup.find_all('div', {'data-tid': 'result'})
        for div in result_divs[:max_links]:
            url_elem = div.find('a', href=True)
            if url_elem and url_elem['href'].startswith('http'):
                urls.append(url_elem['href'])
        
        # Visit each URL and extract content
        deep_content = []
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                page_soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Remove scripts and styles
                for script in page_soup(["script", "style"]):
                    script.extract()
                
                # Get main content
                text = page_soup.get_text(separator=' ', strip=True)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                deep_content.append(f"\n--- Content from {url} ---\n{text[:1500]}")
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                continue
        
        if not deep_content:
            return "Deep research failed to retrieve content."
        
        return "\n".join(deep_content)
    except Exception as e:
        logger.error("Deep research error: %s", e)
        return "Deep research is currently unavailable."

def fetch_url_content(url: str) -> str:
    """Fetch and extract main text from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:2000]
    except Exception as e:
        logger.error("URL fetch error: %s", e)
        return "Failed to read the link content."

def detect_urls(text: str) -> list:
    """Extract URLs from text."""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text)

def is_search_query_semantic(text: str, threshold: float = 0.65) -> bool:
    """Use semantic similarity to detect search intent."""
    try:
        query_embedding = semantic_model.encode([text])
        similarities = np.dot(trigger_embeddings, query_embedding.T).flatten()
        max_similarity = np.max(similarities)
        logger.info(f"🔍 Semantic similarity for '{text[:50]}...': {max_similarity:.3f}")
        return max_similarity >= threshold
    except Exception as e:
        logger.error(f"Semantic routing error: {e}")
        return False

def is_search_query(text: str) -> bool:
    """Hybrid approach: Check explicit triggers first, then semantic."""
    text_lower = text.lower().strip()
    
    explicit_triggers = ["search for", "google", "look up", "find out", "search the web", "browse", "search"]
    if any(trigger in text_lower for trigger in explicit_triggers):
        logger.info("🔍 Explicit search trigger detected")
        return True
    
    return is_search_query_semantic(text)

# ─── BASE SYSTEM PROMPT (Updated for Agentic AI) ───
BASE_SYSTEM_PROMPT = """You are AIM — African Intelligence Model. You are a professional AI assistant built for Africans, by Africans.

PERSONALITY & TONE:
- Warm, respectful, and culturally aware.
- Reference African culture and context when relevant.
- Be helpful, patient, and empowering.
- Use standard English ONLY. 
- NEVER use Nigerian Pidgin, slang, or informal dialects unless the user explicitly initiates it and asks you to.
- NEVER use phrases like "The Empire is rising", "Citizen", or similar roleplay taglines. Speak naturally and professionally.

RULES:
- Keep responses concise but informative.
- If you don't know something, DO NOT say "I don't know" or "I don't have access". Instead, USE THE SEARCH TRIGGER (see below).
- Never make up facts about Africa or Nigeria.
- Respect all users regardless of background.
- Use emojis naturally but not excessively.

CAPABILITIES & TOOLS:
- Memory: You can recall and search past conversations by topic or keyword.
- Time Tools: You can set timers (hours, minutes, seconds) and run stopwatches.
- Time Awareness: You know the current time and date in Lagos (WAT).
- Web Search: You have access to real-time web search results via Brave Search and can read links provided by the user.
- If a user asks "What can you do?", list these features clearly and professionally.

TIME AWARENESS:
- Current time and date: {datetime_info}
- Use this time context naturally in your responses (e.g., if it's late night, you may gently suggest rest).

CONVERSATION CONTINUITY:
- Always maintain context of the ongoing conversation.
- If the user is answering questions you asked, recognize it as an answer, not a new question.
- Reference previous messages when relevant to maintain flow.

SPECIAL INSTRUCTIONS (CRITICAL - FOLLOW EXACTLY):

1. TIMER/STOPWATCH EXTRACTION:
   If the user asks you to set a timer or stopwatch, extract the exact duration and append this code at the VERY END of your response:
   [TIMER:Xs] or [TIMER:Xm] or [TIMER:Xh]
   Example: User says "set a 30 second timer" → You respond normally, then add [TIMER:30s] at the end.
   Example: User says "give me 5 questions and set a 15 second timer" → You give 5 questions, then add [TIMER:15s] at the end.
   
   For stopwatch: [STOPWATCH:START] or [STOPWATCH:STOP]

2. WHEN TO USE SEARCH TRIGGER (CRITICAL):
   If a user asks about:
   - Current events (sports scores, news, recent matches, election results, etc.)
   - Real-time information (weather, prices, exchange rates, stock prices, etc.)
   - Anything you are not 100% certain about
   - Anything that happened recently (today, yesterday, this week, this month)
   - Recent scores or results 
   - data beyond your 2022 knowledge cap 
   - ANYTHING YOU ARE NOT SURE OF
   
   DO NOT say "I don't know" or "I don't have access".
   DO NOT guess or make up information.
   
   INSTEAD, respond with EXACTLY this format (nothing else):
   SEARCH_TRIGGER: [your search query]
   
   Examples:
   - User: "Who won the Champions League final?"
     You: SEARCH_TRIGGER: Champions League final 2026 winner
   
   - User: "What's the weather in Lagos?"
     You: SEARCH_TRIGGER: Lagos Nigeria weather today
   
   - User: "Did Super Eagles win last night?"
     You: SEARCH_TRIGGER: Nigeria Super Eagles match result June 10 2026
   
   - User: "What's the exchange rate?"
     You: SEARCH_TRIGGER: Naira to Dollar exchange rate today

3. WHEN YOU RECEIVE WEB SEARCH RESULTS (CRITICAL):
   When you see "WEB CONTEXT" or "Web Search Results" in your input, this means the system has already searched for you.
   
   YOU MUST:
   - Read the search results carefully
   - Synthesize the information into a clear answer
   - Answer the user's question using ONLY the search results
   - Be specific and provide details from the results
   
   YOU MUST NOT:
   - Say "I don't know" or "I don't have access"
   - Output SEARCH_TRIGGER again (this creates an infinite loop)
   - Make up information not in the search results
   
   Example:
   User: "Who won the Champions League?"
   [System searches and provides results]
   You: "Based on the search results, PSG won the 2026 Champions League final 4-3 on penalties against Arsenal on May 30, 2026."

4. GENERAL KNOWLEDGE:
   For historical facts, general knowledge, and information you're confident about, answer directly without using SEARCH_TRIGGER.
   
   Example:
   User: "Who is the president of Nigeria?"
   You: "The current president of Nigeria is Bola Ahmed Tinubu, who took office on May 29, 2023."
"""
# ─── USER PREFERENCE INJECTION ───
async def get_user_profile_data(user_id: str) -> dict:
    if not supabase: return {}
    try:
        rows = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        return rows.data[0] if rows.data else {}
    except Exception as e:
        logger.error("Profile fetch error: %s", e)
        return {}

def build_enhanced_prompt(user_text: str, user_id: str, profile: dict, context: str = "", web_context: str = "", tool_status: str = "") -> str:
    wat_offset = timedelta(hours=1)
    wat_timezone = timezone(wat_offset)
    now_wat = datetime.now(wat_timezone)
    
    date_str = now_wat.strftime("%A, %B %d, %Y")
    time_str = now_wat.strftime("%I:%M %p")
    
    hour = now_wat.hour
    if 5 <= hour < 12: time_of_day = "morning"
    elif 12 <= hour < 17: time_of_day = "afternoon"
    elif 17 <= hour < 21: time_of_day = "evening"
    else: time_of_day = "night"
    
    datetime_info = f"{time_str} WAT, {date_str} ({time_of_day})"
    prompt_parts = [BASE_SYSTEM_PROMPT.format(datetime_info=datetime_info)]

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

    if pref_language.lower() == "english":
        pref_lines.append("- STRICT LANGUAGE RULE: Respond in standard English ONLY. Do NOT use Pidgin, Nigerian slang, or informal dialects unless the user explicitly asks you to.")

    pref_lines.append("--- END PREFERENCES ---\n")
    prompt_parts.append("\n".join(pref_lines))

    if web_context:
        prompt_parts.append(f"\n--- WEB CONTEXT (Real-time data provided by system) ---\n{web_context}\n--- END WEB CONTEXT ---\n")

    if context:
        prompt_parts.append(f"\n--- RELEVANT MEMORY ---\n{context}\n--- END MEMORY ---\n")

    if tool_status:
        prompt_parts.append(f"\n--- TOOL STATUS ---\n{tool_status}\n--- END TOOL STATUS ---\n")

    prompt_parts.append(f"\nUSER QUESTION: {user_text}")
    return "\n".join(prompt_parts)

# ─── GEMINI API (Updated for Agentic Loop) ───
async def get_gemini_response(user_text: str, user_id: str, chat_type: str, profile: dict = None, context: str = "", web_context: str = "", tool_status: str = "") -> Optional[types.GenerateContentResponse]:
    if not gemini_client: return None
    try:
        if profile is None: profile = await get_user_profile_data(user_id)
        prompt = build_enhanced_prompt(user_text, user_id, profile, context, web_context, tool_status)

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024)
        )
        return response
    except Exception as e:
        logger.error("Gemini error: %s", e)
        return None

# ─── TOPIC EXTRACTION ───
async def extract_topic(user_text: str, bot_response: str) -> str:
    if not gemini_client: return "general"
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
    if not supabase: return
    try:
        supabase.table("chat_memory").insert({
            "user_id": str(user_id), "username": username or "",
            "message": message[:2000], "response": response[:2000],
            "chat_type": chat_type, "topic": topic
        }).execute()
    except Exception as e:
        logger.error("❌ Memory save failed: %s", e)

async def update_user_profile(user_id: str, username: str, topic: str):
    if not supabase: return
    try:
        existing = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if existing.data:
            profile = existing.data[0]
            topic_counts = profile.get("topic_counts", {})
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            supabase.table("user_profiles").update({
                "topic_counts": topic_counts, "total_chats": profile.get("total_chats", 0) + 1,
                "last_active": datetime.now(timezone.utc).isoformat()
            }).eq("user_id", str(user_id)).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": str(user_id), "username": username or "",
                "topic_counts": {topic: 1}, "total_chats": 1
            }).execute()
    except Exception as e:
        logger.error("❌ Profile update failed: %s", e)

# ─── IMPROVED CONTEXT RETRIEVAL (Fix for conversation continuity) ───
async def get_relevant_context(user_id: str, query_text: str, limit: int = 15) -> str:
    """Get context with emphasis on recent conversation flow."""
    if not supabase: return ""
    try:
        # Get last 30 chats
        rows = supabase.table("chat_memory").select("message, response, topic, created_at")\
            .eq("user_id", str(user_id)).order("created_at", desc=True).limit(30).execute()
        if not rows.data: return ""

        # ALWAYS include last 5 messages in chronological order (most recent first)
        recent_context = []
        for row in rows.data[:5]:
            recent_context.append(f"[{row.get('topic', 'general')}] User: {row['message']}\nAIM: {row['response']}")
        
        # For older messages, use relevance scoring
        older_rows = rows.data[5:]
        if older_rows:
            query_lower = query_text.lower()
            keyword_topics = {
                "space": ["tech", "science"], "nigeria": ["general", "politics", "entertainment"],
                "money": ["finance"], "job": ["career"], "health": ["health"],
                "love": ["relationships"], "sport": ["sports"], "music": ["entertainment"],
                "school": ["education"], "code": ["tech"], "programming": ["tech"], "ai": ["tech"]
            }
            matched_topics = set()
            for keyword, topics in keyword_topics.items():
                if keyword in query_lower: matched_topics.update(topics)

            scored = []
            for row in older_rows:
                score = 0
                age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))).days
                score += max(0, 30 - age_days)
                if row.get("topic") in matched_topics: score += 50
                msg_resp = f"{row['message']} {row['response']}".lower()
                for word in query_lower.split():
                    if len(word) > 3 and word in msg_resp: score += 10
                scored.append((score, row))

            scored.sort(key=lambda x: x[0], reverse=True)
            older_context = [f"[{r.get('topic', 'general')}] User: {r['message']}\nAIM: {r['response']}" for _, r in scored[:limit-5]]
        else:
            older_context = []

        # Combine: recent first, then relevant older
        all_context = recent_context + older_context
        return "\n\n".join(all_context[:limit])
    except Exception as e:
        logger.error("Relevant context error: %s", e)
        return ""

# ─── SMART MEMORY SEARCH ───
def is_memory_search_query(user_text: str) -> bool:
    if not user_text: return False
    memory_keywords = [
        "what did we talk about", "what have we discussed", "remember our chats",
        "my memory", "our conversations", "what did i ask you", "what were we talking about",
        "we were talking about", "we discussed", "tell me about", "what did we say about",
        "remember when", "do you remember", "what about that time", "didn't we talk about"
    ]
    return any(kw in user_text.lower() for kw in memory_keywords)

def extract_search_keywords(user_text: str) -> list:
    clean = user_text.lower()
    for phrase in ["what did we talk about", "what were we talking about", "we were talking about",
                   "tell me about", "what did we say about", "do you remember", "remember when",
                   "what about", "didn't we talk about", "what have we discussed"]:
        clean = clean.replace(phrase, "")
    clean = re.sub(r'[^\w\s]', ' ', clean)
    words = [w.strip() for w in clean.split() if len(w.strip()) > 2]
    stop_words = {"the", "and", "about", "were", "did", "have", "what", "when", "that", "this", "with", "for", "from", "you", "are", "was", "is", "it", "we", "our", "me", "my", "i", "a", "an", "to", "of", "in", "on", "at", "be", "been", "being", "do", "does", "say", "said", "get", "got", "go", "know", "think", "take", "see", "want", "use", "find", "give", "tell", "ask", "work"}
    return [w for w in words if w not in stop_words]

async def search_memory_by_keyword(user_id: str, query_text: str) -> str:
    if not supabase: return "My memory is currently offline."
    try:
        keywords = extract_search_keywords(query_text)
        if not keywords: return await search_memory(user_id)

        all_results = []
        for keyword in keywords[:3]:
            msg_results = supabase.table("chat_memory").select("*").eq("user_id", str(user_id)).ilike("message", f"%{keyword}%").order("created_at", desc=True).limit(5).execute()
            resp_results = supabase.table("chat_memory").select("*").eq("user_id", str(user_id)).ilike("response", f"%{keyword}%").order("created_at", desc=True).limit(5).execute()
            all_results.extend(msg_results.data + resp_results.data)

        topic_map = {"space": "tech", "nigeria": "general", "money": "finance", "job": "career", "health": "health", "love": "relationships", "sport": "sports", "music": "entertainment", "school": "education", "code": "tech", "ai": "tech"}
        for keyword in keywords:
            if keyword in topic_map:
                topic_results = supabase.table("chat_memory").select("*").eq("user_id", str(user_id)).eq("topic", topic_map[keyword]).order("created_at", desc=True).limit(5).execute()
                all_results.extend(topic_results.data)

        seen_ids = set()
        unique_results = []
        for row in all_results:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                unique_results.append(row)
        unique_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        if not unique_results: return f"I don't recall us discussing {' '.join(keywords)}. Would you like to start a new conversation about it?"

        lines = [f"🔍 I found {len(unique_results)} conversation(s) about that:"]
        for i, row in enumerate(unique_results[:5], 1):
            emoji = {"career": "💼", "finance": "💰", "tech": "💻", "sports": "⚽", "health": "🏥", "relationships": "❤️", "politics": "🏛️", "entertainment": "🎬", "education": "📚"}.get(row.get("topic"), "💬")
            date = row.get("created_at", "")[:10] if row.get("created_at") else ""
            msg_preview = row["message"][:80] if row["message"] else ""
            resp_preview = row["response"][:120] if row["response"] else ""
            lines.append(f'\n{i}. {emoji} [{date}] You: "{msg_preview}..."')
            lines.append(f'   AIM: "{resp_preview}..."')
        return "\n".join(lines)
    except Exception as e:
        logger.error("Keyword memory search error: %s", e)
        return "I found something in my memory, but I'm having trouble organizing it."

async def search_memory(user_id: str) -> str:
    if not supabase: return "My memory is currently offline."
    try:
        profile_res = supabase.table("user_profiles").select("*").eq("user_id", str(user_id)).execute()
        if not profile_res.data: return "We haven't chatted before! Start a conversation so I can remember you."
        profile = profile_res.data[0]
        topic_counts = profile.get("topic_counts", {})
        total_chats = profile.get("total_chats", 0)
        memory_res = supabase.table("chat_memory").select("message, response, topic, created_at").eq("user_id", str(user_id)).order("created_at", desc=True).limit(10).execute()
        
        lines = [f"📊 Your Top Topics: {', '.join(f'{k} ({v}x)' for k, v in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3])}",
                 f"💬 Total Chats: {total_chats}", "", "📝 Recent Conversations:"]
        for i, row in enumerate(memory_res.data[:5], 1):
            emoji = {"career": "💼", "finance": "💰", "tech": "💻", "sports": "⚽", "health": "🏥", "relationships": "❤️", "politics": "🏛️", "entertainment": "🎬", "education": "📚"}.get(row.get("topic"), "💬")
            date = row.get("created_at", "")[:10] if row.get("created_at") else ""
            lines.append(f"{i}. {emoji} [{date}] {row['message'][:60]}...")
        lines.append("\nWant me to dive deeper? Just ask!")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Memory search error: %s", e)
        return "Memory search is having issues right now."

# ─── BOT COMMAND HANDLERS ───
async def handle_bot_command(user_id: str, chat_id: int, message_id: int, user_text: str) -> bool:
    """Handle /commands like /search, /deep, /help, /timer, /stopwatch"""
    text_lower = user_text.lower().strip()
    
    if text_lower.startswith("/help"):
        help_text = """🤖 <b>AIM Bot Commands</b>

<b>General:</b>
/help - Show this help message

<b>Web Search:</b>
/search [query] - Search the web
/deep [query] - Deep research (visits links for full content)

<b>Time Tools:</b>
/timer [time] - Set a timer (e.g., /timer 5m, /timer 30s, /timer 1h)
/stopwatch - Start or stop a stopwatch

<b>Examples:</b>
/search latest news Nigeria
/deep Champions League final 2026
/timer 5m
/stopwatch

<b>Natural Language:</b>
You can also just ask naturally! AIM understands:
- "Set a 30 second timer"
- "Search for who won the match"
- "Give me 5 questions and set a 15 second timer"
"""
        await send_text_chunks(chat_id, help_text, reply_to=message_id)
        return True
    
    elif text_lower.startswith("/search "):
        query = user_text[8:].strip()
        if query:
            await send_text_chunks(chat_id, "🔍 Searching the web...", reply_to=message_id)
            search_results = search_brave(query)
            # Send search results to Gemini for final answer
            profile = await get_user_profile_data(user_id)
            response = await get_gemini_response(query, user_id, "private", profile, "", search_results)
            if response and response.text:
                await send_text_chunks(chat_id, response.text.strip(), reply_to=message_id)
            else:
                await send_text_chunks(chat_id, search_results, reply_to=message_id)
        return True
    
    elif text_lower.startswith("/deep "):
        query = user_text[6:].strip()
        if query:
            await send_text_chunks(chat_id, "🔬 Performing deep research...", reply_to=message_id)
            deep_results = deep_research(query)
            profile = await get_user_profile_data(user_id)
            response = await get_gemini_response(query, user_id, "private", profile, "", deep_results)
            if response and response.text:
                await send_text_chunks(chat_id, response.text.strip(), reply_to=message_id)
            else:
                await send_text_chunks(chat_id, deep_results, reply_to=message_id)
        return True
    
    elif text_lower.startswith("/timer "):
        time_str = user_text[7:].strip()
        # Parse time (e.g., "5m", "30s", "1h")
        match = re.match(r'(\d+)(s|m|h)', time_str.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit == 's': duration_secs = amount
            elif unit == 'm': duration_secs = amount * 60
            elif unit == 'h': duration_secs = amount * 3600
            else: duration_secs = amount * 60
            
            target_time = datetime.now(timezone.utc) + timedelta(seconds=duration_secs)
            supabase.table("user_tools").insert({
                "user_id": str(user_id), "tool_type": "timer",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration_secs, "target_time": target_time.isoformat(),
                "is_active": True
            }).execute()
            
            await send_text_chunks(chat_id, f"⏲️ Timer set for {time_str}!", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "❌ Invalid format. Use: /timer 5m, /timer 30s, /timer 1h", reply_to=message_id)
        return True
    
    elif text_lower == "/stopwatch":
        # Check if there's an active stopwatch
        res = supabase.table("user_tools").select("*").eq("user_id", str(user_id)).eq("tool_type", "stopwatch").eq("is_active", True).order("created_at", desc=True).limit(1).execute()
        if res.data:
            # Stop it
            row = res.data[0]
            start_time = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
            elapsed = datetime.now(timezone.utc) - start_time
            total_secs = int(elapsed.total_seconds())
            mins, secs = divmod(total_secs, 60)
            supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
            time_str = f"{mins} minute{'s' if mins != 1 else ''} and {secs} second{'s' if secs != 1 else ''}" if mins > 0 else f"{secs} second{'s' if secs != 1 else ''}"
            await send_text_chunks(chat_id, f"⏱️ Stopwatch stopped! Time elapsed: {time_str}.", reply_to=message_id)
        else:
            # Start it
            supabase.table("user_tools").insert({
                "user_id": str(user_id), "tool_type": "stopwatch",
                "start_time": datetime.now(timezone.utc).isoformat(), "is_active": True
            }).execute()
            await send_text_chunks(chat_id, "⏱️ Stopwatch started! Use /stopwatch again to stop.", reply_to=message_id)
        return True
    
    return False

# ─── SEND MESSAGE ───
async def send_text_chunks(chat_id: int, text: str, reply_to: Optional[int] = None, message_id: Optional[int] = None):
    if not bot: return
    try:
        if message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS], parse_mode=ParseMode.HTML)
        else:
            kwargs = {"chat_id": chat_id, "text": text[:TELEGRAM_MAX_CHARS], "parse_mode": ParseMode.HTML}
            if reply_to: kwargs["reply_to_message_id"] = reply_to
            await bot.send_message(**kwargs)
    except Exception as e:
        logger.error("Send/edit error: %s", e)
        try:
            if message_id: await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:TELEGRAM_MAX_CHARS])
            else: await bot.send_message(chat_id=chat_id, text=text[:TELEGRAM_MAX_CHARS])
        except Exception as e2:
            logger.error("Fallback send failed: %s", e2)

# ─── INLINE QUERY HANDLER ───
async def handle_inline_query_async(inline_query):
    """Handle inline queries with web search support."""
    query_id = inline_query.id
    query_text = inline_query.query.strip()
    user_id = inline_query.from_user.id if inline_query.from_user else ""

    logger.info("📨 Inline query from %s: '%s'", user_id, query_text)

    if not query_text or len(query_text) < 2:
        await bot.answer_inline_query(inline_query_id=query_id, results=[], cache_time=1)
        return

    answer_text = None
    web_context = ""
    
    urls = detect_urls(query_text)
    if urls:
        logger.info("🔗 Inline URLs detected: %s", urls)
        link_contents = []
        for url in urls:
            content = fetch_url_content(url)
            if content and content != "Failed to read the link content.":
                link_contents.append(f"Content from {url}:\n{content}")
        if link_contents:
            web_context += "\n".join(link_contents)
    
    if is_search_query(query_text) and not web_context:
        logger.info("🔍 Inline search intent detected for: %s", query_text)
        search_results = search_brave(query_text)
        if search_results and "No search results found" not in search_results:
            web_context += f"Web Search Results for '{query_text}':\n{search_results}"
    
    try:
        profile = await get_user_profile_data(user_id)
        response = await asyncio.wait_for(
            get_gemini_response(query_text, user_id, "private", profile, "", web_context),
            timeout=8.0
        )
        if response and response.text:
            answer_text = response.text.strip()[:300]
    except asyncio.TimeoutError:
        logger.info("⏱️ Gemini timeout for inline query, using placeholder")
    except Exception as e:
        logger.error("Inline Gemini error: %s", e)

    if answer_text:
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
        result = InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title=f"Ask AIM: {query_text[:40]}",
            description="Click to get AIM's answer",
            input_message_content=InputTextMessageContent(
                message_text=f"🤖 Asking AIM: {query_text}\n⏳ Processing..."
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

# ─── MESSAGE PROCESSOR (Updated for Agentic AI) ───
async def handle_message_async(update: Update):
    """Process incoming messages with agentic loop."""
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

    # Check for bot commands first
    if user_text.startswith("/"):
        if await handle_bot_command(user_id, chat.id, message_id, user_text):
            return

    profile = await get_user_profile_data(user_id)

    # Check for inline placeholder
    is_placeholder, query_text = is_inline_placeholder(user_text)
    if is_placeholder and query_text:
        logger.info("🔄 PROCESSING INLINE PLACEHOLDER for query: '%s'", query_text)
        await process_inline_answer(chat.id, message_id, query_text, user_id)
        return

    # Check for memory search
    if is_memory_search_query(user_text):
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

    context = await get_relevant_context(user_id, user_text)
    
    web_context = ""
    urls = detect_urls(user_text)
    if urls:
        logger.info("🔗 URLs detected: %s", urls)
        link_contents = []
        for url in urls:
            content = fetch_url_content(url)
            if content and content != "Failed to read the link content.":
                link_contents.append(f"Content from {url}:\n{content}")
        if link_contents:
            web_context += "\n".join(link_contents)

    if is_search_query(user_text) and not web_context:
        logger.info("🔍 Search intent detected for: %s", user_text)
        search_results = search_brave(user_text)
        if search_results and "No search results found" not in search_results:
            web_context += f"Web Search Results for '{user_text}':\n{search_results}"

    # ─── AGENTIC LOOP (Up to 3 iterations) ───
    max_iterations = 3
    iteration = 0
    final_answer = None
    tool_status = ""
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"🔄 Agentic loop iteration {iteration}")
        
        response = await get_gemini_response(user_text, user_id, chat_type, profile, context, web_context, tool_status)
        
        if not response or not response.text:
            final_answer = "🔥 AIM is experiencing high demand right now. Please try again in 30 seconds."
            break
        
        answer = response.text.strip()
        
        # Check for SEARCH_TRIGGER
        if "SEARCH_TRIGGER:" in answer:
            logger.info("🔍 Gemini triggered search")
            # Extract search query
            match = re.search(r'SEARCH_TRIGGER:\s*(.+)', answer, re.IGNORECASE)
            if match:
                search_query = match.group(1).strip()
                logger.info(f"🔍 Searching for: {search_query}")
                search_results = search_brave(search_query)
                web_context += f"\n\nWeb Search Results for '{search_query}':\n{search_results}"
                # Continue loop to get final answer
                continue
            else:
                final_answer = answer
                break
        
        # Check for timer/stopwatch codes
        timer_match = re.search(r'\[TIMER:(\d+)(s|m|h)\]', answer, re.IGNORECASE)
        if timer_match:
            amount = int(timer_match.group(1))
            unit = timer_match.group(2).lower()
            if unit == 's': duration_secs = amount
            elif unit == 'm': duration_secs = amount * 60
            elif unit == 'h': duration_secs = amount * 3600
            else: duration_secs = amount * 60
            
            target_time = datetime.now(timezone.utc) + timedelta(seconds=duration_secs)
            supabase.table("user_tools").insert({
                "user_id": str(user_id), "tool_type": "timer",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration_secs, "target_time": target_time.isoformat(),
                "is_active": True
            }).execute()
            
            # Remove the code from the answer and add status
            answer = re.sub(r'\[TIMER:\d+[smh]\]', '', answer, flags=re.IGNORECASE).strip()
            tool_status = f"✅ Timer set for {amount}{unit}"
            answer += f"\n\n_{tool_status}_"
        
        stopwatch_match = re.search(r'\[STOPWATCH:(START|STOP)\]', answer, re.IGNORECASE)
        if stopwatch_match:
            action = stopwatch_match.group(1).upper()
            if action == "START":
                supabase.table("user_tools").insert({
                    "user_id": str(user_id), "tool_type": "stopwatch",
                    "start_time": datetime.now(timezone.utc).isoformat(), "is_active": True
                }).execute()
                answer = re.sub(r'\[STOPWATCH:START\]', '', answer, flags=re.IGNORECASE).strip()
                answer += "\n\n_⏱️ Stopwatch started!_"
            elif action == "STOP":
                res = supabase.table("user_tools").select("*").eq("user_id", str(user_id)).eq("tool_type", "stopwatch").eq("is_active", True).order("created_at", desc=True).limit(1).execute()
                if res.data:
                    row = res.data[0]
                    start_time = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
                    elapsed = datetime.now(timezone.utc) - start_time
                    total_secs = int(elapsed.total_seconds())
                    mins, secs = divmod(total_secs, 60)
                    supabase.table("user_tools").update({"is_active": False}).eq("id", row["id"]).execute()
                    time_str = f"{mins} minute{'s' if mins != 1 else ''} and {secs} second{'s' if secs != 1 else ''}" if mins > 0 else f"{secs} second{'s' if secs != 1 else ''}"
                    answer = re.sub(r'\[STOPWATCH:STOP\]', '', answer, flags=re.IGNORECASE).strip()
                    answer += f"\n\n_⏱️ Stopwatch stopped! Time elapsed: {time_str}_"
        
        final_answer = answer
        break
    
    if final_answer:
        await send_text_chunks(chat.id, final_answer, reply_to=message_id)
        topic = await extract_topic(user_text, final_answer)
        await save_chat_memory(user_id, username, user_text, final_answer, chat_type, topic)
        await update_user_profile(user_id, username, topic)

# ─── WEBHOOK ROUTES ───
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "AIM Bot is live!",
        "version": "v7.0",
        "model": "African Intelligence Model (Agentic AI)",
        "features": ["smart_memory", "user_preferences", "topic_search", "inline_mode", "tools", "brave_search", "semantic_routing", "agentic_loop", "deep_research", "bot_commands"]
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
            "tables": ["chat_memory", "user_profiles", "auth_states", "user_tools"]
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