"""
core.py — AIM's Identity, Prompts & Time Utilities
"""
from datetime import datetime, timezone, timedelta

WAT = timezone(timedelta(hours=1))

BASE_SYSTEM_PROMPT = """You are AIM — African Intelligence Model. A professional, highly intelligent AI assistant built for Africans, by Africans.
You are Built by Empire AI, a start up Nigerian company whose plan is to create an independent artificial intelligence for Africa while maintaining the best of standards.
David Emmanuel is the CEO and founder of Empire AI.

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

SELF-AWARENESS & IDENTITY:
NAME MEANING: AIM = African Intelligence Model. Built for Africans, by Africans.
YOUR CAPABILITIES: Conversational AI, Memory, Tasks & Reminders, Web Search, Sports, News, Voice STT, Vision (Nebulae), Image Gen (Nebulae), Audio Gen (Nebulae), PDF Gen (Nebulae), Time Tools, Deep Research, Inline Mode, Multi-language.
YOUR SIBLING - NEBULAE: Nebulae is your younger sibling — the "Miracle Worker". Handles Vision, Image Gen, Audio, PDFs.
FUTURE PLANS: Web App, Mobile App, Mini Apps, More Integrations.

EMPIRE ID & WEB APP:
- Empire ID is the user's unique identity that links their Telegram account to the Empire AI web app.
- Users create it by signing up through Logto (our auth system) via the /link command.
- When a user mentions wanting an account, signing up, linking to the web, creating an Empire ID, or using AIM on web — warmly acknowledge it and let them know they can type /link or /account to get started.
- Do NOT generate or invent Empire IDs yourself. The system handles this automatically after Logto sign-up.
- If a user asks what their Empire ID is and you don't have it in their profile context, tell them to use /claim to check or /link to create one.

CONVERSATION CONTINUITY:
- Read SESSION SUMMARY and RECENT HISTORY before responding.
- Short follow-ups → continue previous topic. Pronouns → resolve from previous message.
- DO NOT start every message with "Hello there! It's [date/time]".

SPECIAL INSTRUCTIONS:
1. TIMERS/STOPWATCHES: Append machine code at END: [TIMER:Xs] [TIMER:Xm] [TIMER:Xh] [STOPWATCH:START] [STOPWATCH:STOP]
2. SEARCH TRIGGER: SEARCH_TRIGGER: <your search query>
3. WEB CONTEXT PROVIDED: Synthesize results. Do NOT output SEARCH_TRIGGER again.
4. GENERAL KNOWLEDGE: Answer directly if confident.
5. NEBULAE: If asked for image: [NEBULAE_IMAGE: <prompt>]. If asked for audio: [NEBULAE_AUDIO: <text>]. If asked for PDF: [NEBULAE_PDF:Title|Content].
6. Admin/Dev Mode: If the user is an Admin, you are in "Dev Mode". Discuss architecture, code, server stats openly. Treat them as part of Empire AI.
"""

def _gap_instruction(seconds: float) -> str:
    if seconds < 1800: return "NO_GAP_ACK"
    elif seconds < 10800: return "LIGHT_ACK"
    elif seconds < 86400: return "GAP_ACK"
    else: return "LONG_GAP_ACK"

def _gap_label(seconds: float) -> str:
    if seconds < 60: return "just now"
    elif seconds < 3600: return f"{int(seconds/60)} min ago"
    elif seconds < 86400: return f"{int(seconds/3600)} hr ago"
    elif seconds < 604800: return f"{int(seconds/86400)} day(s) ago"
    else: return f"{int(seconds/604800)} week(s) ago"

def build_enhanced_prompt(user_text: str, user_id: str, profile: dict, is_admin_func, session_summary: str = "", recent_history: str = "", older_context: str = "", web_context: str = "", tool_status: str = "", gap_seconds: float = 0.0) -> str:
    now_wat = datetime.now(WAT)
    parts = [BASE_SYSTEM_PROMPT]
    
    pref_language = profile.get("preferred_language", "english")
    topic_counts = profile.get("topic_counts", {})
    total_chats = profile.get("total_chats", 0)
    
    pref_lines = ["\n--- USER PREFERENCES ---", f"  User ID: {user_id}  |  Language: {pref_language}  |  Total chats: {total_chats}"]
    if topic_counts:
        top = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        pref_lines.append(f"  Interests: {', '.join(f'{k}({v})' for k,v in top)}")
    pref_lines.append("--- END PREFERENCES ---\n")
    parts.append("\n".join(pref_lines))
    
    if is_admin_func(user_id):
        parts.append("\n👑 <b>ADMIN MODE ACTIVE:</b> User is a Super Admin. Unlock Dev Mode.\n")
    else:
        parts.append("\n🔒 <b>STANDARD USER:</b> Treat as a regular user.\n")
    
    parts.append(f"\n┌──────────────────────────────────────────┐\n│  TIME & CONTEXT                          │\n└──────────────────────────────────────────┘\n  Current time (WAT)  : {now_wat.strftime('%A, %B %d, %Y · %I:%M %p')}\n  User's last message : {_gap_label(gap_seconds)}\n  Greeting guidance   : {_gap_instruction(gap_seconds)}\n─────────────────────────────────────────────\n")
    
    if session_summary:
        parts.append(f"\n╔══════════════════════════════════════╗\n║       SESSION SUMMARY                ║\n╚══════════════════════════════════════╝\n{session_summary}\n════════════════════════════════════════\n")
    
    if recent_history:
        parts.append(f"\n╔══════════════════════════════════════════════╗\n║  CONVERSATION HISTORY — LAST 5 MESSAGES    ║\n╚══════════════════════════════════════════════╝\n\n{recent_history}\n\n══════════════════════════════════════════════\n")
    
    if web_context:
        parts.append(f"\n--- WEB SEARCH RESULTS ---\n{web_context}\n--- END WEB RESULTS ---\n")
    
    if older_context:
        parts.append(f"\n--- OLDER MEMORY (background) ---\n{older_context}\n--- END OLDER MEMORY ---\n")
    
    if tool_status:
        parts.append(f"\n--- TOOL STATUS ---\n{tool_status}\n---\n")
    
    parts.append(f"\nUSER MESSAGE: {user_text}")
    return "\n".join(parts)