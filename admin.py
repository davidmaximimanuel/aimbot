"""
admin.py — Admin System & Controls
"""
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("aimbot")

# Single source of truth for server start time.
# aimbot.py imports this so both files share the same reference.
START_TIME = time.time()

ADMIN_IDS: set[str] = set()


def load_admins(supabase) -> None:
    global ADMIN_IDS
    if not supabase:
        logger.warning("⚠️ Supabase not connected, cannot load admins.")
        return
    try:
        res = supabase.table("admins").select("telegram_id").execute()
        if res.data:
            ADMIN_IDS = {str(row["telegram_id"]) for row in res.data if row.get("telegram_id")}
            logger.info("👑 Loaded %d admin(s): %s", len(ADMIN_IDS), ADMIN_IDS)
        else:
            logger.info("ℹ️ No admins found in database.")
    except Exception as e:
        logger.error("❌ Failed to load admins: %s", e)


def is_admin(user_id: str) -> bool:
    return str(user_id) in ADMIN_IDS


def read_file_safely(filepath: str) -> str:
    """
    Read a file relative to admin.py's directory.
    Rejects path traversal attempts.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    requested_path = os.path.abspath(os.path.join(base_dir, filepath))
    # ✅ Bug 1 fix: log the resolved paths so you can debug hosting-platform
    # issues where __file__ resolves to an unexpected directory.
    logger.info("📂 read_file_safely → base=%s | target=%s", base_dir, requested_path)
    if not requested_path.startswith(base_dir):
        return "❌ Access Denied: Path traversal detected."
    if not os.path.exists(requested_path):
        return f"❌ File not found: {filepath}\n(looked in: {base_dir})"
    try:
        with open(requested_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"❌ Error reading file: {e}"


def get_server_metrics() -> str:
    """
    Returns live server metrics using the START_TIME defined in THIS module.
    Import START_TIME from admin in aimbot.py so they share one reference.
    """
    import resource
    uptime_seconds = time.time() - START_TIME
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    try:
        memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_mb = memory_kb / 1024
    except Exception:
        memory_mb = 0.0
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except Exception:
        load_1, load_5, load_15 = "N/A", "N/A", "N/A"
    return (
        f"📊 <b>Server Health Report:</b>\n\n"
        f"⏱️ <b>Uptime:</b> {hours}h {minutes}m {seconds}s\n"
        f"💾 <b>Memory:</b> {memory_mb:.2f} MB\n"
        f"🧠 <b>CPU Load:</b> 1m: {load_1} | 5m: {load_5} | 15m: {load_15}\n"
        f"🐍 <b>Python:</b> {os.sys.version.split()[0]}\n"
        f"📂 <b>Directory:</b> {os.getcwd()}"
    )


async def handle_admin_command(
    user_id: str,
    chat_id: int,
    message_id: int,
    user_text: str,
    supabase,
    get_ai_response,   # kept for backward compat — no longer used internally
    send_text_chunks,
    USE_DEEPSEEK: bool,
) -> bool:
    """
    Handles all /admin subcommands.
    Called from handle_bot_command in aimbot.py after the /admin check.
    Returns True so handle_bot_command knows the message was handled.

    NOTE: get_ai_response is intentionally no longer called here.
    The analyze command now calls the AI client directly so it gets a clean
    response without the user-chat prompt overhead (history, profile, etc.).
    """
    if not is_admin(user_id):
        await send_text_chunks(chat_id, "❌ Access Denied.", reply_to=message_id)
        return True

    # Parse: "/admin read aimbot.py" -> parts = ["/admin", "read", "aimbot.py"]
    # We work on the ORIGINAL user_text (not lowercased) so filenames stay correct.
    raw_parts  = user_text.strip().split(None, 2)   # max 3 pieces
    tl_parts   = user_text.lower().strip().split(None, 2)
    action     = tl_parts[1] if len(tl_parts) > 1 else "help"
    # filename is the third word if present (preserves original casing)
    filename   = raw_parts[2].strip() if len(raw_parts) > 2 else None

    # ── stats ──────────────────────────────────────────────
    if action == "stats":
        try:
            users  = supabase.table("user_profiles").select("id", count="exact").execute()
            tasks  = supabase.table("user_tasks").select("id", count="exact").execute()
            chats  = supabase.table("chat_memory").select("id", count="exact").execute()
            # linked count — only if logto_id column exists
            try:
                linked = supabase.table("user_profiles").select("id", count="exact").not_.is_("logto_id", "null").execute()
                linked_count = linked.count or 0
            except Exception:
                linked_count = "N/A"
            msg = (
                f"📊 <b>Empire AI Stats:</b>\n\n"
                f"👥 Total Users: {users.count or 0}\n"
                f"🔗 Linked (web): {linked_count}\n"
                f"📋 Active Tasks: {tasks.count or 0}\n"
                f"💬 Total Messages: {chats.count or 0}\n"
                f"🤖 AI Model: {'DeepSeek V4' if USE_DEEPSEEK else 'Gemini'}\n"
                f"👑 Admins: {len(ADMIN_IDS)}"
            )
            await send_text_chunks(chat_id, msg, reply_to=message_id)
        except Exception as e:
            await send_text_chunks(chat_id, f"❌ Error: {e}", reply_to=message_id)

    # ── list ───────────────────────────────────────────────
    elif action == "list":
        admins_list = "\n".join(f"👑 {aid}" for aid in ADMIN_IDS) or "No admins loaded."
        await send_text_chunks(chat_id, f"<b>Current Admins:</b>\n{admins_list}", reply_to=message_id)

    # ── reload ─────────────────────────────────────────────
    elif action == "reload":
        load_admins(supabase)
        await send_text_chunks(chat_id, f"🔄 Admin list reloaded. {len(ADMIN_IDS)} admin(s) active.", reply_to=message_id)

    # ── server ─────────────────────────────────────────────
    elif action == "server":
        await send_text_chunks(chat_id, get_server_metrics(), reply_to=message_id)

    # ── read [file] ────────────────────────────────────────
    # FIX: was `action.startswith("read ")` which is always False because
    # action is just the single word "read" (split into parts[1]).
    elif action == "read":
        target = filename or "aimbot.py"
        content = read_file_safely(target)
        # Telegram HTML parse mode doesn't support <pre><code> well beyond 4096 chars,
        # so we send plain and truncate with a note if needed.
        truncated = len(content) > 3800
        display   = content[:3800]
        header    = f"📄 <b>File: {target}</b>"
        if truncated:
            header += f"\n<i>(truncated — file is longer than 3800 chars)</i>"
        await send_text_chunks(chat_id, f"{header}\n\n<pre>{display}</pre>", reply_to=message_id)

    # ── analyze [file] ─────────────────────────────────────
    elif action == "analyze":
        target = filename or "aimbot.py"

        # ── Step 1: read the file ──────────────────────────
        content = read_file_safely(target)
        if content.startswith("❌"):
            await send_text_chunks(chat_id, content, reply_to=message_id)
            return True

        await send_text_chunks(chat_id, f"🔍 Analyzing <b>{target}</b>...", reply_to=message_id)

        analysis_prompt = (
            f"You are AIM in Dev Mode. Analyze this Python code for the Empire AI bot.\n"
            f"1. Identify any bugs or problems.\n"
            f"2. Suggest 3 concrete improvements.\n"
            f"3. Briefly explain what this file does in the Empire AI architecture.\n\n"
            f"FILE: {target}\n\nCODE:\n{content[:12000]}"
        )

        # ── Step 2: call AI directly (NOT via get_ai_response) ──
        # get_ai_response is built for user chat — it loads conversation
        # history, profiles, builds a heavy prompt, etc. For admin code
        # analysis we just need a clean, direct API call.
        analysis = None
        try:
            if USE_DEEPSEEK:
                import os as _os
                from openai import AsyncOpenAI as _AsyncOpenAI
                _ds = _AsyncOpenAI(
                    api_key=_os.environ.get("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com",
                )
                _r = await _ds.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.4,
                    max_tokens=1500,
                )
                analysis = _r.choices[0].message.content if _r.choices else None
            else:
                import os as _os
                from google import genai as _genai
                from google.genai import types as _types
                _gc = _genai.Client(api_key=_os.environ.get("GEMINI_API_KEY", ""))
                _r  = _gc.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[_types.Content(role="user", parts=[_types.Part(text=analysis_prompt)])],
                    config=_types.GenerateContentConfig(temperature=0.4, max_output_tokens=1500),
                )
                analysis = _r.text if _r and _r.text else None
        except Exception as _e:
            # ✅ Bug 3 fix: show the REAL error instead of a generic message
            logger.error("❌ Admin analyze error: %s", _e)
            await send_text_chunks(chat_id, f"❌ AI call failed: <code>{_e}</code>", reply_to=message_id)
            return True

        if analysis:
            await send_text_chunks(chat_id, f"📊 <b>Analysis: {target}</b>\n\n{analysis}", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "❌ AI returned an empty response. Check your API key env vars.", reply_to=message_id)

    # ── help / unknown ─────────────────────────────────────
    else:
        await send_text_chunks(chat_id, (
            "<b>👑 Admin Commands:</b>\n\n"
            "/admin stats — User & message counts\n"
            "/admin server — Server health & uptime\n"
            "/admin list — List all admin IDs\n"
            "/admin reload — Reload admins from DB\n"
            "/admin read [file] — Read a project file\n"
            "/admin analyze [file] — AI analysis of a file\n\n"
            "<i>Default file for read/analyze: aimbot.py</i>"
        ), reply_to=message_id)

    return True