"""
admin.py — Admin System & Controls
"""
import os
import time
import logging
from typing import Optional
from datetime import datetime, timezone
from telegram.constants import ParseMode

logger = logging.getLogger("aimbot")

ADMIN_IDS = set()

def load_admins(supabase):
    global ADMIN_IDS
    if not supabase:
        logger.warning("⚠️ Supabase not connected, cannot load admins.")
        return
    try:
        res = supabase.table("admins").select("telegram_id").execute()
        if res.data:
            for row in res.data:
                if row.get("telegram_id"):
                    ADMIN_IDS.add(str(row["telegram_id"]))
            logger.info(f"👑 Loaded {len(ADMIN_IDS)} Admins into memory: {ADMIN_IDS}")
        else:
            logger.info("ℹ️ No admins found in database.")
    except Exception as e:
        logger.error(f"❌ Failed to load admins: {e}")

def is_admin(user_id: str) -> bool:
    return str(user_id) in ADMIN_IDS

START_TIME = time.time()

def read_file_safely(filepath: str) -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    requested_path = os.path.abspath(os.path.join(base_dir, filepath))
    if not requested_path.startswith(base_dir):
        return "❌ Access Denied: Path traversal detected."
    if not os.path.exists(requested_path):
        return f"❌ File not found: {filepath}"
    try:
        with open(requested_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"❌ Error reading file: {e}"

def get_server_metrics() -> str:
    import resource
    uptime_seconds = time.time() - START_TIME
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_mb = memory_kb / 1024
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except Exception:
        load_1, load_5, load_15 = "N/A", "N/A", "N/A"
    return (
        f"📊 <b>Server Health Report:</b>\n\n"
        f"⏱️ <b>Uptime:</b> {hours}h {minutes}m {seconds}s\n"
        f"💾 <b>Memory Usage:</b> {memory_mb:.2f} MB\n"
        f"🧠 <b>CPU Load:</b> 1m: {load_1} | 5m: {load_5} | 15m: {load_15}\n"
        f"🐍 <b>Python:</b> {os.sys.version.split()[0]}\n"
        f"📂 <b>Directory:</b> {os.getcwd()}"
    )

async def handle_admin_command(user_id: str, chat_id: int, message_id: int, user_text: str, supabase, get_ai_response, send_text_chunks, USE_DEEPSEEK, ADMIN_IDS, load_admins):
    """Handle all /admin commands"""
    if not is_admin(user_id):
        await send_text_chunks(chat_id, "❌ Access Denied.", reply_to=message_id)
        return True
    
    parts = user_text.lower().strip().split()
    action = parts[1] if len(parts) > 1 else "help"
    
    if action == "stats":
        try:
            users = supabase.table("user_profiles").select("id", count="exact").execute()
            tasks = supabase.table("user_tasks").select("id", count="exact").execute()
            chats = supabase.table("chat_memory").select("id", count="exact").execute()
            linked = supabase.table("user_profiles").select("id", count="exact").not_.is_("logto_id", "null").execute()
            stats_msg = (
                f"📊 <b>Empire AI Stats:</b>\n\n"
                f"👥 Total Users: {users.count}\n"
                f"🔗 Linked (web): {linked.count}\n"
                f"📋 Active Tasks: {tasks.count}\n"
                f"💬 Total Messages: {chats.count}\n"
                f"🤖 AI Model: {'DeepSeek V4' if USE_DEEPSEEK else 'Gemini'}\n"
                f"👑 Admins: {len(ADMIN_IDS)}"
            )
            await send_text_chunks(chat_id, stats_msg, reply_to=message_id)
        except Exception as e:
            await send_text_chunks(chat_id, f"❌ Error: {e}", reply_to=message_id)
    
    elif action == "list":
        admins_list = "\n".join([f"👑 {aid}" for aid in ADMIN_IDS])
        await send_text_chunks(chat_id, f"<b>Current Admins:</b>\n{admins_list}", reply_to=message_id)
    
    elif action == "reload":
        load_admins(supabase)
        await send_text_chunks(chat_id, "🔄 Admin list reloaded.", reply_to=message_id)
    
    elif action == "server":
        metrics = get_server_metrics()
        await send_text_chunks(chat_id, metrics, reply_to=message_id)
    
    elif action.startswith("read "):
        filename = user_text.split(" ", 2)[2] if len(user_text.split(" ", 2)) > 2 else "aimbot.py"
        code_content = read_file_safely(filename)
        formatted_code = f"📄 <b>File: {filename}</b>\n\n```python\n{code_content[:3000]}\n```"
        await send_text_chunks(chat_id, formatted_code, reply_to=message_id)
    
    elif action.startswith("analyze "):
        filename = user_text.split(" ", 2)[2] if len(user_text.split(" ", 2)) > 2 else "aimbot.py"
        code_content = read_file_safely(filename)
        
        if code_content.startswith("❌"):
            await send_text_chunks(chat_id, code_content, reply_to=message_id)
            return True
        
        await send_text_chunks(chat_id, f"🔍 Analyzing <b>{filename}</b>...", reply_to=message_id)
        analysis_prompt = f"You are in Dev Mode. Analyze this Python code for our AIM bot. Identify bugs, suggest 3 improvements, and explain how it fits into Empire AI.\n\nCODE:\n{code_content[:12000]}"
        analysis = await get_ai_response(analysis_prompt, user_id, "private")
        
        if analysis:
            await send_text_chunks(chat_id, f"📊 <b>Analysis of {filename}:</b>\n\n{analysis}", reply_to=message_id)
        else:
            await send_text_chunks(chat_id, "❌ Analysis failed.", reply_to=message_id)
    
    else:
        help_msg = "<b>👑 Admin Commands:</b>\n\n/admin stats\n/admin server\n/admin list\n/admin reload\n/admin read [file]\n/admin analyze [file]"
        await send_text_chunks(chat_id, help_msg, reply_to=message_id)
    
    return True