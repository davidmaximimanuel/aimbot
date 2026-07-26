"""
AIM Language Engine — Type 1 (alphabet-first) language template.
AIM classifies the language, then fills in a JSON schema per-unit
on demand, rather than us hardcoding per-language content.

Caching note: the *static* unit content (which letters exist, their
sounds, example words) is the same fact for every learner of a given
language — so it's generated once per (language, unit_index) and
reused from language_unit_cache. Per-user session_state (progress,
mistakes, chat replies) is never cached — that stays personal and is
always generated fresh, per user, per request.
"""

import json
import logging
from datetime import datetime, timezone

from flask import request, jsonify
from google.genai import types

logger = logging.getLogger("language_engine")

GEMINI_MODEL = "gemini-2.5-flash-lite"  # matches the model used elsewhere in aimbot.py


def register_language_routes(flask_app, supabase_client, gemini_client):

    @flask_app.route("/api/language/start", methods=["POST"])
    def language_start():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        empire_id = data.get("empire_id", "unknown")
        language = data.get("language", "").strip()
        session_length_minutes = data.get("session_length_minutes", 15)

        if not language:
            return jsonify({"error": "language is required"}), 400

        existing = _get_active_session(supabase_client, user_id, language)
        if existing:
            return jsonify({
                "resumed": True,
                "static_content": existing["static_content"],
                "session_state": existing["session_state"],
            })

        unit_zero = _generate_unit(gemini_client, supabase_client, language, unit_index=0, known_symbols=[])
        if unit_zero is None:
            return jsonify({"error": "Failed to generate lesson content"}), 500

        static_content = {
            "language": language,
            "template_type": "type1_alphabet",
            "has_positional_forms": unit_zero.get("has_positional_forms", False),
            "session_length_minutes": session_length_minutes,
            "units": [unit_zero["unit"]],
        }
        session_state = {
            "current_unit_index": 0,
            "units_mastered": [],
            "last_mistake": None,
            "resume_message": f"Let's start learning {language}! We'll begin with the basics.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        _save_session(supabase_client, user_id, empire_id, language, static_content, session_state)

        return jsonify({
            "resumed": False,
            "static_content": static_content,
            "session_state": session_state,
        })

    @flask_app.route("/api/language/next-unit", methods=["POST"])
    def language_next_unit():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        empire_id = data.get("empire_id", "unknown")
        language = data.get("language", "").strip()

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session — call /start first"}), 404

        static_content = session["static_content"]
        session_state = session["session_state"]

        known_symbols = []
        for u in static_content["units"]:
            known_symbols.extend([s["symbol"] for s in u.get("introduces", [])])

        next_index = len(static_content["units"])
        new_unit = _generate_unit(gemini_client, supabase_client, language, unit_index=next_index, known_symbols=known_symbols)
        if new_unit is None:
            return jsonify({"error": "Failed to generate next unit"}), 500

        static_content["units"].append(new_unit["unit"])
        session_state["current_unit_index"] = next_index
        session_state["resume_message"] = f"Now let's move on to: {new_unit['unit']['title']}"
        session_state["updated_at"] = datetime.now(timezone.utc).isoformat()

        _save_session(supabase_client, user_id, empire_id, language, static_content, session_state)

        return jsonify({"static_content": static_content, "session_state": session_state})

    @flask_app.route("/api/language/answer", methods=["POST"])
    def language_answer():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        empire_id = data.get("empire_id", "unknown")
        language = data.get("language", "").strip()
        user_answer = data.get("answer", "")
        expected_context = data.get("context", "")

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session"}), 404

        verdict = _check_answer(gemini_client, language, expected_context, user_answer)

        session_state = session["session_state"]
        if not verdict.get("correct", False):
            session_state["last_mistake"] = verdict.get("feedback", "")
        session_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_session(supabase_client, user_id, empire_id, language,
                      session["static_content"], session_state)

        return jsonify(verdict)

    @flask_app.route("/api/language/chat", methods=["POST"])
    def language_chat():
        data = request.get_json() or {}
        language = data.get("language", "")
        user_message = data.get("message", "")
        unit_title = data.get("unit_title", "")

        reply = _chat_reply(gemini_client, language, unit_title, user_message)
        return jsonify({"response": reply})


# ── Internal helpers ────────────────────────────────────────────

def _get_active_session(supabase, user_id, language):
    if supabase is None:
        return None
    try:
        resp = (supabase.table("language_sessions")
                .select("*")
                .eq("user_id", user_id)
                .eq("language", language)
                .eq("status", "active")
                .single()
                .execute())
        return resp.data
    except Exception:
        return None


def _save_session(supabase, user_id, empire_id, language, static_content, session_state):
    if supabase is None:
        return
    try:
        supabase.table("language_sessions").upsert({
            "user_id": str(user_id),
            "empire_id": str(empire_id),
            "language": language,
            "static_content": static_content,
            "session_state": session_state,
            "last_active_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id,language,status").execute()
    except Exception as e:
        logger.error("Language session save error: %s", e)


def _get_cached_unit(supabase, language, unit_index):
    if supabase is None:
        return None
    try:
        resp = (supabase.table("language_unit_cache")
                .select("*")
                .eq("language", language)
                .eq("unit_index", unit_index)
                .single()
                .execute())
        return resp.data
    except Exception:
        return None


def _save_cached_unit(supabase, language, unit_index, has_positional_forms, unit_data):
    if supabase is None:
        return
    try:
        supabase.table("language_unit_cache").upsert({
            "language": language,
            "unit_index": unit_index,
            "has_positional_forms": has_positional_forms,
            "unit_data": unit_data,
        }, on_conflict="language,unit_index").execute()
    except Exception as e:
        logger.error("Unit cache save error: %s", e)


UNIT_SCHEMA_PROMPT = """You are designing one unit of a language-learning lesson for a beginner learning {language}, using an alphabet-first approach.

This is unit index {unit_index}. The learner already knows these symbols: {known_symbols}.

Return ONLY valid JSON (no markdown, no commentary) matching exactly this shape:
{{
  "has_positional_forms": boolean,
  "unit": {{
    "unit_index": {unit_index},
    "title": "short title",
    "introduces": [
      {{"symbol": "...", "sound": "...", "example_word": "...", "example_translation": "..."}}
    ],
    "assumes_known": [list of symbol strings already known],
    "practice_words": [
      {{"word": "...", "translation": "...", "uses_symbols": ["..."]}}
    ]
  }}
}}

Introduce 3-5 new symbols max for this unit. Only use symbols already known or being introduced in this unit for practice_words. If the language has positional letterforms (like Arabic), set has_positional_forms to true and mention the forms in example_word naturally."""


def _generate_unit(gemini_client, supabase_client, language, unit_index, known_symbols):
    cached = _get_cached_unit(supabase_client, language, unit_index)
    if cached:
        return {
            "has_positional_forms": cached.get("has_positional_forms", False),
            "unit": cached.get("unit_data"),
        }

    if gemini_client is None:
        logger.error("Gemini client not configured (missing GEMINI_API_KEY)")
        return None

    prompt = UNIT_SCHEMA_PROMPT.format(
        language=language, unit_index=unit_index,
        known_symbols=", ".join(known_symbols) if known_symbols else "none yet",
    )
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        result = json.loads(resp.text)
    except Exception as e:
        logger.error("Unit generation error: %s", e)
        return None

    _save_cached_unit(
        supabase_client, language, unit_index,
        result.get("has_positional_forms", False),
        result.get("unit"),
    )
    return result


def _check_answer(gemini_client, language, context, user_answer):
    if gemini_client is None:
        return {"correct": False, "feedback": "Checking is unavailable right now."}
    prompt = (f"A learner of {language} was asked about: {context}. "
              f"They answered: \"{user_answer}\". "
              f"Return ONLY JSON: {{\"correct\": boolean, \"feedback\": \"one short sentence, "
              f"encouraging, in plain English, correcting if wrong\"}}")
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(resp.text)
    except Exception as e:
        logger.error("Answer check error: %s", e)
        return {"correct": False, "feedback": "I couldn't check that just now — let's continue."}


def _chat_reply(gemini_client, language, unit_title, user_message):
    if gemini_client is None:
        return "I'm here to help, but my brain's offline right now — try again in a moment."
    prompt = (f"You are AIM, teaching {language}, currently on unit \"{unit_title}\". "
              f"The learner said: \"{user_message}\". Reply helpfully in 1-3 short sentences, "
              f"mixing in a word or two of {language} they've likely learned if natural, "
              f"but explain in plain English.")
    try:
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text
    except Exception as e:
        logger.error("Chat reply error: %s", e)
        return "Let's keep going — what would you like to know?"