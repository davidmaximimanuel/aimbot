"""
AIM Language Engine — Type 1 (alphabet-first) language template.

Caches (shared across all learners, never personal):
  language_classifications — decided ONCE per language.
  language_unit_cache — decided ONCE per (language, unit_index).

Personal, never cached:
  session_state (progress, mistakes) and native_language.

Chat is grounded in the current unit's actual on-screen content
(symbols, sounds, practice words) rather than just its title, so
AIM can answer questions about what the learner is actually looking
at without needing real screen access.
"""

import json
import logging
from datetime import datetime, timezone

from flask import request, jsonify
from google.genai import types

logger = logging.getLogger("language_engine")

GEMINI_MODEL = "gemini-2.5-flash-lite"  # matches the model used elsewhere in aimbot.py

VALID_TEMPLATE_TYPES = {"type1_alphabet", "type2_grammar_translation", "type3_speech_conversation"}


def _normalize_language(language):
    """
    'Arabic', 'arabic', 'ARABIC ' all need to hit the same cache row and
    the same session row — otherwise casing differences silently create
    duplicate classifications, duplicate unit caches, and mismatched
    session lookups (a real bug we hit in testing). Every route below
    normalizes the incoming language string through this before using
    it anywhere.
    """
    return language.strip().title()


def register_language_routes(flask_app, supabase_client, gemini_client):

    @flask_app.route("/api/language/start", methods=["POST"])
    def language_start():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        empire_id = data.get("empire_id", "unknown")
        language = _normalize_language(data.get("language", ""))
        native_language = data.get("native_language", "English").strip() or "English"
        session_length_minutes = data.get("session_length_minutes", 15)
        preferred_type = data.get("preferred_type")

        if not language:
            return jsonify({"error": "language is required"}), 400

        existing = _get_active_session(supabase_client, user_id, language)
        if existing:
            return jsonify({
                "resumed": True,
                "static_content": existing["static_content"],
                "session_state": existing["session_state"],
            })

        classification = _classify_language(gemini_client, supabase_client, language)
        chosen_type = preferred_type if preferred_type in VALID_TEMPLATE_TYPES else classification["template_type"]

        if chosen_type != "type1_alphabet":
            return jsonify({
                "error": "unsupported_template_type",
                "message": f"{language} is classified as {chosen_type}, which isn't built yet. "
                           f"You can force alphabet-first with preferred_type='type1_alphabet' if you want.",
                "classification": classification,
            }), 501

        unit_zero = _generate_unit(gemini_client, supabase_client, language, unit_index=0, known_symbols=[])
        if unit_zero is None:
            return jsonify({"error": "Failed to generate lesson content"}), 500

        static_content = {
            "language": language,
            "native_language": native_language,
            "template_type": chosen_type,
            "classification_reasoning": classification.get("reasoning", ""),
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

        _save_session(supabase_client, user_id, empire_id, language, native_language, static_content, session_state)

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
        language = _normalize_language(data.get("language", ""))

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session — call /start first"}), 404

        static_content = session["static_content"]
        session_state = session["session_state"]
        native_language = session.get("native_language", "English")

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

        _save_session(supabase_client, user_id, empire_id, language, native_language, static_content, session_state)

        return jsonify({"static_content": static_content, "session_state": session_state})

    @flask_app.route("/api/language/answer", methods=["POST"])
    def language_answer():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        empire_id = data.get("empire_id", "unknown")
        language = _normalize_language(data.get("language", ""))
        user_answer = data.get("answer", "")
        expected_context = data.get("context", "")

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session"}), 404

        native_language = session.get("native_language", "English")
        verdict = _check_answer(gemini_client, language, native_language, expected_context, user_answer)

        session_state = session["session_state"]
        if not verdict.get("correct", False):
            session_state["last_mistake"] = verdict.get("feedback", "")
        session_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_session(supabase_client, user_id, empire_id, language, native_language,
                      session["static_content"], session_state)

        return jsonify(verdict)

    @flask_app.route("/api/language/chat", methods=["POST"])
    def language_chat():
        data = request.get_json() or {}
        language = _normalize_language(data.get("language", ""))
        native_language = data.get("native_language", "English")
        user_message = data.get("message", "")
        unit_title = data.get("unit_title", "")
        # Full current unit (symbols, sounds, practice words) so AIM is
        # grounded in what's actually on screen, not just guessing from
        # the title.
        unit_content = data.get("unit_content") or {}
        # True when this call fires automatically on unit load, rather
        # than in response to something the learner typed.
        auto_intro = bool(data.get("auto_intro", False))

        reply = _chat_reply(gemini_client, language, native_language, unit_title,
                             unit_content, user_message, auto_intro)
        return jsonify({"response": reply})


# ── Internal helpers ────────────────────────────────────────────

def _get_active_session(supabase, user_id, language):
    if supabase is None:
        return None
    try:
        # .limit(1) instead of .single(): single() throws if it finds
        # zero rows OR more than one, which made this silently fail
        # (returning None -> "No active session") whenever duplicate
        # active rows existed for a user+language, even though a real
        # session was sitting right there. This just takes the most
        # recent one instead of erroring out.
        resp = (supabase.table("language_sessions")
                .select("*")
                .eq("user_id", user_id)
                .eq("language", language)
                .eq("status", "active")
                .order("last_active_at", desc=True)
                .limit(1)
                .execute())
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.error("Active session lookup error (user=%s, language=%s): %s", user_id, language, e)
        return None


def _save_session(supabase, user_id, empire_id, language, native_language, static_content, session_state):
    if supabase is None:
        return
    try:
        supabase.table("language_sessions").upsert({
            "user_id": str(user_id),
            "empire_id": str(empire_id),
            "language": language,
            "native_language": native_language,
            "static_content": static_content,
            "session_state": session_state,
            "last_active_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id,language,status").execute()
    except Exception as e:
        logger.error("Language session save error: %s", e)


def _get_cached_classification(supabase, language):
    if supabase is None:
        return None
    try:
        resp = (supabase.table("language_classifications")
                .select("*")
                .eq("language", language)
                .single()
                .execute())
        return resp.data
    except Exception:
        return None


def _save_classification(supabase, language, template_type, reasoning):
    if supabase is None:
        return
    try:
        supabase.table("language_classifications").upsert({
            "language": language,
            "template_type": template_type,
            "reasoning": reasoning,
        }, on_conflict="language").execute()
    except Exception as e:
        logger.error("Classification cache save error: %s", e)


CLASSIFICATION_PROMPT = """Classify how a complete beginner should best start learning {language}, into exactly one of these three types:

- type1_alphabet: learner needs to learn a new alphabet/script/writing system before anything else makes sense (e.g. Arabic, Korean, Russian, Japanese, Greek, Amharic).
- type2_grammar_translation: learner uses a familiar script but the language has grammar/structure different enough that a grammar-and-translation-first approach works best (e.g. German, French, Latin, for a Latin-script speaker).
- type3_speech_conversation: learner should start with everyday speech, listening, and pronunciation before formal grammar (e.g. learning a closely related language, or a language best absorbed conversationally).

Return ONLY valid JSON, no markdown, no commentary:
{{"template_type": "type1_alphabet" | "type2_grammar_translation" | "type3_speech_conversation", "reasoning": "one short sentence"}}"""


def _classify_language(gemini_client, supabase_client, language):
    cached = _get_cached_classification(supabase_client, language)
    if cached:
        return cached

    if gemini_client is None:
        return {"template_type": "type1_alphabet", "reasoning": "Gemini unavailable — defaulted."}

    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=CLASSIFICATION_PROMPT.format(language=language),
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        result = json.loads(resp.text)
        template_type = result.get("template_type", "type1_alphabet")
        if template_type not in VALID_TEMPLATE_TYPES:
            template_type = "type1_alphabet"
        reasoning = result.get("reasoning", "")
    except Exception as e:
        logger.error("Language classification error: %s", e)
        template_type, reasoning = "type1_alphabet", "Classification failed — defaulted."

    _save_classification(supabase_client, language, template_type, reasoning)
    return {"template_type": template_type, "reasoning": reasoning}


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


def _check_answer(gemini_client, language, native_language, context, user_answer):
    if gemini_client is None:
        return {"correct": False, "feedback": "Checking is unavailable right now."}
    prompt = (f"A native {native_language} speaker learning {language} was asked about: {context}. "
              f"They answered: \"{user_answer}\". "
              f"Return ONLY JSON: {{\"correct\": boolean, \"feedback\": \"one short sentence, "
              f"encouraging, in plain {native_language}, correcting if wrong\"}}")
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


def _chat_reply(gemini_client, language, native_language, unit_title, unit_content, user_message, auto_intro=False):
    if gemini_client is None:
        return "I'm here to help, but my brain's offline right now — try again in a moment."

    on_screen = ""
    introduces = unit_content.get("introduces") if isinstance(unit_content, dict) else None
    if introduces:
        symbol_lines = "; ".join(
            f'{s.get("symbol")} (sound: {s.get("sound")}, example: {s.get("example_word")} = {s.get("example_translation")})'
            for s in introduces
        )
        on_screen += f"\nSymbols currently on screen: {symbol_lines}."
    practice_words = unit_content.get("practice_words") if isinstance(unit_content, dict) else None
    if practice_words:
        word_lines = "; ".join(f'{w.get("word")} ({w.get("translation")})' for w in practice_words)
        on_screen += f"\nPractice words currently on screen: {word_lines}."

    if auto_intro:
        prompt = (f"You are AIM, teaching {language} to a native {native_language} speaker, who has just "
                  f"opened the unit \"{unit_title}\".{on_screen}\n"
                  f"Narrate a short spoken-style introduction to this unit, as if walking them through what's "
                  f"on their screen. Name each symbol and its sound, mention how each is used, and give one or "
                  f"two everyday {native_language} words that share a similar sound where that helps (e.g. "
                  f"comparing to a familiar sound in their own language). Keep it warm and conversational, "
                  f"4-6 sentences, entirely in plain {native_language} except for the {language} symbols/words "
                  f"themselves.")
    else:
        prompt = (f"You are AIM, teaching {language} to a native {native_language} speaker, "
                  f"currently on unit \"{unit_title}\".{on_screen}\n"
                  f"The learner said: \"{user_message}\". Reply helpfully in 1-3 short sentences, "
                  f"referencing the symbols/words above if relevant to their question, "
                  f"mixing in a word or two of {language} they've likely learned if natural, "
                  f"but explain in plain {native_language}.")
    try:
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text
    except Exception as e:
        logger.error("Chat reply error: %s", e)
        return "Let's keep going — what would you like to know?"