"""
AIM Language Engine — Type 1 (alphabet-first) language template.

Model routing: every AI call tries DeepSeek first, falls back to Gemini
only if DeepSeek fails or is unavailable — mirrors aimbot.py's own
_route_to_model fallback chain pattern (same idea, same "log which link
answered" behavior), just adapted here to also support JSON-mode calls
for structured content generation.

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

GEMINI_MODEL = "gemini-2.5-flash-lite"
DEEPSEEK_MODEL = "deepseek-v4-flash"

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


def _is_quota_error(e):
    """Detects a quota/rate-limit failure specifically, so the final
    honest message (after BOTH providers have failed) can name the
    real cause instead of a generic 'something went wrong'."""
    msg = str(e).lower()
    return any(s in msg for s in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit"))


# ── Shared model-routing helpers: DeepSeek first, Gemini fallback ──

async def _generate_text(deepseek_client, gemini_client, prompt, temperature=0.7, max_tokens=1024):
    """Plain-text generation. Tries DeepSeek, falls back to Gemini.
    Raises the last exception if both fail — callers decide how to
    present that to the user."""
    last_err = None
    if deepseek_client:
        try:
            r = await deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
            )
            text = r.choices[0].message.content if r.choices else None
            if text:
                return text
        except Exception as e:
            logger.warning("Language engine: DeepSeek text call failed (%s), falling back to Gemini", e)
            last_err = e
    if gemini_client:
        try:
            resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            if resp and resp.text:
                return resp.text
        except Exception as e:
            logger.warning("Language engine: Gemini text call also failed (%s)", e)
            last_err = e
    raise last_err or RuntimeError("No AI provider configured")


async def _generate_json(deepseek_client, gemini_client, prompt, temperature=0.7, max_tokens=1024):
    """JSON-mode generation. Tries DeepSeek (OpenAI-compatible json_object
    mode), falls back to Gemini (response_mime_type=application/json).
    Raises the last exception if both fail."""
    last_err = None
    if deepseek_client:
        try:
            r = await deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            text = r.choices[0].message.content if r.choices else None
            if text:
                return json.loads(text)
        except Exception as e:
            logger.warning("Language engine: DeepSeek JSON call failed (%s), falling back to Gemini", e)
            last_err = e
    if gemini_client:
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            if resp and resp.text:
                return json.loads(resp.text)
        except Exception as e:
            logger.warning("Language engine: Gemini JSON call also failed (%s)", e)
            last_err = e
    raise last_err or RuntimeError("No AI provider configured")


def register_language_routes(flask_app, supabase_client, gemini_client, deepseek_client=None):

    @flask_app.route("/api/language/start", methods=["POST"])
    async def language_start():
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

        classification = await _classify_language(deepseek_client, gemini_client, supabase_client, language)
        chosen_type = preferred_type if preferred_type in VALID_TEMPLATE_TYPES else classification["template_type"]

        if chosen_type != "type1_alphabet":
            return jsonify({
                "error": "unsupported_template_type",
                "message": f"{language} is classified as {chosen_type}, which isn't built yet. "
                           f"You can force alphabet-first with preferred_type='type1_alphabet' if you want.",
                "classification": classification,
            }), 501

        unit_zero = await _generate_unit(deepseek_client, gemini_client, supabase_client, language, unit_index=0, known_symbols=[])
        if unit_zero is None:
            return jsonify({"error": "Failed to generate lesson content"}), 500

        static_content = {
            "language": language,
            "native_language": native_language,
            "template_type": chosen_type,
            "classification_reasoning": classification.get("reasoning", ""),
            "has_positional_forms": unit_zero.get("has_positional_forms", False),
            "total_alphabet_size": unit_zero.get("total_alphabet_size") or 30,
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
    async def language_next_unit():
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
            if u.get("phase") == "words_sentences":
                continue
            known_symbols.extend([s["symbol"] for s in u.get("introduces", [])])

        total_alphabet_size = static_content.get("total_alphabet_size", 30)
        next_index = len(static_content["units"])

        if len(known_symbols) >= total_alphabet_size:
            new_unit = await _generate_sentence_unit(deepseek_client, gemini_client, supabase_client, language,
                                                       native_language, unit_index=next_index, known_symbols=known_symbols)
        else:
            new_unit = await _generate_unit(deepseek_client, gemini_client, supabase_client, language,
                                             unit_index=next_index, known_symbols=known_symbols)

        if new_unit is None:
            return jsonify({"error": "Failed to generate next unit"}), 500

        static_content["units"].append(new_unit["unit"])
        session_state["current_unit_index"] = next_index
        session_state["resume_message"] = f"Now let's move on to: {new_unit['unit']['title']}"
        session_state["updated_at"] = datetime.now(timezone.utc).isoformat()

        _save_session(supabase_client, user_id, empire_id, language, native_language, static_content, session_state)

        return jsonify({"static_content": static_content, "session_state": session_state})

    @flask_app.route("/api/language/answer", methods=["POST"])
    async def language_answer():
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
        verdict = await _check_answer(deepseek_client, gemini_client, language, native_language, expected_context, user_answer)

        session_state = session["session_state"]
        if verdict.get("checked", True) and not verdict.get("correct", False):
            session_state["last_mistake"] = verdict.get("feedback", "")
        session_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_session(supabase_client, user_id, empire_id, language, native_language,
                      session["static_content"], session_state)

        return jsonify(verdict)

    @flask_app.route("/api/language/chat", methods=["POST"])
    async def language_chat():
        data = request.get_json() or {}
        language = _normalize_language(data.get("language", ""))
        native_language = data.get("native_language", "English")
        user_message = data.get("message", "")
        unit_title = data.get("unit_title", "")
        unit_content = data.get("unit_content") or {}
        auto_intro = bool(data.get("auto_intro", False))

        reply = await _chat_reply(deepseek_client, gemini_client, language, native_language, unit_title,
                                   unit_content, user_message, auto_intro)
        return jsonify({"response": reply})

    @flask_app.route("/api/language/quiz/start", methods=["POST"])
    async def language_quiz_start():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        language = _normalize_language(data.get("language", ""))
        level = data.get("level", "intermediate")
        if level not in ("beginner", "intermediate"):
            level = "intermediate"

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session"}), 404

        native_language = session.get("native_language", "English")
        units = session["static_content"].get("units", [])
        if not units:
            return jsonify({"error": "Nothing learned yet to quiz on"}), 400

        questions = await _generate_quiz(deepseek_client, gemini_client, language, native_language, units, level)
        if not questions:
            return jsonify({"error": "Failed to generate quiz"}), 500

        session_state = session["session_state"]
        session_state["active_quiz"] = {
            "questions": questions,
            "answered": {},
            "score": 0,
            "total": len(questions),
        }
        session_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_session(supabase_client, user_id, session.get("empire_id", "unknown"), language,
                      native_language, session["static_content"], session_state)

        safe_questions = [
            {k: v for k, v in q.items() if k != "correct_answer"} for q in questions
        ]
        return jsonify({"questions": safe_questions, "total": len(questions)})

    @flask_app.route("/api/language/quiz/answer", methods=["POST"])
    async def language_quiz_answer():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        language = _normalize_language(data.get("language", ""))
        question_id = data.get("question_id", "")
        user_answer = data.get("answer", "")

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session"}), 404

        native_language = session.get("native_language", "English")
        session_state = session["session_state"]
        active_quiz = session_state.get("active_quiz")
        if not active_quiz:
            return jsonify({"error": "No active quiz — call /quiz/start first"}), 404

        question = next((q for q in active_quiz["questions"] if q["id"] == question_id), None)
        if not question:
            return jsonify({"error": "Unknown question_id for this quiz"}), 400

        if question_id in active_quiz["answered"]:
            return jsonify(active_quiz["answered"][question_id])

        verdict = await _grade_quiz_answer(deepseek_client, gemini_client, native_language, question, user_answer)

        if verdict.get("checked", True):
            active_quiz["answered"][question_id] = verdict
            if verdict.get("correct"):
                active_quiz["score"] += 1
            session_state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_session(supabase_client, user_id, session.get("empire_id", "unknown"), language,
                          native_language, session["static_content"], session_state)

        return jsonify({
            **verdict,
            "score": active_quiz["score"],
            "total": active_quiz["total"],
            "answered_count": len(active_quiz["answered"]),
        })

    @flask_app.route("/api/language/quiz/finish", methods=["POST"])
    async def language_quiz_finish():
        data = request.get_json() or {}
        user_id = data.get("user_id", "unknown")
        language = _normalize_language(data.get("language", ""))

        session = _get_active_session(supabase_client, user_id, language)
        if not session:
            return jsonify({"error": "No active session"}), 404

        native_language = session.get("native_language", "English")
        session_state = session["session_state"]
        active_quiz = session_state.pop("active_quiz", None)
        if not active_quiz:
            return jsonify({"error": "No active quiz to finish"}), 404

        history = session_state.setdefault("quiz_history", [])
        result = {
            "score": active_quiz["score"],
            "total": active_quiz["total"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        history.append(result)
        session_state["updated_at"] = result["completed_at"]

        _save_session(supabase_client, user_id, session.get("empire_id", "unknown"), language,
                      native_language, session["static_content"], session_state)

        return jsonify(result)


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


async def _classify_language(deepseek_client, gemini_client, supabase_client, language):
    cached = _get_cached_classification(supabase_client, language)
    if cached:
        return cached

    try:
        result = await _generate_json(deepseek_client, gemini_client, CLASSIFICATION_PROMPT.format(language=language))
        template_type = result.get("template_type", "type1_alphabet")
        if template_type not in VALID_TEMPLATE_TYPES:
            template_type = "type1_alphabet"
        reasoning = result.get("reasoning", "")
    except Exception as e:
        logger.error("Language classification error (both providers failed): %s", e)
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


def _save_cached_unit(supabase, language, unit_index, has_positional_forms, unit_data, total_alphabet_size=None):
    if supabase is None:
        return
    try:
        supabase.table("language_unit_cache").upsert({
            "language": language,
            "unit_index": unit_index,
            "has_positional_forms": has_positional_forms,
            "unit_data": unit_data,
            "total_alphabet_size": total_alphabet_size,
        }, on_conflict="language,unit_index").execute()
    except Exception as e:
        logger.error("Unit cache save error: %s", e)


UNIT_SCHEMA_PROMPT = """You are designing one unit of a language-learning lesson for a beginner learning {language}, using an alphabet-first approach.

This is unit index {unit_index}. The learner already knows these symbols: {known_symbols}.

Return ONLY valid JSON (no markdown, no commentary) matching exactly this shape:
{{
  "has_positional_forms": boolean,
  "total_alphabet_size": integer (your best estimate of how many distinct symbols/letters this language's script has in total),
  "unit": {{
    "phase": "alphabet",
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

SENTENCE_UNIT_PROMPT = """You are designing one unit for a learner of {language} (native language: {native_language}) who has now learned the full alphabet: {known_symbols}.

This unit moves past the alphabet into simple words and sentences. The sentence, translation, vocabulary, and comprehension question should all use {language} as the main content — the learner already knows the letters, now they're reading real language.

Return ONLY valid JSON (no markdown, no commentary) matching exactly this shape:
{{
  "unit": {{
    "phase": "words_sentences",
    "unit_index": {unit_index},
    "title": "short title",
    "sentence": "a short, simple sentence in {language} (3-6 words)",
    "sentence_translation": "translation in {native_language}",
    "vocabulary": [
      {{"word": "...", "translation": "..."}}
    ],
    "comprehension_prompt": "a short question written in {language} that tests understanding of the sentence, expecting an answer in {language}"
  }}
}}

Keep the sentence appropriate for someone who just finished learning the alphabet — simple, common words, nothing obscure."""


async def _generate_unit(deepseek_client, gemini_client, supabase_client, language, unit_index, known_symbols):
    cached = _get_cached_unit(supabase_client, language, unit_index)
    if cached:
        return {
            "has_positional_forms": cached.get("has_positional_forms", False),
            "total_alphabet_size": cached.get("total_alphabet_size"),
            "unit": cached.get("unit_data"),
        }

    prompt = UNIT_SCHEMA_PROMPT.format(
        language=language, unit_index=unit_index,
        known_symbols=", ".join(known_symbols) if known_symbols else "none yet",
    )
    try:
        result = await _generate_json(deepseek_client, gemini_client, prompt)
    except Exception as e:
        logger.error("Unit generation error (both providers failed): %s", e)
        return None

    _save_cached_unit(
        supabase_client, language, unit_index,
        result.get("has_positional_forms", False),
        result.get("unit"),
        total_alphabet_size=result.get("total_alphabet_size"),
    )
    return result


async def _generate_sentence_unit(deepseek_client, gemini_client, supabase_client, language, native_language, unit_index, known_symbols):
    cached = _get_cached_unit(supabase_client, language, unit_index)
    if cached and cached.get("unit_data", {}).get("phase") == "words_sentences":
        return {"unit": cached.get("unit_data")}

    prompt = SENTENCE_UNIT_PROMPT.format(
        language=language, native_language=native_language, unit_index=unit_index,
        known_symbols=", ".join(known_symbols) if known_symbols else "the full alphabet",
    )
    try:
        result = await _generate_json(deepseek_client, gemini_client, prompt)
    except Exception as e:
        logger.error("Sentence unit generation error (both providers failed): %s", e)
        return None

    _save_cached_unit(supabase_client, language, unit_index, False, result.get("unit"))
    return result


async def _check_answer(deepseek_client, gemini_client, language, native_language, context, user_answer):
    prompt = (f"A native {native_language} speaker learning {language} was asked about: {context}. "
              f"They answered: \"{user_answer}\". "
              f"Return ONLY JSON: {{\"correct\": boolean, \"feedback\": \"one short sentence, "
              f"encouraging, in plain {native_language}, correcting if wrong\"}}")
    try:
        result = await _generate_json(deepseek_client, gemini_client, prompt)
        result["checked"] = True
        return result
    except Exception as e:
        if _is_quota_error(e):
            logger.error("Answer check QUOTA error (both providers): %s", e)
            feedback = "AIM has hit its request limit for the moment — try checking this again in a bit."
        else:
            logger.error("Answer check error (both providers failed): %s", e)
            feedback = "AIM couldn't check that just now — try again in a moment."
        return {"correct": None, "checked": False, "feedback": feedback}


async def _chat_reply(deepseek_client, gemini_client, language, native_language, unit_title, unit_content, user_message, auto_intro=False):
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
        return await _generate_text(deepseek_client, gemini_client, prompt)
    except Exception as e:
        if _is_quota_error(e):
            logger.error("Chat reply QUOTA error (both providers): %s", e)
            return "AIM has hit its request limit for the moment — give it a bit and try again."
        logger.error("Chat reply error (both providers failed): %s", e)
        return "AIM couldn't respond just now — try again in a moment."


QUIZ_GENERATION_PROMPT = """You are creating a short quiz for a learner of {language} (native language: {native_language}) covering ONLY what they've been taught so far. Do not introduce anything new. Difficulty level: {level}.

Known symbols and sounds: {symbol_list}
Known words: {word_list}

Return ONLY valid JSON: an array of up to 5 quiz questions, mixing these types where the known content allows:
- "symbol_to_sound": show a known symbol, ask what sound it makes
- "sound_to_symbol": describe a known sound, ask which known symbol makes it
- "word_meaning": show a known word, ask its meaning in {native_language}

Each item exactly this shape:
{{"id": "q1", "type": "...", "prompt": "...", "expects_free_text": boolean, "correct_answer": "...", "choices": [array of strings, or null]}}

{choices_instruction}

expects_free_text should be true only for word_meaning-style questions with no choices (short phrase answers); false whenever choices are provided or the answer is a single symbol/sound. Use ONLY symbols/words from the known lists above — never invent new ones the learner hasn't seen."""

BEGINNER_CHOICES_INSTRUCTION = (
    "This is BEGINNER level: every question MUST include a \"choices\" array of 3-4 short options "
    "(the correct answer plus 2-3 plausible wrong answers drawn from the other known symbols/words), "
    "in random order, with expects_free_text set to false."
)
INTERMEDIATE_CHOICES_INSTRUCTION = (
    "This is INTERMEDIATE level: set \"choices\" to null for every question — the learner types their own answer."
)


async def _generate_quiz(deepseek_client, gemini_client, language, native_language, units, level="intermediate"):
    symbols, words = [], []
    for u in units:
        for s in u.get("introduces", []):
            symbols.append(f'{s.get("symbol")} (sound: {s.get("sound")})')
        for w in u.get("practice_words", []):
            words.append(f'{w.get("word")} ({w.get("translation")})')

    prompt = QUIZ_GENERATION_PROMPT.format(
        language=language, native_language=native_language, level=level,
        symbol_list="; ".join(symbols) if symbols else "none",
        word_list="; ".join(words) if words else "none",
        choices_instruction=BEGINNER_CHOICES_INSTRUCTION if level == "beginner" else INTERMEDIATE_CHOICES_INSTRUCTION,
    )
    try:
        questions = await _generate_json(deepseek_client, gemini_client, prompt)
        return questions if isinstance(questions, list) else None
    except Exception as e:
        logger.error("Quiz generation error (both providers failed): %s", e)
        return None


async def _grade_quiz_answer(deepseek_client, gemini_client, native_language, question, user_answer):
    correct_answer = question.get("correct_answer", "")

    if not question.get("expects_free_text"):
        is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
        feedback = "Correct!" if is_correct else f"Not quite — the answer was {correct_answer}."
        return {"correct": is_correct, "checked": True, "feedback": feedback}

    prompt = (f"A quiz question was: \"{question.get('prompt')}\". "
              f"Reference answer: \"{correct_answer}\". "
              f"The learner (native {native_language} speaker) answered: \"{user_answer}\". "
              f"Judge if their answer is essentially correct, allowing reasonable phrasing differences. "
              f"Return ONLY JSON: {{\"correct\": boolean, \"feedback\": \"one short encouraging sentence "
              f"in plain {native_language}\"}}")
    try:
        result = await _generate_json(deepseek_client, gemini_client, prompt)
        result["checked"] = True
        return result
    except Exception as e:
        if _is_quota_error(e):
            logger.error("Quiz grading QUOTA error (both providers): %s", e)
            feedback = "AIM has hit its request limit — try this question again shortly."
        else:
            logger.error("Quiz grading error (both providers failed): %s", e)
            feedback = "AIM couldn't check that just now — try again."
        return {"correct": None, "checked": False, "feedback": feedback}