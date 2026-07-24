"""
AIM Chess Engine — AI-powered chess opponent with memory and learning.
This module provides the backend chess logic that replaces Stockfish.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from flask import request, jsonify
from chess import Chess, Move

logger = logging.getLogger("aimbot.chess")

# ─── AI MOVE GENERATION ───
async def get_aim_move(fen: str, difficulty: int, player_color: str, move_history: List[str], user_id: str) -> Optional[str]:
    """
    Ask AIM (Gemini/DeepSeek) to generate a chess move.
    Returns UCI notation (e.g., 'e2e4') or None.
    """
    from aimbot import gemini_client, deepseek_client, USE_DEEPSEEK
    from google.genai import types

    # Build the prompt
    history_str = " ".join(move_history[-10:]) if move_history else "No moves yet."

    prompt = f"""You are AIM, a chess coach and opponent. The user is playing against you.

Current board position (FEN): {fen}
You are playing as: {'Black' if player_color == 'white' else 'White'}
Difficulty level: {difficulty}/15 (1=beginner, 15=grandmaster)
Recent moves: {history_str}

Rules:
- Respond with ONLY a valid UCI move (4-5 characters, e.g., 'e2e4', 'g1f3', 'e7e8q')
- The move must be legal in the current position
- At difficulty 1-3: make occasional mistakes, don't always play the best move
- At difficulty 5-8: play solidly, occasional tactical mistakes
- At difficulty 10-12: play strong, find good tactics
- At difficulty 13-15: play like a strong club player, find best moves
- If checkmate is possible, take it
- If the user blundered, punish it

Respond with ONLY the move, no explanation."""

    try:
        move_text = None

        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            move_text = r.choices[0].message.content.strip() if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=10)
            )
            move_text = r.text.strip() if r and r.text else None

        if not move_text:
            return None

        # Clean up — extract just the move
        move_text = move_text.lower().strip()
        # Remove any punctuation or extra text
        move_text = ''.join(c for c in move_text if c.isalnum())

        # Validate it's a proper UCI move (4 or 5 chars)
        if len(move_text) < 4 or len(move_text) > 5:
            logger.warning(f"AIM returned invalid move format: {move_text}")
            return None

        # Validate it's a legal move
        game = Chess(fen)
        legal_moves = [m.uci() for m in game.legal_moves]

        if move_text not in legal_moves:
            # Try to find a close match (sometimes AI returns slightly wrong notation)
            logger.warning(f"AIM move {move_text} not legal, trying fallback")
            return None

        logger.info(f"AIM move: {move_text} (difficulty {difficulty})")
        return move_text

    except Exception as e:
        logger.error(f"AIM move generation error: {e}")
        return None


# ─── POSITION EVALUATION ───
async def evaluate_position(fen: str) -> Dict:
    """
    Ask AIM to evaluate a chess position.
    Returns {score: float, assessment: str}
    """
    from aimbot import gemini_client, deepseek_client, USE_DEEPSEEK
    from google.genai import types

    prompt = f"""Evaluate this chess position (FEN: {fen}).
Respond with ONLY a JSON object:
{{"score": float (positive=white advantage, negative=black advantage, in pawns),
  "assessment": "brief description"}}
Example: {{"score": 0.5, "assessment": "White has a slight space advantage"}}"""

    try:
        eval_text = None

        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            eval_text = r.choices[0].message.content.strip() if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=100)
            )
            eval_text = r.text.strip() if r and r.text else None

        if not eval_text:
            return {"score": 0.0, "assessment": "Position unclear"}

        # Extract JSON
        import re
        json_match = re.search(r'\{.*?\}', eval_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "score": float(data.get("score", 0)),
                "assessment": data.get("assessment", "Position unclear")
            }

        return {"score": 0.0, "assessment": "Position unclear"}

    except Exception as e:
        logger.error(f"Position evaluation error: {e}")
        return {"score": 0.0, "assessment": "Position unclear"}


# ─── MOVE ANALYSIS / COACHING ───
async def analyze_move(fen_before: str, fen_after: str, move_san: str, player_color: str) -> str:
    """
    Ask AIM to analyze the user's move and give coaching feedback.
    """
    from aimbot import gemini_client, deepseek_client, USE_DEEPSEEK
    from google.genai import types

    prompt = f"""You are AIM, an encouraging chess coach. A student just played {move_san}.

Position before: {fen_before}
Position after: {fen_after}
Student plays: {player_color}

Give a SHORT, encouraging coaching message (1-2 sentences):
- If it's a great move: celebrate it
- If it's a good move: acknowledge it
- If it's a mistake: gently point it out, suggest what to look for
- If it's a blunder: be kind but clear about why it loses material

Keep it under 100 characters. Use emojis. Be encouraging!"""

    try:
        coaching = None

        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            coaching = r.choices[0].message.content.strip() if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=150)
            )
            coaching = r.text.strip() if r and r.text else None

        return coaching or "Good move! Keep thinking ahead."

    except Exception as e:
        logger.error(f"Move analysis error: {e}")
        return "Good move! Keep thinking ahead."


# ─── POST-GAME SUMMARY ───
async def generate_game_summary(move_history: List[str], game_result: str) -> str:
    """
    Writes a short, encouraging recap of the finished game — why the
    player won or lost, and the turning point(s) — sent to them via
    Telegram after the game ends (see send_summary_to_telegram below).
    """
    from aimbot import gemini_client, deepseek_client, USE_DEEPSEEK
    from google.genai import types

    moves_text = " ".join(move_history) if move_history else "(no moves recorded)"

    prompt = f"""You are AIM, an encouraging chess coach. A student just finished a game.

Result: {game_result} (from the student's perspective)
Full move list (SAN): {moves_text}

Write a short, warm recap for the student, covering:
1. The overall result and how the game felt (a few words)
2. The 1-2 key turning point moves — name the actual move(s) from the list and briefly why they mattered
3. One concrete thing to work on next time

Keep it to 3-4 short sentences total. Use emojis. Be honest but kind — this
is a student learning, not a grandmaster being reviewed."""

    try:
        summary = None

        if USE_DEEPSEEK and deepseek_client:
            r = await deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            summary = r.choices[0].message.content.strip() if r.choices else None
        elif gemini_client:
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=300)
            )
            summary = r.text.strip() if r and r.text else None

        return summary or f"Game over — result: {game_result}. Keep practicing! ♟️"

    except Exception as e:
        logger.error(f"Game summary generation error: {e}")
        return f"Game over — result: {game_result}. Keep practicing! ♟️"


async def send_summary_to_telegram(empire_id: str, summary_text: str) -> bool:
    """
    Delivers the post-game summary via Telegram — the only channel that
    exists right now regardless of whether the game was played on the
    website or Telegram itself. (Web delivery can be added once AIM's
    own web chat is live — this is the one place that'll need updating.)
    Looks up the Telegram chat id from the MAIN aimbot Supabase project's
    user_profiles table, keyed by empire_id (populated via /link).
    """
    from aimbot import supabase, bot, send_text_chunks

    if not supabase or not bot:
        logger.error("send_summary_to_telegram: supabase or bot not configured")
        return False

    try:
        res = supabase.table("user_profiles").select("user_id").eq("empire_id", empire_id).execute()
        if not res.data:
            logger.warning(f"send_summary_to_telegram: no linked Telegram account for empire_id={empire_id}")
            return False

        telegram_user_id = res.data[0]["user_id"]
        message = f"♟️ <b>Game Summary</b>\n\n{summary_text}"
        await send_text_chunks(int(telegram_user_id), message)
        return True

    except Exception as e:
        logger.error(f"send_summary_to_telegram error: {e}")
        return False


# ─── PATTERN LEARNING ───
def extract_patterns(move_history: List[str], game_result: str) -> Dict:
    """
    Analyze a completed game to extract learning patterns.
    Returns weaknesses, strengths, key moments.
    """
    patterns = {
        "weaknesses": [],
        "strengths": [],
        "key_moments": [],
        "opening": None,
        "blunders": 0,
        "brilliancies": 0
    }

    if not move_history:
        return patterns

    # Simple heuristic-based pattern extraction
    # In production, this would use a real engine or deeper AI analysis

    # Check opening
    if len(move_history) >= 2:
        first_moves = " ".join(move_history[:2]).lower()
        if "e4" in first_moves:
            patterns["opening"] = "King's Pawn Opening"
        elif "d4" in first_moves:
            patterns["opening"] = "Queen's Pawn Opening"
        elif "c4" in first_moves:
            patterns["opening"] = "English Opening"
        elif "nf3" in first_moves:
            patterns["opening"] = "Réti Opening"

    # Count moves (very basic heuristics)
    for i, move in enumerate(move_history):
        move_lower = move.lower()

        # Blunder indicators (simplified)
        if any(x in move_lower for x in ["??", "blunder"]):
            patterns["blunders"] += 1
            patterns["weaknesses"].append("tactical awareness")

        # Good move indicators
        if any(x in move_lower for x in ["!", "excellent", "brilliant"]):
            patterns["brilliancies"] += 1
            patterns["strengths"].append("tactical vision")

    # Deduplicate
    patterns["weaknesses"] = list(set(patterns["weaknesses"]))[:3]
    patterns["strengths"] = list(set(patterns["strengths"]))[:3]

    return patterns


# ─── SUPABASE HELPERS ───
async def save_move_to_db(user_id: str, empire_id: str, game_id: str, 
                          move_san: str, fen: str, move_number: int, 
                          is_player_move: bool, coaching: str = ""):
    """Save a single move to the database."""
    from aimbot import supabase
    if not supabase:
        return

    try:
        supabase.table("chess_moves").insert({
            "user_id": user_id,
            "empire_id": empire_id,
            "game_id": game_id,
            "move_san": move_san,
            "fen": fen,
            "move_number": move_number,
            "is_player_move": is_player_move,
            "coaching": coaching,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Save move error: {e}")


async def update_learning_profile(user_id: str, empire_id: str, game_result: str, 
                                  patterns: Dict, total_moves: int):
    """Update user's learning profile after a game."""
    from aimbot import supabase
    if not supabase:
        return

    try:
        # Get existing profile
        res = supabase.table("user_learning_profiles")            .select("*")            .eq("user_id", user_id)            .single()            .execute()

        if res.data:
            # Update existing
            current = res.data
            games_played = current.get("chess_games_played", 0) + 1
            wins = current.get("chess_wins", 0) + (1 if game_result == "win" else 0)
            losses = current.get("chess_losses", 0) + (1 if game_result == "loss" else 0)
            draws = current.get("chess_draws", 0) + (1 if game_result == "draw" else 0)

            # Simple ELO adjustment
            elo = current.get("chess_elo", 400)
            if game_result == "win":
                elo += 15
            elif game_result == "loss":
                elo -= 10
            elif game_result == "draw":
                elo += 5
            elo = max(100, min(3000, elo))

            # Merge weaknesses and strengths
            old_weak = set(current.get("chess_weaknesses", []) or [])
            new_weak = set(patterns.get("weaknesses", []))
            old_strong = set(current.get("chess_strengths", []) or [])
            new_strong = set(patterns.get("strengths", []))

            supabase.table("user_learning_profiles").update({
                "chess_games_played": games_played,
                "chess_wins": wins,
                "chess_losses": losses,
                "chess_draws": draws,
                "chess_elo": elo,
                "chess_weaknesses": list(old_weak | new_weak)[:5],
                "chess_strengths": list(old_strong | new_strong)[:5],
                "last_chess_session": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("user_id", user_id).execute()

        else:
            # Create new profile
            supabase.table("user_learning_profiles").insert({
                "user_id": user_id,
                "empire_id": empire_id,
                "chess_games_played": 1,
                "chess_wins": 1 if game_result == "win" else 0,
                "chess_losses": 1 if game_result == "loss" else 0,
                "chess_draws": 1 if game_result == "draw" else 0,
                "chess_elo": 415 if game_result == "win" else 390 if game_result == "loss" else 405,
                "chess_weaknesses": patterns.get("weaknesses", []),
                "chess_strengths": patterns.get("strengths", []),
                "last_chess_session": datetime.now(timezone.utc).isoformat(),
                "current_learning_topics": ["chess"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).execute()

    except Exception as e:
        logger.error(f"Update learning profile error: {e}")


# ─── FLASK API ENDPOINTS ───
# These will be added to aimbot.py's Flask app

def register_chess_routes(flask_app, supabase_client):
    """Register chess API routes on the Flask app.

    NOTE: these routes must be plain `def`, not `async def` — this Flask
    app has no async-view support installed (no asgiref/flask[async]),
    so an async route crashes with an unhandled 500 (an HTML error page,
    not JSON) before its own try/except ever runs. That was silently
    breaking every one of these three endpoints — the frontend's
    getAIMMove()/getCoaching() calls always failed and fell back to a
    random legal move / a hardcoded coaching message respectively. The
    async helper functions above (get_aim_move, evaluate_position,
    analyze_move, update_learning_profile) still work fine — they're
    just bridged in via aimbot's existing run_async() helper, the same
    pattern already used for /set-webhook and /delete-webhook."""

    @flask_app.route("/api/chess/move", methods=["POST"])
    def chess_move():
        """Handle a chess move from the frontend."""
        from aimbot import run_async
        try:
            data = request.get_json(silent=True) or {}
            fen = data.get("fen")
            difficulty = data.get("difficulty", 5)
            player_color = data.get("player_color", "white")
            move_history = data.get("move_history", [])
            user_id = data.get("user_id")
            empire_id = data.get("empire_id")

            if not fen or not user_id or not empire_id:
                return jsonify({"error": "Missing required fields"}), 400

            # Get AIM's move
            aim_move = run_async(get_aim_move(fen, difficulty, player_color, move_history, user_id)).result(timeout=15)

            if not aim_move:
                # Fallback: random legal move
                game = Chess(fen)
                legal = [m.uci() for m in game.legal_moves]
                if legal:
                    import random
                    aim_move = random.choice(legal)
                else:
                    return jsonify({"error": "No legal moves"}), 400

            # Evaluate position
            eval_result = run_async(evaluate_position(fen)).result(timeout=15)

            return jsonify({
                "move": aim_move,
                "evaluation": eval_result["score"],
                "assessment": eval_result["assessment"]
            })

        except Exception as e:
            logger.error(f"Chess move API error: {e}")
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/api/chess/analyze", methods=["POST"])
    def chess_analyze():
        """Analyze a user's move and return coaching."""
        from aimbot import run_async
        try:
            data = request.get_json(silent=True) or {}
            fen_before = data.get("fen_before")
            fen_after = data.get("fen_after")
            move_san = data.get("move_san")
            player_color = data.get("player_color", "white")

            if not all([fen_before, fen_after, move_san]):
                return jsonify({"error": "Missing required fields"}), 400

            coaching = run_async(analyze_move(fen_before, fen_after, move_san, player_color)).result(timeout=15)

            return jsonify({"coaching": coaching})

        except Exception as e:
            logger.error(f"Chess analyze API error: {e}")
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/api/chess/save", methods=["POST"])
    def chess_save():
        """Save game data after completion, then send a post-game summary
        via Telegram — the only delivery channel that exists right now,
        even for games played on the website (see send_summary_to_telegram)."""
        from aimbot import run_async
        try:
            data = request.get_json(silent=True) or {}
            user_id = data.get("user_id")
            empire_id = data.get("empire_id")
            game_result = data.get("result")
            move_history = data.get("move_history", [])

            if not all([user_id, empire_id, game_result]):
                return jsonify({"error": "Missing required fields"}), 400

            # Extract patterns
            patterns = extract_patterns(move_history, game_result)

            # Update learning profile
            run_async(update_learning_profile(user_id, empire_id, game_result, patterns, len(move_history))).result(timeout=15)

            # Generate + send the post-game summary. Best-effort: a failure
            # here shouldn't fail the whole save (the game itself is
            # already safely recorded above).
            summary_sent = False
            try:
                summary_text = run_async(generate_game_summary(move_history, game_result)).result(timeout=20)
                summary_sent = run_async(send_summary_to_telegram(empire_id, summary_text)).result(timeout=15)
            except Exception as e:
                logger.error(f"Post-game summary delivery failed (non-fatal): {e}")

            return jsonify({"success": True, "patterns": patterns, "summary_sent": summary_sent})

        except Exception as e:
            logger.error(f"Chess save API error: {e}")
            return jsonify({"error": str(e)}), 500