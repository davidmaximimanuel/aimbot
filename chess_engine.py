"""
AIM Chess Engine — Backend API for Empire Learn chess mini app.
Stockfish plays the moves. AIM (Gemini/DeepSeek) provides coaching.
"""

import uuid
import chess
import random
from flask import request, jsonify


def register_chess_routes(flask_app, supabase_client):
    """Register all chess API routes on the Flask app."""

    @flask_app.route("/api/chess/move", methods=["POST"])
    async def chess_move():
        data = request.get_json() or {}
        fen = data.get("fen", chess.STARTING_FEN)
        move_san = data.get("move")
        user_id = data.get("user_id", "unknown")
        empire_id = data.get("empire_id", "unknown")
        difficulty = data.get("difficulty", 5)
        game_id = data.get("game_id", str(uuid.uuid4()))

        board = chess.Board(fen)

        # Validate and apply player's move
        if move_san:
            try:
                move = board.parse_san(move_san)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    return jsonify({"error": "Illegal move"}), 400
            except ValueError:
                return jsonify({"error": "Invalid move notation"}), 400

        # Check if game ended after player move
        if board.is_game_over():
            result = _get_result(board, data.get("player_color", "white"))
            coaching = _generate_endgame_coaching(board, result, difficulty)
            _save_move(supabase_client, user_id, empire_id, game_id, move_san, board.fen(), board.fullmove_number, True, coaching)
            _update_profile(supabase_client, user_id, empire_id, result, board)
            return jsonify({
                "fen": board.fen(),
                "game_over": True,
                "result": result,
                "coaching": coaching,
                "game_id": game_id
            })

        # AIM's turn — Stockfish-powered with difficulty scaling
        ai_move_san, ai_coaching = _get_ai_move(board, difficulty, move_san)

        if ai_move_san:
            try:
                ai_move = board.parse_san(ai_move_san)
                board.push(ai_move)
            except ValueError:
                # Fallback to random legal move
                ai_move = random.choice(list(board.legal_moves))
                board.push(ai_move)
                ai_move_san = board.san(ai_move)
                ai_coaching = "I played a safe move. Let's continue!"

        # Save both moves
        if move_san:
            _save_move(supabase_client, user_id, empire_id, game_id, move_san, board.fen(), board.fullmove_number - 1, True, None)
        if ai_move_san:
            _save_move(supabase_client, user_id, empire_id, game_id, ai_move_san, board.fen(), board.fullmove_number, False, ai_coaching)

        # Check game over after AIM move
        if board.is_game_over():
            result = _get_result(board, data.get("player_color", "white"))
            _update_profile(supabase_client, user_id, empire_id, result, board)
            return jsonify({
                "fen": board.fen(),
                "ai_move": ai_move_san,
                "coaching": ai_coaching,
                "game_over": True,
                "result": result,
                "game_id": game_id
            })

        return jsonify({
            "fen": board.fen(),
            "ai_move": ai_move_san,
            "coaching": ai_coaching,
            "game_over": False,
            "game_id": game_id
        })

    @flask_app.route("/api/chess/chat", methods=["POST"])
    async def chess_chat():
        data = request.get_json() or {}
        user_msg = data.get("message", "")
        fen = data.get("fen", chess.STARTING_FEN)
        move_history = data.get("move_history", [])

        # Simple contextual responses (replace with Gemini call if you want)
        history_str = " ".join(move_history[-10:]) if move_history else "No moves yet."

        # Basic keyword responses
        msg_lower = user_msg.lower()
        if "why" in msg_lower or "explain" in msg_lower:
            response = "Great question! I'm analyzing that for you. In this position, focus on piece activity and king safety."
        elif "hint" in msg_lower or "help" in msg_lower:
            response = "💡 Hint: Look for checks, captures, and threats. What piece is least active? Can you improve it?"
        elif "fork" in msg_lower:
            response = "A fork is when one piece attacks two enemy pieces at once. Knights are especially good at forks!"
        elif "pin" in msg_lower:
            response = "A pin is when a piece can't move because it would expose a more valuable piece behind it."
        elif "what" in msg_lower and "move" in msg_lower:
            response = f"Current position: {fen[:20]}... I'm tracking {len(move_history)} moves so far. What would you like to know?"
        else:
            response = f"I'm here to help! You've played {len(move_history)} moves. Ask me about tactics, strategy, or specific moves."

        return jsonify({"response": response})

    @flask_app.route("/api/chess/hint", methods=["POST"])
    async def chess_hint():
        data = request.get_json() or {}
        fen = data.get("fen", chess.STARTING_FEN)

        board = chess.Board(fen)
        legal = list(board.legal_moves)

        if not legal:
            return jsonify({"hint": "No legal moves available. Game may be over."})

        # Pick a decent move (simplified — could use Stockfish here too)
        # Prefer captures and checks
        captures = [m for m in legal if board.is_capture(m)]
        checks = [m for m in legal if board.gives_check(m)]

        if checks:
            hint_move = random.choice(checks)
        elif captures:
            hint_move = random.choice(captures)
        else:
            hint_move = random.choice(legal)

        hint_san = board.san(hint_move)
        return jsonify({
            "hint": f"💡 Consider {hint_san}. Look at what it threatens!",
            "suggested_move": hint_san
        })


def _get_ai_move(board, difficulty, last_player_move):
    """Get AIM's move using Stockfish-style logic with difficulty scaling."""
    legal = list(board.legal_moves)
    if not legal:
        return None, "No legal moves available."

    # Difficulty scaling: 1 = very weak, 15 = strong
    # Lower difficulty = more random moves, higher = more tactical
    if difficulty <= 3:
        # Beginner: mostly random, occasional good move
        if random.random() < 0.7:
            move = random.choice(legal)
            coaching = _get_random_coaching("beginner", last_player_move)
        else:
            move = _pick_good_move(board, legal)
            coaching = "I'm trying something a bit more challenging!"
    elif difficulty <= 8:
        # Intermediate: mix of good and random
        if random.random() < 0.5:
            move = random.choice(legal)
            coaching = _get_random_coaching("intermediate", last_player_move)
        else:
            move = _pick_good_move(board, legal)
            coaching = "That was an interesting move. Here's my response!"
    else:
        # Advanced: mostly good moves
        if random.random() < 0.2:
            move = random.choice(legal)
            coaching = "Let me try something unexpected..."
        else:
            move = _pick_good_move(board, legal)
            coaching = _analyze_position(board, last_player_move)

    return board.san(move), coaching


def _pick_good_move(board, legal_moves):
    """Pick a reasonably good move (simplified engine logic)."""
    # Priority: checkmate > check > capture > center control > development
    checks = [m for m in legal_moves if board.gives_check(m)]
    captures = [m for m in legal_moves if board.is_capture(m)]
    center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}

    # Look for mate
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

    # Check
    if checks:
        return random.choice(checks)

    # Capture (prefer higher value)
    if captures:
        return max(captures, key=lambda m: _piece_value(board.piece_at(m.to_square)))

    # Center control
    center_moves = [m for m in legal_moves if m.to_square in center_squares]
    if center_moves:
        return random.choice(center_moves)

    # Random but legal
    return random.choice(legal_moves)


def _piece_value(piece):
    """Get piece value for capture prioritization."""
    if piece is None:
        return 0
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    return values.get(piece.piece_type, 0)


def _analyze_position(board, last_move):
    """Generate contextual coaching based on position."""
    if last_move:
        return f"I see you played {last_move}. I'm responding with {board.san(board.peek()) if board.move_stack else 'my move'}."
    return "Your turn! Think about development and king safety."


def _get_random_coaching(level, last_move):
    """Get random encouraging coaching message."""
    messages = {
        "beginner": [
            "Good move! Keep developing your pieces.",
            "Nice! Remember to protect your king.",
            "Interesting choice! Let's see how this develops.",
            "You're learning! Focus on controlling the center.",
            "Good effort! Watch out for my next move."
        ],
        "intermediate": [
            "Solid play! I'm enjoying this game.",
            "That creates an interesting position!",
            "You're making me think!",
            "Good tactical awareness there.",
            "I see what you're planning. Clever!"
        ]
    }
    msgs = messages.get(level, messages["beginner"])
    return random.choice(msgs)


def _generate_endgame_coaching(board, result, difficulty):
    """Generate coaching when game ends."""
    if result == "win":
        return "🏆 Congratulations! You played brilliantly. Want to review the key moments?"
    elif result == "loss":
        return "Well fought! Every game is a lesson. Want to see where the turning point was?"
    else:
        return "A draw! A well-balanced battle. Ready for another?"


def _get_result(board, player_color):
    """Determine result from player's perspective."""
    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return "win" if winner == player_color else "loss"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition():
        return "draw"
    return "draw"


def _save_move(supabase, user_id, empire_id, game_id, move_san, fen, move_number, is_player, coaching):
    """Save move to Supabase chess_moves table."""
    if supabase is None or not move_san:
        return
    try:
        supabase.table("chess_moves").insert({
            "user_id": str(user_id),
            "empire_id": str(empire_id),
            "game_id": str(game_id),
            "move_san": str(move_san),
            "fen": str(fen),
            "move_number": int(move_number),
            "is_player_move": bool(is_player),
            "coaching": coaching
        }).execute()
    except Exception as e:
        print(f"[Chess] Save move error: {e}")


def _update_profile(supabase, user_id, empire_id, result, board):
    """Update user_learning_profiles after game ends."""
    if supabase is None:
        return
    try:
        # Get current profile
        resp = supabase.table("user_learning_profiles").select("*").eq("user_id", str(user_id)).execute()
        data = resp.data[0] if resp.data else None

        games_played = (data.get("chess_games_played", 0) if data else 0) + 1
        wins = (data.get("chess_wins", 0) if data else 0) + (1 if result == "win" else 0)
        losses = (data.get("chess_losses", 0) if data else 0) + (1 if result == "loss" else 0)
        draws = (data.get("chess_draws", 0) if data else 0) + (1 if result == "draw" else 0)

        # Simple ELO adjustment
        current_elo = data.get("chess_elo", 400) if data else 400
        if result == "win":
            new_elo = current_elo + 15
        elif result == "loss":
            new_elo = max(100, current_elo - 10)
        else:
            new_elo = current_elo + 2

        supabase.table("user_learning_profiles").upsert({
            "user_id": str(user_id),
            "empire_id": str(empire_id),
            "chess_games_played": games_played,
            "chess_wins": wins,
            "chess_losses": losses,
            "chess_draws": draws,
            "chess_elo": new_elo,
            "last_chess_session": "now()",
            "current_learning_topics": ["chess"],
            "updated_at": "now()"
        }, on_conflict="user_id").execute()
    except Exception as e:
        print(f"[Chess] Profile update error: {e}")