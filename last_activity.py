"""
Generic "resume where I left off" lookup — checks every learning
mini-app's session table and returns whichever was most recently
active. Add a new table/topic here whenever a new template ships,
so LearnLanding's "Resume from AIM" button never has to hardcode
a topic again.
"""

import logging
from flask import request, jsonify

logger = logging.getLogger("last_activity")


def register_last_activity_route(flask_app, supabase_client):

    @flask_app.route("/api/user/last-activity", methods=["POST"])
    def last_activity():
        data = request.get_json() or {}
        user_id = data.get("user_id", "")

        if not user_id or supabase_client is None:
            return jsonify({"found": False})

        candidates = []

        # Chess
        try:
            resp = (supabase_client.table("chess_sessions")
                    .select("last_move_at")
                    .eq("user_id", user_id)
                    .eq("status", "active")
                    .order("last_move_at", desc=True)
                    .limit(1)
                    .execute())
            if resp.data:
                candidates.append({
                    "topic": "chess",
                    "url": "/learn/chess",
                    "last_active_at": resp.data[0]["last_move_at"],
                })
        except Exception as e:
            logger.error("last_activity chess check error: %s", e)

        # Language (any language, any active session)
        try:
            resp = (supabase_client.table("language_sessions")
                    .select("language,last_active_at")
                    .eq("user_id", user_id)
                    .eq("status", "active")
                    .order("last_active_at", desc=True)
                    .limit(1)
                    .execute())
            if resp.data:
                row = resp.data[0]
                candidates.append({
                    "topic": "language",
                    "url": f"/learn/language?language={row['language']}",
                    "last_active_at": row["last_active_at"],
                })
        except Exception as e:
            logger.error("last_activity language check error: %s", e)

        if not candidates:
            return jsonify({"found": False})

        most_recent = max(candidates, key=lambda c: c["last_active_at"])
        return jsonify({"found": True, **most_recent})