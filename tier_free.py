"""
tier_free.py — Free plan limits & feature flags for AIM.

Import TIER_CONFIG wherever you need to check what a free user can do,
e.g. in aimbot.py: `from tier_free import TIER_CONFIG as FREE_CONFIG`.

Keep this file to CONFIG ONLY — no enforcement logic here. Enforcement
(checking a user's message count against these limits, etc.) belongs in
aimbot.py so all tiers are checked through one consistent code path.
"""

TIER_CONFIG = {
    "tier_name": "free",

    # --- Model access ---
    "models": {
        "aim_mini": "unlimited",       # Gemini 2.5 Flash Lite
        "aim_alpha1": "limited",       # Gemini 2.5 Flash
        "aim_alpha1_pro": "very_limited",  # DeepSeek V4 / V4 Pro
    },

    # --- Context window trimming (see cost-control discussion) ---
    "context": {
        "recent_messages": 5,     # how many raw recent turns to include
        "older_context": False,   # no semantic lookback into older history
        "session_summary": False, # no rolling summary for free tier
    },

    # --- Rate limits (soft daily caps, not hard walls) ---
    "rate_limits": {
        "messages_per_day": 60,
        "aim_alpha1_messages_per_day": 15,
        "aim_alpha1_pro_messages_per_day": 3,
    },

    # --- Feature access ---
    "features": {
        "medical_mode": True,
        "nebulae_1": True,
        "nebulae_1_pro": False,
        "cross_device_integration": True,
        "app_connections": "limited",
        "premade_tasks": True,
        "custom_tasks": False,
        "background_workers": True,   # timers/stopwatches
        "learn_chess": True,
        "learn_language": "trial_30_days",   # 1 language, 30-day trial
        "learn_language_slots": 1,
        "learn_subjects": False,
        "neurabridge": False,
        "github_access": False,
        "device_ambient_audio": False,  # Bluetooth-aware ambient music feature
    },
}