"""
tier_basic.py — Basic plan ($2-3/mo, price TBD) limits & feature flags for AIM.
"""

TIER_CONFIG = {
    "tier_name": "basic",

    "models": {
        "aim_mini": "unlimited",
        "aim_alpha1": "unlimited",
        "aim_alpha1_pro": "limited_higher",  # higher cap than free, still capped
    },

    "context": {
        "recent_messages": 15,
        "older_context": True,
        "session_summary": True,   # rolling AI-generated summary enabled
    },

    "rate_limits": {
        "messages_per_day": 400,
        "aim_alpha1_pro_messages_per_day": 40,
        "github_actions_per_day": 20,
    },

    "features": {
        "medical_mode": True,
        "nebulae_1": "unlimited",
        "nebulae_1_pro": "limited",
        "cross_platform_integration": True,
        "app_connections": True,
        "premade_tasks": True,
        "custom_tasks": "limited",
        "background_workers": True,
        "learn_chess": True,
        "learn_language": True,
        "learn_language_slots": 3,        # up to 3 languages, 1 active at a time
        "learn_subjects": True,
        "learn_subject_slots": 3,          # up to 3 subjects, 1 active at a time
        "neurabridge": "multiple",
        "github_access": True,             # rate-limited, see rate_limits above
        "device_ambient_audio": False,
    },
}