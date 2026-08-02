"""
tier_pro.py — Pro plan ($5-7/mo, price TBD) limits & feature flags for AIM.

Note: language/subject slot caps (3 each, 1 active at a time) are kept
intentionally on Pro too, per product decision — not a cost constraint,
just a deliberate design choice for focus.
"""

TIER_CONFIG = {
    "tier_name": "pro",

    "models": {
        "aim_mini": "unlimited",
        "aim_alpha1": "unlimited",
        "aim_alpha1_pro": "unlimited",
    },

    "context": {
        "recent_messages": 40,
        "older_context": True,
        "session_summary": True,
    },

    "rate_limits": {
        "messages_per_day": None,   # no hard cap; still logged for abuse detection
        "github_actions_per_day": None,
    },

    "features": {
        "medical_mode": True,
        "nebulae_1": "unlimited",
        "nebulae_1_pro": "unlimited",
        "cross_platform_integration": True,
        "app_connections": True,
        "premade_tasks": True,
        "custom_tasks": "unlimited",
        "background_workers": True,
        "learn_chess": True,
        "learn_language": True,
        "learn_language_slots": 3,
        "learn_subjects": True,
        "learn_subject_slots": 3,
        "neurabridge": "unlimited",
        "github_access": "unlimited",
        "device_ambient_audio": True,   # Bluetooth-aware ambient music/singing feature
    },
}