"""
modes.py — AIM's "modes" registry (Medical, DeepThink, DeepSearch, Sandbox, ...)

TO ADD A NEW MODE YOURSELF:
  1. Add one entry to MODES below with a short prompt describing the
     behavior change.
  2. If the mode needs special data-gathering (like DeepSearch visiting
     URLs, or Sandbox restricting to an uploaded doc), add a small
     function here and reference it from apply_mode() at the bottom.
  3. That's it — nothing else in aimbot.py needs to change; every mode
     flows through apply_mode() below.

A "mode" is just: an extra instruction appended to the system prompt,
plus (optionally) some extra context-gathering before the AI is called.
"""

from typing import Optional

# ─── MODE DEFINITIONS ───
# key = the mode identifier sent from the frontend's Plus-button menu.
# prompt = extra instruction injected into the system prompt when active.
MODES = {
    "medical": {
        "label": "Medical Mode",
        "prompt": (
            "\n\n--- MEDICAL MODE ACTIVE ---\n"
            "The user has turned on Medical Mode. Rules:\n"
            "1. At the START of your very first reply in this mode, state "
            "plainly and briefly that you are an AI, not a doctor, and that "
            "this is informational support for someone without easy access "
            "to a real doctor right now — not a diagnosis.\n"
            "2. Ask about symptoms, duration, severity, and relevant history "
            "like a careful, caring clinician would, then give your honest "
            "read on what it might be and whether it's worth seeing a doctor.\n"
            "3. If the user mentions taking medication, offer to check in on "
            "them later — ask how they're doing, whether the medication is "
            "working, or whether it's time to go back to the doctor.\n"
            "4. If symptoms sound serious or you're unsure, always err "
            "toward recommending they see a real doctor rather than "
            "reassuring them.\n"
            "--- END MEDICAL MODE ---\n"
        ),
    },
    "deepthink": {
        "label": "DeepThink",
        "prompt": (
            "\n\n--- DEEPTHINK MODE ACTIVE ---\n"
            "Before answering, reason through the problem carefully and "
            "step-by-step in your head — consider multiple angles, check "
            "your own reasoning for mistakes, and only THEN give your final "
            "answer. Do not skip straight to a surface-level answer. If it "
            "would help the user, briefly show your key reasoning steps "
            "(not every micro-step) before the final answer, clearly "
            "separated from it.\n"
            "--- END DEEPTHINK MODE ---\n"
        ),
        # DeepThink also gets a bigger token budget + lower temperature —
        # see MODE_GENERATION_OVERRIDES below.
    },
    "deepsearch": {
        "label": "DeepSearch",
        "prompt": (
            "\n\n--- DEEPSEARCH MODE ACTIVE ---\n"
            "The system has already searched the web AND visited the "
            "actual pages of the top results (not just search snippets) — "
            "see WEB SEARCH RESULTS below, which contains full page content, "
            "not just summaries. Base your answer on that fuller content, "
            "and mention that you did a deeper look, not just a quick search.\n"
            "--- END DEEPSEARCH MODE ---\n"
        ),
        # DeepSearch's real behavior (fetching top links, not just snippets)
        # lives in aim_deepsearch() below — it must be called by aimbot.py
        # BEFORE the AI call, with its output passed in as web_context.
    },
    "search": {
        "label": "Search",
        "prompt": (
            "\n\n--- SEARCH MODE ACTIVE ---\n"
            "The user explicitly asked you to search rather than answer "
            "from memory. Web search has already been run for this message "
            "regardless of whether it looked search-worthy — see WEB SEARCH "
            "RESULTS below and base your answer on it.\n"
            "--- END SEARCH MODE ---\n"
        ),
        # Behavior: forces a normal (snippet-level) search every time,
        # bypassing the usual is_search_query() judgment call — enforced
        # in aimbot.py via forces_search() below.
    },
    "sandbox": {
        "label": "Sandbox Mode",
        "prompt": (
            "\n\n--- SANDBOX MODE ACTIVE ---\n"
            "The user is in Sandbox Mode. You must answer ONLY using the "
            "document/content they provided in this conversation — do NOT "
            "use outside knowledge, do NOT search the web, and do NOT fill "
            "in gaps from what you generally know about the topic. If the "
            "provided content doesn't contain the answer, say so plainly "
            "rather than guessing. At the end of every reply in this mode, "
            "briefly note that your answer is based only on the document "
            "provided, not external knowledge.\n"
            "--- END SANDBOX MODE ---\n"
        ),
        # Sandbox also disables web search / SEARCH_TRIGGER entirely for
        # the duration — enforced in aimbot.py by skipping the search step
        # when mode == "sandbox", not just by prompting.
    },
}

# Per-mode generation parameter overrides (falls back to caller's defaults
# if a mode isn't listed here).
MODE_GENERATION_OVERRIDES = {
    "deepthink": {"temperature": 0.4, "max_tokens": 2048},
}


def apply_mode(base_system_prompt: str, mode: Optional[str]) -> str:
    """Returns the system prompt with the active mode's instructions
    appended, or the prompt unchanged if mode is None/unrecognized."""
    if not mode or mode not in MODES:
        return base_system_prompt
    return base_system_prompt + MODES[mode]["prompt"]


def get_generation_overrides(mode: Optional[str]) -> dict:
    """Returns {"temperature": ..., "max_tokens": ...} overrides for a mode,
    or {} if the mode has no overrides (caller keeps its own defaults)."""
    return MODE_GENERATION_OVERRIDES.get(mode, {})


def disables_web_search(mode: Optional[str]) -> bool:
    """Sandbox mode must never search the web, even if the user's message
    would normally trigger a search."""
    return mode == "sandbox"


def requires_deepsearch(mode: Optional[str]) -> bool:
    """DeepSearch mode needs the fuller fetch-and-visit search pipeline
    instead of the normal snippet-only search."""
    return mode == "deepsearch"


def forces_search(mode: Optional[str]) -> bool:
    """Search mode always searches, bypassing the normal is_search_query()
    judgment call — the user explicitly asked for it."""
    return mode == "search"