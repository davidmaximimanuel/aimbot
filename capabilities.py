"""
capabilities.py — Semantic Search Routing
"""
import numpy as np
from sentence_transformers import SentenceTransformer
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
SEARCH_TRIGGER_PHRASES = [
    "who won the match", "what is the score", "latest news about", "current events",
    "what happened today", "search for information", "look up", "find out about",
    "who knocked out", "eliminated from", "when did they win", "what is the price",
    "exchange rate", "weather forecast", "stock price", "bitcoin price",
    "currency conversion", "flight status", "traffic update", "road conditions",
    "event schedule", "concert tickets", "movie release date", "album release",
    "who is the president", "who is the governor", "latest update on",
    "recent developments", "breaking news", "current situation", "what is happening now",
    "live update", "real-time information", "who won the election", "match result",
    "game outcome", "tournament winner", "championship result", "final score",
    "standings table", "league table", "fixture list", "upcoming matches",
    "next game", "who is playing", "schedule for", "when is the match",
    "kickoff time", "venue information", "ticket prices", "how to watch",
    "broadcast information", "streaming options", "what did this celebrity do",
    "Politics", "Sports", "Entertainment",
    "formula 1 result", "f1 race winner", "grand prix results",
    "nba score", "basketball result", "tennis result", "wimbledon winner",
    "boxing match result", "ufc fight night", "mma result",
    "rugby result", "cricket score", "ipl result", "who won the super bowl",
    "who stopped them from qualifying", "who stopped nigeria", "who knocked nigeria out",
    "why did nigeria not qualify", "who beat nigeria", "did nigeria qualify",
    "nigeria world cup", "super eagles result", "super eagles match", "afcon result",
    "african cup of nations", "world cup qualification africa",
    "who invented", "what caused", "why did", "how did", "when did", "what year did",
    "tell me about", "give me information on", "what do you know about",
    "news about", "update on", "facts about", "history of", "background on",
    "what is going on with", "recent news", "what happened with", "explain what happened",
    "naira exchange rate", "dollar to naira", "fuel price nigeria",
    "nigerian government", "nigerian politics", "tinubu", "lagos news", "abuja news",
    "nigeria economy", "nigeria inflation", "nigeria election", "nigeria insecurity",
]
trigger_embeddings = semantic_model.encode(SEARCH_TRIGGER_PHRASES)
def is_search_query_semantic(text: str, threshold: float = 0.45) -> bool:
    try:
        sims = np.dot(trigger_embeddings, semantic_model.encode([text]).T).flatten()
        max_sim = float(np.max(sims))
        result = max_sim >= threshold
        return result
    except Exception:
        return False
def is_search_query(text: str) -> bool:
    tl = text.lower().strip()
    if any(t in tl for t in ["search for","google","look up","find out","search the web","browse","search"]):
        return True
    return is_search_query_semantic(text)