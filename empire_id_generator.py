"""
EMPIRE ID GENERATOR — Standalone Module
========================================
This module handles Empire ID generation and storage.
Connects to the EmpireID Supabase project (separate from AIM's main DB).

Usage:
    from empire_id_generator import create_empire_id, get_user_by_logto, get_user_by_empire_id
"""

import os
import random
import string
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# These should point to the NEW EmpireID Supabase project (not AIM's main DB).
# Uses the SECRET key (full access, bypasses RLS) since this runs server-side
# and needs to read/write every user's Empire ID record. Falls back to the old
# EMPIRE_ID_SUPABASE_SERVICE_KEY name in case Railway hasn't been updated yet.
EMPIRE_ID_SUPABASE_URL = os.environ.get("EMPIRE_ID_SUPABASE_URL", "")
EMPIRE_ID_SUPABASE_SECRET_KEY = os.environ.get("EMPIRE_ID_SUPABASE_SECRET_KEY", "") or os.environ.get("EMPIRE_ID_SUPABASE_SERVICE_KEY", "")

# Initialize Supabase client
empire_id_client: Optional[Client] = None
if EMPIRE_ID_SUPABASE_URL and EMPIRE_ID_SUPABASE_SECRET_KEY:
    try:
        empire_id_client = create_client(EMPIRE_ID_SUPABASE_URL, EMPIRE_ID_SUPABASE_SECRET_KEY)
        logger.info("✅ EmpireID Supabase connected")
    except Exception as e:
        logger.error(f"❌ EmpireID Supabase connection failed: {e}")
else:
    logger.warning("⚠️ EmpireID Supabase not configured")

# ─── EMPIRE ID GENERATION ──────────────────────────────────────────────────────

def generate_empire_id() -> str:
    """
    Generates a unique Empire ID in format: EMP-XXXXXXXX
    where X is uppercase letter or digit (8 characters)
    """
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"EMP-{random_str}"

def is_empire_id_unique(empire_id: str) -> bool:
    """
    Checks if an Empire ID already exists in the database.
    Returns True if it's unique (doesn't exist), False if it's taken.
    """
    if not empire_id_client:
        logger.error("EmpireID client not initialized")
        return False
    
    try:
        response = empire_id_client.table("empire_ids").select("id").eq("empire_id", empire_id).execute()
        return len(response.data) == 0
    except Exception as e:
        logger.error(f"Error checking Empire ID uniqueness: {e}")
        return False

def create_empire_id(
    logto_id: str,
    username: str,
    email: str,
    source: str = "telegram_bot"  # Options: "telegram_bot", "website", "admin", "mobile_app"
) -> Tuple[bool, str]:
    """
    Creates a new Empire ID and stores it in the EmpireID Supabase project.
    
    Args:
        logto_id: The unique user ID from Logto (the 'sub' claim)
        username: User's display name
        email: User's email from Logto
        source: Where the ID was created from
    
    Returns:
        Tuple[success: bool, message: str]
        - If success: (True, "EMP-XXXXXXXX")
        - If failed: (False, "Error message")
    """
    if not empire_id_client:
        return False, "❌ EmpireID database is offline"
    
    # Check if user already has an Empire ID
    try:
        existing = empire_id_client.table("empire_ids").select("empire_id").eq("logto_id", logto_id).execute()
        if existing.data:
            existing_id = existing.data[0].get("empire_id")
            return False, f"✅ User already has Empire ID: {existing_id}"
    except Exception as e:
        logger.error(f"Error checking existing Empire ID: {e}")
        return False, f"❌ Database error: {str(e)}"
    
    # Generate unique Empire ID (with retry logic)
    max_attempts = 10
    for attempt in range(max_attempts):
        empire_id = generate_empire_id()
        
        if is_empire_id_unique(empire_id):
            # Store in database
            try:
                row = {
                    "logto_id": logto_id.strip(),
                    "empire_id": empire_id,
                    "username": username.strip() if username else None,
                    "email": email.strip().lower() if email else None,
                    "source": source,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                
                empire_id_client.table("empire_ids").insert(row).execute()
                logger.info(f"✅ Created Empire ID {empire_id} for Logto user {logto_id[:20]}...")
                return True, empire_id
                
            except Exception as e:
                logger.error(f"Error storing Empire ID: {e}")
                return False, f"❌ Failed to save Empire ID: {str(e)}"
        else:
            logger.warning(f"Empire ID {empire_id} collision, retrying... (attempt {attempt + 1}/{max_attempts})")
            continue
    
    return False, "❌ Failed to generate unique Empire ID after multiple attempts"

# ─── LOOKUP FUNCTIONS ──────────────────────────────────────────────────────────

def get_user_by_logto(logto_id: str) -> Optional[dict]:
    """
    Fetches user data from EmpireID database using their Logto ID.
    Returns the user record or None if not found.
    """
    if not empire_id_client:
        return None
    
    try:
        response = empire_id_client.table("empire_ids").select("*").eq("logto_id", logto_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching user by Logto ID: {e}")
        return None

def get_user_by_empire_id(empire_id: str) -> Optional[dict]:
    """
    Fetches user data from EmpireID database using their Empire ID.
    Returns the user record or None if not found.
    """
    if not empire_id_client:
        return None
    
    try:
        response = empire_id_client.table("empire_ids").select("*").eq("empire_id", empire_id.upper()).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching user by Empire ID: {e}")
        return None

def get_user_stats(source: Optional[str] = None) -> dict:
    """
    Gets statistics about Empire IDs.
    If source is provided, filters by that source.
    """
    if not empire_id_client:
        return {"total": 0, "error": "Database not connected"}
    
    try:
        query = empire_id_client.table("empire_ids").select("id", count="exact")
        if source:
            query = query.eq("source", source)
        
        response = query.execute()
        
        # Get breakdown by source
        sources_query = empire_id_client.rpc("get_source_counts") if source is None else None
        
        return {
            "total": response.count or 0,
            "by_source": "Use SQL function for breakdown"  # You can implement this later
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"total": 0, "error": str(e)}

# ─── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────

def validate_empire_id_format(empire_id: str) -> bool:
    """
    Validates if a string matches the Empire ID format: EMP-XXXXXXXX
    """
    import re
    pattern = r'^EMP-[A-Z0-9]{8}$'
    return bool(re.match(pattern, empire_id.upper()))

# ─── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: Create a new Empire ID
    success, result = create_empire_id(
        logto_id="user123_from_logto",
        username="David Emmanuel",
        email="david@empireunion.xyz",
        source="telegram_bot"
    )
    
    if success:
        print(f"✅ Empire ID created: {result}")
    else:
        print(f"❌ Error: {result}")
    
    # Example: Look up user by Logto ID
    user = get_user_by_logto("user123_from_logto")
    if user:
        print(f"👤 User found: {user}")
    else:
        print("👤 User not found")