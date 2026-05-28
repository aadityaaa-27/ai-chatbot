"""
Authentication & Role-Based Access Control
=========================================
Roles (high → low privilege):
  admin    — everything, including user management
  hr       — all data, all HR tabs, upload
  payroll  — restricted to Finance_Payroll dataset + salary analytics
  manager  — all data, but filtered to their own department

Users are stored in the Supabase `app_users` table.
Passwords are SHA-256 hashed (no extra dependencies).
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

# Compute the .env path ONCE at import time — most reliable approach
# (avoids any cwd issues; __file__ is always the absolute path of auth.py)
_AUTH_DIR = Path(__file__).resolve().parent
_ENV_FILE  = _AUTH_DIR / ".env"

# ── Role definitions ──────────────────────────────────────────────────────────

ROLES: dict[str, dict] = {
    "admin": {
        "label":           "Administrator",
        "icon":            "👑",
        "tabs":            ["chat", "analytics", "upload", "admin"],
        "can_see_salary":  True,
        "datasets":        None,   # None = access to all datasets
        "dept_filter":     False,  # no forced department filter
    },
    "hr": {
        "label":           "HR Manager",
        "icon":            "🧑‍💼",
        "tabs":            ["chat", "analytics", "upload"],
        "can_see_salary":  True,
        "datasets":        None,
        "dept_filter":     False,
    },
    "payroll": {
        "label":           "Payroll Department",
        "icon":            "💰",
        "tabs":            ["chat", "analytics"],
        "can_see_salary":  True,
        "datasets":        ["Finance_Payroll_2024"],  # hard-restricted
        "dept_filter":     False,
    },
    "manager": {
        "label":           "Department Manager",
        "icon":            "🏢",
        "tabs":            ["chat", "analytics"],
        "can_see_salary":  False,  # managers cannot see salary data
        "datasets":        None,
        "dept_filter":     True,   # auto-filtered to user's department
    },
}

# ── Default users (seeded into Supabase on first login) ───────────────────────

_SEED_USERS = [
    {
        "username":   "admin",
        "password":   "admin123",
        "role":       "admin",
        "department": None,
        "full_name":  "System Administrator",
    },
    {
        "username":   "hr_manager",
        "password":   "hr123",
        "role":       "hr",
        "department": None,
        "full_name":  "HR Manager",
    },
    {
        "username":   "payroll_user",
        "password":   "payroll123",
        "role":       "payroll",
        "department": "Finance",
        "full_name":  "Payroll Officer",
    },
    {
        "username":   "sales_head",
        "password":   "sales123",
        "role":       "manager",
        "department": "Sales",
        "full_name":  "Sales Department Head",
    },
    {
        "username":   "tech_head",
        "password":   "tech123",
        "role":       "manager",
        "department": "Research & Development",
        "full_name":  "Tech Department Head",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    """SHA-256 hash — deterministic, no extra deps."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_auth_client():
    """
    Create a raw Supabase client for authentication.
    Independent of SQLEngine — no Gemini required.
    Uses an absolute path to .env so it works regardless of Streamlit's cwd.
    Falls back to st.secrets for Render / Streamlit Cloud.
    Returns the Supabase client, or None on failure.
    """
    # Load .env using the module-level constant — computed at import time, never
    # depends on cwd.  override=True ensures we replace any stale empty strings
    # that may have been placed in the environment before dotenv ran.
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_ENV_FILE, override=True)
    except Exception:
        pass

    # Manual fallback: parse .env ourselves if dotenv didn't populate the vars
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        try:
            with open(_ENV_FILE, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _, _v = _line.partition("=")
                        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        except Exception:
            pass

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    # Streamlit secrets fallback (Render / Streamlit Cloud)
    if not url or not key:
        try:
            import streamlit as st
            url = url or str(st.secrets.get("SUPABASE_URL", "")).strip()
            key = key or str(st.secrets.get("SUPABASE_KEY", "")).strip()
        except Exception:
            pass

    if not url or not key:
        print(f"[Auth] SUPABASE_URL={'SET' if url else 'MISSING'}, "
              f"SUPABASE_KEY={'SET' if key else 'MISSING'}")
        return None

    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        print(f"[Auth] get_auth_client failed: {e}")
        return None


def role_info(role: str) -> dict:
    """Return the role config dict, defaulting to 'hr' if unknown."""
    return ROLES.get(role, ROLES["hr"])


# ── Core auth functions ───────────────────────────────────────────────────────

def authenticate(sb, username: str, password: str) -> Optional[dict]:
    """
    Verify credentials against the app_users table.
    Returns the full user row dict on success, None on failure.
    """
    try:
        hashed = _hash(password)
        rows = (
            sb.table("app_users")
            .select("*")
            .eq("username", username)
            .eq("password_hash", hashed)
            .execute()
        )
        if rows.data:
            user = rows.data[0]
            # Best-effort: stamp last_login (ignore errors)
            try:
                from datetime import datetime, timezone
                sb.table("app_users").update(
                    {"last_login": datetime.now(timezone.utc).isoformat()}
                ).eq("id", user["id"]).execute()
            except Exception:
                pass
            return user
    except Exception as e:
        print(f"[Auth] authenticate error: {e}")
    return None


def seed_users(sb) -> int:
    """
    Insert the default demo users if the table is empty.
    Safe to call on every startup — skips if users already exist.
    Returns the number of users inserted.
    """
    try:
        existing = sb.table("app_users").select("id").limit(1).execute()
        if existing.data:   # at least one user already exists
            return 0
        inserted = 0
        for u in _SEED_USERS:
            try:
                sb.table("app_users").insert({
                    "username":      u["username"],
                    "password_hash": _hash(u["password"]),
                    "role":          u["role"],
                    "department":    u.get("department"),
                    "full_name":     u["full_name"],
                }).execute()
                inserted += 1
            except Exception as e:
                print(f"[Auth] seed: could not insert {u['username']}: {e}")
        return inserted
    except Exception as e:
        print(f"[Auth] seed_users error: {e}")
        return 0


# ── User management (admin only) ──────────────────────────────────────────────

def get_all_users(sb) -> list:
    try:
        return (
            sb.table("app_users")
            .select("id, username, role, department, full_name, last_login")
            .order("id")
            .execute()
            .data
            or []
        )
    except Exception as e:
        print(f"[Auth] get_all_users error: {e}")
        return []


def create_user(sb, username: str, password: str, role: str,
                department: Optional[str], full_name: str) -> tuple[bool, str]:
    """Returns (success, error_message)."""
    if not username.strip():
        return False, "Username cannot be empty"
    if len(password) < 4:
        return False, "Password must be at least 4 characters"
    if role not in ROLES:
        return False, f"Invalid role '{role}'"
    try:
        sb.table("app_users").insert({
            "username":      username.strip(),
            "password_hash": _hash(password),
            "role":          role,
            "department":    department.strip() if department else None,
            "full_name":     full_name.strip(),
        }).execute()
        return True, ""
    except Exception as e:
        msg = str(e)
        if "duplicate" in msg.lower() or "unique" in msg.lower():
            return False, f"Username '{username}' already exists"
        return False, msg


def delete_user(sb, user_id: int) -> bool:
    try:
        sb.table("app_users").delete().eq("id", user_id).execute()
        return True
    except Exception as e:
        print(f"[Auth] delete_user error: {e}")
        return False


def update_password(sb, user_id: int, new_password: str) -> bool:
    try:
        sb.table("app_users").update(
            {"password_hash": _hash(new_password)}
        ).eq("id", user_id).execute()
        return True
    except Exception as e:
        print(f"[Auth] update_password error: {e}")
        return False
