"""
Authentication & Role-Based Access Control — Multi-Tenant
=========================================================
Roles (high → low privilege):
  super_admin — platform owner; manages companies, sees nothing by default
                (no data access — only company/user management)
  admin       — company admin; manages users within their own company only
  hr          — all data + upload within their company
  payroll     — restricted to Finance_Payroll dataset within their company
  manager     — all datasets, filtered to their own department

Every user belongs to a company (company_id).  All data queries are
hard-scoped to the logged-in user's company — no cross-company leakage.

Passwords are SHA-256 hashed (no extra dependencies).
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

_AUTH_DIR = Path(__file__).resolve().parent
_ENV_FILE  = _AUTH_DIR / ".env"

# ── Role definitions ──────────────────────────────────────────────────────────

ROLES: dict[str, dict] = {
    "super_admin": {
        "label":          "Platform Admin",
        "icon":           "🌐",
        "tabs":           ["admin"],          # only the admin panel
        "can_see_salary": False,
        "datasets":       None,
        "dept_filter":    False,
    },
    "admin": {
        "label":          "Company Administrator",
        "icon":           "👑",
        "tabs":           ["chat", "analytics", "upload", "admin"],
        "can_see_salary": True,
        "datasets":       None,
        "dept_filter":    False,
    },
    "hr": {
        "label":          "HR Manager",
        "icon":           "🧑‍💼",
        "tabs":           ["chat", "analytics", "upload"],
        "can_see_salary": True,
        "datasets":       None,
        "dept_filter":    False,
    },
    "payroll": {
        "label":          "Payroll Department",
        "icon":           "💰",
        "tabs":           ["chat", "analytics"],
        "can_see_salary": True,
        "datasets":       ["Finance_Payroll_2024"],
        "dept_filter":    False,
    },
    "manager": {
        "label":          "Department Manager",
        "icon":           "🏢",
        "tabs":           ["chat", "analytics"],
        "can_see_salary": False,
        "datasets":       None,
        "dept_filter":    True,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_auth_client():
    """
    Supabase client for auth — independent of SQLEngine, no Gemini needed.
    Loads .env by absolute path so it works regardless of Streamlit's cwd.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_ENV_FILE, override=True)
    except Exception:
        pass

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
    return ROLES.get(role, ROLES["hr"])


# ── Core auth ─────────────────────────────────────────────────────────────────

def authenticate(sb, username: str, password: str) -> Optional[dict]:
    """
    Verify credentials.  Returns full user row (including company_id) on
    success, None on failure.
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


def ensure_super_admin(sb) -> bool:
    """
    Create the platform super_admin account if it doesn't exist.
    Credentials come from SUPER_ADMIN_USER / SUPER_ADMIN_PASS env vars.
    Falls back to 'superadmin' / a random hex token printed to the console.
    Returns True if the account already existed or was just created.
    """
    try:
        existing = (
            sb.table("app_users")
            .select("id")
            .eq("role", "super_admin")
            .limit(1)
            .execute()
        )
        if existing.data:
            return True

        username = os.environ.get("SUPER_ADMIN_USER", "superadmin").strip()
        password = os.environ.get("SUPER_ADMIN_PASS", "").strip()
        if not password:
            import secrets
            password = secrets.token_hex(12)
            print(f"\n[Auth] *** SUPER ADMIN CREATED ***")
            print(f"[Auth]   username : {username}")
            print(f"[Auth]   password : {password}")
            print(f"[Auth] Save these — they will NOT be shown again.\n")

        sb.table("app_users").insert({
            "username":      username,
            "password_hash": _hash(password),
            "role":          "super_admin",
            "full_name":     "Platform Administrator",
            "company_id":    None,
        }).execute()
        return True
    except Exception as e:
        print(f"[Auth] ensure_super_admin error: {e}")
        return False


# ── Company management (super_admin only) ─────────────────────────────────────

def get_all_companies(sb) -> list:
    try:
        res = sb.rpc("run_employee_query", {
            "query_sql": "SELECT id, name, slug, created_at FROM companies ORDER BY id"
        }).execute()
        return res.data or []
    except Exception as e:
        print(f"[Auth] get_all_companies error: {e}")
        return []


def create_company(sb, name: str, slug: str) -> tuple[bool, str, Optional[int]]:
    """Returns (success, error_message, new_company_id)."""
    if not name.strip() or not slug.strip():
        return False, "Name and slug cannot be empty", None
    safe_name = name.strip().replace("'", "''")
    safe_slug = slug.strip().lower().replace(" ", "-").replace("'", "")
    try:
        sb.rpc("run_employee_write", {"query_sql":
            f"INSERT INTO companies (name, slug) VALUES ('{safe_name}', '{safe_slug}')"
        }).execute()
        res = sb.rpc("run_employee_query", {"query_sql":
            f"SELECT id FROM companies WHERE slug = '{safe_slug}' LIMIT 1"
        }).execute()
        company_id = res.data[0]["id"] if res.data else None
        return True, "", company_id
    except Exception as e:
        msg = str(e)
        if "unique" in msg.lower() or "duplicate" in msg.lower():
            return False, f"Company slug '{safe_slug}' already exists", None
        return False, msg, None


def delete_company(sb, company_id: int) -> tuple[bool, str]:
    """Delete a company and all its users (employees data is NOT deleted)."""
    try:
        sb.rpc("run_employee_write", {"query_sql":
            f"DELETE FROM app_users WHERE company_id = {int(company_id)}"
        }).execute()
        sb.rpc("run_employee_write", {"query_sql":
            f"DELETE FROM companies WHERE id = {int(company_id)}"
        }).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


# ── User management ───────────────────────────────────────────────────────────

def get_all_users(sb, company_id: Optional[int] = None) -> list:
    """
    super_admin passes company_id to scope to one company.
    company admin passes their own company_id (enforced by caller).
    """
    try:
        q = sb.table("app_users").select(
            "id, username, role, department, full_name, last_login, company_id"
        )
        if company_id is not None:
            q = q.eq("company_id", company_id)
        return q.order("id").execute().data or []
    except Exception as e:
        print(f"[Auth] get_all_users error: {e}")
        return []


def create_user(sb, username: str, password: str, role: str,
                department: Optional[str], full_name: str,
                company_id: int, email: str = "") -> tuple[bool, str]:
    """Returns (success, error_message)."""
    if not username.strip():
        return False, "Username cannot be empty"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if role not in ROLES:
        return False, f"Invalid role '{role}'"
    if role == "super_admin":
        return False, "Cannot create super_admin via this form"
    try:
        row = {
            "username":      username.strip(),
            "password_hash": _hash(password),
            "role":          role,
            "department":    department.strip() if department else None,
            "full_name":     full_name.strip(),
            "company_id":    company_id,
        }
        if email:
            row["email"] = email.strip().lower()
        sb.table("app_users").insert(row).execute()
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


def update_password(sb, user_id: int, new_password: str) -> tuple[bool, str]:
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters"
    try:
        sb.table("app_users").update(
            {"password_hash": _hash(new_password)}
        ).eq("id", user_id).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Invite link system ────────────────────────────────────────────────────────

def create_invite_token(sb, company_name: str,
                        hours: int = 48) -> tuple[bool, str, str]:
    """
    Generate a one-time invite token for a new company.
    Returns (success, error_message, token).
    """
    if not company_name.strip():
        return False, "Company name cannot be empty", ""
    try:
        import secrets
        from datetime import datetime, timezone, timedelta
        token      = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
        safe_name  = company_name.strip().replace("'", "''")
        sb.rpc("run_employee_write", {"query_sql":
            f"INSERT INTO invite_tokens (token, company_name, expires_at) "
            f"VALUES ('{token}', '{safe_name}', '{expires_at}')"
        }).execute()
        return True, "", token
    except Exception as e:
        return False, str(e), ""


def get_invite_token(sb, token: str) -> Optional[dict]:
    """
    Look up a token. Returns the row dict if valid & unused, else None.
    """
    try:
        from datetime import datetime, timezone
        safe_token = token.replace("'", "''")
        res = sb.rpc("run_employee_query", {"query_sql":
            f"SELECT id, company_name, used, expires_at "
            f"FROM invite_tokens WHERE token = '{safe_token}' LIMIT 1"
        }).execute()
        if not res.data:
            return None
        row = res.data[0]
        if row["used"]:
            return None
        expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires:
            return None
        return row
    except Exception as e:
        print(f"[Auth] get_invite_token error: {e}")
        return None


def use_invite_token(sb, token: str, used_by: str) -> bool:
    """Mark a token as used."""
    try:
        safe_token = token.replace("'", "''")
        safe_user  = used_by.replace("'", "''")
        sb.rpc("run_employee_write", {"query_sql":
            f"UPDATE invite_tokens SET used = TRUE, used_by = '{safe_user}' "
            f"WHERE token = '{safe_token}'"
        }).execute()
        return True
    except Exception as e:
        print(f"[Auth] use_invite_token error: {e}")
        return False


def get_all_invite_tokens(sb) -> list:
    """List all invite tokens (for super_admin panel)."""
    try:
        res = sb.rpc("run_employee_query", {"query_sql":
            "SELECT id, company_name, used, used_by, expires_at, created_at "
            "FROM invite_tokens ORDER BY created_at DESC LIMIT 50"
        }).execute()
        return res.data or []
    except Exception as e:
        print(f"[Auth] get_all_invite_tokens error: {e}")
        return []


def revoke_invite_token(sb, token_id: int) -> bool:
    """Revoke (mark used) an invite by id."""
    try:
        sb.rpc("run_employee_write", {"query_sql":
            f"UPDATE invite_tokens SET used = TRUE WHERE id = {int(token_id)}"
        }).execute()
        return True
    except Exception:
        return False
