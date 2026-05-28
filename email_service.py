"""
Email Service — OTP delivery via Resend API.
Reads RESEND_API_KEY and SMTP_EMAIL (used as 'from' address) from .env.
"""

import os
import random
import hashlib
import urllib.request
import urllib.error
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

_DIR = Path(__file__).resolve().parent
_ENV = _DIR / ".env"


def _load_env():
    if not os.environ.get("RESEND_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=_ENV, override=True)
        except Exception:
            pass
    if not os.environ.get("RESEND_API_KEY"):
        try:
            with open(_ENV, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


def _hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"


def send_otp_email(to_email: str, otp: str, company_name: str) -> tuple[bool, str]:
    """Send OTP via Resend API. Returns (success, error_message)."""
    _load_env()
    api_key    = os.environ.get("RESEND_API_KEY", "").strip()
    from_email = os.environ.get("SMTP_EMAIL", "").strip()

    if not api_key:
        return False, "RESEND_API_KEY not set in .env"
    if not from_email:
        return False, "SMTP_EMAIL (sender address) not set in .env"

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;
                background:#0f0f1a;border-radius:12px;color:#e2e8f0;">
      <h2 style="background:linear-gradient(135deg,#6eb6ff,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin:0 0 8px;">HR Analytics Platform</h2>
      <p style="color:#94a3b8;margin:0 0 24px;">Company registration verification</p>
      <p>Hi there,</p>
      <p>You requested to register <strong>{company_name}</strong> on the HR Analytics
         platform. Use the code below to complete your registration:</p>
      <div style="background:#1e1e2e;border-radius:10px;padding:24px;text-align:center;
                  margin:24px 0;border:1px solid #2a2a3d;">
        <span style="font-size:40px;font-weight:700;letter-spacing:12px;
                     color:#6eb6ff;">{otp}</span>
      </div>
      <p style="color:#94a3b8;font-size:13px;">
        This code expires in <strong>10 minutes</strong>.<br>
        If you didn't request this, ignore this email.
      </p>
    </div>
    """

    payload = json.dumps({
        "from":    f"HR Platform <{from_email}>",
        "to":      [to_email],
        "subject": f"Your verification code: {otp}",
        "html":    html,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                return True, ""
            body = resp.read().decode()
            return False, f"Resend API error {resp.status}: {body}"
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            msg = json.loads(body).get("message", body)
        except Exception:
            msg = body
        if e.code == 401:
            return False, "Invalid RESEND_API_KEY — check your key at resend.com"
        if e.code == 422:
            return False, f"Resend rejected the request: {msg}"
        return False, f"Resend API error {e.code}: {msg}"
    except Exception as e:
        return False, str(e)


# ── OTP DB helpers ────────────────────────────────────────────────────────────

def store_otp(sb, email: str, otp: str, company_name: str,
              admin_name: str) -> tuple[bool, str]:
    """Store hashed OTP in otp_requests table."""
    try:
        safe_email = email.replace("'", "''")
        sb.rpc("run_employee_write", {"query_sql":
            f"UPDATE otp_requests SET used = TRUE "
            f"WHERE email = '{safe_email}' AND used = FALSE"
        }).execute()

        expires      = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        safe_company = company_name.replace("'", "''")
        safe_name    = admin_name.replace("'", "''")
        otp_h        = _hash_otp(otp)

        sb.rpc("run_employee_write", {"query_sql":
            f"INSERT INTO otp_requests (email, otp_hash, company_name, admin_name, expires_at) "
            f"VALUES ('{safe_email}', '{otp_h}', '{safe_company}', '{safe_name}', '{expires}')"
        }).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


def verify_otp(sb, email: str, otp: str) -> tuple[bool, str, dict]:
    """Verify OTP, mark as used on success."""
    try:
        safe_email = email.replace("'", "''")
        otp_h      = _hash_otp(otp)
        res = sb.rpc("run_employee_query", {"query_sql":
            f"SELECT id, company_name, admin_name, expires_at, used "
            f"FROM otp_requests "
            f"WHERE email = '{safe_email}' AND otp_hash = '{otp_h}' "
            f"ORDER BY created_at DESC LIMIT 1"
        }).execute()

        if not res.data:
            return False, "Invalid code — please check and try again.", {}

        row = res.data[0]
        if row["used"]:
            return False, "This code has already been used. Request a new one.", {}

        expiry = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expiry:
            return False, "Code has expired. Please request a new one.", {}

        sb.rpc("run_employee_write", {"query_sql":
            f"UPDATE otp_requests SET used = TRUE WHERE id = {row['id']}"
        }).execute()

        return True, "", {
            "company_name": row["company_name"],
            "admin_name":   row["admin_name"],
            "email":        email,
        }
    except Exception as e:
        return False, str(e), {}
