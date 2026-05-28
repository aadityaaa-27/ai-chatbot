"""
Email Service — OTP delivery via Outlook / Office 365 SMTP.
Reads SMTP_EMAIL and SMTP_PASSWORD from environment / .env.
"""

import os
import random
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime, timezone, timedelta

_DIR = Path(__file__).resolve().parent
_ENV = _DIR / ".env"


def _load_env():
    if not os.environ.get("SMTP_EMAIL"):
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=_ENV, override=True)
        except Exception:
            pass
    if not os.environ.get("SMTP_EMAIL"):
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
    """
    Send OTP via Gmail SMTP.
    Returns (success, error_message).
    """
    _load_env()
    smtp_email = os.environ.get("SMTP_EMAIL", "").strip()
    smtp_pass  = os.environ.get("SMTP_PASSWORD", "").strip()

    if not smtp_email or not smtp_pass:
        return False, "SMTP_EMAIL or SMTP_PASSWORD not configured in .env"

    subject = f"Your verification code for HR Analytics — {otp}"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;
                background:#0f0f1a;border-radius:12px;color:#e2e8f0;">
      <h2 style="background:linear-gradient(135deg,#6eb6ff,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin:0 0 8px;">HR Analytics Platform</h2>
      <p style="color:#94a3b8;margin:0 0 24px;">Company registration verification</p>

      <p>Hi there,</p>
      <p>You requested to register <strong>{company_name}</strong> on the HR Analytics platform.
         Use the code below to complete your registration:</p>

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

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_email
        msg["To"]      = to_email
        msg.attach(MIMEText(html, "html"))

        # Outlook / Office 365 — STARTTLS on port 587
        with smtplib.SMTP("smtp.office365.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_email, smtp_pass)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, (
            "Outlook authentication failed — check SMTP_EMAIL and SMTP_PASSWORD. "
            "If your org uses MFA, ask IT to enable SMTP AUTH for your account, "
            "or generate an app password in your Microsoft account security settings."
        )
    except Exception as e:
        return False, str(e)


# ── OTP DB helpers ────────────────────────────────────────────────────────────

def store_otp(sb, email: str, otp: str, company_name: str,
              admin_name: str) -> tuple[bool, str]:
    """Store hashed OTP in otp_requests table. Returns (success, error)."""
    try:
        # Invalidate any previous unused OTPs for this email
        sb.rpc("run_employee_write", {"query_sql":
            f"UPDATE otp_requests SET used = TRUE "
            f"WHERE email = '{email.replace(chr(39), chr(39)*2)}' AND used = FALSE"
        }).execute()

        expires = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        safe_email   = email.replace("'", "''")
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
    """
    Check OTP. Returns (valid, error_message, request_data).
    On success marks the OTP as used.
    """
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
            return False, "Invalid OTP — please check the code and try again.", {}

        row = res.data[0]
        if row["used"]:
            return False, "This OTP has already been used. Request a new one.", {}

        expiry = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expiry:
            return False, "OTP has expired. Please request a new code.", {}

        # Mark used
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
