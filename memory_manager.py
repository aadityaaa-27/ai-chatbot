import json
import re
import uuid
from pathlib import Path
from datetime import datetime

DATA_DIR     = Path("data")
MEMORY_FILE  = DATA_DIR / "user_memory.json"
HISTORY_FILE = DATA_DIR / "chat_history.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"


class MemoryManager:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        self.memory   = self._load(MEMORY_FILE,  {"user_facts": {}})
        self.history  = self._load(HISTORY_FILE, [])
        self.sessions = self._load(SESSIONS_FILE, {})

    def _load(self, path: Path, default):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save(self, path: Path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_memory(self):   self._save(MEMORY_FILE,   self.memory)
    def save_history(self):  self._save(HISTORY_FILE,  self.history)
    def save_sessions(self): self._save(SESSIONS_FILE, self.sessions)

    # ── Sessions ──────────────────────────────────────────────────────────────

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:8]
        self.sessions[sid] = {
            "id":         sid,
            "title":      "New Chat",
            "created_at": datetime.now().isoformat(),
            "msg_count":  0,
        }
        self.save_sessions()
        return sid

    def set_session_title(self, sid: str, title: str):
        if sid in self.sessions:
            self.sessions[sid]["title"] = title
            self.save_sessions()

    def get_sessions(self) -> list:
        """All sessions newest-first."""
        return sorted(self.sessions.values(), key=lambda s: s["created_at"], reverse=True)

    def get_session_messages(self, sid: str) -> list:
        return [m for m in self.history if m.get("session_id") == sid]

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, session_id: str = ""):
        self.history.append({
            "role":       role,
            "content":    content,
            "timestamp":  datetime.now().isoformat(),
            "session_id": session_id,
        })
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["msg_count"] += 1
            self.save_sessions()
        self.save_history()

    def get_history_for_gemini(self, session_id: str = "", max_messages: int = 60) -> list:
        """
        Return alternating user/model history for the Gemini chat API.
        Excludes the last message (just-added user message sent via send_message).
        If session_id given, only use messages from that session.
        """
        if session_id:
            raw = [m for m in self.history if m.get("session_id") == session_id]
        else:
            raw = self.history
        raw = raw[-(max_messages + 1):-1]
        result, expected = [], "user"
        for msg in raw:
            role = "user" if msg["role"] == "user" else "model"
            if role == expected:
                result.append({"role": role, "parts": [{"text": msg["content"]}]})
                expected = "model" if expected == "user" else "user"
        return result

    # ── Fact extraction ───────────────────────────────────────────────────────

    def extract_facts_from_message(self, message: str) -> dict:
        facts = {}
        msg   = message.lower().strip()
        _skip = {
            "a", "an", "the", "not", "going", "trying", "just", "here",
            "back", "new", "good", "fine", "ok", "okay", "ready", "happy",
            "glad", "sure", "sorry", "right", "wrong", "interested", "learning",
            "student", "developer", "using", "working", "looking",
        }
        for pat in [
            r"my name is ([a-zA-Z][a-zA-Z\s]{1,30}?)(?:\.|,|!|\?|$)",
            r"call me ([a-zA-Z][a-zA-Z\s]{1,20}?)(?:\.|,|!|\?|$)",
            r"name's ([a-zA-Z][a-zA-Z\s]{1,20}?)(?:\.|,|!|\?|$)",
        ]:
            m = re.search(pat, msg)
            if m:
                name = m.group(1).strip().title()
                if name.lower().split()[0] not in _skip and len(name) > 1:
                    facts["name"] = name
                    break

        m = re.search(r"i(?:'m| am) (\d{1,2}) years old|my age is (\d{1,2})", msg)
        if m:
            facts["age"] = m.group(1) or m.group(2)

        m = re.search(r"i (?:live|stay|am) (?:in|from) ([a-zA-Z\s,]+?)(?:\.|,|!|\?|$)", msg)
        if m:
            facts["location"] = m.group(1).strip().title()

        m = re.search(
            r"i(?:'m| am) (?:a |an )([a-zA-Z\s]{3,30}?)(?:\.|,|!|\?|$)|"
            r"i work as (?:a |an )([a-zA-Z\s]{3,30}?)(?:\.|,|!|\?|$)", msg)
        if m:
            job = (m.group(1) or m.group(2) or "").strip()
            if job and job.split()[0] not in _skip and len(job) > 2:
                facts["occupation"] = job.title()

        return facts

    def update_facts(self, facts: dict):
        if facts:
            self.memory["user_facts"].update(facts)
            self.save_memory()

    def get_memory_context(self) -> str:
        if not self.memory["user_facts"]:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in self.memory["user_facts"].items())

    # ── Clear ─────────────────────────────────────────────────────────────────

    def delete_session(self, sid: str):
        """Remove a session and all its messages."""
        self.sessions.pop(sid, None)
        self.history = [m for m in self.history if m.get("session_id") != sid]
        self.save_sessions()
        self.save_history()

    def clear_history(self):
        self.history  = []
        self.sessions = {}
        self.save_history()
        self.save_sessions()

    def clear_all(self):
        self.history  = []
        self.sessions = {}
        self.memory   = {"user_facts": {}}
        self.save_history()
        self.save_sessions()
        self.save_memory()

    def get_stats(self) -> dict:
        return {
            "total_messages": len(self.history),
            "facts_count":    len(self.memory["user_facts"]),
            "known_facts":    dict(self.memory["user_facts"]),
        }
