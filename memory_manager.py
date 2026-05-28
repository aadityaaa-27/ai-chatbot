import re
import uuid
from datetime import datetime


class MemoryManager:
    """
    Persistent chat memory backed by Supabase.
    Requires setup_memory.sql to have been run once in Supabase.
    Falls back to in-memory-only mode if the tables are missing.
    """

    def __init__(self, user_id: int):
        from rag_engine import _secret
        self._user_id = user_id
        self._sb = None
        self._db_ready = False
        self._sessions: dict = {}
        self._history: list = []
        self._facts: dict = {}

        url = _secret("SUPABASE_URL")
        key = _secret("SUPABASE_KEY")
        if url and key:
            try:
                from supabase import create_client
                self._sb = create_client(url, key)
                self._db_ready = self._check_tables()
            except Exception as e:
                print(f"[Memory] Supabase init failed: {e}")

        if self._db_ready:
            self._load_from_db()

    # ── DB connectivity ───────────────────────────────────────────────────────

    def _check_tables(self) -> bool:
        try:
            self._sb.table("chat_sessions").select("id").limit(1).execute()
            return True
        except Exception:
            print("[Memory] chat_sessions table not found — run setup_memory.sql in Supabase.")
            return False

    def _load_from_db(self):
        try:
            res = self._sb.table("chat_sessions").select(
                "id, title, msg_count, created_at"
            ).eq("user_id", self._user_id).order("created_at", desc=True).limit(100).execute()
            for row in (res.data or []):
                self._sessions[row["id"]] = {
                    "id":         row["id"],
                    "title":      row["title"],
                    "created_at": row["created_at"],
                    "msg_count":  row["msg_count"],
                }
        except Exception as e:
            print(f"[Memory] load sessions error: {e}")

        try:
            res = self._sb.table("chat_messages").select(
                "session_id, role, content, created_at"
            ).eq("user_id", self._user_id).order("created_at").limit(500).execute()
            for row in (res.data or []):
                self._history.append({
                    "role":       row["role"],
                    "content":    row["content"],
                    "timestamp":  row["created_at"],
                    "session_id": row["session_id"],
                })
        except Exception as e:
            print(f"[Memory] load messages error: {e}")

        try:
            res = self._sb.table("user_facts").select(
                "key, value"
            ).eq("user_id", self._user_id).execute()
            for row in (res.data or []):
                self._facts[row["key"]] = row["value"]
        except Exception as e:
            print(f"[Memory] load facts error: {e}")

    # ── Sessions ──────────────────────────────────────────────────────────────

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:8]
        now = datetime.utcnow().isoformat()
        session = {
            "id":         sid,
            "title":      "New Chat",
            "created_at": now,
            "msg_count":  0,
        }
        self._sessions[sid] = session
        if self._db_ready:
            try:
                self._sb.table("chat_sessions").insert({
                    "id":       sid,
                    "user_id":  self._user_id,
                    "title":    "New Chat",
                    "created_at": now,
                }).execute()
            except Exception as e:
                print(f"[Memory] new_session DB error: {e}")
        return sid

    def set_session_title(self, sid: str, title: str):
        if sid in self._sessions:
            self._sessions[sid]["title"] = title
            if self._db_ready:
                try:
                    self._sb.table("chat_sessions").update(
                        {"title": title}
                    ).eq("id", sid).eq("user_id", self._user_id).execute()
                except Exception as e:
                    print(f"[Memory] set_session_title DB error: {e}")

    def get_sessions(self) -> list:
        """All sessions newest-first."""
        return sorted(self._sessions.values(), key=lambda s: s["created_at"], reverse=True)

    def get_session_messages(self, sid: str) -> list:
        return [m for m in self._history if m.get("session_id") == sid]

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, session_id: str = ""):
        now = datetime.utcnow().isoformat()
        msg = {
            "role":       role,
            "content":    content,
            "timestamp":  now,
            "session_id": session_id,
        }
        self._history.append(msg)
        if session_id and session_id in self._sessions:
            self._sessions[session_id]["msg_count"] += 1

        if self._db_ready:
            try:
                self._sb.table("chat_messages").insert({
                    "session_id": session_id or None,
                    "user_id":    self._user_id,
                    "role":       role,
                    "content":    content,
                    "created_at": now,
                }).execute()
            except Exception as e:
                print(f"[Memory] add_message DB error: {e}")
            if session_id and session_id in self._sessions:
                try:
                    self._sb.table("chat_sessions").update({
                        "msg_count": self._sessions[session_id]["msg_count"]
                    }).eq("id", session_id).eq("user_id", self._user_id).execute()
                except Exception as e:
                    print(f"[Memory] msg_count update DB error: {e}")

    def get_history_for_gemini(self, session_id: str = "", max_messages: int = 60) -> list:
        """
        Return alternating user/model history for the Gemini chat API.
        Excludes the last message (just-added user message sent via send_message).
        """
        if session_id:
            raw = [m for m in self._history if m.get("session_id") == session_id]
        else:
            raw = self._history
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
        if not facts:
            return
        self._facts.update(facts)
        if self._db_ready:
            for key, value in facts.items():
                try:
                    self._sb.table("user_facts").upsert({
                        "user_id":    self._user_id,
                        "key":        key,
                        "value":      str(value),
                        "updated_at": datetime.utcnow().isoformat(),
                    }, on_conflict="user_id,key").execute()
                except Exception as e:
                    print(f"[Memory] update_facts DB error: {e}")

    def get_memory_context(self) -> str:
        if not self._facts:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in self._facts.items())

    # ── Clear ─────────────────────────────────────────────────────────────────

    def delete_session(self, sid: str):
        """Remove a session and all its messages."""
        self._sessions.pop(sid, None)
        self._history = [m for m in self._history if m.get("session_id") != sid]
        if self._db_ready:
            try:
                self._sb.table("chat_sessions").delete().eq(
                    "id", sid
                ).eq("user_id", self._user_id).execute()
            except Exception as e:
                print(f"[Memory] delete_session DB error: {e}")

    def clear_history(self):
        self._history  = []
        self._sessions = {}
        if self._db_ready:
            try:
                self._sb.table("chat_sessions").delete().eq(
                    "user_id", self._user_id
                ).execute()
            except Exception as e:
                print(f"[Memory] clear_history DB error: {e}")

    def clear_all(self):
        self._history  = []
        self._sessions = {}
        self._facts    = {}
        if self._db_ready:
            try:
                self._sb.table("chat_sessions").delete().eq(
                    "user_id", self._user_id
                ).execute()
                self._sb.table("user_facts").delete().eq(
                    "user_id", self._user_id
                ).execute()
            except Exception as e:
                print(f"[Memory] clear_all DB error: {e}")

    def get_stats(self) -> dict:
        return {
            "total_messages": len(self._history),
            "facts_count":    len(self._facts),
            "known_facts":    dict(self._facts),
        }
