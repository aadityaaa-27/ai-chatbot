"""
RAG Engine — Supabase pgvector + Gemini text-embedding-004
Searches company data and returns relevant context for the chatbot.
"""
import os
from typing import List, Dict

from google import genai


def _secret(key: str) -> str:
    """
    Read credentials in priority order:
    1. data/db_config.json  (user-set via the in-app Settings panel)
    2. Environment variables / .env
    3. st.secrets (Streamlit Cloud / Render secrets)
    """
    # 1 — in-app saved config (overrides everything)
    try:
        import json, pathlib
        cfg_path = pathlib.Path("data/db_config.json")
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            if cfg.get(key):
                return cfg[key]
    except Exception:
        pass

    # 2 — env vars / .env
    val = os.getenv(key, "")
    if val:
        return val

    # 3 — st.secrets
    try:
        import streamlit as st
        val = str(st.secrets.get(key, ""))
    except Exception:
        pass
    return val


class RAGEngine:
    EMBED_MODEL = "text-embedding-004"

    def __init__(self):
        self._sb     = None
        self._client = None
        self._ready  = False
        url = _secret("SUPABASE_URL")
        key = _secret("SUPABASE_KEY")
        if not url or not key:
            return
        try:
            from supabase import create_client
            self._sb     = create_client(url, key)
            self._client = genai.Client(api_key=_secret("GEMINI_API_KEY"))
            self._ready  = True
        except Exception as e:
            print(f"[RAG] init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self.EMBED_MODEL,
            contents=text[:8000],
        )
        return result.embeddings[0].values

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._ready:
            return []
        try:
            emb = self.embed(query)
            res = self._sb.rpc(
                "search_company_data",
                {
                    "query_embedding": emb,
                    "match_threshold":  0.3,
                    "match_count":      top_k,
                },
            ).execute()
            return res.data or []
        except Exception as e:
            print(f"[RAG] search error: {e}")
            return []

    def get_context(self, query: str) -> str:
        """Return formatted context string to inject into the system prompt."""
        hits = self.search(query)
        if not hits:
            return ""
        lines = [f"• {h['content']}" for h in hits]
        return "RELEVANT COMPANY DATA (use this to answer accurately):\n" + "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def record_count(self) -> int:
        if not self._ready:
            return 0
        try:
            r = self._sb.table("company_data").select("id", count="exact").execute()
            return r.count or 0
        except Exception:
            return 0
