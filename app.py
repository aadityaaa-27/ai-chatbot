import json
import os
import re
from datetime import datetime

import google.generativeai as genai
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from memory_manager import MemoryManager
from rag_engine import RAGEngine, _secret
from sql_engine import SQLEngine

load_dotenv()
GEMINI_API_KEY = _secret("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Base styles ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

html, body, .stApp { background: #0d0d0d !important; }
.block-container    { padding: 0 !important; max-width: 100% !important; }

/* Remove column gap */
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── Left panel (fixed via JS below) ── */
#left-panel-inner {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 1rem 0.85rem;
    box-sizing: border-box;
}

/* Session history cards */
.scard {
    background: #181818;
    border: 1px solid #252525;
    border-radius: 7px;
    padding: 8px 10px 6px 10px;
    margin: 3px 0;
    cursor: pointer;
    transition: border-color .15s;
}
.scard:hover { border-color: #383838; }
.scard.active { border-color: #1a73e8; background: #0f1f3d; }
.scard-title { font-size: 0.82rem; font-weight: 600; color: #ddd;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scard-meta  { font-size: 0.65rem; color: #444; margin-top: 2px; }

/* Fact chips */
.chip {
    display: inline-block; background: #1a56db; color: #fff !important;
    padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; margin: 2px 1px;
}

/* Section label */
.sec {
    font-size: 0.64rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #3a3a3a; margin: 0.7rem 0 0.25rem 0;
}

/* Welcome */
.welcome {
    height: 65vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center;
}
.welcome h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
.welcome p  { color: #555; font-size: 0.9rem; margin-top: 6px; }

/* Tighten metrics */
[data-testid="stMetric"]      { padding: 0 !important; }
[data-testid="stMetricValue"] { font-size: 1rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.68rem !important; }

hr { border-color: #1e1e1e !important; margin: 0.4rem 0 !important; }

/* ── Form-based chat input ── */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
.chat-form-wrap {
    position: sticky;
    bottom: 0;
    background: #0d0d0d;
    padding: 0.6rem 1rem 0.8rem 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    border-top: 1px solid #1a1a1a;
}
/* Text input inside form */
.chat-form-wrap [data-testid="stTextInput"] {
    flex: 1;
}
.chat-form-wrap [data-testid="stTextInput"] > div {
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    background: #161616 !important;
    box-shadow: none !important;
}
.chat-form-wrap [data-testid="stTextInput"] > div:focus-within {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 2px rgba(26,115,232,0.15) !important;
}
.chat-form-wrap [data-testid="stTextInput"] input {
    height: 44px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 0.92rem !important;
    color: #e0e0e0 !important;
    padding: 0 1rem !important;
}
/* Send button */
.chat-form-wrap [data-testid="stFormSubmitButton"] button {
    height: 44px !important;
    width: 44px !important;
    min-width: 44px !important;
    background: #1a73e8 !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 1.1rem !important;
    padding: 0 !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# ── JS: fix left column + compact chat input ─────────────────────────────────
components.html("""
<script>
(function init() {
    var doc = window.parent.document;

    /* ── 1. Fix left column position ── */
    function fixLayout() {
        var blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
        if (!blocks.length) { setTimeout(fixLayout, 80); return; }
        var left  = blocks[0].querySelector(':scope > div:first-child');
        var right = blocks[0].querySelector(':scope > div:last-child');
        if (!left || !right) { setTimeout(fixLayout, 80); return; }
        left.style.cssText = [
            'position:fixed','top:0','left:0','height:100vh','width:24vw',
            'background:#111','border-right:1px solid #1e1e1e',
            'overflow-y:auto','z-index:200','box-sizing:border-box'
        ].join(';');
        right.style.cssText = 'margin-left:24vw;width:76vw;max-width:76vw';
    }
    fixLayout();

})();
</script>
""", height=0)


# ── Gemini helpers ────────────────────────────────────────────────────────────

def build_system_prompt(memory_manager: MemoryManager, rag_context: str = "") -> str:
    base = (
        "You are a smart, helpful, friendly AI assistant with deep knowledge across "
        "all domains: science, math, technology, programming, history, literature, "
        "philosophy, arts, health, business, and everyday topics.\n\n"
        "- Be conversational and warm; use the user's name when you know it.\n"
        "- Give accurate, well-structured answers with markdown when helpful.\n"
        "- Be honest about uncertainty — never fabricate facts.\n"
        "- When asked about past conversations, refer to the chat history."
    )
    if rag_context:
        base += f"\n\n{rag_context}\n\nAlways prioritise the company data above. Cite specific values when answering."
    ctx = memory_manager.get_memory_context()
    if ctx:
        base += f"\n\nKNOWN FACTS ABOUT THIS USER:\n{ctx}\nAddress them by name naturally."
    return base


def make_model(system_prompt: str) -> genai.GenerativeModel:
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.85, top_p=0.95, max_output_tokens=8192,
        ),
    )


def generate_title(model: genai.GenerativeModel, first_msg: str) -> str:
    try:
        r = model.generate_content(
            f'Give a 3-5 word title for a chat that starts with: "{first_msg[:200]}"\n'
            'Reply with ONLY the title. No quotes, no punctuation at the end.'
        )
        return r.text.strip()[:55]
    except Exception:
        return first_msg[:45]


def ai_extract_facts(model: genai.GenerativeModel, msg: str) -> dict:
    try:
        r = model.generate_content(
            f'Extract personal facts explicitly stated in: "{msg}"\n'
            'Return ONLY a flat JSON object (name, age, location, occupation, etc.).'
            ' If none, return {}.'
        )
        text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
        text = re.sub(r"\s*```$", "", text)
        facts = json.loads(text)
        return {k: v for k, v in facts.items()
                if v and str(v).lower() not in ("not mentioned", "unknown", "none", "n/a")}
    except Exception:
        return {}


def chat(memory_manager: MemoryManager, user_input: str, session_id: str,
         rag: RAGEngine = None, sql: SQLEngine = None):
    # 1. SQL engine — structured employee queries (runs first, most precise)
    sql_ctx = sql.format_context(user_input) if (sql and sql.ready) else ""
    # 2. RAG — semantic search over company docs (fallback / supplement)
    rag_ctx = rag.get_context(user_input) if (rag and rag.ready and not sql_ctx) else ""
    combined = "\n\n".join(filter(None, [sql_ctx, rag_ctx]))
    prompt  = build_system_prompt(memory_manager, combined)
    model   = make_model(prompt)
    hist    = memory_manager.get_history_for_gemini(session_id=session_id)
    return model.start_chat(history=hist).send_message(user_input).text, model


# ── Left panel ────────────────────────────────────────────────────────────────

def render_left(mm: MemoryManager):
    st.markdown("### 🤖 AI Chatbot")
    rag: RAGEngine = st.session_state.get("rag")
    sql: SQLEngine = st.session_state.get("sql")
    emp_n = sql.employee_count() if (sql and sql.ready) else 0
    rag_n = rag.record_count()   if (rag and rag.ready) else 0
    if emp_n:
        st.caption(f"Gemini 2.0 Flash · 🟢 {emp_n:,} Employees · Memory")
    elif rag_n:
        st.caption(f"Gemini 2.0 Flash · 🟢 Company DB ({rag_n:,}) · Memory")
    else:
        st.caption("Gemini 2.0 Flash · Memory enabled")

    if st.button("✏️  New Chat", type="primary", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.current_sid    = None
        st.session_state.title_generated = False
        st.rerun()

    st.divider()

    # ── Session History ───────────────────────────────────────────────────────
    st.markdown('<div class="sec">💬 Chats</div>', unsafe_allow_html=True)

    sessions = mm.get_sessions()
    if not sessions:
        st.caption("No chats yet.")
    else:
        cur_sid = st.session_state.get("current_sid")
        with st.container(height=280):
            for s in sessions:
                is_active = s["id"] == cur_sid
                card_cls  = "scard active" if is_active else "scard"
                created   = ""
                try:
                    created = datetime.fromisoformat(s["created_at"]).strftime("%d %b")
                except Exception:
                    pass
                meta = f"{created} · {s['msg_count']} msgs"
                st.markdown(
                    f'<div class="{card_cls}">'
                    f'<div class="scard-title">{s["title"]}</div>'
                    f'<div class="scard-meta">{meta}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                btn_load, btn_del = st.columns([3, 1])
                with btn_load:
                    if st.button("Load", key=f"load_{s['id']}", use_container_width=True):
                        msgs = mm.get_session_messages(s["id"])
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]} for m in msgs
                        ]
                        st.session_state.current_sid     = s["id"]
                        st.session_state.title_generated = True
                        st.rerun()
                with btn_del:
                    if st.button("🗑️", key=f"del_{s['id']}", use_container_width=True, help="Delete this chat"):
                        mm.delete_session(s["id"])
                        if st.session_state.get("current_sid") == s["id"]:
                            st.session_state.messages    = []
                            st.session_state.current_sid = None
                            st.session_state.title_generated = False
                        st.rerun()

    st.divider()

    # ── Memory ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🧠 Memory</div>', unsafe_allow_html=True)
    stats = mm.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Messages", stats["total_messages"])
    c2.metric("Facts",    stats["facts_count"])
    if stats["known_facts"]:
        st.markdown(
            "".join(f'<span class="chip">{k}: {v}</span>'
                    for k, v in stats["known_facts"].items()),
            unsafe_allow_html=True,
        )
    else:
        st.caption("Tell me your name, job, location…")

    st.divider()

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">⚙️ Controls</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat, keep memory"):
            mm.clear_history()
            st.session_state.messages    = []
            st.session_state.current_sid = None
            st.session_state.title_generated = False
            st.rerun()
    with cb:
        if st.button("🧹 Reset", use_container_width=True, help="Clear everything"):
            mm.clear_all()
            st.session_state.messages    = []
            st.session_state.current_sid = None
            st.session_state.title_generated = False
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine()
    if "sql" not in st.session_state:
        st.session_state.sql = SQLEngine()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_sid" not in st.session_state:
        st.session_state.current_sid = None
    if "title_generated" not in st.session_state:
        st.session_state.title_generated = False

    mm: MemoryManager = st.session_state.memory_manager

    left_col, right_col = st.columns([1, 3], gap="small")

    with left_col:
        render_left(mm)

    with right_col:
        if not st.session_state.messages:
            st.markdown(
                '<div class="welcome"><h1>🤖 AI Chatbot</h1>'
                '<p>Ask me anything — I remember everything you share, across all sessions.</p></div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Compact form input ────────────────────────────────────────────────
        st.markdown('<div class="chat-form-wrap">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True, border=False):
            col_in, col_btn = st.columns([12, 1])
            with col_in:
                user_input = st.text_input(
                    "msg", placeholder="Ask me anything…",
                    label_visibility="collapsed", key="chat_text",
                )
            with col_btn:
                submitted = st.form_submit_button("↑", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not (submitted and user_input.strip()):
        return
    user_input = user_input.strip()

    # Create session on first message
    if not st.session_state.current_sid:
        st.session_state.current_sid     = mm.new_session()
        st.session_state.title_generated = False

    sid = st.session_state.current_sid

    with right_col:
        with st.chat_message("user"):
            st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    mm.add_message("user", user_input, session_id=sid)

    pf = mm.extract_facts_from_message(user_input)
    if pf:
        mm.update_facts(pf)

    with right_col:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response_text, model = chat(
                        mm, user_input, sid,
                        rag=st.session_state.rag,
                        sql=st.session_state.sql,
                    )
                    st.markdown(response_text)
                    mm.add_message("assistant", response_text, session_id=sid)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                    # Generate session title from first message
                    if not st.session_state.title_generated:
                        title = generate_title(model, user_input)
                        mm.set_session_title(sid, title)
                        st.session_state.title_generated = True

                    # AI fact extraction (non-blocking)
                    if not pf:
                        try:
                            af = ai_extract_facts(model, user_input)
                            if af:
                                mm.update_facts(af)
                        except Exception:
                            pass

                except Exception as e:
                    err = str(e)
                    if "API_KEY_INVALID" in err or "api key" in err.lower():
                        st.error("Invalid API key — check your .env file.")
                    elif "quota" in err.lower() or "429" in err:
                        st.error("API quota exceeded. Try again in a moment.")
                    elif "safety" in err.lower():
                        st.warning("Message blocked by safety filters.")
                    else:
                        st.error(f"Error: {err}")
                    return

    st.rerun()


if __name__ == "__main__":
    main()
