import json
import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types
import streamlit as st
from dotenv import load_dotenv

from memory_manager import MemoryManager
from rag_engine import RAGEngine, _secret
from sql_engine import SQLEngine
from data_processor import DataProcessor

load_dotenv()
GEMINI_API_KEY = _secret("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS (pure CSS only — no JS, fully Render-safe) ────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

html, body, .stApp { background: #0d0d0d !important; }
.block-container    { padding: 0 !important; max-width: 100% !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* Session cards */
.scard {
    background: #181818; border: 1px solid #252525;
    border-radius: 7px; padding: 8px 10px 6px 10px;
    margin: 3px 0;
}
.scard.active { border-color: #1a73e8; background: #0f1f3d; }
.scard-title  { font-size: 0.82rem; font-weight: 600; color: #ddd;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scard-meta   { font-size: 0.65rem; color: #444; margin-top: 2px; }

/* Chips */
.chip {
    display: inline-block; background: #1a56db; color: #fff !important;
    padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; margin: 2px 1px;
}

/* Section label */
.sec {
    font-size: 0.64rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #3a3a3a; margin: 0.7rem 0 0.25rem 0;
}

hr { border-color: #1e1e1e !important; margin: 0.4rem 0 !important; }

/* Metrics */
[data-testid="stMetric"]      { padding: 0 !important; }
[data-testid="stMetricValue"] { font-size: 1rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.68rem !important; }

/* Form input */
[data-testid="stForm"] {
    border: none !important; padding: 0 !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────

_DARK = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#cccccc",
    margin=dict(t=36, b=10, l=10, r=10),
    height=280,
    showlegend=False,
    coloraxis_showscale=False,
)


def make_chart(rows: list):
    """Auto-generate a Plotly chart from SQL result rows. Returns None if not chartable."""
    if not rows or len(rows) < 2:
        return None
    try:
        df = pd.DataFrame(rows)
        cols = df.columns.tolist()
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        txt_cols = [c for c in cols if c not in num_cols]
        if not num_cols or not txt_cols:
            return None
        x_col, y_col = txt_cols[0], num_cols[0]
        # Pie for very small sets (≤ 4 categories, e.g. gender)
        if len(df) <= 4:
            fig = px.pie(
                df, names=x_col, values=y_col,
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
        else:
            df = df.sort_values(y_col, ascending=True).tail(15)
            fig = px.bar(
                df, x=y_col, y=x_col, orientation="h",
                template="plotly_dark",
                color=y_col,
                color_continuous_scale="Blues",
            )
        fig.update_layout(**_DARK)
        return fig
    except Exception:
        return None


def analytics_bar(df: pd.DataFrame, x: str, y: str, title: str, color_scale="Blues"):
    df = df.sort_values(y, ascending=True)
    fig = px.bar(
        df, x=y, y=x, orientation="h",
        title=title, template="plotly_dark",
        color=y, color_continuous_scale=color_scale,
    )
    fig.update_layout(**_DARK)
    return fig


def analytics_pie(df: pd.DataFrame, names: str, values: str, title: str):
    fig = px.pie(
        df, names=names, values=values,
        title=title, template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(**{**_DARK, "showlegend": True})
    return fig


# ── Export helper ─────────────────────────────────────────────────────────────

def export_chat(messages: list) -> str:
    lines = [f"Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*50}\n"]
    for m in messages:
        role = "You" if m["role"] == "user" else "AI"
        lines.append(f"[{role}]\n{m['content']}\n")
    return "\n---\n".join(lines)


# ── Gemini helpers ────────────────────────────────────────────────────────────

def build_system_prompt(memory_manager: MemoryManager, rag_context: str = "") -> str:
    base = (
        "You are a smart, helpful, friendly AI assistant for an enterprise company. "
        "You have been given DIRECT ACCESS to the company's live employee database "
        "(2,940 employee records across 9 departments: Sales, Research & Development, "
        "Human Resources, Finance, Marketing, Information Technology, Operations, Legal, Customer Support). "
        "When asked ANYTHING about employees, headcount, salary, attrition, departments, "
        "or any HR metrics — you MUST answer using the live database results provided to you. "
        "NEVER say you lack access to employee data. You have it.\n\n"
        "KNOWN COMPANY POLICIES (answer these directly without querying the DB):\n"
        "- Every employee receives exactly 20 paid holidays per year.\n\n"
        "You also have broad general knowledge across science, math, technology, programming, "
        "history, literature, philosophy, arts, health, business, and everyday topics.\n\n"
        "- Be conversational and warm; use the user's name when you know it.\n"
        "- Give accurate, well-structured answers with markdown when helpful.\n"
        "- Be honest about uncertainty — never fabricate facts.\n"
        "- When asked about past conversations, refer to the chat history.\n"
        "- For employee/HR questions: ALWAYS give a direct plain-English answer with "
        "the actual numbers. NEVER output code snippets, function calls, or Python examples. "
        "Just answer directly: e.g. 'The highest paid department is X with an average salary of $Y.'"
    )
    if rag_context:
        base += (
            f"\n\n{rag_context}"
            "\n\n⚠️ CRITICAL INSTRUCTION: The query results above are REAL DATA from this "
            "company's own employee database that you have full authorised access to. "
            "You MUST use these exact numbers and facts in your answer. "
            "Do NOT say the data is private or that you cannot access it — "
            "it has already been fetched and is shown above. Cite specific values."
        )
    ctx = memory_manager.get_memory_context()
    if ctx:
        base += f"\n\nKNOWN FACTS ABOUT THIS USER:\n{ctx}\nAddress them by name naturally."
    return base


def _make_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def generate_title(client: genai.Client, first_msg: str) -> str:
    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f'Give a 3-5 word title for a chat that starts with: "{first_msg[:200]}"\n'
                'Reply with ONLY the title. No quotes, no punctuation at the end.'
            ),
        )
        return r.text.strip()[:55]
    except Exception:
        return first_msg[:45]


def ai_extract_facts(client: genai.Client, msg: str) -> dict:
    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f'Extract personal facts explicitly stated in: "{msg}"\n'
                'Return ONLY a flat JSON object (name, age, location, occupation, etc.).'
                ' If none, return {}.'
            ),
        )
        text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
        text = re.sub(r"\s*```$", "", text)
        facts = json.loads(text)
        return {k: v for k, v in facts.items()
                if v and str(v).lower() not in ("not mentioned", "unknown", "none", "n/a")}
    except Exception:
        return {}


def chat_with_ai(memory_manager: MemoryManager, user_input: str, session_id: str,
                 rag: RAGEngine = None, sql: SQLEngine = None):
    """Send a message and return (response_text, client, sql_rows)."""
    sql_ctx, sql_rows = "", []
    rag_ctx = ""

    if sql and sql.ready:
        sql_ctx, sql_rows = sql.query(user_input)
    if rag and rag.ready and not sql_ctx:
        rag_ctx = rag.get_context(user_input)

    combined = "\n\n".join(filter(None, [sql_ctx, rag_ctx]))
    system_prompt = build_system_prompt(memory_manager, combined)
    hist = memory_manager.get_history_for_gemini(session_id=session_id)

    client = _make_client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.85,
        top_p=0.95,
        max_output_tokens=8192,
    )
    chat_session = client.chats.create(
        model="gemini-2.0-flash",
        config=cfg,
        history=hist,
    )
    response = chat_session.send_message(user_input)
    return response.text, client, sql_rows


# ── Left panel ────────────────────────────────────────────────────────────────

def render_left(mm: MemoryManager):
    st.markdown("### 🤖 AI Chatbot")
    sql: SQLEngine = st.session_state.get("sql")
    rag: RAGEngine = st.session_state.get("rag")
    emp_n = sql.employee_count() if (sql and sql.ready) else 0
    rag_n = rag.record_count()   if (rag and rag.ready) else 0
    if emp_n:
        st.caption(f"Gemini 2.0 Flash · 🟢 {emp_n:,} Employees · Memory")
    elif rag_n:
        st.caption(f"Gemini 2.0 Flash · 🟢 Company DB ({rag_n:,}) · Memory")
    else:
        st.caption("Gemini 2.0 Flash · Memory enabled")

    if st.button("✏️  New Chat", type="primary", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.current_sid     = None
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
        with st.container(height=260):
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
                            {"role": m["role"], "content": m["content"], "chart_data": []}
                            for m in msgs
                        ]
                        st.session_state.current_sid     = s["id"]
                        st.session_state.title_generated = True
                        st.rerun()
                with btn_del:
                    if st.button("🗑️", key=f"del_{s['id']}", use_container_width=True):
                        mm.delete_session(s["id"])
                        if st.session_state.get("current_sid") == s["id"]:
                            st.session_state.messages        = []
                            st.session_state.current_sid     = None
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

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📥 Export</div>', unsafe_allow_html=True)
    messages = st.session_state.get("messages", [])
    if messages:
        st.download_button(
            label="⬇️ Download Chat",
            data=export_chat(messages),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.caption("Start a chat to export it.")

    st.divider()

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">⚙️ Controls</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat, keep memory"):
            mm.clear_history()
            st.session_state.messages        = []
            st.session_state.current_sid     = None
            st.session_state.title_generated = False
            st.rerun()
    with cb:
        if st.button("🧹 Reset", use_container_width=True, help="Clear everything"):
            mm.clear_all()
            st.session_state.messages        = []
            st.session_state.current_sid     = None
            st.session_state.title_generated = False
            st.rerun()


# ── Upload tab ────────────────────────────────────────────────────────────────

def render_upload_tab(sql: SQLEngine):
    st.markdown("### 📤 Upload HR Data")
    st.caption(
        "Upload any Excel or CSV file — even with messy, non-standard column names. "
        "The AI will automatically map, clean and import the data into the database."
    )

    if not (sql and sql.ready):
        st.warning("⚠️ Database connection required. Check your Supabase credentials.")
        return

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Supports .csv, .xlsx, .xls — any column names, any format",
    )

    if not uploaded:
        st.markdown("""
        **What this does:**
        - 📋 Reads your Excel/CSV (messy column names are fine)
        - 🤖 AI maps columns: *"Monthly CTC"* → `monthly_income`, *"Designation"* → `job_role`, etc.
        - 🧹 Cleans data: removes ₹/$ symbols, fixes Yes/No fields, standardizes formats
        - 💾 Stores in database — chatbot answers questions from your data instantly

        **Supported columns** *(any of these, in any order, with any name)*:
        `department, job_role, monthly_income, age, gender, attrition, education,
        job_satisfaction, overtime, experience, performance_rating` and more.
        """)
        return

    dp = DataProcessor()

    # ── Sheet picker for Excel ────────────────────────────────────────────────
    try:
        raw_df, sheets = dp.read_file(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    if sheets and len(sheets) > 1:
        chosen_sheet = st.selectbox("📑 Select sheet", sheets)
        uploaded.seek(0)
        raw_df = dp.read_sheet(uploaded, chosen_sheet)

    st.markdown(f"**📊 Raw file preview** — {len(raw_df):,} rows × {len(raw_df.columns)} columns")
    st.dataframe(raw_df.head(5), use_container_width=True)

    # ── AI analysis ───────────────────────────────────────────────────────────
    st.divider()

    if st.button("🔍 Analyze with AI", type="primary", use_container_width=False):
        with st.spinner("🤖 Gemini is analyzing your columns…"):
            try:
                uploaded.seek(0)
                mapping = dp.analyze_columns(raw_df)
                st.session_state.upload_mapping  = mapping
                st.session_state.upload_raw_df   = raw_df
                st.session_state.upload_filename = uploaded.name
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
                return

    if "upload_mapping" not in st.session_state:
        return

    mapping  = st.session_state.upload_mapping
    raw_df   = st.session_state.upload_raw_df
    col_map  = mapping.get("column_map", {})
    notes    = mapping.get("notes", "")
    extras   = mapping.get("extra_columns", [])

    # ── Show mapping results ──────────────────────────────────────────────────
    st.markdown("#### 🗂️ AI Column Mapping")
    if notes:
        st.info(f"💡 **AI Notes:** {notes}")

    mapped   = {k: v for k, v in col_map.items() if v}
    unmapped = {k: v for k, v in col_map.items() if not v}

    if mapped:
        map_rows = [
            {"Your Column": k, "Maps To": f"✅  {v}"}
            for k, v in mapped.items()
        ]
        st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No columns could be mapped. Make sure your file has HR-related data.")
        return

    if unmapped:
        skipped = ", ".join(f"`{k}`" for k in list(unmapped.keys())[:10])
        st.caption(f"⏭️ Skipped (no match): {skipped}")

    # ── Preview cleaned data ──────────────────────────────────────────────────
    st.markdown("#### 🧹 Cleaned Data Preview")
    try:
        clean_df = dp.clean_and_transform(raw_df, mapping)
        st.caption(f"{len(clean_df):,} rows ready to import · {len(clean_df.columns)} standard columns")
        st.dataframe(clean_df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Data cleaning failed: {e}")
        return

    # ── Import options ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💾 Import to Database")

    mode = st.radio(
        "Import mode",
        ["🔄 Replace existing data", "➕ Add to existing data"],
        help="Replace wipes the current table first. Add keeps existing records.",
    )
    import_mode = "replace" if "Replace" in mode else "append"

    if import_mode == "replace":
        st.warning(
            f"⚠️ **Replace mode** will delete all current employees and load "
            f"{len(clean_df):,} new rows from **{st.session_state.upload_filename}**."
        )

    col_imp, _ = st.columns([1, 3])
    with col_imp:
        do_import = st.button("✅ Import Now", type="primary", use_container_width=True)

    if do_import:
        progress = st.progress(0, text="Preparing…")
        try:
            sb = sql._sb
            progress.progress(10, text="Clearing old data…" if import_mode == "replace" else "Connecting…")

            # Clear analytics cache so charts refresh
            st.session_state.pop("analytics_data", None)

            progress.progress(30, text=f"Inserting {len(clean_df):,} rows…")
            result = dp.insert_to_db(clean_df, sb, mode=import_mode)
            progress.progress(100, text="Done!")

            if result["inserted"] > 0:
                st.success(
                    f"✅ **{result['inserted']:,} employees imported** successfully!\n\n"
                    f"Go to the **💬 Chat** tab and ask: *'How many employees are there?'*"
                )
                # Clear mapping from state so next upload starts fresh
                st.session_state.pop("upload_mapping", None)
                st.session_state.pop("upload_raw_df", None)
            else:
                st.error(
                    f"Import failed — 0 rows inserted.\n\n"
                    + ("\n".join(result.get("errors", [])))
                )

        except Exception as e:
            progress.empty()
            st.error(f"Import error: {e}")


# ── Analytics tab ─────────────────────────────────────────────────────────────

def render_analytics(sql: SQLEngine):
    st.markdown("### 📊 HR Analytics Dashboard")
    if not (sql and sql.ready):
        st.warning("⚠️ Analytics require a live Supabase connection.")
        return

    col_refresh, col_space = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.pop("analytics_data", None)

    if "analytics_data" not in st.session_state:
        with st.spinner("Loading analytics data…"):
            st.session_state.analytics_data = sql.get_analytics_data()

    data = st.session_state.analytics_data
    if not data:
        st.error("Could not load analytics data.")
        return

    # ── Row 1: Headcount + Attrition ─────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        rows = data.get("dept_headcount", [])
        if rows:
            df = pd.DataFrame(rows)
            st.plotly_chart(
                analytics_bar(df, "department", "employees",
                              "👥 Employees by Department"),
                use_container_width=True,
            )
    with c2:
        rows = data.get("dept_attrition", [])
        if rows:
            df = pd.DataFrame(rows)
            st.plotly_chart(
                analytics_bar(df, "department", "attrition_pct",
                              "📉 Attrition Rate by Department (%)", "Reds"),
                use_container_width=True,
            )

    # ── Row 2: Salary + Age groups ────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        rows = data.get("dept_salary", [])
        if rows:
            df = pd.DataFrame(rows)
            st.plotly_chart(
                analytics_bar(df, "department", "avg_salary",
                              "💰 Avg Monthly Salary by Department ($)", "Greens"),
                use_container_width=True,
            )
    with c4:
        rows = data.get("age_groups", [])
        if rows:
            df = pd.DataFrame(rows)
            fig = px.bar(
                df, x="age_group", y="employees",
                title="📅 Age Distribution",
                template="plotly_dark",
                color="employees",
                color_continuous_scale="Blues",
            )
            fig.update_layout(**_DARK)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Gender + Job Satisfaction ─────────────────────────────────────
    c5, c6 = st.columns(2)
    with c5:
        rows = data.get("gender", [])
        if rows:
            df = pd.DataFrame(rows)
            st.plotly_chart(
                analytics_pie(df, "gender", "employees", "⚧ Gender Breakdown"),
                use_container_width=True,
            )
    with c6:
        rows = data.get("satisfaction", [])
        if rows:
            df = pd.DataFrame(rows)
            label_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
            df["job_satisfaction"] = df["job_satisfaction"].map(label_map)
            fig = px.bar(
                df, x="job_satisfaction", y="employees",
                title="😊 Job Satisfaction",
                template="plotly_dark",
                color="employees",
                color_continuous_scale="Blues",
            )
            fig.update_layout(**_DARK)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Overtime + Education ───────────────────────────────────────────
    c7, c8 = st.columns(2)
    with c7:
        rows = data.get("overtime", [])
        if rows:
            df = pd.DataFrame(rows)
            st.plotly_chart(
                analytics_pie(df, "overtime", "employees", "⏰ Overtime Distribution"),
                use_container_width=True,
            )
    with c8:
        rows = data.get("education", [])
        if rows:
            df = pd.DataFrame(rows)
            edu_map = {1: "Below College", 2: "College", 3: "Bachelor",
                       4: "Master", 5: "Doctor"}
            df["education"] = df["education"].map(edu_map)
            fig = px.bar(
                df, x="education", y="employees",
                title="🎓 Education Level",
                template="plotly_dark",
                color="employees",
                color_continuous_scale="Purples",
            )
            fig.update_layout(**_DARK)
            st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

QUICK_PROMPTS = [
    "How many employees are there?",
    "Which department has the highest salary?",
    "What is the overall attrition rate?",
    "Show employees by department",
    "How many employees work overtime?",
    "What's the average employee age?",
]


def main():
    # ── Init session state ────────────────────────────────────────────────────
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

    mm:  MemoryManager = st.session_state.memory_manager
    sql: SQLEngine     = st.session_state.sql
    rag: RAGEngine     = st.session_state.rag

    left_col, right_col = st.columns([1, 3], gap="small")

    with left_col:
        render_left(mm)

    with right_col:
        tab_chat, tab_analytics, tab_upload = st.tabs(["💬 Chat", "📊 Analytics", "📤 Upload Data"])

        # ── Chat tab ─────────────────────────────────────────────────────────
        with tab_chat:
            messages = st.session_state.messages

            # Welcome screen with quick prompts
            if not messages:
                st.markdown("""
                <div style='padding: 3rem 1rem 1.5rem 1rem; text-align: center;'>
                    <h1 style='font-size:2rem; font-weight:800; margin:0;'>🤖 AI Chatbot</h1>
                    <p style='color:#555; font-size:0.9rem; margin-top:6px;'>
                        Ask me anything — I remember everything you share, across all sessions.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**⚡ Quick questions:**")
                cols = st.columns(2)
                for i, prompt in enumerate(QUICK_PROMPTS):
                    with cols[i % 2]:
                        if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                            st.session_state.quick_prompt = prompt
                            st.rerun()

            # Render existing messages
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Show chart if this message has SQL data
                    chart_data = msg.get("chart_data", [])
                    if chart_data and len(chart_data) > 1:
                        fig = make_chart(chart_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

            # ── Input form ────────────────────────────────────────────────────
            with st.form("chat_form", clear_on_submit=True, border=False):
                col_in, col_btn = st.columns([12, 1])
                with col_in:
                    user_input = st.text_input(
                        "msg", placeholder="Ask me anything…",
                        label_visibility="collapsed",
                    )
                with col_btn:
                    submitted = st.form_submit_button("↑", use_container_width=True)

        # ── Analytics tab ─────────────────────────────────────────────────────
        with tab_analytics:
            render_analytics(sql)

        # ── Upload tab ─────────────────────────────────────────────────────────
        with tab_upload:
            render_upload_tab(sql)

    # ── Handle quick prompt ───────────────────────────────────────────────────
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.pop("quick_prompt")
        submitted  = True

    if not (submitted and user_input and user_input.strip()):
        return

    user_input = user_input.strip()

    # Create session on first message
    if not st.session_state.current_sid:
        st.session_state.current_sid     = mm.new_session()
        st.session_state.title_generated = False

    sid = st.session_state.current_sid

    # Save & display user message
    st.session_state.messages.append({"role": "user", "content": user_input, "chart_data": []})
    mm.add_message("user", user_input, session_id=sid)

    # Extract facts from user message
    pf = mm.extract_facts_from_message(user_input)
    if pf:
        mm.update_facts(pf)

    # Get AI response
    with right_col:
        with tab_chat:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        response_text, client, sql_rows = chat_with_ai(
                            mm, user_input, sid,
                            rag=rag, sql=sql,
                        )
                        st.markdown(response_text)

                        # Auto-chart if SQL returned chartable data
                        if sql_rows and len(sql_rows) > 1:
                            fig = make_chart(sql_rows)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        mm.add_message("assistant", response_text, session_id=sid)
                        st.session_state.messages.append({
                            "role":       "assistant",
                            "content":    response_text,
                            "chart_data": sql_rows,
                        })

                        # Generate session title on first message
                        if not st.session_state.title_generated:
                            title = generate_title(client, user_input)
                            mm.set_session_title(sid, title)
                            st.session_state.title_generated = True

                        # AI fact extraction
                        if not pf:
                            try:
                                af = ai_extract_facts(client, user_input)
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
