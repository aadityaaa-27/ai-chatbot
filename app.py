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
    background: #141414; border: 1px solid #222;
    border-radius: 8px; padding: 9px 12px 7px 12px;
    margin: 3px 0; transition: border-color .15s;
}
.scard:hover { border-color: #333; }
.scard.active { border-color: #1a73e8; background: #0d1f40; }
.scard-title  { font-size: 0.83rem; font-weight: 600; color: #e0e0e0;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scard-meta   { font-size: 0.64rem; color: #3a3a3a; margin-top: 2px; }

/* Section label */
.sec {
    font-size: 0.60rem; font-weight: 700; letter-spacing: .10em;
    text-transform: uppercase; color: #333; margin: 0.8rem 0 0.3rem 0;
}

/* Status badge */
.status-ok  { color: #4ade80; font-size: 0.75rem; }
.status-err { color: #f87171; font-size: 0.75rem; }

/* Setup notice */
.setup-box {
    background: #1a1200; border: 1px solid #3d2e00;
    border-radius: 8px; padding: 10px 12px; font-size: 0.78rem; color: #f5c518;
}

/* Dataset source badge on AI messages */
.ds-badge {
    display: inline-block; background: #1a2a1a; border: 1px solid #2a3d2a;
    color: #4ade80; border-radius: 6px; font-size: 0.68rem;
    padding: 2px 8px; margin-top: 6px;
}

hr { border-color: #1a1a1a !important; margin: 0.5rem 0 !important; }

/* Sticky chat input — dark theme */
[data-testid="stChatInput"] {
    background: #0d0d0d !important;
    border-top: 1px solid #1e1e1e !important;
    padding: 6px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #141414 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 1px #1a73e8 !important;
}

/* Per-chart AI answer bubble */
.chart-ai-ans {
    background: #0a1a0a; border: 1px solid #1a3a1a;
    border-radius: 8px; padding: 9px 13px;
    font-size: 0.80rem; color: #b8d8b8;
    margin-top: 4px; line-height: 1.55;
}

/* Tighten tab font */
[data-testid="stTabs"] button { font-size: 0.84rem !important; }
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


# ── Per-chart AI helpers ──────────────────────────────────────────────────────

# Human-readable labels for each analytics key
_ANALYTICS_LABELS = {
    "dept_headcount":    "Headcount by Department",
    "dept_attrition":    "Attrition Rate by Department (%)",
    "dept_salary":       "Avg Monthly Salary by Department",
    "age_groups":        "Headcount by Age Group",
    "age_salary":        "Avg Monthly Salary by Age Group",
    "gender":            "Overall Gender Breakdown",
    "gender_salary":     "Avg Monthly Salary by Gender",
    "gender_by_dept":    "Gender Breakdown by Department",
    "satisfaction":      "Job Satisfaction Levels",
    "overtime":          "Overtime Distribution (Yes/No)",
    "overtime_by_age":   "Overtime Rate (%) by Age Group",
    "overtime_by_dept":  "Overtime Rate (%) by Department",
    "overtime_by_gender":"Overtime Rate (%) by Gender",
    "education":         "Education Level Distribution",
    "education_salary":  "Avg Salary by Education Level",
    "role_salary":       "Avg Monthly Salary by Job Role (top 15)",
    "salary_by_jobrole": "Avg Monthly Salary by Job Role",
    "dept_avg_age":      "Avg Age & Salary by Department",
    "attrition_age":     "Attrition Rate by Age Group",
    "attrition_by_gender":"Attrition Rate by Gender",
    "attrition_by_dept": "Attrition Rate by Department",
    "satisfaction_by_dept":"Avg Job Satisfaction by Department",
}

# Keyword pairs → analytics key (for chart override matching)
# First matching rule wins, so order from most specific to least.
_CHART_KEYWORD_MAP = [
    (["overtime", "age"],            "overtime_by_age"),
    (["overtime", "department"],     "overtime_by_dept"),
    (["overtime", "dept"],           "overtime_by_dept"),
    (["overtime", "gender"],         "overtime_by_gender"),
    (["attrition", "age"],           "attrition_age"),
    (["attrition", "gender"],        "attrition_by_gender"),
    (["attrition", "department"],    "attrition_by_dept"),
    (["attrition", "dept"],          "attrition_by_dept"),
    (["salary", "age"],              "age_salary"),
    (["paid", "age"],                "age_salary"),
    (["income", "age"],              "age_salary"),
    (["earn", "age"],                "age_salary"),
    (["salary", "gender"],           "gender_salary"),
    (["paid", "gender"],             "gender_salary"),
    (["salary", "role"],             "role_salary"),
    (["salary", "job"],              "role_salary"),
    (["salary", "education"],        "education_salary"),
    (["salary", "degree"],           "education_salary"),
    (["education", "salary"],        "education_salary"),
    (["satisfaction", "department"], "satisfaction_by_dept"),
    (["satisfaction", "dept"],       "satisfaction_by_dept"),
    (["gender", "department"],       "gender_by_dept"),
    (["gender", "dept"],             "gender_by_dept"),
    (["age", "department"],          "dept_avg_age"),
    (["age", "dept"],                "dept_avg_age"),
    (["headcount", "age"],           "age_groups"),
    (["employee", "age"],            "age_groups"),
    (["age", "group"],               "age_groups"),
]


def _find_chart_data(question: str, all_analytics: dict,
                     sql: SQLEngine, source_file: str) -> list:
    """
    Return the best list-of-dicts to use as a chart override.
    Strategy:
      1. Match question keywords against _CHART_KEYWORD_MAP → pre-built table
      2. Fall back to sql.answer() dynamic SQL if no match
      3. Return [] if nothing chartable is found
    """
    q = question.lower()

    # ── 1. Keyword match → pre-built table ────────────────────────────────────
    for keywords, key in _CHART_KEYWORD_MAP:
        if all(kw in q for kw in keywords):
            rows = all_analytics.get(key, [])
            if rows and len(rows) >= 2:
                if make_chart(rows) is not None:
                    return rows

    # ── 2. Dynamic SQL fallback ────────────────────────────────────────────────
    if sql and sql.ready:
        try:
            result = sql.answer(question, source_file=source_file)
            if not result.get("error") and result.get("data"):
                rows = result["data"]
                if len(rows) >= 2 and make_chart(rows) is not None:
                    return rows
        except Exception:
            pass

    return []


def ask_about_chart(chart_title: str, chart_rows: list, question: str,
                    source_file: str = "", sql: SQLEngine = None,
                    all_analytics: dict = None) -> str:
    """
    Answer a question about a chart using ALL pre-computed analytics data.
    Pre-built queries (age_salary, gender_salary, role_salary, etc.) are
    fetched once when the analytics tab loads, so cross-chart questions like
    'which age group is most paid?' work reliably without dynamic SQL.
    """
    if not chart_rows:
        return "No data available for this chart."

    sf_ctx = f"Dataset: {source_file}" if source_file else "Dataset: all data combined"

    # ── Build comprehensive context from ALL pre-computed analytics ───────────
    sections = []
    analytics = all_analytics or {}
    for key, label in _ANALYTICS_LABELS.items():
        rows = analytics.get(key, [])
        if rows:
            df_str = pd.DataFrame(rows).to_string(index=False)
            sections.append(f"[{label}]\n{df_str}")

    # Always include the primary chart's data first (in case it's not in analytics)
    primary_tbl = pd.DataFrame(chart_rows).to_string(index=False)
    all_data_ctx = (
        f"[{chart_title} — primary chart]\n{primary_tbl}\n\n"
        + "\n\n".join(sections)
    )

    prompt = (
        f"You are an expert HR analyst with access to full employee analytics.\n"
        f"{sf_ctx}\n\n"
        f"ALL AVAILABLE ANALYTICS DATA:\n"
        f"{all_data_ctx}\n\n"
        f"Question (asked in the context of the '{chart_title}' chart): {question}\n\n"
        "Rules:\n"
        "- Use whichever table(s) from the data above best answer the question.\n"
        "- Always cite specific numbers, department names, or group labels.\n"
        "- Highlight the key insight or actionable finding.\n"
        "- Plain English, 2–3 sentences. No markdown headers, no code."
    )
    client = _make_client()
    r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return r.text.strip()


def _render_chart(chart_key: str, fig, rows: list, chart_title: str,
                  source_file: str, sql: SQLEngine = None):
    """
    Render a chart that can be replaced in-place by an AI-generated chart
    when the user asks a question.  Then renders the AI chat widget below.
    """
    override_key = f"chart_override_{chart_key}"
    override     = st.session_state.get(override_key)

    if override:
        try:
            dyn_fig = make_chart(override["rows"])
            if dyn_fig:
                dyn_fig.update_layout(
                    title=f"🔍 {override['question'][:60]}",
                    **_DARK,
                )
                st.plotly_chart(dyn_fig, use_container_width=True)
            else:
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.plotly_chart(fig, use_container_width=True)

        if st.button("↩ Reset to original chart", key=f"reset_{chart_key}",
                     use_container_width=True):
            st.session_state.pop(override_key, None)
            st.session_state.pop(f"ca_{chart_key}", None)
            st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True)

    _chart_ai(chart_key, chart_title, rows, source_file, sql)


def _chart_ai(chart_key: str, chart_title: str, rows: list,
              source_file: str = "", sql: SQLEngine = None):
    """Render the mini-chat widget below a chart.
    On submit:  (a) generates a text answer using all analytics data,
                (b) tries to run a DB query and replace the chart if chartable."""
    ans_key = f"ca_{chart_key}"

    with st.form(key=f"cf_{chart_key}", clear_on_submit=True, border=False):
        col_q, col_b = st.columns([6, 1])
        with col_q:
            question = st.text_input(
                "q", placeholder="Ask about this chart…",
                label_visibility="collapsed",
                key=f"ci_{chart_key}",
            )
        with col_b:
            ask = st.form_submit_button("Ask ✨", use_container_width=True)

    if ask and question.strip():
        all_analytics = st.session_state.get("analytics_data", {})
        with st.spinner("Analysing…"):
            # ── Text answer ───────────────────────────────────────────────────
            st.session_state[ans_key] = ask_about_chart(
                chart_title, rows, question, source_file,
                sql=sql, all_analytics=all_analytics,
            )
            # ── Chart override: keyword-matched pre-built OR dynamic SQL ──────
            dyn_rows = _find_chart_data(question, all_analytics, sql, source_file)
            if dyn_rows:
                st.session_state[f"chart_override_{chart_key}"] = {
                    "rows": dyn_rows,
                    "question": question,
                }

    if ans_key in st.session_state:
        st.markdown(
            f'<div class="chart-ai-ans">🤖 {st.session_state[ans_key]}</div>',
            unsafe_allow_html=True,
        )
        if st.button("✕ clear", key=f"clr_{chart_key}",
                     type="secondary", use_container_width=False):
            st.session_state.pop(ans_key, None)
            st.session_state.pop(f"chart_override_{chart_key}", None)
            st.rerun()


# ── Gemini helpers ────────────────────────────────────────────────────────────

def build_system_prompt(memory_manager: MemoryManager, rag_context: str = "",
                        emp_count: int = 0, source_file: str = "") -> str:
    count_str   = f"{emp_count:,}" if emp_count else "several thousand"
    dataset_ctx = (
        f"You are currently querying the dataset from file: '{source_file}' "
        f"({count_str} records).\n"
        if source_file else
        f"You have access to all datasets combined ({count_str} total records).\n"
    )
    base = (
        "You are a smart, helpful, friendly AI assistant for an enterprise company. "
        "You have been given DIRECT ACCESS to the company's live employee database. "
        f"{dataset_ctx}"
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
        if rag_context.startswith("DATA AVAILABILITY CHECK:"):
            # Special case: column is all-NULL in the selected file
            base += (
                f"\n\n{rag_context}"
                "\n\n⚠️ CRITICAL INSTRUCTION: The check above shows that the requested "
                "information does NOT exist in the uploaded file. "
                "You MUST tell the user that this specific data is not available in their file — "
                "do NOT make up numbers, do NOT use information from any other source, "
                "do NOT say you lack access. Just explain clearly and helpfully that "
                "this particular column/field was not present or was empty in the uploaded dataset."
            )
        else:
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
                 rag: RAGEngine = None, sql: SQLEngine = None,
                 source_file: str = ""):
    """Send a message and return (response_text, client, sql_rows)."""
    sql_ctx, sql_rows = "", []
    rag_ctx = ""

    emp_count = sql.employee_count(source_file=source_file) if (sql and sql.ready) else 0

    if sql and sql.ready:
        sql_ctx, sql_rows = sql.query(user_input, source_file=source_file)
    if rag and rag.ready and not sql_ctx:
        rag_ctx = rag.get_context(user_input)

    combined = "\n\n".join(filter(None, [sql_ctx, rag_ctx]))
    system_prompt = build_system_prompt(
        memory_manager, combined,
        emp_count=emp_count, source_file=source_file,
    )
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
    sql: SQLEngine = st.session_state.get("sql")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🤖 AI Chatbot")
    if sql and sql.ready:
        st.markdown('<span class="status-ok">● Connected</span> · Gemini 2.0 Flash',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">● No database</span> · Gemini 2.0 Flash',
                    unsafe_allow_html=True)

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
        with st.container(height=240):
            for s in sessions:
                is_active = s["id"] == cur_sid
                card_cls  = "scard active" if is_active else "scard"
                try:
                    created = datetime.fromisoformat(s["created_at"]).strftime("%d %b, %H:%M")
                except Exception:
                    created = ""
                meta = f"{created}  ·  {s['msg_count']} msgs"
                st.markdown(
                    f'<div class="{card_cls}">'
                    f'<div class="scard-title">{s["title"]}</div>'
                    f'<div class="scard-meta">{meta}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                col_l, col_d = st.columns([5, 1])
                with col_l:
                    if st.button("↩ Open", key=f"load_{s['id']}", use_container_width=True):
                        msgs = mm.get_session_messages(s["id"])
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"], "chart_data": []}
                            for m in msgs
                        ]
                        st.session_state.current_sid     = s["id"]
                        st.session_state.title_generated = True
                        st.rerun()
                with col_d:
                    if st.button("🗑", key=f"del_{s['id']}", use_container_width=True):
                        mm.delete_session(s["id"])
                        if st.session_state.get("current_sid") == s["id"]:
                            st.session_state.messages        = []
                            st.session_state.current_sid     = None
                            st.session_state.title_generated = False
                        st.rerun()

    st.divider()

    # ── Dataset selector ──────────────────────────────────────────────────────
    st.markdown('<div class="sec">📂 Dataset</div>', unsafe_allow_html=True)

    sql: SQLEngine = st.session_state.get("sql")

    # Show setup notice if source_file column is missing
    if sql and sql.ready and not sql.has_source_file_column():
        st.markdown(
            '<div class="setup-box">⚠️ <b>One-time setup needed</b><br>'
            'Run <code>setup_source_file.sql</code> in Supabase → SQL Editor<br>'
            'to enable dataset tracking.</div>',
            unsafe_allow_html=True,
        )
        st.caption("")

    datasets = sql.get_source_files() if (sql and sql.ready) else []

    if not datasets:
        st.caption("No datasets yet. Upload a file →")
    else:
        ds_names    = [d["source_file"] for d in datasets]
        ds_labels   = [f"{d['source_file']}  ({int(d['count']):,} rows)" for d in datasets]
        all_label   = f"📊 All datasets  ({sum(int(d['count']) for d in datasets):,} rows)"
        options     = [all_label] + ds_labels

        prev = st.session_state.get("active_source_file", "")
        # Find current index
        try:
            cur_idx = ds_names.index(prev) + 1 if prev in ds_names else 0
        except ValueError:
            cur_idx = 0

        selected_idx = st.selectbox(
            "Active dataset",
            range(len(options)),
            format_func=lambda i: options[i],
            index=cur_idx,
            label_visibility="collapsed",
            key="dataset_selector",
        )

        new_sf = "" if selected_idx == 0 else ds_names[selected_idx - 1]
        if new_sf != prev:
            st.session_state.active_source_file = new_sf
            st.session_state.pop("analytics_data", None)   # refresh charts
            st.rerun()

        # Actions for selected dataset
        if new_sf:
            col_ren, col_del = st.columns(2)
            with col_del:
                if st.button("🗑️ Delete", use_container_width=True, key="del_dataset_btn"):
                    with st.spinner("Deleting…"):
                        if sql.delete_source_file(new_sf):
                            st.session_state.active_source_file = ""
                            st.session_state.pop("analytics_data", None)
                            st.rerun()
                        else:
                            st.error("Delete failed.")
            with col_ren:
                if st.button("✏️ Rename", use_container_width=True, key="rename_dataset_btn"):
                    st.session_state.renaming_dataset = new_sf

        # Rename input
        if st.session_state.get("renaming_dataset"):
            old = st.session_state.renaming_dataset
            new_name = st.text_input(f"New name for '{old}'", key="rename_input",
                                     placeholder="e.g. Q1_2025_employees")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("✅ Save", use_container_width=True, key="rename_save"):
                    if new_name.strip():
                        with st.spinner("Renaming…"):
                            if sql.retag_dataset(old, new_name.strip()):
                                st.session_state.active_source_file = new_name.strip()
                                st.session_state.pop("renaming_dataset", None)
                                st.session_state.pop("analytics_data", None)
                                st.rerun()
                            else:
                                st.error("Rename failed.")
            with rc2:
                if st.button("Cancel", use_container_width=True, key="rename_cancel"):
                    st.session_state.pop("renaming_dataset", None)
                    st.rerun()

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

    filename = st.session_state.get("upload_filename", "upload")
    # Strip extension for cleaner dataset name
    dataset_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    mode = st.radio(
        "Import mode",
        [
            f"➕ Add as new dataset  (keeps all existing data)",
            f"🔄 Replace '{dataset_name}'  (removes old version of this file only)",
            f"🗑️ Replace ALL data  (wipes everything, loads only this file)",
        ],
        help="Add: keeps all existing datasets. Replace file: re-imports just this file. Replace all: full reset.",
    )

    if "Replace ALL" in mode:
        import_mode = "replace_all"
        st.warning(f"⚠️ This will delete **all** existing data and load only this file.")
    elif "Replace" in mode:
        import_mode = "replace_dataset"
        st.info(f"ℹ️ Only rows from **'{dataset_name}'** will be replaced.")
    else:
        import_mode = "add_new"

    col_imp, _ = st.columns([1, 3])
    with col_imp:
        do_import = st.button("✅ Import Now", type="primary", use_container_width=True)

    if do_import:
        progress = st.progress(0, text="Preparing…")
        try:
            sb = sql._sb
            progress.progress(20, text=f"Processing {len(clean_df):,} rows…")
            st.session_state.pop("analytics_data", None)

            progress.progress(40, text="Inserting into database…")
            result = dp.insert_to_db(clean_df, sb,
                                     mode=import_mode,
                                     source_file=dataset_name)
            progress.progress(100, text="Done!")

            if result["inserted"] > 0:
                st.success(
                    f"✅ **{result['inserted']:,} rows imported** as dataset **'{dataset_name}'**!\n\n"
                    f"Select it in the **📂 Dataset** panel → ask questions in **💬 Chat**."
                )
                st.session_state.active_source_file = dataset_name
                st.session_state.pop("upload_mapping", None)
                st.session_state.pop("upload_raw_df", None)
            else:
                st.error(
                    "Import failed — 0 rows inserted.\n\n"
                    + "\n".join(result.get("errors", []))
                )

        except Exception as e:
            progress.empty()
            st.error(f"Import error: {e}")


# ── Analytics tab ─────────────────────────────────────────────────────────────

def render_analytics(sql: SQLEngine, source_file: str = ""):
    label = f"📊 Analytics — {source_file}" if source_file else "📊 HR Analytics Dashboard"
    st.markdown(f"### {label}")
    if not (sql and sql.ready):
        st.warning("⚠️ Analytics require a live Supabase connection.")
        return

    col_refresh, col_space = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.pop("analytics_data", None)

    if "analytics_data" not in st.session_state:
        with st.spinner("Loading analytics data…"):
            st.session_state.analytics_data = sql.get_analytics_data(source_file=source_file)

    data = st.session_state.analytics_data
    if not data:
        st.error("Could not load analytics data.")
        return

    # ── Row 1: Headcount + Attrition ─────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        rows = data.get("dept_headcount", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_bar(df, "department", "employees", "👥 Employees by Department")
            _render_chart("dept_headcount", fig, rows, "Employees by Department", source_file, sql)
    with c2:
        rows = data.get("dept_attrition", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_bar(df, "department", "attrition_pct",
                                "📉 Attrition Rate by Department (%)", "Reds")
            _render_chart("dept_attrition", fig, rows, "Attrition Rate % by Department", source_file, sql)

    # ── Row 2: Salary + Age groups ────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        rows = data.get("dept_salary", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_bar(df, "department", "avg_salary",
                                "💰 Avg Monthly Salary by Department ($)", "Greens")
            _render_chart("dept_salary", fig, rows, "Avg Monthly Salary by Department", source_file, sql)
    with c4:
        rows = data.get("age_groups", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = px.bar(df, x="age_group", y="employees", title="📅 Age Distribution",
                         template="plotly_dark", color="employees",
                         color_continuous_scale="Blues")
            fig.update_layout(**_DARK)
            _render_chart("age_groups", fig, rows, "Age Distribution of Employees", source_file, sql)

    # ── Row 3: Gender + Job Satisfaction ─────────────────────────────────────
    c5, c6 = st.columns(2)
    with c5:
        rows = data.get("gender", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_pie(df, "gender", "employees", "⚧ Gender Breakdown")
            _render_chart("gender", fig, rows, "Overall Gender Breakdown", source_file, sql)
    with c6:
        rows = data.get("satisfaction", [])
        if rows:
            df = pd.DataFrame(rows)
            label_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
            df["job_satisfaction"] = df["job_satisfaction"].map(label_map)
            fig = px.bar(df, x="job_satisfaction", y="employees",
                         title="😊 Job Satisfaction", template="plotly_dark",
                         color="employees", color_continuous_scale="Blues")
            fig.update_layout(**_DARK)
            _render_chart("satisfaction", fig, df.to_dict(orient="records"),
                          "Job Satisfaction Distribution", source_file, sql)

    # ── Row 4: Overtime + Education ───────────────────────────────────────────
    c7, c8 = st.columns(2)
    with c7:
        rows = data.get("overtime", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_pie(df, "overtime", "employees", "⏰ Overtime Distribution")
            _render_chart("overtime", fig, rows, "Overtime Distribution", source_file, sql)
    with c8:
        rows = data.get("education", [])
        if rows:
            df = pd.DataFrame(rows)
            edu_map = {1: "Below College", 2: "College", 3: "Bachelor",
                       4: "Master", 5: "Doctor"}
            df["education"] = df["education"].map(edu_map)
            fig = px.bar(df, x="education", y="employees", title="🎓 Education Level",
                         template="plotly_dark", color="employees",
                         color_continuous_scale="Purples")
            fig.update_layout(**_DARK)
            _render_chart("education", fig, df.to_dict(orient="records"),
                          "Education Level Distribution", source_file, sql)

    # ── Row 5: Gender by Department (full-width) ──────────────────────────────
    rows = data.get("gender_by_dept", [])
    if rows:
        df  = pd.DataFrame(rows)
        fig = px.bar(df, x="department", y="employees", color="gender",
                     barmode="group", title="⚧ Gender Breakdown by Department",
                     template="plotly_dark",
                     color_discrete_map={"Male": "#1a73e8", "Female": "#e84a1a"})
        fig.update_layout(**{**_DARK, "showlegend": True, "height": 320})
        _render_chart("gender_by_dept", fig, rows, "Gender Breakdown by Department", source_file, sql)


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
                    # Dataset source badge on assistant messages
                    if msg["role"] == "assistant":
                        sf_label = msg.get("source_file", "")
                        badge = sf_label if sf_label else "All datasets"
                        st.markdown(
                            f'<div class="ds-badge">📂 {badge}</div>',
                            unsafe_allow_html=True,
                        )
                    # Show chart if this message has SQL data
                    chart_data = msg.get("chart_data", [])
                    if chart_data and len(chart_data) > 1:
                        fig = make_chart(chart_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

            # ── Sticky chat input (always visible at bottom of tab) ───────────────
            user_input = st.chat_input("Ask me anything…")
            submitted  = bool(user_input)

        # ── Analytics tab ─────────────────────────────────────────────────────
        with tab_analytics:
            active_sf = st.session_state.get("active_source_file", "")
            render_analytics(sql, source_file=active_sf)

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
                        active_sf = st.session_state.get("active_source_file", "")
                        response_text, client, sql_rows = chat_with_ai(
                            mm, user_input, sid,
                            rag=rag, sql=sql,
                            source_file=active_sf,
                        )
                        st.markdown(response_text)
                        # Dataset badge — shown immediately on new response
                        badge = active_sf if active_sf else "All datasets"
                        st.markdown(
                            f'<div class="ds-badge">📂 {badge}</div>',
                            unsafe_allow_html=True,
                        )

                        # Auto-chart if SQL returned chartable data
                        if sql_rows and len(sql_rows) > 1:
                            fig = make_chart(sql_rows)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        mm.add_message("assistant", response_text, session_id=sid)
                        st.session_state.messages.append({
                            "role":        "assistant",
                            "content":     response_text,
                            "chart_data":  sql_rows,
                            "source_file": active_sf,   # which dataset answered this
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
