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
from auth import (authenticate, ensure_super_admin, get_all_users, create_user,
                  delete_user, update_password, role_info, ROLES, get_auth_client,
                  get_all_companies, create_company, delete_company,
                  create_invite_token, get_invite_token, use_invite_token,
                  get_all_invite_tokens, revoke_invite_token)

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
/* ── Hide Streamlit chrome ─────────────────────────────────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, .stApp {
    background: #09090f !important;
    color: #e2e8f0 !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a3d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4f7fff; }

/* ── Gradient headings ─────────────────────────────────────────────────────── */
h1, h2, h3 {
    background: linear-gradient(135deg, #6eb6ff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
}
/* h4/h5 stay plain white */
h4, h5, h6 { color: #cbd5e1 !important; }

/* ── Dividers ──────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 0.6rem 0 !important; }

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
/* Primary */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4f7fff 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 0 18px rgba(79,127,255,0.30) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 0 28px rgba(79,127,255,0.50) !important;
    transform: translateY(-1px) !important;
}
/* Secondary / default */
.stButton > button:not([kind="primary"]),
[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(79,127,255,0.25) !important;
    color: #94a3b8 !important;
}
.stButton > button:not([kind="primary"]):hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: rgba(79,127,255,0.55) !important;
    color: #6eb6ff !important;
    background: rgba(79,127,255,0.07) !important;
    box-shadow: 0 0 12px rgba(79,127,255,0.15) !important;
}
/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4f7fff 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 0 14px rgba(79,127,255,0.25) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 4px !important;
}
[data-testid="stTabs"] button {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 6px 16px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button:hover { color: #94a3b8 !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6eb6ff !important;
    border-bottom: 2px solid #4f7fff !important;
    background: rgba(79,127,255,0.06) !important;
}

/* ── Chat input ────────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #09090f !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    padding: 8px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #0f0f1a !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(79,127,255,0.20) !important;
    border-radius: 14px !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4f7fff !important;
    box-shadow: 0 0 0 2px rgba(79,127,255,0.18), 0 0 22px rgba(79,127,255,0.10) !important;
}

/* ── Text inputs ───────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #0f0f1a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #4f7fff !important;
    box-shadow: 0 0 0 2px rgba(79,127,255,0.15) !important;
}

/* ── Selectbox ─────────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #0f0f1a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Radio ─────────────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label p { color: #94a3b8 !important; }

/* ── File uploader ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background: rgba(79,127,255,0.03) !important;
    border: 1.5px dashed rgba(79,127,255,0.28) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(79,127,255,0.55) !important;
    background: rgba(79,127,255,0.06) !important;
}

/* ── Plotly chart cards ────────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] > div {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stPlotlyChart"] > div:hover {
    border-color: rgba(79,127,255,0.22) !important;
}

/* ── Progress bar ──────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #4f7fff 0%, #8b5cf6 100%) !important;
    border-radius: 4px !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(79,127,255,0.20) !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Alerts ────────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Session cards ─────────────────────────────────────────────────────────── */
.scard {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 9px 12px 7px 12px;
    margin: 3px 0; transition: all 0.18s ease;
}
.scard:hover {
    border-color: rgba(79,127,255,0.30);
    background: rgba(79,127,255,0.05);
}
.scard.active {
    border-color: #4f7fff;
    background: rgba(79,127,255,0.10);
    box-shadow: 0 0 16px rgba(79,127,255,0.14);
}
.scard-title {
    font-size: 0.83rem; font-weight: 600; color: #cbd5e1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.scard-meta { font-size: 0.64rem; color: #334155; margin-top: 2px; }

/* ── Section labels ────────────────────────────────────────────────────────── */
.sec {
    font-size: 0.60rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #334155;
    margin: 0.9rem 0 0.3rem 0;
}

/* ── Status badges ─────────────────────────────────────────────────────────── */
.status-ok  { color: #34d399; font-size: 0.75rem; }
.status-err { color: #f87171; font-size: 0.75rem; }

/* ── Setup notice ──────────────────────────────────────────────────────────── */
.setup-box {
    background: rgba(245,197,24,0.05); border: 1px solid rgba(245,197,24,0.20);
    border-radius: 10px; padding: 10px 12px; font-size: 0.78rem; color: #f5c518;
}

/* ── Dataset source badge ──────────────────────────────────────────────────── */
.ds-badge {
    display: inline-block;
    background: rgba(79,127,255,0.08); border: 1px solid rgba(79,127,255,0.22);
    color: #6eb6ff; border-radius: 20px; font-size: 0.68rem;
    padding: 2px 10px; margin-top: 6px;
}

/* ── Per-chart AI answer bubble ────────────────────────────────────────────── */
.chart-ai-ans {
    background: rgba(79,127,255,0.06);
    border: 1px solid rgba(79,127,255,0.20);
    border-left: 3px solid #4f7fff;
    border-radius: 10px; padding: 10px 14px;
    font-size: 0.80rem; color: #bdd3ff;
    margin-top: 6px; line-height: 1.60;
}

/* ── Login page ────────────────────────────────────────────────────────────── */
.login-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(79,127,255,0.20);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-top: 1rem;
}
.login-logo {
    text-align: center;
    font-size: 3.5rem;
    margin-bottom: 0.4rem;
}
.login-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6eb6ff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.login-sub {
    text-align: center;
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 1.6rem;
}

/* ── User info badge (left panel) ──────────────────────────────────────────── */
.user-badge {
    background: rgba(79,127,255,0.06);
    border: 1px solid rgba(79,127,255,0.18);
    border-radius: 10px;
    padding: 8px 10px;
    margin-bottom: 6px;
}
.user-badge .name { font-size:0.85rem; font-weight:600; color:#cbd5e1; }
.user-badge .role { font-size:0.70rem; color:#4f7fff; margin-top:2px; }

/* ── Access denied notice ──────────────────────────────────────────────────── */
.access-denied {
    background: rgba(248,113,113,0.06);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #fca5a5;
}

/* ══════════════════════════════════════════════════════════════════════════════
   MOBILE RESPONSIVE  (≤ 768 px)
   ══════════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {

    /* ── Columns → stack vertically ─────────────────────────────────────── */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
        border-right: none !important;
    }

    /* ── Left panel: compact horizontal strip ───────────────────────────── */
    .user-badge {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 10px;
    }
    .user-badge .name { font-size: 0.80rem; }
    .user-badge .role { font-size: 0.65rem; margin-top: 0; }

    /* ── Section labels: tighter spacing ───────────────────────────────── */
    .sec { margin: 0.5rem 0 0.2rem 0; }

    /* ── Headings ───────────────────────────────────────────────────────── */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.0rem !important; }

    /* ── Tabs: scrollable, no wrapping ─────────────────────────────────── */
    [data-testid="stTabs"] > div:first-child {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    [data-testid="stTabs"] > div:first-child::-webkit-scrollbar { display: none; }
    [data-testid="stTabs"] button {
        font-size: 0.78rem !important;
        padding: 5px 12px !important;
        white-space: nowrap !important;
    }

    /* ── Buttons: full width + bigger tap target ────────────────────────── */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.88rem !important;
        min-height: 44px !important;
    }

    /* ── Inputs: bigger touch targets ──────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div {
        font-size: 16px !important;   /* prevents iOS auto-zoom */
        min-height: 44px !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;   /* prevents iOS auto-zoom */
    }

    /* ── Charts: shorter on mobile ──────────────────────────────────────── */
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container {
        height: 220px !important;
        max-height: 220px !important;
    }

    /* ── Login: remove side columns, full width ─────────────────────────── */
    .login-card {
        padding: 1.5rem 1rem !important;
        margin-top: 0.5rem !important;
        border-radius: 14px !important;
    }
    .login-logo  { font-size: 2.5rem !important; }
    .login-title { font-size: 1.3rem !important; }
    .login-sub   { font-size: 0.75rem !important; }

    /* ── Chat messages: smaller avatars ────────────────────────────────── */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
    }
    [data-testid="stChatMessage"] > div:first-child {
        width: 28px !important;
        height: 28px !important;
        font-size: 0.75rem !important;
    }

    /* ── Dataframe: horizontal scroll ───────────────────────────────────── */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
    }

    /* ── File uploader ───────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] section {
        padding: 1rem !important;
    }

    /* ── Expanders ───────────────────────────────────────────────────────── */
    [data-testid="stExpander"] summary {
        font-size: 0.82rem !important;
    }

    /* ── Chat input: always visible at bottom ───────────────────────────── */
    [data-testid="stChatInput"] {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
        padding: 6px 0 !important;
    }

    /* ── Block container: add side padding on mobile ────────────────────── */
    .block-container {
        padding: 0.5rem !important;
    }
}

/* ── Very small phones (≤ 400 px) ─────────────────────────────────────────── */
@media (max-width: 400px) {
    [data-testid="stTabs"] button {
        font-size: 0.70rem !important;
        padding: 4px 8px !important;
    }
    h1 { font-size: 1.2rem !important; }
    .login-title { font-size: 1.1rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────

_DARK = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8",
    margin=dict(t=40, b=12, l=12, r=12),
    height=290,
    showlegend=False,
    coloraxis_showscale=False,
    title_font=dict(size=13, color="#cbd5e1"),
    hoverlabel=dict(
        bgcolor="#1e1e2e",
        bordercolor="rgba(79,127,255,0.4)",
        font_color="#e2e8f0",
        font_size=12,
    ),
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
        _PIE_COLORS = ["#4f7fff", "#8b5cf6", "#06b6d4", "#34d399", "#f59e0b"]
        # Pie for very small sets (≤ 4 categories, e.g. gender)
        if len(df) <= 4:
            fig = px.pie(
                df, names=x_col, values=y_col,
                template="plotly_dark",
                color_discrete_sequence=_PIE_COLORS,
            )
            fig.update_traces(
                hole=0.42,
                textposition="inside",
                textinfo="percent+label",
                marker_line=dict(color="#09090f", width=2),
            )
        else:
            df = df.sort_values(y_col, ascending=True).tail(15)
            fig = px.bar(
                df, x=y_col, y=x_col, orientation="h",
                template="plotly_dark",
                color=y_col,
                color_continuous_scale=[[0,"#1e2a5e"],[1,"#4f7fff"]],
            )
            fig.update_traces(marker_line_width=0)
        fig.update_layout(**_DARK)
        return fig
    except Exception:
        return None


def analytics_bar(df: pd.DataFrame, x: str, y: str, title: str,
                  color_scale="Blues"):
    # Limit to top 8 rows so mobile bars don't get too small
    df = df.sort_values(y, ascending=False).head(8).sort_values(y, ascending=True)
    fig = px.bar(
        df, x=y, y=x, orientation="h",
        title=title, template="plotly_dark",
        color=y, color_continuous_scale=color_scale,
    )
    fig.update_layout(**_DARK)
    fig.update_xaxes(
        tickfont_size=10,
        automargin=True,
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
    )
    fig.update_yaxes(
        tickfont_size=10,
        automargin=True,
        tickmode="array",
        # Truncate long labels so they don't overflow on mobile
        ticktext=[str(v)[:18] + "…" if len(str(v)) > 18 else str(v)
                  for v in df[x].tolist()],
        tickvals=df[x].tolist(),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def analytics_pie(df: pd.DataFrame, names: str, values: str, title: str):
    _PIE_COLORS = ["#4f7fff", "#8b5cf6", "#06b6d4", "#34d399", "#f59e0b", "#f87171"]
    fig = px.pie(
        df, names=names, values=values,
        title=title, template="plotly_dark",
        color_discrete_sequence=_PIE_COLORS,
    )
    fig.update_layout(**{
        **_DARK,
        "showlegend": True,
        # Legend inside chart, horizontal at bottom — compact on mobile
        "legend": dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.15, yanchor="top",
            font_size=10,
            bgcolor="rgba(0,0,0,0)",
        ),
        "margin": dict(t=40, b=50, l=8, r=8),
    })
    fig.update_traces(
        hole=0.45,
        # Only show percent inside the slice — no label text, no overflow
        textposition="inside",
        textinfo="percent",
        textfont_size=11,
        marker_line=dict(color="#09090f", width=2),
        # Pull slices slightly so legend doesn't overlap
        pull=[0.02] * len(df),
    )
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
      1. Match question keywords → pre-built table already in all_analytics
      2. Cache miss? Refresh full analytics from DB once, then retry
      3. Still no match? Try sql.answer() dynamic SQL
      4. Return [] if nothing chartable found
    """
    q = question.lower()
    _cache_refreshed = False

    def _try_key(key: str) -> list:
        rows = all_analytics.get(key, [])
        if rows and len(rows) >= 2 and make_chart(rows) is not None:
            return rows
        return []

    # ── 1. Keyword match → pre-built table ───────────────────────────────────
    for keywords, key in _CHART_KEYWORD_MAP:
        if all(kw in q for kw in keywords):
            rows = _try_key(key)
            if rows:
                return rows

            # ── 2. Cache miss — refresh analytics once and retry ──────────────
            if not _cache_refreshed and sql and sql.ready:
                try:
                    fresh = sql.get_analytics_data(source_file=source_file,
                                                   company_id=st.session_state.get("user_company_id") or 0)
                    st.session_state["analytics_data"] = fresh
                    all_analytics.update(fresh)      # mutate in-place so caller sees it
                    _cache_refreshed = True
                    rows = _try_key(key)
                    if rows:
                        return rows
                except Exception:
                    pass

    # ── 3. Dynamic SQL fallback for unrecognised questions ────────────────────
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
            for _k in [override_key, f"ca_{chart_key}", f"ch_{chart_key}"]:
                st.session_state.pop(_k, None)
            st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True)

    _chart_ai(chart_key, chart_title, rows, source_file, sql)


def _chart_ai(chart_key: str, chart_title: str, rows: list,
              source_file: str = "", sql: SQLEngine = None):
    """
    Mini-chat widget with full conversation history.
    Each question appends to a per-chart history (capped at 5) so the user
    can drill down across 4-5 follow-up questions without losing context.
    Each question also attempts to update the chart above via _find_chart_data().
    """
    hist_key = f"ch_{chart_key}"   # list[{"q": str, "a": str}]

    # ── Input form ────────────────────────────────────────────────────────────
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
            answer = ask_about_chart(
                chart_title, rows, question, source_file,
                sql=sql, all_analytics=all_analytics,
            )
            # Append to per-chart history (keep last 5 turns)
            hist = st.session_state.get(hist_key, [])
            hist = (hist + [{"q": question, "a": answer}])[-5:]
            st.session_state[hist_key] = hist

            # ── Chart override: keyword map first, then SQL fallback ──────────
            dyn_rows = _find_chart_data(question, all_analytics, sql, source_file)
            if dyn_rows:
                st.session_state[f"chart_override_{chart_key}"] = {
                    "rows": dyn_rows,
                    "question": question,
                }
                st.rerun()   # triggers _render_chart() to swap the chart

    # ── Conversation history display ──────────────────────────────────────────
    hist = st.session_state.get(hist_key, [])
    if hist:
        # Latest answer — expanded green bubble
        latest = hist[-1]
        st.markdown(
            f'<div class="chart-ai-ans">🤖 {latest["a"]}</div>',
            unsafe_allow_html=True,
        )
        # Previous answers — collapsed expanders
        for item in reversed(hist[:-1]):
            label = item["q"][:48] + ("…" if len(item["q"]) > 48 else "")
            with st.expander(f"Q: {label}"):
                st.markdown(item["a"])

        if st.button("✕ clear all", key=f"clr_{chart_key}",
                     type="secondary", use_container_width=False):
            for _k in [hist_key, f"ca_{chart_key}", f"chart_override_{chart_key}"]:
                st.session_state.pop(_k, None)
            st.rerun()


# ── Gemini helpers ────────────────────────────────────────────────────────────

def build_system_prompt(memory_manager: MemoryManager, rag_context: str = "",
                        emp_count: int = 0, source_file: str = "",
                        user_name: str = "", user_role: str = "",
                        user_dept: str = "") -> str:
    count_str   = f"{emp_count:,}" if emp_count else "several thousand"
    dataset_ctx = (
        f"You are currently querying the dataset from file: '{source_file}' "
        f"({count_str} records).\n"
        if source_file else
        f"You have access to all datasets combined ({count_str} total records).\n"
    )

    # Build user identity context so AI knows who it's talking to
    if user_name and user_role:
        role_label = role_info(user_role).get("label", user_role)
        if user_dept:
            identity_ctx = (
                f"The logged-in user is {user_name} ({role_label}, "
                f"{user_dept} department). "
                f"When they say 'my department', they mean '{user_dept}'. "
                f"All employee queries are automatically scoped to the "
                f"'{user_dept}' department for this user.\n"
            )
        else:
            identity_ctx = (
                f"The logged-in user is {user_name} ({role_label}).\n"
            )
    else:
        identity_ctx = ""

    base = (
        "You are a smart, helpful, friendly AI assistant for an enterprise company. "
        "You have been given DIRECT ACCESS to the company's live employee database. "
        f"{dataset_ctx}"
        f"{identity_ctx}"
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
                 source_file: str = "", user_name: str = "",
                 user_role: str = "", user_dept: str = "",
                 company_id: int = 0):
    """Send a message and return (response_text, client, sql_rows)."""
    sql_ctx, sql_rows = "", []
    rag_ctx = ""

    dept_scope = user_dept if user_role == "manager" else ""

    emp_count = sql.employee_count(source_file=source_file,
                                   company_id=company_id) if (sql and sql.ready) else 0

    if sql and sql.ready:
        sql_ctx, sql_rows = sql.query(user_input, source_file=source_file,
                                      dept_filter=dept_scope,
                                      company_id=company_id)
    if rag and rag.ready and not sql_ctx:
        rag_ctx = rag.get_context(user_input)

    combined = "\n\n".join(filter(None, [sql_ctx, rag_ctx]))
    system_prompt = build_system_prompt(
        memory_manager, combined,
        emp_count=emp_count, source_file=source_file,
        user_name=user_name, user_role=user_role, user_dept=user_dept,
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


# ── Login page ────────────────────────────────────────────────────────────────

def _get_sb(sql: SQLEngine):
    """Return a live Supabase client, preferring the already-open sql connection."""
    if sql and sql.ready and sql._sb:
        return sql._sb
    return get_auth_client()


def show_login_page(sql: SQLEngine):
    """Login page — handles normal login AND invite-link registration."""

    sb = _get_sb(sql)

    # ── Check for invite token in URL ─────────────────────────────────────────
    invite_token = st.query_params.get("invite", "")
    invite_data  = None
    if invite_token and sb:
        invite_data = get_invite_token(sb, invite_token)

    # ── Invite registration flow (full screen, no tabs) ───────────────────────
    if invite_data:
        _, col, _ = st.columns([0.5, 2, 0.5])
        with col:
            company_name = invite_data["company_name"]
            st.markdown(f"""
            <div class="login-card">
              <div class="login-logo">🏢</div>
              <div class="login-title">{company_name}</div>
              <div class="login-sub">You've been invited — set up your account below</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            with st.form("invite_setup_form", clear_on_submit=False):
                i_fullname = st.text_input("Your Full Name *", placeholder="Jane Smith")
                i_username = st.text_input("Choose a Username *", placeholder="jane_smith")
                i_password = st.text_input("Set Password * (min 6 chars)", type="password")
                i_confirm  = st.text_input("Confirm Password *", type="password")
                i_submit   = st.form_submit_button(
                    "Create Account →", type="primary", use_container_width=True
                )

            if i_submit:
                if not i_fullname.strip() or not i_username.strip():
                    st.error("Name and username are required.")
                elif len(i_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif i_password != i_confirm:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Setting up your company…"):
                        # Create company
                        import time
                        slug = company_name.lower().replace(" ", "-").replace("'", "")
                        ok_co, err_co, cid = create_company(sb, company_name, slug)
                        if not ok_co:
                            slug = f"{slug}-{int(time.time()) % 10000}"
                            ok_co, err_co, cid = create_company(sb, company_name, slug)

                        if ok_co and cid:
                            ok_u, err_u = create_user(
                                sb,
                                username=i_username.strip(),
                                password=i_password,
                                role="admin",
                                department=None,
                                full_name=i_fullname.strip(),
                                company_id=cid,
                            )
                            if ok_u:
                                use_invite_token(sb, invite_token, i_username.strip())
                                # Clear token from URL and show success
                                st.query_params.clear()
                                st.success(
                                    f"Welcome! **{company_name}** is all set up. "
                                    f"Sign in with username `{i_username.strip()}`."
                                )
                                st.rerun()
                            else:
                                st.error(f"Account creation failed: {err_u}")
                        else:
                            st.error(f"Company setup failed: {err_co}")
        return   # don't show normal login when invite is active

    # ── Normal login ──────────────────────────────────────────────────────────
    _, col, _ = st.columns([0.5, 2, 0.5])
    with col:
        st.markdown("""
        <div class="login-card">
          <div class="login-logo">🤖</div>
          <div class="login-title">AI HR Platform</div>
          <div class="login-sub">Sign in with your credentials to continue</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username  = st.text_input("Username", placeholder="Enter username")
            password  = st.text_input("Password", type="password",
                                      placeholder="Enter password")
            submitted = st.form_submit_button(
                "Sign In →", type="primary", use_container_width=True
            )

        if submitted:
            if not username.strip() or not password:
                st.error("Please enter both username and password.")
                return
            with st.spinner("Signing in…"):
                if sb is None:
                    st.error("Cannot reach the database. Check Supabase credentials.")
                    return
                try:
                    ensure_super_admin(sb)
                except Exception:
                    pass
                user = authenticate(sb, username.strip(), password)

            if user:
                st.session_state["logged_in"]       = True
                st.session_state["user"]            = user
                st.session_state["user_role"]       = user["role"]
                st.session_state["user_dept"]       = user.get("department")
                st.session_state["user_name"]       = user.get("full_name", username)
                st.session_state["user_company_id"] = user.get("company_id")
                st.rerun()
            else:
                st.error("Incorrect username or password.")


# ── Admin panel ───────────────────────────────────────────────────────────────

def render_admin_tab(sql: SQLEngine):
    """Admin panel — scoped by role: super_admin manages companies, admin manages users."""
    if not (sql and sql.ready):
        st.warning("Database not connected.")
        return

    sb         = sql._sb
    viewer_role= st.session_state.get("user_role", "admin")
    company_id = st.session_state.get("user_company_id") or None

    # ═══════════════════════════════════════════════════════════════════════════
    # SUPER ADMIN — company management
    # ═══════════════════════════════════════════════════════════════════════════
    if viewer_role == "super_admin":
        st.markdown("### 🌐 Company Management")

        companies = get_all_companies(sb)
        if companies:
            st.dataframe(pd.DataFrame(companies), use_container_width=True, hide_index=True)
        else:
            st.info("No companies yet.")

        st.divider()

        with st.expander("➕ Create New Company"):
            with st.form("create_company_form", clear_on_submit=True):
                c_name = st.text_input("Company Name *", placeholder="Acme Corp")
                c_slug = st.text_input("Slug * (unique, no spaces)", placeholder="acme")
                c_admin_user = st.text_input("First Admin Username *")
                c_admin_pass = st.text_input("First Admin Password *", type="password")
                c_admin_name = st.text_input("Admin Full Name *")
                c_submit = st.form_submit_button("Create Company + Admin", type="primary")

            if c_submit:
                ok, err, cid = create_company(sb, c_name, c_slug)
                if ok and cid:
                    u_ok, u_err = create_user(sb, c_admin_user, c_admin_pass,
                                              "admin", None, c_admin_name, cid)
                    if u_ok:
                        st.success(f"Company '{c_name}' created (id={cid}) with admin '{c_admin_user}'.")
                        st.rerun()
                    else:
                        st.error(f"Company created but admin failed: {u_err}")
                else:
                    st.error(f"Failed: {err}")

        with st.expander("🗑️ Delete Company"):
            if companies:
                del_opts = {f"{c['name']} (id={c['id']})": c["id"] for c in companies}
                del_sel  = st.selectbox("Select company", list(del_opts.keys()), key="del_co")
                st.warning("This will delete the company and ALL its users. Employee data is kept.")
                if st.button("Delete Company", type="secondary"):
                    ok, err = delete_company(sb, del_opts[del_sel])
                    if ok:
                        st.success("Company deleted.")
                        st.rerun()
                    else:
                        st.error(f"Failed: {err}")

        # ── Invite Link Generator ─────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔗 Invite Links")
        st.caption("Generate a one-time link and share it with a new company. "
                   "They click it, set a username + password, and their account is ready.")

        with st.form("invite_form", clear_on_submit=True):
            inv_company = st.text_input("Company Name *", placeholder="Acme Corp")
            inv_hours   = st.selectbox("Link expires in", [48, 24, 72, 168],
                                       format_func=lambda h: f"{h} hours"
                                       if h < 168 else "7 days")
            inv_submit  = st.form_submit_button("Generate Invite Link", type="primary")

        if inv_submit:
            if not inv_company.strip():
                st.error("Enter a company name.")
            else:
                ok, err, token = create_invite_token(sb, inv_company.strip(), hours=inv_hours)
                if ok:
                    # Build the app URL dynamically
                    try:
                        base_url = st.query_params.get("_base_url", "")
                    except Exception:
                        base_url = ""
                    if not base_url:
                        # Fallback: use the Render URL from env or a placeholder
                        base_url = os.environ.get("APP_URL", "https://your-app.onrender.com")
                    invite_url = f"{base_url}?invite={token}"
                    st.success(f"Invite link created for **{inv_company.strip()}**!")
                    st.code(invite_url, language=None)
                    st.caption(f"Expires in {inv_hours} hours · Single use · "
                               "Share via Teams, WhatsApp, or email.")
                else:
                    st.error(f"Failed: {err}")

        # ── Active invite tokens table ────────────────────────────────────────
        tokens = get_all_invite_tokens(sb)
        if tokens:
            with st.expander(f"📋 All invite links ({len(tokens)})"):
                df_t = pd.DataFrame(tokens)
                df_t["status"] = df_t["used"].map(
                    lambda u: "✅ Used" if u else "⏳ Pending"
                )
                st.dataframe(
                    df_t[["company_name", "status", "used_by", "expires_at", "created_at"]],
                    use_container_width=True, hide_index=True
                )
                revoke_opts = {
                    f"{r['company_name']} (created {str(r['created_at'])[:10]})": r["id"]
                    for r in tokens if not r["used"]
                }
                if revoke_opts:
                    rev_sel = st.selectbox("Revoke a pending link",
                                           list(revoke_opts.keys()), key="rev_inv")
                    if st.button("Revoke", type="secondary"):
                        revoke_invite_token(sb, revoke_opts[rev_sel])
                        st.success("Link revoked.")
                        st.rerun()

        st.divider()
        st.markdown("### 👥 All Users (across all companies)")
        all_users = get_all_users(sb)  # no company_id filter for super_admin

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPANY ADMIN — user management (own company only)
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 👑 User Management")
        all_users = get_all_users(sb, company_id=company_id)

    # ── Users table ───────────────────────────────────────────────────────────
    if all_users:
        df_u = pd.DataFrame(all_users)
        df_u["role_label"] = df_u["role"].map(
            lambda r: f"{ROLES.get(r,{}).get('icon','')} {ROLES.get(r,{}).get('label', r)}"
        )
        display_cols = ["username", "role_label", "department", "full_name", "last_login"]
        if viewer_role == "super_admin":
            display_cols = ["company_id"] + display_cols
        display_cols = [c for c in display_cols if c in df_u.columns]
        st.dataframe(df_u[display_cols].rename(columns={"role_label": "role"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No users found.")

    if viewer_role == "super_admin":
        return   # super_admin doesn't create users here — done per-company above

    st.divider()

    # ── Add new user (company-scoped) ─────────────────────────────────────────
    with st.expander("➕ Add New User"):
        with st.form("add_user_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                nu_username = st.text_input("Username *")
                nu_password = st.text_input("Password * (min 6 chars)", type="password")
                nu_fullname = st.text_input("Full Name *")
            with col2:
                addable_roles = [r for r in ROLES if r != "super_admin"]
                nu_role = st.selectbox(
                    "Role *", addable_roles,
                    format_func=lambda r: f"{ROLES[r]['icon']}  {ROLES[r]['label']}"
                )
                nu_dept = st.text_input(
                    "Department (required for Manager role)",
                    placeholder="e.g. Sales, Finance, R&D"
                )
            nu_submit = st.form_submit_button("Create User", type="primary")

        if nu_submit:
            ok, err = create_user(sb, nu_username, nu_password,
                                  nu_role, nu_dept, nu_fullname,
                                  company_id=company_id)
            if ok:
                st.success(f"User '{nu_username}' created.")
                st.rerun()
            else:
                st.error(f"Failed: {err}")

    # ── Change password ───────────────────────────────────────────────────────
    with st.expander("🔑 Change Password"):
        with st.form("change_pw_form", clear_on_submit=True):
            u_fresh = get_all_users(sb, company_id=company_id)
            opts    = {f"{u['username']} ({u['role']})": u["id"] for u in u_fresh}
            sel_user = st.selectbox("Select user", list(opts.keys()))
            new_pw   = st.text_input("New password (min 6 chars)", type="password")
            pw_submit = st.form_submit_button("Update Password")
        if pw_submit and new_pw:
            ok, err = update_password(sb, opts[sel_user], new_pw)
            if ok:
                st.success("Password updated.")
            else:
                st.error(f"Failed: {err}")

    # ── Delete user ───────────────────────────────────────────────────────────
    with st.expander("🗑️ Delete User"):
        u_fresh2  = get_all_users(sb, company_id=company_id)
        del_opts  = {f"{u['username']} ({u['role']})": u["id"] for u in u_fresh2}
        del_sel   = st.selectbox("Select user to delete", list(del_opts.keys()),
                                 key="del_user_sel")
        if st.button("Delete User", type="secondary"):
            if delete_user(sb, del_opts[del_sel]):
                st.success("User deleted.")
                st.rerun()
            else:
                st.error("Failed to delete user.")


# ── Left panel ────────────────────────────────────────────────────────────────

def render_left(mm: MemoryManager):
    sql: SQLEngine = st.session_state.get("sql")

    # ── User badge ────────────────────────────────────────────────────────────
    user_role    = st.session_state.get("user_role", "hr")
    user_name    = st.session_state.get("user_name", "User")
    user_company = st.session_state.get("user_company_id") or 0
    rcfg         = role_info(user_role)
    st.markdown(
        f'<div class="user-badge">'
        f'<div class="name">{user_name}</div>'
        f'<div class="role">{rcfg["icon"]} {rcfg["label"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

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

    all_datasets = sql.get_source_files(company_id=user_company) if (sql and sql.ready) else []

    # ── Payroll role: restrict to allowed datasets only ───────────────────────
    allowed_ds = rcfg.get("datasets")   # None = all; list = restricted
    if allowed_ds is not None:
        datasets = [d for d in all_datasets if d["source_file"] in allowed_ds]
        if not datasets and all_datasets:
            st.markdown(
                '<div class="setup-box">⚠️ Your role only has access to '
                f'<b>{", ".join(allowed_ds)}</b>. '
                'Upload that dataset first.</div>',
                unsafe_allow_html=True,
            )
    else:
        datasets = all_datasets

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

    st.divider()

    # ── Sign out ──────────────────────────────────────────────────────────────
    if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
        for _k in ["logged_in", "user", "user_role", "user_dept", "user_name",
                   "analytics_data", "active_source_file"]:
            st.session_state.pop(_k, None)
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
                                     source_file=dataset_name,
                                     company_id=st.session_state.get("user_company_id") or 1)
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

def render_analytics(sql: SQLEngine, source_file: str = "",
                     role: str = "hr", dept_filter: str = ""):
    """
    role        — controls salary chart visibility
    dept_filter — non-empty string forces a department sub-filter (manager role)
    """
    can_salary = role_info(role).get("can_see_salary", True)

    # For manager role, enforce their department as the source_file filter
    # by appending a note; the actual SQL filtering is done via source_file
    # (managers see all datasets but analytics are scoped to their dept via
    # a secondary WHERE added below when dept_filter is set)
    effective_sf = source_file   # used for analytics queries

    label = f"📊 Analytics — {source_file}" if source_file else "📊 HR Analytics Dashboard"
    if dept_filter:
        label += f" · {dept_filter} only"
    st.markdown(f"### {label}")

    if not (sql and sql.ready):
        st.warning("⚠️ Analytics require a live Supabase connection.")
        return

    if not can_salary:
        st.info(
            "ℹ️ Salary data is not visible for your role. "
            "Contact HR or Admin if you need access."
        )

    col_refresh, col_space = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.pop("analytics_data", None)

    if "analytics_data" not in st.session_state:
        with st.spinner("Loading analytics data…"):
            st.session_state.analytics_data = sql.get_analytics_data(
                source_file=source_file,
                company_id=st.session_state.get("user_company_id") or 0
            )

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
            fig = analytics_bar(df, "department", "employees",
                                "👥 Employees by Department",
                                color_scale=[[0,"#1e3a6e"],[1,"#4f7fff"]])
            _render_chart("dept_headcount", fig, rows, "Employees by Department",
                          source_file, sql)
    with c2:
        rows = data.get("dept_attrition", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_bar(df, "department", "attrition_pct",
                                "📉 Attrition Rate by Department (%)",
                                color_scale=[[0,"#3d1a1a"],[1,"#f87171"]])
            _render_chart("dept_attrition", fig, rows,
                          "Attrition Rate % by Department", source_file, sql)

    # ── Row 2: Salary (role-gated) + Age groups ──────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        if can_salary:
            rows = data.get("dept_salary", [])
            if rows:
                df  = pd.DataFrame(rows)
                fig = analytics_bar(df, "department", "avg_salary",
                                    "💰 Avg Monthly Salary by Department ($)",
                                    color_scale=[[0,"#103528"],[1,"#34d399"]])
                _render_chart("dept_salary", fig, rows,
                              "Avg Monthly Salary by Department", source_file, sql)
        else:
            st.markdown(
                '<div class="access-denied">'
                '🔒 Salary data hidden for your role</div>',
                unsafe_allow_html=True,
            )
    with c4:
        rows = data.get("age_groups", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = px.bar(df, x="age_group", y="employees",
                         title="📅 Age Distribution",
                         template="plotly_dark", color="employees",
                         color_continuous_scale=[[0,"#2d1f6e"],[1,"#8b5cf6"]])
            fig.update_layout(**_DARK)
            fig.update_traces(marker_line_width=0)
            _render_chart("age_groups", fig, rows,
                          "Age Distribution of Employees", source_file, sql)

    # ── Row 3: Gender + Job Satisfaction ─────────────────────────────────────
    c5, c6 = st.columns(2)
    with c5:
        rows = data.get("gender", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_pie(df, "gender", "employees",
                                "👫 Male / Female Distribution")
            _render_chart("gender", fig, rows,
                          "Male / Female Distribution", source_file, sql)
    with c6:
        rows = data.get("satisfaction", [])
        if rows:
            df = pd.DataFrame(rows)
            label_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
            df["job_satisfaction"] = df["job_satisfaction"].map(label_map)
            fig = px.bar(df, x="job_satisfaction", y="employees",
                         title="😊 Job Satisfaction",
                         template="plotly_dark", color="employees",
                         color_continuous_scale=[[0,"#1a2a1a"],[1,"#34d399"]])
            fig.update_layout(**_DARK)
            fig.update_traces(marker_line_width=0)
            _render_chart("satisfaction", fig, df.to_dict(orient="records"),
                          "Job Satisfaction Distribution", source_file, sql)

    # ── Row 4: Overtime + Education ───────────────────────────────────────────
    c7, c8 = st.columns(2)
    with c7:
        rows = data.get("overtime", [])
        if rows:
            df  = pd.DataFrame(rows)
            fig = analytics_pie(df, "overtime", "employees",
                                "⏰ Overtime Distribution")
            _render_chart("overtime", fig, rows,
                          "Overtime Distribution", source_file, sql)
    with c8:
        rows = data.get("education", [])
        if rows:
            df = pd.DataFrame(rows)
            edu_map = {1: "Below College", 2: "College", 3: "Bachelor",
                       4: "Master", 5: "Doctor"}
            df["education"] = df["education"].map(edu_map)
            fig = px.bar(df, x="education", y="employees",
                         title="🎓 Education Level",
                         template="plotly_dark", color="employees",
                         color_continuous_scale=[[0,"#2a1a3d"],[1,"#a78bfa"]])
            fig.update_layout(**_DARK)
            fig.update_traces(marker_line_width=0)
            _render_chart("education", fig, df.to_dict(orient="records"),
                          "Education Level Distribution", source_file, sql)

    # ── Row 5: Gender by Department (full-width) ──────────────────────────────
    rows = data.get("gender_by_dept", [])
    if rows:
        df  = pd.DataFrame(rows)
        fig = px.bar(df, x="department", y="employees", color="gender",
                     barmode="group", title="👫 Male / Female by Department",
                     template="plotly_dark",
                     color_discrete_map={"Male": "#4f7fff", "Female": "#a78bfa"})
        fig.update_layout(**{**_DARK, "showlegend": True, "height": 320,
                             "legend": dict(
                                 bgcolor="rgba(0,0,0,0)",
                                 font_color="#94a3b8",
                             )})
        fig.update_traces(marker_line_width=0)
        _render_chart("gender_by_dept", fig, rows,
                      "Gender Breakdown by Department", source_file, sql)


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
    # Re-init if a previously-cached instance failed to connect (stale session state)
    if not st.session_state.sql.ready:
        st.session_state.sql = SQLEngine()
    if not st.session_state.rag.ready:
        st.session_state.rag = RAGEngine()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_sid" not in st.session_state:
        st.session_state.current_sid = None
    if "title_generated" not in st.session_state:
        st.session_state.title_generated = False

    mm:  MemoryManager = st.session_state.memory_manager
    sql: SQLEngine     = st.session_state.sql
    rag: RAGEngine     = st.session_state.rag

    # ── Auth gate ─────────────────────────────────────────────────────────────
    if not st.session_state.get("logged_in"):
        show_login_page(sql)
        return

    # ── Role context ──────────────────────────────────────────────────────────
    user_role    = st.session_state.get("user_role", "hr")
    user_dept    = st.session_state.get("user_dept") or ""
    user_name    = st.session_state.get("user_name", "User")
    user_company = st.session_state.get("user_company_id") or 0   # 0 = super_admin (no company)
    rcfg         = role_info(user_role)
    allowed_tabs = rcfg["tabs"]

    left_col, right_col = st.columns([1, 3], gap="small")

    with left_col:
        render_left(mm)

    with right_col:
        # Build tab list dynamically from role config
        tab_labels = []
        if "chat"      in allowed_tabs: tab_labels.append("💬 Chat")
        if "analytics" in allowed_tabs: tab_labels.append("📊 Analytics")
        if "upload"    in allowed_tabs: tab_labels.append("📤 Upload Data")
        if "admin"     in allowed_tabs: tab_labels.append("👑 Admin")

        all_tabs = st.tabs(tab_labels)
        tab_map  = {label: tab for label, tab in zip(tab_labels, all_tabs)}

        # ── Chat tab ─────────────────────────────────────────────────────────
        if "💬 Chat" in tab_map:
          with tab_map["💬 Chat"]:
            messages = st.session_state.messages

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

            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        sf_label = msg.get("source_file", "")
                        badge = sf_label if sf_label else "All datasets"
                        st.markdown(
                            f'<div class="ds-badge">📂 {badge}</div>',
                            unsafe_allow_html=True,
                        )
                    chart_data = msg.get("chart_data", [])
                    if chart_data and len(chart_data) > 1:
                        fig = make_chart(chart_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

            user_input = st.chat_input("Ask me anything…")
            submitted  = bool(user_input)

        # ── Analytics tab ─────────────────────────────────────────────────────
        if "📊 Analytics" in tab_map:
          with tab_map["📊 Analytics"]:
            active_sf  = st.session_state.get("active_source_file", "")
            dept_scope = user_dept if rcfg.get("dept_filter") else ""
            render_analytics(sql, source_file=active_sf,
                             role=user_role, dept_filter=dept_scope)

        # ── Upload tab ────────────────────────────────────────────────────────
        if "📤 Upload Data" in tab_map:
          with tab_map["📤 Upload Data"]:
            render_upload_tab(sql)

        # ── Admin tab ─────────────────────────────────────────────────────────
        if "👑 Admin" in tab_map:
          with tab_map["👑 Admin"]:
            render_admin_tab(sql)

    # ── Handle quick prompt ───────────────────────────────────────────────────
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.pop("quick_prompt")
        submitted  = True
    else:
        # user_input / submitted may not exist if Chat tab isn't in role's tabs
        user_input = locals().get("user_input")
        submitted  = locals().get("submitted", False)

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
        with tab_map.get("💬 Chat", st.container()):
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        active_sf = st.session_state.get("active_source_file", "")
                        response_text, client, sql_rows = chat_with_ai(
                            mm, user_input, sid,
                            rag=rag, sql=sql,
                            source_file=active_sf,
                            user_name=user_name,
                            user_role=user_role,
                            user_dept=user_dept,
                            company_id=user_company,
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
