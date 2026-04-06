import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import json, io
from datetime import datetime

st.set_page_config(page_title="DataPrep Studio", page_icon="🧪",
                   layout="wide", initial_sidebar_state="collapsed")

# ─── Session State ─────────────────────────────────────────────────────────────
for k, v in {
    'screen': 'landing', 'working_df': None, 'original_df': None,
    'file_name': '', 'history': [], 'df_snapshots': [], 'recipe': [],
    'violations_df': None, 'viz_selected': None, 'clean_section': None,
    'viz_palette': 'Sky Blue',
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_num(df):  return df.select_dtypes(include='number').columns.tolist()
def get_cat(df):  return df.select_dtypes(include=['object','category']).columns.tolist()
def get_dt(df):   return df.select_dtypes(include=['datetime','datetimetz']).columns.tolist()
def shape_str(df): return f"{df.shape[0]:,} rows × {df.shape[1]} cols"

def classify_col(series):
    dtype = str(series.dtype)
    if 'datetime' in dtype: return 'datetime'
    if dtype in ('int64','float64','int32','float32') or 'int' in dtype or 'float' in dtype:
        return 'numerical'
    if dtype == 'bool': return 'categorical'
    if hasattr(series, 'cat') or dtype == 'category': return 'categorical'
    if series.dtype == object:
        sample = series.dropna().head(20).astype(str)
        dt_hits = 0
        for v in sample:
            try:
                pd.to_datetime(v); dt_hits += 1
            except: pass
        if dt_hits / max(len(sample),1) > 0.7: return 'datetime'
        uniq_ratio = series.nunique() / max(len(series), 1)
        if uniq_ratio > 0.5 and series.str.len().mean() > 20: return 'text'
        return 'categorical'
    return 'categorical'

COL_TYPE_META = {
    'numerical':  {'label': 'Numerical',   'icon': '🔢', 'badge_bg': '#E0F2FE', 'badge_fg': '#0284C7'},
    'categorical':{'label': 'Categorical', 'icon': '🔠', 'badge_bg': '#FEF3C7', 'badge_fg': '#D97706'},
    'datetime':   {'label': 'Datetime',    'icon': '📅', 'badge_bg': '#EDE9FE', 'badge_fg': '#7C3AED'},
    'text':       {'label': 'Text',        'icon': '📝', 'badge_bg': '#F0FDF4', 'badge_fg': '#16A34A'},
}

def col_types_dict(df):
    return {c: classify_col(df[c]) for c in df.columns}

def push_history(desc, sb, sa, re=None):
    st.session_state['history'].append({
        'step': len(st.session_state['history'])+1,
        'description': desc, 'shape_before': sb, 'shape_after': sa,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
    })
    if re: st.session_state['recipe'].append(re)

def save_snap(): st.session_state['df_snapshots'].append(st.session_state['working_df'].copy())

def commit(new_df, desc, sb, re=None):
    sa = shape_str(new_df)
    st.session_state['working_df'] = new_df
    push_history(desc, sb, sa, re)
    return sa

def load_df(f, sample=False):
    if sample:
        np.random.seed(42); n = 300
        df = pd.DataFrame({
            'age':        np.where(np.random.rand(n)<.1, np.nan, np.random.randint(18,70,n).astype(float)),
            'salary':     ['$'+str(int(x)) for x in np.random.normal(55000,15000,n)],
            'score':      np.random.uniform(0,100,n),
            'department': np.random.choice(['HR','Engineering','Sales','engineering ','hr',None],n),
            'bonus_pct':  [f"{round(x,1)}%" for x in np.random.uniform(0,20,n)],
            'years_exp':  np.random.randint(0,30,n).astype(float),
            'rating':     np.random.choice([1,2,3,4,5,None],n),
            'hire_date':  pd.date_range('2010-01-01',periods=n,freq='D').astype(str),
            'region':     np.random.choice(['North','South','East','West'],n),
            'notes':      ['Employee note: ' + ' '.join(['word']*np.random.randint(3,15)) for _ in range(n)],
        })
        df.loc[np.random.choice(n,5,replace=False),'age'] = 150
        df.loc[np.random.choice(n,3,replace=False),'score'] = -999
        return df, 'sample_data.csv'
    name = f.name
    if name.endswith('.csv'):    return pd.read_csv(f), name
    elif name.endswith('.json'): return pd.read_json(f), name
    else:                        return pd.read_excel(f), name

# ─── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    sidebar_on = st.session_state['screen'] == 'studio'
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body {{ box-sizing:border-box; margin:0; padding:0; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: #F8FAFC !important;
    font-family: 'Inter', -apple-system, 'Apple SD Gothic Neo', sans-serif !important;
    color: #1E293B !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
    display: {'flex' if sidebar_on else 'none'} !important;
    background: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important;
}}
[data-testid="stSidebarContent"] {{ padding: 1.2rem 1rem !important; }}

/* Buttons */
.stButton > button {{
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 0.875rem !important; transition: all 0.18s ease !important;
    border-radius: 8px !important;
}}
.btn-primary .stButton > button {{
    background: #0EA5E9 !important; color: #fff !important; border: none !important;
    padding: 0.55rem 1.4rem !important; box-shadow: 0 1px 3px rgba(14,165,233,0.3) !important;
}}
.btn-primary .stButton > button:hover {{
    background: #0284C7 !important; box-shadow: 0 4px 12px rgba(14,165,233,0.35) !important;
    transform: translateY(-1px) !important;
}}
.btn-ghost .stButton > button {{
    background: transparent !important; color: #64748B !important;
    border: 1.5px solid #E2E8F0 !important; padding: 0.45rem 1rem !important;
    box-shadow: none !important;
}}
.btn-ghost .stButton > button:hover {{
    border-color: #0EA5E9 !important; color: #0EA5E9 !important;
    transform: none !important; box-shadow: none !important; background: transparent !important;
}}
.btn-sample .stButton > button {{
    background: transparent !important; color: #0EA5E9 !important;
    border: 1.5px solid #BAE6FD !important; padding: 0.45rem 1.1rem !important;
    box-shadow: none !important; font-weight: 500 !important;
}}
.btn-sample .stButton > button:hover {{
    background: #F0F9FF !important; border-color: #0EA5E9 !important;
    transform: none !important; box-shadow: none !important;
}}
.sidebar-btn .stButton > button {{
    background: transparent !important; color: #475569 !important;
    border: 1px solid #E2E8F0 !important; width: 100% !important;
    text-align: left !important; padding: 0.5rem 0.8rem !important;
    font-size: 0.82rem !important; box-shadow: none !important;
}}
.sidebar-btn .stButton > button:hover {{
    background: #F1F5F9 !important; border-color: #CBD5E1 !important;
    transform: none !important; box-shadow: none !important; color: #1E293B !important;
}}
.btn-run .stButton > button {{
    background: #0EA5E9 !important; color: #fff !important; border: none !important;
    padding: 0.5rem 1.2rem !important; box-shadow: 0 1px 3px rgba(14,165,233,0.25) !important;
}}
.btn-run .stButton > button:hover {{
    background: #0284C7 !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(14,165,233,0.3) !important;
}}
.btn-download .stDownloadButton > button {{
    background: #0EA5E9 !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.875rem !important; width: 100% !important;
    box-shadow: 0 1px 3px rgba(14,165,233,0.3) !important; transition: all 0.18s !important;
}}
.btn-download .stDownloadButton > button:hover {{
    background: #0284C7 !important; box-shadow: 0 4px 12px rgba(14,165,233,0.4) !important;
    transform: translateY(-1px) !important;
}}
.btn-download-purple .stDownloadButton > button {{
    background: #8B5CF6 !important;
    border: none !important; border-radius: 8px !important; padding: 0.6rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.875rem !important; width: 100% !important; color: #fff !important;
    box-shadow: 0 1px 3px rgba(139,92,246,0.3) !important; transition: all 0.18s !important;
}}
.btn-download-purple .stDownloadButton > button:hover {{
    background: #7C3AED !important; box-shadow: 0 4px 12px rgba(139,92,246,0.4) !important;
    transform: translateY(-1px) !important;
}}
.btn-download-indigo .stDownloadButton > button {{
    background: #6366F1 !important;
    border: none !important; border-radius: 8px !important; padding: 0.6rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.875rem !important; width: 100% !important; color: #fff !important;
    box-shadow: 0 1px 3px rgba(99,102,241,0.3) !important; transition: all 0.18s !important;
}}
.btn-download-indigo .stDownloadButton > button:hover {{
    background: #4F46E5 !important; box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}}

/* ══ Force ALL Streamlit red/orange → theme blue #0EA5E9 ══ */

/* Primary button */
.stButton > button[kind="primary"] {{
    background: #0EA5E9 !important; border-color: #0EA5E9 !important;
}}
.stButton > button[kind="secondary"] {{
    background: transparent !important; color: #475569 !important;
    border-color: #E2E8F0 !important;
}}

/* ── Radio: JS handles dot color to avoid collateral blue backgrounds ── */

/* ── Slider: thumb, filled track, tooltip ── */
/* Thumb (handle) */
div[data-baseweb="slider"] [role="slider"] {{
    background: #0EA5E9 !important;
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 0 4px rgba(14,165,233,.18) !important;
}}
/* Filled portion of track (the colored left part) */
div[data-baseweb="slider"] > div > div > div:nth-child(4),
div[data-baseweb="slider"] > div > div > div:nth-child(3) {{
    background: #0EA5E9 !important;
}}
/* All inner track-fill divs — belt-and-suspenders */
div[data-baseweb="slider"] div[style*="background-color: rgb(255"] {{
    background-color: #0EA5E9 !important;
}}
div[data-baseweb="slider"] div[style*="background: rgb(255"] {{
    background: #0EA5E9 !important;
}}
/* Tooltip bubble above thumb */
div[data-baseweb="slider"] [data-baseweb="tooltip"] div {{
    background: #0EA5E9 !important;
}}
/* Streamlit's own slider inner fill via CSS var override */
[data-testid="stSlider"] * {{ --primary: #0EA5E9 !important; }}

/* ── Selectbox / multiselect tags ── */
[data-baseweb="select"] [data-baseweb="tag"] {{
    background-color: #E0F2FE !important; color: #0284C7 !important;
}}
[data-baseweb="select"] [data-baseweb="tag"] span {{ color: #0284C7 !important; }}
[data-baseweb="tag"] [data-baseweb="tag-action"] svg path {{ fill: #0284C7 !important; }}

/* ── Input focus ── */
input[type="number"]:focus, input[type="text"]:focus, textarea:focus {{
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 0 2px rgba(14,165,233,.15) !important;
}}
/* ── Checkbox ── */
[data-baseweb="checkbox"] [data-checked="true"] span {{
    background-color: #0EA5E9 !important; border-color: #0EA5E9 !important;
}}
/* ── Progress bar ── */
.stProgress > div > div {{ background-color: #0EA5E9 !important; }}
/* ── Streamlit global CSS variable (overrides theme token) ── */
:root {{
    --primary-color: #0EA5E9 !important;
    --secondary-color: #0284C7 !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important; border-bottom: 1px solid #E2E8F0 !important;
    gap: 0 !important; padding: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; padding: 0.6rem 1.2rem !important;
    font-weight: 500 !important; font-size: 0.875rem !important;
    color: #64748B !important; transition: all 0.18s !important; margin-bottom: -1px !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: #0EA5E9 !important; background: transparent !important; }}
.stTabs [aria-selected="true"] {{
    background: transparent !important; color: #0EA5E9 !important;
    border-bottom: 2px solid #0EA5E9 !important; font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab"]::after {{ display:none !important; }}
[data-testid="stTabsContent"] {{ padding-top: 1.2rem !important; }}

/* Inputs */
.stSelectbox>div>div, .stMultiSelect>div>div {{
    background: #F8FAFC !important; border-color: #E2E8F0 !important;
    border-radius: 8px !important; font-size: 0.875rem !important;
}}
input, textarea {{ background: #F8FAFC !important; border-radius:8px !important; font-size: 0.875rem !important; }}
[data-testid="stRadio"] label {{ color:#475569 !important; font-size:0.875rem !important; }}
[data-testid="stCheckbox"] label {{ color:#475569 !important; font-size:0.875rem !important; }}
[data-testid="stMetricValue"] {{ color:#0EA5E9 !important; font-weight:700 !important; }}
[data-testid="stDataFrame"] {{ border-radius:10px !important; overflow:hidden !important; border:1px solid #E2E8F0 !important; }}
.stCaption {{ color:#94A3B8 !important; font-size:0.78rem !important; }}

/* Step bar */
.stepbar {{
    display:flex; align-items:center; justify-content:center;
    gap:0; padding:0.8rem 0 1.6rem;
}}
.step-wrap {{ display:flex; align-items:center; gap:0.45rem; }}
.step-dot {{
    width:30px; height:30px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-weight:700; font-size:0.78rem;
    transition: all 0.2s;
}}
.dot-active {{ background:#0EA5E9; color:#fff; box-shadow:0 0 0 3px rgba(14,165,233,0.2); }}
.dot-done {{ background:#BAE6FD; color:#0284C7; }}
.dot-idle {{ background:#fff; color:#CBD5E1; border:1.5px solid #E2E8F0; }}
.step-lbl {{ font-size:0.78rem; font-weight:500; transition: color 0.2s; }}
.lbl-active {{ color:#0EA5E9; font-weight:600; }}
.lbl-done   {{ color:#94A3B8; }}
.lbl-idle   {{ color:#CBD5E1; }}
.step-line {{ width:52px; height:1.5px; margin:0 0.35rem; }}
.line-done {{ background:#BAE6FD; }}
.line-idle {{ background:#E2E8F0; }}

/* Stat cards */
.stat-box {{ background:#fff; border:1px solid #E2E8F0; border-radius:10px;
    padding:1rem 0.8rem; text-align:center; }}
.stat-num {{ font-size:1.75rem; font-weight:700; color:#0EA5E9;
    font-variant-numeric:tabular-nums; line-height:1.1; }}
.stat-lbl {{ font-size:0.68rem; font-weight:500; color:#94A3B8;
    text-transform:uppercase; letter-spacing:0.07em; margin-top:0.25rem; }}
.stat-sub {{ font-size:0.72rem; color:#64748B; margin-top:0.2rem; }}

/* Health check card */
.health-card {{
    background:#fff; border:1px solid #E2E8F0; border-radius:12px;
    padding:1.2rem 1.5rem; margin-bottom:1rem;
}}
.health-row {{
    display:flex; align-items:center; justify-content:space-between;
    padding:0.45rem 0; border-bottom:1px solid #F8FAFC;
    font-size:0.85rem;
}}
.health-row:last-child {{ border-bottom:none; }}
.health-key {{ color:#64748B; font-weight:500; }}
.health-val {{ color:#1E293B; font-weight:600; }}

/* Progress bar */
.miss-prog-wrap {{ display:flex; align-items:center; gap:0.5rem; }}
.miss-prog-bar {{
    flex:1; height:5px; background:#F1F5F9; border-radius:3px; overflow:hidden;
}}
.miss-prog-fill {{ height:100%; border-radius:3px; transition:width .3s; }}

/* Column info table */
.col-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
.col-table th {{
    padding:0.5rem 0.75rem; text-align:left; color:#94A3B8;
    font-weight:500; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:0.06em; border-bottom:1px solid #F1F5F9;
    background:#FAFBFC; position:sticky; top:0;
}}
.col-table td {{ padding:0.55rem 0.75rem; border-bottom:1px solid #F8FAFC;
    color:#475569; vertical-align:middle; }}
.col-table tr:hover td {{ background:#FAFBFC; }}

/* Type badges */
.badge {{
    display:inline-flex; align-items:center; gap:0.25rem;
    padding:0.18rem 0.55rem; border-radius:5px;
    font-size:0.7rem; font-weight:600;
}}
.badge-numerical {{ background:#E0F2FE; color:#0284C7; }}
.badge-categorical {{ background:#FEF3C7; color:#D97706; }}
.badge-datetime   {{ background:#EDE9FE; color:#7C3AED; }}
.badge-text       {{ background:#DCFCE7; color:#16A34A; }}

.miss-ok  {{ color:#16A34A; }}
.miss-warn {{ color:#D97706; }}
.miss-crit {{ color:#DC2626; font-weight:600; }}

.sec-hdr {{ font-size:0.9rem; font-weight:600; color:#334155;
    border-left:3px solid #0EA5E9; padding-left:0.65rem; margin:1rem 0 0.65rem; }}
.shape-badge {{ display:inline-block; background:#F0F9FF;
    border:1px solid #BAE6FD; border-radius:6px;
    padding:0.3rem 0.8rem; font-size:0.78rem; color:#0284C7;
    font-family:'Inter',monospace; margin:0.15rem; }}
.hist-item {{ background:#F8FAFC; border-left:2px solid #0EA5E9;
    padding:0.35rem 0.75rem; margin:0.18rem 0; border-radius:0 6px 6px 0;
    font-size:0.72rem; color:#64748B; line-height:1.4; }}

/* ── Card styles for Cleaning & Visualization ── */
.func-card {{
    background: #fff;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.1rem 1rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    margin-bottom: 0.1rem;
}}
.func-card-icon {{
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; background: #F0F9FF;
    margin-bottom: 0.1rem;
}}
.func-card-title {{ font-size: 0.9rem; font-weight: 700; color: #1E293B; }}
.func-card-desc  {{ font-size: 0.73rem; color: #94A3B8; line-height: 1.45; min-height: 2.6rem; }}

/* Real visible button inside each card */
.func-card-btn .stButton > button {{
    background: #F0F9FF !important;
    color: #0EA5E9 !important;
    border: 1.5px solid #BAE6FD !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.9rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    box-shadow: none !important;
    transition: all .18s ease !important;
}}
.func-card-btn .stButton > button:hover {{
    background: #0EA5E9 !important;
    color: #fff !important;
    border-color: #0EA5E9 !important;
    transform: none !important;
    box-shadow: 0 2px 8px rgba(14,165,233,.25) !important;
}}

/* Color palette picker */
.palette-row {{
    display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
    margin: 0.3rem 0 0.8rem;
}}
.palette-swatch {{
    width: 22px; height: 22px; border-radius: 50%;
    cursor: pointer; border: 2px solid transparent;
    transition: transform .15s, border-color .15s;
    display: inline-block;
}}
.palette-swatch:hover {{ transform: scale(1.2); }}
.palette-swatch.selected {{ border-color: #1E293B !important; transform: scale(1.18); }}
/* Invisible palette picker buttons */
[key^="pal_"] .stButton > button, .stButton > button[data-testid*="pal_"] {{
    opacity: 0 !important; position: absolute !important;
    inset: 0 !important; width: 100% !important; height: 100% !important;
    border: none !important; background: transparent !important;
    cursor: pointer !important;
}}

/* Section detail header: title left, back button RIGHT — same row */
.sec-detail-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.2rem 0;
    margin-bottom: 0.2rem;
}}
.sec-detail-left {{
    display: flex; align-items: center; gap: 0.7rem;
}}
.sec-detail-icon {{
    width: 36px; height: 36px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; background: #F0F9FF; flex-shrink: 0;
}}
/* Back button inside header row */
.sec-back-btn .stButton > button {{
    background: transparent !important; color: #64748B !important;
    border: 1.5px solid #E2E8F0 !important; padding: 0.3rem 0.85rem !important;
    font-size: 0.8rem !important; box-shadow: none !important;
    white-space: nowrap !important;
}}
.sec-back-btn .stButton > button:hover {{
    border-color: #0EA5E9 !important; color: #0EA5E9 !important;
    background: transparent !important; transform: none !important;
}}

/* Export cards */
.exp-card {{ background:#fff; border:1px solid #E2E8F0; border-radius:12px;
    padding:2rem 1.5rem 1.5rem; text-align:center; }}
.exp-icon {{ font-size:2.2rem; margin-bottom:0.75rem; }}
.exp-title {{ font-size:1rem; font-weight:600; color:#1E293B; margin-bottom:0.2rem; }}
.exp-sub {{ font-size:0.78rem; color:#94A3B8; margin-bottom:1.2rem; }}

/* Upload */
.upload-zone {{ background:#fff; border:2px dashed #CBD5E1; border-radius:16px;
    padding:2.5rem 2rem 1.8rem; text-align:center; transition:border-color .2s; }}
.fmt-tag {{ display:inline-block; padding:0.18rem 0.6rem;
    background:#F1F5F9; border:1px solid #E2E8F0; border-radius:5px;
    font-size:0.72rem; color:#64748B; margin:0.15rem; font-family:'Inter',monospace; }}
.landing-title {{
    font-size:2.8rem; font-weight:700; text-align:center;
    background:linear-gradient(135deg,#0EA5E9,#6366F1);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1.2; margin-bottom:0.6rem;
}}
.landing-sub {{ text-align:center; color:#64748B; font-size:1rem; margin-bottom:1.8rem; }}
</style>
<script>
(function patchRed() {{
    var BLUE = '#0EA5E9';
    var RED_PATTERNS = [
        'rgb(255, 75, 75)', 'rgb(255,75,75)',
        '#ff4b4b', '#FF4B4B', 'red',
        'rgb(255, 49, 49)', 'rgb(229, 61, 61)'
    ];

    function isRed(v) {{
        if (!v) return false;
        for (var i = 0; i < RED_PATTERNS.length; i++) {{
            if (v === RED_PATTERNS[i]) return true;
        }}
        // catch rgb(255, XX, XX) pattern broadly
        return /rgb\(25[0-9],\s*[0-9]{{1,2}},\s*[0-9]{{1,2}}\)/.test(v);
    }}

    function fixEl(el) {{
        if (!el || el.nodeType !== 1) return;
        var props = ['backgroundColor','background','borderColor',
                     'borderTopColor','borderBottomColor','borderLeftColor',
                     'borderRightColor','fill','stroke'];
        props.forEach(function(p) {{
            if (isRed(el.style[p])) el.style[p] = BLUE;
        }});
        // SVG fill attribute
        var fillAttr = el.getAttribute && el.getAttribute('fill');
        if (fillAttr && isRed(fillAttr)) el.setAttribute('fill', BLUE);
    }}

    function runAll() {{
        document.querySelectorAll('*').forEach(fixEl);
    }}

    var mo = new MutationObserver(function(muts) {{
        muts.forEach(function(m) {{
            m.addedNodes.forEach(function(n) {{
                if (n.nodeType === 1) {{
                    fixEl(n);
                    n.querySelectorAll('*').forEach(fixEl);
                }}
            }});
            if (m.type === 'attributes') fixEl(m.target);
        }});
    }});

    function init() {{
        mo.observe(document.body, {{
            childList: true, subtree: true,
            attributes: true, attributeFilter: ['style','fill','stroke']
        }});
        runAll();
    }}

    if (document.body) init();
    else document.addEventListener('DOMContentLoaded', init);
    setTimeout(runAll, 400);
    setTimeout(runAll, 1200);
    setTimeout(runAll, 3000);
}})();
</script>
""", unsafe_allow_html=True)

inject_css()

# ─── Step Bar ──────────────────────────────────────────────────────────────────
def step_bar(active):
    def dot(n):
        if n == active: cls = "dot-active"
        elif n < active: cls = "dot-done"
        else: cls = "dot-idle"
        lbl = "✓" if n < active else str(n)
        return f'<div class="step-dot {cls}">{lbl}</div>'
    def lbl(name, n):
        if n == active: cls = "lbl-active"
        elif n < active: cls = "lbl-done"
        else: cls = "lbl-idle"
        return f'<span class="step-lbl {cls}">{name}</span>'
    def line(n):
        cls = "line-done" if n < active else "line-idle"
        return f'<div class="step-line {cls}"></div>'
    return (f'<div class="stepbar">'
            f'<div class="step-wrap">{dot(1)}{lbl("Upload",1)}</div>{line(1)}'
            f'<div class="step-wrap">{dot(2)}{lbl("Overview",2)}</div>{line(2)}'
            f'<div class="step-wrap">{dot(3)}{lbl("Studio",3)}</div></div>')

def sec(title):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)

def shape_delta(sb, sa):
    arrow = '<span style="color:#94A3B8;margin:0 0.4rem">→</span>'
    st.markdown(f'<div class="shape-badge">Before: {sb}</div>{arrow}'
                f'<div class="shape-badge">After: {sa}</div>', unsafe_allow_html=True)

def miss_color_cls(pct):
    if pct == 0: return "miss-ok"
    if pct < 10: return "miss-warn"
    return "miss-crit"

def progress_bar_html(pct, color):
    return (f'<div class="miss-prog-wrap">'
            f'<div class="miss-prog-bar">'
            f'<div class="miss-prog-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div>'
            f'</div>'
            f'<span style="font-size:0.72rem;color:{color};font-weight:600;min-width:3rem">{pct:.1f}%</span>'
            f'</div>')

def type_badge_html(t):
    m = COL_TYPE_META.get(t, COL_TYPE_META['categorical'])
    return f'<span class="badge badge-{t}">{m["icon"]} {m["label"]}</span>'

# ──────────────────────────────────────────────────────────────────────────────
# SCREEN 1 — LANDING
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state['screen'] == 'landing':
    st.markdown(step_bar(1), unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown('<div class="landing-title">Your Personal<br>DataPrep Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="landing-sub">Upload your dataset — clean, explore, and export in minutes.</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:1.2rem">', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=['csv','xlsx','xls','json'], label_visibility="collapsed")
        st.markdown('<div style="margin-top:0.6rem">', unsafe_allow_html=True)
        if uploaded:
            st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            if st.button("Continue", use_container_width=True, key="continue_btn"):
                try:
                    df, fname = load_df(uploaded)
                    st.session_state.update({'working_df':df,'original_df':df.copy(),'file_name':fname,
                        'history':[],'df_snapshots':[],'recipe':[],'screen':'overview',
                        'clean_section': None, 'viz_selected': None})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        _, right_btn = st.columns([3, 1])
        with right_btn:
            st.markdown('<div class="btn-sample" style="margin-top:0.6rem">', unsafe_allow_html=True)
            if st.button("Random Dataset", key="sample_btn", use_container_width=True):
                df, fname = load_df(None, sample=True)
                st.session_state.update({'working_df':df,'original_df':df.copy(),'file_name':fname,
                    'history':[],'df_snapshots':[],'recipe':[],'screen':'overview',
                    'clean_section': None, 'viz_selected': None})
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SCREEN 2 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
elif st.session_state['screen'] == 'overview':
    df = st.session_state['working_df']
    fname = st.session_state['file_name']
    st.markdown(step_bar(2), unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.25rem">'
        f'<h2 style="font-size:1.4rem;font-weight:700;color:#1E293B;margin:0">Overview</h2>'
        f'<span style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:6px;'
        f'padding:0.2rem 0.7rem;font-size:0.75rem;color:#0284C7;font-family:monospace">'
        f'📄 {fname}</span></div>', unsafe_allow_html=True)

    ctypes = col_types_dict(df)
    n_num  = sum(1 for t in ctypes.values() if t == 'numerical')
    n_cat  = sum(1 for t in ctypes.values() if t == 'categorical')
    n_dt   = sum(1 for t in ctypes.values() if t == 'datetime')
    n_txt  = sum(1 for t in ctypes.values() if t == 'text')
    total_cells = df.shape[0] * df.shape[1]
    miss_total = int(df.isnull().sum().sum())
    miss_pct   = miss_total / total_cells * 100 if total_cells > 0 else 0
    mem_mb     = df.memory_usage(deep=True).sum() / 1024 / 1024

    m1, m2, m3, m4, m5 = st.columns(5)
    for cw, val, sub, lbl in [
        (m1, f"{df.shape[0]:,}", "", "ROWS"),
        (m2, str(df.shape[1]), "", "COLUMNS"),
        (m5, f"{mem_mb:.1f} MB", "", "MEMORY"),
        (m4, f"{miss_total:,} cells ({miss_pct:.1f}%)", "MISSING", ""),
        (m3, f"🔢{n_num} 🔠{n_cat} 📅{n_dt} 📝{n_txt}", "COLUMN TYPES", ""),
    ]:
        cw.markdown(
            f'<div class="stat-box"><div class="stat-num">{val}</div>'
            f'{"<div class=stat-sub>"+sub+"</div>" if sub else ""}'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

    num_cols = [c for c, t in ctypes.items() if t == 'numerical']
    cat_cols = [c for c, t in ctypes.items() if t == 'categorical']
    dt_cols  = [c for c, t in ctypes.items() if t == 'datetime']
    txt_cols = [c for c, t in ctypes.items() if t == 'text']

    tab_all, tab_num, tab_cat, tab_dt, tab_txt = st.tabs([
        "📋 All Columns", "🔢 Numerical", "🔠 Categorical", "📅 Datetime", "📝 String"
    ])

    with tab_all:
        st.markdown('<div style="font-size:0.88rem;font-weight:600;color:#334155;margin-bottom:0.6rem">Column Schema & Statistics</div>', unsafe_allow_html=True)
        rows_html = ""
        for col in df.columns:
            t     = ctypes[col]
            miss  = df[col].isnull().sum()
            mpct  = miss / len(df) * 100
            uniq  = df[col].nunique()
            mc    = "#DC2626" if mpct >= 10 else ("#D97706" if mpct > 0 else "#16A34A")
            badge = type_badge_html(t)
            prog  = progress_bar_html(mpct, mc)
            rows_html += (
                f'<tr>'
                f'<td style="font-weight:500;color:#1E293B">{col}</td>'
                f'<td>{badge}</td>'
                f'<td>{prog}</td>'
                f'<td style="color:#64748B;text-align:right">{uniq:,}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="overflow-y:auto;max-height:360px;border-radius:10px;'
            f'border:1px solid #E2E8F0;background:#fff">'
            f'<table class="col-table"><thead><tr>'
            f'<th>Column</th><th>Type</th><th>Missing</th><th style="text-align:right">Unique</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        with st.expander("Show Data Preview (up to 500 rows)"):
            st.dataframe(df.head(500), use_container_width=True)

    with tab_num:
        if not num_cols:
            st.info("No numerical columns detected.")
        else:
            rows_html = ""
            for col in num_cols:
                s    = df[col].dropna()
                miss = df[col].isnull().sum()
                mpct = miss / len(df) * 100
                mc   = "#DC2626" if mpct >= 10 else ("#D97706" if mpct > 0 else "#16A34A")
                prog = progress_bar_html(mpct, mc)
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr    = q3 - q1
                n_out  = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
                rows_html += (
                    f'<tr>'
                    f'<td style="font-weight:500;color:#1E293B">{col}</td>'
                    f'<td>{prog}</td>'
                    f'<td style="color:#475569;text-align:right">{s.mean():.2f}</td>'
                    f'<td style="color:#475569;text-align:right">{s.std():.2f}</td>'
                    f'<td style="color:#475569;text-align:right">{s.min():.2f}</td>'
                    f'<td style="color:#475569;text-align:right">{s.median():.2f}</td>'
                    f'<td style="color:#475569;text-align:right">{s.max():.2f}</td>'
                    f'<td style="color:{"#DC2626" if n_out > 0 else "#94A3B8"};text-align:right;font-weight:{"600" if n_out > 0 else "400"}">{n_out}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="overflow-y:auto;border-radius:10px;border:1px solid #E2E8F0;background:#fff">'
                f'<table class="col-table"><thead><tr>'
                f'<th>Column</th><th>Missing</th>'
                f'<th style="text-align:right">Mean</th><th style="text-align:right">Std</th>'
                f'<th style="text-align:right">Min</th><th style="text-align:right">Median</th>'
                f'<th style="text-align:right">Max</th><th style="text-align:right">Outliers(IQR)</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True)

    with tab_cat:
        if not cat_cols:
            st.info("No categorical columns detected.")
        else:
            rows_html = ""
            for col in cat_cols:
                miss  = df[col].isnull().sum()
                mpct  = miss / len(df) * 100
                mc    = "#DC2626" if mpct >= 10 else ("#D97706" if mpct > 0 else "#16A34A")
                prog  = progress_bar_html(mpct, mc)
                uniq  = df[col].nunique()
                vc    = df[col].value_counts()
                mode_val  = vc.index[0] if len(vc) > 0 else "—"
                mode_cnt  = int(vc.iloc[0]) if len(vc) > 0 else 0
                mode_pct  = round(mode_cnt / len(df) * 100, 1) if len(df) > 0 else 0
                rows_html += (
                    f'<tr>'
                    f'<td style="font-weight:500;color:#1E293B">{col}</td>'
                    f'<td>{prog}</td>'
                    f'<td style="color:#475569;text-align:right">{uniq:,}</td>'
                    f'<td style="color:#1E293B;font-weight:500">{str(mode_val)[:30]}</td>'
                    f'<td style="color:#475569;text-align:right">{mode_cnt:,}</td>'
                    f'<td style="color:#0284C7;text-align:right;font-weight:600">{mode_pct}%</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="overflow-y:auto;border-radius:10px;border:1px solid #E2E8F0;background:#fff">'
                f'<table class="col-table"><thead><tr>'
                f'<th>Column</th><th>Missing</th><th style="text-align:right">Unique</th>'
                f'<th>Mode</th><th style="text-align:right">Mode Count</th><th style="text-align:right">Mode %</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True)

    with tab_dt:
        if not dt_cols:
            candidate_dt = []
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].dropna().head(10).astype(str)
                    hits = 0
                    for v in sample:
                        try: pd.to_datetime(v); hits += 1
                        except: pass
                    if hits / max(len(sample), 1) > 0.7:
                        candidate_dt.append(col)
            if candidate_dt:
                st.info(f"No datetime columns found, but these text columns may contain dates: **{', '.join(candidate_dt)}**. Convert them in the Cleaning → Types tab.")
            else:
                st.info("No datetime columns detected.")
        else:
            rows_html = ""
            for col in dt_cols:
                s     = pd.to_datetime(df[col], errors='coerce').dropna()
                miss  = df[col].isnull().sum()
                mpct  = miss / len(df) * 100
                mc    = "#DC2626" if mpct >= 10 else ("#D97706" if mpct > 0 else "#16A34A")
                prog  = progress_bar_html(mpct, mc)
                mn    = s.min().strftime('%Y-%m-%d') if len(s) else "—"
                mx    = s.max().strftime('%Y-%m-%d') if len(s) else "—"
                rng   = str((s.max() - s.min()).days) + " days" if len(s) > 1 else "—"
                if len(s) > 2:
                    diffs = s.sort_values().diff().dropna()
                    med_d = diffs.dt.days.median()
                    freq = "Daily" if med_d <= 1.5 else ("Weekly" if med_d <= 8 else ("Monthly" if med_d <= 32 else "Irregular"))
                else: freq = "—"
                gaps = 0
                if len(s) > 1:
                    s_sorted = s.sort_values()
                    diffs2 = (s_sorted.diff().dropna().dt.days)
                    gaps = int((diffs2 > diffs2.median() * 3).sum())
                rows_html += (
                    f'<tr>'
                    f'<td style="font-weight:500;color:#1E293B">{col}</td>'
                    f'<td>{prog}</td>'
                    f'<td style="color:#475569">{mn}</td>'
                    f'<td style="color:#475569">{mx}</td>'
                    f'<td style="color:#475569">{rng}</td>'
                    f'<td style="color:#475569">{freq}</td>'
                    f'<td style="color:{"#DC2626" if gaps>0 else "#94A3B8"};font-weight:{"600" if gaps>0 else "400"}">{gaps}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="overflow-y:auto;border-radius:10px;border:1px solid #E2E8F0;background:#fff">'
                f'<table class="col-table"><thead><tr>'
                f'<th>Column</th><th>Missing</th><th>Min Date</th><th>Max Date</th>'
                f'<th>Range</th><th>Freq</th><th>Gaps</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True)

    with tab_txt:
        if not txt_cols:
            st.info("No text (unstructured string) columns detected.")
        else:
            rows_html = ""
            for col in txt_cols:
                miss   = df[col].isnull().sum()
                mpct   = miss / len(df) * 100
                mc     = "#DC2626" if mpct >= 10 else ("#D97706" if mpct > 0 else "#16A34A")
                prog   = progress_bar_html(mpct, mc)
                uniq   = df[col].nunique()
                s_len  = df[col].dropna().astype(str).str.len()
                avg_l  = round(s_len.mean(), 1) if len(s_len) else 0
                min_l  = int(s_len.min()) if len(s_len) else 0
                max_l  = int(s_len.max()) if len(s_len) else 0
                sample = str(df[col].dropna().iloc[0])[:60] + "…" if df[col].notna().any() else "—"
                rows_html += (
                    f'<tr>'
                    f'<td style="font-weight:500;color:#1E293B">{col}</td>'
                    f'<td>{prog}</td>'
                    f'<td style="color:#475569;text-align:right">{uniq:,}</td>'
                    f'<td style="color:#475569;text-align:right">{avg_l}</td>'
                    f'<td style="color:#475569;text-align:right">{min_l} / {max_l}</td>'
                    f'<td style="color:#475569;font-size:0.75rem;max-width:220px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{sample}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="overflow-y:auto;border-radius:10px;border:1px solid #E2E8F0;background:#fff">'
                f'<table class="col-table"><thead><tr>'
                f'<th>Column</th><th>Missing</th><th style="text-align:right">Unique</th>'
                f'<th style="text-align:right">Avg Length</th><th style="text-align:right">Min/Max Len</th>'
                f'<th>Sample</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    bc, _, gc = st.columns([1, 3, 1])
    with bc:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("← Upload Again"):
            st.session_state.update({'screen':'landing','working_df':None}); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with gc:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("Go to Studio →", use_container_width=True):
            # Reset clean_section and viz_selected when entering studio
            st.session_state['clean_section'] = None
            st.session_state['viz_selected'] = None
            st.session_state['screen'] = 'studio'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SCREEN 3 — STUDIO
# ──────────────────────────────────────────────────────────────────────────────
elif st.session_state['screen'] == 'studio':
    df = st.session_state['working_df']
    fname = st.session_state['file_name']

    with st.sidebar:
        st.markdown('<div style="font-size:1rem;font-weight:700;color:#1E293B;margin-bottom:0.8rem">🧪 Studio</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:7px;'
            f'padding:0.35rem 0.7rem;font-size:0.75rem;color:#0284C7;font-family:monospace;'
            f'margin-bottom:0.5rem">📄 {fname}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#94A3B8;margin-bottom:0.8rem">'
            f'{df.shape[0]:,} rows · {df.shape[1]} cols · '
            f'{len(get_num(df))} numeric · {df.isnull().sum().sum()} missing</div>',
            unsafe_allow_html=True)
        st.markdown('<hr style="border:none;border-top:1px solid #F1F5F9;margin:0.5rem 0">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
        if st.button("↩ Undo Last Step", use_container_width=True):
            if st.session_state['df_snapshots']:
                st.session_state['working_df'] = st.session_state['df_snapshots'].pop()
                if st.session_state['history']: st.session_state['history'].pop()
                if st.session_state['recipe']:  st.session_state['recipe'].pop()
                st.success("Undone!"); st.rerun()
            else: st.warning("Nothing to undo")
        # FIX: typo 'st.rerufn()' → 'st.rerun()'
        if st.button("← Back to Overview", use_container_width=True):
            st.session_state['screen'] = 'overview'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr style="border:none;border-top:1px solid #F1F5F9;margin:0.5rem 0">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:0.4rem">History</div>', unsafe_allow_html=True)
        if st.session_state['history']:
            for h in reversed(st.session_state['history'][-6:]):
                st.markdown(f'<div class="hist-item">#{h["step"]} {h["timestamp"]}<br>{h["description"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hist-item" style="color:#CBD5E1">No transformations yet</div>', unsafe_allow_html=True)

    st.markdown(step_bar(3), unsafe_allow_html=True)
    tab_clean, tab_viz, tab_export = st.tabs(["🔧 Cleaning", "📊 Visualization", "⬇️ Export"])

    # ═══════════════════════════════════════════════════════════════
    # CLEANING
    # ═══════════════════════════════════════════════════════════════
    with tab_clean:
        df = st.session_state['working_df']
        num_cols = get_num(df); cat_cols = get_cat(df)

        CLEAN_SECTIONS = [
            ("missing",  "⚠️", "Missing Values",   "Handle nulls with mean, median, mode or drop."),
            ("dupes",    "🗂️", "Duplicates",        "Find and remove redundant rows in your data."),
            ("types",    "🔄", "Types & Parsing",   "Convert data types or clean dirty numeric strings."),
            ("cat",      "🏷️", "Categorical",       "Standardize text, map values, or OHE."),
            ("outliers", "📐", "Outliers",          "Detect and cap/remove statistical outliers."),
            ("norm",     "📏", "Normalization",     "Scale features using Min-Max or Z-Score."),
            ("colops",   "🔩", "Column Operations",        "Rename, drop, or create new calculated columns."),
            ("val",      "✅", "Validation",        "Check data integrity and constraints."),
        ]

        active_sec = st.session_state.get('clean_section', None)

        # ── Card grid ──────────────────────────────────────────────
        if active_sec is None:
            with st.expander("Show Data Preview", expanded=False):
                st.dataframe(df.head(500), use_container_width=True)

            st.markdown(
                '<div style="font-size:1.25rem;font-weight:700;color:#1E293B;margin-top:0.8rem;margin-bottom:0.15rem">🔧 Choose Cleaning Operation</div>'
                '<div style="color:#94A3B8;font-size:0.84rem;margin-bottom:1.5rem">Select a tool below to start cleaning your data.</div>',
                unsafe_allow_html=True)

            rows = [CLEAN_SECTIONS[i:i+4] for i in range(0, len(CLEAN_SECTIONS), 4)]
            for row in rows:
                cols4 = st.columns(4)
                for col_w, (sid, icon, label, desc) in zip(cols4, row):
                    with col_w:
                        st.markdown(
                            f'<div class="func-card">'
                            f'<div class="func-card-icon">{icon}</div>'
                            f'<div class="func-card-title">{label}</div>'
                            f'<div class="func-card-desc">{desc}</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                        st.markdown('<div class="func-card-btn">', unsafe_allow_html=True)
                        if st.button("Open", key=f"btn_nav_{sid}", use_container_width=True):
                            st.session_state['clean_section'] = sid
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

        # ── Detail view ────────────────────────────────────────────
        else:
            cdef = next(d for d in CLEAN_SECTIONS if d[0] == active_sec)
            sid, icon, label, desc = cdef

            # Header row: title LEFT, back button RIGHT — same vertical level
            title_col, back_col = st.columns([5, 1])
            with title_col:
                st.markdown(
                    f'<div class="sec-detail-left">'
                    f'<div class="sec-detail-icon">{icon}</div>'
                    f'<div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#1E293B;line-height:1.2">{label}</div>'
                    f'<div style="font-size:0.78rem;color:#94A3B8">{desc}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            with back_col:
                st.markdown('<div class="sec-back-btn" style="display:flex;align-items:center;height:100%;padding-top:0.25rem">', unsafe_allow_html=True)
                if st.button("← Back", key="clean_back"):
                    st.session_state['clean_section'] = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<hr style="border:none;border-top:1px solid #F1F5F9;margin:0.5rem 0 1.2rem">', unsafe_allow_html=True)

            # ── Tool logic ──────────────────────────────────────────
            if active_sec == "missing":
                ms = pd.DataFrame({'Column':df.columns,'Missing':df.isnull().sum().values,
                    'Pct(%)': (df.isnull().sum().values/len(df)*100).round(1),
                    'Dtype':df.dtypes.astype(str).values}).query('Missing > 0').reset_index(drop=True)

                st.markdown("##### 📊 Missing Value Summary")
                if ms.empty:
                    st.success("🎉 No missing values!")
                else:
                    st.dataframe(ms, use_container_width=True, hide_index=True)
                    st.markdown("##### 🔧 Fix Strategy")
                    miss_cols = ms['Column'].tolist()
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        tgt = st.multiselect("Target columns", miss_cols, default=miss_cols[:1], key="m_tgt")
                        meth = st.selectbox("Strategy", ["Mean (numeric)","Median (numeric)","Mode / Most Frequent",
                            "Constant Value","Forward Fill","Backward Fill","Drop Rows","Drop Columns above threshold %"], key="m_meth")
                        fv=""; thresh=50
                        if meth=="Constant Value": fv=st.text_input("Fill value","0",key="m_fv")
                        if meth=="Drop Columns above threshold %": thresh=st.slider("Threshold %",0,100,50,key="m_thr")
                    with c2:
                        if tgt:
                            st.dataframe(pd.DataFrame({'Column':tgt,
                                'Missing':[df[c].isnull().sum() for c in tgt],
                                'Pct(%)':[round(df[c].isnull().sum()/len(df)*100,1) for c in tgt]}), use_container_width=True, hide_index=True)

                    if st.button("▶ Apply Fix", key="run_miss", use_container_width=True):
                        if tgt:
                            save_snap(); sb=shape_str(df); new_df=df.copy()
                            if meth=="Drop Columns above threshold %":
                                new_df=new_df.drop(columns=[c for c in tgt if new_df[c].isnull().sum()/len(new_df)*100>=thresh])
                            else:
                                for col in tgt:
                                    if meth=="Mean (numeric)" and col in num_cols: new_df[col]=new_df[col].fillna(new_df[col].mean())
                                    elif meth=="Median (numeric)" and col in num_cols: new_df[col]=new_df[col].fillna(new_df[col].median())
                                    elif meth=="Mode / Most Frequent": new_df[col]=new_df[col].fillna(new_df[col].mode()[0] if not new_df[col].mode().empty else np.nan)
                                    elif meth=="Constant Value": new_df[col]=new_df[col].fillna(fv)
                                    elif meth=="Forward Fill": new_df[col]=new_df[col].ffill()
                                    elif meth=="Backward Fill": new_df[col]=new_df[col].bfill()
                                    elif meth=="Drop Rows": new_df=new_df.dropna(subset=[col])
                            sa=commit(new_df, f"Missing→{meth}: {tgt}", sb)
                            st.success("✅ Done"); st.rerun()

            elif active_sec == "dupes":
                st.markdown("##### 🔍 Duplicate Detection")
                c1, c2 = st.columns(2)
                with c1:
                    mode = st.radio("Scope", ["Full-row", "By subset of columns"], key="d_mode")
                    subset = st.multiselect("Key columns", df.columns.tolist(), key="d_sub") if mode == "By subset of columns" else None
                    keep = st.radio("Keep", ["first", "last", "none (remove all)"], key="d_keep")
                    kv = False if keep == "none (remove all)" else keep
                with c2:
                    try:
                        dc = df.duplicated(subset=subset, keep=False).sum()
                        st.metric("Duplicate Rows", dc)
                    except: dc = 0
                if dc > 0:
                    if st.button("▶ Remove Duplicates", key="run_dup", use_container_width=True):
                        save_snap(); sb=shape_str(df)
                        new_df = df.drop_duplicates(subset=subset, keep=kv)
                        commit(new_df, f"Dedup ({mode})", sb)
                        st.success("✅ Done"); st.rerun()

            elif active_sec == "types":
                st.markdown("##### 🔄 Convert Column Type")
                tc = st.selectbox("Column", df.columns.tolist(), key="tc_col")
                tt = st.selectbox("Convert to", ["float64", "int64", "str / object", "datetime", "category"], key="tc_tgt")
                if st.button("▶ Convert", key="run_type", use_container_width=True):
                    save_snap(); sb=shape_str(df); new_df=df.copy()
                    try:
                        if tt == "datetime": new_df[tc] = pd.to_datetime(new_df[tc], errors='coerce')
                        elif tt == "str / object": new_df[tc] = new_df[tc].astype(str)
                        else: new_df[tc] = pd.to_numeric(new_df[tc], errors='coerce').astype(tt)
                        commit(new_df, f"Cast {tc} to {tt}", sb)
                        st.success("✅ Done"); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            elif active_sec == "cat":
                st.markdown("##### 🏷️ Categorical Standardize")
                sc2 = st.selectbox("Select Column", cat_cols if cat_cols else df.columns.tolist(), key="cat_sc")
                ops = st.multiselect("Operations", ["Trim whitespace", "Lowercase", "Title Case", "UPPERCASE"], key="cat_ops")
                if st.button("▶ Apply", key="run_std"):
                    save_snap(); sb=shape_str(df); new_df=df.copy(); s=new_df[sc2].astype(str)
                    if "Trim whitespace" in ops: s=s.str.strip()
                    if "Lowercase" in ops: s=s.str.lower()
                    if "Title Case" in ops: s=s.str.title()
                    if "UPPERCASE" in ops: s=s.str.upper()
                    new_df[sc2]=s
                    commit(new_df, f"Std {sc2}", sb); st.rerun()

            elif active_sec == "outliers":
                st.markdown("##### 📐 Outlier Treatment")
                if not num_cols:
                    st.info("No numerical columns available.")
                else:
                    oc = st.selectbox("Numeric Column", num_cols, key="out_col")
                    method = st.radio("Detection", ["IQR", "Z-Score"], horizontal=True)
                    act = st.selectbox("Action", ["Cap / Winsorize", "Remove Rows"])
                    s = df[oc].dropna()
                    if method == "IQR":
                        q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
                        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                    else:
                        lo = s.mean() - 3*s.std(); hi = s.mean() + 3*s.std()
                    n_out = int(((s < lo) | (s > hi)).sum())
                    st.info(f"Detected **{n_out}** outliers (range: {lo:.2f} – {hi:.2f})")
                    if st.button("▶ Handle Outliers", key="run_out"):
                        save_snap(); sb=shape_str(df); new_df=df.copy()
                        if act == "Cap / Winsorize":
                            new_df[oc] = new_df[oc].clip(lower=lo, upper=hi)
                        else:
                            new_df = new_df[(new_df[oc].isna()) | ((new_df[oc] >= lo) & (new_df[oc] <= hi))]
                        commit(new_df, f"Outlier {act} on {oc}", sb); st.rerun()

            elif active_sec == "norm":
                st.markdown("##### 📏 Scaling & Normalization")
                if not num_cols:
                    st.info("No numerical columns available.")
                else:
                    sm = st.radio("Method", ["Min-Max [0,1]", "Z-Score"], key="norm_sm")
                    sc3 = st.multiselect("Columns", num_cols, key="norm_sc")
                    if st.button("▶ Apply Scaling", key="run_norm"):
                        save_snap(); sb=shape_str(df); new_df=df.copy()
                        for c in sc3:
                            if sm == "Min-Max [0,1]":
                                rng = new_df[c].max() - new_df[c].min()
                                if rng != 0: new_df[c] = (new_df[c]-new_df[c].min())/rng
                            else:
                                std = new_df[c].std()
                                if std != 0: new_df[c] = (new_df[c]-new_df[c].mean())/std
                        commit(new_df, f"Scale {sm}: {sc3}", sb); st.rerun()

            elif active_sec == "colops":
                st.markdown("##### 🔩 Column Operations")
                op_type = st.radio("Type", ["Rename", "Drop", "New Formula"], horizontal=True)
                if op_type == "Rename":
                    c1, c2 = st.columns(2)
                    old_n = c1.selectbox("Source", df.columns)
                    new_n = c2.text_input("New Name")
                    if st.button("Rename") and new_n:
                        save_snap(); commit(df.rename(columns={old_n:new_n}), f"Rename {old_n}→{new_n}", shape_str(df)); st.rerun()
                elif op_type == "Drop":
                    to_drop = st.multiselect("Columns to remove", df.columns)
                    if st.button("Drop Columns") and to_drop:
                        save_snap(); commit(df.drop(columns=to_drop), f"Drop {to_drop}", shape_str(df)); st.rerun()
                elif op_type == "New Formula":
                    new_col_name = st.text_input("New column name", key="formula_name")
                    formula = st.text_area("Formula (use column names, e.g. `age * 2 + years_exp`)", key="formula_expr")
                    st.caption("Available columns: " + ", ".join(df.columns.tolist()))
                    if st.button("Create Column") and new_col_name and formula:
                        save_snap(); sb=shape_str(df); new_df=df.copy()
                        try:
                            new_df[new_col_name] = new_df.eval(formula)
                            commit(new_df, f"New col: {new_col_name}", sb); st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

            elif active_sec == "val":
                st.markdown("##### ✅ Data Validation")
                st.markdown("**Null summary**")
                null_df = df.isnull().sum().reset_index()
                null_df.columns = ['Column', 'Null Count']
                null_df['Null %'] = (null_df['Null Count'] / len(df) * 100).round(1)
                null_df = null_df[null_df['Null Count'] > 0]
                if null_df.empty:
                    st.success("No nulls found!")
                else:
                    st.dataframe(null_df, use_container_width=True, hide_index=True)

                st.markdown("**Duplicate rows**")
                n_dup = df.duplicated().sum()
                if n_dup == 0:
                    st.success("No duplicate rows.")
                else:
                    st.warning(f"{n_dup} duplicate rows found.")

                if num_cols:
                    st.markdown("**Numeric range check**")
                    vc = st.selectbox("Column", num_cols, key="val_col")
                    c1, c2 = st.columns(2)
                    vmin = c1.number_input("Min allowed", value=float(df[vc].min()), key="val_min")
                    vmax = c2.number_input("Max allowed", value=float(df[vc].max()), key="val_max")
                    violations = df[(df[vc] < vmin) | (df[vc] > vmax)]
                    st.info(f"{len(violations)} violations found outside [{vmin}, {vmax}]")
                    if not violations.empty:
                        st.dataframe(violations.head(50), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    with tab_viz:
        df = st.session_state['working_df']
        num_cols = get_num(df); cat_cols = get_cat(df); all_cols = df.columns.tolist()

        CHART_DEFS = [
            ("histogram", "📊", "blue",   "Histogram",    "Distribution of a numeric column with adjustable bins."),
            ("box",       "📦", "teal",   "Box Plot",     "Median, quartiles and outliers. Group by category."),
            ("scatter",   "✦",  "indigo", "Scatter Plot", "Relationship between two numeric variables."),
            ("line",      "📈", "sky",    "Line Chart",   "Trend over time or ordered index."),
            ("bar",       "▮",  "emerald","Bar Chart",    "Compare categories with aggregation & Top-N."),
            ("heatmap",   "🔥", "amber",  "Heatmap",      "Correlation matrix for numeric columns."),
            ("violin",    "🎻", "violet", "Violin Plot",  "Distribution shape + box plot combined."),
            ("pie",       "🍩", "rose",   "Pie / Donut",  "Part-to-whole for categorical columns."),
        ]

        # Map color names to our theme palette
        ICON_BG = {
            "blue":   "#DBEAFE", "teal":   "#CCFBF1", "indigo": "#E0E7FF",
            "sky":    "#E0F2FE", "emerald":"#D1FAE5", "amber":  "#FEF3C7",
            "violet": "#EDE9FE", "rose":   "#FFE4E6",
        }

        sel = st.session_state['viz_selected']

        if sel is None:
            st.markdown(
                '<div style="font-size:1.25rem;font-weight:700;color:#1E293B;margin-bottom:0.15rem">📊 Choose Chart Type</div>'
                '<div style="color:#94A3B8;font-size:0.84rem;margin-bottom:1rem">Click a card to build a visualization</div>',
                unsafe_allow_html=True)

            rows = [CHART_DEFS[i:i+4] for i in range(0, len(CHART_DEFS), 4)]
            for row in rows:
                cols4 = st.columns(4)
                for col_w, (cid, icon, color, title, desc) in zip(cols4, row):
                    icon_bg = ICON_BG.get(color, "#F0F9FF")
                    with col_w:
                        st.markdown(
                            f'<div class="func-card">'
                            f'<div class="func-card-icon" style="background:{icon_bg}">{icon}</div>'
                            f'<div class="func-card-title">{title}</div>'
                            f'<div class="func-card-desc">{desc}</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                        st.markdown('<div class="func-card-btn">', unsafe_allow_html=True)
                        if st.button("Open", key=f"pick_{cid}", use_container_width=True):
                            st.session_state['viz_selected'] = cid; st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

        else:
            cdef = next(d for d in CHART_DEFS if d[0]==sel)
            cid, icon, color, title, desc = cdef
            icon_bg = ICON_BG.get(color, "#F0F9FF")

            # Header row: title LEFT, back button RIGHT — same vertical level
            title_col, back_col = st.columns([5, 1])
            with title_col:
                st.markdown(
                    f'<div class="sec-detail-left">'
                    f'<div class="sec-detail-icon" style="background:{icon_bg}">{icon}</div>'
                    f'<div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#1E293B;line-height:1.2">{title}</div>'
                    f'<div style="font-size:0.78rem;color:#94A3B8">{desc}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            with back_col:
                st.markdown('<div class="sec-back-btn" style="display:flex;align-items:center;height:100%;padding-top:0.25rem">', unsafe_allow_html=True)
                if st.button("← Back", key="viz_back"):
                    st.session_state['viz_selected'] = None; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

            build_col, filter_col = st.columns([3, 1])
            pt = 'plotly_white'
            lk = dict(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#475569', size=11),
                xaxis=dict(gridcolor='#F1F5F9', linecolor='#E2E8F0'),
                yaxis=dict(gridcolor='#F1F5F9', linecolor='#E2E8F0'))
            # ── Color palette picker ──
            PALETTES = {
                'Sky Blue':   ['#0EA5E9','#38BDF8','#7DD3FC','#BAE6FD','#0284C7','#075985','#60A5FA','#93C5FD'],
                'Indigo':     ['#6366F1','#818CF8','#A5B4FC','#C7D2FE','#4F46E5','#3730A3','#8B5CF6','#A78BFA'],
                'Emerald':    ['#10B981','#34D399','#6EE7B7','#A7F3D0','#059669','#047857','#14B8A6','#2DD4BF'],
                'Rose':       ['#F43F5E','#FB7185','#FDA4AF','#FECDD3','#E11D48','#BE123C','#F97316','#FB923C'],
                'Amber':      ['#F59E0B','#FCD34D','#FDE68A','#FEF3C7','#D97706','#B45309','#EF4444','#FCA5A5'],
                'Slate':      ['#475569','#64748B','#94A3B8','#CBD5E1','#1E293B','#334155','#0EA5E9','#38BDF8'],
                'Vivid Mix':  ['#0EA5E9','#6366F1','#10B981','#F59E0B','#F43F5E','#8B5CF6','#14B8A6','#F97316'],
            }
            PAL_SWATCHES = {k: v[0] for k, v in PALETTES.items()}  # first color as preview

            st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#475569;margin-bottom:0.2rem">🎨 Chart Color Palette</div>', unsafe_allow_html=True)
            pal_cols = st.columns(len(PALETTES))
            for i, (pname, phex) in enumerate(PAL_SWATCHES.items()):
                with pal_cols[i]:
                    is_sel = st.session_state['viz_palette'] == pname
                    border = "3px solid #1E293B" if is_sel else "2px solid #E2E8F0"
                    st.markdown(
                        f'<div style="display:flex;flex-direction:column;align-items:center;gap:0.2rem;cursor:pointer">',
                        unsafe_allow_html=True)
                    # Show 4 color mini swatches
                    swatches_html = ''.join(
                        f'<div style="width:14px;height:14px;border-radius:3px;background:{c};display:inline-block;margin:1px"></div>'
                        for c in PALETTES[pname][:4]
                    )
                    st.markdown(
                        f'<div style="border:{border};border-radius:8px;padding:4px 5px;cursor:pointer;display:flex;gap:1px;flex-wrap:wrap;width:fit-content;background:{"#F0F9FF" if is_sel else "#fff"}">' + swatches_html + '</div>',
                        unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:0.62rem;color:{"#0EA5E9" if is_sel else "#94A3B8"};font-weight:{"700" if is_sel else "400"};text-align:center;margin-top:1px">{pname}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    if st.button(" ", key=f"pal_{pname}", help=pname):
                        st.session_state['viz_palette'] = pname; st.rerun()

            st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

            THEME_COLORS = PALETTES[st.session_state['viz_palette']]
            prim = THEME_COLORS[0]
            fig = None; plot_df = df.copy()

            with filter_col:
                st.markdown('<div style="font-size:0.85rem;font-weight:600;color:#334155;margin-bottom:0.4rem">🔍 Filters</div>', unsafe_allow_html=True)
                if cat_cols:
                    fc = st.selectbox("By category", ["(none)"]+cat_cols, key="f_cc")
                    if fc != "(none)":
                        uv = sorted(plot_df[fc].dropna().unique().tolist())
                        cv2 = st.multiselect("Keep", uv, default=uv, key="f_cv")
                        if cv2: plot_df = plot_df[plot_df[fc].isin(cv2)]
                if num_cols:
                    fn = st.selectbox("Range on", ["(none)"]+num_cols, key="f_nc")
                    if fn != "(none)":
                        cmin = float(plot_df[fn].min()); cmax = float(plot_df[fn].max())
                        if cmin < cmax:
                            rng = st.slider("Range", cmin, cmax, (cmin, cmax), key="f_nr")
                            plot_df = plot_df[plot_df[fn].between(rng[0], rng[1])]
                st.markdown(f'<div style="font-size:0.72rem;color:#94A3B8;margin-top:0.4rem">{len(plot_df):,} rows</div>', unsafe_allow_html=True)

            with build_col:
                AGG = {"None (raw)":None,"Sum":"sum","Mean":"mean","Count":"count","Median":"median"}
                if cid == "histogram":
                    a,b,c = st.columns(3)
                    hcc = a.selectbox("Column", num_cols or["—"], key="hi_c")
                    bins = b.slider("Bins", 5, 100, 30, key="hi_b")
                    clr = c.selectbox("Color by", ["None"]+cat_cols, key="hi_clr")
                    if num_cols:
                        fig = px.histogram(plot_df, x=hcc, nbins=bins,
                            color=(None if clr=="None" else clr),
                            template=pt, color_discrete_sequence=THEME_COLORS)

                elif cid == "box":
                    a,b,c = st.columns(3)
                    yc = a.selectbox("Value", num_cols or["—"], key="bx_y")
                    xc = b.selectbox("Group by", ["None"]+cat_cols, key="bx_x")
                    clr = c.selectbox("Color by", ["None"]+cat_cols, key="bx_clr")
                    if num_cols:
                        fig = px.box(plot_df, x=(None if xc=="None" else xc), y=yc,
                            color=(None if clr=="None" else clr),
                            template=pt, color_discrete_sequence=THEME_COLORS)

                elif cid == "scatter":
                    if len(num_cols) < 2: st.warning("Need ≥ 2 numeric columns")
                    else:
                        a,b,c,d = st.columns(4)
                        xc = a.selectbox("X", num_cols, key="sc_x")
                        yc = b.selectbox("Y", num_cols, index=min(1,len(num_cols)-1), key="sc_y")
                        clr = c.selectbox("Color", ["None"]+all_cols, key="sc_c")
                        szc = d.selectbox("Size", ["None"]+num_cols, key="sc_s")
                        fig = px.scatter(plot_df, x=xc, y=yc,
                            color=(None if clr=="None" else clr),
                            size=(None if szc=="None" else szc),
                            template=pt, color_discrete_sequence=THEME_COLORS,
                            color_continuous_scale='Blues')

                elif cid == "line":
                    a,b,c = st.columns(3)
                    xc = a.selectbox("X", ["Index"]+all_cols, key="ln_x")
                    yc = b.selectbox("Y", num_cols or["—"], key="ln_y")
                    clr = c.selectbox("Color", ["None"]+cat_cols, key="ln_c")
                    if num_cols:
                        _df2 = plot_df.reset_index() if xc=="Index" else plot_df
                        _x = "index" if xc=="Index" else xc
                        fig = px.line(_df2, x=_x, y=yc,
                            color=(None if clr=="None" else clr),
                            template=pt, color_discrete_sequence=THEME_COLORS)

                elif cid == "bar":
                    a,b,c,d = st.columns(4)
                    xcat = a.selectbox("Category (X)", cat_cols or all_cols, key="br_x")
                    ynum = b.selectbox("Value (Y)", num_cols or["—"], key="br_y")
                    agl = c.selectbox("Aggregation", list(AGG.keys()), index=2, key="br_agg")
                    tn = d.slider("Top N", 3, 50, 15, key="br_tn")
                    clr = st.selectbox("Color by", ["None"]+cat_cols, key="br_clr")
                    if cat_cols and num_cols:
                        af = AGG[agl]
                        if af and af != "count": grp = plot_df.groupby(xcat)[ynum].agg(af).reset_index()
                        elif af == "count": grp = plot_df[xcat].value_counts().reset_index(); grp.columns=[xcat,ynum]
                        else: grp = plot_df[[xcat,ynum]].dropna()
                        grp = grp.nlargest(tn, ynum)
                        fig = px.bar(grp, x=xcat, y=ynum,
                            color=(None if clr=="None" else(clr if clr in grp.columns else None)),
                            template=pt, color_discrete_sequence=THEME_COLORS, text_auto=True)
                        fig.update_traces(textposition='outside')

                elif cid == "heatmap":
                    if len(num_cols) < 2: st.warning("Need ≥ 2 numeric columns")
                    else:
                        sc4 = st.multiselect("Columns", num_cols, default=num_cols, key="hm_c")
                        if len(sc4) >= 2:
                            corr = plot_df[sc4].corr()
                            fig = go.Figure(go.Heatmap(
                                z=corr.values, x=corr.columns, y=corr.index,
                                colorscale=[[0,'#EFF6FF'],[0.5,'#60A5FA'],[1,'#1E40AF']],
                                zmid=0, text=corr.values.round(2), texttemplate='%{text}', showscale=True))
                            fig.update_layout(template=pt)

                elif cid == "violin":
                    a,b,c = st.columns(3)
                    yc = a.selectbox("Value", num_cols or["—"], key="vi_y")
                    xc = b.selectbox("Group by", ["None"]+cat_cols, key="vi_x")
                    clr = c.selectbox("Color by", ["None"]+cat_cols, key="vi_c")
                    if num_cols:
                        fig = px.violin(plot_df, x=(None if xc=="None" else xc), y=yc,
                            color=(None if clr=="None" else clr),
                            box=True, template=pt, color_discrete_sequence=THEME_COLORS)

                elif cid == "pie":
                    if not cat_cols: st.warning("No categorical columns.")
                    else:
                        a,b,c = st.columns(3)
                        pc = a.selectbox("Column", cat_cols, key="pi_c")
                        hole = b.slider("Donut", 0., 0.75, .4, key="pi_h")
                        topn = c.slider("Top N", 2, 20, 8, key="pi_n")
                        counts = plot_df[pc].value_counts().head(topn)
                        fig = go.Figure(go.Pie(
                            labels=counts.index, values=counts.values, hole=hole,
                            marker_colors=THEME_COLORS[:len(counts)]))
                        fig.update_layout(template=pt)

                if fig is not None:
                    fig.update_layout(**lk)
                    st.plotly_chart(fig, use_container_width=True)

                    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dl1, dl2, dl3, _ = st.columns([1, 1, 1, 3])

                    # ── PNG (requires kaleido) ──
                    with dl1:
                        try:
                            import kaleido  # noqa: check availability
                            png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
                            st.markdown('<div class="btn-download">', unsafe_allow_html=True)
                            st.download_button("⬇️ PNG", png_bytes,
                                f"{cid}_{ts_str}.png", "image/png", key="dl_png")
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception:
                            st.markdown(
                                '<div style="font-size:0.7rem;color:#94A3B8;padding-top:0.4rem">'
                                'PNG: install<br><code>kaleido</code></div>',
                                unsafe_allow_html=True)

                    # ── SVG (requires kaleido) ──
                    with dl2:
                        try:
                            import kaleido  # noqa
                            svg_bytes = fig.to_image(format="svg")
                            st.markdown('<div class="btn-download-purple">', unsafe_allow_html=True)
                            st.download_button("⬇️ SVG", svg_bytes,
                                f"{cid}_{ts_str}.svg", "image/svg+xml", key="dl_svg")
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception:
                            pass

                    # ── HTML (always works, no extra dependency) ──
                    with dl3:
                        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                        st.markdown('<div class="btn-download-indigo">', unsafe_allow_html=True)
                        st.download_button("⬇️ HTML", html_bytes,
                            f"{cid}_{ts_str}.html", "text/html", key="dl_html")
                        st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # EXPORT
    # ═══════════════════════════════════════════════════════════════
    with tab_export:
        df = st.session_state['working_df']
        num_cols = get_num(df)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="exp-card"><div class="exp-icon">⬇️</div>'
                '<div class="exp-title">Cleaned CSV</div>'
                f'<div class="exp-sub">{shape_str(df)}</div>',
                unsafe_allow_html=True)
            buf = io.BytesIO(); df.to_csv(buf, index=False, encoding='utf-8-sig')
            st.markdown('<div class="btn-download">', unsafe_allow_html=True)
            st.download_button("Download CSV", buf.getvalue(), f"cleaned_{ts}.csv", "text/csv",
                use_container_width=True, key="dl_csv")
            st.markdown('</div></div>', unsafe_allow_html=True)
        with c2:
            recipe_data = {'created_at': datetime.now().isoformat(), 'source_file': fname,
                'original_shape': shape_str(st.session_state['original_df']),
                'final_shape': shape_str(df), 'total_steps': len(st.session_state['recipe']),
                'steps': st.session_state['recipe'], 'history': st.session_state['history']}
            rj = json.dumps(recipe_data, ensure_ascii=False, indent=2)
            st.markdown(
                '<div class="exp-card"><div class="exp-icon">📋</div>'
                '<div class="exp-title">Recipe JSON (Recommended)</div>'
                f'<div class="exp-sub">{len(st.session_state["recipe"])} steps recorded</div>',
                unsafe_allow_html=True)
            st.markdown('<div class="btn-download-purple">', unsafe_allow_html=True)
            st.download_button("Download Recipe", rj.encode('utf-8'), f"recipe_{ts}.json",
                "application/json", use_container_width=True, key="dl_recipe")
            st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:#334155;margin-bottom:0.5rem">📋 Applied Steps</div>', unsafe_allow_html=True)
        if st.session_state['history']:
            st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True, hide_index=True)
        else:
            st.info("No steps applied yet.")
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:#334155;margin:1rem 0 0.5rem">🔍 Final Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        if num_cols:
            st.markdown('<div style="font-size:0.9rem;font-weight:600;color:#334155;margin:1rem 0 0.5rem">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
