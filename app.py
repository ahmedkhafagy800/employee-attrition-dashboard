import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

# ── Theme Toggle ──────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
is_dark = st.session_state.dark_mode

# ── Brand Colors ──────────────────────────────────────────────────
if is_dark:
    BRAND = {
        'primary':      '#58A6FF',
        'secondary':    '#1F6FEB',
        'light':        '#161B22',
        'bg':           '#0D1117',
        'stayed':       '#3FB950',
        'left':         '#F85149',
        'chart_title':  '#58A6FF',
        'palette':      ['#58A6FF', '#3FB950', '#F85149', '#D2A8FF', '#FFA657'],
        'font_color':   '#E6EDF3',
        'grid_color':   '#2A3A4A',
        'line_color':   '#444444',
        'paper_bg':     '#0D1117',
        'plot_bg':      '#161B22',
        'card_border':  '#30363D',
        'sidebar_bg':   '#1F6FEB',
        'card_stayed':  '#1a4d2e',
        'card_left':    '#4d1a1a',
        'card_total':   '#1F3A6E',
        'card_rate':    '#4d3a1a',
        'tab_list_bg':  '#161B22',
        'metric_bg':    '#161B22',
        'input_bg':     '#161B22',
        'info_bg':      '#1C2A3A',
    }
else:
    BRAND = {
        'primary':      '#2563EB',
        'secondary':    '#1D4ED8',
        'light':        '#EFF6FF',
        'bg':           '#F8FAFC',
        'stayed':       '#16A34A',
        'left':         '#DC2626',
        'chart_title':  '#1E40AF',
        'palette':      ['#2563EB', '#16A34A', '#DC2626', '#7C3AED', '#D97706'],
        'font_color':   '#1E293B',
        'grid_color':   '#CBD5E1',
        'line_color':   '#94A3B8',
        'paper_bg':     '#FFFFFF',
        'plot_bg':      '#F1F5F9',
        'card_border':  '#BFDBFE',
        'sidebar_bg':   '#2563EB',
        'card_stayed':  '#DCFCE7',
        'card_left':    '#FEE2E2',
        'card_total':   '#DBEAFE',
        'card_rate':    '#FEF3C7',
        'tab_list_bg':  '#EFF6FF',
        'metric_bg':    '#EFF6FF',
        'input_bg':     '#FFFFFF',
        'info_bg':      '#EFF6FF',
    }

# ── Inject CSS FIRST before anything else ────────────────────────
st.markdown(f"""
<style>
/* ══ FORCE BACKGROUNDS ══════════════════════════════════════════ */
html, body,
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stMain"],
[data-testid="stMain"] > div,
.main,
.main > div,
.block-container,
section.main > div.block-container
{{
    background-color: {BRAND['bg']} !important;
    color: {BRAND['font_color']} !important;
}}
/* Fix top bar / header strip */
header, header[data-testid="stHeader"] {{
    background-color: {BRAND['bg']} !important;
    border-bottom: 1px solid {BRAND['primary']}22 !important;
}}

/* ══ SIDEBAR ════════════════════════════════════════════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"]
{{
    background-color: {BRAND['sidebar_bg']} !important;
}}
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] *
{{
    color: #FFFFFF !important;
}}
/* Sidebar selectbox dropdown — keep it themed, not white */
[data-testid="stSidebar"] [data-baseweb="popover"],
[data-testid="stSidebar"] [data-baseweb="menu"],
[data-testid="stSidebar"] [data-baseweb="select"] > div
{{
    background-color: {BRAND['sidebar_bg']} !important;
    border-color: rgba(255,255,255,0.3) !important;
}}
[data-testid="stSidebar"] [data-baseweb="option"],
[data-testid="stSidebar"] [data-baseweb="menu"] li
{{
    background-color: {BRAND['sidebar_bg']} !important;
    color: #FFFFFF !important;
}}
[data-testid="stSidebar"] [data-baseweb="option"]:hover,
[data-testid="stSidebar"] [data-baseweb="menu"] li:hover
{{
    background-color: rgba(255,255,255,0.2) !important;
    color: #FFFFFF !important;
}}

/* ══ HEADINGS ═══════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {{
    color: {BRAND['primary']} !important;
    background-color: transparent !important;
}}

/* ══ BODY TEXT ══════════════════════════════════════════════════ */
p, span, div, label, .stMarkdown, .stText {{
    color: {BRAND['font_color']} !important;
}}
.stCaption, small, [data-testid="stCaptionContainer"] * {{
    color: {BRAND['font_color']} !important;
    opacity: 0.75;
}}

/* ══ TABS ═══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {BRAND['tab_list_bg']} !important;
    border-radius: 8px;
    padding: 4px;
}}
.stTabs [aria-selected="true"] {{
    background-color: {BRAND['primary']} !important;
    color: #FFFFFF !important;
    border-radius: 6px;
    border: none !important;
    outline: none !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-bottom: none !important;
    box-shadow: none !important;
    color: {BRAND['font_color']} !important;
    background-color: transparent !important;
}}
button[role="tab"] {{
    border-bottom: none !important;
    box-shadow: none !important;
    color: {BRAND['font_color']} !important;
    background-color: transparent !important;
}}
button[role="tab"]:focus {{
    box-shadow: none !important;
    outline: none !important;
}}
button[role="tab"][aria-selected="true"] {{
    color: #FFFFFF !important;
}}

/* ══ METRIC CARDS ═══════════════════════════════════════════════ */
[data-testid="metric-container"] {{
    background-color: {BRAND['metric_bg']} !important;
    border: 1px solid {BRAND['primary']} !important;
    border-radius: 10px;
    padding: 14px;
}}
[data-testid="metric-container"] * {{
    color: {BRAND['font_color']} !important;
}}

/* ══ INFO / ALERT BOXES ═════════════════════════════════════════ */
[data-testid="stAlert"],
[data-testid="stAlert"] > div,
[data-testid="stNotification"] {{
    background-color: {BRAND['info_bg']} !important;
    border-left: 4px solid {BRAND['primary']} !important;
}}
[data-testid="stAlert"] *,
[data-testid="stNotification"] * {{
    color: {BRAND['font_color']} !important;
}}

/* ══ SELECTBOX (main content) ═══════════════════════════════════ */
[data-testid="stSelectbox"] label {{
    color: {BRAND['font_color']} !important;
}}
[data-baseweb="select"] > div,
[data-baseweb="select"] > div:hover {{
    background-color: {BRAND['input_bg']} !important;
    border-color: {BRAND['primary']} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] div {{
    color: {BRAND['font_color']} !important;
    background-color: transparent !important;
}}
/* Main dropdown popup */
[data-baseweb="popover"],
[data-baseweb="menu"] {{
    background-color: {BRAND['input_bg']} !important;
    border: 1px solid {BRAND['primary']} !important;
}}
[data-baseweb="menu"] li,
[data-baseweb="option"] {{
    background-color: {BRAND['input_bg']} !important;
    color: #1E293B !important;
}}
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {{
    background-color: {BRAND['primary']} !important;
    color: #FFFFFF !important;
}}

/* ══ DIVIDER ════════════════════════════════════════════════════ */
hr {{
    border-color: {BRAND['primary']} !important;
    opacity: 0.3;
}}

/* ══ PLOTLY — legend text & chart bg ════════════════════════════ */
.js-plotly-plot .plotly .svg-container {{
    background: transparent !important;
}}
/* Legend text color */
.js-plotly-plot .legendtext {{
    fill: {BRAND['font_color']} !important;
}}
/* Legend background */
.js-plotly-plot .legend {{
    fill: {BRAND['plot_bg']} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Logo ──────────────────────────────────────────────────────────
def get_logo_base64():
    try:
        with open("kayfaio_logo.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

logo_b64  = get_logo_base64()
logo_html = (
    f'<img src="data:image/jpeg;base64,{logo_b64}" width="240" style="margin-right:20px;"/>'
    if logo_b64 else ""
)

# ── Header ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; padding:16px 0 24px 0;
            border-bottom:2px solid {BRAND['primary']}; margin-bottom:24px;">
    {logo_html}
    <div>
        <h1 style="margin:0; color:{BRAND['primary']}; font-size:2rem; background:transparent;">
            Employee Attrition Analytics
        </h1>
        <p style="margin:4px 0 0 0; color:{BRAND['font_color']}; font-size:1rem;">
            Predict whether an employee <b>Stayed</b> or <b>Left</b>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load & Clean Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    df    = pd.concat([train, test], ignore_index=True)
    df.drop("Employee ID", axis=1, inplace=True)
    yc_condition = (df["Years at Company"] > 42) | (df["Years at Company"] < 1)
    df = df[~yc_condition]
    df = df.drop(columns=["Company Tenure"])
    mi_condition = (df["Monthly Income"] > 13712) | (df["Monthly Income"] < 816)
    df.loc[mi_condition, "Monthly Income"] = np.nan
    df["Monthly Income"].fillna(df["Monthly Income"].median(), inplace=True)
    return df.copy()

df_full = load_data()
numeric_cols     = df_full.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in df_full.select_dtypes(include="object").columns if c != "Attrition"]

# ── Sidebar ───────────────────────────────────────────────────────
if logo_b64:
    st.sidebar.markdown(
        f'<img src="data:image/jpeg;base64,{logo_b64}" width="200" '
        f'style="display:block; margin:0 auto 16px auto; '
        f'filter: brightness(0) invert(1);"/>',
        unsafe_allow_html=True
    )
st.sidebar.title("🔍 Filters")

new_dark = st.sidebar.toggle(
    "🌙 Dark Mode" if is_dark else "☀️ Light Mode",
    value=st.session_state.dark_mode,
    key="theme_toggle"
)
if new_dark != st.session_state.dark_mode:
    st.session_state.dark_mode = new_dark
    st.rerun()

st.sidebar.markdown("---")
attrition_filter = st.sidebar.selectbox("Attrition", ["All", "Stayed", "Left"])
df = df_full[df_full["Attrition"] == attrition_filter] if attrition_filter != "All" else df_full.copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Records:** {len(df):,}")
st.sidebar.markdown(f"**Attrition Rate:** {(df['Attrition'] == 'Left').mean():.1%}")

# ── Chart Theme ───────────────────────────────────────────────────
def apply_theme(fig, height=None):
    layout = dict(
        paper_bgcolor=BRAND['paper_bg'],
        plot_bgcolor=BRAND['plot_bg'],
        font=dict(family='Arial', color=BRAND['font_color']),
        title_font=dict(color=BRAND['chart_title'], size=16),
        legend=dict(
            font=dict(color=BRAND['font_color']),
            bgcolor=BRAND['plot_bg'],
            bordercolor=BRAND['grid_color'],
        ),
    )
    if height:
        layout['height'] = height
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=BRAND['grid_color'], linecolor=BRAND['line_color'],
        tickfont=dict(color=BRAND['font_color']), title_font=dict(color=BRAND['font_color'])
    )
    fig.update_yaxes(
        gridcolor=BRAND['grid_color'], linecolor=BRAND['line_color'],
        tickfont=dict(color=BRAND['font_color']), title_font=dict(color=BRAND['font_color'])
    )
    return fig

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "📈 Univariate", "🔗 Bivariate", "🌐 Multivariate", "💡 Business Questions",
])

# ════ TAB 1 — Overview ═══════════════════════════════════════════
with tab1:
    st.subheader("Data Overview")
    st.caption("High-level summary of the dataset including attrition breakdown and outlier detection.")

    stayed_count = (df['Attrition'] == 'Stayed').sum()
    left_count   = (df['Attrition'] == 'Left').sum()
    total        = len(df)
    att_rate     = (df['Attrition'] == 'Left').mean()

    col_donut, col_cards = st.columns([2, 1])
    with col_donut:
        fig_donut = go.Figure(go.Pie(
            labels=['Stayed', 'Left'], values=[stayed_count, left_count], hole=0.6,
            marker_colors=[BRAND['stayed'], BRAND['left']],
            textinfo='label+percent', textfont=dict(color=BRAND['font_color'], size=14),
        ))
        fig_donut.update_layout(
            title="Attrition Breakdown",
            annotations=[dict(text=f"{att_rate:.1%}<br>Attrition", x=0.5, y=0.5,
                              font_size=18, font_color=BRAND['font_color'], showarrow=False)],
            showlegend=True, legend=dict(font=dict(color=BRAND['font_color'])),
        )
        apply_theme(fig_donut, height=450)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_cards:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        for label, value, bg in [
            ("👥 Total Employees", f"{total:,}",       BRAND['card_total']),
            ("✅ Stayed",          f"{stayed_count:,}", BRAND['card_stayed']),
            ("🚪 Left",            f"{left_count:,}",  BRAND['card_left']),
            ("📉 Attrition Rate",  f"{att_rate:.1%}",  BRAND['card_rate']),
        ]:
            st.markdown(f"""
            <div style="background-color:{bg}; border-radius:10px; padding:16px 20px;
                        margin-bottom:12px; border:1px solid {BRAND['card_border']};">
                <div style="color:{BRAND['font_color']}; opacity:0.75; font-size:13px; margin-bottom:4px;">{label}</div>
                <div style="color:{BRAND['font_color']}; font-size:26px; font-weight:600;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

# ════ TAB 2 — Univariate ═════════════════════════════════════════
with tab2:
    st.subheader("Univariate Analysis")
    st.info("💡 Explores each variable independently — distribution, spread, and shape.")
    st.markdown("#### Categorical Columns")
    st.caption("Select a categorical column to view its value distribution.")
    selected_cat = st.selectbox("Select Column", categorical_cols, key="uni_cat")
    fig = px.bar(df[selected_cat].value_counts().reset_index(), x=selected_cat, y='count',
                 title=f"Distribution of {selected_cat}", color_discrete_sequence=BRAND['palette'])
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ════ TAB 3 — Bivariate ══════════════════════════════════════════
with tab3:
    st.subheader("Bivariate Analysis vs Attrition")
    st.info("💡 Explores the relationship between each feature and the Attrition target.")
    st.markdown("#### Categorical vs Attrition")
    st.caption("Compare attrition counts across categories.")
    selected_cat_bi = st.selectbox("Select Column", categorical_cols, key="bi_cat")
    temp = df.groupby([selected_cat_bi, 'Attrition']).size().reset_index(name='count')
    fig  = px.bar(temp, x=selected_cat_bi, y='count', color='Attrition', barmode='group',
                  title=f"{selected_cat_bi} vs Attrition",
                  color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                  category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ════ TAB 4 — Multivariate ═══════════════════════════════════════
with tab4:
    st.subheader("Multivariate Analysis")
    st.info("💡 Examines relationships between multiple variables simultaneously.")
    st.markdown("#### Correlation Heatmap")
    st.caption("Pearson correlation between all numeric features.")
    corr = df[numeric_cols].corr()
    fig  = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, title="Correlation Heatmap")
    apply_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Values above 0.7 or below -0.7 indicate strong linear relationships.")

# ════ TAB 5 — Business Questions ═════════════════════════════════
with tab5:
    st.subheader("Business Questions")

    st.markdown("#### Q1 · Overall attrition & job role breakdown")
    ac = df["Attrition"].value_counts().reset_index()
    ac.columns = ["Attrition", "Count"]
    fig = px.bar(ac, x="Attrition", y="Count", title="Attrition Distribution", color="Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Q2 · Overtime vs Attrition")
    fig = px.bar(df.groupby("Overtime")["Attrition"].value_counts().reset_index(),
                 x="Overtime", y="count", color="Attrition", barmode="group",
                 title="Attrition by Overtime",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                 category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Q3 · Remote Work vs Attrition")
    fig = px.bar(df.groupby("Remote Work")["Attrition"].value_counts().reset_index(),
                 x="Remote Work", y="count", color="Attrition", barmode="group",
                 title="Remote Work vs Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                 category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Q4 · Monthly Income vs Attrition by Job Level")
    fig = px.box(df, x="Job Level", y="Monthly Income", color="Attrition",
                 title="Monthly Income vs Attrition by Job Level",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Q5 · Attrition by Years at Company")
    fig = px.histogram(df, x="Years at Company", color="Attrition", barmode="group",
                       title="Attrition by Years at Company",
                       color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                       category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Q6 · Job Satisfaction × Work-Life Balance")
    combo = df.groupby(["Job Satisfaction", "Work-Life Balance"])["Attrition"].apply(
        lambda x: (x == "Left").mean() * 100).reset_index()
    combo.columns = ["Job Satisfaction", "Work-Life Balance", "Attrition Rate"]
    fig = px.bar(combo.sort_values("Attrition Rate", ascending=False),
                 x="Work-Life Balance", y="Attrition Rate", color="Job Satisfaction",
                 barmode="group", title="Attrition Rate by Job Satisfaction & Work-Life Balance",
                 color_discrete_sequence=BRAND['palette'])
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Q7 · Life-stage factors vs Attrition")
    fig = go.Figure()
    for val, col in zip(["Stayed", "Left"], [BRAND['stayed'], BRAND['left']]):
        sub = df[df["Attrition"] == val]["Age"]
        kde = gaussian_kde(sub)
        xr  = np.linspace(df["Age"].min(), df["Age"].max(), 300)
        fig.add_trace(go.Scatter(x=xr, y=kde(xr), mode="lines", name=val, line=dict(color=col)))
    fig.update_layout(title="Age Distribution by Attrition (KDE)")
    apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    for col_name, title in [("Marital Status", "Attrition by Marital Status"),
                             ("Number of Dependents", "Attrition by Number of Dependents")]:
        fig = px.bar(df.groupby([col_name, "Attrition"]).size().reset_index(name="count"),
                     x=col_name, y="count", color="Attrition", barmode="group", title=title,
                     color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                     category_orders={"Attrition": ["Stayed", "Left"]})
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Q8 · Career Stagnation")
    st.caption("Does feeling stuck line up with leaving? Promotions and job level tell the story.")
    col1, col2 = st.columns(2)
    for c, field, label in [
        (col1, "Number of Promotions", "Attrition Rate by Number of Promotions"),
        (col2, "Job Level",            "Attrition Rate by Job Level"),
    ]:
        d = df.groupby(field)["Attrition"].apply(lambda x: (x == "Left").mean() * 100).reset_index()
        d.columns = [field, "Attrition Rate (%)"]
        fig = px.bar(d, x=field, y="Attrition Rate (%)", title=label,
                     color="Attrition Rate (%)",
                     color_continuous_scale=[[0, BRAND['stayed']], [1, BRAND['left']]])
        apply_theme(fig)
        with c: st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    for c, field, label in [
        (col3, "Leadership Opportunities",  "Attrition Rate by Leadership Opportunities"),
        (col4, "Innovation Opportunities",  "Attrition Rate by Innovation Opportunities"),
    ]:
        d = df.groupby(field)["Attrition"].apply(lambda x: (x == "Left").mean() * 100).reset_index()
        d.columns = [field, "Attrition Rate (%)"]
        fig = px.bar(d, x=field, y="Attrition Rate (%)", title=label,
                     color_discrete_sequence=BRAND['palette'])
        apply_theme(fig)
        with c: st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key finding:** 0–2 promotions → ~49% attrition. 3–4 promotions → ~24%. "
        "Entry-level attrition is 63.3% vs Senior at 20.3%.\n\n"
        "**Recommendation:** Accelerate the Entry → Mid pipeline with transparent promotion timelines."
    )

    st.markdown("---")
    st.markdown("#### Q9 · Highest-Risk Employee Profile")
    st.caption("Combining 4 factors to construct the single highest-risk group.")
    baseline   = (df["Attrition"] == "Left").mean() * 100
    mask = (
        (df["Job Level"] == "Entry") & (df["Remote Work"] == "No") &
        (df["Leadership Opportunities"] == "No") & (df["Innovation Opportunities"] == "No") &
        (df["Overtime"] == "Yes")
    )
    risk_group = df[mask]
    risk_rate  = (risk_group["Attrition"] == "Left").mean() * 100
    risk_count = len(risk_group)
    r1, r2, r3 = st.columns(3)
    r1.metric("Group Attrition Rate", f"{risk_rate:.1f}%",
              delta=f"+{risk_rate - baseline:.1f}pp vs baseline", delta_color="inverse")
    r2.metric("Company Baseline",  f"{baseline:.1f}%")
    r3.metric("Employees at Risk", f"{risk_count:,}")
    st.markdown("""
    **Profile:** Entry-level · No remote work · No leadership/innovation opportunities · Overtime = Yes

    This group is **26.6pp above baseline** — nearly 3 in 4 employees matching this profile will leave.
    """)

    st.markdown("---")
    st.markdown("#### Q10 · What Moves the Needle")
    st.caption("Top drivers ranked by spread between best and worst group attrition rate.")
    drivers_df = pd.DataFrame({
        "Factor":      ["Job Level","Remote Work","Work-Life Balance","Job Satisfaction","Overtime","Innovation Opps","Leadership Opps"],
        "Spread (pp)": [43.0, 28.1, 24.5, 8.0, 6.0, 3.0, 2.8],
    }).sort_values("Spread (pp)", ascending=True)
    fig = px.bar(drivers_df, x="Spread (pp)", y="Factor", orientation="h",
                 title="Attrition Driver Spread (pp between best & worst group)",
                 color="Spread (pp)",
                 color_continuous_scale=[[0, BRAND['secondary']], [1, BRAND['primary']]])
    apply_theme(fig, height=350); st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**#1 Fix: Remote Work policy** — delivers a 28pp gap and can be rolled out in one quarter.\n\n"
        "**Estimated impact:** ~1,600–2,000 additional employees retained per cycle."
    )
