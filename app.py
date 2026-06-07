import base64
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

# ── Detect Theme ──────────────────────────────────────────────────
is_dark = st.get_option("theme.base") != "light"

# ── Brand Colors (theme-aware) ────────────────────────────────────
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
        'card_total':   '#1F6FEB',
        'card_rate':    '#4d3a1a',
    }
else:
    BRAND = {
        'primary':      '#2563EB',       # deep blue — headers & accents
        'secondary':    '#1D4ED8',       # darker blue
        'light':        '#EFF6FF',       # very light blue tint — tab bar bg
        'bg':           '#F8FAFC',       # near-white page background
        'stayed':       '#16A34A',       # strong green
        'left':         '#DC2626',       # strong red
        'chart_title':  '#1E40AF',       # dark blue chart titles
        'palette':      ['#2563EB', '#16A34A', '#DC2626', '#7C3AED', '#D97706'],
        'font_color':   '#1E293B',       # near-black text
        'grid_color':   '#CBD5E1',       # light grey grid
        'line_color':   '#94A3B8',       # medium grey axis lines
        'paper_bg':     '#FFFFFF',       # white chart paper
        'plot_bg':      '#F1F5F9',       # very light slate chart area
        'card_border':  '#BFDBFE',       # light blue card border
        'sidebar_bg':   '#2563EB',       # sidebar stays branded blue
        'card_stayed':  '#DCFCE7',       # light green card
        'card_left':    '#FEE2E2',       # light red card
        'card_total':   '#DBEAFE',       # light blue card
        'card_rate':    '#FEF3C7',       # light amber card
    }

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

# ── Global CSS ────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {BRAND['bg']}; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{ background-color: {BRAND['sidebar_bg']}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] .stSelectbox label {{ color: white !important; }}

    /* ── Headings ── */
    h1, h2, h3 {{ color: {BRAND['primary']} !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {BRAND['light']};
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BRAND['primary']} !important;
        color: white !important;
        border-radius: 6px;
        border: none !important;
        outline: none !important;
    }}
    /* Unselected tab text — readable in light mode */
    .stTabs [data-baseweb="tab"] {{
        border-bottom: none !important;
        box-shadow: none !important;
        color: {BRAND['font_color']} !important;
    }}
    button[role="tab"] {{ border-bottom: none !important; box-shadow: none !important; }}
    button[role="tab"]:focus {{ box-shadow: none !important; outline: none !important; }}

    /* ── Metric cards ── */
    [data-testid="metric-container"] {{
        background-color: {'#161B22' if is_dark else '#EFF6FF'};
        border: 1px solid {BRAND['primary']};
        border-radius: 10px;
        padding: 14px;
    }}

    /* ── Body text ── */
    p, span, label {{ color: {BRAND['font_color']}; }}

    /* ── Info / warning boxes ── */
    {'/* dark: default Streamlit info box */' if is_dark else '''
    [data-testid="stAlert"] {
        background-color: #EFF6FF !important;
        border-left: 4px solid #2563EB !important;
        color: #1E293B !important;
    }
    '''}
</style>

<div style="display:flex; align-items:center; padding:16px 0 24px 0;
            border-bottom:2px solid {BRAND['primary']}; margin-bottom:24px;">
    {logo_html}
    <div>
        <h1 style="margin:0; color:{BRAND['primary']}; font-size:2rem;">Employee Attrition Analytics</h1>
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

numeric_cols = df_full.select_dtypes(include="number").columns.tolist()
categorical_cols = [
    col for col in df_full.select_dtypes(include="object").columns
    if col != "Attrition"
]

# ── Sidebar ───────────────────────────────────────────────────────
if logo_b64:
    st.sidebar.markdown(
        f'<img src="data:image/jpeg;base64,{logo_b64}" width="240" '
        f'style="display:block; margin:0 auto 16px auto;"/>',
        unsafe_allow_html=True
    )

st.sidebar.title("🔍 Filters")
attrition_filter = st.sidebar.selectbox("Attrition", ["All", "Stayed", "Left"])
df = df_full[df_full["Attrition"] == attrition_filter] if attrition_filter != "All" else df_full.copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Records:** {len(df):,}")
st.sidebar.markdown(f"**Attrition Rate:** {(df['Attrition'] == 'Left').mean():.1%}")

# ── Chart Theme Helper ────────────────────────────────────────────
def apply_theme(fig, height=None):
    layout = dict(
        paper_bgcolor=BRAND['paper_bg'],
        plot_bgcolor=BRAND['plot_bg'],
        font=dict(family='Arial', color=BRAND['font_color']),
        title_font=dict(color=BRAND['chart_title'], size=16),
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
    "📋 Overview",
    "📈 Univariate",
    "🔗 Bivariate",
    "🌐 Multivariate",
    "💡 Business Questions",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════
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
            labels=['Stayed', 'Left'],
            values=[stayed_count, left_count],
            hole=0.6,
            marker_colors=[BRAND['stayed'], BRAND['left']],
            textinfo='label+percent',
            textfont=dict(color=BRAND['font_color'], size=14),
        ))
        fig_donut.update_layout(
            title="Attrition Breakdown",
            annotations=[dict(
                text=f"{att_rate:.1%}<br>Attrition",
                x=0.5, y=0.5, font_size=18,
                font_color=BRAND['font_color'], showarrow=False
            )],
            showlegend=True,
            legend=dict(font=dict(color=BRAND['font_color'])),
        )
        apply_theme(fig_donut, height=450)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_cards:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        cards = [
            ("👥 Total Employees", f"{total:,}",       BRAND['card_total']),
            ("✅ Stayed",          f"{stayed_count:,}", BRAND['card_stayed']),
            ("🚪 Left",            f"{left_count:,}",  BRAND['card_left']),
            ("📉 Attrition Rate",  f"{att_rate:.1%}",  BRAND['card_rate']),
        ]
        txt = BRAND['font_color']
        for label, value, bg in cards:
            st.markdown(f"""
            <div style="background-color:{bg}; border-radius:10px; padding:16px 20px;
                        margin-bottom:12px; border:1px solid {BRAND['card_border']};">
                <div style="color:{txt}; opacity:0.7; font-size:13px; margin-bottom:4px;">{label}</div>
                <div style="color:{txt}; font-size:26px; font-weight:600;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — Univariate
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Univariate Analysis")
    st.info("💡 Explores each variable independently — distribution, spread, and shape.")

    st.markdown("#### Categorical Columns")
    st.caption("Select a categorical column to view its value distribution.")
    selected_cat = st.selectbox("Select Column", categorical_cols, key="uni_cat")
    fig = px.bar(
        df[selected_cat].value_counts().reset_index(),
        x=selected_cat, y='count',
        title=f"Distribution of {selected_cat}",
        color_discrete_sequence=BRAND['palette']
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — Bivariate
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Bivariate Analysis vs Attrition")
    st.info("💡 Explores the relationship between each feature and the Attrition target.")

    st.markdown("#### Categorical vs Attrition")
    st.caption("Compare attrition counts across categories.")
    selected_cat_bi = st.selectbox("Select Column", categorical_cols, key="bi_cat")
    temp = df.groupby([selected_cat_bi, 'Attrition']).size().reset_index(name='count')
    fig  = px.bar(temp, x=selected_cat_bi, y='count', color='Attrition',
                  barmode='group', title=f"{selected_cat_bi} vs Attrition",
                  color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                  category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — Multivariate
# ════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════
# TAB 5 — Business Questions
# ════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Business Questions")

    # Q1
    st.markdown("#### Q1 · Overall attrition & job role breakdown")
    attrition_counts = df["Attrition"].value_counts().reset_index()
    attrition_counts.columns = ["Attrition", "Count"]
    fig = px.bar(attrition_counts, x="Attrition", y="Count",
                 title="Attrition Distribution", color="Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q2
    st.markdown("#### Q2 · Overtime vs Attrition")
    overtime_attrition = df.groupby("Overtime")["Attrition"].value_counts().reset_index()
    fig = px.bar(overtime_attrition, x="Overtime", y="count", color="Attrition",
                 barmode="group", title="Attrition by Overtime",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                 category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q3
    st.markdown("#### Q3 · Remote Work vs Attrition")
    remote_attrition = df.groupby("Remote Work")["Attrition"].value_counts().reset_index()
    fig = px.bar(remote_attrition, x="Remote Work", y="count", color="Attrition",
                 barmode="group", title="Remote Work vs Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                 category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q4
    st.markdown("#### Q4 · Monthly Income vs Attrition by Job Level")
    fig = px.box(df, x="Job Level", y="Monthly Income", color="Attrition",
                 title="Monthly Income vs Attrition by Job Level",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q5
    st.markdown("#### Q5 · Attrition by Years at Company")
    fig = px.histogram(df, x="Years at Company", color="Attrition",
                       barmode="group", title="Attrition by Years at Company",
                       color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                       category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q6
    st.markdown("#### Q6 · Job Satisfaction × Work-Life Balance")
    combo = df.groupby(["Job Satisfaction", "Work-Life Balance"])["Attrition"].apply(
        lambda x: (x == "Left").mean() * 100
    ).reset_index()
    combo.columns = ["Job Satisfaction", "Work-Life Balance", "Attrition Rate"]
    combo = combo.sort_values("Attrition Rate", ascending=False)
    fig = px.bar(combo, x="Work-Life Balance", y="Attrition Rate",
                 color="Job Satisfaction", barmode="group",
                 title="Attrition Rate by Job Satisfaction & Work-Life Balance",
                 color_discrete_sequence=BRAND['palette'])
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q7
    st.markdown("#### Q7 · Life-stage factors vs Attrition")
    fig = go.Figure()
    for attrition_val, color in zip(["Stayed", "Left"], [BRAND['stayed'], BRAND['left']]):
        subset  = df[df["Attrition"] == attrition_val]["Age"]
        kde     = gaussian_kde(subset)
        x_range = np.linspace(df["Age"].min(), df["Age"].max(), 300)
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode="lines",
                                 name=attrition_val, line=dict(color=color)))
    fig.update_layout(title="Age Distribution by Attrition (KDE)")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df.groupby(["Marital Status", "Attrition"]).size().reset_index(name="count"),
        x="Marital Status", y="count", color="Attrition", barmode="group",
        title="Attrition by Marital Status",
        color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
        category_orders={"Attrition": ["Stayed", "Left"]}
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df.groupby(["Number of Dependents", "Attrition"]).size().reset_index(name="count"),
        x="Number of Dependents", y="count", color="Attrition", barmode="group",
        title="Attrition by Number of Dependents",
        color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
        category_orders={"Attrition": ["Stayed", "Left"]}
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Q8
    st.markdown("---")
    st.markdown("#### Q8 · Career Stagnation")
    st.caption("Does feeling stuck line up with leaving? Promotions and job level tell the story.")

    col1, col2 = st.columns(2)
    with col1:
        promo_data = df.groupby("Number of Promotions")["Attrition"].apply(
            lambda x: (x == "Left").mean() * 100
        ).reset_index()
        promo_data.columns = ["Number of Promotions", "Attrition Rate (%)"]
        fig = px.bar(promo_data, x="Number of Promotions", y="Attrition Rate (%)",
                     title="Attrition Rate by Number of Promotions",
                     color="Attrition Rate (%)",
                     color_continuous_scale=[[0, BRAND['stayed']], [1, BRAND['left']]])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        level_data = df.groupby("Job Level")["Attrition"].apply(
            lambda x: (x == "Left").mean() * 100
        ).reset_index()
        level_data.columns = ["Job Level", "Attrition Rate (%)"]
        fig = px.bar(level_data, x="Job Level", y="Attrition Rate (%)",
                     title="Attrition Rate by Job Level",
                     color="Attrition Rate (%)",
                     color_continuous_scale=[[0, BRAND['stayed']], [1, BRAND['left']]])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        lead_data = df.groupby("Leadership Opportunities")["Attrition"].apply(
            lambda x: (x == "Left").mean() * 100
        ).reset_index()
        lead_data.columns = ["Leadership Opportunities", "Attrition Rate (%)"]
        fig = px.bar(lead_data, x="Leadership Opportunities", y="Attrition Rate (%)",
                     title="Attrition Rate by Leadership Opportunities",
                     color_discrete_sequence=BRAND['palette'])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        innov_data = df.groupby("Innovation Opportunities")["Attrition"].apply(
            lambda x: (x == "Left").mean() * 100
        ).reset_index()
        innov_data.columns = ["Innovation Opportunities", "Attrition Rate (%)"]
        fig = px.bar(innov_data, x="Innovation Opportunities", y="Attrition Rate (%)",
                     title="Attrition Rate by Innovation Opportunities",
                     color_discrete_sequence=BRAND['palette'])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key finding:** 0–2 promotions → ~49% attrition. 3–4 promotions → ~24%. "
        "Entry-level attrition is 63.3% vs Senior at 20.3%. Leadership/innovation opportunities alone show only ~3pp difference — "
        "actual level progression matters, not just opportunities.\n\n"
        "**Recommendation:** Accelerate the Entry → Mid pipeline. Set transparent promotion timelines "
        "tied to milestones, not just tenure."
    )

    # Q9
    st.markdown("---")
    st.markdown("#### Q9 · Highest-Risk Employee Profile")
    st.caption("Combining 4 factors to construct the single highest-risk group.")

    baseline = (df["Attrition"] == "Left").mean() * 100
    mask = (
        (df["Job Level"] == "Entry") &
        (df["Remote Work"] == "No") &
        (df["Leadership Opportunities"] == "No") &
        (df["Innovation Opportunities"] == "No") &
        (df["Overtime"] == "Yes")
    )
    risk_group = df[mask]
    risk_rate  = (risk_group["Attrition"] == "Left").mean() * 100
    risk_count = len(risk_group)
    risk_delta = risk_rate - baseline

    r1, r2, r3 = st.columns(3)
    r1.metric("Group Attrition Rate", f"{risk_rate:.1f}%",  delta=f"+{risk_delta:.1f}pp vs baseline", delta_color="inverse")
    r2.metric("Company Baseline",     f"{baseline:.1f}%")
    r3.metric("Employees at Risk",    f"{risk_count:,}")

    st.markdown("""
    **Profile:** Entry-level · No remote work · No leadership opportunities · No innovation opportunities · Overtime = Yes

    This group is **26.6pp above baseline** — nearly 3 in 4 employees matching this profile will leave.
    With **6,324 employees** fitting this description, it's large enough to warrant targeted intervention.
    """)

    # Q10
    st.markdown("---")
    st.markdown("#### Q10 · What Moves the Needle")
    st.caption("Top drivers ranked by spread between best and worst group attrition rate.")

    drivers_df = pd.DataFrame({
        "Factor":      ["Job Level", "Remote Work", "Work-Life Balance", "Job Satisfaction", "Overtime", "Innovation Opps", "Leadership Opps"],
        "Spread (pp)": [43.0, 28.1, 24.5, 8.0, 6.0, 3.0, 2.8],
    }).sort_values("Spread (pp)", ascending=True)

    fig = px.bar(drivers_df, x="Spread (pp)", y="Factor", orientation="h",
                 title="Attrition Driver Spread (pp between best & worst group)",
                 color="Spread (pp)",
                 color_continuous_scale=[[0, BRAND['secondary']], [1, BRAND['primary']]])
    apply_theme(fig, height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**#1 Fix: Remote Work policy** — Job Level has the biggest spread (43pp) but is slow and expensive to change. "
        "Remote Work delivers a 28pp gap and can be rolled out as policy in one quarter with zero promotion budget.\n\n"
        "**Estimated impact:** If 20% of in-office employees shift to remote, attrition in that group drops from ~53% → ~40%, "
        "retaining an estimated **1,600–2,000 additional employees** per cycle."
    )
