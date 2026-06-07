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
 
# ── Brand Colors ──────────────────────────────────────────────────
BRAND = {
    'primary':     '#58A6FF',
    'secondary':   '#1F6FEB',
    'light':       '#161B22',
    'bg':          '#0D1117',
    'stayed':      '#3FB950',
    'left':        '#F85149',
    'chart_title': '#58A6FF',
    'palette':     ['#58A6FF', '#3FB950', '#F85149', '#D2A8FF', '#FFA657'],
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
    [data-testid="stSidebar"] {{ background-color: {BRAND['primary']}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] .stSelectbox label {{ color: white !important; }}
    h1, h2, h3 {{ color: {BRAND['primary']} !important; }}
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
    .stTabs [data-baseweb="tab"] {{ border-bottom: none !important; box-shadow: none !important; }}
    button[role="tab"] {{ border-bottom: none !important; box-shadow: none !important; }}
    button[role="tab"]:focus {{ box-shadow: none !important; outline: none !important; }}
    [data-testid="metric-container"] {{
        background-color: white;
        border: 1px solid {BRAND['primary']};
        border-radius: 10px;
        padding: 14px;
    }}
</style>
 
<div style="display:flex; align-items:center; padding:16px 0 24px 0;
            border-bottom:2px solid {BRAND['primary']}; margin-bottom:24px;">
    {logo_html}
    <div>
        <h1 style="margin:0; color:{BRAND['primary']}; font-size:2rem;">Employee Attrition Analytics</h1>
        <p style="margin:4px 0 0 0; color:#ffffff; font-size:1rem;">
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
 
    # Years at Company — remove rows outside [1, 42]
    yc_condition = (df["Years at Company"] > 42) | (df["Years at Company"] < 1)
    df = df[~yc_condition]
 
    # Company Tenure — too many outliers, drop column
    df = df.drop(columns=["Company Tenure"])
 
    # Monthly Income — cap outliers with median
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
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        font=dict(family='Arial', color='#E6EDF3'),
        title_font=dict(color=BRAND['chart_title'], size=16),
    )
    if height:
        layout['height'] = height
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor='#2A3A4A', linecolor='#444',
                     tickfont=dict(color='#E6EDF3'), title_font=dict(color='#E6EDF3'))
    fig.update_yaxes(gridcolor='#2A3A4A', linecolor='#444',
                     tickfont=dict(color='#E6EDF3'), title_font=dict(color='#E6EDF3'))
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
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees",  f"{len(df):,}")
    c2.metric("Stayed",           f"{(df['Attrition'] == 'Stayed').sum():,}")
    c3.metric("Left",             f"{(df['Attrition'] == 'Left').sum():,}")
    c4.metric("Attrition Rate",   f"{(df['Attrition'] == 'Left').mean():.1%}")
 
    st.markdown("---")
    st.subheader("Outlier Detection (IQR Method)")
    st.caption("Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged. Not all flagged values are errors.")
 
    Q1  = df[numeric_cols].quantile(0.25)
    Q3  = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers     = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
    outlier_cols = outliers[outliers > 0].index.tolist()
 
    st.dataframe(outliers.rename("Outlier Count").to_frame(), use_container_width=True)
 
    if outlier_cols:
        rows = math.ceil(len(outlier_cols) / 2)
        fig  = make_subplots(rows=rows, cols=2, subplot_titles=outlier_cols)
        for i, col in enumerate(outlier_cols):
            r, c = i // 2 + 1, i % 2 + 1
            fig.add_trace(
                go.Box(y=df[col], name=col, showlegend=False,
                       marker_color=BRAND['primary'], line_color=BRAND['primary']),
                row=r, col=c
            )
        apply_theme(fig, height=350 * rows)
        fig.update_layout(title="Outlier Boxplots")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points beyond the whiskers are potential outliers.")
 
# ════════════════════════════════════════════════════════════════
# TAB 2 — Univariate
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Univariate Analysis")
    st.info("💡 Explores each variable independently — distribution, spread, and shape.")
 
    # Categorical
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
 
    # Categorical vs Attrition
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
 
    # Q1 — Attrition distribution
    st.markdown("#### Q1 · Overall attrition & job role breakdown")
    attrition_counts = df["Attrition"].value_counts().reset_index()
    attrition_counts.columns = ["Attrition", "Count"]
    fig = px.bar(attrition_counts, x="Attrition", y="Count",
                 title="Attrition Distribution",
                 color="Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 
    # Q2 — Overtime
    st.markdown("#### Q2 · Overtime vs Attrition")
    overtime_attrition = df.groupby("Overtime")["Attrition"].value_counts().reset_index()
    fig = px.bar(overtime_attrition, x="Overtime", y="count", color="Attrition",
                 barmode="group", title="Attrition by Overtime",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                 category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 
    # Q3 — Remote Work
    st.markdown("#### Q3 · Remote Work vs Attrition")
    remote_attrition = df.groupby("Remote Work")["Attrition"].value_counts().reset_index()
    fig = px.bar(remote_attrition, x="Remote Work", y="count", color="Attrition",
                 barmode="group", title="Remote Work vs Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                 category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 
    # Q4 — Income vs Job Level
    st.markdown("#### Q4 · Monthly Income vs Attrition by Job Level")
    fig = px.box(df, x="Job Level", y="Monthly Income", color="Attrition",
                 title="Monthly Income vs Attrition by Job Level",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 
    # Q5 — Years at Company
    st.markdown("#### Q5 · Attrition by Years at Company")
    fig = px.histogram(df, x="Years at Company", color="Attrition",
                       barmode="group", title="Attrition by Years at Company",
                       color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
                       category_orders={"Attrition": ["Stayed", "Left"]})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 
    # Q6 — Job Satisfaction x Work-Life Balance
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
 
    # Q7 — Age, Marital Status, Dependents
    st.markdown("#### Q7 · Life-stage factors vs Attrition")
 
    # Age KDE
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
 
    # Marital Status
    fig = px.bar(
        df.groupby(["Marital Status", "Attrition"]).size().reset_index(name="count"),
        x="Marital Status", y="count", color="Attrition", barmode="group",
        title="Attrition by Marital Status",
        color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
        category_orders={"Attrition": ["Stayed", "Left"]}
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 
    # Number of Dependents
    fig = px.bar(
        df.groupby(["Number of Dependents", "Attrition"]).size().reset_index(name="count"),
        x="Number of Dependents", y="count", color="Attrition", barmode="group",
        title="Attrition by Number of Dependents",
        color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']},
        category_orders={"Attrition": ["Stayed", "Left"]}
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
 








