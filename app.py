import base64
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

# ── Brand Colors ──────────────────────────────────────────────────
BRAND = {
    'primary': '#58A6FF', 'secondary': '#1F6FEB',
    'light': '#161B22', 'bg': '#0D1117',
    'stayed': '#3FB950', 'left': '#F85149',
    'title_color': '#58A6FF', 'chart_title': '#58A6FF',
    'palette': ['#58A6FF', '#3FB950', '#F85149', '#D2A8FF', '#FFA657'],
}

# ── Logo ──────────────────────────────────────────────────────────
def get_logo_base64():
    try:
        with open("kayfaio_logo.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

logo_b64 = get_logo_base64()
logo_html = (
    f'<img src="data:image/jpeg;base64,{logo_b64}" width="100" style="margin-left:20px;"/>'
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
    .stTabs [data-baseweb="tab"] {{
        border-bottom: none !important;
        box-shadow: none !important;
    }}
    button[role="tab"] {{
        border-bottom: none !important;
        box-shadow: none !important;
    }}
    button[role="tab"]:focus {{
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="metric-container"] {{
        background-color: white;
        border: 1px solid {BRAND['primary']};
        border-radius: 10px;
        padding: 14px;
    }}
</style>

<div style="display:flex; align-items:center; padding:16px 0 24px 0; border-bottom:2px solid {BRAND['primary']}; margin-bottom:24px;">
    {logo_html}
    <div>
        <h1 style="margin:0; color:{BRAND['primary']}; font-size:2rem;">Employee Attrition Analytics</h1>
        <p style="margin:4px 0 0 0; color:#ffffff; font-size:1rem;">Predict whether an employee <b>Stayed</b> or <b>Left</b></p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    data  = pd.concat([train, test], ignore_index=True)
    median_tenure = data['Company Tenure'].median()
    data.loc[data['Company Tenure'] > 40, 'Company Tenure'] = median_tenure
    return data.copy()

df_full = load_data()

numeric_cols = [
    'Age', 'Years at Company', 'Monthly Income',
    'Number of Promotions', 'Distance from Home',
    'Number of Dependents', 'Company Tenure'
]

categorical_cols = [
    col for col in [
        'Gender', 'Job Role', 'Work-Life Balance', 'Job Satisfaction',
        'Performance Rating', 'Overtime', 'Education Level', 'Marital Status',
        'Job Level', 'Company Size', 'Remote Work', 'Leadership Opportunities',
        'Innovation Opportunities', 'Company Reputation', 'Employee Recognition'
    ] if col != 'Attrition'
]

# ── Sidebar ───────────────────────────────────────────────────────
if logo_b64:
    st.sidebar.markdown(
        f'<img src="data:image/jpeg;base64,{logo_b64}" width="100" style="display:block; margin:0 auto 16px auto;"/>',
        unsafe_allow_html=True
    )

st.sidebar.title("🔍 Filters")
attrition_filter = st.sidebar.selectbox("Attrition", ["All", "Stayed", "Left"])
df = df_full[df_full["Attrition"] == attrition_filter] if attrition_filter != "All" else df_full.copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Records:** {len(df):,}")
st.sidebar.markdown(f"**Attrition Rate:** {(df['Attrition'] == 'Left').mean():.1%}")

# ── Chart theme helper ────────────────────────────────────────────
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
    fig.update_xaxes(
        gridcolor='#2A3A4A', linecolor='#444',
        tickfont=dict(color='#E6EDF3'),
        title_font=dict(color='#E6EDF3')
    )
    fig.update_yaxes(
        gridcolor='#2A3A4A', linecolor='#444',
        tickfont=dict(color='#E6EDF3'),
        title_font=dict(color='#E6EDF3')
    )
    return fig

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview",
    "📈 Univariate",
    "🔗 Bivariate",
    "🌐 Multivariate"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Data Overview")
    st.caption("A high-level summary of the dataset including total employees, attrition breakdown, and key statistics.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(df):,}")
    c2.metric("Stayed", f"{(df['Attrition'] == 'Stayed').sum():,}")
    c3.metric("Left", f"{(df['Attrition'] == 'Left').sum():,}")
    c4.metric("Attrition Rate", f"{(df['Attrition'] == 'Left').mean():.1%}")

    st.markdown("---")
    st.subheader("Sample Data")
    st.caption("First 10 rows of the combined train and test dataset.")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Data Info")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Values**")
        st.caption("Columns with missing data that may require imputation or removal.")
        missing = df.isnull().sum()
        st.dataframe(
            missing[missing > 0] if missing.any()
            else pd.DataFrame({"Status": ["No missing values"]})
        )
    with c2:
        st.markdown("**Numeric Summary**")
        st.caption("Descriptive statistics (mean, std, min, max, quartiles) for all numeric columns.")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Outlier Detection (IQR Method)")
    st.caption("Outliers are detected using the IQR rule: values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged. Note that flagged values are not necessarily errors — they may represent real edge cases.")
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
    st.dataframe(outliers.rename("Outlier Count").to_frame(), use_container_width=True)

    outlier_cols = outliers[outliers > 0].index.tolist()
    if outlier_cols:
        cols_per_row = 2
        rows = math.ceil(len(outlier_cols) / cols_per_row)
        fig = make_subplots(rows=rows, cols=cols_per_row, subplot_titles=outlier_cols)
        for i, col in enumerate(outlier_cols):
            r = i // cols_per_row + 1
            c = i % cols_per_row + 1
            fig.add_trace(
                go.Box(y=df[col], name=col, showlegend=False,
                       marker_color=BRAND['primary'], line_color=BRAND['primary']),
                row=r, col=c
            )
        apply_theme(fig, height=350 * rows)
        fig.update_layout(title="Outlier Boxplots")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each box shows the median, Q1, Q3, and whiskers at 1.5×IQR. Points beyond the whiskers are potential outliers.")


# ════════════════════════════════════════════════════════════════
# TAB 2 — Univariate
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Univariate Analysis")
    st.info("💡 Univariate analysis explores each variable independently — looking at its distribution, spread, and shape without considering other variables.")

    st.markdown("#### Numeric Columns")
    st.caption("Histograms showing the frequency distribution of each numeric feature. Look for skewness, peaks, and unusual gaps.")
    cols_per_row = 3
    rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig = make_subplots(rows=rows, cols=cols_per_row, subplot_titles=numeric_cols)
    for i, col in enumerate(numeric_cols):
        r = i // cols_per_row + 1
        c = i % cols_per_row + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False,
                         marker_color=BRAND['secondary'], opacity=0.85),
            row=r, col=c
        )
    apply_theme(fig, height=300 * rows)
    fig.update_layout(title="Numeric Distributions")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Right-skewed distributions (long tail to the right) suggest a few high values pulling the mean up — common in Income and Tenure columns.")

    st.markdown("#### Categorical Columns")
    st.caption("Select a categorical column to see how its values are distributed across the dataset.")
    selected_cat = st.selectbox("Select Categorical Column", categorical_cols, key="uni_cat")
    fig = px.bar(
        df[selected_cat].value_counts().reset_index(),
        x=selected_cat, y='count',
        title=f"Distribution of {selected_cat}",
        color_discrete_sequence=BRAND['palette']
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"This chart shows the count of each unique value in '{selected_cat}'. Imbalanced categories may affect model performance.")


# ════════════════════════════════════════════════════════════════
# TAB 3 — Bivariate
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Bivariate Analysis vs Attrition")
    st.info("💡 Bivariate analysis explores the relationship between each feature and the target variable (Attrition). This helps identify which factors are most associated with employees leaving.")

    st.markdown("#### Numeric vs Attrition")
    st.caption("Box plots comparing the distribution of each numeric feature between employees who Stayed and those who Left. A visible gap between the two boxes suggests the feature is predictive of attrition.")
    cols_per_row = 2
    rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig = make_subplots(rows=rows, cols=cols_per_row, subplot_titles=numeric_cols)
    for i, col in enumerate(numeric_cols):
        r = i // cols_per_row + 1
        c = i % cols_per_row + 1
        for val, color in zip(['Stayed', 'Left'], [BRAND['stayed'], BRAND['left']]):
            fig.add_trace(
                go.Box(y=df[df['Attrition'] == val][col], name=val,
                       marker_color=color, showlegend=(i == 0)),
                row=r, col=c
            )
    apply_theme(fig, height=350 * rows)
    fig.update_layout(boxmode='group', title="Numeric vs Attrition")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = Stayed, Red = Left. Features where the two boxes are clearly separated are strong attrition indicators.")

    st.markdown("#### Categorical vs Attrition")
    st.caption("Select a categorical feature to compare attrition rates across its groups. Taller red bars indicate higher attrition within that group.")
    selected_cat_bi = st.selectbox("Select Categorical Column", categorical_cols, key="bi_cat")
    temp = df.groupby([selected_cat_bi, 'Attrition']).size().reset_index(name='count')
    fig = px.bar(temp, x=selected_cat_bi, y='count', color='Attrition',
                 barmode='group', title=f"{selected_cat_bi} vs Attrition",
                 color_discrete_map={'Stayed': BRAND['stayed'], 'Left': BRAND['left']})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Groups with a high Left-to-Stayed ratio in '{selected_cat_bi}' are worth investigating further as potential attrition risk factors.")


# ════════════════════════════════════════════════════════════════
# TAB 4 — Multivariate
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Multivariate Analysis")
    st.info("💡 Multivariate analysis examines relationships between multiple variables simultaneously — revealing patterns that single-variable analysis would miss.")

    st.markdown("#### Correlation Heatmap")
    st.caption("Pearson correlation between all numeric features. Values close to +1 indicate a strong positive relationship; values close to -1 indicate a strong negative relationship. Values near 0 suggest no linear relationship.")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, title="Correlation Heatmap")
    apply_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Focus on values above 0.7 or below -0.7 — these indicate strong linear relationships worth investigating before modeling to avoid multicollinearity.")
