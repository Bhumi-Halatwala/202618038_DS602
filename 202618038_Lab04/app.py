import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import streamlit as st
import joblib

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "Data" / "insurance.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "insurance_ols.pkl"
META_PATH  = BASE_DIR / "artifacts" / "model_metadata.json"

ALPHA = 0.05

st.set_page_config(page_title="Insurance Cost Dashboard",
                   page_icon="🏥", layout="wide")


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


df    = load_data()
model = load_model()
meta  = load_meta()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🏥 Medical Insurance Cost — Statistical Dashboard")
st.caption("M.Sc. Data Science · Lab-4 · Roll No: 202618038")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)


# ==================================================================
# TAB 1 — Data Exploration
# ==================================================================
with tab1:
    st.header("Interactive Data Exploration")

    with st.sidebar:
        st.markdown("### 🔎 Filters")
        age_r = st.slider("Age range",
                          int(df["age"].min()), int(df["age"].max()),
                          (int(df["age"].min()), int(df["age"].max())))
        bmi_r = st.slider("BMI range",
                          float(df["bmi"].min()), float(df["bmi"].max()),
                          (float(df["bmi"].min()), float(df["bmi"].max())))
        sex_s   = st.multiselect("Sex",    sorted(df["sex"].unique()),    sorted(df["sex"].unique()))
        smok_s  = st.multiselect("Smoker", sorted(df["smoker"].unique()), sorted(df["smoker"].unique()))
        reg_s   = st.multiselect("Region", sorted(df["region"].unique()), sorted(df["region"].unique()))

    mask = (df["age"].between(*age_r)
            & df["bmi"].between(*bmi_r)
            & df["sex"].isin(sex_s)
            & df["smoker"].isin(smok_s)
            & df["region"].isin(reg_s))
    dff = df[mask]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dff):,}")
    c2.metric("Avg Charges", f"${dff['charges'].mean():,.0f}" if len(dff) else "—")
    c3.metric("Avg BMI", f"{dff['bmi'].mean():.2f}" if len(dff) else "—")
    c4.metric("Avg Age", f"{dff['age'].mean():.1f}" if len(dff) else "—")

    st.subheader("Summary Statistics")
    st.dataframe(dff.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Distribution")
    num_col = st.selectbox("Numeric column", ["age", "bmi", "children", "charges"])
    fig_hist = px.histogram(dff, x=num_col, nbins=40, marginal="box",
                            color="smoker",
                            title=f"Distribution of {num_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Scatter Explorer")
    col1, col2 = st.columns(2)
    x_ax = col1.selectbox("X-axis", ["age", "bmi", "children"])
    c_by = col2.selectbox("Color by", ["smoker", "sex", "region"])
    fig_scatter = px.scatter(dff, x=x_ax, y="charges", color=c_by,
                             opacity=0.65, title=f"Charges vs {x_ax}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = dff[["age", "bmi", "children", "charges"]].corr().round(3)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Pearson Correlation")
    st.plotly_chart(fig_corr, use_container_width=True)


# ==================================================================
# TAB 2 — Hypothesis Testing Lab
# ==================================================================
with tab2:
    st.header("Hypothesis Testing Lab")
    st.caption(f"Significance level α = {ALPHA}")

    test_type = st.radio(
        "Choose a test",
        ["Two-Group (t-test / Mann-Whitney)",
         "Multi-Group (One-Way ANOVA)"],
        horizontal=True,
    )

    # ---------- Two-group ----------
    if test_type.startswith("Two"):
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Categorical factor", ["smoker", "sex"])
        num_col = c2.selectbox("Numeric metric", ["charges", "bmi", "age"])

        groups = sorted(df[cat_col].unique())
        c1, c2 = st.columns(2)
        g1 = c1.selectbox("Group A", groups, index=0)
        g2 = c2.selectbox("Group B", groups, index=1 if len(groups) > 1 else 0)

        if g1 == g2:
            st.warning("Choose two different groups.")
        else:
            a = df.loc[df[cat_col] == g1, num_col]
            b = df.loc[df[cat_col] == g2, num_col]

            st.markdown(f"**H0:** Distribution of `{num_col}` is the same for "
                        f"`{cat_col}={g1}` and `{cat_col}={g2}`.")
            st.markdown(f"**H1:** Distribution of `{num_col}` differs.")

            sh_a = stats.shapiro(a.sample(min(len(a), 5000), random_state=0))
            sh_b = stats.shapiro(b.sample(min(len(b), 5000), random_state=0))
            lev  = stats.levene(a, b, center="median")

            normal    = (sh_a.pvalue > ALPHA) and (sh_b.pvalue > ALPHA)
            equal_var = lev.pvalue > ALPHA

            if normal:
                stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
                test_name = "Two-Sample t-test"
            else:
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                test_name = "Mann-Whitney U test"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test", test_name)
            c2.metric("Statistic", f"{stat:,.3f}")
            c3.metric("p-value", f"{p:.10f}")
            c4.metric("Decision", "Reject H0" if p < ALPHA else "Fail to Reject H0")

            st.markdown("#### Assumption Checks")
            st.write(f"- Shapiro-Wilk `{g1}`: W = {sh_a.statistic:.4f}, p = {sh_a.pvalue:.10f}")
            st.write(f"- Shapiro-Wilk `{g2}`: W = {sh_b.statistic:.4f}, p = {sh_b.pvalue:.10f}")
            st.write(f"- Levene's test: stat = {lev.statistic:.4f}, p = {lev.pvalue:.10f}")

            st.markdown("#### Group Summary")
            st.dataframe(
                pd.DataFrame({
                    "Group":  [g1, g2],
                    "N":      [len(a), len(b)],
                    "Mean":   [a.mean(), b.mean()],
                    "Median": [a.median(), b.median()],
                    "Std":    [a.std(), b.std()],
                }).set_index("Group").round(3),
                use_container_width=True,
            )

            fig_box = px.box(df, x=cat_col, y=num_col, color=cat_col,
                             title=f"{num_col} by {cat_col}")
            st.plotly_chart(fig_box, use_container_width=True)

    # ---------- One-way ANOVA ----------
    else:
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Categorical factor", ["region"])
        num_col = c2.selectbox("Numeric metric", ["charges", "bmi", "age"])

        groups  = sorted(df[cat_col].unique())
        samples = [df.loc[df[cat_col] == g, num_col].values for g in groups]

        st.markdown(f"**H0:** Mean `{num_col}` is equal across all `{cat_col}` groups.")
        st.markdown(f"**H1:** At least one group mean differs.")

        lev      = stats.levene(*samples, center="median")
        f_stat, p = stats.f_oneway(*samples)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test", "One-Way ANOVA")
        c2.metric("F-statistic", f"{f_stat:.3f}")
        c3.metric("p-value", f"{p:.10f}")
        c4.metric("Decision", "Reject H0" if p < ALPHA else "Fail to Reject H0")

        st.write(f"- Levene's test: stat = {lev.statistic:.4f}, p = {lev.pvalue:.10f}")

        st.markdown("#### Group Summary")
        st.dataframe(
            df.groupby(cat_col)[num_col].agg(["count", "mean", "std", "median"]).round(3),
            use_container_width=True,
        )

        fig_box = px.box(df, x=cat_col, y=num_col, color=cat_col,
                         title=f"{num_col} by {cat_col}")
        st.plotly_chart(fig_box, use_container_width=True)


# ==================================================================
# TAB 3 — Live Prediction & Diagnostics
# ==================================================================
with tab3:
    st.header("Live Prediction & Diagnostics")

    st.markdown("### 🎛️ Enter Beneficiary Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.slider("Age", 18, 64, 35)
        bmi      = st.slider("BMI", 15.0, 55.0, 28.0, 0.1)
    with c2:
        children = st.slider("Children", 0, 5, 1)
        sex      = st.selectbox("Sex", sorted(df["sex"].unique()))
    with c3:
        smoker   = st.selectbox("Smoker", sorted(df["smoker"].unique()))
        region   = st.selectbox("Region", sorted(df["region"].unique()))

    row = {
        "const":            1.0,
        "age":              age,
        "bmi":              bmi,
        "children":         children,
        "sex_male":         int(sex == "male"),
        "smoker_yes":       int(smoker == "yes"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
        "smoker_bmi":       bmi * int(smoker == "yes"),
    }
    X_new = pd.DataFrame([row])[meta["train_columns"]]

    pred = model.get_prediction(X_new).summary_frame(alpha=ALPHA)

    point = float(pred["mean"].iloc[0])
    ci_lo = float(pred["mean_ci_lower"].iloc[0])
    ci_hi = float(pred["mean_ci_upper"].iloc[0])
    pi_lo = float(pred["obs_ci_lower"].iloc[0])
    pi_hi = float(pred["obs_ci_upper"].iloc[0])

    st.markdown("### 💰 Predicted Charges")
    c1, c2, c3 = st.columns(3)
    c1.metric("Point Estimate",        f"${point:,.0f}")
    c2.metric("95% CI (mean)",         f"${ci_lo:,.0f} – ${ci_hi:,.0f}")
    c3.metric("95% PI (individual)",   f"${pi_lo:,.0f} – ${pi_hi:,.0f}")

    st.markdown("### 📈 Model Diagnostics")
    fitted = model.fittedvalues
    resid  = model.resid

    col1, col2 = st.columns(2)

    fig1 = px.scatter(x=fitted, y=resid, opacity=0.5,
                      labels={"x": "Fitted", "y": "Residuals"},
                      title="Residuals vs Fitted")
    fig1.add_hline(y=0, line_dash="dash", line_color="red")
    col1.plotly_chart(fig1, use_container_width=True)

    qq = stats.probplot(resid, dist="norm", fit=True)
    qq_x, qq_y = qq[0][0], qq[0][1]
    slope, intercept = qq[1][0], qq[1][1]
    fig2 = px.scatter(x=qq_x, y=qq_y,
                      labels={"x": "Theoretical Quantiles",
                              "y": "Sample Quantiles"},
                      title="Q-Q Plot of Residuals")
    fig2.add_scatter(x=qq_x, y=slope * qq_x + intercept,
                     mode="lines", name="Fit",
                     line=dict(color="red"))
    col2.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📊 Model Fit Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (train)",     f"{meta['model_stats']['r2']:.4f}")
    c2.metric("Adj. R²",        f"{meta['model_stats']['adj_r2']:.4f}")
    c3.metric("RMSE (test)",    f"${meta['model_stats']['rmse_test']:,.0f}")
    c4.metric("MAE (test)",     f"${meta['model_stats']['mae_test']:,.0f}")

    with st.expander("📋 Full Coefficient Table"):
        coef_df = pd.DataFrame({
            "Coef":    model.params,
            "Std Err": model.bse,
            "t":       model.tvalues,
            "p-value": model.pvalues,
            "CI Low":  model.conf_int()[0],
            "CI High": model.conf_int()[1],
        }).round(4)
        st.dataframe(coef_df, use_container_width=True)

st.markdown("---")
st.caption("© 2026 · M.Sc. Data Science · Roll No: 202618038")