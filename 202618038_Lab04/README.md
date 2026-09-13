# Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

> **Course:** Statistical Modeling with Python (DS602)  
> **Program:** M.Sc. Data Science — Semester 1  
> **Roll No:** 202618038  
> **Dataset:** Medical Insurance Costs (1,337 records, 7 features)  
> **Tools:** Python 3.13 · statsmodels · scipy · Plotly · Streamlit · VS Code

---

## 📌 Overview

This project delivers a complete end-to-end statistical analysis of a medical insurance cost dataset, culminating in an **interactive Streamlit dashboard**. It covers:

1. **Exploratory Data Analysis (EDA)** — descriptive statistics, distributional analysis, correlation study.
2. **Hypothesis Testing** — parametric and non-parametric tests with formal assumption checks.
3. **Multiple Linear Regression** — OLS estimation via `statsmodels`, with full Gauss-Markov diagnostics.
4. **Interactive Web Delivery** — a 3-tab Streamlit dashboard enabling live data exploration, hypothesis testing, and real-time prediction with confidence/prediction intervals.

---

## 📁 Repository Structure

```
202618038_Lab04/
├── Data/
│   └── insurance.csv                    # Raw dataset (1,337 rows after dedup)
├── artifacts/                           # Created after running the notebook
│   ├── insurance_ols.pkl                # Serialized trained OLS model
│   └── model_metadata.json              # Feature list + performance metrics
├── 202618038_Lab04.ipynb                # Full analysis notebook (Parts 1 & 2)
├── app.py                               # Streamlit dashboard (Part 3)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 📊 Dataset

**Source:** [Medical Insurance Cost Dataset (Kaggle)](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age of primary beneficiary (18–64) |
| `sex` | categorical | male / female |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents covered (0–5) |
| `smoker` | categorical | yes / no |
| `region` | categorical | northeast / northwest / southeast / southwest |
| `charges` | float | **Target** — annual medical cost billed (USD) |

- **Size:** 1,338 rows (1,337 after removing 1 duplicate)
- **Missing values:** None
- **Target range:** $1,122 – $63,770

---

## 🔬 Methodology

### Part 1 — EDA & Hypothesis Testing

**Descriptive statistics** computed for all numeric features: mean, median, standard deviation, IQR, skewness, kurtosis.

**Visual exploration:** histograms + KDE, scatter plots (charges vs age / BMI), and a correlation heatmap.

**Hypothesis Test 1 (2 groups):**
- **H0:** Charges distribution is the same for smokers and non-smokers.
- **H1:** Charges differ between the two groups.
- Assumption checks: Shapiro-Wilk (normality), Levene's (equal variance).
- Since both groups violate normality → **Mann-Whitney U test** (non-parametric).

**Hypothesis Test 2 (3+ groups):**
- **H0:** Mean charges are equal across all 4 US regions.
- **H1:** At least one region's mean differs.
- **One-Way ANOVA** with Levene's variance check.

### Part 2 — Multiple Linear Regression

**Model:**
```
charges = β₀ + β₁·age + β₂·bmi + β₃·children + β₄·sex_male
        + β₅·smoker_yes + β₆·region_nw + β₇·region_se + β₈·region_sw
        + β₉·(smoker × bmi) + ε
```

Fitted using `statsmodels.api.OLS` with an 80/20 train/test split.

**Gauss-Markov diagnostics:**
- **Linearity & Homoscedasticity** — Residuals vs Fitted plot
- **Normality of residuals** — Q-Q plot, Jarque-Bera, Omnibus
- **Multicollinearity** — Variance Inflation Factor (VIF)
- **Heteroscedasticity** — Breusch-Pagan test

### Part 3 — Interactive Dashboard

Three tabs in Streamlit:

| Tab | Purpose |
|---|---|
| **📊 Data Exploration** | Sidebar filters (age, BMI, sex, smoker, region) + reactive Plotly visuals + live summary stats |
| **🧪 Hypothesis Testing Lab** | Dropdowns to select categorical factor + numeric metric → auto-runs appropriate test with clear Reject / Fail-to-Reject conclusion |
| **🔮 Live Prediction & Diagnostics** | Sliders/number inputs → live point prediction + 95% CI + 95% PI, plus residual diagnostic plots |

---

## 📈 Key Findings

### Hypothesis Tests

| Test | Statistic | p-value | Decision |
|---|---|---|---|
| Shapiro-Wilk (smoker=yes) | W = 0.9396 | ≈ 0 | Normality rejected |
| Shapiro-Wilk (smoker=no) | W = 0.8729 | ≈ 0 | Normality rejected |
| Levene's | 332.47 | ≈ 0 | Unequal variances |
| **Mann-Whitney U** (smoker vs non) | 283,859 | ≈ 0 | ✅ **Reject H0** |
| **One-Way ANOVA** (regions) | F = 2.926 | **0.0328** | ✅ **Reject H0** |

- **Smokers pay 3.8× more** than non-smokers on average ($32,050 vs $8,441).
- **Region** has a statistically significant but weak effect. The **southeast** shows the highest mean charges ($14,735), likely due to higher smoking prevalence.

### Regression Model

| Metric | Value |
|---|---|
| **R² (train)** | **0.8247** |
| **Adjusted R² (train)** | **0.8232** |
| **R² (test)** | **0.8862** |
| **RMSE (test)** | **$4,572.81** |
| **MAE (test)** | **$2,828.97** |
| **F-statistic** | 553.70 (p ≈ 0) |
| **N (train / test)** | 1,069 / 268 |

**No overfitting** — test R² exceeds train R².

**Significant predictors (p < 0.05):**
- `age` (+$259/year)
- `children` (+$570/child)
- `sex_male` (−$647)
- `region_southeast` (−$896 vs northeast)
- `region_southwest` (−$885 vs northeast)
- **`smoker_bmi` (+$1,477 per BMI point for smokers)** ← dominant effect

**Interaction interpretation:** The smoking premium is not fixed — it grows with BMI:

| BMI | Smoker premium |
|---|---|
| 20 | +$7,844 |
| 30 | +$22,610 |
| 40 | +$37,376 |

### Gauss-Markov Diagnostics

| Assumption | Test | Result |
|---|---|---|
| Linearity | Residuals vs Fitted | ⚠️ Violated (curved pattern) |
| Homoscedasticity | Breusch-Pagan | ✅ LM = 6.19, p = 0.72 (holds) |
| Normality of residuals | Jarque-Bera | ❌ stat = 3,696, p ≈ 0 |
| Normality of residuals | Omnibus | ❌ stat = 590, p ≈ 0 |
| Multicollinearity | VIF | ⚠️ `smoker_yes` (25.2) and `smoker_bmi` (25.5) high — structural, from the interaction term. All others < 2. |

**Notable:** Residuals are non-normal and there is mild non-linearity, but the large sample size (n ≈ 1,070) protects coefficient inference via the Central Limit Theorem. Standard errors remain approximately valid.

---

## 🎯 Business Interpretation

1. **Smoking is the dominant cost driver.** The interaction with BMI shows that public-health interventions targeting **obese smokers** yield the highest cost savings.
2. **Age is a clean, predictable effect.** Every year adds ~$259 — useful for actuarial forecasting.
3. **Region effect is weak.** After controlling for smoking and BMI, the southeast no longer appears expensive (Simpson's paradox — its raw high cost is due to higher smoking prevalence).
4. **Gender effect is negligible** (−$647, borderline p = 0.033).
5. **Recommendation:** Any predictive pricing model for insurance should absolutely include the **smoker × BMI interaction term** — it alone explains a large chunk of variance.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Statistics | `scipy.stats`, `statsmodels` |
| Modeling | `statsmodels.api.OLS`, `scikit-learn` |
| Visualization (notebook) | `matplotlib`, `seaborn`, `plotly.express` |
| Visualization (dashboard) | `plotly.express` + `streamlit` |
| Serialization | `joblib`, `json` |

---

## 📚 References

1. Kaggle — [Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)
2. `statsmodels` documentation — https://www.statsmodels.org
3. Streamlit documentation — https://docs.streamlit.io
4. `scipy.stats` reference — https://docs.scipy.org/doc/scipy/reference/stats.html

---

## 👤 Author

**Roll No:** 202618038  
**Program:** M.Sc. Data Science — Semester 1  
**Institution:** Dhirubhai Ambani University

---

## 📜 License

This project is submitted as academic coursework. Dataset is licensed under the original Kaggle dataset terms.
