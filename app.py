"""
Streamlit App — Cardiotocography (CTG) Fetal State Classifier
Predicts fetal state (Normal / Suspect / Pathologic) using a trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CTG Fetal State Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Prediction result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .result-suspect {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border: 2px solid #ffc107;
    }
    .result-pathologic {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-desc {
        font-size: 1rem;
        color: #495057;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Feature Configuration
# ──────────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "LB":       {"label": "Baseline Value (LB)",                      "min": 106.0, "max": 160.0, "default": 133.0, "step": 1.0,  "desc": "Fetal heart rate baseline (bpm)"},
    "AC":       {"label": "Accelerations (AC)",                       "min": 0.0,   "max": 26.0,  "default": 3.0,   "step": 1.0,  "desc": "Number of accelerations per second"},
    "FM":       {"label": "Fetal Movement (FM)",                      "min": 0.0,   "max": 564.0, "default": 7.0,   "step": 1.0,  "desc": "Number of fetal movements per second"},
    "UC":       {"label": "Uterine Contractions (UC)",                "min": 0.0,   "max": 23.0,  "default": 4.0,   "step": 1.0,  "desc": "Number of uterine contractions per second"},
    "ASTV":     {"label": "Abnormal Short Term Variability (ASTV %)", "min": 12.0,  "max": 87.0,  "default": 47.0,  "step": 1.0,  "desc": "Percentage of time with abnormal STV"},
    "MSTV":     {"label": "Mean Short Term Variability (MSTV)",       "min": 0.2,   "max": 7.0,   "default": 1.3,   "step": 0.1,  "desc": "Mean value of short term variability"},
    "ALTV":     {"label": "Abnormal Long Term Variability (ALTV %)",  "min": 0.0,   "max": 91.0,  "default": 10.0,  "step": 1.0,  "desc": "Percentage of time with abnormal LTV"},
    "MLTV":     {"label": "Mean Long Term Variability (MLTV)",        "min": 0.0,   "max": 50.7,  "default": 8.2,   "step": 0.1,  "desc": "Mean value of long term variability"},
    "DL":       {"label": "Light Decelerations (DL)",                 "min": 0.0,   "max": 16.0,  "default": 0.0,   "step": 1.0,  "desc": "Number of light decelerations per second"},
    "DS":       {"label": "Severe Decelerations (DS)",                "min": 0.0,   "max": 1.0,   "default": 0.0,   "step": 1.0,  "desc": "Number of severe decelerations per second"},
    "DP":       {"label": "Prolonged Decelerations (DP)",             "min": 0.0,   "max": 4.0,   "default": 0.0,   "step": 1.0,  "desc": "Number of prolonged decelerations per second"},
    "Width":    {"label": "Histogram Width",                          "min": 3.0,   "max": 180.0, "default": 70.0,  "step": 1.0,  "desc": "Width of FHR histogram"},
    "Min":      {"label": "Histogram Min",                            "min": 50.0,  "max": 159.0, "default": 94.0,  "step": 1.0,  "desc": "Minimum of FHR histogram"},
    "Max":      {"label": "Histogram Max",                            "min": 122.0, "max": 238.0, "default": 164.0, "step": 1.0,  "desc": "Maximum of FHR histogram"},
    "Nmax":     {"label": "Histogram Peaks (Nmax)",                   "min": 0.0,   "max": 18.0,  "default": 4.0,   "step": 1.0,  "desc": "Number of histogram peaks"},
    "Nzeros":   {"label": "Histogram Zeros (Nzeros)",                 "min": 0.0,   "max": 10.0,  "default": 0.0,   "step": 1.0,  "desc": "Number of histogram zeros"},
    "Mode":     {"label": "Histogram Mode",                           "min": 60.0,  "max": 187.0, "default": 137.0, "step": 1.0,  "desc": "Histogram mode"},
    "Mean":     {"label": "Histogram Mean",                           "min": 73.0,  "max": 182.0, "default": 135.0, "step": 1.0,  "desc": "Histogram mean"},
    "Median":   {"label": "Histogram Median",                         "min": 77.0,  "max": 186.0, "default": 138.0, "step": 1.0,  "desc": "Histogram median"},
    "Variance": {"label": "Histogram Variance",                       "min": 0.0,   "max": 269.0, "default": 19.0,  "step": 1.0,  "desc": "Histogram variance"},
    "Tendency": {"label": "Histogram Tendency",                       "min": -1.0,  "max": 1.0,   "default": 0.0,   "step": 1.0,  "desc": "-1=left asymmetric, 0=symmetric, 1=right asymmetric"},
}

NSP_MAP = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
NSP_COLORS = {"Normal": "#28a745", "Suspect": "#ffc107", "Pathologic": "#dc3545"}
NSP_CSS = {"Normal": "result-normal", "Suspect": "result-suspect", "Pathologic": "result-pathologic"}
NSP_EMOJI = {"Normal": "✅", "Suspect": "⚠️", "Pathologic": "🚨"}
NSP_DESC = {
    "Normal": "The fetal heart rate pattern appears normal. No immediate intervention required.",
    "Suspect": "The pattern shows some concerning features. Further monitoring is recommended.",
    "Pathologic": "The pattern indicates a potentially dangerous state. Immediate medical attention is advised.",
}


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model and scaler from disk."""
    model_path = "rf_best_model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path):
        return None, None, f"Model file '{model_path}' not found. Please run Section 3.7 in model.ipynb first."
    if not os.path.exists(scaler_path):
        return None, None, f"Scaler file '{scaler_path}' not found. Please run Section 3.7 in model.ipynb first."

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, None


@st.cache_data
def load_classification_report():
    """Load classification report from JSON."""
    path = "classification_report.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    """Load the CTG dataset for reference statistics."""
    path = "CTG_data.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    cols_drop = ["FileName", "Date", "SegFile", "b", "e", "LBE",
                 "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP", "CLASS", "DR"]
    df.drop(columns=cols_drop, inplace=True)
    df = df.dropna(subset=["NSP"])
    df["NSP"] = df["NSP"].astype(int)
    return df


# ──────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CTG Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>Cardiotocography Fetal State<br>Classification System</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# PAGE 1: PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.markdown('<p class="main-header">Fetal State Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enter cardiotocographic measurements to predict the fetal state.</p>',
        unsafe_allow_html=True,
    )

    model, scaler, error = load_model()

    if error:
        st.error(error)
        st.info("Please run the model serialization cell in model.ipynb to generate the required .pkl files.")
        st.stop()

    # --- Input Form ---
    st.markdown("### 📋 Input Features")

    # Organize sliders in 3 columns
    feature_names = list(FEATURE_CONFIG.keys())
    input_values = {}

    col_groups = [feature_names[i:i + 7] for i in range(0, len(feature_names), 7)]

    cols = st.columns(3)
    for col_idx, group in enumerate(col_groups):
        with cols[col_idx]:
            for feat in group:
                cfg = FEATURE_CONFIG[feat]
                input_values[feat] = st.slider(
                    cfg["label"],
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"],
                    step=cfg["step"],
                    help=cfg["desc"],
                )

    st.markdown("---")

    # --- Predict Button ---
    if st.button("🔮 Predict Fetal State", use_container_width=True, type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_values])

        # Ensure column order matches scaler's expected features
        expected_features = list(scaler.feature_names_in_)
        input_df = input_df[expected_features]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        pred_label = NSP_MAP[prediction]

        # --- Display Result ---
        st.markdown("### Prediction Result")

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            css_class = NSP_CSS[pred_label]
            emoji = NSP_EMOJI[pred_label]
            desc = NSP_DESC[pred_label]
            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div style="font-size:3rem;">{emoji}</div>
                    <div class="result-label">{pred_label}</div>
                    <div class="result-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            # Probability bar chart
            prob_df = pd.DataFrame({
                "Class": list(NSP_MAP.values()),
                "Probability": [round(p * 100, 2) for p in probabilities],
            })
            fig = go.Figure(go.Bar(
                x=prob_df["Class"],
                y=prob_df["Probability"],
                marker_color=[NSP_COLORS[c] for c in prob_df["Class"]],
                text=[f"{p:.1f}%" for p in prob_df["Probability"]],
                textposition="outside",
                textfont=dict(size=14, color="white"),
            ))
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 105],
                template="plotly_dark",
                height=350,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show input summary
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">Model Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Compare base and tuned model performance metrics.</p>',
        unsafe_allow_html=True,
    )

    report = load_classification_report()
    if report is None:
        st.error("classification_report.json not found.")
        st.stop()

    # --- Accuracy Comparison ---
    st.markdown("### 🏆 Model Accuracy Comparison")

    comparison_data = []
    for section_key, section_label in [("base_models", "Base"), ("tuned_models", "Tuned")]:
        if section_key in report:
            for model_name, metrics in report[section_key].items():
                comparison_data.append({
                    "Model": model_name,
                    "Type": section_label,
                    "Accuracy": round(metrics["accuracy"] * 100, 2),
                    "Weighted F1": round(metrics["weighted avg"]["f1-score"] * 100, 2),
                    "Macro F1": round(metrics["macro avg"]["f1-score"] * 100, 2),
                })

    comp_df = pd.DataFrame(comparison_data)

    fig = px.bar(
        comp_df,
        x="Model",
        y="Accuracy",
        color="Type",
        barmode="group",
        text="Accuracy",
        color_discrete_map={"Base": "#667eea", "Tuned": "#28a745"},
        title="Accuracy: Base vs Tuned Models",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 105],
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Detailed Metrics Tables ---
    st.markdown("### 📋 Detailed Classification Reports")

    for section_key, section_label in [("base_models", "Base Models"), ("tuned_models", "Tuned Models")]:
        if section_key not in report:
            continue

        st.markdown(f"#### {section_label}")
        tabs = st.tabs(list(report[section_key].keys()))

        for tab, (model_name, metrics) in zip(tabs, report[section_key].items()):
            with tab:
                # Per-class metrics
                class_data = []
                for cls in ["Normal", "Suspect", "Pathologic"]:
                    if cls in metrics:
                        class_data.append({
                            "Class": cls,
                            "Precision": f"{metrics[cls]['precision']:.4f}",
                            "Recall": f"{metrics[cls]['recall']:.4f}",
                            "F1-Score": f"{metrics[cls]['f1-score']:.4f}",
                            "Support": int(metrics[cls]["support"]),
                        })

                st.dataframe(
                    pd.DataFrame(class_data).set_index("Class"),
                    use_container_width=True,
                )

                # Overall metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                m2.metric("Weighted F1", f"{metrics['weighted avg']['f1-score']:.4f}")
                m3.metric("Macro F1", f"{metrics['macro avg']['f1-score']:.4f}")

    # --- F1 Score Comparison ---
    st.markdown("### 📊 F1-Score Comparison (Tuned Models)")

    if "tuned_models" in report:
        f1_data = []
        for model_name, metrics in report["tuned_models"].items():
            for cls in ["Normal", "Suspect", "Pathologic"]:
                if cls in metrics:
                    f1_data.append({
                        "Model": model_name,
                        "Class": cls,
                        "F1-Score": round(metrics[cls]["f1-score"] * 100, 2),
                    })

        f1_df = pd.DataFrame(f1_data)
        fig2 = px.bar(
            f1_df,
            x="Model",
            y="F1-Score",
            color="Class",
            barmode="group",
            text="F1-Score",
            color_discrete_map=NSP_COLORS,
            title="Per-Class F1-Score (Tuned Models)",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            yaxis_range=[0, 105],
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">About This Application</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Dataset: Cardiotocography (CTG)

    This application uses the **Cardiotocography dataset** from the UCI Machine Learning Repository.
    The dataset contains **2,126 fetal cardiotocograms (CTGs)** that were automatically processed
    and their respective diagnostic features measured.

    The CTGs were classified by **three expert obstetricians** into three classes:

    | Class | Label | Description |
    |---|---|---|
    | 1 | **Normal** | Healthy fetal state |
    | 2 | **Suspect** | Potentially concerning patterns |
    | 3 | **Pathologic** | Dangerous state requiring intervention |
    """)

    st.markdown("---")

    st.markdown("### Feature Descriptions")

    feat_table = []
    for feat, cfg in FEATURE_CONFIG.items():
        feat_table.append({
            "Feature": feat,
            "Description": cfg["desc"],
            "Min": cfg["min"],
            "Max": cfg["max"],
        })
    st.dataframe(pd.DataFrame(feat_table).set_index("Feature"), use_container_width=True)

    st.markdown("---")

    st.markdown("""
    ### Model Information

    | Property | Value |
    |---|---|
    | **Algorithm** | Random Forest Classifier (Tuned) |
    | **Optimization** | GridSearchCV with Stratified K-Fold CV |
    | **Serialization** | Joblib (.pkl format) |
    | **Preprocessing** | StandardScaler (fitted on training data) |

    ### Reference

    > D. Ayres de Campos et al. (2000). *SisPorto 2.0: A Program for Automated Analysis of
    > Cardiotocograms*. J Matern Fetal Med 5:311-318.
    >
    > Source: J. Bernardes, Faculty of Medicine, University of Porto, Portugal.
    """)

    # Show dataset stats if available
    df = load_dataset()
    if df is not None:
        st.markdown("---")
        st.markdown("### Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        col2.metric("Features", len(df.columns) - 1)
        col3.metric("Classes", df["NSP"].nunique())

        st.markdown("#### Class Distribution")
        dist = df["NSP"].value_counts().sort_index()
        dist.index = [NSP_MAP[i] for i in dist.index]
        fig = px.pie(
            values=dist.values,
            names=dist.index,
            color=dist.index,
            color_discrete_map=NSP_COLORS,
            title="Target Class Distribution",
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
