# Cardiotocography-CTG-Fetal-State-Classifier
Cardiotocography (CTG) is a medical technique used to monitor fetal heart rate (FHR) and uterine contractions during pregnancy. This project builds and evaluates multiple ML classification models on the CTG dataset to assist healthcare professionals in identifying potentially dangerous fetal states early.


### Key Highlights

- **Dataset**: 2,126 CTG recordings with 21 diagnostic features
- **Task**: Multi-class classification (Normal / Suspect / Pathologic)
- **Best Model**: Random Forest — **92.9% accuracy**
- **Deployment**: Interactive Streamlit web app with real-time predictions

---

## 📂 Project Structure

```
ML_Portfolio/
├── model.ipynb                 # Full ML pipeline (EDA, training, evaluation, tuning)
├── app.py                      # Streamlit web application
├── CTG_data.csv                # Dataset (CSV format)
├── requirements.txt            # Python dependencies with versions
└── README.md
```

---

## 🔬 ML Pipeline (model.ipynb)

The notebook covers the complete machine learning workflow:

| Step | Description |
|------|-------------|
| **3.1** | Load data and import libraries |
| **3.2** | Data preprocessing, EDA (outlier handling, VIF multicollinearity analysis), and stratified train-test split |
| **3.3** | Train 3 base models: Logistic Regression, Random Forest, KNN |
| **3.4** | Evaluate with classification reports, confusion matrices, K-Fold & Stratified K-Fold cross-validation |
| **3.5** | Test on new instances (inference demo) |
| **3.6** | Hyperparameter tuning with GridSearchCV |
| **3.7** | Model serialization using Joblib |

---

## 📊 Model Performance

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Logistic Regression | 87.7% | 87.3% |
| **Random Forest** | **92.9%** | **92.5%** |
| KNN | 91.9% | 91.7% |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML_Portfolio.git
   cd ML_Portfolio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook** (optional — to retrain models)
   
   Open `model.ipynb` in Jupyter and run all cells to generate `rf_best_model.pkl` and `scaler.pkl`.

4. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Streamlit App Features

The web app has 3 pages:

- **🔮 Predict** — Input 21 CTG features via sliders and get real-time fetal state predictions with probability scores
- **📊 Model Performance** — Compare base vs tuned model metrics with interactive charts
- **ℹ️ About** — Dataset description, feature explanations, and class distribution

---

## 📁 Dataset

**Source**: [UCI Machine Learning Repository — Cardiotocography](https://archive.ics.uci.edu/ml/datasets/cardiotocography)

| Property | Value |
|----------|-------|
| Samples | 2,126 |
| Features | 21 (numeric) |
| Target | NSP (1=Normal, 2=Suspect, 3=Pathologic) |
| Class Balance | Imbalanced (78% Normal, 14% Suspect, 8% Pathologic) |

### Features Used

| Feature | Description |
|---------|-------------|
| LB | Fetal heart rate baseline (bpm) |
| AC | Accelerations per second |
| FM | Fetal movements per second |
| UC | Uterine contractions per second |
| ASTV | % time with abnormal short term variability |
| MSTV | Mean short term variability |
| ALTV | % time with abnormal long term variability |
| MLTV | Mean long term variability |
| DL | Light decelerations per second |
| DS | Severe decelerations per second |
| DP | Prolonged decelerations per second |
| Width | FHR histogram width |
| Min | FHR histogram minimum |
| Max | FHR histogram maximum |
| Nmax | Number of histogram peaks |
| Nzeros | Number of histogram zeros |
| Mode | Histogram mode |
| Mean | Histogram mean |
| Median | Histogram median |
| Variance | Histogram variance |
| Tendency | Histogram tendency (-1=left, 0=symmetric, 1=right) |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Statsmodels** — VIF multicollinearity analysis
- **Streamlit** — Web app deployment
- **Plotly** — Interactive charts
- **Joblib** — Model serialization

---

## 📖 Reference

> D. Ayres de Campos et al. (2000). *SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms*. J Matern Fetal Med 5:311-318.
>
> Source: J. Bernardes, Faculty of Medicine, University of Porto, Portugal.

---

## 📄 License

This project is for educational purposes as part of a Machine Learning Portfolio.
