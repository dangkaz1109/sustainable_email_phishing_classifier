# Phishing Email Classifier

A professional and modular machine learning project designed to classify emails as Phishing or Legitimate. This repository trains and compares three classifiers:
1. **LR Word**: Logistic Regression on word-level TF-IDF (1-2 n-grams)
2. **SVM Char**: LinearSVC on character-level TF-IDF (3-5 n-grams)
3. **HistGradientBoosting**: HistGradientBoostingClassifier utilizing engineered numeric features

---

## Directory Structure

```
sustainable_email_phishing_classifier/
│
├── data/
│   └── ephishLLM.json            # Raw data
│
├── models/                       # Directory for serialized models & vectorizers
│   ├── lr_word_model.joblib
│   ├── vectorizer_word.joblib
│   ├── svm_char_model.joblib
│   ├── vectorizer_char.joblib
│   └── hgb_model.joblib
│
├── src/                          # Core source module
│   ├── __init__.py               # Marks directory as package
│   ├── config.py                 # Paths, seeds, and configurations
│   ├── data_loader.py            # Data loading and stratified train/test split
│   ├── features.py               # Preprocessing and numeric feature engineering
│   ├── train.py                  # Training pipeline and metric evaluations
│   ├── predict.py                # Inference service classes
│   └── benchmark.py              # Computational latency & ML benchmarking logic
│
├── main.py                       # Root pipeline runner (trains & evaluates all models)
├── predict.py                    # Root prediction CLI convenience wrapper
├── visualize.py                  # Root visualizer runner (compiles benchmarking plots & HTML dashboard)
├── requirements.txt              # Standard package requirements
└── README.md                     # Documentation
```

---

## Installation & Setup

1. Make sure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Training the Pipeline
To load the dataset, perform stratified train-test splits, train all three models, print comparison metrics, and serialize the trained classifiers to `models/`, run:
```bash
python main.py
```

### 2. Evaluating & Visualizing Performance
To benchmark both **machine learning metrics** (Accuracy, F1-Score, ROC AUC) and **inference computing performance** (latency in milliseconds per email, throughput in emails/second, and model file footprint on disk) for all three models:
```bash
python visualize.py
```
This command outputs a terminal summary table and generates two visual artifacts:
- **`performance_comparison.png`**: A 2x2 grid of matplotlib charts showing the trade-offs.
- **`benchmark_report.html`**: A gorgeous, dark-themed interactive glassmorphic dashboard containing charts (powered by Chart.js) and recommendations.

### 3. Running Inference on Single Emails
To test individual emails using any of the trained classifiers, run `predict.py` with custom text:

```bash
# Using the default character-level SVM model (SVM Char)
python predict.py --subject "Urgent security update required" --body "Please verify your credentials at http://fake-login.com"

# Using the Logistic Regression model (LR Word)
python predict.py --model lr_word --subject "Project meeting" --body "Hey Team, let's meet tomorrow at 10 AM to discuss the timeline."

# Using the HistGradientBoosting model (engineered numeric features)
python predict.py --model hgb --body "Verify your account immediately: http://phish-link.org/reset"
```
