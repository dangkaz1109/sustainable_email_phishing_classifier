import os
import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import MODELS_DIR, LR_MAX_ITER, LR_RANDOM_STATE, SVM_RANDOM_STATE, HGB_RANDOM_STATE, TRANSFORMER_BATCH_SIZE, TRANSFORMER_EPOCHS, TRANSFORMER_LR

# Ensure model directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def calculate_evaluation_metrics(y_true, y_pred, y_prob):
    """
    Computes performance metrics: Accuracy, Precision, Recall, F1, and ROC AUC.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["ROC_AUC"] = 0.5
    return metrics

def train_lr_word(train_df, test_df, text_column):
    """
    Trains Logistic Regression on word-level TF-IDF.
    """
    print("\nTraining [LR Word] model...")
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df[text_column])
    X_test = vectorizer.transform(test_df[text_column])
    
    model = LogisticRegression(max_iter=LR_MAX_ITER, random_state=LR_RANDOM_STATE)
    model.fit(X_train, train_df["label"])
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_evaluation_metrics(test_df["label"], preds, probs)
    
    # Save artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "lr_word_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer_word.joblib"))
    print("Saved LR Word model and vectorizer to models/")
    
    return metrics

def train_svm_char(train_df, test_df, text_column):
    """
    Trains LinearSVC on character-level TF-IDF.
    """
    print("Training [SVM Char] model...")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(train_df[text_column])
    X_test = vectorizer.transform(test_df[text_column])
    
    model = LinearSVC(random_state=SVM_RANDOM_STATE, dual=False)
    model.fit(X_train, train_df["label"])
    
    preds = model.predict(X_test)
    decision_scores = model.decision_function(X_test)
    probs = 1 / (1 + np.exp(-decision_scores))  # Calibrate decision scores to probabilities
    
    metrics = calculate_evaluation_metrics(test_df["label"], preds, probs)
    
    # Save artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "svm_char_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer_char.joblib"))
    print("Saved SVM Char model and vectorizer to models/")
    
    return metrics

def train_hgb_numeric(train_df, test_df, numeric_features):
    """
    Trains HistGradientBoosting on numeric features.
    """
    print("Training [HistGradientBoosting] model on numeric features...")
    model = HistGradientBoostingClassifier(random_state=HGB_RANDOM_STATE)
    model.fit(train_df[numeric_features], train_df["label"])
    
    preds = model.predict(test_df[numeric_features])
    probs = model.predict_proba(test_df[numeric_features])[:, 1]
    
    metrics = calculate_evaluation_metrics(test_df["label"], preds, probs)
    
    # Save artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "hgb_model.joblib"))
    print("Saved HistGradientBoosting model to models/")
    
    return metrics

class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_transformer(train_df, test_df, text_column, model_name):
    print(f"\nTraining [{model_name}] model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    train_texts = train_df[text_column].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df[text_column].tolist()
    test_labels = test_df["label"].tolist()
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    
    train_dataset = PhishingDataset(train_encodings, train_labels)
    test_dataset = PhishingDataset(test_encodings, test_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=TRANSFORMER_EPOCHS,
        per_device_train_batch_size=TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=TRANSFORMER_BATCH_SIZE,
        learning_rate=TRANSFORMER_LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return calculate_evaluation_metrics(labels, predictions, probs)
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    
    safe_name = model_name.replace("/", "_")
    model_save_path = os.path.join(MODELS_DIR, f"{safe_name}_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Saved {model_name} model to {model_save_path}")
    
    return eval_results
