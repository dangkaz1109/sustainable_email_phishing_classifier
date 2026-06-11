import os
import re
import joblib
import numpy as np
import pandas as pd
from src.config import MODELS_DIR, NUMERIC_FEATURES

class EmailPredictor:
    """
    Inference class to load trained classifiers/vectorizers and predict on new emails.
    """
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self._load_all_models()
        
    def _load_all_models(self):
        # LR Word
        lr_path = os.path.join(MODELS_DIR, "lr_word_model.joblib")
        vec_word_path = os.path.join(MODELS_DIR, "vectorizer_word.joblib")
        if os.path.exists(lr_path) and os.path.exists(vec_word_path):
            self.models["lr_word"] = joblib.load(lr_path)
            self.vectorizers["lr_word"] = joblib.load(vec_word_path)
            
        # SVM Char
        svm_path = os.path.join(MODELS_DIR, "svm_char_model.joblib")
        vec_char_path = os.path.join(MODELS_DIR, "vectorizer_char.joblib")
        if os.path.exists(svm_path) and os.path.exists(vec_char_path):
            self.models["svm_char"] = joblib.load(svm_path)
            self.vectorizers["svm_char"] = joblib.load(vec_char_path)
            
        # HGB Numeric
        hgb_path = os.path.join(MODELS_DIR, "hgb_model.joblib")
        if os.path.exists(hgb_path):
            self.models["hgb"] = joblib.load(hgb_path)

    def is_model_available(self, model_type):
        return model_type in self.models

    def predict(self, subject, body, model_type="svm_char"):
        """
        Predicts if a single email is phishing.
        """
        model_type = model_type.lower()
        if not self.is_model_available(model_type):
            raise FileNotFoundError(
                f"Model type '{model_type}' is not loaded. Please train the model first by running main.py."
            )
            
        # Standardize strings
        subject = subject or ""
        body = body or ""
        
        if model_type == "hgb":
            # Extract engineered features for HGB
            feat_dict = {
                "subject_len": len(subject),
                "body_len": len(body),
                "subject_word_count": len(subject.split()),
                "body_word_count": len(body.split()),
                "body_digit_count": sum(c.isdigit() for c in body),
                "body_url_count": len(re.findall(r"https?://\S+|www\.\S+", body)),
                "body_special_char_count": len(re.findall(r"[!@#$%^&*(),.?\":{}|<>]", body))
            }
            X = pd.DataFrame([feat_dict])[NUMERIC_FEATURES]
            model = self.models["hgb"]
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]
            
        elif model_type in ["lr_word", "svm_char"]:
            vectorizer = self.vectorizers[model_type]
            model = self.models[model_type]
            
            X = vectorizer.transform([body])
            pred = model.predict(X)[0]
            
            if model_type == "lr_word":
                prob = model.predict_proba(X)[0][1]
            else:  # svm_char
                decision_score = model.decision_function(X)[0]
                prob = 1 / (1 + np.exp(-decision_score))
                
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return {
            "is_phishing": bool(pred == 1),
            "probability": float(prob),
            "label": int(pred),
            "model_used": model_type
        }
