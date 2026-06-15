import sys
from src.config import DATA_PATH, NUMERIC_FEATURES
from src.data_loader import load_data, split_data
from src.features import get_primary_text_column, engineer_numeric_features
from src.train import train_lr_word, train_svm_char, train_hgb_numeric, train_transformer

def main():
    try:
        # 1. Load the dataset
        df = load_data(DATA_PATH)
        
        # 2. Get primary text column and fill NaNs
        text_column = get_primary_text_column(df)
        print(f"Using '{text_column}' as the primary text column.")
        
        # 3. Feature Engineering
        print("Engineering numeric features from text...")
        df = engineer_numeric_features(df, text_column)
        
        # 4. Stratified Split (80% train, 20% test)
        train_df, test_df = split_data(df)
        
        # 5. Model Training & Evaluation
        metrics_lr_word = train_lr_word(train_df, test_df, text_column)
        metrics_svm_char = train_svm_char(train_df, test_df, text_column)
        metrics_hgb = train_hgb_numeric(train_df, test_df, NUMERIC_FEATURES)
        metrics_mobilebert = train_transformer(train_df, test_df, text_column, "google/mobilebert-uncased")
        metrics_distilbert = train_transformer(train_df, test_df, text_column, "distilbert-base-uncased")
        metrics_electra = train_transformer(train_df, test_df, text_column, "google/electra-base-discriminator")
        
        # 6. Output Comparison
        print(f"\n{'='*60}\nEVALUATION RESULTS\n{'='*60}")
        
        print(f"[{text_column.upper()} - LR Word]")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics_lr_word.items()]))
        print("-" * 60)
        
        print(f"[{text_column.upper()} - SVM Char]")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics_svm_char.items()]))
        print("-" * 60)
        
        print("[NUMERIC - HistGradientBoosting]")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics_hgb.items()]))
        print("-" * 60)

        print("[BODY - MobileBERT]")
        print(" | ".join([f"{k.replace('eval_', '')}: {v:.4f}" for k, v in metrics_mobilebert.items() if k.replace('eval_', '') in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]))
        print("-" * 60)

        print("[BODY - DistilBERT]")
        print(" | ".join([f"{k.replace('eval_', '')}: {v:.4f}" for k, v in metrics_distilbert.items() if k.replace('eval_', '') in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]))
        print("-" * 60)

        print("[BODY - Electra]")
        print(" | ".join([f"{k.replace('eval_', '')}: {v:.4f}" for k, v in metrics_electra.items() if k.replace('eval_', '') in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]))
        print("-" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

