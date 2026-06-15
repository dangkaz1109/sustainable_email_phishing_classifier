import sys
import argparse
from src.predict import EmailPredictor

def main():
    parser = argparse.ArgumentParser(description="Predict if an email is phishing or legitimate.")
    parser.add_argument(
        "--subject", 
        type=str, 
        default="", 
        help="The subject of the email."
    )
    parser.add_argument(
        "--body", 
        type=str, 
        required=True, 
        help="The body content of the email."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="svm_char", 
        choices=["lr_word", "svm_char", "hgb", "mobilebert", "distilbert", "electra"],
        help="The classification model to use (default: svm_char)."
    )
    
    args = parser.parse_args()
    
    try:
        predictor = EmailPredictor()
        
        # Check if requested model is trained/available
        if not predictor.is_model_available(args.model):
            print(
                f"\n[ERROR] The model '{args.model}' is not trained or found in 'models/'.\n"
                "Please train the models first by running: python main.py", 
                file=sys.stderr
            )
            sys.exit(1)
            
        # Run prediction
        result = predictor.predict(args.subject, args.body, args.model)
        
        print("\n" + "=" * 50)
        print("PHISHING EMAIL CLASSIFIER INFERENCE")
        print("=" * 50)
        print(f"Model used  : {result['model_used'].upper()}")
        print(f"Subject     : {args.subject[:60] + ('...' if len(args.subject) > 60 else '')}")
        print(f"Body snippet: {args.body[:60] + ('...' if len(args.body) > 60 else '')}")
        print("-" * 50)
        
        status = "⚠️  PHISHING" if result["is_phishing"] else "✅ LEGITIMATE"
        color_code = "\033[91m" if result["is_phishing"] else "\033[92m"
        reset_code = "\033[0m"
        
        # Output with colored label
        print(f"Result      : {color_code}{status}{reset_code}")
        print(f"Probability : {result['probability']:.4f}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
