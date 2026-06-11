import sys
from src.config import DATA_PATH
from src.data_loader import load_data
from src.features import get_primary_text_column
from src.benchmark import run_benchmark, generate_plots, generate_html_report

def main():
    print("=" * 60)
    print("PHISHING EMAIL CLASSIFIER BENCHMARK & VISUALIZATION")
    print("=" * 60)
    
    try:
        # 1. Load data
        df = load_data(DATA_PATH)
        
        # 2. Get text column
        text_column = get_primary_text_column(df)
        
        # 3. Run benchmark (computes ML metrics, latency, throughput, model sizes)
        results = run_benchmark(df, text_column)
        
        # 4. Generate comparison plots (PNG)
        generate_plots(results)
        
        # 5. Generate interactive HTML dashboard
        generate_html_report(results)
        
        # 6. Print Command Line Summary
        print(f"\n{'='*85}\nBENCHMARK SUMMARY TABLE\n{'='*85}")
        print(f"{'Model Classifier':<16} | {'Accuracy':<8} | {'F1-Score':<8} | {'Latency':<10} | {'Throughput':<12} | {'Size':<8} | {'CO2 Footprint':<13}")
        print("-" * 90)
        for mt, res in results.items():
            name = mt.upper()
            acc = res["metrics"]["Accuracy"]
            f1 = res["metrics"]["F1"]
            lat = f"{res['latency_ms']:.2f} ms"
            tput = f"{res['throughput_emails_per_sec']:.1f}/s"
            size = f"{res['model_size_mb']:.2f} MB"
            co2 = f"{res.get('emissions_mg', 0.0):.4f} mg"
            print(f"{name:<16} | {acc:<8.4f} | {f1:<8.4f} | {lat:<10} | {tput:<12} | {size:<8} | {co2:<13}")
        print("=" * 85)
        
        print("\nAll visualizer artifacts generated successfully:")
        print(" -> Static chart  : performance_comparison.png")
        print(" -> HTML Dashboard: benchmark_report.html\n")
        
    except FileNotFoundError as fnfe:
        print(f"\n[ERROR] Missing models: {str(fnfe)}", file=sys.stderr)
        print("Please train your models first by running: python main.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Benchmarking failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
