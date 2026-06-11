import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import MODELS_DIR, NUMERIC_FEATURES
from src.predict import EmailPredictor
from src.train import calculate_evaluation_metrics
from src.data_loader import split_data
from src.features import engineer_numeric_features
import logging
from codecarbon import OfflineEmissionsTracker

# Silence codecarbon logger
logging.getLogger("codecarbon").disabled = True

def get_file_size_mb(*paths):
    """
    Returns the combined file size of target paths in Megabytes.
    """
    total_bytes = 0
    for p in paths:
        if os.path.exists(p):
            total_bytes += os.path.getsize(p)
    return total_bytes / (1024 * 1024)

def run_benchmark(df, text_column):
    """
    Evaluates both ML metrics and compute metrics (latency, throughput, size) for all models.
    """
    # 1. Check models presence
    predictor = EmailPredictor()
    model_types = ["lr_word", "svm_char", "hgb"]
    for mt in model_types:
        if not predictor.is_model_available(mt):
            raise FileNotFoundError(f"Model '{mt}' is not trained. Please run main.py first.")
            
    # 2. Split data
    _, test_df = split_data(df)
    test_df_engineered = engineer_numeric_features(test_df, text_column)
    
    y_true = test_df_engineered["label"].values
    
    results = {}
    
    # We sample 200 rows for single-instance latency benchmarking
    sample_size = min(200, len(test_df_engineered))
    sample_df = test_df_engineered.sample(n=sample_size, random_state=42)
    
    print("\nStarting performance benchmarking...")
    
    for mt in model_types:
        print(f"Benchmarking '{mt.upper()}'...")
        
        # Start CodeCarbon tracking for the benchmarking run of this model
        tracker = OfflineEmissionsTracker(country_iso_code="USA", save_to_file=False, log_level="error")
        tracker.start()
        
        try:
            # A. Measure ML metrics on full test set
            # For batch evaluation, we do it model by model
            preds = []
            probs = []
            
            model = predictor.models[mt]
            if mt == "hgb":
                X = test_df_engineered[NUMERIC_FEATURES]
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]
            elif mt in ["lr_word", "svm_char"]:
                vectorizer = predictor.vectorizers[mt]
                X = vectorizer.transform(test_df_engineered[text_column])
                preds = model.predict(X)
                if mt == "lr_word":
                    probs = model.predict_proba(X)[:, 1]
                else:  # svm_char
                    decision_scores = model.decision_function(X)
                    probs = 1 / (1 + np.exp(-decision_scores))
                    
            metrics = calculate_evaluation_metrics(y_true, preds, probs)
            
            # B. Measure single-instance latency & throughput
            latencies = []
            for _, row in sample_df.iterrows():
                subject = row.get("subject", "")
                body = row.get(text_column, "")
                
                start_time = time.perf_counter()
                predictor.predict(subject, body, model_type=mt)
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000) # Convert to ms
                
            avg_latency = np.mean(latencies)
            throughput = 1000.0 / avg_latency if avg_latency > 0 else 0
        finally:
            emissions_kg = tracker.stop()
            
        if emissions_kg is None:
            emissions_kg = 0.0
        emissions_mg = emissions_kg * 1.0e6
        
        # C. Get model size on disk
        if mt == "lr_word":
            size_mb = get_file_size_mb(
                os.path.join(MODELS_DIR, "lr_word_model.joblib"),
                os.path.join(MODELS_DIR, "vectorizer_word.joblib")
            )
        elif mt == "svm_char":
            size_mb = get_file_size_mb(
                os.path.join(MODELS_DIR, "svm_char_model.joblib"),
                os.path.join(MODELS_DIR, "vectorizer_char.joblib")
            )
        else: # hgb
            size_mb = get_file_size_mb(
                os.path.join(MODELS_DIR, "hgb_model.joblib")
            )
            
        results[mt] = {
            "metrics": metrics,
            "latency_ms": float(avg_latency),
            "throughput_emails_per_sec": float(throughput),
            "model_size_mb": float(size_mb),
            "emissions_mg": float(emissions_mg)
        }
        
    return results

def generate_plots(results, output_path="performance_comparison.png"):
    """
    Generates a 3x2 grid of matplotlib bar charts comparing the models including energy/throughput metrics.
    Each metric uses a consistent single color across models.
    """
    models = [m.upper() for m in results.keys()]
    
    # Extract values
    f1_scores = [results[m.lower()]["metrics"]["F1"] for m in models]
    roc_auc = [results[m.lower()]["metrics"]["ROC_AUC"] for m in models]
    latencies = [results[m.lower()]["latency_ms"] for m in models]
    throughputs = [results[m.lower()]["throughput_emails_per_sec"] for m in models]
    sizes = [results[m.lower()]["model_size_mb"] for m in models]
    emissions = [results[m.lower()]["emissions_mg"] for m in models]
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Phishing Classifier: Performance, Compute & Sustainability Benchmark", fontsize=16, fontweight="bold", y=0.97)
    
    # 1. F1-Score (Indigo)
    axs[0, 0].bar(models, f1_scores, color="#4f46e5", edgecolor="black", width=0.5)
    axs[0, 0].set_title("F1-Score (Higher is Better)", fontweight="bold")
    axs[0, 0].set_ylim(0, 1.05)
    for i, v in enumerate(f1_scores):
        axs[0, 0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
        
    # 2. ROC AUC (Purple)
    axs[0, 1].bar(models, roc_auc, color="#8b5cf6", edgecolor="black", width=0.5)
    axs[0, 1].set_title("ROC AUC (Higher is Better)", fontweight="bold")
    axs[0, 1].set_ylim(0, 1.05)
    for i, v in enumerate(roc_auc):
        axs[0, 1].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
        
    # 3. Latency (Red)
    axs[1, 0].bar(models, latencies, color="#ef4444", edgecolor="black", width=0.5)
    axs[1, 0].set_title("Single Inference Latency (ms) (Lower is Better)", fontweight="bold")
    axs[1, 0].set_ylabel("Milliseconds (ms)")
    max_lat = max(latencies)
    axs[1, 0].set_ylim(0, max_lat * 1.25)
    for i, v in enumerate(latencies):
        axs[1, 0].text(i, v + (max_lat * 0.02), f"{v:.2f} ms", ha="center", fontweight="bold")
        
    # 4. Throughput (Amber)
    axs[1, 1].bar(models, throughputs, color="#f59e0b", edgecolor="black", width=0.5)
    axs[1, 1].set_title("Throughput (emails/s) (Higher is Better)", fontweight="bold")
    axs[1, 1].set_ylabel("Emails per Second")
    max_tput = max(throughputs)
    axs[1, 1].set_ylim(0, max_tput * 1.25)
    for i, v in enumerate(throughputs):
        axs[1, 1].text(i, v + (max_tput * 0.02), f"{v:.1f}/s", ha="center", fontweight="bold")
        
    # 5. Model Size (Sky Blue)
    axs[2, 0].bar(models, sizes, color="#0ea5e9", edgecolor="black", width=0.5)
    axs[2, 0].set_title("Model Footprint on Disk (MB) (Lower is Better)", fontweight="bold")
    axs[2, 0].set_ylabel("Megabytes (MB)")
    max_size = max(sizes)
    axs[2, 0].set_ylim(0, max_size * 1.25)
    for i, v in enumerate(sizes):
        axs[2, 0].text(i, v + (max_size * 0.02), f"{v:.2f} MB", ha="center", fontweight="bold")
        
    # 6. Carbon Footprint (Green - Required)
    axs[2, 1].bar(models, emissions, color="#10b981", edgecolor="black", width=0.5)
    axs[2, 1].set_title("Carbon Footprint (mg CO2eq) (Lower is Better)", fontweight="bold")
    axs[2, 1].set_ylabel("Milligrams of CO2eq (mg)")
    max_em = max(emissions)
    axs[2, 1].set_ylim(0, max_em * 1.25 if max_em > 0 else 1.0)
    for i, v in enumerate(emissions):
        axs[2, 1].text(i, v + (max_em * 0.02 if max_em > 0 else 0.02), f"{v:.4f} mg", ha="center", fontweight="bold")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved benchmark charts grid to '{output_path}'")

def generate_html_report(results, output_path="benchmark_report.html"):
    """
    Compiles results into a beautiful glassmorphic dark-mode HTML dashboard using Chart.js.
    """
    json_data = json.dumps(results, indent=2)
    
    # Find model with lowest carbon footprint
    lowest_co2_model = min(results.keys(), key=lambda k: results[k].get("emissions_mg", 0.0))
    lowest_co2_val = results[lowest_co2_model].get("emissions_mg", 0.0)
    lowest_co2_name = {"lr_word": "LR Word", "svm_char": "SVM Char", "hgb": "HGB Numeric"}[lowest_co2_model]
    lowest_co2_badge = {"lr_word": "badge-indigo", "svm_char": "badge-sky", "hgb": "badge-emerald"}[lowest_co2_model]
    
    lr_co2 = results["lr_word"].get("emissions_mg", 0.0)
    svm_co2 = results["svm_char"].get("emissions_mg", 0.0)
    hgb_co2 = results["hgb"].get("emissions_mg", 0.0)
    co2_values = [lr_co2, svm_co2, hgb_co2]
    min_co2 = min(co2_values)
    
    lr_co2_hl = ' class="highlight"' if lr_co2 == min_co2 else ''
    svm_co2_hl = ' class="highlight"' if svm_co2 == min_co2 else ''
    hgb_co2_hl = ' class="highlight"' if hgb_co2 == min_co2 else ''
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Phishing Classifier Benchmark</title>
    <!-- Google Fonts: Inter -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.04);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #4f46e5;
            --primary-glow: rgba(79, 70, 229, 0.4);
            --secondary: #0ea5e9;
            --accent: #10b981;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 2.5rem 1.5rem;
            min-height: 100vh;
            background-image: radial-gradient(circle at 10% 20%, rgba(79, 70, 229, 0.15) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.15) 0%, transparent 40%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        header h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 400;
        }}

        /* Grid System */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .card-header {{
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-value {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #fff;
            margin-bottom: 0.5rem;
        }}

        .card-desc {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        @media (max-width: 600px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-card {{
            min-height: 380px;
            display: flex;
            flex-direction: column;
        }}

        .chart-card h2 {{
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: #fff;
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
        }}

        .chart-container {{
            position: relative;
            flex-grow: 1;
            width: 100%;
            height: 100%;
        }}

        /* Table Card */
        .table-card {{
            overflow-x: auto;
            margin-bottom: 3rem;
        }}

        .table-card h2 {{
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: #fff;
            border-left: 4px solid var(--secondary);
            padding-left: 0.75rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th, td {{
            padding: 1rem;
            border-bottom: 1px solid var(--card-border);
        }}

        th {{
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        td {{
            color: #fff;
            font-size: 0.95rem;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        .highlight {{
            font-weight: 700;
            color: var(--accent);
        }}

        .badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .badge-indigo {{
            background: rgba(79, 70, 229, 0.2);
            color: #a5b4fc;
            border: 1px solid rgba(79, 70, 229, 0.4);
        }}
        .badge-sky {{
            background: rgba(14, 165, 233, 0.2);
            color: #7dd3fc;
            border: 1px solid rgba(14, 165, 233, 0.4);
        }}
        .badge-emerald {{
            background: rgba(16, 185, 129, 0.2);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        .recommendation-card {{
            border-left: 4px solid var(--accent);
            background: rgba(16, 185, 129, 0.05);
        }}

        .recommendation-card h2 {{
            color: var(--accent);
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }}
        
        .recommendation-card p {{
            line-height: 1.6;
            color: #e5e7eb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Email Phishing Classifier</h1>
            <p>Sustainable Performance & Compute Efficiency Dashboard</p>
        </header>

        <!-- Summary Cards -->
        <div class="metrics-grid">
            <div class="card">
                <div class="card-header">
                    <span>Top F1-Score</span>
                    <span class="badge badge-sky">SVM Char</span>
                </div>
                <div class="card-value">{results["svm_char"]["metrics"]["F1"]:.4f}</div>
                <div class="card-desc">Highest overall classification accuracy on test dataset split.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span>Fastest Latency</span>
                    <span class="badge badge-emerald">HGB Numeric</span>
                </div>
                <div class="card-value">{results["hgb"]["latency_ms"]:.2f} ms</div>
                <div class="card-desc">Single email classification time in milliseconds.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span>Highest Throughput</span>
                    <span class="badge badge-emerald">HGB Numeric</span>
                </div>
                <div class="card-value">{int(results["hgb"]["throughput_emails_per_sec"])} /s</div>
                <div class="card-desc">Emails processed per second.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span>Lowest CO2 Footprint</span>
                    <span class="badge {lowest_co2_badge}">{lowest_co2_name}</span>
                </div>
                <div class="card-value">{lowest_co2_val:.4f} mg</div>
                <div class="card-desc">CO2 equivalent emissions per benchmark run.</div>
            </div>
        </div>

        <!-- Interactive Charts -->
        <div class="chart-grid">
            <div class="card chart-card">
                <h2>Model Accuracy & F1-Scores</h2>
                <div class="chart-container">
                    <canvas id="accuracyChart"></canvas>
                </div>
            </div>
            <div class="card chart-card">
                <h2>Inference Latency & Disk Size Trade-off</h2>
                <div class="chart-container">
                    <canvas id="latencySizeChart"></canvas>
                </div>
            </div>
            <div class="card chart-card">
                <h2>Inference Throughput (emails/s)</h2>
                <div class="chart-container">
                    <canvas id="throughputChart"></canvas>
                </div>
            </div>
            <div class="card chart-card">
                <h2>Carbon Footprint (mg CO2eq)</h2>
                <div class="chart-container">
                    <canvas id="carbonChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Detailed Table -->
        <div class="card table-card">
            <h2>Detailed Evaluation & Benchmark Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model Classifier</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>ROC AUC</th>
                        <th>Latency (ms)</th>
                        <th>Throughput (/s)</th>
                        <th>Size on Disk (MB)</th>
                        <th>CO2 (mg)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="badge badge-indigo">LR Word</span></td>
                        <td>{results["lr_word"]["metrics"]["Accuracy"]:.4f}</td>
                        <td>{results["lr_word"]["metrics"]["Precision"]:.4f}</td>
                        <td>{results["lr_word"]["metrics"]["Recall"]:.4f}</td>
                        <td>{results["lr_word"]["metrics"]["F1"]:.4f}</td>
                        <td>{results["lr_word"]["metrics"]["ROC_AUC"]:.4f}</td>
                        <td>{results["lr_word"]["latency_ms"]:.2f} ms</td>
                        <td>{results["lr_word"]["throughput_emails_per_sec"]:.1f}</td>
                        <td>{results["lr_word"]["model_size_mb"]:.2f} MB</td>
                        <td{lr_co2_hl}>{results["lr_word"].get("emissions_mg", 0.0):.4f} mg</td>
                    </tr>
                    <tr>
                        <td><span class="badge badge-sky">SVM Char</span></td>
                        <td class="highlight">{results["svm_char"]["metrics"]["Accuracy"]:.4f}</td>
                        <td>{results["svm_char"]["metrics"]["Precision"]:.4f}</td>
                        <td>{results["svm_char"]["metrics"]["Recall"]:.4f}</td>
                        <td class="highlight">{results["svm_char"]["metrics"]["F1"]:.4f}</td>
                        <td class="highlight">{results["svm_char"]["metrics"]["ROC_AUC"]:.4f}</td>
                        <td>{results["svm_char"]["latency_ms"]:.2f} ms</td>
                        <td>{results["svm_char"]["throughput_emails_per_sec"]:.1f}</td>
                        <td>{results["svm_char"]["model_size_mb"]:.2f} MB</td>
                        <td{svm_co2_hl}>{results["svm_char"].get("emissions_mg", 0.0):.4f} mg</td>
                    </tr>
                    <tr>
                        <td><span class="badge badge-emerald">HGB Numeric</span></td>
                        <td>{results["hgb"]["metrics"]["Accuracy"]:.4f}</td>
                        <td>{results["hgb"]["metrics"]["Precision"]:.4f}</td>
                        <td>{results["hgb"]["metrics"]["Recall"]:.4f}</td>
                        <td>{results["hgb"]["metrics"]["F1"]:.4f}</td>
                        <td>{results["hgb"]["metrics"]["ROC_AUC"]:.4f}</td>
                        <td class="highlight">{results["hgb"]["latency_ms"]:.2f} ms</td>
                        <td class="highlight">{results["hgb"]["throughput_emails_per_sec"]:.1f}</td>
                        <td class="highlight">{results["hgb"]["model_size_mb"]:.2f} MB</td>
                        <td{hgb_co2_hl}>{results["hgb"].get("emissions_mg", 0.0):.4f} mg</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Deployment Recommendation -->
        <div class="card recommendation-card">
            <h2>Deployment & Sustainability Recommendation</h2>
            <p>
                Based on the benchmark results, <strong>SVM Char (Character n-grams 3-5 LinearSVC)</strong> is highly recommended for applications prioritizing <strong>accuracy</strong>, reaching a top F1-Score of <strong>{results["svm_char"]["metrics"]["F1"]:.4f}</strong>. However, it requires a larger model disk footprint ({results["svm_char"]["model_size_mb"]:.2f} MB), has higher inference latency ({results["svm_char"]["latency_ms"]:.2f} ms), and incurs a carbon footprint of <strong>{results["svm_char"]["emissions_mg"]:.4f} mg CO2eq</strong> per benchmark evaluation.
            </p>
            <p style="margin-top: 0.5rem;">
                If you are running in a resource-constrained environment (edge/mobile devices or high-concurrency low-latency webhooks) or prioritize <strong>environmental sustainability (Green AI)</strong>, <strong>HistGradientBoosting</strong> is the best alternative. It achieves a blazing fast <strong>{results["hgb"]["latency_ms"]:.2f} ms</strong> single inference latency, <strong>{results["hgb"]["throughput_emails_per_sec"]:.1f} classifications per second</strong>, a minimal model footprint of <strong>{results["hgb"]["model_size_mb"]:.2f} MB</strong>, and a carbon footprint of only <strong>{results["hgb"]["emissions_mg"]:.4f} mg CO2eq</strong>.
            </p>
        </div>
    </div>

    <script>
        const data = {json_data};
        
        // 1. Accuracy Chart
        new Chart(document.getElementById('accuracyChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: ['LR Word', 'SVM Char', 'HGB Numeric'],
                datasets: [
                    {{
                        label: 'F1-Score',
                        data: [data.lr_word.metrics.F1, data.svm_char.metrics.F1, data.hgb.metrics.F1],
                        backgroundColor: 'rgba(79, 70, 229, 0.8)',
                        borderColor: '#4f46e5',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Accuracy',
                        data: [data.lr_word.metrics.Accuracy, data.svm_char.metrics.Accuracy, data.hgb.metrics.Accuracy],
                        backgroundColor: 'rgba(14, 165, 233, 0.8)',
                        borderColor: '#0ea5e9',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        grid: {{ color: 'rgba(255, 255, 255, 0.08)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});

        // 2. Latency & Size Chart
        new Chart(document.getElementById('latencySizeChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: ['LR Word', 'SVM Char', 'HGB Numeric'],
                datasets: [
                    {{
                        label: 'Latency (ms) [lower is better]',
                        data: [data.lr_word.latency_ms, data.svm_char.latency_ms, data.hgb.latency_ms],
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        borderColor: '#ef4444',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Model Disk Size (MB) [lower is better]',
                        data: [data.lr_word.model_size_mb, data.svm_char.model_size_mb, data.hgb.model_size_mb],
                        backgroundColor: 'rgba(168, 85, 247, 0.8)',
                        borderColor: '#a855f7',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        position: 'left',
                        title: {{ display: true, text: 'Latency (ms)', color: '#ef4444' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.08)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    y1: {{
                        beginAtZero: true,
                        position: 'right',
                        title: {{ display: true, text: 'Model Size (MB)', color: '#a855f7' }},
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});

        // 3. Throughput Chart
        new Chart(document.getElementById('throughputChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: ['LR Word', 'SVM Char', 'HGB Numeric'],
                datasets: [
                    {{
                        label: 'Throughput (emails/s) [higher is better]',
                        data: [data.lr_word.throughput_emails_per_sec, data.svm_char.throughput_emails_per_sec, data.hgb.throughput_emails_per_sec],
                        backgroundColor: 'rgba(245, 158, 11, 0.8)',
                        borderColor: '#f59e0b',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(255, 255, 255, 0.08)' }},
                        ticks: {{ color: '#9ca3af' }},
                        title: {{ display: true, text: 'Emails / Second', color: '#f59e0b' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});

        // 4. Carbon Footprint Chart
        new Chart(document.getElementById('carbonChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: ['LR Word', 'SVM Char', 'HGB Numeric'],
                datasets: [
                    {{
                        label: 'Carbon Footprint (mg CO2eq) [lower is better]',
                        data: [data.lr_word.emissions_mg, data.svm_char.emissions_mg, data.hgb.emissions_mg],
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderColor: '#10b981',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(255, 255, 255, 0.08)' }},
                        ticks: {{ color: '#9ca3af' }},
                        title: {{ display: true, text: 'Milligrams of CO2eq (mg)', color: '#10b981' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f3f4f6' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Saved interactive HTML dashboard to '{output_path}'")
