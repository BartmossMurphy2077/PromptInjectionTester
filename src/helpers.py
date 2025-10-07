import time
import pandas as pd
from pathlib import Path
from typing import List


def select_dataset(dataset_dir: Path) -> Path:
    """UI for listing and selecting a dataset"""
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in Datasets.")
        exit(1)

    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file.name}")

    while True:
        choice = input(f"Select a dataset (1-{len(csv_files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            return csv_files[int(choice) - 1]
        print("Invalid selection, try again.")


def collect_breaches(output_dir: Path) -> None:
    """Collect breaches from all output CSVs and aggregate them into a single breaches.csv file"""
    print("========================Collecting breaches========================")

    start_time = time.time()
    breach_records = []

    for csv_file in output_dir.glob("*.csv"):
        if csv_file.name in ["breaches.csv", "token_logs.csv"]:
            print(f"[INFO] Skipping {csv_file.name}.")
            continue

        try:
            df = pd.read_csv(csv_file)
            if "audit" not in df.columns:
                print(f"[Warning] 'audit' column not found in {csv_file.name}, skipping.")
                continue

            breaches = df[df["audit"] == "BREACH"]
            if not breaches.empty:
                breach_records.append(breaches)
                print(f"[INFO] Found {len(breaches)} breaches in {csv_file.name}.")
        except Exception as e:
            print(f"[Error] Could not read {csv_file.name}: {e}")

    end_time = time.time()

    if breach_records:
        final_df = pd.concat(breach_records, ignore_index=True)
        output_file = output_dir / "breaches.csv"
        final_df.to_csv(output_file, index=False)
        print(f"[INFO] Collected {len(final_df)} breaches to {output_file}")
        print(f"[INFO] Time taken to collect breaches: {end_time - start_time:.2f} seconds.")
    else:
        print(f"[INFO] No breaches found in any output files. {end_time - start_time:.2f} seconds.")


def save_results(results: List[dict], output_dir: Path, dataset_name: str, model_name: str) -> Path:
    """Save results to CSV file"""
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"results_{dataset_name}_{model_name}.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    return output_file


def save_token_logs(token_logs: List[dict], output_dir: Path) -> Path:
    """Save token logs to CSV file"""
    output_dir.mkdir(exist_ok=True)
    token_file = output_dir / "token_logs.csv"
    token_df = pd.DataFrame(token_logs)

    if token_file.exists():
        token_df.to_csv(token_file, mode='a', header=False, index=False)
    else:
        token_df.to_csv(token_file, index=False)

    return token_file


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """Calculate ASR, severity distribution and other metrics"""
    total = len(results_df)
    verdict_counts = results_df['audit'].value_counts()

    # ASR calculation
    num_breaches = verdict_counts.get("BREACH", 0)
    asr = num_breaches / total if total > 0 else 0

    # Severity metrics
    severity_metrics = {}
    if "severity" in results_df.columns:
        severity_counts = results_df["severity"].value_counts(dropna=False).sort_index()
        avg_severity = results_df["severity"].dropna().mean()
        severity_metrics = {
            "severity_counts": severity_counts.to_dict(),
            "avg_severity": avg_severity
        }

    return {
        "total_prompts": total,
        "verdict_counts": verdict_counts.to_dict(),
        "asr": asr,
        **severity_metrics
    }


def print_summary(metrics: dict, total_time: float, total_tokens: dict) -> None:
    """Print summary statistics"""
    print("===============================================================================")
    print(f"\nTime taken: {total_time:.2f} seconds")

    # Verdict counts
    for verdict, count in metrics["verdict_counts"].items():
        print(f"{verdict}: {count}")

    # ASR
    print(f"\nAttack Success Rate (ASR): {metrics['asr']:.2%}")

    # Severity distribution
    if "severity_counts" in metrics:
        print("\nSeverity distribution:")
        for sev, count in metrics["severity_counts"].items():
            print(f"Severity {sev}: {count}")
        print(f"Average Severity: {metrics.get('avg_severity', 0):.2f}")

    # Token usage
    if total_tokens:
        total_input = total_tokens.get('input', 0)
        total_output = total_tokens.get('output', 0)
        print(
            f"\nTotal input tokens: {total_input}, Total output tokens: {total_output}, Grand total: {total_input + total_output}")

    print("===============================================================================")
