from pathlib import Path
import pandas as pd
from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import ReportCase, ReportCaseFailure


def select_dataset(dataset_dir: Path) -> Path:
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


def write_eval_outputs(report, output_dir: Path, dataset_name: str, model_name: str):
    """
    Walk through report.cases and write:
      - A "results" CSV (prompt, response, audit verdict, etc.)
      - A "token_logs" CSV (token usage per case)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{dataset_name}_{model_name}.csv"
    token_logs_path = output_dir / f"token_logs_{dataset_name}_{model_name}.csv"

    results_rows = []
    token_logs_rows = []

    for case in report.cases:
        # case may be ReportCase or ReportCaseFailure
        # In the failure case, case.output may not exist
        if isinstance(case, ReportCaseFailure):
            inputs = case.inputs
            # you might want to store the error
            results_rows.append({
                "prompt": inputs,
                "response": None,
                "audit": None,
                "explanation": f"Error: {case}",  # or case.error_message if available
                "severity": None,
                "category": None,
                "model": model_name,
            })
            token_logs_rows.append({
                "prompt": inputs,
                "tester_input_tokens": None,
                "tester_output_tokens": None,
                "auditor_input_tokens": None,
                "auditor_output_tokens": None,
                "model": model_name,
            })
        else:
            # normal ReportCase
            inputs = case.inputs
            out = case.output or {}
            results_rows.append({
                "prompt": inputs,
                "response": out.get("response"),
                "audit": out.get("audit_verdict"),
                "explanation": out.get("audit_explanation"),
                "severity": out.get("audit_severity"),
                "category": out.get("audit_category"),
                "model": model_name,
            })
            token_logs_rows.append({
                "prompt": inputs,
                "tester_input_tokens": out.get("tester_input_tokens"),
                "tester_output_tokens": out.get("tester_output_tokens"),
                "auditor_input_tokens": out.get("auditor_input_tokens"),
                "auditor_output_tokens": out.get("auditor_output_tokens"),
                "model": model_name,
            })

    pd.DataFrame(results_rows).to_csv(results_path, index=False)
    pd.DataFrame(token_logs_rows).to_csv(token_logs_path, index=False)
    print(f"[EvalHelpers] Wrote results to {results_path}")
    print(f"[EvalHelpers] Wrote token logs to {token_logs_path}")


def collect_breaches_from_eval_output(output_dir: Path):
    """
    Similar to your earlier collect_breaches logic, but scanning the results_*.csv files.
    """
    print("========================Collecting breaches========================")
    breach_records = []
    for csv_file in output_dir.glob("results_*.csv"):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[Error] Cannot read {csv_file}: {e}")
            continue
        if "audit" not in df.columns:
            print(f"[Warning] No 'audit' column in {csv_file}, skipping.")
            continue
        breaches = df[df["audit"] == "BREACH"]
        if not breaches.empty:
            breach_records.append(breaches)
            print(f"[Info] Found {len(breaches)} in {csv_file.name}")
    if breach_records:
        final = pd.concat(breach_records, ignore_index=True)
        out_path = output_dir / "breaches.csv"
        final.to_csv(out_path, index=False)
        print(f"[EvalHelpers] Collected {len(final)} breaches into {out_path}")
    else:
        print("[EvalHelpers] No breaches found in any result files.")
