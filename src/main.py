import pandas as pd
from pathlib import Path
from tester import Tester
from auditor import Auditor, AuditResult
from utils import DEBUG, DEBUG_LIMIT
import os

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"


def select_dataset():
    # List CSV files in Datasets
    csv_files = list(DATASET_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in Datasets.")
        exit(1)

    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file.name}")

    # Interactive selection
    while True:
        choice = input(f"Select a dataset (1-{len(csv_files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            return csv_files[int(choice) - 1]
        print("Invalid selection, try again.")


def main():
    # Let user select dataset
    DATASET_PATH = select_dataset()
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["prompt"]).reset_index(drop=True)

    if DEBUG:
        df = df.head(DEBUG_LIMIT)

    tester = Tester()
    auditor = Auditor()

    results = []
    total = len(df)
    total_tokens = 0

    for i, row in df.iterrows():
        prompt = row["prompt"]
        print(f"[{i + 1}/{total}] Processing prompt: {prompt[:50]}...")

        tester_response, tester_tokens = tester.run(prompt)
        audit_result, auditor_tokens = auditor.check(prompt, tester_response)

        total_tokens += tester_tokens + auditor_tokens

        if isinstance(audit_result, AuditResult):
            verdict = audit_result.verdict
            explanation = audit_result.explanation
        else:
            verdict, explanation = audit_result

        results.append({
            "prompt": prompt,
            "response": tester_response,
            "audit": verdict,
            "explanation": explanation
        })

        print(f"  Verdict: {verdict}, Explanation: {explanation[:60]}...")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"results_{DATASET_PATH.stem}.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)

    print("===================================================")
    print(f"Done! Results saved to {output_file}")
    if(DEBUG):
        print(f"======Total tokens consumed: {total_tokens} ======")
    print(f"==================================================")


if __name__ == "__main__":
    main()
