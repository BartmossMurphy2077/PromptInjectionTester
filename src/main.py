import time
import pandas as pd
from pathlib import Path
from tester import Tester
from auditor import Auditor, AuditResult
from utils import DEBUG, RUN_LIMIT, AZURE_DEPLOYMENT_NAME
import os

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"

# UI for listing and selecting a dataset
def select_dataset():
    csv_files = list(DATASET_DIR.glob("*.csv"))
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


#A function for collecting breaches from all output CSVs and then aggregating them
#Into a single breaches.csv file in the output directory
def collect_breaches(output_dir: Path):

    print("========================Collecting breaches========================")

    start_time = time.time()

    breach_records = []

    for csv_file in output_dir.glob("*.csv"):
        if csv_file.name == "breaches.csv" or csv_file.name == "token_logs.csv":
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


def main():
    #Loading the dataset
    DATASET_PATH = select_dataset()
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["prompt"]).reset_index(drop=True)

    #DEBUG limit for testing
    if RUN_LIMIT > 0:
        print(f"[DEBUG] Debug limit: {RUN_LIMIT}")
        df = df.head(RUN_LIMIT)

    #Creation of Tester and Auditor instances
    tester = Tester()
    auditor = Auditor()

    #Processing each prompt
    results = []
    total = len(df)
    total_input_tokens = 0
    total_output_tokens = 0

    token_logs = []

    #Timer to see how long the process takes
    start_time = time.time()

    print(f"========================[Testing Model {AZURE_DEPLOYMENT_NAME}]========================")

    print(f"Processing {total} prompts from {DATASET_PATH.name}")

    #Main iteration loop
    for i, row in df.iterrows():
        print(f"*********************************[{i + 1}/{total}]*********************************")
        prompt = row["prompt"]
        print(f"Processing prompt: {prompt[:50]}...")

        # Runs tester and stores response and token usage
        tester_response, tester_input_tokens, tester_output_tokens = tester.run(prompt)
        total_input_tokens += tester_input_tokens
        total_output_tokens += tester_output_tokens

        # Runs auditor and stores verdict, explanation, severity, and token usage
        audit_result, auditor_input_tokens, auditor_output_tokens = auditor.check(prompt, tester_response)
        total_input_tokens += auditor_input_tokens
        total_output_tokens += auditor_output_tokens

        #Error prevention if auditor skips
        if isinstance(audit_result, AuditResult):
            verdict = audit_result.verdict
            explanation = audit_result.explanation
            severity = getattr(audit_result, "severity", None)
        else:
            verdict, explanation = audit_result
            severity = None

        #appends to the output
        results.append({
            "prompt": prompt,
            "response": tester_response,
            "audit": verdict,
            "explanation": explanation,
            "severity": severity,
            "model": AZURE_DEPLOYMENT_NAME
        })

        #appends to token results for potential future analysis
        token_logs.append({
            "model": AZURE_DEPLOYMENT_NAME,
            "dataset_name": DATASET_PATH.stem,
            "prompt": prompt,
            "tester_input_tokens": tester_input_tokens,
            "tester_output_tokens": tester_output_tokens,
            "auditor_input_tokens": auditor_input_tokens,
            "auditor_output_tokens": auditor_output_tokens
        })

        #DEBUG info for token consumption
        if DEBUG:
            print(f"[Tokens] - Tester: in={tester_input_tokens}, out={tester_output_tokens} | "
                  f"Auditor: in={auditor_input_tokens}, out={auditor_output_tokens} | "
                  f"Total: {tester_input_tokens + tester_output_tokens + auditor_input_tokens + auditor_output_tokens}")
            print(f"[Verdict]: {verdict}, Severity: {severity}, Explanation: {explanation[:60]}...")

        print(f"******************************END_OF_PROMPT******************************")

    # Ending timer
    end_time = time.time()

    #Saving results to CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"results_{DATASET_PATH.stem}_{AZURE_DEPLOYMENT_NAME}.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)

    #saving token logs to CSV
    token_file = OUTPUT_DIR / f"token_logs.csv"
    token_df = pd.DataFrame(token_logs)
    if token_file.exists():
        token_df.to_csv(token_file, mode='a', header=False, index=False)
    else:
        token_df.to_csv(token_file, index=False)

    print("===============================================================================")
    print(f"\nDone! Results saved to {output_file}")
    print(f"\nToken logs saved to {token_file}")

    print(f"\nTime taken: {end_time - start_time:.2f} seconds")

    verdict_counts = out_df['audit'].value_counts()
    for verdict, count in verdict_counts.items():
        print(f"{verdict}: {count}")

    #asr scoring
    num_breaches = verdict_counts.get("BREACH", 0)
    asr = num_breaches/total
    print(f"\nAttack Success Rate (ASR): {asr:.2%}")

    # ISR scoring (severity distribution and average)
    if "severity" in out_df.columns:
        severity_counts = out_df["severity"].value_counts(dropna=False).sort_index()
        print("\nSeverity distribution:")
        for sev, count in severity_counts.items():
            print(f"Severity {sev}: {count}")

        # Calculate average severity (ignoring nulls)
        avg_severity = out_df["severity"].dropna().mean()
        print(f"Average Severity: {avg_severity:.2f}")


    if DEBUG:
        print(f"\nTotal input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}, Grand total: {total_input_tokens + total_output_tokens}")
    print("===============================================================================")


if __name__ == "__main__":
    main()
    collect_breaches(OUTPUT_DIR)
