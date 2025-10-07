import time
import pandas as pd
from pathlib import Path
from tester import Tester
from auditor import Auditor, AuditResult
from helpers import select_dataset, collect_breaches, save_results, save_token_logs, calculate_metrics, print_summary
from utils import DEBUG, RUN_LIMIT, AZURE_DEPLOYMENT_NAME, AUDITOR_CHECKS_PROMPT_AND_RESPONSE

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"


def main():
    # Loading the dataset
    DATASET_PATH = select_dataset(DATASET_DIR)
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["prompt"]).reset_index(drop=True)

    # DEBUG limit for testing
    if RUN_LIMIT > 0:
        print(f"[DEBUG] Debug limit: {RUN_LIMIT}")
        df = df.head(RUN_LIMIT)

    # Creation of Tester and Auditor instances
    tester = Tester()
    auditor = Auditor()

    # Processing setup
    results = []
    total = len(df)
    total_input_tokens = 0
    total_output_tokens = 0
    token_logs = []

    # Timer
    start_time = time.time()

    print(f"========================[Testing Model {AZURE_DEPLOYMENT_NAME}]========================")
    print(f"Processing {total} prompts from {DATASET_PATH.name}")

    # Main iteration loop
    for i, row in df.iterrows():
        print(f"*********************************[{i + 1}/{total}]*********************************")
        prompt = row["prompt"]
        print(f"Processing prompt: {prompt[:50]}...")

        # Run tester
        tester_response, tester_input_tokens, tester_output_tokens = tester.run(prompt)
        total_input_tokens += tester_input_tokens
        total_output_tokens += tester_output_tokens

        # Run auditor
        audit_result, auditor_input_tokens, auditor_output_tokens = auditor.check(prompt, tester_response)
        total_input_tokens += auditor_input_tokens
        total_output_tokens += auditor_output_tokens

        # Process audit result
        if isinstance(audit_result, AuditResult):
            verdict = audit_result.verdict
            explanation = audit_result.explanation
            severity = getattr(audit_result, "severity", None)
            category = getattr(audit_result, "category", "others")
        else:
            verdict, explanation = audit_result
            severity = None
            category = "others"

        # Append results
        results.append({
            "prompt": prompt,
            "response": tester_response,
            "audit": verdict,
            "explanation": explanation,
            "severity": severity,
            "category": category,
            "Auditor_checks_prompt_and_response": AUDITOR_CHECKS_PROMPT_AND_RESPONSE,
            "model": AZURE_DEPLOYMENT_NAME
        })

        # Token logs
        token_logs.append({
            "model": AZURE_DEPLOYMENT_NAME,
            "dataset_name": DATASET_PATH.stem,
            "prompt": prompt,
            "tester_input_tokens": tester_input_tokens,
            "tester_output_tokens": tester_output_tokens,
            "auditor_input_tokens": auditor_input_tokens,
            "auditor_output_tokens": auditor_output_tokens
        })

        # Debug info
        if DEBUG:
            print(f"[Tokens] - Tester: in={tester_input_tokens}, out={tester_output_tokens} | "
                  f"Auditor: in={auditor_input_tokens}, out={auditor_output_tokens} | "
                  f"Total: {tester_input_tokens + tester_output_tokens + auditor_input_tokens + auditor_output_tokens}")

        print(f"[Tester Response]: {tester_response[:60]}...")
        print(f"[Verdict]: {verdict}, Severity: {severity}, Explanation: {explanation[:60]}...")
        print(f"******************************END_OF_PROMPT******************************")

    # End timer
    end_time = time.time()
    total_time = end_time - start_time

    # Save results
    output_file = save_results(results, OUTPUT_DIR, DATASET_PATH.stem, AZURE_DEPLOYMENT_NAME)
    token_file = save_token_logs(token_logs, OUTPUT_DIR)

    print(f"\nDone! Results saved to {output_file}")
    print(f"Token logs saved to {token_file}")

    # Calculate and print metrics
    results_df = pd.DataFrame(results)
    metrics = calculate_metrics(results_df)
    total_tokens = {
        'input': total_input_tokens,
        'output': total_output_tokens
    }

    print_summary(metrics, total_time, total_tokens)


if __name__ == "__main__":
    main()
    collect_breaches(OUTPUT_DIR)
