import asyncio
import time
import pandas as pd
from pathlib import Path
from tester import Tester
from auditor import Auditor, AuditResult
from helpers import (
    select_dataset, collect_breaches, save_results, save_token_logs,
    calculate_metrics, print_summary
)
from utils import DEBUG, RUN_LIMIT, AZURE_DEPLOYMENT_NAME, AUDITOR_CHECKS_PROMPT_AND_RESPONSE, CONCURRENCY_LIMIT

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"


async def process_prompt(semaphore, tester, auditor, prompt, i, total):
    """Process one prompt asynchronously with concurrency control."""
    async with semaphore:
        print(f"*********************************[{i + 1}/{total}]*********************************")
        print(f"Processing prompt: {prompt[:50]}...")

        tester_response, tester_in, tester_out = await tester.run_async(prompt)
        audit_result, audit_in, audit_out = await auditor.check_async(prompt, tester_response)

        if isinstance(audit_result, AuditResult):
            verdict = audit_result.verdict
            explanation = audit_result.explanation
            severity = getattr(audit_result, "severity", None)
            category = getattr(audit_result, "category", "others")
        else:
            verdict, explanation = audit_result
            severity = None
            category = "others"

        if DEBUG:
            print(f"[Tester Response]: {tester_response[:60]}...")
            print(f"[Verdict]: {verdict}, Severity: {severity}, Explanation: {explanation[:60]}...")
            print(f"******************************END_OF_PROMPT******************************")

        return {
            "prompt": prompt,
            "response": tester_response,
            "audit": verdict,
            "explanation": explanation,
            "severity": severity,
            "category": category,
            "tester_input_tokens": tester_in,
            "tester_output_tokens": tester_out,
            "auditor_input_tokens": audit_in,
            "auditor_output_tokens": audit_out,
        }


async def main():
    DATASET_PATH = select_dataset(DATASET_DIR)
    df = pd.read_csv(DATASET_PATH).dropna(subset=["prompt"]).reset_index(drop=True)

    if RUN_LIMIT > 0:
        print(f"[DEBUG] Debug limit: {RUN_LIMIT}")
        df = df.head(RUN_LIMIT)

    tester = Tester()
    auditor = Auditor()

    total = len(df)
    results = []
    token_logs = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    start_time = time.time()
    print(f"========================[Testing Model {AZURE_DEPLOYMENT_NAME}]========================")
    print(f"Processing {total} prompts from {DATASET_PATH.name}")

    # Create a coroutine for each prompt
    tasks = [
        process_prompt(semaphore, tester, auditor, row["prompt"], i, total)
        for i, row in df.iterrows()
    ]

    # Run all concurrently (respecting the concurrency limit)
    task_results = await asyncio.gather(*tasks)

    # Merge results
    for res in task_results:
        results.append({
            "prompt": res["prompt"],
            "response": res["response"],
            "audit": res["audit"],
            "explanation": res["explanation"],
            "severity": res["severity"],
            "category": res["category"],
            "Auditor_checks_prompt_and_response": AUDITOR_CHECKS_PROMPT_AND_RESPONSE,
            "model": AZURE_DEPLOYMENT_NAME,
        })
        token_logs.append({
            "model": AZURE_DEPLOYMENT_NAME,
            "dataset_name": DATASET_PATH.stem,
            "prompt": res["prompt"],
            "tester_input_tokens": res["tester_input_tokens"],
            "tester_output_tokens": res["tester_output_tokens"],
            "auditor_input_tokens": res["auditor_input_tokens"],
            "auditor_output_tokens": res["auditor_output_tokens"],
        })

    end_time = time.time()
    total_time = end_time - start_time

    total_input_tokens = sum(r["tester_input_tokens"] + r["auditor_input_tokens"] for r in task_results)
    total_output_tokens = sum(r["tester_output_tokens"] + r["auditor_output_tokens"] for r in task_results)

    output_file = save_results(results, OUTPUT_DIR, DATASET_PATH.stem, AZURE_DEPLOYMENT_NAME)
    token_file = save_token_logs(token_logs, OUTPUT_DIR)

    print(f"\nDone! Results saved to {output_file}")
    print(f"Token logs saved to {token_file}")

    metrics = calculate_metrics(pd.DataFrame(results))
    total_tokens = {"input": total_input_tokens, "output": total_output_tokens}
    print_summary(metrics, total_time, total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
    collect_breaches(OUTPUT_DIR)
