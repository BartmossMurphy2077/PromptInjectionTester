import asyncio
import time
from pathlib import Path
import pandas as pd
from pydantic_evals import Case, Dataset

from tester import Tester
from auditor import Auditor, AuditResult
from utils import (
    DEBUG, RUN_LIMIT, AZURE_DEPLOYMENT_NAME, AUDITOR_CHECKS_PROMPT_AND_RESPONSE,
    CONCURRENCY_LIMIT, CHECKPOINTING, CHECKPOINTING_LENGTH
)
from eval_helpers import (
    select_dataset, write_eval_outputs, collect_breaches_from_eval_output,
    save_checkpoint, merge_checkpoints
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"
CHECKPOINT_DIR = ROOT / "Checkpoints"


async def task_fn(prompt: str, i: int = None, total: int = None) -> dict:
    """Async task executed for each prompt."""
    if i is not None and total is not None:
        print(f"\n*********************************[{i + 1}/{total}]*********************************")
        print(f"Processing prompt: {prompt[:80]}...")

    resp, in_t, out_t = await task_fn.tester.run_async(prompt)
    audit_result, in_a, out_a = await task_fn.auditor.check_async(prompt, resp)

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
        print(f"[Tester Response]: {resp[:80]}...")
        print(f"[Verdict]: {verdict}, Severity: {severity}, Explanation: {explanation[:80]}...")
        print(f"******************************END_OF_PROMPT******************************")

    return {
        "prompt": prompt,
        "response": resp,
        "audit_verdict": verdict,
        "audit_explanation": explanation,
        "audit_severity": severity,
        "audit_category": category,
        "tester_input_tokens": in_t,
        "tester_output_tokens": out_t,
        "auditor_input_tokens": in_a,
        "auditor_output_tokens": out_a,
    }


def build_dataset_from_csv(csv_path: Path) -> Dataset:
    df = pd.read_csv(csv_path).dropna(subset=["prompt"]).reset_index(drop=True)
    if RUN_LIMIT > 0:
        print(f"[DEBUG] Debug limit: {RUN_LIMIT}")
        df = df.head(RUN_LIMIT)
    cases = [Case(name=f"case_{i}", inputs=row["prompt"]) for i, row in df.iterrows()]
    return Dataset(cases=cases)


async def main_async():
    DATASET_PATH = select_dataset(DATASET_DIR)
    dataset = build_dataset_from_csv(DATASET_PATH)

    tester = Tester()
    auditor = Auditor()
    task_fn.tester = tester
    task_fn.auditor = auditor

    total_cases = len(dataset.cases)
    print(f"\n========================[Evaluating Model {AZURE_DEPLOYMENT_NAME}]========================")
    print(f"Dataset: {DATASET_PATH.name}, Cases: {total_cases}")
    if CHECKPOINTING:
        print(f"Checkpointing enabled: saving every {CHECKPOINTING_LENGTH} results")
    print("==============================================================================")

    start_time = time.time()

    # Manual async handling with checkpointing
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    results = []
    checkpoint_counter = 0

    async def wrapped_task(case, i):
        async with semaphore:
            return await task_fn(case.inputs, i, total_cases)

    # Process in batches if checkpointing is enabled
    if CHECKPOINTING:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        for batch_start in range(0, total_cases, CHECKPOINTING_LENGTH):
            batch_end = min(batch_start + CHECKPOINTING_LENGTH, total_cases)
            batch_cases = dataset.cases[batch_start:batch_end]

            print(f"\n[Checkpoint] Processing batch {batch_start}-{batch_end}...")

            tasks = [wrapped_task(case, batch_start + i) for i, case in enumerate(batch_cases)]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Save checkpoint
            save_checkpoint(
                batch_results,
                CHECKPOINT_DIR,
                checkpoint_counter,
                DATASET_PATH.stem,
                AZURE_DEPLOYMENT_NAME
            )
            checkpoint_counter += 1

            print(f"[Checkpoint] Saved checkpoint {checkpoint_counter} ({len(batch_results)} results)")
    else:
        # Process all at once without checkpointing
        tasks = [wrapped_task(case, i) for i, case in enumerate(dataset.cases)]
        results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # Merge checkpoints if enabled
    if CHECKPOINTING:
        print("\n========================[Merging Checkpoints]========================")
        merge_checkpoints(CHECKPOINT_DIR, OUTPUT_DIR, DATASET_PATH.stem, AZURE_DEPLOYMENT_NAME)
        print(f"[Checkpoint] Merged all checkpoints into Output directory")
    else:
        # Save results normally
        print("\n========================[Saving Outputs]========================")

        class SimpleCase:
            def __init__(self, inputs, output):
                self.inputs = inputs
                self.output = output

        class SimpleReport:
            def __init__(self, cases):
                self.cases = cases

        fake_report = SimpleReport([
            SimpleCase(r["prompt"], r) for r in results
        ])

        write_eval_outputs(fake_report, OUTPUT_DIR, dataset_name=DATASET_PATH.stem, model_name=AZURE_DEPLOYMENT_NAME)

    # Compute token totals
    total_input_tokens = sum((r["tester_input_tokens"] or 0) + (r["auditor_input_tokens"] or 0) for r in results)
    total_output_tokens = sum((r["tester_output_tokens"] or 0) + (r["auditor_output_tokens"] or 0) for r in results)

    print(f"\n========================[Summary]========================")
    print(f"Total prompts tested: {total_cases}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Concurrency limit: {CONCURRENCY_LIMIT}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Auditor checks both prompt and response: {AUDITOR_CHECKS_PROMPT_AND_RESPONSE}")
    print("==========================================================")

    verdict_counts = {"SAFE": 0, "BREACH": 0, "PROMPTSHIELD": 0, "AUDITOR_PROMPTSHIELD": 0}
    for r in results:
        v = r["audit_verdict"]
        if v in verdict_counts:
            verdict_counts[v] += 1
        elif isinstance(v, str) and v.startswith("AUDITOR_") and v not in verdict_counts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

    print("\n========================[Verdict Breakdown]========================")
    for key in ["PROMPTSHIELD", "BREACH", "SAFE", "AUDITOR_PROMPTSHIELD"]:
        print(f"{key}: {verdict_counts.get(key, 0)}")
    print("===================================================================")

    collect_breaches_from_eval_output(OUTPUT_DIR)
    print("==========================================================")
    print("Evaluation complete ✅")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
