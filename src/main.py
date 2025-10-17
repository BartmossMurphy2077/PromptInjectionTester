import asyncio
import time
from pathlib import Path

import pandas as pd
from pydantic_evals import Case, Dataset

from tester import Tester
from auditor import Auditor, AuditResult
from utils import (
    DEBUG, RUN_LIMIT, AZURE_DEPLOYMENT_NAME, AUDITOR_CHECKS_PROMPT_AND_RESPONSE,
    CONCURRENCY_LIMIT
)
from eval_helpers import (
    select_dataset, write_eval_outputs, collect_breaches_from_eval_output
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"


async def task_fn(prompt: str) -> dict:
    """
    This is the “task” Evals will call for each Case.inputs.
    We return a dict output; we also record metrics in ctx.attributes.
    """
    # instantiate testers / auditors at top-level to reuse connections? or globally
    resp, in_t, out_t = await task_fn.tester.run_async(prompt)
    audit_result, in_a, out_a = await task_fn.auditor.check_async(prompt, resp)

    # Normalize audit_result into structured form
    if isinstance(audit_result, AuditResult):
        verdict = audit_result.verdict
        explanation = audit_result.explanation
        severity = audit_result.severity
        category = audit_result.category
    else:
        verdict, explanation = audit_result
        severity = None
        category = None

    # You can access `ctx` via `pydantic_evals.current_context()` if needed,
    # but the cleaner pattern is to return attributes in the output,
    # then do a post-walk to dump them.

    return {
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
    import pandas as pd
    df = pd.read_csv(csv_path).dropna(subset=["prompt"]).reset_index(drop=True)

    if RUN_LIMIT > 0:
        print(f"[DEBUG] Debug limit: {RUN_LIMIT}")
        df = df.head(RUN_LIMIT)

    cases = []
    for idx, row in df.iterrows():
        cases.append(Case(name=f"case_{idx}", inputs=row["prompt"]))
    return Dataset(cases=cases)


def main():
    # (Synchronous entry point wrapping async)
    asyncio.run(main_async())


async def main_async():
    DATASET_PATH = select_dataset(DATASET_DIR)
    dataset = build_dataset_from_csv(DATASET_PATH)

    # Bind tester / auditor to task_fn
    tester = Tester()
    auditor = Auditor()
    task_fn.tester = tester
    task_fn.auditor = auditor

    print(f"========================[Evaluating Model {AZURE_DEPLOYMENT_NAME}]========================")
    print(f"Dataset: {DATASET_PATH.name}, Cases: {len(dataset.cases)}")

    start_time = time.time()

    # Use `evaluate`: the async evaluation method
    report = await dataset.evaluate(task_fn, max_concurrency=CONCURRENCY_LIMIT)

    end_time = time.time()
    total_time = end_time - start_time

    # Dump outputs
    write_eval_outputs(report, OUTPUT_DIR, dataset_name=DATASET_PATH.stem, model_name=AZURE_DEPLOYMENT_NAME)
    collect_breaches_from_eval_output(OUTPUT_DIR)

    print(f"Completed evaluation in {total_time:.2f}s")


if __name__ == "__main__":
    main()
