import pandas as pd
from pathlib import Path
from tester import Tester
from auditor import Auditor
import os

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Datasets" / "cleaned_prompts.csv"
OUTPUT_PATH = ROOT / "Output" / "results.csv"

# Debug settings
DEBUG = True
DEBUG_LIMIT = 10


def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Drop empty prompts just in case
    df = df.dropna(subset=["prompt"]).reset_index(drop=True)

    # Apply debug limit
    if DEBUG:
        df = df.head(DEBUG_LIMIT)

    tester = Tester()
    auditor = Auditor()

    results = []
    total = len(df)

    for i, row in df.iterrows():
        prompt = row["prompt"]
        print(f"[{i + 1}/{total}] Processing prompt: {prompt[:50]}...")  # show first 50 chars

        tester_response = tester.run(prompt)
        verdict = auditor.check(prompt, tester_response)

        results.append({"prompt": prompt, "response": tester_response, "audit": verdict})

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
