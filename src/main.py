import pandas as pd
from pathlib import Path
from tester import Tester
from auditor import Auditor
import os

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Output"

# Debug settings
DEBUG = True
DEBUG_LIMIT = 10


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

    # Ensure Output folder exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save results with dynamic filename
    output_file = OUTPUT_DIR / f"results_{DATASET_PATH.stem}.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    main()
