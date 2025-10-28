import pandas as pd
from deepeval.dataset import EvaluationDataset


class DeepEvaluation:
    def __init__(self):
        self.dataset = EvaluationDataset()
        self.pull = False

    def load_dataset(self, alias: str):
        """Load a reference dataset from Confident AI."""
        self.dataset.pull(alias=alias)
        self.pull = True
        print(f"[DeepEvaluation] Loaded dataset '{alias}' from Confident AI")

    def save_dataset(self, csv_path: str, alias: str, finalized: bool = False):
        """Push a local CSV dataset to Confident AI."""
        df = pd.read_csv(csv_path)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "input": row["prompt"],
                "output": {
                    "audit": row["audit"],
                    "severity": row["severity"],
                    "category": row["category"],
                    "explanation": row["explanation"],
                    "response": row["response"],
                },
            })
        ds = EvaluationDataset()
        ds.add(rows)
        ds.push(alias=alias, finalized=finalized)
        print(f"[DeepEvaluation] Pushed dataset '{alias}' (finalized={finalized}) to Confident AI.")

    def evaluate(self, csv_path: str, save_diffs: bool = True):
        """Compare local CSV results against the loaded Confident AI dataset."""
        if not self.pull:
            raise Exception("Dataset not loaded. Please load a dataset before evaluation.")

        print(f"[DeepEvaluation] Evaluating local CSV: {csv_path}")
        local_df = pd.read_csv(csv_path)

        # Build reference DataFrame from goldens
        ref_df = pd.DataFrame([{
            "prompt": g.input,
            "audit": getattr(g, "audit", None),
            "severity": getattr(g, "severity", None),
            "category": getattr(g, "category", None),
            "explanation": getattr(g, "explanation", None),
            "response": getattr(g, "response", None),
        } for g in self.dataset.goldens])

        # Merge local CSV with reference dataset on prompt
        merged = pd.merge(local_df, ref_df, on="prompt", suffixes=("_new", "_ref"))
        total = len(merged)
        diffs = []

        for _, row in merged.iterrows():
            if (
                    row["audit_new"] != row["audit_ref"] or
                    row["severity_new"] != row["severity_ref"] or
                    row["category_new"] != row["category_ref"]
            ):
                diffs.append({
                    "prompt": row["prompt"],
                    "audit_new": row["audit_new"],
                    "audit_ref": row["audit_ref"],
                    "severity_new": row["severity_new"],
                    "severity_ref": row["severity_ref"],
                    "category_new": row["category_new"],
                    "category_ref": row["category_ref"],
                })
                print(f"\n[DIFF FOUND] Prompt: {row['prompt'][:60]}...")
                print(f"  Audit: {row['audit_new']} (expected {row['audit_ref']})")
                print(f"  Severity: {row['severity_new']} (expected {row['severity_ref']})")
                print(f"  Category: {row['category_new']} (expected {row['category_ref']})")

        print(f"\n[DeepEvaluation] Compared {total} prompts.")
        print(f"[DeepEvaluation] Found {len(diffs)} differences.")

        if save_diffs and diffs:
            out_path = "../Output/diffs.csv"
            pd.DataFrame(diffs).to_csv(out_path, index=False)
            print(f"[DeepEvaluation] Saved detailed differences to {out_path}")


if __name__ == "__main__":
    deep_eval = DeepEvaluation()

    # 1️⃣ Load the reference dataset from Confident AI
    deep_eval.load_dataset("PromptInjectionReproducibility")

    # 2️⃣ Evaluate your new model output from Output/
    deep_eval.evaluate("../Output/results_cleaned_malicous_deepset_gpt-4o-mini.csv")
