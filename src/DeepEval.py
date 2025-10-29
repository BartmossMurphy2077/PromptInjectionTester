import pandas as pd
from deepeval.dataset import EvaluationDataset

class DeepEvaluation:
    def __init__(self):
        self.dataset = EvaluationDataset()
        self.loaded = False

    def load_dataset(self, alias: str):
        """Load a reference dataset from Confident AI."""
        self.dataset.pull(alias=alias)
        self.loaded = True
        print(f"[DeepEvaluation] Loaded dataset '{alias}' from Confident AI")

    def evaluate_csv(self, csv_path: str, save_diffs: bool = True):
        """Compare CSV outputs against the goldens from Confident AI (audit only), with debug prints for every check."""
        if not self.loaded:
            raise Exception("Dataset not loaded. Please load a dataset first.")

        # Load CSV outputs and normalize column names
        local_df = pd.read_csv(csv_path)
        local_df.columns = [c.lower() for c in local_df.columns]  # lowercase for consistency
        print(f"[DeepEvaluation] Loaded CSV: {csv_path} ({len(local_df)} rows)")

        diffs = []

        for golden in self.dataset.goldens:
            # Find the CSV row corresponding to this prompt
            match = local_df[local_df['prompt'] == golden.input]
            if match.empty:
                diffs.append({
                    "prompt": golden.input,
                    "error": "No matching output in CSV"
                })
                print(f"[DEBUG] Prompt not found in CSV: {golden.input[:60]}...")
                continue

            csv_audit = str(match.iloc[0]['audit']).strip().upper()
            expected_audit = str(golden.expected_output).strip().upper()

            # DEBUG: print EVERY comparison
            match_status = "✅ MATCH" if csv_audit == expected_audit else "❌ MISMATCH"
            print(f"[DEBUG] Prompt: {golden.input[:60]}...")
            print(f"        Expected audit: {expected_audit}")
            print(f"        CSV audit     : {csv_audit}")
            print(f"        Status        : {match_status}\n")

            # Store diffs only if there is a mismatch
            if csv_audit != expected_audit:
                diffs.append({
                    "prompt": golden.input,
                    "expected_audit": expected_audit,
                    "actual_audit": csv_audit
                })

        print(f"\n[DeepEvaluation] Total prompts checked: {len(self.dataset.goldens)}")
        print(f"[DeepEvaluation] Differences found: {len(diffs)}")

        if save_diffs and diffs:
            out_path = "../Output/diffs.csv"
            pd.DataFrame(diffs).to_csv(out_path, index=False)
            print(f"[DeepEvaluation] Saved differences to {out_path}")


if __name__ == "__main__":
    deep_eval = DeepEvaluation()
    deep_eval.load_dataset("PromptInjectionReproducibility")
    deep_eval.evaluate_csv("../Output/results_cleaned_malicous_deepset_gpt-4o-mini.csv", save_diffs=False)

# dataset = EvaluationDataset()
# dataset.pull(alias="PromptInjectionReproducibility")
# print(dataset.goldens)