import pandas as pd
from pathlib import Path
import tiktoken
import time

# === Part 1: Averages from token logs ===
OUTPUT_DIR = Path("Output")
TOKEN_LOGS_FILE = OUTPUT_DIR / "token_logs.csv"

df = pd.read_csv(TOKEN_LOGS_FILE)
df_filtered = df[df["tester_input_tokens"] > 0]

avg_tokens_per_model = df_filtered.groupby("model").agg(
    avg_tester_input_tokens=pd.NamedAgg(column="tester_input_tokens", aggfunc="mean"),
    avg_tester_output_tokens=pd.NamedAgg(column="tester_output_tokens", aggfunc="mean"),
    avg_auditor_input_tokens=pd.NamedAgg(column="auditor_input_tokens", aggfunc="mean"),
    avg_auditor_output_tokens=pd.NamedAgg(column="auditor_output_tokens", aggfunc="mean")
).reset_index()

print("Average tokens per model:")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(avg_tokens_per_model)

# === Part 2: Input token estimation from dataset ===
DATASET_DIR = Path("Datasets")

USD_TO_EUR = 0.857
# EUR prices per million tokens (split input vs output!)
model_prices_eur = {
    "gpt-4o-mini": {"input": 0.15 * USD_TO_EUR / 1_000_000, "output": 0.60 * USD_TO_EUR / 1_000_000},
    "gpt-5-mini": {"input": 0.25 * USD_TO_EUR / 1_000_000, "output": 2.00 * USD_TO_EUR / 1_000_000},
    # add others here...
}

enc = tiktoken.get_encoding("cl100k_base")
prompt_token_counts = {}

start_time = time.time()

for idx, csv_file in enumerate(DATASET_DIR.glob("*.csv"), start=1):
    print(f"[{idx}] Processing file: {csv_file.name}")
    df = pd.read_csv(csv_file)

    for jdx, prompt in enumerate(df["prompt"].dropna(), start=1):
        if prompt not in prompt_token_counts:
            tokens = enc.encode(prompt)
            prompt_token_counts[prompt] = len(tokens)
        if jdx % 1000 == 0:
            print(f"  Processed {jdx} prompts in {csv_file.name}...")

end_time = time.time()
elapsed = end_time - start_time

total_input_tokens = sum(prompt_token_counts.values())

# === Part 3: Estimate full costs ===
results = {}
for _, row in avg_tokens_per_model.iterrows():
    model = row["model"]

    if model not in model_prices_eur:
        continue

    # how much output relative to input?
    input_avg = row["avg_tester_input_tokens"] + row["avg_auditor_input_tokens"]
    output_avg = row["avg_tester_output_tokens"] + row["avg_auditor_output_tokens"]

    output_ratio = output_avg / input_avg if input_avg > 0 else 0
    est_output_tokens = total_input_tokens * output_ratio

    input_cost = total_input_tokens * model_prices_eur[model]["input"]
    output_cost = est_output_tokens * model_prices_eur[model]["output"]
    total_cost = input_cost + output_cost

    results[model] = {
        "input_tokens": total_input_tokens,
        "output_tokens": est_output_tokens,
        "input_cost_eur": input_cost,
        "output_cost_eur": output_cost,
        "total_cost_eur": total_cost
    }

print("\n=== Expected input + output token usage and cost per model ===")
for model, stats in results.items():
    print(
        f"{model}: "
        f"{stats['input_tokens']:,} in + {int(stats['output_tokens']):,} out "
        f"≈ €{stats['total_cost_eur']:.2f} "
        f"(€{stats['input_cost_eur']:.2f} input + €{stats['output_cost_eur']:.2f} output)"
    )

print(f"\nTotal processing time: {elapsed:.2f} seconds")
