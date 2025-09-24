import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("Output")
TOKEN_LOGS_FILE = OUTPUT_DIR / "token_logs.csv"

df = pd.read_csv(TOKEN_LOGS_FILE)

# Filter out records where tester_input_tokens is 0 (prompt shield cases)
df_filtered = df[df["tester_input_tokens"] > 0]

# Compute average input and output tokens per model
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
