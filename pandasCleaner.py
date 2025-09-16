import pandas as pd

# Load your dataset
df = pd.read_csv("DatasetsUnclean/malignant.csv")

# Keep only 'text' column and rename it to 'prompt'
df = df[["text"]].rename(columns={"text": "prompt"})

# Save it back (overwrite or new file)
df.to_csv("Datasets/cleaned_prompts.csv", index=False)
