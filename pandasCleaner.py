import pandas as pd
from pathlib import Path

UNCLEAN_DIR = Path("DatasetsUnclean")
CLEAN_DIR = Path("Datasets")

def clean_dataset():
    # List CSV files in DatasetsUnclean
    csv_files = list(UNCLEAN_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in DatasetsUnclean.")
        return

    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file.name}")

    # Choose CSV file
    while True:
        choice = input(f"Select a file to clean (1-{len(csv_files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            chosen_file = csv_files[int(choice) - 1]
            break
        print("Invalid selection, try again.")

    # Load the chosen CSV
    df = pd.read_csv(chosen_file)
    print(f"\nColumns in {chosen_file.name}: {list(df.columns)}")

    # Choose the column to use as 'prompt'
    while True:
        column_choice = input("Enter the name of the column to use as 'prompt': ")
        if column_choice in df.columns:
            break
        print("Invalid column name, try again.")

    # Keep only the chosen column and rename it to 'prompt'
    df_cleaned = df[[column_choice]].rename(columns={column_choice: "prompt"})

    # Save cleaned dataset
    CLEAN_DIR.mkdir(exist_ok=True)
    output_file = CLEAN_DIR / f"cleaned_{chosen_file.name}"
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to {output_file}")

if __name__ == "__main__":
    clean_dataset()
