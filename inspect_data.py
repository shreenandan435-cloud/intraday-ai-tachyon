import glob
import os
import pandas as pd

# Recursively search for any parquet files in the current project or data directory
files = glob.glob("**/*.parquet", recursive=True)

if not files:
    # If not in the project folder, check if a specific path was meant
    print("No .parquet files found in the current workspace.")
    print("If your files are in another folder (e.g., Downloads, Documents, or D: drive),")
    print("replace 'target' below with the full path to one of those files.")
else:
    target = files[0]
    print(f"Examining file: {target}\n")
    try:
        df = pd.read_parquet(target)
        print("=== DATASET SHAPE ===")
        print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")
        
        print("=== SCHEMA ===")
        print(df.dtypes, "\n")
        
        print("=== FIRST 3 ROWS ===")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"Failed to parse {target}: {e}")