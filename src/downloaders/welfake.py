from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
ds = load_dataset("davanstrien/WELFake")

# The default split is 'train'; convert to a pandas DataFrame
df = ds["train"].to_pandas()

# Inspect structure
print(df.head())
print(df.columns)

# Save to CSV so your Week-1 pipeline can use it
df.to_csv("data/WELFake.csv", index=False)
print("✅  Saved to data/WELFake.csv")
