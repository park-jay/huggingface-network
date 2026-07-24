import pandas as pd
from tqdm import tqdm

INPUT_FILE = "huggingface_models.csv"
OUTPUT_FILE = "task_generation_dataset.csv"

CHUNK_SIZE = 50000  # good for large files

first_chunk = True

print("Filtering text-generation models...")

for chunk in tqdm(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)):

    # filter rows
    filtered = chunk[chunk["task"] == "text-generation"]

    # write to output
    filtered.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=first_chunk
    )

    first_chunk = False

print("Finished!")
print(f"Saved to {OUTPUT_FILE}")

