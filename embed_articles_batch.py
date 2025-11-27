import pandas as pd
import ollama
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

MODEL = "nomic-embed-text"
BATCH_SIZE = 200  # smaller batches for CPU
MAX_WORKERS = 4   # number of threads

data_file_path = "data/news_last_1_year.parquet"
partial_file = "data/news_with_embeddings_1yr_partial.parquet"
final_file = "data/news_with_embeddings_1yr.parquet"

# Load existing partial embeddings if available
if os.path.exists(partial_file):
    df = pd.read_parquet(partial_file, engine="pyarrow")
    print("Resuming from partial embeddings file...")
else:
    df = pd.read_parquet(data_file_path, engine="pyarrow")
    df["embedding"] = None

# Only process articles without embeddings
df_to_embed = df[df["embedding"].isna()]

def build_text(row):
    return f"{row.title}\n{row.body}"

def embed_row(row):
    text = build_text(row)
    result = ollama.embeddings(MODEL, text)
    return row.Index, result["embedding"]

num_batches = math.ceil(len(df_to_embed) / BATCH_SIZE)
print(f"Processing {len(df_to_embed)} articles in {num_batches} batches...")

start_time = time.time()

for i in range(num_batches):
    batch = df_to_embed.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(embed_row, row) for row in batch.itertuples(index=True)]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i+1}/{num_batches}"):
            idx, emb = future.result()
            df.at[idx, "embedding"] = emb

    # Save intermediate results
    df.to_parquet(partial_file, index=False, compression="snappy")
    
    # Print dynamic ETA
    elapsed = time.time() - start_time
    batches_done = i + 1
    batches_left = num_batches - batches_done
    eta = (elapsed / batches_done) * batches_left
    print(f"Batch {i+1}/{num_batches} done. Estimated remaining time: {eta/60:.1f} minutes")

# Save final embeddings
df.to_parquet(final_file, index=False, compression="snappy")
print(df.head(1))
print("All embeddings saved successfully!")
