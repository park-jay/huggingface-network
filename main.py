import pandas as pd
import csv
import re
import os
import time
from tqdm import tqdm
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

INPUT_FILE = "task_generation_dataset.csv"
OUTPUT_FILE = "task_generation_enriched.csv"

TRACK_FILE = "processed_models.txt"


# ------------------------------------------
# Load already processed models (resume)
# ------------------------------------------

def load_processed():

    if os.path.exists(TRACK_FILE):

        with open(TRACK_FILE) as f:

            return set(line.strip() for line in f)

    return set()


def mark_processed(model_id):

    with open(TRACK_FILE,"a") as f:

        f.write(model_id + "\n")


# ------------------------------------------
# Extract base model from model card
# ------------------------------------------

def extract_base_model(card_text):

    if not isinstance(card_text,str):

        return ""

    match = re.search(r'base_model:\s*(.*)', card_text)

    if match:

        return match.group(1).strip()

    return ""


# ------------------------------------------
# Setup output CSV
# ------------------------------------------

if not os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE,"w",newline='',encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "model_id",
            "tasks",
            "creator",
            "model_name",
            "tags",
            "base_model",
            "downloads",
            "created_at"
        ])


processed = load_processed()


df = pd.read_csv(INPUT_FILE)


print("Starting metadata enrichment...")


for _, row in tqdm(df.iterrows(), total=len(df)):

    creator = row["creator"]

    model_name = row["model_name"]

    task = row["task"]

    model_id = f"{creator}/{model_name}"


    if model_id in processed:

        continue


    try:

        info = api.model_info(model_id)

    except:

        continue


    tags = info.tags if info.tags else []

    downloads = info.downloads if info.downloads else 0

    created_at = ""

    if info.created_at:

        created_at = info.created_at.isoformat()


    base_model = ""

    if hasattr(info,"base_models") and info.base_models:

        base_model = ",".join(info.base_models)

    else:

        base_model = extract_base_model(row["model_card"])


    with open(OUTPUT_FILE,"a",newline='',encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            model_id,
            task,
            creator,
            model_name,
            ",".join(tags),
            base_model,
            downloads,
            created_at
        ])


    mark_processed(model_id)


    time.sleep(0.1)