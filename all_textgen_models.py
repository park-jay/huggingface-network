from huggingface_hub import HfApi
import pandas as pd
from tqdm import tqdm

api = HfApi()

print("Fetching all text-generation models from Hugging Face...")

models = api.list_models(
    filter="text-generation",
    full=True
)

rows = []

for model in tqdm(models):

    model_id = model.id

    creator, model_name = model_id.split("/",1)

    rows.append({
        "model_id": model_id,
        "tasks": model.pipeline_tag,
        "creator": creator,
        "model_name": model_name,
        "tags": ",".join(model.tags) if model.tags else "",
        "downloads": model.downloads,
        "created_at": model.created_at
    })

df = pd.DataFrame(rows)

df.to_csv("all_text_generation_models.csv", index=False)

print("Finished!")
print("Total models:", len(df))