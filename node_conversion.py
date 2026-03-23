import pandas as pd

df = pd.read_csv("all_text_generation_models.csv")

nodes = df.rename(columns={"model_id": "Id"})

nodes.to_csv("nodes.csv", index=False)

print("Nodes:", len(nodes))