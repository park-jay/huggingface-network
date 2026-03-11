import pandas as pd

df=pd.read_csv("huggingface_models.csv")
print(df["url"].duplicated().sum())