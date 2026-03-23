import pandas as pd
import re

# ----------------------------
# Step 1: Load files
# ----------------------------
print("Loading files...")

# Original scraper output — has url and image_count
df_scraper = pd.read_csv("huggingface_models.csv")

# Enriched nodes — has model_id, tags, downloads, etc.
df_nodes = pd.read_csv("merged_dataset.csv")

# Edges
df_edges = pd.read_csv("edges.csv")

print(f"Scraper rows:       {len(df_scraper)}")
print(f"Nodes rows:         {len(df_nodes)}")
print(f"Edges rows:         {len(df_edges)}")
print(f"Scraper columns:    {df_scraper.columns.tolist()}")
print(f"Nodes columns:      {df_nodes.columns.tolist()}")


# ----------------------------
# Step 2: Build model_id in scraper data and merge
# ----------------------------
print("\nMerging url and image_count into nodes...")

# scraper.py saves creator and model_name separately
df_scraper["model_id"] = df_scraper["creator"] + "/" + df_scraper["model_name"]

# Keep only what we need from scraper
scraper_extra = df_scraper[["model_id", "url", "image_count"]].drop_duplicates(subset="model_id")

# Merge into nodes
df_nodes = df_nodes.merge(scraper_extra, on="model_id", how="left")

print(f"URL coverage:          {df_nodes['url'].notna().sum()} / {len(df_nodes)} ({df_nodes['url'].notna().mean()*100:.1f}%)")
print(f"Image count coverage:  {df_nodes['image_count'].notna().sum()} / {len(df_nodes)} ({df_nodes['image_count'].notna().mean()*100:.1f}%)")


# ----------------------------
# Step 3: URL validation
# ----------------------------
print("\n" + "=" * 60)
print("URL ANALYSIS")
print("=" * 60)

# Validate URL format: should be https://huggingface.co/creator/model
hf_pattern = re.compile(r'^https://huggingface\.co/[\w\-\.]+/[\w\-\.]+$')

df_nodes["url_valid"] = df_nodes["url"].apply(
    lambda u: bool(hf_pattern.match(str(u))) if pd.notna(u) else False
)

valid_urls   = df_nodes["url_valid"].sum()
invalid_urls = df_nodes["url"].notna().sum() - valid_urls
missing_urls = df_nodes["url"].isna().sum()

print(f"Valid HuggingFace URLs:    {valid_urls} ({valid_urls/len(df_nodes)*100:.1f}%)")
print(f"Invalid/unexpected URLs:   {invalid_urls}")
print(f"Missing URLs:              {missing_urls}")

# Show any invalid ones
invalid_sample = df_nodes[df_nodes["url"].notna() & ~df_nodes["url_valid"]]["url"].head(10)
if len(invalid_sample) > 0:
    print(f"\nSample invalid URLs:")
    for u in invalid_sample:
        print(f"  {u}")

# Cross-check: does URL match model_id?
def url_matches_id(row):
    if pd.isna(row["url"]):
        return None
    expected = f"https://huggingface.co/{row['model_id']}"
    return str(row["url"]).strip() == expected

df_nodes["url_matches_id"] = df_nodes.apply(url_matches_id, axis=1)
mismatches = df_nodes[df_nodes["url_matches_id"] == False]
print(f"\nURL ↔ model_id mismatches: {len(mismatches)}")
if len(mismatches) > 0:
    print("Sample mismatches:")
    for _, row in mismatches.head(5).iterrows():
        print(f"  model_id: {row['model_id']}")
        print(f"  url:      {row['url']}")


# ----------------------------
# Step 4: Image count analysis
# ----------------------------
print("\n" + "=" * 60)
print("IMAGE COUNT ANALYSIS")
print("=" * 60)

df_nodes["image_count"] = pd.to_numeric(df_nodes["image_count"], errors="coerce").fillna(0).astype(int)

total      = len(df_nodes)
no_images  = (df_nodes["image_count"] == 0).sum()
has_images = (df_nodes["image_count"] > 0).sum()
rich_docs  = (df_nodes["image_count"] >= 3).sum()

print(f"Models with 0 images:      {no_images} ({no_images/total*100:.1f}%)")
print(f"Models with 1+ images:     {has_images} ({has_images/total*100:.1f}%)")
print(f"Models with 3+ images:     {rich_docs} ({rich_docs/total*100:.1f}%) ← well-documented")
print(f"Max image count:           {df_nodes['image_count'].max()}")
print(f"Mean (models with images): {df_nodes[df_nodes['image_count']>0]['image_count'].mean():.1f}")

print(f"\nImage count distribution:")
for threshold, label in [(0,"0"),(1,"1"),(2,"2"),(3,"3-5"),(6,"6-10"),(11,"11+")]:
    if label == "3-5":
        count = ((df_nodes["image_count"] >= 3) & (df_nodes["image_count"] <= 5)).sum()
    elif label == "6-10":
        count = ((df_nodes["image_count"] >= 6) & (df_nodes["image_count"] <= 10)).sum()
    elif label == "11+":
        count = (df_nodes["image_count"] >= 11).sum()
    else:
        count = (df_nodes["image_count"] == threshold).sum()
    bar = "█" * min(int(count / total * 60), 60)
    print(f"  {label:>4} images: {bar} {count}")


# ----------------------------
# Step 5: Are hub (parent) models better documented?
# ----------------------------
print("\n" + "=" * 60)
print("DOCUMENTATION VS LINEAGE ROLE")
print("=" * 60)

parent_ids = set(df_edges["Source"].tolist())
child_ids  = set(df_edges["Target"].tolist())
leaf_ids   = child_ids - parent_ids   # has children but is not a parent
root_ids   = parent_ids - child_ids   # is a parent but has no known parent

df_nodes["lineage_role"] = "isolated"
df_nodes.loc[df_nodes["model_id"].isin(parent_ids) & ~df_nodes["model_id"].isin(child_ids), "lineage_role"] = "root"
df_nodes.loc[df_nodes["model_id"].isin(child_ids) & ~df_nodes["model_id"].isin(parent_ids), "lineage_role"] = "leaf"
df_nodes.loc[df_nodes["model_id"].isin(parent_ids) & df_nodes["model_id"].isin(child_ids), "lineage_role"] = "intermediate"

print("\nAverage image count by lineage role:")
role_stats = df_nodes.groupby("lineage_role")["image_count"].agg(["mean","median","count"])
print(role_stats.round(2))

print("\nAverage downloads by lineage role:")
df_nodes["downloads"] = pd.to_numeric(df_nodes["downloads"], errors="coerce").fillna(0)
dl_stats = df_nodes.groupby("lineage_role")["downloads"].agg(["mean","median","count"])
print(dl_stats.round(0))


# ----------------------------
# Step 6: Save enriched nodes
# ----------------------------
OUTPUT = "nodes_enriched.csv"
df_nodes.to_csv(OUTPUT, index=False)
print(f"\nSaved enriched nodes to {OUTPUT}")
print(f"Final columns: {df_nodes.columns.tolist()}")