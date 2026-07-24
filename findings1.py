"""
RQ1 - Model Genealogy and Base Model Identification
=====================================================
How can base models be formally defined in the HuggingFace text-generation
ecosystem, and what structural properties distinguish them from derivative models?

Definition: A base model = any node with in-degree = 0 in the lineage graph.

Outputs:
  - rq1_base_models.csv          : all identified base models with stats
  - rq1_structural_comparison.csv: base vs derivative property comparison
  - rq1_creator_breakdown.csv    : which orgs produce the most base models
  - rq1_summary.txt              : key statistics for the paper
"""

import pandas as pd
import networkx as nx
import re
from collections import Counter

EDGES_FILE = "edges.csv"
NODES_FILE = "merged_dataset.csv"   # full 341k nodes
MIN_DOWNLOADS = 0                   # set to e.g. 50 to filter out test accounts

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)
print(f"Edges: {len(edges):,}  |  Nodes: {len(nodes):,}")

# Build directed graph from edges
G = nx.from_pandas_edgelist(
    edges, source="Source", target="Target",
    edge_attr=["Transformation", "Confidence"],
    create_using=nx.DiGraph()
)
print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())

#Identify base models as those with in-degree = 0
base_model_ids = [n for n, d in in_deg.items() if d == 0]
print(f"\nBase models (in-degree=0): {len(base_model_ids):,}")

# Assign lineage role to every node  
parent_set = set(edges["Source"].tolist())
child_set  = set(edges["Target"].tolist())

def assign_role(model_id):
    is_parent = model_id in parent_set
    is_child  = model_id in child_set
    if is_parent and not is_child:
        return "root"
    elif is_parent and is_child:
        return "intermediate"
    elif not is_parent and is_child:
        return "leaf"
    else:
        return "isolated"

nodes["lineage_role"]   = nodes["model_id"].apply(assign_role)
nodes["in_degree"]      = nodes["model_id"].map(lambda x: in_deg.get(x, 0))
nodes["out_degree"]     = nodes["model_id"].map(lambda x: out_deg.get(x, 0))
nodes["downloads"]      = pd.to_numeric(nodes["downloads"], errors="coerce").fillna(0)
nodes["created_at"]     = pd.to_datetime(nodes["created_at"], utc=True, errors="coerce")

# Base model table
base_df = nodes[nodes["lineage_role"] == "root"].copy()

if MIN_DOWNLOADS > 0:
    base_df = base_df[base_df["downloads"] >= MIN_DOWNLOADS]
    print(f"After download filter (>={MIN_DOWNLOADS}): {len(base_df):,} base models")

base_df = base_df.sort_values("out_degree", ascending=False)

# Tag model family from model_id
def extract_family(model_id):
    mid = str(model_id).lower()
    if "llama"     in mid: return "LLaMA"
    if "qwen"      in mid: return "Qwen"
    if "mistral"   in mid: return "Mistral"
    if "gpt"       in mid: return "GPT"
    if "deepseek"  in mid: return "DeepSeek"
    if "gemma"     in mid: return "Gemma"
    if "falcon"    in mid: return "Falcon"
    if "bloom"     in mid: return "BLOOM"
    if "phi"       in mid: return "Phi"
    if "opt"       in mid: return "OPT"
    if "t5"        in mid: return "T5"
    return "Other"

base_df["model_family"] = base_df["model_id"].apply(extract_family)

base_df[[
    "model_id", "creator", "model_family",
    "out_degree", "downloads", "created_at", "tags"
]].to_csv("rq1_base_models.csv", index=False)
print(f"Saved rq1_base_models.csv ({len(base_df):,} rows)")

#Structural comparison: base vs derivative models
print("\n=== STRUCTURAL COMPARISON ===")
roles = ["root", "intermediate", "leaf", "isolated"]
comparison = nodes.groupby("lineage_role").agg(
    count       = ("model_id",    "count"),
    avg_downloads = ("downloads", "mean"),
    med_downloads = ("downloads", "median"),
    avg_out_degree = ("out_degree", "mean"),
    avg_in_degree  = ("in_degree",  "mean"),
).round(2)
print(comparison)
comparison.to_csv("rq1_structural_comparison.csv")
print("Saved rq1_structural_comparison.csv")

#Creator breakdown
print("\n=== TOP BASE MODEL CREATORS ===")
creator_counts = base_df.groupby("creator").agg(
    base_model_count = ("model_id",    "count"),
    total_derivatives = ("out_degree", "sum"),
    avg_downloads    = ("downloads",   "mean"),
).sort_values("base_model_count", ascending=False)
print(creator_counts.head(20))
creator_counts.to_csv("rq1_creator_breakdown.csv")
print("Saved rq1_creator_breakdown.csv")

#Model family breakdown
print("\n=== BASE MODELS BY FAMILY ===")
family_stats = base_df.groupby("model_family").agg(
    count      = ("model_id",    "count"),
    total_deriv = ("out_degree", "sum"),
    avg_dl     = ("downloads",   "mean"),
).sort_values("total_deriv", ascending=False)
print(family_stats)

#Summary stats
total_nodes     = G.number_of_nodes()
total_base      = len(base_df)
total_deriv     = len(nodes[nodes["lineage_role"].isin(["leaf","intermediate"])])
top10           = base_df.head(10)[["model_id","out_degree","downloads"]].to_string()

summary = f"""
RQ1 SUMMARY STATISTICS
=======================
Total nodes in graph:          {total_nodes:,}
Total base models (in-deg=0):  {total_base:,}  ({total_base/total_nodes*100:.1f}% of graph)
Total derivative models:       {total_deriv:,}

Top 10 base models by derivative count:
{top10}

Structural comparison (base vs derivative):
{comparison.to_string()}

Base models by family:
{family_stats.to_string()}
"""
with open("rq1_summary.txt", "w") as f:
    f.write(summary)
print(summary)
print("Saved rq1_summary.txt")