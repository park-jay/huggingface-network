"""
RQ5 - Transformation Type Distribution Across Lineage Depth
============================================================
Does the type of transformation change predictably as you move
deeper into the lineage?

Hypothesis: Early generations (depth 1-2) tend to be fine-tunes and
adapters. Deeper generations (depth 3+) tend to be quantizations of
already fine-tuned models — a "refinement funnel."

Outputs:
  - rq5_depth_transformation_matrix.csv : depth × transformation counts
  - rq5_depth_transformation_pct.csv    : same, as percentages
  - rq5_per_family_depth.csv            : breakdown per model family
  - rq5_clustering_top100.csv           : clustering coefficient for top 100 nodes
  - rq5_summary.txt                     : key statistics + hypothesis test
"""

import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats

EDGES_FILE = "edges.csv"
NODES_FILE = "nodes.csv"
TOP_N_BASE = 100    # base models to run BFS from

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

if "Id" in nodes.columns and "model_id" not in nodes.columns:
    nodes = nodes.rename(columns={"Id": "model_id"})
G = nx.from_pandas_edgelist(
    edges, source="Source", target="Target",
    edge_attr=["Transformation", "Confidence"],
    create_using=nx.DiGraph()
)

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
nodes["downloads"] = pd.to_numeric(nodes["downloads"], errors="coerce").fillna(0)

# ── Identify base models ──────────────────────────────────────────────────────
base_models = [n for n, d in in_deg.items() if d == 0]
top_base    = sorted(base_models, key=lambda n: out_deg.get(n, 0), reverse=True)[:TOP_N_BASE]
print(f"Running BFS from top {TOP_N_BASE} base models...")

# ── Assign depth to every reachable node ─────────────────────────────────────
node_depth = {}   # node → min depth from any base model
for bm in top_base:
    try:
        lengths = nx.single_source_shortest_path_length(G, bm)
        for node, depth in lengths.items():
            if node not in node_depth or depth < node_depth[node]:
                node_depth[node] = depth
    except Exception:
        continue

print(f"Nodes assigned a depth: {len(node_depth):,}")

# ── Build depth × transformation matrix ──────────────────────────────────────
print("\n=== DEPTH × TRANSFORMATION MATRIX ===")
rows = []
for _, edge in edges.iterrows():
    target_depth = node_depth.get(edge["Target"])
    if target_depth is None or target_depth == 0:
        continue
    rows.append({
        "depth":          target_depth,
        "transformation": edge["Transformation"],
        "confidence":     edge["Confidence"],
        "source":         edge["Source"],
        "target":         edge["Target"],
    })

depth_edge_df = pd.DataFrame(rows)
print(f"Edges with depth assigned: {len(depth_edge_df):,}")

# Count matrix
matrix = depth_edge_df.groupby(["depth","transformation"]).size().unstack(fill_value=0)
print("\nRaw counts:")
print(matrix)
matrix.to_csv("rq5_depth_transformation_matrix.csv")

# Percentage matrix (row-normalised)
matrix_pct = matrix.div(matrix.sum(axis=1), axis=0).round(4) * 100
print("\nRow-normalised % (each row sums to 100%):")
print(matrix_pct.round(1))
matrix_pct.to_csv("rq5_depth_transformation_pct.csv")

print("\nSaved rq5_depth_transformation_matrix.csv")
print("Saved rq5_depth_transformation_pct.csv")

# ── Hypothesis test ───────────────────────────────────────────────────────────
print("\n=== HYPOTHESIS TEST: refinement funnel ===")
# H0: transformation type distribution is independent of depth
# Test: chi-squared on depth (1,2,3+) × transformation type

depth_edge_df["depth_group"] = depth_edge_df["depth"].apply(
    lambda d: "1" if d == 1 else ("2" if d == 2 else "3+")
)
contingency = pd.crosstab(
    depth_edge_df["depth_group"],
    depth_edge_df["transformation"]
)
print("Contingency table (depth group × transformation):")
print(contingency)

chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency)
print(f"\nChi-squared: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p:.6f}")
if p < 0.05:
    print("RESULT: Significant — transformation type DOES change with depth (reject H0)")
else:
    print("RESULT: Not significant — no clear depth-transformation relationship")

# ── Per-family depth breakdown ────────────────────────────────────────────────
print("\n=== TRANSFORMATION BY DEPTH PER FAMILY ===")
def get_family(model_id):
    mid = str(model_id).lower()
    if "llama"    in mid: return "LLaMA"
    if "qwen"     in mid: return "Qwen"
    if "mistral"  in mid: return "Mistral"
    if "gpt"      in mid: return "GPT"
    if "deepseek" in mid: return "DeepSeek"
    if "gemma"    in mid: return "Gemma"
    if "phi"      in mid: return "Phi"
    return "Other"

depth_edge_df["source_family"] = depth_edge_df["source"].apply(get_family)
family_depth = depth_edge_df.groupby(
    ["source_family","depth","transformation"]
).size().reset_index(name="count")
family_depth.to_csv("rq5_per_family_depth.csv", index=False)
print(family_depth[family_depth["source_family"].isin(["LLaMA","Qwen","Mistral"])]\
    .head(30).to_string(index=False))
print("Saved rq5_per_family_depth.csv")

# ── Clustering coefficient for top 100 nodes ─────────────────────────────────
print("\n=== CLUSTERING COEFFICIENT (top 100 nodes by degree) ===")

# Convert to undirected for clustering
G_undirected = G.to_undirected()

# Remove self-loops
G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

# Top 100 by total degree
all_degrees  = dict(G_undirected.degree())
top100_nodes = sorted(all_degrees, key=all_degrees.get, reverse=True)[:100]

cc_rows = []
for node in top100_nodes:
    cc     = nx.clustering(G_undirected, node)
    deg    = all_degrees[node]
    in_d   = in_deg.get(node, 0)
    out_d  = out_deg.get(node, 0)
    family = get_family(node)

    meta = nodes[nodes["model_id"] == node]
    downloads = int(meta["downloads"].values[0]) if len(meta) > 0 else 0

    cc_rows.append({
        "model_id":             node,
        "family":               family,
        "clustering_coefficient": round(cc, 4),
        "total_degree":         deg,
        "in_degree":            in_d,
        "out_degree":           out_d,
        "downloads":            downloads,
    })

cc_df = pd.DataFrame(cc_rows).sort_values("clustering_coefficient", ascending=False)
cc_df.to_csv("rq5_clustering_top100.csv", index=False)

print(f"\nTop 20 nodes by clustering coefficient:")
print(cc_df.head(20).to_string(index=False))
print(f"\nBottom 10 (most star-like / spoke patterns):")
print(cc_df.tail(10).to_string(index=False))
print(f"\nAvg clustering coefficient (top 100): {cc_df['clustering_coefficient'].mean():.4f}")
print(f"Saved rq5_clustering_top100.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
RQ5 SUMMARY STATISTICS
=======================
Edges with depth assigned:  {len(depth_edge_df):,}

Hypothesis test (chi-squared):
  Chi2 = {chi2:.2f}, df = {dof}, p = {p:.6f}
  {"REJECT H0 — transformation type changes significantly with depth" if p < 0.05 else "FAIL TO REJECT H0"}

Depth × Transformation counts:
{matrix.to_string()}

Depth × Transformation percentages:
{matrix_pct.round(1).to_string()}

Clustering coefficient (top 100 nodes):
  Average:  {cc_df['clustering_coefficient'].mean():.4f}
  Max:      {cc_df['clustering_coefficient'].max():.4f}
  Min:      {cc_df['clustering_coefficient'].min():.4f}
  Median:   {cc_df['clustering_coefficient'].median():.4f}

Top 10 by clustering coefficient:
{cc_df.head(10)[['model_id','family','clustering_coefficient','in_degree']].to_string(index=False)}
"""
with open("rq5_summary.txt", "w") as f:
    f.write(summary)
print(summary)
print("Saved rq5_summary.txt")