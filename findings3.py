"""
RQ3 - Lineage Depth and Derivation Statistics
=============================================
How many generational levels does each base model's lineage span,
and what does the depth distribution reveal about long-term reuse?

Outputs:
  - rq3_base_model_depths.csv    : depth + descendant stats per base model
  - rq3_depth_distribution.csv   : how many models live at each depth level
  - rq3_summary.txt              : key statistics for the paper
"""

import pandas as pd
import networkx as nx
from collections import defaultdict

EDGES_FILE = "edges.csv"
NODES_FILE = "all_text_generation_models.csv"
TOP_N_BASE  = 100    # analyse top N base models by derivative count

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
print(f"Base models (in-degree=0): {len(base_models):,}")

# ── Connected components overview ─────────────────────────────────────────────
wcc = list(nx.weakly_connected_components(G))
print(f"\nWeakly connected components: {len(wcc):,}")
print(f"Largest component:           {max(len(c) for c in wcc):,} nodes")
print(f"Singleton components:        {sum(1 for c in wcc if len(c)==1):,}")

# ── BFS depth analysis for top base models ────────────────────────────────────
print(f"\nRunning BFS for top {TOP_N_BASE} base models...")

top_base = sorted(base_models, key=lambda n: out_deg.get(n, 0), reverse=True)[:TOP_N_BASE]

depth_rows = []
all_depth_counts = defaultdict(int)

for bm in top_base:
    try:
        lengths = nx.single_source_shortest_path_length(G, bm)
    except Exception:
        continue

    descendants      = len(lengths) - 1
    if descendants == 0:
        continue

    max_depth        = max(lengths.values())
    avg_depth        = sum(v for k, v in lengths.items() if k != bm) / descendants
    depth_dist       = defaultdict(int)

    for node, depth in lengths.items():
        if node != bm:
            depth_dist[depth] += 1
            all_depth_counts[depth] += 1

    # get metadata
    meta = nodes[nodes["model_id"] == bm]
    downloads  = int(meta["downloads"].values[0])  if len(meta) > 0 else 0
    creator    = meta["creator"].values[0]          if len(meta) > 0 else "unknown"
    created_at = meta["created_at"].values[0]       if len(meta) > 0 else None

    # model family
    mid = bm.lower()
    if "llama"    in mid: family = "LLaMA"
    elif "qwen"   in mid: family = "Qwen"
    elif "mistral"in mid: family = "Mistral"
    elif "gpt"    in mid: family = "GPT"
    elif "deepseek"in mid:family = "DeepSeek"
    elif "gemma"  in mid: family = "Gemma"
    elif "phi"    in mid: family = "Phi"
    else:                  family = "Other"

    depth_rows.append({
        "model_id":      bm,
        "creator":       creator,
        "family":        family,
        "downloads":     downloads,
        "created_at":    created_at,
        "direct_derivatives": out_deg.get(bm, 0),
        "total_descendants":  descendants,
        "max_depth":     max_depth,
        "avg_depth":     round(avg_depth, 2),
        "depth_1_count": depth_dist.get(1, 0),
        "depth_2_count": depth_dist.get(2, 0),
        "depth_3_count": depth_dist.get(3, 0),
        "depth_4plus_count": sum(v for k, v in depth_dist.items() if k >= 4),
    })

depth_df = pd.DataFrame(depth_rows).sort_values("max_depth", ascending=False)
depth_df.to_csv("rq3_base_model_depths.csv", index=False)
print(f"Saved rq3_base_model_depths.csv ({len(depth_df)} rows)")

# ── Print top results ─────────────────────────────────────────────────────────
print("\n=== TOP 20 DEEPEST LINEAGES ===")
cols = ["model_id","family","direct_derivatives","total_descendants","max_depth","avg_depth"]
print(depth_df[cols].head(20).to_string(index=False))

# ── Depth distribution across ALL descendants ─────────────────────────────────
print("\n=== GLOBAL DEPTH DISTRIBUTION (from top base models) ===")
dist_df = pd.DataFrame([
    {"depth": d, "model_count": c}
    for d, c in sorted(all_depth_counts.items())
])
dist_df["cumulative_pct"] = (dist_df["model_count"].cumsum() / dist_df["model_count"].sum() * 100).round(1)
print(dist_df.to_string(index=False))
dist_df.to_csv("rq3_depth_distribution.csv", index=False)
print("Saved rq3_depth_distribution.csv")

# ── Family-level depth stats ──────────────────────────────────────────────────
print("\n=== DEPTH STATS BY MODEL FAMILY ===")
family_stats = depth_df.groupby("family").agg(
    base_model_count = ("model_id",  "count"),
    avg_max_depth    = ("max_depth", "mean"),
    max_max_depth    = ("max_depth", "max"),
    avg_descendants  = ("total_descendants", "mean"),
    total_descendants= ("total_descendants", "sum"),
).round(2).sort_values("max_max_depth", ascending=False)
print(family_stats)

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
RQ3 SUMMARY STATISTICS
=======================
Total base models analysed:    {len(depth_df)}
Weakly connected components:   {len(wcc):,}
Largest component:             {max(len(c) for c in wcc):,} nodes

Depth statistics across top {TOP_N_BASE} base models:
  Max depth overall:    {depth_df['max_depth'].max()}
  Avg max depth:        {depth_df['max_depth'].mean():.2f}
  Median max depth:     {depth_df['max_depth'].median():.1f}

Top 10 deepest lineages:
{depth_df[['model_id','family','max_depth','total_descendants']].head(10).to_string(index=False)}

Family-level depth stats:
{family_stats.to_string()}

Global depth distribution:
{dist_df.to_string(index=False)}
"""
with open("rq3_summary.txt", "w") as f:
    f.write(summary)
print(summary)
print("Saved rq3_summary.txt")