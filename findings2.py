"""
RQ2 - Multi-Parent Reuse and Knowledge Diversity
=================================================
How prevalent is multi-parent derivation in the HuggingFace ecosystem,
and which combinations of base models are most frequently reused together?

Outputs:
  - rq2_multi_parent_nodes.csv   : all nodes with in-degree >= 2
  - rq2_parent_pairs.csv         : most common base model co-occurrence pairs
  - rq2_family_combinations.csv  : which model families get combined most
  - rq2_summary.txt              : key statistics for the paper
"""

import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter

EDGES_FILE = "edges.csv"
NODES_FILE = "all_text_generation_models.csv"

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

# Handle both Id (all_text_generation_models.csv) and model_id (merged_dataset.csv)
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

# ── Model family helper ───────────────────────────────────────────────────────
def extract_family(model_id):
    mid = str(model_id).lower()
    if "llama"    in mid: return "LLaMA"
    if "qwen"     in mid: return "Qwen"
    if "mistral"  in mid: return "Mistral"
    if "gpt"      in mid: return "GPT"
    if "deepseek" in mid: return "DeepSeek"
    if "gemma"    in mid: return "Gemma"
    if "falcon"   in mid: return "Falcon"
    if "bloom"    in mid: return "BLOOM"
    if "phi"      in mid: return "Phi"
    if "opt"      in mid: return "OPT"
    return "Other"

# ── Find all multi-parent nodes ───────────────────────────────────────────────
print("\n=== MULTI-PARENT PREVALENCE ===")
total_nodes   = G.number_of_nodes()
single_parent = sum(1 for d in in_deg.values() if d == 1)
multi_2plus   = sum(1 for d in in_deg.values() if d >= 2)
multi_3plus   = sum(1 for d in in_deg.values() if d >= 3)
multi_5plus   = sum(1 for d in in_deg.values() if d >= 5)
multi_10plus  = sum(1 for d in in_deg.values() if d >= 10)
max_in        = max(in_deg.values())

print(f"Total nodes:             {total_nodes:,}")
print(f"Single-parent (deg=1):   {single_parent:,} ({single_parent/total_nodes*100:.1f}%)")
print(f"Multi-parent (deg>=2):   {multi_2plus:,} ({multi_2plus/total_nodes*100:.1f}%)")
print(f"Multi-parent (deg>=3):   {multi_3plus:,} ({multi_3plus/total_nodes*100:.1f}%)")
print(f"Multi-parent (deg>=5):   {multi_5plus:,} ({multi_5plus/total_nodes*100:.1f}%)")
print(f"Multi-parent (deg>=10):  {multi_10plus:,} ({multi_10plus/total_nodes*100:.1f}%)")
print(f"Max in-degree:           {max_in}")

# Pre-build lookup dicts ONCE — avoids 12k x 341k row scans in the loop
print("Building lookup tables...")
downloads_map   = nodes.set_index("model_id")["downloads"].to_dict()
creator_map     = nodes.set_index("model_id")["creator"].to_dict()
edge_transforms = edges.groupby("Target")["Transformation"].apply(list).to_dict()

# ── Build multi-parent node table ─────────────────────────────────────────────
print("Building multi-parent node table...")
multi_rows = []
for node, deg in in_deg.items():
    if deg >= 2:
        parents      = list(G.predecessors(node))
        parent_fams  = [extract_family(p) for p in parents]
        unique_fams  = list(set(parent_fams))
        is_cross_fam = len(unique_fams) > 1

        downloads  = int(downloads_map.get(node, 0))
        creator    = creator_map.get(node, "unknown")
        transforms = pd.Series(edge_transforms.get(node, [])).value_counts().to_dict()

        multi_rows.append({
            "model_id":          node,
            "creator":           creator,
            "in_degree":         deg,
            "out_degree":        out_deg.get(node, 0),
            "downloads":         downloads,
            "parents":           "|".join(parents[:10]),
            "parent_families":   "|".join(unique_fams),
            "n_unique_families": len(unique_fams),
            "is_cross_family":   is_cross_fam,
            "transformations":   str(transforms),
        })

multi_df = pd.DataFrame(multi_rows).sort_values("in_degree", ascending=False)
multi_df.to_csv("rq2_multi_parent_nodes.csv", index=False)
print(f"\nSaved rq2_multi_parent_nodes.csv ({len(multi_df):,} rows)")

print(f"\nCross-family merges:  {multi_df['is_cross_family'].sum():,} ({multi_df['is_cross_family'].mean()*100:.1f}% of multi-parent)")
print(f"Same-family merges:   {(~multi_df['is_cross_family']).sum():,}")

# ── Parent pair co-occurrence ─────────────────────────────────────────────────
print("\n=== MOST COMMON BASE MODEL PAIRS ===")
pair_counter = Counter()
family_pair_counter = Counter()

for _, row in multi_df.iterrows():
    parents = [p for p in row["parents"].split("|") if p]
    if len(parents) >= 2:
        for pair in combinations(sorted(parents), 2):
            pair_counter[pair] += 1

        fams = [extract_family(p) for p in parents]
        for fpair in combinations(sorted(set(fams)), 2):
            family_pair_counter[fpair] += 1

top_pairs = pd.DataFrame([
    {"parent_a": a, "parent_b": b, "co_occurrence_count": c,
     "family_a": extract_family(a), "family_b": extract_family(b)}
    for (a, b), c in pair_counter.most_common(50)
])
if len(top_pairs) > 0:
    top_pairs.to_csv("rq2_parent_pairs.csv", index=False)
    print(top_pairs.head(15).to_string(index=False))
    print(f"\nSaved rq2_parent_pairs.csv")

# ── Family combination breakdown ─────────────────────────────────────────────
print("\n=== FAMILY COMBINATIONS ===")
fam_pairs = pd.DataFrame([
    {"family_a": a, "family_b": b, "count": c}
    for (a, b), c in family_pair_counter.most_common(30)
])
if len(fam_pairs) > 0:
    fam_pairs.to_csv("rq2_family_combinations.csv", index=False)
    print(fam_pairs.head(15).to_string(index=False))
    print(f"\nSaved rq2_family_combinations.csv")

# ── Downloads comparison: multi vs single parent ──────────────────────────────
print("\n=== DOWNLOADS: MULTI-PARENT vs SINGLE-PARENT ===")
multi_ids  = set(multi_df["model_id"].tolist())
single_ids = set(n for n, d in in_deg.items() if d == 1)

multi_nodes  = nodes[nodes["model_id"].isin(multi_ids)]
single_nodes = nodes[nodes["model_id"].isin(single_ids)]

print(f"Multi-parent  avg downloads: {multi_nodes['downloads'].mean():.0f}")
print(f"Multi-parent  med downloads: {multi_nodes['downloads'].median():.0f}")
print(f"Single-parent avg downloads: {single_nodes['downloads'].mean():.0f}")
print(f"Single-parent med downloads: {single_nodes['downloads'].median():.0f}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
RQ2 SUMMARY STATISTICS
=======================
Total nodes in graph:              {total_nodes:,}
Multi-parent nodes (in-deg >= 2):  {multi_2plus:,} ({multi_2plus/total_nodes*100:.2f}%)
Multi-parent nodes (in-deg >= 3):  {multi_3plus:,}
Multi-parent nodes (in-deg >= 5):  {multi_5plus:,}
Multi-parent nodes (in-deg >= 10): {multi_10plus:,}
Maximum in-degree:                 {max_in}

Cross-family merges: {multi_df['is_cross_family'].sum():,} ({multi_df['is_cross_family'].mean()*100:.1f}% of multi-parent nodes)
Same-family merges:  {(~multi_df['is_cross_family']).sum():,}

Downloads comparison:
  Multi-parent  avg: {multi_nodes['downloads'].mean():.0f}   median: {multi_nodes['downloads'].median():.0f}
  Single-parent avg: {single_nodes['downloads'].mean():.0f}   median: {single_nodes['downloads'].median():.0f}

Top 10 most-merged base model pairs:
{top_pairs.head(10).to_string(index=False) if len(top_pairs) > 0 else 'N/A'}

Top family combinations:
{fam_pairs.head(10).to_string(index=False) if len(fam_pairs) > 0 else 'N/A'}
"""
with open("rq2_summary.txt", "w") as f:
    f.write(summary)
print(summary)
print("Saved rq2_summary.txt")