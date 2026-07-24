"""
RQ2 v2 - Multi-Parent Reuse with 2-Hop Family Resolution
=========================================================
Fix for the 48.5% unresolved problem:
  Parents like TheBloke/Llama-2-7B-GGUF don't match "llama" in their name,
  but their own base_model tag points to meta-llama/Llama-2-7B.
  We follow the chain up to 2 hops to resolve the true ancestor family.

Outputs:
  - rq2_multi_parent_nodes_v2.csv
  - rq2_family_pairs_v2.csv
  - rq2_indegree_breakdown_v2.csv
  - rq2_summary_v2.txt
"""

import pandas as pd
import networkx as nx
import re
from itertools import combinations
from collections import Counter
import statistics

EDGES_FILE = "edges.csv"
NODES_FILE = "all_text_generation_models.csv"

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

if "Id" in nodes.columns and "model_id" not in nodes.columns:
    nodes = nodes.rename(columns={"Id": "model_id"})

nodes["downloads"] = pd.to_numeric(nodes["downloads"], errors="coerce").fillna(0)

G = nx.from_pandas_edgelist(
    edges, source="Source", target="Target",
    edge_attr=["Transformation", "Confidence"],
    create_using=nx.DiGraph()
)

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())

# ── Pre-build fast lookup maps ────────────────────────────────────────────────
downloads_map  = nodes.set_index("model_id")["downloads"].to_dict()
creator_map    = nodes.set_index("model_id")["creator"].to_dict()
tags_map       = nodes.set_index("model_id")["tags"].to_dict()
edge_transforms = edges.groupby("Target")["Transformation"].apply(list).to_dict()

# ── Family patterns ───────────────────────────────────────────────────────────
FAMILY_PATTERNS = [
    ("LLaMA",    [r"llama"]),
    ("Qwen",     [r"\bqwen\b"]),
    ("Mistral",  [r"mistral"]),
    ("GPT",      [r"\bgpt\b", r"openai"]),
    ("DeepSeek", [r"deepseek"]),
    ("Gemma",    [r"gemma"]),
    ("Falcon",   [r"falcon"]),
    ("BLOOM",    [r"bloom"]),
    ("Phi",      [r"\bphi\b"]),
    ("OPT",      [r"\bopt\b"]),
    ("T5",       [r"\bt5\b"]),
    ("Yi",       [r"\byi\b"]),
    ("Nemotron", [r"nemotron"]),
]

def match_family_string(text):
    if not text or pd.isna(text):
        return None
    s = str(text).lower()
    for fam, pats in FAMILY_PATTERNS:
        for pat in pats:
            if re.search(pat, s):
                return fam
    return None

def extract_base_model_from_tags(tags_str):
    """Extract first base_model:xxx value from tags string."""
    if not tags_str or pd.isna(tags_str):
        return None
    # prioritise base_model:finetune: or base_model: without sub-type
    matches = re.findall(r"base_model:(?:finetune:|quantized:|adapter:|merge:)?([^,\s]+)", str(tags_str))
    return matches[0].strip() if matches else None

def resolve_family(model_id, depth=0, visited=None):
    """
    2-hop family resolution:
      1. Try to classify model_id by its own name
      2. Try to classify by its base_model tag
      3. Follow that base_model's own base_model tag (1 more hop)
    Returns (family, method_used)
    """
    if visited is None:
        visited = set()
    if model_id in visited or depth > 2:
        return None, None
    visited.add(model_id)

    # Hop 0: try model name directly
    fam = match_family_string(model_id)
    if fam:
        return fam, f"model_id_hop{depth}"

    # Hop 1: try base_model tag
    tags = tags_map.get(model_id, "")
    base = extract_base_model_from_tags(tags)
    if base:
        fam = match_family_string(base)
        if fam:
            return fam, f"base_model_tag_hop{depth}"
        # Hop 2: follow the base model's own tags
        if depth < 2:
            return resolve_family(base, depth + 1, visited)

    return None, None

# ── Pre-resolve all nodes in graph ────────────────────────────────────────────
print("Resolving family for all nodes (2-hop)...")
family_cache = {}
for node in G.nodes():
    fam, method = resolve_family(node)
    family_cache[node] = fam

resolved_count = sum(1 for v in family_cache.values() if v is not None)
print(f"Resolved: {resolved_count:,} / {len(family_cache):,} ({resolved_count/len(family_cache)*100:.1f}%)")

# Family distribution
from collections import Counter
fam_dist = Counter(v for v in family_cache.values() if v)
print("\nFamily distribution in graph:")
for fam, cnt in fam_dist.most_common():
    print(f"  {fam}: {cnt:,}")

# ── Multi-parent analysis ─────────────────────────────────────────────────────
print("\nBuilding multi-parent analysis...")

multi_rows   = []
pair_counter = Counter()
fam_combo_counter_by_deg = {2: Counter(), 3: Counter(), 4: Counter(), 5: Counter()}

for node, deg in in_deg.items():
    if deg < 2:
        continue

    parents     = list(G.predecessors(node))
    parent_fams = [family_cache.get(p) for p in parents]
    resolved    = [f for f in parent_fams if f is not None]
    unique_fams = sorted(set(resolved))

    if len(unique_fams) == 0:
        status = "unresolved"
    elif len(unique_fams) == 1:
        status = "same_family"
    else:
        status = "cross_family"

    downloads = int(downloads_map.get(node, 0))
    creator   = creator_map.get(node, "unknown")
    transforms = pd.Series(edge_transforms.get(node, [])).value_counts().to_dict()
    node_family = family_cache.get(node)

    multi_rows.append({
        "model_id":            node,
        "creator":             creator,
        "node_family":         node_family,
        "in_degree":           deg,
        "out_degree":          out_deg.get(node, 0),
        "downloads":           downloads,
        "parents":             "|".join(parents),
        "parent_families":     "|".join(unique_fams),
        "n_parent_families":   len(unique_fams),
        "n_resolved_parents":  len(resolved),
        "family_status":       status,
        "transformations":     str(transforms),
    })

    # Pair co-occurrence
    if len(unique_fams) >= 2:
        for pair in combinations(unique_fams, 2):
            pair_counter[pair] += 1

    # Per-degree family combos
    exact_deg = deg if deg <= 5 else None
    if exact_deg and exact_deg >= 2:
        combo = tuple(unique_fams) if unique_fams else ("unresolved",)
        fam_combo_counter_by_deg[exact_deg][combo] += 1

multi_df = pd.DataFrame(multi_rows).sort_values("in_degree", ascending=False)
multi_df.to_csv("rq2_multi_parent_nodes_v2.csv", index=False)
print(f"Saved rq2_multi_parent_nodes_v2.csv ({len(multi_df):,} rows)")

# ── In-degree breakdown ───────────────────────────────────────────────────────
print("\n=== IN-DEGREE BREAKDOWN ===")
deg_rows = []
for exact_deg in [2, 3, 4, 5]:
    subset = multi_df[multi_df["in_degree"] == exact_deg]
    total  = len(subset)
    same   = (subset["family_status"] == "same_family").sum()
    cross  = (subset["family_status"] == "cross_family").sum()
    unres  = (subset["family_status"] == "unresolved").sum()
    resolved_total = same + cross
    print(f"in-degree=={exact_deg}: {total:,} nodes | "
          f"same={same:,} ({same/max(total,1)*100:.1f}%) | "
          f"cross={cross:,} ({cross/max(total,1)*100:.1f}%) | "
          f"unresolved={unres:,} ({unres/max(total,1)*100:.1f}%)")
    if resolved_total > 0:
        print(f"   Of resolved: same={same/resolved_total*100:.1f}% | cross={cross/resolved_total*100:.1f}%")

    deg_rows.append({
        "in_degree":      exact_deg,
        "total_nodes":    total,
        "same_family":    same,
        "cross_family":   cross,
        "unresolved":     unres,
        "same_pct":       round(same/max(total,1)*100, 1),
        "cross_pct":      round(cross/max(total,1)*100, 1),
        "unresolved_pct": round(unres/max(total,1)*100, 1),
        "same_of_resolved_pct":  round(same/max(resolved_total,1)*100, 1),
        "cross_of_resolved_pct": round(cross/max(resolved_total,1)*100, 1),
    })

    # Top family combos for this degree
    print(f"   Top 5 family combos:")
    for combo, cnt in fam_combo_counter_by_deg[exact_deg].most_common(5):
        print(f"     {' + '.join(combo)}: {cnt}")
    print()

indeg_df = pd.DataFrame(deg_rows)
indeg_df.to_csv("rq2_indegree_breakdown_v2.csv", index=False)
print("Saved rq2_indegree_breakdown_v2.csv")

# ── Family pair table ─────────────────────────────────────────────────────────
print("\n=== TOP FAMILY PAIRS (cross-family merges) ===")
pairs_df = pd.DataFrame([
    {"family_a": a, "family_b": b, "co_occurrence_count": c}
    for (a, b), c in pair_counter.most_common(30)
])
if len(pairs_df) > 0:
    print(pairs_df.head(15).to_string(index=False))
    pairs_df.to_csv("rq2_family_pairs_v2.csv", index=False)
    print("Saved rq2_family_pairs_v2.csv")

# ── Overall stats ─────────────────────────────────────────────────────────────
print("\n=== OVERALL MULTI-PARENT STATS ===")
total     = len(multi_df)
same_tot  = (multi_df["family_status"] == "same_family").sum()
cross_tot = (multi_df["family_status"] == "cross_family").sum()
unres_tot = (multi_df["family_status"] == "unresolved").sum()
res_tot   = same_tot + cross_tot

multi_degs = multi_df["in_degree"].tolist()
print(f"Total multi-parent nodes:       {total:,}")
print(f"Same-family:                    {same_tot:,} ({same_tot/total*100:.1f}%)")
print(f"Cross-family:                   {cross_tot:,} ({cross_tot/total*100:.1f}%)")
print(f"Unresolved:                     {unres_tot:,} ({unres_tot/total*100:.1f}%)")
if res_tot > 0:
    print(f"Of resolved — same:             {same_tot/res_tot*100:.1f}%")
    print(f"Of resolved — cross:            {cross_tot/res_tot*100:.1f}%")
print(f"Avg in-degree (multi-parent):   {sum(multi_degs)/len(multi_degs):.2f}")
print(f"Median in-degree:               {statistics.median(multi_degs):.1f}")
print(f"Max in-degree:                  {max(multi_degs)}")

# Downloads comparison
single_ids  = set(n for n, d in in_deg.items() if d == 1)
multi_ids_set = set(multi_df["model_id"])
single_nodes  = nodes[nodes["model_id"].isin(single_ids)]
multi_nodes_f = nodes[nodes["model_id"].isin(multi_ids_set)]
print(f"\nDownloads — multi-parent  avg: {multi_nodes_f['downloads'].mean():.0f}  median: {multi_nodes_f['downloads'].median():.0f}")
print(f"Downloads — single-parent avg: {single_nodes['downloads'].mean():.0f}  median: {single_nodes['downloads'].median():.0f}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
RQ2 v2 SUMMARY (with 2-hop family resolution)
=============================================
Total multi-parent nodes:    {total:,}
Same-family:                 {same_tot:,} ({same_tot/total*100:.1f}%)
Cross-family:                {cross_tot:,} ({cross_tot/total*100:.1f}%)
Unresolved:                  {unres_tot:,} ({unres_tot/total*100:.1f}%)
Of resolved — same:          {same_tot/max(res_tot,1)*100:.1f}%
Of resolved — cross:         {cross_tot/max(res_tot,1)*100:.1f}%

Avg in-degree (multi-parent): {sum(multi_degs)/len(multi_degs):.2f}
Max in-degree:               {max(multi_degs)}

In-degree breakdown:
{indeg_df.to_string(index=False)}

Top family pairs:
{pairs_df.head(10).to_string(index=False) if len(pairs_df) > 0 else 'None'}

Family distribution in graph:
{chr(10).join(f"  {fam}: {cnt:,}" for fam, cnt in fam_dist.most_common())}
"""
with open("rq2_summary_v2.txt", "w") as f:
    f.write(summary)
print(summary)
print("Saved rq2_summary_v2.txt")