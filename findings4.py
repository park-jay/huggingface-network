"""
RQ4 - Temporal Evolution of Lineage
=====================================
How has the structure of the model lineage graph evolved over time?
Are newer base models being adopted faster, and is the ecosystem
becoming more or less diverse?

Outputs:
  - rq4_quarterly_stats.csv      : graph metrics sliced by quarter
  - rq4_adoption_speed.csv       : time from base model creation to first derivative
  - rq4_diversity_over_time.csv  : number of active base models per quarter
  - rq4_summary.txt              : key statistics for the paper
"""

import pandas as pd
import networkx as nx
from collections import defaultdict

EDGES_FILE = "edges.csv"
NODES_FILE = "all_text_generation_models.csv"

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

if "Id" in nodes.columns and "model_id" not in nodes.columns:
    nodes = nodes.rename(columns={"Id": "model_id"})

# ── Parse dates ───────────────────────────────────────────────────────────────
nodes["created_at"] = pd.to_datetime(nodes["created_at"], utc=True, errors="coerce")
nodes["downloads"]  = pd.to_numeric(nodes["downloads"], errors="coerce").fillna(0)
nodes["quarter"]    = nodes["created_at"].dt.to_period("Q")

# ── Join dates onto edges ─────────────────────────────────────────────────────
print("Joining dates onto edges...")
date_map = nodes.set_index("model_id")["created_at"].to_dict()

edges["source_date"] = edges["Source"].map(date_map)
edges["target_date"] = edges["Target"].map(date_map)
edges["target_quarter"] = pd.to_datetime(
    edges["target_date"], utc=True, errors="coerce"
).dt.to_period("Q")

dated_edges = edges.dropna(subset=["source_date","target_date"]).copy()
print(f"Edges with both dates: {len(dated_edges):,} / {len(edges):,}")

# ── Adoption speed ────────────────────────────────────────────────────────────
print("\n=== ADOPTION SPEED ===")
# Time from parent creation to child creation
dated_edges["source_dt"] = pd.to_datetime(dated_edges["source_date"], utc=True, errors="coerce")
dated_edges["target_dt"] = pd.to_datetime(dated_edges["target_date"], utc=True, errors="coerce")
dated_edges["days_to_derive"] = (
    dated_edges["target_dt"] - dated_edges["source_dt"]
).dt.days

valid_gaps = dated_edges[dated_edges["days_to_derive"] >= 0]
print(f"Valid temporal edges (child after parent): {len(valid_gaps):,}")
print(f"Avg days from parent to child: {valid_gaps['days_to_derive'].mean():.0f}")
print(f"Med days from parent to child: {valid_gaps['days_to_derive'].median():.0f}")
print(f"Min days:                      {valid_gaps['days_to_derive'].min()}")
print(f"Max days:                      {valid_gaps['days_to_derive'].max()}")

# ── Adoption speed — fully vectorized, no loop ───────────────────────────────
print("Calculating adoption speed (vectorized)...")

# Base model IDs = appear as Source but never as Target
source_set  = set(edges["Source"].unique())
target_set  = set(edges["Target"].unique())
base_id_set = source_set - target_set

# Filter to base model edges with valid dates
base_edges = valid_gaps[valid_gaps["Source"].isin(base_id_set)].copy()

# Vectorized: first derivative date and total count per base model
first_deriv_s = base_edges.groupby("Source")["target_dt"].min().rename("first_derivative_date")
total_deriv_s = base_edges.groupby("Source")["Target"].count().rename("total_derivatives")

# Base model creation dates
bm_dates = pd.to_datetime(pd.Series(date_map), utc=True, errors="coerce")

adoption_df = pd.concat([first_deriv_s, total_deriv_s, bm_dates.rename("created_at")], axis=1).dropna()
adoption_df.index.name = "base_model"
adoption_df = adoption_df.reset_index()
adoption_df["days_to_first_derivative"] = (
    adoption_df["first_derivative_date"] - adoption_df["created_at"]
).dt.days
adoption_df = adoption_df[adoption_df["days_to_first_derivative"] >= 0]
adoption_df = adoption_df.sort_values("days_to_first_derivative")

adoption_df.to_csv("rq4_adoption_speed.csv", index=False)
print(f"\nSaved rq4_adoption_speed.csv ({len(adoption_df):,} rows)")
# ── Adoption speed over time (NEW) ───────────────────────────────────────────
print("\n=== ADOPTION SPEED OVER TIME ===")

adoption_df["creation_quarter"] = adoption_df["created_at"].dt.to_period("Q")

adoption_trend = (
    adoption_df
    .groupby("creation_quarter")["days_to_first_derivative"]
    .agg(["median", "mean", "count"])
    .reset_index()
    .sort_values("creation_quarter")
)

print(adoption_trend.to_string(index=False))

adoption_trend.to_csv("rq4_adoption_trend.csv", index=False)
print("Saved rq4_adoption_trend.csv")
print(f"Fastest adoption:  {adoption_df['days_to_first_derivative'].min()} days")
print(f"Avg adoption speed: {adoption_df['days_to_first_derivative'].mean():.0f} days")
print(f"Med adoption speed: {adoption_df['days_to_first_derivative'].median():.0f} days")
print("\nTop 10 fastest-adopted base models:")
print(adoption_df[["base_model","days_to_first_derivative","total_derivatives"]].head(10).to_string(index=False))

# ── Quarterly diversity: unique active base models ────────────────────────────
print("\n=== QUARTERLY DIVERSITY ===")
q_stats = []
for quarter, group in dated_edges.groupby("target_quarter"):
    active_bases   = group["Source"].nunique()
    active_targets = group["Target"].nunique()
    n_edges        = len(group)
    cross_family   = 0  # placeholder for family diversity
    transform_dist = group["Transformation"].value_counts().to_dict()

    q_stats.append({
        "quarter":          str(quarter),
        "new_edges":        n_edges,
        "unique_sources":   active_bases,
        "unique_targets":   active_targets,
        "fine_tune_pct":    round(transform_dist.get("fine-tune",0) / max(n_edges,1) * 100, 1),
        "quantization_pct": round(transform_dist.get("quantization",0) / max(n_edges,1) * 100, 1),
        "adapter_pct":      round(transform_dist.get("adapter",0) / max(n_edges,1) * 100, 1),
        "merge_pct":        round(transform_dist.get("merge",0) / max(n_edges,1) * 100, 1),
    })

q_df = pd.DataFrame(q_stats).sort_values("quarter")
q_df.to_csv("rq4_quarterly_stats.csv", index=False)
print(q_df.to_string(index=False))
print(f"\nSaved rq4_quarterly_stats.csv")

# ── Ecosystem diversity: Herfindahl index per quarter ────────────────────────
# HHI: sum of squared market shares — lower = more diverse
print("\n=== ECOSYSTEM CONCENTRATION (HHI) BY QUARTER ===")
hhi_rows = []
for quarter, group in dated_edges.groupby("target_quarter"):
    source_counts = group["Source"].value_counts()
    total         = source_counts.sum()
    shares        = source_counts / total
    hhi           = (shares ** 2).sum()
    hhi_rows.append({"quarter": str(quarter), "hhi": round(hhi, 4), "n_edges": total})

hhi_df = pd.DataFrame(hhi_rows).sort_values("quarter")
print(hhi_df.to_string(index=False))
print("(HHI close to 0 = diverse, close to 1 = monopoly by one base model)")

diversity_df = q_df.merge(hhi_df, on="quarter")
diversity_df.to_csv("rq4_diversity_over_time.csv", index=False)
print("\nSaved rq4_diversity_over_time.csv")



# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
RQ4 SUMMARY STATISTICS
=======================
Edges with datable parent + child:    {len(dated_edges):,}
Valid temporal edges (child > parent): {len(valid_gaps):,}

Adoption speed (base → first derivative):
  Average: {adoption_df['days_to_first_derivative'].mean():.0f} days
  Median:  {adoption_df['days_to_first_derivative'].median():.0f} days
  Fastest: {adoption_df['days_to_first_derivative'].min()} days


Adoption speed over time (median days):
{adoption_trend[['creation_quarter','median']].to_string(index=False)}

Quarterly ecosystem stats:
{q_df.to_string(index=False)}

Concentration (HHI) over time:
{hhi_df.to_string(index=False)}
"""
with open("rq4_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print(summary)
print("Saved rq4_summary.txt")