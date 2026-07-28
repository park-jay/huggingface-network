## Towards AI Model Lineage: Understanding Model Reuse Patterns at Scale on Hugging Face
This repository contains materials for paper "Towards AI Model Lineage: Understanding Model Reuse Patterns at Scale on Hugging Face"

## Abstract
The rapid growth of AI model sharing platforms has intensified the reuse and derivation of large language models (LLMs), yet the structural patterns of such reuse remain underexplored. This study examines AI model lineage at scale by analyzing parent–child relationships among 341,070 text‑generation models hosted on Hugging Face. We operationalize model reuse as a directed lineage graph and investigate (1) whether multi‑parent descendant models arise from same‑family or cross‑family mergers and (2) how lineage depth varies across model families. Our findings show that multi‑parent reuse is predominantly same‑family, indicating strong architectural homophily, while cross‑family mergers are rare and primarily occur within shared cultural or geopolitical contexts. We further find that lineage depth is generally shallow across the ecosystem, with only a few dominant model families sustaining deep, multi‑generational derivation chains. These results suggest that open model sharing does not necessarily promote architectural or cultural diversity, but instead reinforces existing foundations.

## RQ 1: To what extent are descendant models the product of cross-family mergers? (Inbreed vs. outbreed analysis)
The code for this research question is available in the [RQ1 file](https://github.com/anuska58/hugging_face_ecosystem/blob/main/rq1.ipynb)

## RQ 2: To what extent do lineage depths vary across model families? (Lineage depth analysis)
The code for this research question is available in the [RQ2 file](https://github.com/anuska58/hugging_face_ecosystem/blob/main/rq2.ipynb)
