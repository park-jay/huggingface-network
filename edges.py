import pandas as pd
import re
import csv
from tqdm import tqdm

INPUT_FILE = "merged_dataset.csv"
OUTPUT_FILE = "edges.csv"


# ----------------------------
# Validate HuggingFace model ID
# Must be in format: "org/model-name"
# ----------------------------
HF_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$')

def is_valid_hf_id(text):
    if not isinstance(text, str):
        return False
    text = text.strip()
    return bool(HF_ID_PATTERN.match(text)) and len(text) > 3


# ----------------------------
# Method 1: Extract base_model from tags column
# Tags look like: "...,base_model:Qwen/Qwen3.5-27B,base_model:finetune:Qwen/Qwen3.5-27B,..."
# This is the most reliable source — explicit metadata from HuggingFace
# ----------------------------
def extract_parents_from_tags(tags_str):
    if not isinstance(tags_str, str):
        return []

    parents = []
    for tag in tags_str.split(','):
        tag = tag.strip()
        if tag.startswith('base_model:'):
            # Strip prefix variants:
            # base_model:org/model
            # base_model:finetune:org/model
            # base_model:quantized:org/model
            # base_model:adapter:org/model
            # base_model:merge:org/model
            # base_model:revision:org/model
            val = re.sub(
                r'^base_model:(finetune:|quantized:|adapter:|merge:|revision:|distil:|pruned:)?',
                '', tag
            ).strip()
            if is_valid_hf_id(val):
                parents.append(val)

    return list(set(parents))


# ----------------------------
# Method 2: Extract parents from model card text (NLP)
# Only accepts values that look like real HuggingFace model IDs
# ----------------------------
def extract_parents_from_card(text):
    if not isinstance(text, str):
        return []

    patterns = [
        r"fine[- ]?tuned from\s+([\w\-\.]+/[\w\-\.]+)",
        r"based on\s+([\w\-\.]+/[\w\-\.]+)",
        r"derived from\s+([\w\-\.]+/[\w\-\.]+)",
        r"initialized from\s+([\w\-\.]+/[\w\-\.]+)",
        r"trained on top of\s+([\w\-\.]+/[\w\-\.]+)",
        r"built on\s+([\w\-\.]+/[\w\-\.]+)",
        r"checkpoint of\s+([\w\-\.]+/[\w\-\.]+)",
        r"starting from\s+([\w\-\.]+/[\w\-\.]+)",
        r"distilled from\s+([\w\-\.]+/[\w\-\.]+)",
        r"pruned from\s+([\w\-\.]+/[\w\-\.]+)",
        r"merged from\s+([\w\-\.]+/[\w\-\.]+)",
    ]

    parents = []
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            if is_valid_hf_id(m):
                parents.append(m)

    return list(set(parents))


# ----------------------------
# Method 3: Tag heuristics
# Only fires for strong quantization/adapter signals
# Strips suffix from model name to guess parent
# ----------------------------
def extract_parents_from_heuristic(tags_str, model_name, creator, model_id):
    tags_lower = str(tags_str).lower()
    model_lower = str(model_name).lower()

    quant_signals  = ["gptq", "gguf", "awq", "4bit", "8bit", "fp8", "nvfp4", "exl2", "imatrix"]
    adapter_signals = ["lora", "qlora", "peft"]

    has_quant   = any(x in tags_lower or x in model_lower for x in quant_signals)
    has_adapter = any(x in tags_lower or x in model_lower for x in adapter_signals)

    if not (has_quant or has_adapter):
        return []

    base_guess = re.sub(
        r'[-_](gptq|gguf|awq|lora|qlora|4bit|8bit|fp8|fp4|nvfp4|exl2|imatrix|quantized|q4|q5|q6|q8).*$',
        '', model_name, flags=re.IGNORECASE
    ).strip()

    if base_guess and base_guess.lower() != model_name.lower():
        guessed_parent = f"{creator}/{base_guess}"
        if is_valid_hf_id(guessed_parent) and guessed_parent != model_id:
            return [guessed_parent]

    return []


# ----------------------------
# Detect transformation type and confidence
# ----------------------------
def detect_transformation(tags_str, model_name, card_text, source):
    tags_lower  = str(tags_str).lower()
    model_lower = str(model_name).lower()
    card_lower  = str(card_text).lower()

    # Quantization
    if any(x in tags_lower for x in ["gptq", "gguf", "awq", "4bit", "8bit", "exl2", "imatrix", "nvfp4", "fp8", "fp4"]):
        return "quantization", (1.0 if source == "base_model_field" else 0.9)
    if any(x in model_lower for x in ["gptq", "gguf", "awq", "4bit", "8bit", "fp8", "fp4", "nvfp4", "exl2"]):
        return "quantization", (1.0 if source == "base_model_field" else 0.9)

    # Adapter / LoRA
    if any(x in tags_lower for x in ["lora", "qlora", "peft", "adapter"]):
        return "adapter", (1.0 if source == "base_model_field" else 0.9)
    if any(x in model_lower for x in ["lora", "qlora"]):
        return "adapter", (1.0 if source == "base_model_field" else 0.9)

    # Merge
    if any(x in tags_lower for x in ["merge", "mergekit", "merged"]):
        return "merge", (1.0 if source == "base_model_field" else 0.8)
    if any(x in model_lower for x in ["merge", "merged"]):
        return "merge", (1.0 if source == "base_model_field" else 0.8)

    # Distillation
    if any(x in tags_lower for x in ["distil", "distilled", "distillation"]):
        return "distillation", (1.0 if source == "base_model_field" else 0.9)
    if any(x in model_lower for x in ["distil", "distilled"]):
        return "distillation", (1.0 if source == "base_model_field" else 0.9)

    # Pruning
    if "prun" in tags_lower or "prun" in model_lower:
        return "pruning", (1.0 if source == "base_model_field" else 0.9)

    # Fine-tune (inferred from card text)
    if re.search(r"fine[- ]?tun", card_lower):
        return "fine-tune", (1.0 if source == "base_model_field" else 0.9)

    # Default: assume fine-tune if we found a parent but no other signal
    return "fine-tune", (1.0 if source == "base_model_field" else 0.6)


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} models")

# Check available columns
print(f"Columns: {df.columns.tolist()}")

# Check if model_card column exists
has_card = "model_card" in df.columns
print(f"Model card column available: {has_card}")
print()

edges = []
source_counts = {
    "base_model_field": 0,
    "model_card_text":  0,
    "tag_heuristic":    0
}

print("Building edges...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    model_id   = str(row.get("model_id",   "")).strip()
    model_name = str(row.get("model_name", "")).strip()
    creator    = str(row.get("creator",    "")).strip()
    tags       = row.get("tags",           "")
    card       = row.get("model_card",     "") if has_card else ""

    if not is_valid_hf_id(model_id):
        continue

    parents = []
    source  = None

    # -----------------------------------------------
    # Method 1: base_model tags (explicit, most reliable)
    # -----------------------------------------------
    tag_parents = extract_parents_from_tags(str(tags))
    if tag_parents:
        parents = tag_parents
        source  = "base_model_field"
        source_counts["base_model_field"] += len(tag_parents)

    # -----------------------------------------------
    # Method 2: model card NLP
    # -----------------------------------------------
    if not parents and has_card:
        card_parents = extract_parents_from_card(card)
        if card_parents:
            parents = card_parents
            source  = "model_card_text"
            source_counts["model_card_text"] += len(card_parents)

    # -----------------------------------------------
    # Method 3: tag heuristic (last resort)
    # -----------------------------------------------
    if not parents:
        heuristic_parents = extract_parents_from_heuristic(tags, model_name, creator, model_id)
        if heuristic_parents:
            parents = heuristic_parents
            source  = "tag_heuristic"
            source_counts["tag_heuristic"] += len(heuristic_parents)

    # -----------------------------------------------
    # Build edges (deduplicate parents first)
    # -----------------------------------------------
    seen = set()
    for parent in parents:
        parent = parent.strip()
        if not is_valid_hf_id(parent):
            continue
        if parent == model_id:
            continue
        if parent in seen:
            continue
        seen.add(parent)

        transformation, confidence = detect_transformation(tags, model_name, card, source)

        edges.append([
            parent,
            model_id,
            "Directed",
            transformation,
            confidence,
            source
        ])


# ----------------------------
# Save edges
# ----------------------------
with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Source", "Target", "Type", "Transformation", "Confidence", "ExtractionSource"])
    writer.writerows(edges)

# ----------------------------
# Summary report
# ----------------------------
edges_df = pd.DataFrame(
    edges,
    columns=["Source", "Target", "Type", "Transformation", "Confidence", "ExtractionSource"]
)

print("\n✅ Done!")
print(f"Total edges: {len(edges)}")

print(f"\nEdges by extraction method:")
for method, count in source_counts.items():
    pct = count / len(edges) * 100 if edges else 0
    print(f"  {method}: {count} ({pct:.1f}%)")

print(f"\nTransformation breakdown:")
print(edges_df["Transformation"].value_counts())

print(f"\nConfidence distribution:")
print(edges_df["Confidence"].value_counts())

print(f"\nUnique source nodes:  {edges_df['Source'].nunique()}")
print(f"Unique target nodes:  {edges_df['Target'].nunique()}")
print(f"Total unique nodes in graph: {pd.concat([edges_df['Source'], edges_df['Target']]).nunique()}")

print(f"\nTop 15 most-derived-from models (highest in-degree):")
top_parents = edges_df["Source"].value_counts().head(15)
for model, count in top_parents.items():
    print(f"  {model}: {count} derivatives")