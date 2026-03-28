import json, random, os, re

# ── Cleaning ─────────────────────────────────────────────────
def clean_item(item):
    instruction = item.get("prompt", "").strip()
    response    = item.get("completion", "").strip()

    instruction = re.sub(r'^User:\s*', '', instruction, flags=re.IGNORECASE)
    response    = re.sub(r'^Coach:\s*', '', response, flags=re.IGNORECASE)
    response    = re.sub(
        r'\.?\s*If pain occurs,?\s*stop and reduce intensity\.?',
        '', response, flags=re.IGNORECASE
    ).strip().strip(' .,')

    item["prompt"]      = instruction
    item["completion"]  = response
    return item

# ── Load & validate ──────────────────────────────────────────
def load_and_validate(path, domain_tag):
    items, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item = clean_item(item)

                prompt     = item.get("prompt", "")
                completion = item.get("completion", "")

                if not prompt or not completion:
                    skipped += 1; continue
                if len(completion) < 40:
                    skipped += 1; continue
                if len(prompt) > 512:
                    skipped += 1; continue

                items.append({
                    "prompt":     prompt,
                    "completion": completion,
                    "domain":     domain_tag
                })
            except json.JSONDecodeError:
                skipped += 1

    print(f"[{domain_tag}] loaded: {len(items)}, skipped: {skipped}")
    return items

# ── Format ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI fitness and nutrition coach. Your role is to provide safe, science-based, and personalized guidance.

Rules you must always follow:
- Give structured, clear responses using headers and bullet points
- Be specific with numbers (sets, reps, calories, macros)
- Never invent a conversation or roleplay as the user
- If a question is outside fitness or nutrition, politely redirect
- Always end your response cleanly — never repeat yourself"""

def to_phi3_format(item):
    tag        = f"[{item['domain']}]"
    prompt     = item['prompt'].strip()
    completion = item['completion'].strip()
    return {
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{tag} {prompt}<|end|>\n"
            f"<|assistant|>\n{completion}<|end|>"
        )
    }

# ── Save ─────────────────────────────────────────────────────
def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ── Run ──────────────────────────────────────────────────────
fitness   = load_and_validate("data/raw/fitness_qa.jsonl",   "FITNESS")
nutrition = load_and_validate("data/raw/nutrition_qa.jsonl", "NUTRITION")

merged = [to_phi3_format(i) for i in fitness + nutrition]
random.seed(42)
random.shuffle(merged)

split      = int(len(merged) * 0.9)
train_data = merged[:split]
val_data   = merged[split:]

save_jsonl(train_data, "data/splits/train.jsonl")
save_jsonl(val_data,   "data/splits/val.jsonl")
save_jsonl(merged,     "data/processed/full_dataset.jsonl")

print(f"\nTotal: {len(merged)} | Train: {len(train_data)} | Val: {len(val_data)}")

# ── Visual check ─────────────────────────────────────────────
print("\n── Sample 1 ──")
print(train_data[0]["text"])
print("\n── Sample 2 ──")
print(train_data[1]["text"]) 