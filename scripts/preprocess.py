import json
import os
import random


def resolve_input_path(*candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"None of the candidate input files exist: {', '.join(candidates)}"
    )


def load_and_validate(path, domain_tag):
    items, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                instruction = item.get("instruction") or item.get("prompt", "")
                response = item.get("response") or item.get("completion", "")

                instruction = instruction.strip()
                response = response.strip()

                if not instruction or not response:
                    skipped += 1
                    continue
                if len(response) < 40:
                    skipped += 1
                    continue
                if len(instruction) > 512:
                    skipped += 1
                    continue

                items.append(
                    {
                        "instruction": instruction,
                        "response": response,
                        "domain": domain_tag,
                    }
                )
            except json.JSONDecodeError:
                skipped += 1

    print(f"[{domain_tag}] loaded: {len(items)}, skipped: {skipped}")
    return items


def to_mistral_format(item):
    return {
        "text": (
            f"<s>[INST] [{item['domain']}] {item['instruction']} [/INST] "
            f"{item['response']} </s>"
        )
    }


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


fitness_path = resolve_input_path(
    "data/raw/fitness_qa.jsonl",
    "data/raw/fitness_qa.jsonl.jsonl",
)
nutrition_path = resolve_input_path(
    "data/raw/nutrition_qa.jsonl",
    "data/raw/nutrition_qa.jsonl.jsonl",
)

fitness = load_and_validate(fitness_path, "FITNESS")
nutrition = load_and_validate(nutrition_path, "NUTRITION")

merged = [to_mistral_format(i) for i in fitness + nutrition]
random.seed(42)
random.shuffle(merged)

split = int(len(merged) * 0.9)
train_data = merged[:split]
val_data = merged[split:]

save_jsonl(train_data, "data/splits/train.jsonl")
save_jsonl(val_data, "data/splits/val.jsonl")
save_jsonl(merged, "data/processed/full_dataset.jsonl")

print(f"\nTotal: {len(merged)} | Train: {len(train_data)} | Val: {len(val_data)}")

if train_data:
    print("\n-- Sample 1 --")
    print(train_data[0]["text"][:400])

if len(train_data) > 1:
    print("\n-- Sample 2 --")
    print(train_data[1]["text"][:400])

