import json
from transformers import AutoTokenizer
# Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

total_tokens = 0
total_papers = 0

with open("pubmed_nutrition_final.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        paper = json.loads(line)

        text = paper["title"] + "\n" + paper["abstract"]

        total_tokens += len(
            tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

        total_papers += 1

print("=" * 60)
print("LLAMA TOKEN STATISTICS")
print("=" * 60)

print(f"Total Papers      : {total_papers:,}")
print(f"Total Tokens      : {total_tokens:,}")
print(f"Average Tokens    : {total_tokens / total_papers:.2f}")