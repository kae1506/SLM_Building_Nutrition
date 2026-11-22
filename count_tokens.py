import json

TOTAL_WORDS = 0
TOTAL_CHARACTERS = 0
TOTAL_PAPERS = 0

with open("pubmed_nutrition_final.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        paper = json.loads(line)

        text = paper["title"] + " " + paper["abstract"]

        TOTAL_WORDS += len(text.split())
        TOTAL_CHARACTERS += len(text)
        TOTAL_PAPERS += 1

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Total Papers      : {TOTAL_PAPERS:,}")
print(f"Total Words       : {TOTAL_WORDS:,}")
print(f"Total Characters  : {TOTAL_CHARACTERS:,}")
print(f"Average Words     : {TOTAL_WORDS / TOTAL_PAPERS:.2f}")
print(f"Average Characters: {TOTAL_CHARACTERS / TOTAL_PAPERS:.2f}")
print(f"Estimated Tokens  : {TOTAL_CHARACTERS / 4:,.0f}")