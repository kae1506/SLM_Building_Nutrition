
"""
main.py

Runs the complete PubMed nutrition preprocessing pipeline.
Appends KEEP papers to pubmed_nutrition_final.jsonl
and BORDERLINE papers to borderline.jsonl.
"""

import json
from pathlib import Path

from downloader import download_file
from extractor import extract_gz, delete_xml
from parser import parse_pubmed
from scorer import score_paper


START_FILE = 1
END_FILE = 1334

OUTPUT_FILE = Path("pubmed_nutrition_final.jsonl")
BORDERLINE_FILE = Path("borderline.jsonl")


RESET_OUTPUT = False

if RESET_OUTPUT:
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    if BORDERLINE_FILE.exists():
        BORDERLINE_FILE.unlink()


def append_jsonl(filename, papers):
    if not papers:
        return

    with open(filename, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")


total_papers = 0

total_keep = 0
total_borderline = 0
total_discard = 0

total_words = 0
total_characters = 0


for i in range(START_FILE, END_FILE + 1):

    print("\n" + "=" * 80)
    print(f"Processing pubmed26n{i:04d}")
    print("=" * 80)


    gz_path = download_file(i)


    xml_path = extract_gz(gz_path)


    papers = parse_pubmed(xml_path)

    print(f"Loaded {len(papers)} papers")

    total_papers += len(papers)


    keep = []
    borderline = []

    keep_count = 0
    borderline_count = 0
    discard_count = 0

    for paper in papers:

        paper = score_paper(paper)

        if paper["label"] == "KEEP":

            text = paper["title"] + " " + paper["abstract"]

            word_count = len(text.split())
            char_count = len(text)

            paper["word_count"] = word_count
            paper["char_count"] = char_count

            total_words += word_count
            total_characters += char_count

            keep.append(paper)
            keep_count += 1

        elif paper["label"] == "BORDERLINE":

            borderline.append(paper)
            borderline_count += 1

        else:

            discard_count += 1

    total_keep += keep_count
    total_borderline += borderline_count
    total_discard += discard_count


    append_jsonl(OUTPUT_FILE, keep)
    append_jsonl(BORDERLINE_FILE, borderline)


    delete_xml(xml_path)


    print(f"KEEP       : {keep_count}")
    print(f"BORDERLINE : {borderline_count}")
    print(f"DISCARD    : {discard_count}")

    print(f"Appended KEEP papers       : {OUTPUT_FILE}")
    print(f"Appended BORDERLINE papers : {BORDERLINE_FILE}")


print("\n" + "=" * 80)
print("PROCESSING SUMMARY")
print("=" * 80)

print(f"Files Processed    : {END_FILE - START_FILE + 1}")
print(f"Total Papers       : {total_papers:,}")

print(f"KEEP Papers        : {total_keep:,}")
print(f"BORDERLINE Papers  : {total_borderline:,}")
print(f"DISCARD Papers     : {total_discard:,}")

print()

print(f"Total Words        : {total_words:,}")
print(f"Total Characters   : {total_characters:,}")

if total_keep > 0:
    print(f"Average Words/Paper      : {total_words / total_keep:.2f}")
    print(f"Average Characters/Paper : {total_characters / total_keep:.2f}")

estimated_tokens = total_characters / 4

print(f"Estimated Tokens   : {estimated_tokens:,.0f}")

print("\nKEEP dataset saved to:")
print(OUTPUT_FILE)

print("\nBORDERLINE dataset saved to:")
print(BORDERLINE_FILE)
