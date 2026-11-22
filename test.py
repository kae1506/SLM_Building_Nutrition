# from parser import parse_pubmed
# from scorer import score_paper

# papers = parse_pubmed("pubmed26n0215.xml")

# paper = score_paper(papers[100])

# print(paper)
from extractor import extract_gz
from parser import parse_pubmed

# Extract XML from the .gz file
xml_path = extract_gz("downloads/pubmed26n0200.xml.gz")

# Parse it
papers = parse_pubmed(xml_path)

print("First 10 papers:\n")

for paper in papers[:10]:
    print(
        f"{paper['year']} | {paper['journal']} | {paper['title']}"
    )
