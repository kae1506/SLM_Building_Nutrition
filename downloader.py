"""
downloader.py

Downloads PubMed baseline XML.GZ files.
"""

from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"

DOWNLOAD_DIR = Path("downloads")

DOWNLOAD_DIR.mkdir(exist_ok=True)


def download_file(index):

    filename = f"pubmed26n{index:04d}.xml.gz"

    destination = DOWNLOAD_DIR / filename

    if destination.exists():

        print(f"{filename} already exists")

        return destination

    url = f"{BASE_URL}/{filename}"

    print(f"Downloading {filename}...")

    urlretrieve(url, destination)

    print("Done.")

    return destination
