"""
extractor.py

Extracts PubMed .xml.gz files into XML.
"""

import gzip
import shutil
from pathlib import Path


DOWNLOAD_DIR = Path("downloads")
TEMP_DIR = Path("temp")

TEMP_DIR.mkdir(parents=True, exist_ok=True)


def extract_gz(gz_path):
    """
    Extract a .xml.gz file to temp/.

    Returns:
        Path to extracted XML.
    """

    gz_path = Path(gz_path)

    xml_name = gz_path.stem          # removes .gz

    xml_path = TEMP_DIR / xml_name

    if xml_path.exists():

        print(f"{xml_name} already extracted.")

        return xml_path

    print(f"Extracting {gz_path.name}...")

    with gzip.open(gz_path, "rb") as f_in:

        with open(xml_path, "wb") as f_out:

            shutil.copyfileobj(f_in, f_out)

    print("Done.")

    return xml_path


def delete_xml(xml_path):

    xml_path = Path(xml_path)

    if xml_path.exists():

        xml_path.unlink()

        print(f"Deleted {xml_path.name}")


if __name__ == "__main__":

    print("Use main.py to run the complete pipeline.")