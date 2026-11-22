"""
parser.py

Parses PubMed XML files and extracts metadata from each paper.
"""

import xml.etree.ElementTree as ET


def extract_paper(article_xml):
    """
    Extract metadata from one PubMed article.
    """
    pmid_node = article_xml.find(".//PMID")
    pmid = pmid_node.text if pmid_node is not None else ""

    title_node = article_xml.find(".//ArticleTitle")

    title = (
        title_node.text.strip()
        if title_node is not None and title_node.text
        else ""
    )

    abstract_parts = article_xml.findall(".//AbstractText")

    abstract = " ".join(
        part.text.strip()
        for part in abstract_parts
        if part.text
    )

    language = ""
    journal = ""
    year = ""
    publication_types = []

    article = article_xml.find(".//Article")

    if article is not None:

        lang_node = article.find("Language")
        language = (
            lang_node.text
            if lang_node is not None and lang_node.text
            else ""
        )

        journal_node = article.find("Journal/Title")
        journal = (
            journal_node.text
            if journal_node is not None and journal_node.text
            else ""
        )

        year_node = article.find(
            "Journal/JournalIssue/PubDate/Year"
        )

        year = (
            year_node.text
            if year_node is not None and year_node.text
            else ""
        )

        pub_nodes = article.findall(
            "PublicationTypeList/PublicationType"
        )

        publication_types = [
            node.text
            for node in pub_nodes
            if node.text
        ]


    mesh_terms = []

    medline = article_xml.find("MedlineCitation")

    if medline is not None:

        mesh_nodes = medline.findall(
            "MeshHeadingList/MeshHeading"
        )

        for mesh in mesh_nodes:

            descriptor = mesh.find("DescriptorName")

            if descriptor is not None and descriptor.text:
                mesh_terms.append(descriptor.text)


    return {

        "pmid": pmid,

        "title": title,

        "abstract": abstract,

        "language": language,

        "journal": journal,

        "year": year,

        "publication_types": publication_types,

        "mesh_terms": mesh_terms

    }


def parse_pubmed(xml_file):
    """
    Parse an entire PubMed XML file.

    Returns:
        List of paper dictionaries.
    """

    tree = ET.parse(xml_file)

    root = tree.getroot()

    articles = root.findall("PubmedArticle")

    papers = []

    for article_xml in articles:

        paper = extract_paper(article_xml)

        papers.append(paper)

    return papers