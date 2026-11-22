"""
scorer.py

Quality filtering + nutrition relevance scoring.
"""

import re

import config


def score_paper(paper):
    """
    Scores a parsed PubMed paper.

    Returns:
        Updated paper dictionary with
        quality flags, nutrition score,
        matched terms and final label.
    """

    # ==================================================
    # Copy paper
    # ==================================================

    paper = paper.copy()

    # ==================================================
    # Extract fields
    # ==================================================

    title = paper["title"]
    abstract = paper["abstract"]
    language = paper["language"]
    journal = paper["journal"]

    mesh_terms = paper["mesh_terms"]
    publication_types = paper["publication_types"]

    title_lower = title.lower()
    abstract_lower = abstract.lower()

    # ==================================================
    # Quality Filtering
    # ==================================================

    quality_pass = True

    # English only
    if language != "eng":
        quality_pass = False

    # Title required
    if title.strip() == "":
        quality_pass = False

    # Abstract required
    if abstract.strip() == "":
        quality_pass = False

    # Remove unwanted publication types
    for pub_type in publication_types:

        if pub_type in config.REMOVE_PUBLICATION_TYPES:

            quality_pass = False

            break

    # ==================================================
    # Nutrition Scoring
    # ==================================================

    score = 0

    matched_strong_mesh = []

    matched_context_mesh = []

    matched_keywords = set()

    # --------------------------------------------------
    # Strong MeSH
    # --------------------------------------------------

    for mesh in mesh_terms:

        if mesh in config.STRONG_MESH:

            matched_strong_mesh.append(mesh)

            score += config.STRONG_MESH_SCORE

    # --------------------------------------------------
    # Context MeSH
    # --------------------------------------------------

    for mesh in mesh_terms:

        if mesh in config.CONTEXT_MESH:

            matched_context_mesh.append(mesh)

    context_count = len(matched_context_mesh)

    if context_count >= 2:

        score += 2

    elif context_count == 1:

        score += 1

    # --------------------------------------------------
    # Keyword Matching
    # --------------------------------------------------

    for keyword in config.NUTRITION_KEYWORDS:

        pattern = rf"\b{re.escape(keyword.lower())}\b"

        # Prefer title match
        if re.search(pattern, title_lower):

            matched_keywords.add(keyword)

            score += config.TITLE_KEYWORD_SCORE

        # Otherwise abstract
        elif re.search(pattern, abstract_lower):

            matched_keywords.add(keyword)

            score += config.ABSTRACT_KEYWORD_SCORE

    # --------------------------------------------------
    # Nutrition Journal
    # --------------------------------------------------

    if journal in config.NUTRITION_JOURNALS:

        score += config.JOURNAL_SCORE

    # ==================================================
    # Final Decision
    # ==================================================

    has_strong = len(matched_strong_mesh) > 0

    has_keyword = len(matched_keywords) > 0

    if not quality_pass:

        label = "DISCARD"

    elif has_strong:

        label = "KEEP"

    elif score >= config.KEEP_THRESHOLD and has_keyword:

        label = "KEEP"

    elif score >= config.BORDERLINE_THRESHOLD:

        label = "BORDERLINE"

    else:

        label = "DISCARD"

    # ==================================================
    # Store Results
    # ==================================================

    paper["quality_pass"] = quality_pass

    paper["nutrition_score"] = score

    paper["label"] = label

    paper["matched_strong_mesh"] = matched_strong_mesh

    paper["matched_context_mesh"] = matched_context_mesh

    paper["matched_keywords"] = sorted(list(matched_keywords))

    return paper