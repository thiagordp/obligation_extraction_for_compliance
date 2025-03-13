# Code to retrieve the context of other articles or sections.
# Add a key with context content with string for the given article
import copy
import logging
from typing import List


def standardize_citations(data):
    # Collect all "par_id" values from the top-level objects
    all_par_ids = set(entry["par_id"] for entry in data)

    for entry in data:
        for obj in entry["potential_deontic"]:
            new_references = []
            for ref in obj["references"]:
                if isinstance(ref, str):
                    # Case 1: Pattern "001.005" - retain as is
                    if "." in ref and ref.split(".")[-1].isdigit():
                        new_references.append(ref)

                    # Case 2: Pattern "001." - expand based on "par_id"
                    elif "." in ref and ref.endswith("."):
                        expanded_refs = [pid for pid in all_par_ids if pid.startswith(ref)]
                        new_references.extend(expanded_refs)

                # Case 3: It's a list of lists - flatten and include all values
                elif isinstance(ref, list):
                    for sublist in ref:
                        if isinstance(sublist, list):
                            new_references.extend(sublist)
                        else:
                            new_references.append(sublist)

            # Update the references for the object
            obj["references"] = sorted(list(set(new_references)))  # Remove duplicates

    return data


def retrieve_paragraph_content(par_id: str, dataset) -> str:
    for paragraph in dataset:
        if paragraph["par_id"] == par_id:
            return paragraph["text"].strip()
    return ""


def retrieve_context_for_reference(reference: str, dataset) -> str | None:
    standard_string = """
@Article @Paragraph
@Content
"""
    try:
        this_section_id, this_par_id = [int(x) for x in reference.split(".")]

        standard_string = standard_string.replace("@Article", f"Article {this_section_id}")
        standard_string = standard_string.replace("@Paragraph", f"Paragraph {this_par_id}")
        content = retrieve_paragraph_content(reference, dataset=dataset)

        standard_string = standard_string.replace("@Content", content.strip())
        return standard_string
    except Exception as e:
        logging.error(f"Error while processing {reference}: {e}")
        return None


def retrieve_citation_contents(sentence: dict, dataset: dict) -> list:
    sentence_context = []

    # Terminate if no references found
    if len(sentence["references"]) == 0:
        return []

    for ref in sentence["references"]:
        content = retrieve_context_for_reference(ref, dataset)
        if content:
            sentence_context.append(content)

    return sentence_context


def paragraph_exists(par_id: str, dataset) -> bool:
    for paragraph in dataset:
        if paragraph["par_id"] == par_id:
            return True
    return False


def retrieve_context(reference: str, dataset: List[dict], retrieve_surrounding_paragraphs=False) -> list:
    current_paragraph_content = retrieve_paragraph_content(reference, dataset)
    if not retrieve_surrounding_paragraphs:
        return [current_paragraph_content]

    this_section_id, this_par_id = [int(x) for x in reference.split(".")]
    before_par_id = this_par_id - 1
    after_par_id = this_par_id + 1

    before_reference = f"{this_section_id:03d}.{before_par_id:03d}"
    after_reference = f"{this_section_id:03d}.{after_par_id:03d}"

    before_paragraph_content = after_paragraph_content = ""
    if paragraph_exists(before_reference, dataset):
        before_paragraph_content = retrieve_paragraph_content(before_reference, dataset)
    if paragraph_exists(after_reference, dataset):
        after_paragraph_content = retrieve_paragraph_content(after_reference, dataset)

    return [
        before_paragraph_content,
        current_paragraph_content,
        after_paragraph_content,
    ]
