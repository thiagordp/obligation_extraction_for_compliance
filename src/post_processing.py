import os
from typing import Dict, List
from pathlib import Path

from src.constants import DATASET_NAME, OBLIGATION_FILTERING_FOLDER


def split_by_filtering_label(dataset: List[Dict]) -> None:
    """
    Splits dataset sentences by filtering label and stores them in corresponding files.

    Args:
        dataset (List[Dict]): The dataset containing obligations and their filtering classifications.

    Returns:
        None
    """

    phrase_labeling = {}

    phrase_template = """================================
Article:     {article}
Paragraph:   {paragraph}
Sentence:    
{sentence}
"""

    # Process each paragraph in the dataset
    for paragraph_content in dataset:
        article, paragraph = paragraph_content["par_id"].split(".")

        for sentence in paragraph_content["potential_deontic"]:
            if "filtering" in sentence and "output" in sentence["filtering"]:
                filtering_label = sentence["filtering"]["output"]["classification"]

                # Format the sentence into the template
                formatted_text = phrase_template.format(
                    article=article,
                    paragraph=paragraph,
                    sentence=sentence["sentence"]
                )

                # Append to the corresponding category
                if filtering_label not in phrase_labeling:
                    phrase_labeling[filtering_label] = []

                phrase_labeling[filtering_label].append(formatted_text)

    # Ensure output directory exists
    os.makedirs(OBLIGATION_FILTERING_FOLDER, exist_ok=True)

    # Write each label's filtered obligations to a separate file
    for label, sentences in phrase_labeling.items():
        output_path = Path(OBLIGATION_FILTERING_FOLDER) / f"{label}.txt"

        with output_path.open("w", encoding="utf-8") as file:
            file.write("\n".join(sentences))

