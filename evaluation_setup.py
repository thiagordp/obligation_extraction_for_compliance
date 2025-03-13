"""
Setup evaluation files

"""
import random
from pathlib import Path

DATASETS = ["AI_Act", "DSA", "GDPR"]
REVIEWERS = ["R1", "R2", "R3", "R4"]
GROUP_REVIEWS = {"G1": ["R1", "R4"], "G2": ["R2", "R3"]}
TASKS = ["obligation_filtering", "obligation_analysis"]
OUTPUT_DIR = "setup_evaluation/"

DOCS_PER_REVIEWER_PER_DATASET = 15


def select_sample_from_dataset(task: str, dataset: str, n_samples: int):
    path_to_dataset = Path(f"data/validation/{task}/{dataset}/{task}_system/")

    docs = list(path_to_dataset.glob("*.txt"))
    random.shuffle(docs)

    return docs[:n_samples]


def setup_evaluation_files(task: str):


    for group_review in GROUP_REVIEWS:
        for dataset in DATASETS:
            docs_to_review = select_sample_from_dataset(task, dataset, DOCS_PER_REVIEWER_PER_DATASET)

            for reviewer in GROUP_REVIEWS[group_review]:
                output_folder_path = Path(OUTPUT_DIR) / f"{reviewer}/{task}/{dataset}/"

                for file_to_review in docs_to_review:
                    file_to_review = Path(file_to_review)
                    output_file_path = output_folder_path / file_to_review.name

                    # Copy the document from the file_to_review to the output_file_path
                    output_folder_path.mkdir(parents=True, exist_ok=True)
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)

                    output_file_path.write_text(file_to_review.read_text())

    # For each group review
    #   For each dataset
    #       retrieve DOCS_PER_REVIEWER_PER_DATASET documents
    #


def evaluation_setup():
    random.seed(42)
    setup_evaluation_files("obligation_filtering")
    setup_evaluation_files("obligation_analysis")

if __name__ == "__main__":
    evaluation_setup()