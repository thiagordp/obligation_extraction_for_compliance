import argparse
import json
import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

from src.validation import validation_filtering_results, validation_analysis_results

# Setup
load_dotenv()
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Name of the dataset")
args = parser.parse_args()

if args.dataset:
    os.environ["DATASET_NAME"] = args.dataset  # Set the environment variable
    # Reload constants to reflect new dataset name
    import importlib

    from src import constants

    importlib.reload(constants)

from src.llm import setup_together_client
from src.post_processing import split_by_filtering_label
from src.solve_references import standardize_citations
from src.constants import DATASET_PATH, PARAGRAPHS_TO_ANALYSE, DATASET_NAME, \
    OBLIGATION_FILTERING_RETRIEVE_SURROUNDING_PARAGRAPHS, OBLIGATION_FILTERING_RETRIEVE_CITATIONS_INSIDE_REGULATION, \
    OBLIGATION_ANALYSIS_RETRIEVE_SURROUNDING_PARAGRAPHS, OBLIGATION_ANALYSIS_RETRIEVE_CITATIONS_INSIDE_REGULATION, \
    SHALL_TYPES_OF_INTEREST, LLM_MODEL, TEMPERATURE, OBLIGATION_FILTERING_SYS_PROMPT, OBLIGATION_ANALYSIS_SYS_PROMPT, \
    OBLIGATION_FILTERING_POTENTIAL
from src.obligation_filtering import obligation_filtering, store_obligation_filtering_results
from src.obligation_analysis import obligation_analysis, store_obligation_analysis_results
from src.potential_obligations import load_dataset
from src.utils import setup_logging
import logging
import random


def present_metadata():
    logging.info("============================================================")
    logging.info(f"Dataset:                  {DATASET_NAME}")
    logging.info(f"Sample size:              {PARAGRAPHS_TO_ANALYSE}")
    logging.info(f"LLM:                      {LLM_MODEL}")
    logging.info(f"LLM Temperature:          {TEMPERATURE}")
    logging.info(f"Filtering prompt:         {OBLIGATION_FILTERING_SYS_PROMPT}")
    logging.info(f"Analysis prompt:          {OBLIGATION_ANALYSIS_SYS_PROMPT}")
    logging.info(f"Filtering sentence types: {SHALL_TYPES_OF_INTEREST}")
    logging.info(f"Filtering context:        {OBLIGATION_FILTERING_RETRIEVE_SURROUNDING_PARAGRAPHS}")
    logging.info(f"Filtering citations:      {OBLIGATION_FILTERING_RETRIEVE_CITATIONS_INSIDE_REGULATION}")
    logging.info(f"Analysis context:         {OBLIGATION_ANALYSIS_RETRIEVE_SURROUNDING_PARAGRAPHS}")
    logging.info(f"Analysis citations:       {OBLIGATION_ANALYSIS_RETRIEVE_CITATIONS_INSIDE_REGULATION}")
    logging.info("============================================================")


def main():
    setup_logging()

    present_metadata()

    together_client = setup_together_client()

    #
    # Loading the data
    #
    dataset = load_dataset(
        dataset_path=DATASET_PATH
    )

    # For reproducibility, set a seed value here.
    # random.seed(42)

    # For final validation, use a different seed.
    seed_value = int(time.time() * 1000000) ^ int.from_bytes(os.urandom(8), 'big') ^ uuid.uuid4().int
    seed_value = seed_value % 2 ** 32
    random.seed(seed_value)

    random.shuffle(dataset)
    dataset = standardize_citations(dataset)

    #
    # Obligation Filtering
    #
    logging.info("Starting obligation filtering...")
    obligations_labeled = obligation_filtering(
        client=together_client,
        dataset=dataset,
        retrieve_surrounding_paragraphs=OBLIGATION_FILTERING_RETRIEVE_SURROUNDING_PARAGRAPHS,
        retrieve_citations_inside_regulation=OBLIGATION_FILTERING_RETRIEVE_CITATIONS_INSIDE_REGULATION
    )
    store_obligation_filtering_results(obligations_labeled)
    logging.info(f"Obligation filtering completed.")

    #
    # Obligation Analysis
    #
    logging.info("Starting obligation analysis...")
    dataset = json.load(open(OBLIGATION_FILTERING_POTENTIAL))
    random.shuffle(dataset)

    obligation_extracted = obligation_analysis(
        client=together_client,
        dataset=dataset,
        retrieve_surrounding_paragraphs=OBLIGATION_ANALYSIS_RETRIEVE_SURROUNDING_PARAGRAPHS,
        retrieve_citations_inside_regulation=OBLIGATION_ANALYSIS_RETRIEVE_CITATIONS_INSIDE_REGULATION
    )
    store_obligation_analysis_results(obligation_extracted)
    logging.info(f"Obligation analysis completed.")


def post_processing():
    dataset = json.load(
        open(Path(f"data/processed/obligations_filtered/{DATASET_NAME}/obligation_analysis_system/AI_Act.json")))
    dataset = sorted(dataset, key=lambda x: x['par_id'])
    split_by_filtering_label(dataset)


if __name__ == "__main__":
    # main()
    validation_filtering_results()
    #validation_analysis_results()
    # post_processing()
