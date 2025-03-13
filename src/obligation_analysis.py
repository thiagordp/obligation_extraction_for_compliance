import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

import tqdm
from together import Together

from src.constants import OBLIGATION_ANALYSIS_SYS_PROMPT, OBLIGATION_ANALYSIS_USER_PROMPT, SHALL_TYPES_OF_INTEREST, \
    LLM_MODEL, TEMPERATURE, OBLIGATION_ANALYSIS_RESULTS, OBLIGATION_ANALYSIS_VALIDATION_TEMPLATE, \
    OBLIGATION_ANALYSIS_SINGLE_TEMPLATE, DATASET_NAME, VALIDATION_OBLIGATION_ANALYSIS_FOLDER, \
    OBLIGATION_FILTERING_SYS_PROMPT, PROMPTS_PATH, PARAGRAPHS_TO_ANALYSE
from src.llm import execute_prompt, count_tokens_io
from src.solve_references import retrieve_citation_contents, retrieve_context
from src.utils import extract_dict, display_stats


def obligation_analysis(client: Together, dataset: List[dict], retrieve_citations_inside_regulation=False,
                        retrieve_surrounding_paragraphs=False) -> List[dict]:
    """
    Perform obligation analysis on the given obligations.

    Args:
        dataset (List[dict]): A dictionary containing the obligations to be analyzed.
        retrieve_citations_inside_regulation (bool): Whether to include citations in the analysis.
        retrieve_surrounding_paragraphs (bool): Whether to retrieve the surrounding paragraphs.
    """

    def store_obligation_analysis():
        try:
            with open(OBLIGATION_ANALYSIS_RESULTS, "w") as json_file:
                json.dump(dataset, json_file, indent=4)
                logging.info("Saved obligation analysis results to obligation_analysis.json successfully.")
        except Exception as e:
            logging.error(f"Error while saving obligation analysis results to JSON: {e}")

    def load_detection_obligation_prompt():
        system_prompt_template = open(OBLIGATION_ANALYSIS_SYS_PROMPT).read()
        user_prompt_template = open(OBLIGATION_ANALYSIS_USER_PROMPT).read()

        return system_prompt_template, user_prompt_template

    # Load prompts
    system_prompt, user_prompt = load_detection_obligation_prompt()
    tokens = {
        "input": [],
        "output": [],
        "total": []
    }
    # For each paragraph in the obligations
    for paragraph in tqdm.tqdm(dataset[:PARAGRAPHS_TO_ANALYSE]):
        potential_sentences = paragraph["potential_deontic"]
        par_id = paragraph["par_id"]

        for index_sentence, sentence in enumerate(potential_sentences):
            # Check if the sentence is an obligation
            if not "filtering" in sentence.keys() or not "output" in sentence["filtering"].keys() or \
                    not sentence["filtering"]["output"]["classification"] in SHALL_TYPES_OF_INTEREST:
                continue
            sentence_user_prompt = user_prompt.replace("@Sentence", sentence["sentence"])

            # Add sourrounding context to the prompt or just the paragraph itself.
            context_paragraphs = retrieve_context(
                reference=par_id,
                dataset=dataset,
                retrieve_surrounding_paragraphs=True
            )
            sentence_user_prompt = sentence_user_prompt.replace("@Context", "\n".join(context_paragraphs).strip())

            citations = retrieve_citation_contents(
                sentence=sentence,
                dataset=dataset
            ) if retrieve_citations_inside_regulation else []

            citation_content = "\n".join(citations).strip() if len(citations) > 0 else "No Citation"
            sentence_user_prompt = sentence_user_prompt.replace("@Citations", citation_content)

            dict_results = {}
            successful_attempt = False
            number_attempts = 0
            token_count = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            while number_attempts < 5 and not successful_attempt:
                try:
                    results = execute_prompt(
                        client=client,
                        model=LLM_MODEL,
                        user_prompt=sentence_user_prompt,
                        system_prompt=system_prompt,
                        temperature=TEMPERATURE,
                        show_execution_time=False
                    )

                    dict_results = extract_dict(results)
                    token_count = count_tokens_io(
                        input_text=sentence_user_prompt + system_prompt,
                        output_text=results
                    )

                    tokens["input"].append(token_count["input_tokens"])
                    tokens["output"].append(token_count["output_tokens"])
                    tokens["total"].append(token_count["total_tokens"])

                    sentence["analysis"] = {
                        "output": dict_results,
                        "tokens": token_count,
                        "prompts": {
                            # "system_prompt": system_prompt, # In principle, not necessary
                            "user_prompt": sentence_user_prompt
                        }
                    }
                    successful_attempt = True

                except Exception as e:
                    logging.error(f"Error while sending request: {e}")
                    number_attempts += 1
                    logging.error(f"Attempt {number_attempts}")

                    time.sleep(5)
            if not successful_attempt:
                logging.error(f"Failed to process {par_id}")

    display_stats(tokens)

    while True:
        logging.info("Sleeping for 1 minute before checking again...")
        time.sleep(60)  # Wait for 1 minute before checking again

    store_obligation_analysis()

    return dataset


def store_obligation_analysis_results(dataset: List[dict]) -> None:
    """
    For every paragraph in dataset:
        Check if there is the analysis object.
            If not, go to the next.
        
        Replace the general information (tokens, prompt, etc.)
        For each extracted obligation
            Create a validation object for it.
            Replace the with the corresponding index and Predicate for clarity.
            Put it the Obligation Analysis validation struture.
        ...
    """
    logging.info("Storing obligation analysis results...")
    system_prompt = OBLIGATION_ANALYSIS_SYS_PROMPT
    output_base_path = (Path(VALIDATION_OBLIGATION_ANALYSIS_FOLDER) / system_prompt
                        .replace(PROMPTS_PATH, "").replace(".txt", ""))
    os.makedirs(name=output_base_path, exist_ok=True)

    dataset_name = DATASET_NAME
    llm_model = LLM_MODEL
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    template_analysis_content = open(OBLIGATION_ANALYSIS_VALIDATION_TEMPLATE).read()
    template_obligation_analysis_single_content = open(OBLIGATION_ANALYSIS_SINGLE_TEMPLATE).read()

    template_analysis_content = template_analysis_content.replace("@DATASET", dataset_name)
    template_analysis_content = template_analysis_content.replace("@LLM_MODEL", llm_model)
    template_analysis_content = template_analysis_content.replace("@PROMPT", system_prompt)
    template_analysis_content = template_analysis_content.replace("@TIMESTAMP", timestamp)

    skipped_sentences = 0
    for paragraph_content in dataset:
        article, paragraph = paragraph_content["par_id"].split(".")

        sentence_analysis_content = str(template_analysis_content)
        sentence_analysis_content = sentence_analysis_content.replace("@PARAGRAPH", paragraph)
        sentence_analysis_content = sentence_analysis_content.replace("@ARTICLE", article)
        for index_sentence, sentence in enumerate(paragraph_content["potential_deontic"]):
            if "analysis" not in sentence.keys():
                skipped_sentences += 1
                continue

            analysis = sentence["analysis"]

            llm_input = analysis["prompts"]["user_prompt"]
            llm_output = analysis["output"]

            if llm_output is None:
                print("LLM OUTPUT EMPTY")
                print(sentence)

            sentence_analysis_content = sentence_analysis_content.replace("@SENTENCE", str(index_sentence + 1))
            sentence_analysis_content = sentence_analysis_content.replace("@INPUT_TOKENS",
                                                                          str(analysis["tokens"]["input_tokens"]))
            sentence_analysis_content = sentence_analysis_content.replace("@OUTPUT_TOKENS",
                                                                          str(analysis["tokens"]["output_tokens"]))

            sentence_analysis_content = sentence_analysis_content.replace("@LLM_INPUT", llm_input)
            sentence_analysis_content = sentence_analysis_content.replace("@LLM_OUTPUT",
                                                                          json.dumps(llm_output, indent=3))
            for index_obligation, obligation_extracted in enumerate(llm_output):
                obligation_analysis_single_content = str(template_obligation_analysis_single_content)
                obligation_analysis_single_content = obligation_analysis_single_content.replace(
                    "@OBLIGATION_PREDICATE", obligation_extracted["Predicate"]["value"])
                obligation_analysis_single_content = obligation_analysis_single_content.replace(
                    "@OBLIGATION_INDEX", str(index_obligation + 1))

                sentence_analysis_content = sentence_analysis_content.replace("@OBLIGATION_STRUCTURE_VALIDATION",
                                                                              f"{obligation_analysis_single_content}\n\n@OBLIGATION_STRUCTURE_VALIDATION")

            sentence_analysis_content = sentence_analysis_content.replace("@OBLIGATION_STRUCTURE_VALIDATION", "")

            sentence_output_filename = f"Validation_{dataset_name}_A-{article}_P-{paragraph}_S-{index_sentence}.txt"
            final_output_path = output_base_path / sentence_output_filename

            with open(final_output_path, "w") as json_file:
                json_file.write(sentence_analysis_content)

    logging.warning(f"Skipped {skipped_sentences} sentences due to lack of analysis.")
