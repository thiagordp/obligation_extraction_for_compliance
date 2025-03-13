import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tqdm
from together import Together
from together.cli.api.files import retrieve

from src.constants import OBLIGATION_FILTERING_SYS_PROMPT, OBLIGATION_FILTERING_USER_PROMPT, \
    OBLIGATION_FILTERING_POTENTIAL, \
    LLM_MODEL, TEMPERATURE, OBLIGATION_FILTERING_SHALL_FREQUENCY, VALIDATION_OBLIGATION_FILTERING_FOLDER, PROMPTS_PATH, \
    DATASET_PATH, OBLIGATION_FILTERING_VALIDATION_TEMPLATE, DATASETS_PATH, PARAGRAPHS_TO_ANALYSE, \
    VALIDATION_OBLIGATION_ANALYSIS_FOLDER, WAIT_FOR_NEXT_CALL
from src.llm import execute_prompt, count_tokens_io
from src.solve_references import retrieve_citation_contents, retrieve_context
from src.utils import extract_dict, increment_dict_count, display_stats


def obligation_filtering(client: Together, dataset: dict, retrieve_citations_inside_regulation: bool = False,
                         retrieve_surrounding_paragraphs: bool = False):
    def load_detection_obligation_prompt():
        system_prompt = open(OBLIGATION_FILTERING_SYS_PROMPT).read()
        user_prompt = open(OBLIGATION_FILTERING_USER_PROMPT).read()

        return system_prompt, user_prompt

    def store_frequent_shall_types():
        with open(OBLIGATION_FILTERING_SHALL_FREQUENCY, "w") as f:
            json.dump(frequent_types_shall, f, indent=4)

    def store_filtered_obligations():
        # Save the updated potential_obligations dictionary to a JSON file
        try:
            with open(OBLIGATION_FILTERING_POTENTIAL, "w") as json_file:
                json.dump(dataset, json_file, indent=4)
                logging.info("Saved potential_sentences to potential_sentences.json successfully.")
        except Exception as e:
            logging.error(f"Error while saving potential_sentences to JSON: {e}")

    logging.info("Starting obligation filtering...")

    sys_prompt, user_prompt = load_detection_obligation_prompt()

    # For each paragraph
    frequent_types_shall = {}
    frequent_obligation_analysis = {}

    tokens = {
        "input": [],
        "output": [],
        "total": []
    }
    for paragraph in tqdm.tqdm(dataset[:PARAGRAPHS_TO_ANALYSE]):
        potential_sentences = paragraph["potential_deontic"]
        par_id = paragraph["par_id"]

        # For each sentence
        for index_sentence, sentence in enumerate(potential_sentences):
            # Build prompt           
            sentence_user_prompt = user_prompt.replace("@Sentence", sentence["sentence"].strip())

            # Add surrounding context to the prompt
            context_paragraphs = retrieve_context(
                reference=par_id,
                dataset=dataset,
                retrieve_surrounding_paragraphs=retrieve_surrounding_paragraphs
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
            token_count = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            while not successful_attempt:
                try:
                    # Send the call
                    results = execute_prompt(
                        client=client,
                        model=LLM_MODEL,
                        user_prompt=sentence_user_prompt,
                        system_prompt=sys_prompt,
                        temperature=TEMPERATURE,
                        show_execution_time=False
                    )

                    # Receive the results
                    dict_results = extract_dict(results)

                    # ---  LLM Output Processing ---
                    token_count = count_tokens_io(
                        input_text=sentence_user_prompt + sys_prompt,
                        output_text=results
                    )

                    tokens["input"].append(token_count["input_tokens"])
                    tokens["output"].append(token_count["output_tokens"])
                    tokens["total"].append(token_count["total_tokens"])

                    shall_type = dict_results["classification"]
                    if shall_type not in frequent_types_shall.keys():
                        frequent_types_shall[shall_type] = 1
                    else:
                        frequent_types_shall[shall_type] += 1

                    sentence["filtering"] = {}
                    sentence["filtering"]["output"] = dict_results
                    sentence["filtering"]["tokens"] = token_count
                    sentence["filtering"]["prompts"] = {
                        # "system_prompt": sys_prompt, # In principle, not necessary
                        "user_prompt": sentence_user_prompt
                    }

                    successful_attempt = True
                except Exception as e:
                    logging.error(f"Error while sending request: {e}")
                    time.sleep(5)
                finally:
                    time.sleep(WAIT_FOR_NEXT_CALL)

    # Storing data
    store_frequent_shall_types()
    store_filtered_obligations()

    display_stats(tokens)

    logging.info("Finished obligation filtering")

    return dataset



def store_obligation_filtering_results(potential_deontic: dict) -> None:
    logging.info("Storing obligation filtering results...")
    output_base_path = (Path(VALIDATION_OBLIGATION_FILTERING_FOLDER) / OBLIGATION_FILTERING_SYS_PROMPT
                        .replace(PROMPTS_PATH, "").replace(".txt", ""))
    os.makedirs(name=output_base_path, exist_ok=True)

    dataset_name = DATASET_PATH.replace(DATASETS_PATH, "").replace(".json", "")
    llm_model = LLM_MODEL
    system_prompt = OBLIGATION_FILTERING_SYS_PROMPT
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Read the obligation filtering template.
    template_content = open(OBLIGATION_FILTERING_VALIDATION_TEMPLATE).read()

    template_content = template_content.replace("@DATASET", dataset_name)
    template_content = template_content.replace("@LLM_MODEL", llm_model)
    template_content = template_content.replace("@PROMPT", system_prompt)
    template_content = template_content.replace("@TIMESTAMP", timestamp)

    skipped_sentences = 0
    for paragraph_content in potential_deontic:
        article, paragraph = paragraph_content["par_id"].split(".")
        for index_sentence, sentence in enumerate(paragraph_content["potential_deontic"]):

            if "filtering" not in sentence.keys():
                skipped_sentences += 1
                continue
            filtering_results = sentence["filtering"]

            input_tokens = filtering_results["tokens"]["input_tokens"]
            output_tokens = filtering_results["tokens"]["output_tokens"]

            llm_input = filtering_results["prompts"]["user_prompt"]
            llm_output = dict(filtering_results["output"])

            sentence_validation_template = str(template_content)  # Copy value
            sentence_validation_template = sentence_validation_template.replace("@ARTICLE", article)
            sentence_validation_template = sentence_validation_template.replace("@PARAGRAPH", paragraph)
            sentence_validation_template = sentence_validation_template.replace("@SENTENCE", str(index_sentence + 1))
            sentence_validation_template = sentence_validation_template.replace("@INPUT_TOKENS", str(input_tokens))
            sentence_validation_template = sentence_validation_template.replace("@OUTPUT_TOKENS", str(output_tokens))
            sentence_validation_template = sentence_validation_template.replace("@LLM_INPUT", llm_input)
            sentence_validation_template = sentence_validation_template.replace("@LLM_OUTPUT",
                                                                                json.dumps(llm_output, indent=3))

            sentence_output_filename = f"Validation_{dataset_name}_A-{article}_P-{paragraph}_S-{index_sentence}.txt"

            final_output_path = output_base_path / sentence_output_filename
            with open(final_output_path, "w") as json_file:
                json_file.write(sentence_validation_template)

    logging.warning(f"Skipped {skipped_sentences} sentences due to lack of filtering data.")
