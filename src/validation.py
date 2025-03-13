import glob
import json
import os
import time
import zipfile
from pathlib import Path
import re

import numpy as np
import pandas as pd
from nltk import accuracy
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DistributedSampler

from src.constants import VALIDATION_RESULTS_FOLDER

# DATASET = "GDPR"
DATASET = "AI_Act"
# DATASET = "DSA"
#TASK = "obligation_analysis"
TASK = "obligation_filtering"
COMPARISON_PAIRS = [
    ["R1", "R4", "R5"],
    ["R2", "R3", "R6"],
]
COUNT_DISAGREEMENTS = 0


#
# Obligation Filtering
#
def evaluate_filtering_results():
    pass


#
# Obligation Analysis
#
def evaluate_analysis_results():
    pass


def retrieve_filtering_results(target: str):
    pattern = {
        "Dataset": r"Dataset:\s+(.*)",
        "Article": r"Article:\s+(\d+)",
        "Paragraph": r"Paragraph:\s+(\d+)",
        "Sentence": r"Sentence:\s+(\d+)",
        "LLM Model": r"LLM Model:\s+(.+)",
        "Prompt": r"Prompt:\s+(.+)",
        "Timestamp": r"Timestamp:\s+(.+)",
        "In Tokens": r"In Tokens:\s+(\d+)",
        "Out Tokens": r"Out Tokens:\s+(\d+)",
        "Classification": r"Classification:\s+\[(\d+|X)\]",
        "Justification": r"Justification:\s+\[(\d+|X)\]"
    }

    extracted_info = {}
    for key, regex in pattern.items():
        match = re.search(regex, target)
        if match:
            value = match.group(1)

            # Convert specific numeric fields and handle "X" as 0
            if key in ["In Tokens", "Out Tokens", "Classification", "Justification"]:
                extracted_info[key] = 0 if value == "X" else int(value)
            else:
                extracted_info[key] = value

    return extracted_info


def retrieve_analysis_results(target: str):
    pattern = {
        "Dataset": r"Dataset:\s+(.*)",
        "Article": r"Article:\s+(\d+)",
        "Paragraph": r"Paragraph:\s+(\d+)",
        "Sentence": r"Sentence:\s+(\d+)",
        "LLM Model": r"LLM Model:\s+(.+)",
        "Prompt": r"Prompt:\s+(.+)",
        "Timestamp": r"Timestamp:\s+(.+)",
        "In Tokens": r"In Tokens:\s+(\d+)",
        "Out Tokens": r"Out Tokens:\s+(\d+)",
        "No of obligations identified": r"No of obligations identified:\s+\[(\d+)\]"
    }

    base_obligation = {}
    for key, regex in pattern.items():
        match = re.search(regex, target)
        if match:
            value = match.group(1)
            base_obligation[key] = int(value) if value.isdigit() else value

    # Extract Obligation Evaluations
    obligation_evaluations = []
    obligations_identified = base_obligation.get("No of obligations identified", 0)

    # extracted_info["No of obligations identified"] = obligations_identified

    # Extract obligations
    obligation_pattern = re.compile(
        r"### Obligation (\d+) \(Predicate: '([^']+)'\)\n\n"
        r"1\. ObligationTypeClassification:\s+\[(\d+)\]\n"
        r"2\. Addressees:\n"
        r"\s+- Value\s+\[(\d+)\]\n"
        r"\s+- Extraction Method\s+\[(\d+)\]\n"
        r"3\. Predicates:\n"
        r"\s+- Value\s+\[(\d+)\]\n"
        r"\s+- Extraction Method\s+\[(\d+)\]\n"
        r"4\. Objects:\n"
        r"\s+- Value\s+\[(\d+)\]\n"
        r"\s+- Extraction Method\s+\[(\d+)\]\n"
        r"5\. Specifications:\n"
        r"\s+- Value\s+\[(\d+)\]\n"
        r"\s+- Extraction Method\s+\[(\d+)\]\n"
        r"6\. Pre-Conditions\n"
        r"\s+- Value\s+\[(\d+)\]\n"
        r"\s+- Extraction Method\s+\[(\d+)\]\n"
        r"7\. Beneficiaries\n"
        r"\s+- Value\s+\[(\d+)\]\n"
        r"\s+- Extraction Method\s+\[(\d+)\]"
    )

    for match in obligation_pattern.finditer(target):
        obligation_meta = dict(base_obligation)

        obligation_number = int(match.group(1))
        predicate = match.group(2)

        obligation_data = {
            "Obligation Number": obligation_number,
            "Predicate": predicate,
            "ObligationTypeClassification": int(match.group(3)),
            "Addressees": {
                "Value": int(match.group(4)),
                "Extraction Method": int(match.group(5))
            },
            "Predicates": {
                "Value": int(match.group(6)),
                "Extraction Method": int(match.group(7))
            },
            "Objects": {
                "Value": int(match.group(8)),
                "Extraction Method": int(match.group(9))
            },
            "Specifications": {
                "Value": int(match.group(10)),
                "Extraction Method": int(match.group(11))
            },
            "Pre-Conditions": {
                "Value": int(match.group(12)),
                "Extraction Method": int(match.group(13))
            },
            "Beneficiaries": {
                "Value": int(match.group(14)),
                "Extraction Method": int(match.group(15))
            }
        }

        obligation_data = {**obligation_meta, **obligation_data}
        if obligations_identified == 1:
            obligation_evaluations.append(obligation_data)
        else:
            # Only add obligations with at least one non-zero value
            if any(value != 0 for section in obligation_data.values() if isinstance(section, dict) for value in
                   section.values()) or obligation_data["ObligationTypeClassification"] != 0:
                obligation_evaluations.append(obligation_data)

    return obligation_evaluations


def load_file_results(target):
    content = open(target).read()

    if TASK == "obligation_filtering":
        results = retrieve_filtering_results(content)

        if "Classification" not in results or "Justification" not in results:
            raise Exception(f"Missing classification or justification for {target}")

        results["ID"] = f"{int(results['Article']):03d}.{int(results['Paragraph']):03d}.{int(results['Sentence']):03d}"
        results["TASK"] = TASK

        return results
    elif TASK == "obligation_analysis":
        results = retrieve_analysis_results(content)
        # Mudar o output results pra uma lista de dicts.
        for result in results:
            result[
                "ID"] = f"{int(result['Article']):03d}.{int(result['Paragraph']):03d}.{int(result['Sentence']):03d}.{int(result['Obligation Number']):03d}"
            result["TASK"] = TASK

        return results

    # print(f"Results for {target}: {json.dumps(results, indent=4)}")


def validation_analysis_results():
    global COUNT_DISAGREEMENTS

    def load_results(reviewer):
        final_result = []
        path_to_results = Path(VALIDATION_RESULTS_FOLDER) / reviewer / TASK / DATASET
        validation_files = list(path_to_results.glob("*.txt"))

        for file in validation_files:
            results = load_file_results(file)
            final_result.append(results)

        return final_result

    def retrieve_by_key_in_array(array, target_id):
        for elem in array:
            if elem["ID"] == target_id:
                return elem
        return None

    def compare_results(r1, r2):
        l1 = len(r1)
        l2 = len(r2)

        if l1 != l2:
            raise Exception(f"Two results have different lengths: {l1} vs {l2}")

        for doc_i in range(l1):
            for obligation_i in r1[doc_i]:
                ref_id = obligation_i["ID"]
                result_r1 = retrieve_by_key_in_array(r1, ref_id)
                result_r2 = retrieve_by_key_in_array(r2, ref_id)

    def add_to_results(reviewer_pair, data):

        for provision_results in data:
            for doc in provision_results:
                doc_id = doc["ID"]

                elements = [
                    "Addressees",
                    "Objects",
                    "Predicates",
                    "Specifications",
                    "Pre-Conditions",
                    "Beneficiaries",
                ]
                if doc_id not in final_results:

                    final_results[doc_id] = {
                        "ReviewerPair": reviewer_pair,
                        "ObligationTypeClassification": [doc["ObligationTypeClassification"]],
                        "Predicate": doc["Predicate"],
                    }
                    for element in elements:
                        value_key = f"{element}-Value"
                        extraction_key = f"{element}-Extraction Method"
                        final_results[doc_id][value_key] = [doc[element]["Value"]]
                        final_results[doc_id][extraction_key] = [doc[element]["Extraction Method"]]
                else:
                    final_results[doc_id]["ObligationTypeClassification"].append(doc["ObligationTypeClassification"])
                    for element in elements:
                        value_key = f"{element}-Value"
                        extraction_key = f"{element}-Extraction Method"
                        final_results[doc_id][value_key].append(doc[element]["Value"])
                        final_results[doc_id][extraction_key].append(doc[element]["Extraction Method"])

    # unzip_files()
    final_results = {

    }

    # Loop through comparison pairs
    for pair in COMPARISON_PAIRS:
        reviewer_a, reviewer_b, reviewer_c = pair

        results_r1 = load_results(reviewer_a)
        results_r2 = load_results(reviewer_b)
        results_r3 = load_results(reviewer_c)

        add_to_results(f"{reviewer_a}-{reviewer_b}-{reviewer_c}", results_r1)
        add_to_results(f"{reviewer_a}-{reviewer_b}-{reviewer_c}", results_r2)
        add_to_results(f"{reviewer_a}-{reviewer_b}-{reviewer_c}", results_r3)


    print(final_results)

    # Compare the results
    # compare_results(results_r1, results_r2)

    # Display DataFrame

    # Convert dictionary into DataFrame
    df = pd.DataFrame.from_dict(final_results, orient='index').reset_index()
    # Rename columns
    df.rename(columns={'index': 'ID'}, inplace=True)

    output_path = f"data/validation/results_analysis/{TASK}_{DATASET}.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist'
    df.to_excel(output_path, index=False)

    elements = [
        "ObligationTypeClassification",
        "Addressees-Value",
        "Addressees-Extraction Method",
        "Objects-Value",
        "Objects-Extraction Method",
        "Predicates-Value",
        "Predicates-Extraction Method",
        "Specifications-Value",
        "Specifications-Extraction Method",
        "Pre-Conditions-Value",
        "Pre-Conditions-Extraction Method",
        "Beneficiaries-Value",
        "Beneficiaries-Extraction Method",
    ]

    # Compute agreement/disagreement for each entry
    for key, value in final_results.items():
        for element in elements:
            value_mean = sum(value[element]) / len(value[element])
            value[element] = int(np.round(value_mean))
            COUNT_DISAGREEMENTS += 1 if value_mean in [0, 1] else 0
            value[element + "_Agreement"] = 1 if value_mean in [0, 1] else 0

    # Convert dictionary into DataFrame
    df = pd.DataFrame.from_dict(final_results, orient='index').reset_index()
    # Rename columns
    df.rename(columns={'index': 'ID'}, inplace=True)

    def calculate_metrics(column):
        """ Compute classification metrics for a given column (Classification or Justification). """
        correct_answers = (df[column] >= 0.5).sum()  # Correct answers (1 means correct)
        wrong_answers = (df[column] < 0.5).sum()  # Wrong answers (0 means wrong)

        accuracy = correct_answers / (correct_answers + wrong_answers) if (correct_answers + wrong_answers) > 0 else 0

        return {
            "Accuracy": accuracy
        }

    accuracy_results = {}
    for element in elements:
        element_metrics = calculate_metrics(element)
        # print(f"\n=== Metrics for {element} ===")

        accuracy_results[element] = element_metrics["Accuracy"]
        #
        # for key, value in element_metrics.items():
        #     print(f"{key}: {value:.4f}")
    accuracy_results_agreement = {}
    for element in elements:
        element_metrics = calculate_metrics(element + "_Agreement")

        accuracy_results_agreement[element] = element_metrics["Accuracy"]
        # print(f"\n=== Agreement Metrics for {element} ===")
        # for key, value in element_metrics.items():
        #     print(f"{key}: {value:.4f}")

    # print("Accuracy")
    # print(json.dumps(accuracy_results, indent=4))
    #
    # print("Agreement")
    # print(json.dumps(accuracy_results_agreement, indent=4))

    # Create DataFrame
    df = pd.DataFrame({
        "Lines": accuracy_results.keys(),
        "Accuracy": accuracy_results.values(),
        "Agreement": accuracy_results_agreement.values()
    })

    df.to_excel(f"data/validation/results_analysis/{TASK}_{DATASET}_Accuracy_Agreement.xlsx", index=False)


def validation_filtering_results():
    def unzip_files():
        # Loop through all files in the folder
        for file in os.listdir(VALIDATION_RESULTS_FOLDER):
            if file.endswith(".zip"):  # Process only .zip files
                zip_path = os.path.join(VALIDATION_RESULTS_FOLDER, file)
                extract_folder = os.path.join(VALIDATION_RESULTS_FOLDER)  # Create a folder named after the zip file

                # Create directory if it doesn't exist
                os.makedirs(extract_folder, exist_ok=True)

                # Extract contents
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

                print(f"Extracted {file} into {extract_folder}")

    def load_results(reviewer):
        final_results = []
        path_to_results = Path(VALIDATION_RESULTS_FOLDER) / reviewer / TASK / DATASET
        validation_files = list(path_to_results.glob("*.txt"))

        for file in validation_files:
            results = load_file_results(file)
            final_results.append(results)

        return final_results

    def retrieve_by_key_in_array(array, target_id):
        for elem in array:
            if elem["ID"] == target_id:
                return elem
        return None

    def compare_results(r1, r2):
        l1 = len(r1)
        l2 = len(r2)

        if l1 != l2:
            raise Exception(f"Two results have different lengths: {l1} vs {l2}")

        for i in range(l1):
            ref_id = r1[i]["ID"]
            result_r1 = retrieve_by_key_in_array(r1, ref_id)
            result_r2 = retrieve_by_key_in_array(r2, ref_id)
            pass

    def add_to_results(reviewer_pair, data):
        for doc in data:
            doc_id = doc["ID"]
            if doc_id not in final_results:
                final_results[doc_id] = {
                    "ReviewerPair": reviewer_pair,
                    "Classification": [doc["Classification"]],
                    "Justification": [doc["Justification"]]
                }
            else:
                final_results[doc_id]["Classification"].append(doc["Classification"])
                final_results[doc_id]["Justification"].append(doc["Justification"])
        pass

    # unzip_files()
    final_results = {

    }

    # Loop through comparison pairs
    for pair in COMPARISON_PAIRS:
        reviewer_a, reviewer_b, reviewer_c = pair

        results_r1 = load_results(reviewer_a)
        results_r2 = load_results(reviewer_b)
        results_r3 = load_results(reviewer_c)

        add_to_results(f"{reviewer_a}-{reviewer_b}-{reviewer_c}", results_r1)
        add_to_results(f"{reviewer_a}-{reviewer_b}-{reviewer_c}", results_r2)
        add_to_results(f"{reviewer_a}-{reviewer_b}-{reviewer_c}", results_r3)

        # Compare the results
        # compare_results(results_r1, results_r2)

        # Display DataFrame

    def calculate_agreement(df: pd.DataFrame, column: str):
        """
        Calculates the agreement between two evaluators for a given column in a DataFrame.
        Assumes each cell contains a list of two evaluations.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column name containing evaluator assessments.

        Returns:
            dict: A dictionary with percentage agreement and Cohen's Kappa.
        """
        # Extract the evaluations and ensure they are valid
        evaluations = df[column].dropna().tolist()

        # Ensure each row contains exactly two evaluations
        valid_evaluations = [e for e in evaluations if isinstance(e, list) and len(e) == 3]
        if not valid_evaluations:
            raise ValueError("No valid evaluation pairs found in the column.")

        # Separate into two lists: evaluator 1 and evaluator 2
        evaluator_1, evaluator_2, evaluator_3 = zip(*valid_evaluations)

        # Compute percentage agreement
        agreements = [(int(e1 == e2) + int(e2 == e3) + int(e1 == e3)) / 3.0
                      for e1, e2, e3 in zip(evaluator_1, evaluator_2, evaluator_3)]

        percentage_agreement = np.mean(agreements)

        # Compute Cohen's Kappa
        kappa = cohen_kappa_score(evaluator_1, evaluator_2)

        return {"percentage_agreement": percentage_agreement, "cohen_kappa": kappa}

    global COUNT_DISAGREEMENTS
    # Convert dictionary into DataFrame
    df = pd.DataFrame.from_dict(final_results, orient='index').reset_index()
    # Rename columns
    df.rename(columns={'index': 'ID'}, inplace=True)

    output_path = f"data/validation/results_analysis/{TASK}_{DATASET}.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist'
    df.to_excel(output_path, index=False)

    # Calculate agreement for each entry
    agreement_classification = calculate_agreement(df, "Classification")
    agreement_justification = calculate_agreement(df, "Justification")

    print(agreement_classification, agreement_justification)

    # Compute agreement/disagreement for each entry
    for key, value in final_results.items():
        classification_mean = sum(value["Classification"]) / len(value["Classification"])
        justification_mean = sum(value["Justification"]) / len(value["Justification"])

        value["Classification"] = int(np.round(classification_mean))  # Floor to 0 or 1
        value["Justification"] = int(np.round(justification_mean))  # Floor to 0 or 1

        COUNT_DISAGREEMENTS += 1 if classification_mean in [0, 1] else 0
        COUNT_DISAGREEMENTS += 1 if justification_mean in [0, 1] else 0
        # value["Classification_Agreement"] = 1 if classification_mean in [0, 1] else 0
        # value["Justification_Agreement"] = 1 if justification_mean in [0, 1] else 0

    # Convert dictionary into DataFrame
    df = pd.DataFrame.from_dict(final_results, orient='index').reset_index()
    # Rename columns
    df.rename(columns={'index': 'ID'}, inplace=True)

    def calculate_metrics(column):
        """ Compute classification metrics for a given column (Classification or Justification). """
        correct_answers = (df[column] >= 0.5).sum()  # Correct answers (1 means correct)
        wrong_answers = (df[column] < 0.5).sum()  # Wrong answers (0 means wrong)

        accuracy = correct_answers / (correct_answers + wrong_answers) if (correct_answers + wrong_answers) > 0 else 0

        return {
            "Accuracy": accuracy
        }

    # Compute metrics separately
    classification_metrics = calculate_metrics("Classification")
    justification_metrics = calculate_metrics("Justification")


    # # Calculate agreement percentage
    # classification_agreement_rate = df["Classification_Agreement"].mean()
    # justification_agreement_rate = df["Justification_Agreement"].mean()

    # Display results
    print("=== Classification Metrics ===")
    for key, value in classification_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Justification Metrics ===")
    for key, value in justification_metrics.items():
        print(f"{key}: {value:.4f}")

    elements = [
        "Classification",
        "Justification"
    ]

    accuracy_results = {}
    for element in elements:
        element_metrics = calculate_metrics(element)
        # print(f"\n=== Metrics for {element} ===")

        accuracy_results[element] = element_metrics["Accuracy"]
        #
        # for key, value in element_metrics.items():
        #     print(f"{key}: {value:.4f}")
    accuracy_results_agreement = {
        "Classification": agreement_classification["percentage_agreement"],
        "Justification": agreement_justification["percentage_agreement"]
    }


    # print(f"\n=== Agreement Metrics for {element} ===")
    # for key, value in element_metrics.items():
    #     print(f"{key}: {value:.4f}")

    # print("Accuracy")
    # print(json.dumps(accuracy_results, indent=4))
    #
    # print("Agreement")
    # print(json.dumps(accuracy_results_agreement, indent=4))

    # Create DataFrame
    df = pd.DataFrame({
        "Lines": accuracy_results.keys(),
        "Accuracy": accuracy_results.values(),
        "Agreement": accuracy_results_agreement.values()
    })
    print("GLOBAL", COUNT_DISAGREEMENTS)
    df.to_excel(f"data/validation/results_analysis/{TASK}_{DATASET}_Accuracy_Agreement.xlsx", index=False)
    # # Display agreement rates
    # print(f"Classification Agreement Rate: {classification_agreement_rate:.4f}")
    # print(f"Justification Agreement Rate: {justification_agreement_rate:.4f}")

# Unzip files in folder.


# AIA     0.8621
# DSA     0.8333
# GDPR    0.8276
