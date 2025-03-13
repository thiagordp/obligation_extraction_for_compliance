from datetime import datetime
import logging
import json
import re
import logging
from typing import Optional, Dict

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from src.constants import DATASET_NAME


def setup_logging(level=logging.INFO, log_dir="logs"):
    """
    Set up logging to output messages to both the console and a log file.

    Args:
        level (int): The minimum logging level. Defaults to logging.INFO.
        log_dir (str): Directory where log files will be stored. Defaults to 'logs'.

    Returns:
        None
    """
    # Ensure the log directory exists
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Get current timestamp for log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_dir_path / f"log_{DATASET_NAME}_{timestamp}.log"

    # Define the log format to include timestamp, file, function, and message
    log_format = (
        "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
    )

    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(level)  # Set the minimum logging level

    # Avoid adding duplicate handlers if the logger is already set up
    if not logger.hasHandlers():
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

        # Create a console handler to output logs to the screen
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Log an initialization message
    logger.info("Logging has been successfully set up.")
    logger.info(f"Logs will be saved to: {log_filename}")


def display_stats(tokens):
    logging.info("Token Statistics:")
    for key, values in tokens.items():
        median = np.median(values)
        std_dev = np.std(values, ddof=1)  # Sample standard deviation (ddof=1)
        logging.info(f"{key.capitalize()} Tokens: Median = {median:.2f}, Std Dev = {std_dev:.2f}")


def extract_dict(data: str) -> Optional[Dict]:
    """
    Extracts a dictionary from a string containing a JSON object enclosed in triple backticks (```json).

    Args:
        data (str): The string containing the JSON object.

    Returns:
        Optional[Dict]: The extracted dictionary, or None if no valid JSON object is found.

    Logs:
        - Error messages if JSON decoding fails.
    """
    # Normalize the string to handle variations in JSON code block markers
    data = data.replace("```json", "```")

    # Extract the content within triple backticks
    json_match = re.search(r'```(.*?)```', data, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()  # Extract and clean the matched JSON string
        try:
            return json.loads(json_str)  # Attempt to parse the JSON string into a dictionary
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
    else:
        logging.warning("No JSON object enclosed in triple backticks found.")
        print(data)

    raise Exception("JSON not found")


def increment_dict_count(target: dict, key: str) -> None:
    """
    Increment the count for a given key in the dictionary.

    If the key doesn't exist, initialize it with 0 and then increment it.

    Parameters:
    - target (dict): The dictionary to update.
    - key (str): The key whose count should be incremented.
    """

    target[key] = target.get(key, 0) + 1
