import json
from pathlib import Path
from typing import Dict, List


def load_dataset(dataset_path: Path | str) -> List[Dict]:
    """
    Load potential obligations from a JSON file.

    Args:
        dataset_path (Path): The path to the JSON file containing potential obligations.

    Returns:
        List[Dict]: The dictionary representation of the JSON data.

    Raises:
        ValueError: If the file does not contain valid JSON.
        FileNotFoundError: If the file does not exist.
        Exception: For any other I/O errors.
    """
    try:
        if type(dataset_path) is str:
            dataset_path = Path(dataset_path)

        # Ensure the path exists and is a file
        if not dataset_path.is_file():
            raise FileNotFoundError(f"The file '{dataset_path}' does not exist or is not a file.")

        # Open the file and load JSON data
        with dataset_path.open('r', encoding='utf-8') as file:
            return json.load(file)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from '{dataset_path}': {e}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")
