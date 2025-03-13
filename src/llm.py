import json
import logging
import os
import os
import time
from typing import Any

import tiktoken
from together import Together


def check_structure(data: Any, structure: Any) -> bool:
    """
    Recursively checks if the given `data` matches the specified `structure`.

    Args:
        data (Any): The JSON data to be validated.
        structure (Any): The expected structure for `data`.

    Returns:
        bool: True if `data` matches the `structure`, False otherwise.
    """
    if isinstance(structure, dict):
        if not isinstance(data, dict):
            return False
        for key, value_type in structure.items():
            if key not in data or not check_structure(data[key], value_type):
                return False
        return True
    elif isinstance(structure, list):
        if not isinstance(data, list):
            return False
        if not structure:
            return True
        return all(check_structure(item, structure[0]) for item in data)
    else:
        return isinstance(data, structure)


def setup_together_client() -> Together:
    """
    Sets up and initializes the Together client using the API key from environment variables.

    Returns:
        Together: An initialized Together client instance.

    Raises:
        EnvironmentError: If the TOGETHER_API_KEY environment variable is not set.
    """
    # Retrieve the API key from environment variables
    api_key = os.getenv('TOGETHER_API_KEY')

    if not api_key:
        raise EnvironmentError(
            "The 'TOGETHER_API_KEY' environment variable is not set. Please add it to your environment."
        )

    # Initialize and return the Together client
    return Together(api_key=api_key)


def execute_prompt(client, user_prompt, system_prompt=None, temperature=0.7,
                   model="meta-llama/Llama-3.3-70B-Instruct-Turbo", show_execution_time=False):
    """
    Execute prompts using the Together API and return the result.

    Args:
        user_prompt (str): The user's input prompt. Required.
        system_prompt (str, optional): The system's input prompt. Defaults to None.
        temperature (float, optional): The temperature for response generation. Defaults to 0.7.
        model (str, optional): The model to use for response generation. Defaults to 'meta-llama/Llama-3.3-70B-Instruct-Turbo'.
        show_execution_time (bool, optional): Whether to display execution time. Defaults to False.

    Returns:
        str: The content of the response from the assistant.

    Raises:
        ValueError: If user_prompt is not provided or invalid.
        Exception: If the API request fails.
    """
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    if system_prompt and not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string or None.")
    if not (0 <= temperature <= 1):
        raise ValueError("temperature must be between 0 and 1.")

    # Construct the messages list
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        # Measure execution time if required
        start_time = time.time() if show_execution_time else None

        # Call the Together API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=10000
        )

        if show_execution_time:
            elapsed_time = time.time() - start_time

        logging.info(
            response.choices[0].message
        )

        # Return the assistant's response content
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Failed to execute prompt: {str(e)}")


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Counts the number of tokens in the given text for a specific model.

    Args:
        text (str): The input string to tokenize.
        model (str): The model for which to calculate tokens, default is "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Initialize encoding for the specific model
        encoding = tiktoken.encoding_for_model(model)

        # Encode the text to get tokens
        tokens = encoding.encode(text)

        # Return the count of tokens
        return len(tokens)
    except Exception as e:
        raise ValueError(f"Error counting tokens for model '{model}': {e}")


def count_tokens_io(input_text: str, output_text: str, model: str = "gpt-3.5-turbo") -> dict:
    """
    Counts the number of tokens in the input and output texts for a specific model.

    Args:
        input_text (str): The input prompt to the LLM.
        output_text (str): The output response from the LLM.
        model (str): The model for which to calculate tokens, default is "gpt-3.5-turbo".

    Returns:
        dict: A dictionary with the token counts for input, output, and total.
    """
    input_tokens = count_tokens(input_text, model=model)
    output_tokens = count_tokens(output_text, model=model)
    total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
