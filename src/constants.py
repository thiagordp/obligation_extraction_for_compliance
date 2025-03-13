import os

# LLM Setup
#LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DATASET = "AI_Act"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TEMPERATURE = 0.6

WAIT_FOR_NEXT_CALL = 1

OBLIGATION_FILTERING_RETRIEVE_SURROUNDING_PARAGRAPHS = False
OBLIGATION_FILTERING_RETRIEVE_CITATIONS_INSIDE_REGULATION = False

OBLIGATION_ANALYSIS_RETRIEVE_SURROUNDING_PARAGRAPHS = True
OBLIGATION_ANALYSIS_RETRIEVE_CITATIONS_INSIDE_REGULATION = True

PARAGRAPHS_TO_ANALYSE = 2  # Example: limit for testing; replace with None for full processing
RETRY_DELAY = 5  # Seconds between retries

###### SHALL types of interest ######
SHALL_TYPES_OF_INTEREST = ["Deontic obligation", "Deontic prohibition"]

###### Paths #####
DATASETS_PATH = "data/raw/datasets/"
# Default value if no argument is provided
DATASET_NAME = os.getenv("DATASET_NAME", "")

os.makedirs(name=DATASETS_PATH, exist_ok=True)
DATASET_PATH = DATASETS_PATH + DATASET_NAME + ".json"

PROMPTS_PATH = "data/raw/prompts/"
os.makedirs(name=PROMPTS_PATH, exist_ok=True)
OBLIGATION_FILTERING_SYS_PROMPT_FILE = "obligation_filtering_system.txt"
OBLIGATION_FILTERING_SYS_PROMPT = PROMPTS_PATH + OBLIGATION_FILTERING_SYS_PROMPT_FILE
OBLIGATION_FILTERING_USER_PROMPT = PROMPTS_PATH + "obligation_filtering_user.txt"

OBLIGATION_ANALYSIS_SYS_PROMPT_FILE = "obligation_analysis_system.txt"
OBLIGATION_ANALYSIS_SYS_PROMPT = PROMPTS_PATH + OBLIGATION_ANALYSIS_SYS_PROMPT_FILE
OBLIGATION_ANALYSIS_USER_PROMPT = PROMPTS_PATH + "obligation_analysis_user.txt"

ofspf_name = OBLIGATION_ANALYSIS_SYS_PROMPT_FILE.replace(".txt", "")
OBLIGATION_FILTERING_FOLDER = f"data/processed/obligations_filtered/{DATASET_NAME}/{ofspf_name}/"
os.makedirs(name=OBLIGATION_FILTERING_FOLDER, exist_ok=True)
OBLIGATION_FILTERING_POTENTIAL = OBLIGATION_FILTERING_FOLDER + f"{DATASET_NAME}.json"
OBLIGATION_FILTERING_SHALL_FREQUENCY = OBLIGATION_FILTERING_FOLDER + "counting_shall_types.json"

oaspf_name = OBLIGATION_ANALYSIS_SYS_PROMPT_FILE.replace(".txt", "")
OBLIGATION_ANALYSIS_FOLDER = f"data/processed/obligations_analysis/{DATASET_NAME}/{oaspf_name}/"
os.makedirs(name=OBLIGATION_ANALYSIS_FOLDER, exist_ok=True)
OBLIGATION_ANALYSIS_RESULTS = OBLIGATION_ANALYSIS_FOLDER + f"{DATASET_NAME}.json"

VALIDATION_FOLDER = "data/validation/"
VALIDATION_OBLIGATION_FILTERING_FOLDER = VALIDATION_FOLDER + f"obligation_filtering/{DATASET_NAME}/"
os.makedirs(name=VALIDATION_OBLIGATION_FILTERING_FOLDER, exist_ok=True)
VALIDATION_OBLIGATION_ANALYSIS_FOLDER = VALIDATION_FOLDER + f"obligation_analysis/{DATASET_NAME}/"
os.makedirs(name=VALIDATION_OBLIGATION_FILTERING_FOLDER, exist_ok=True)

TEMPLATES_PATH = "data/templates/"
os.makedirs(name=TEMPLATES_PATH, exist_ok=True)
OBLIGATION_FILTERING_VALIDATION_TEMPLATE = TEMPLATES_PATH + "obligation_filtering_labeling.txt"

OBLIGATION_ANALYSIS_VALIDATION_TEMPLATE = TEMPLATES_PATH + "obligation_analysis_labeling.txt"
OBLIGATION_ANALYSIS_SINGLE_TEMPLATE = TEMPLATES_PATH + "obligation_analysis_single_obligation.txt"


#
# Validation results
#
VALIDATION_RESULTS_FOLDER = "data/validation/results/"
os.makedirs(name=VALIDATION_RESULTS_FOLDER, exist_ok=True)

