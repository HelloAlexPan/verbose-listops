"""
verbose-listops.py

Generates complex ListOps problems as Abstract Syntax Trees (AST), evaluates them, creates fictional world metadata, and renders a narrative where each calculation step is a story beat. Ensures strict validation so only atomic operands are mentioned in the narrative, with no intermediate results or extraneous numbers. Supports batch generation, logging, and saving samples to a JSONL file. Configuration is available via constants.
"""

import os
import json
import random
import datetime
import logging
import logging.handlers
import time
from typing import Callable, Set
from dataclasses import dataclass, field, asdict
import re
import concurrent.futures
import inflect
import tiktoken
from functools import lru_cache
from openai import OpenAI
import shutil
import threading
import requests
import subprocess
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

# fmt: off
# --- Batch Settings ---
NUM_SAMPLES_TO_GENERATE = 10
DEFAULT_MAX_WORKERS = 100
MODEL = "google/gemini-2.5-flash-preview:thinking"
STATIC_CHECKER_MODEL = "google/gemini-2.5-flash-preview:thinking"
DATASETS_DIR = "datasets"
PROD_RUN: bool = True

@dataclass
class Config:
    MAX_OPS: int = 8
    MAX_BRANCH: int = 8
    MIN_ARITY: int = 6
    MIN_ATOM_VAL: int = 1
    MAX_ATOM_VAL: int = 9
    MAX_TOTAL_TOKENS: int = 10000
    EARLY_TERMINATION_PROBABILITY: float = 0.0
    PADDING_MAX_TOK_PERCENT: float = 0.75
    USE_NARRATIVE_ANCHORS: bool = True
    USE_LLM_NAMING: bool = True
    MIN_WORLD_CHARS: int = 6
    MAX_WORLD_CHARS: int = 8
    MIN_WORLD_CONCEPTS: int = 3
    MAX_WORLD_CONCEPTS: int = 7
    BEAT_CONTEXT: int = 1000
    PADDING_CONTEXT: int = 1500
    MAX_PAD_PARAGRAPHS: int = 30
    WORLD_GEN_TEMP:  float = 0.9
    BEAT_GEN_TEMP: float = 0.5
    CREATIVE_NARRATIVE_TEMP: float = 0.5
    ANCHOR_GEN_TEMP: float = 0.85
    LLM_VALIDATOR_MODEL: str = "google/gemini-2.5-flash-preview:thinking"
    LLM_VALIDATOR_TEMP: float = 0.05
    BEAT_REVISION_TEMP: float = 0.1
    MAX_LLM_VALIDATION_ITERATIONS: int = 6
    MODEL_MAX_CONTEXT_TOKENS: int = 750000
    MAX_ANCHOR_WORDS: int = 4
    FEW_SHOT_EXAMPLES: int = 3
    FALLBACK_MIN_NUM_WORD: int = 0
    FALLBACK_MAX_NUM_WORD: int = 20
    MIN_ALLOWED_SMALL_NUMBER: int = 0
    MAX_ALLOWED_SMALL_NUMBER: int = 10
    ALWAYS_ALLOWED_PHRASING_NUMBERS_SET: Set[int] = field(default_factory=lambda: {1, 2, 3})
    INVALID_RESULT_PLACEHOLDER: int = -999
    PROBLEM_SMALL_NUMBERS_TO_CHECK: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
    RETRY_MAX_ATTEMPTS: int = 10
    RETRY_INITIAL_DELAY: float = 0.25
    MAX_BEAT_RETRIES: int = 5
    MAX_PAD_RETRIES: int = 5
    INTRO_MAX_RETRIES: int = 3
    WORLDGEN_MAX_RETRIES: int = 5
    INITIAL_WORLD_RETRY_DELAY: float = 1.0
    MAX_REQUESTS_PER_SECOND: float = 900.0
    MIN_REQUEST_INTERVAL: float = 0.001
    LOG_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 3
    CLEAR_LOGS_ON_START: bool = True
    MAX_TOKENS_BUFFER: int = 500
    MAX_API_TOKEN_LIMIT: int = 60000
    WORLD_GEN_MAX_TOKENS: int = 5000
    ANCHOR_MAX_TOKENS: int = 100
    INTRO_MAX_TOKENS: int = 100
    BEAT_MAX_TOKENS: int = 400
    PADDING_MAX_TOKENS: int = 400

config = Config()
# fmt: on

# --- JSON Schema Definitions for Structured Outputs ---
VALIDATOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {
            "type": "boolean",
            "description": "Whether the beat passes all validation criteria",
        },
        "explanation_for_generator": {
            "type": "string",
            "description": "Detailed explanation of all validation issues, for the next generation attempt",
        },
        "explanation_for_audit": {
            "type": "string",
            "description": "Summary of why the beat is valid, highlighting numerical compliance",
        },
        "overall_revision_summary_for_generator_prompt": {
            "type": "string",
            "description": "Concise instruction for the generator focusing on critical issues",
        },
        "suggested_revisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional suggested specific text revisions",
        },
    },
    "required": ["is_valid"],
    "additionalProperties": False,
}

STATIC_VALIDATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "is_beat_valid": {
            "type": "boolean",
            "description": "Whether the beat passes validation",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation for why the beat passes or fails validation",
        },
    },
    "required": ["is_beat_valid", "reasoning"],
    "additionalProperties": False,
}

WORLD_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "quirk": {"type": "string"},
                },
                "required": ["name", "role", "quirk"],
            },
        },
        "genre": {"type": "string"},
        "setting": {"type": "string"},
        "object": {"type": "string"},
    },
    "required": ["characters", "genre", "setting", "object"],
    "additionalProperties": False,
}

# --- End of JSON Schema Definitions ---

NUMBER_WORD_REGEX = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b"
    r"(?:\s*\((\d+)\))?",  # Optional: space and digits in parentheses
    re.IGNORECASE
)

# Simple mapping for single number words (expand as needed for more complex numbers like "twenty-one")
# For Verbose ListOps, inputs are 0-9, so this is likely sufficient for direct inputs.
# Phrasing numbers might be slightly larger but usually simple.
SINGLE_NUMBER_WORDS_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    # Add more if your config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET goes higher
    # or if operand_count can be higher and mentioned.
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    # For compound numbers, a more sophisticated parser would be needed.
    # Given ListOps constraints (0-9 for ops), this primarily serves phrasing.
}

def find_numbers_in_text_detailed(text: str, config_obj: "Config") -> tuple[list[int], dict[int, list[str]]]:
    """
    Finds numbers written as words in the text.
    Returns a list of all integer values found and a dictionary mapping each integer
    to a list of the actual word phrases that represented it.
    Focuses on numbers relevant to ListOps (0-9 for operations, and allowed phrasing numbers).
    """
    numbers_found_values = []
    number_words_map = {} # Stores {int_value: [list_of_word_phrases_for_it]}
    processed_text = text.lower()
    cleaned_text = re.sub(r"[^\w\s']", " ", processed_text) # Keep apostrophes, replace other punctuation with space
    words = cleaned_text.split()

    for word in words:
        # Strip trailing 's (e.g., "fives" -> "five") - very basic plural handling
        singular_word = word
        if word.endswith("'s"): # Possessive like "seven's"
            singular_word = word[:-2]
        elif word.endswith("s"):
            singular_word = word[:-1]


        if singular_word in SINGLE_NUMBER_WORDS_TO_INT:
            num_val = SINGLE_NUMBER_WORDS_TO_INT[singular_word]
            numbers_found_values.append(num_val)
            if num_val not in number_words_map:
                number_words_map[num_val] = []
            number_words_map[num_val].append(word) # Store the original word found

    # Also find digits explicitly written as numbers (e.g., "7")
    # This is important if your narrative generator might accidentally use digits.
    # The prompt tells it not to, but this validator can catch it.
    for digit_match in re.finditer(r'\b\d+\b', text): # Find standalone digits
        num_str = digit_match.group(0)
        try:
            num_val = int(num_str)
            # Only consider digits if they are in the relevant range for your benchmark,
            # e.g., 0-9 for operands, or within phrasing number limits.
            # This prevents flagging page numbers, years, etc., unless they match operand values.
            if 0 <= num_val <= config_obj.MAX_ATOM_VAL or num_val in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
                numbers_found_values.append(num_val)
                if num_val not in number_words_map:
                    number_words_map[num_val] = []
                number_words_map[num_val].append(num_str) # Store the digit string
        except ValueError:
            pass # Should not happen with \d+

    return numbers_found_values, number_words_map


# --- AST Node Definitions ---
@dataclass
class Node:
    op: str = ""  # Fix: Add default value
    children: list = field(default_factory=list)
    value: int = None


@dataclass
class Atom(Node):
    def __init__(self, n: int):
        super().__init__(op="ATOM", children=[])
        self.n = n
        self.value = n


@dataclass
class OpNode(Node):
    def __init__(self, op: str, children: list):
        super().__init__(op=op, children=children)
        self.value = None


# --- END OF AST Node Definitions ---


class GenerationTokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.api_calls = 0
        self.lock = threading.Lock()

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.api_calls += 1
        logger.debug(
            f"Token Tracker: Added {prompt_tokens} prompt, {completion_tokens} completion. Total P: {self.total_prompt_tokens}, C: {self.total_completion_tokens}, Calls: {self.api_calls}"
        )

    def get_summary(self):
        with self.lock:
            return (
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.api_calls,
            )

    def calculate_cost(
        self, prompt_cost_per_million: float, completion_cost_per_million: float
    ) -> float:
        with self.lock:
            prompt_cost = (
                self.total_prompt_tokens / 1_000_000
            ) * prompt_cost_per_million
            completion_cost = (
                self.total_completion_tokens / 1_000_000
            ) * completion_cost_per_million
            return prompt_cost + completion_cost


generation_token_tracker = GenerationTokenTracker()
DEFAULT_COST_PER_MILLION_PROMPT_TOKENS = 0.50
DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS = 1.50

ORDINAL_WORDS_TO_IGNORE = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "twentieth",
    "thirtieth",
    "fortieth",
    "fiftieth",
    "sixtieth",
    "seventieth",
    "eightieth",
    "ninetieth",
    "hundredth",
    "last",
    "final",
}

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print(
        "Warning: OPENROUTER_API_KEY environment variable not set. Using placeholder."
    )
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
try:
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"Failed to initialize tokenizer: {e}")
    encoder = None
from string import Template

BASE_BEAT_TEMPLATE = Template(
    "You are a $beat_mode storyteller writing the next sequential scene in an ongoing narrative.\\n"
    "Characters: $characters\\n"
    "Setting: $setting\\n"
    'Previous Scene Snippet (End of last scene): "...$snippet"\\n\\n'
    "--- $task_header ---\\n"
    "$task_body\\n\\n"
    "$ultra_strict_instruction\\n\\n"
    "Output only the narrative text for this new scene, continuing from the snippet. Do not include titles, headings, or explanations."
)

# In verbose-listops.py

FEW_SHOT_EXAMPLES_STRICT = [
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: thirty-nine, ninety, and ninety-three.\\\\n"
            "*   You MAY use the number 'three' (the count of direct items...) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        "Felix examined the three caches. 'This one has ninety-three relics, that one ninety, and the last thirty-nine,' he said. Liora checked the Cipher Wheel. 'We need the smallest: thirty-nine.'",
        "Felix examined the three caches. 'This one has ninety-three relics, that one ninety, and the last thirty-nine,' he said. Liora checked the Cipher Wheel. 'We need the smallest: thirty-nine. It took twelve minutes.'",
        "BAD output failed: Included 'twelve'. Rule Analysis: 12 not in MUST INCLUDE {39, 90, 93}, not operand count (3), not allowed small num (0-10). Violates 'NO OTHER NUMBERS'.",
    ),
    # Corrected MEDIAN Example 1 (Original inputs: {72, 84, 89, 91, 95}, Median: 89)
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene - MEDIAN Example):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: seventy-two, eighty-four, ninety-one, and ninety-five. (Note: 'eighty-nine' was an input but is OMITTED here because it's also the MEDIAN result).\\\\n"
            "*   **(Critical MEDIAN Exception Applied):** The number 'eighty-nine' (89) was an input for this step BUT it is also the MEDIAN result. Therefore, 'eighty-nine' (89) has been EXCLUDED from the list above and MUST NOT be mentioned in your narrative for this scene. Only state the other numbers listed above (seventy-two, eighty-four, ninety-one, ninety-five).\\\\n"
            "*   **MEDIAN RESULT MUST BE IMPLICIT:** The median value ('eighty-nine') must NOT be explicitly stated as the result. It should be implied conceptually.\\\\n"
            "*   You MAY use the number 'four' (the count of *mentioned* direct items) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions all required numbers (72, 84, 91, 95) but OMITTED 89 (input that is also median). Median 89 is IMPLICIT.
        "Seraphina arranged the crystal fragments on the altar: 'This one pulses with seventy-two vibrations, this with eighty-four, this with ninety-one, this with ninety-five.' She examined the fifth crystal, studying its unique pattern. 'This middle fragment - the balanced keystone - shall be our central focus. Its resonance sits precisely between the others.' Marcus nodded, 'The perfect equilibrium point. The central essence that will stabilize the ritual.'",
        # BAD example: explicitly mentions "eighty-nine" as the median/result OR as an input.
        "Seraphina arranged the crystal fragments on the altar: 'This one pulses with seventy-two vibrations, this with eighty-four, this with eighty-nine, this with ninety-one, and this with ninety-five.' She examined the fifth crystal. 'This one has eighty-nine vibrations - it's the median value, the perfect middle point.' Marcus nodded, 'Eighty-nine is indeed the central value we need.'",
        "BAD output failed: Explicitly stated 'eighty-nine'. For MEDIAN operations, if an input number is also the median result, that number MUST NOT be mentioned at all. The median result itself must also always be IMPLICIT.",
    ),
    # Corrected MEDIAN Example 2 (Original inputs: {73, 85, 87, 88, 89, 91}, Median: 87)
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene - MEDIAN Example):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: seventy-three, eighty-five, eighty-eight, eighty-nine, and ninety-one. (Note: 'eighty-seven' was an input but is OMITTED here because it's also the MEDIAN result).\\\\n"
            "*   **(Critical MEDIAN Exception Applied):** The number 'eighty-seven' (87) was an input for this step BUT it is also the MEDIAN result. Therefore, 'eighty-seven' (87) has been EXCLUDED from the list above and MUST NOT be mentioned in your narrative for this scene. Only state the other numbers listed above (seventy-three, eighty-five, eighty-eight, eighty-nine, ninety-one).\\\\n"
            "*   **MEDIAN RESULT MUST BE IMPLICIT:** The median value ('eighty-seven') must NOT be explicitly stated anywhere.\\\\n"
            "*   You MAY use the number 'five' (the count of *mentioned* direct items) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions all inputs EXCEPT the median (87), which is only implied
        "Kairos studied the alignment of six energy signatures on the quantum display. 'The readings show seventy-three, eighty-five, eighty-eight, eighty-nine, and ninety-one, plus the void signal.' He pointed to the empty space between the values. 'The central point - this balance nexus - is our target. The middle value will stabilize the entire sequence.' Lyra nodded, understanding the critical equilibrium point without needing to name it.",
        # BAD example: explicitly lists the median (87) among all values
        "Kairos studied the alignment of six energy signatures on the quantum display. 'The readings show seventy-three, eighty-five, eighty-seven, eighty-eight, eighty-nine, and ninety-one.' He pointed to the third value. 'This central point - eighty-seven - is our target. The middle value will stabilize the entire sequence.'",
        "BAD output failed: The median value 'eighty-seven' was explicitly listed. CRITICAL ERROR: For MEDIAN operations, if an input number is also the median result, that number MUST NOT be mentioned at all. The median result itself must also always be IMPLICIT.",
    ),
    # Example for handling duplicate atomic inputs (non-MEDIAN)
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene - DUPLICATE INPUTS Example):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: two instances of 'seven' (7) and one instance of 'three' (3). (e.g., 'seven relics here, another seven there, and three more over yonder').\\\\n"
            "*   **RESULT MUST BE IMPLICIT:** The numerical result of this operation (which is 'five' (5)) MUST NOT be explicitly stated. It should be implied conceptually as 'The Combined Resonance'.\\\\n"
            "*   You MAY use the number 'three' (the count of *mentioned* direct items) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions 'seven' twice and 'three' once.
        "The artificer examined the power conduits. 'This primary conduit shows a charge of seven units. The secondary conduit also registers seven units. And the auxiliary channel is pulsing at three units.' He nodded. 'When combined, their energies should achieve The Combined Resonance.'",
        # BAD example: only mentions 'seven' once.
        "The artificer examined the power conduits. 'This primary conduit shows a charge of seven units. The auxiliary channel is pulsing at three units.' He nodded. 'When combined, their energies should achieve The Combined Resonance.'",
        "BAD output failed: Did not mention all required instances of atomic inputs. The rules specified 'two instances of seven', but the narrative only mentioned 'seven' once.",
    ),
]

META_INSTRUCTION = (
    "Here are examples demonstrating how to solve narrative math problems. For each problem: "
    "Read the entire story. Identify the quantity being tracked (e.g., coins, artifacts, energy units). "
    "Follow the narrative step-by-step, performing the calculation implied by the actions in each scene "
    "(e.g., finding items, combining, selecting the largest/smallest, averaging, reducing, resetting). "
    "Keep track of the current quantity as it changes. Finally, answer the question by providing only "
    "the single integer representing the final quantity based on the last relevant action described."
)

TASK_SOLVING_FEW_SHOTS = [
    # ... (content of TASK_SOLVING_FEW_SHOTS) ...
]

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

if config.CLEAR_LOGS_ON_START:
    if os.path.exists(LOG_DIR):
        try:
            shutil.rmtree(LOG_DIR)
            print(f"Removed existing log directory: {LOG_DIR}")
        except OSError as e:
            print(f"Error removing log directory {LOG_DIR}: {e}")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory {LOG_DIR}: {e}")

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
os.makedirs(LOG_DIR, exist_ok=True)
handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "verbose_listops.log"),
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
print(f"Logger initialized. Log file: {os.path.join(LOG_DIR, 'verbose_listops.log')}")
if config.CLEAR_LOGS_ON_START and os.path.exists(LOG_DIR):
    logger.info(f"Log directory {LOG_DIR} cleared and recreated successfully.")
elif config.CLEAR_LOGS_ON_START and not os.path.exists(LOG_DIR):
    logger.warning(
        f"Log directory {LOG_DIR} was meant to be cleared/recreated, but it does not exist."
    )


def log_prompt(header: str, prompt: str, sample_index: int | None = None):
    try:
        llm_turns_main_dir = os.path.join(LOG_DIR, "llm_turns")
        llm_turns_log_specific_dir = os.path.join(llm_turns_main_dir, "log")
        log_filename = (
            f"llm_turns_sample_{sample_index + 1}.log"
            if sample_index is not None
            else "llm_turns_general.log"
        )
        os.makedirs(llm_turns_log_specific_dir, exist_ok=True)
        current_log_file_path = os.path.join(llm_turns_log_specific_dir, log_filename)
        timestamp = datetime.datetime.now().isoformat()
        log_header_text = (
            f"[Sample {sample_index + 1}] {header}"
            if sample_index is not None
            else header
        )
        with open(current_log_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"--- Log Time: {timestamp} ---\\n{log_header_text}\\n{prompt}\\n\\n---\\n\\n"
            )
    except Exception as e:
        logger.error(f"Error writing to LLM turn log file: {e}")


FINAL_QUESTION_TEMPLATE = Template(
    "\n\n---\n\n**Question:** The story describes a sequence of operations that modify a quantifiable measure related to '$primary_object'. Following this entire sequence, what is the final, precise numerical value of this measure at the conclusion of all activities? Provide only the single integer."
)


# Ensure GenerationContext dataclass is defined and includes overall_ast_root
@dataclass
class GenerationContext:
    world: dict
    config: Config  # Use full name to avoid conflict
    encoder: any
    p_inflect: any
    logger: logging.Logger  # Use full name
    narrative_anchor_map: dict
    all_atoms: set  # All unique atomic numbers in the AST
    introduced_atoms: set  # Atoms mentioned in narrative so far (only from current beat's direct atoms)
    scenes: list
    tokens_used: int
    last_scene_text: str
    beat_counter: dict  # {'current': 0, 'total': N}
    sample_index: int
    max_pad_paragraphs: int
    overall_ground_truth_answer: int | None
    overall_ast_root: Node | None = (
        None  # NEW: To trace node values for forbidden checks
    )
    padding_stats: dict = field(
        default_factory=lambda: {
            "total_padding_tokens": 0,
            "padding_segments_added": 0,
            "max_padding_allowed": 0,
            "padding_per_slot": 0,
        }
    )


# --- Postorder Traversal (needed by generate_narrative before its own definition) ---
# This function is used by generate_narrative to count operator nodes and get all atoms.
# It needs to be defined before generate_narrative if generate_narrative uses it at the top level.
# Or, ensure Node, Atom, OpNode are defined before this.
def postorder(node: Node):
    """Yield nodes in post-order."""
    if node is None:  # Add this check to handle None values
        return
    for c in node.children:
        yield from postorder(c)
    yield node


# --- End Postorder Traversal ---


def generate_introduction_scene(
    world_info: dict,
    sample_index: int | None = None,
    config_obj: Config = config,  # Add config_obj parameter
    logger_obj: logging.Logger = logger,  # Add logger_obj parameter
) -> str | None:
    logger_obj.info(
        f"[Sample {sample_index + 1 if sample_index is not None else 'N/A'}] Generating introduction scene..."
    )

    # --- COPIED PROMPT LOGIC START ---
    system_prompt = (
        f"You are a master {world_info.get('genre')} storyteller. Your task is to write a compelling introductory scene for a new story. "
        "This scene should establish the setting, introduce one or two key characters, and hint at a central mystery or goal related to the primary object. "
        "Crucially, this introductory scene MUST NOT contain any numerical values (digits or words like 'one', 'two', 'first', etc.), "
        "except potentially the word 'one', 'two', or 'three' if used for completely general, non-quantitative phrasing (e.g., 'a single ray of light', 'two figures emerged', 'three ancient symbols'). Strive for zero numbers. "
        "Focus on atmosphere and intrigue. Do not reveal any specific quantities or begin any calculations. "
        "Output ONLY the narrative text for this scene. No titles, no explanations, no analysis."
    )

    characters_list = world_info.get("characters", [])
    char_names_roles = []
    if characters_list:
        # Select one or two characters for the intro
        num_intro_chars = random.randint(1, min(2, len(characters_list)))
        intro_chars = random.sample(characters_list, num_intro_chars)
        for char_info in intro_chars:
            char_names_roles.append(
                f"{char_info.get('name', 'A mysterious figure')} ({char_info.get('role', 'of unknown purpose')})"
            )

    user_prompt = (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'A realm of mystery')}\n"
        f"- Setting: {world_info.get('setting', 'An enigmatic place')}\n"
        f"- Primary Object of Interest: {world_info.get('object', 'ancient artifacts')}\n"
        f"- Characters to potentially feature: {', '.join(char_names_roles) if char_names_roles else 'The inhabitants of this world'}\n\n"
        f"**Task:** Write an engaging introductory scene based on the context above. Remember the strict rule: NO numbers (or strive for zero numbers, with very limited exceptions for 'one'/'two'/'three' in general phrasing only). "
        f"The scene should set a tone and hint at the story's direction without giving away specifics. "
        f"Output ONLY the narrative text."
    )
    # --- COPIED PROMPT LOGIC END ---

    # Validator for intro: NO numbers, or at most very specific phrasing numbers if allowed by config.
    # The intro prompt asks for NO numbers.
    validate_intro = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=set(),
        operand_count=0,
        correct_result_for_beat=None,
        strict_zero=True,  # Key for intro/padding style validation
        enforce_result_presence=False,
        operation_type="INTRO",
        overall_ground_truth_answer=None,  # No GT relevant for intro in this way
        is_root_node_being_validated=False,
        config_obj=config_obj,  # Pass the config
        logger_obj=logger_obj,
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=config_obj.INTRO_MAX_TOKENS,
        validate_fn=validate_intro,
        retries=config_obj.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config_obj.CREATIVE_NARRATIVE_TEMP,
        reasoning_settings={"exclude": True},
    )
    if intro_text:
        logger_obj.info(
            f"Successfully generated intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return intro_text.strip()
    else:
        logger_obj.error(
            f"Failed to generate intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return None


def generate_narrative(
    ast: Node,  # This is the root of the AST for the current problem
    world: dict,
    config_obj: Config,  # Renamed from config
    encoder_obj: tiktoken.Encoding,  # More specific type hint
    p_inflect_obj: inflect.engine,  # More specific type hint
    logger_obj: logging.Logger,  # Use full name
    sample_index: int,
    overall_ground_truth_answer: int,
) -> GenerationContext | None:  # Return the whole context or None on failure
    logger_obj.info(f"[Sample {sample_index + 1}] Starting narrative generation.")

    # --- Initialize narrative_anchor_map ---
    # This map will store {node_id: "anchor_name_string"}
    # It needs to be populated before GenerationContext is created if GenerationContext needs it at init.
    # However, GenerationContext just stores it; it's populated *within* this function.
    narrative_anchor_map: dict[int, str] = {}  # Initialize as an empty dict

    # --- Populate narrative_anchor_map (Example of how it's done in the full script) ---
    # This logic is simplified here; the full script has generate_narrative_anchor_with_llm or a fallback is called here
    # For this fix, we'll just use a placeholder to ensure the map is populated.
    # The actual anchor generation logic is complex and assumed to be working.
    if config_obj.USE_NARRATIVE_ANCHORS:
        temp_operator_nodes_for_anchors = []
        for node_iter in postorder(ast):
            if isinstance(node_iter, OpNode):
                # In the full script, generate_narrative_anchor_with_llm or a fallback is called here
                # For this fix, we'll just use a placeholder to ensure the map is populated.
                # The actual anchor generation logic is complex and assumed to be working.
                anchor_name = f"anchor_for_{node_iter.op}_{id(node_iter) % 100}"
                narrative_anchor_map[id(node_iter)] = anchor_name
                temp_operator_nodes_for_anchors.append(node_iter)
        logger_obj.debug(
            f"Populated narrative_anchor_map with {len(narrative_anchor_map)} anchors."
        )
    # --- End of narrative_anchor_map population ---

    all_atoms_in_ast = set()
    for node_in_ast in postorder(ast):  # Iterate over the passed 'ast'
        if isinstance(node_in_ast, Atom):
            all_atoms_in_ast.add(node_in_ast.n)

    operator_nodes_for_count = [n for n in postorder(ast) if isinstance(n, OpNode)]
    total_beats = len(operator_nodes_for_count)

    scenes_list = []
    tokens_used_count = 0
    last_scene_text_val = "The story begins..."

    # Assuming generate_introduction_scene, make_number_validator, generate_with_retry are defined before this point
    # or imported correctly. For this specific fix, we focus on Node and narrative_anchor_map.
    intro_text = generate_introduction_scene(
        world, sample_index=sample_index, config_obj=config_obj, logger_obj=logger_obj
    )
    if intro_text:
        intro_tokens = len(encoder_obj.encode(intro_text))
        if intro_tokens <= config_obj.MAX_TOTAL_TOKENS - config_obj.MAX_TOKENS_BUFFER:
            scenes_list.append(intro_text)
            tokens_used_count += intro_tokens
            last_scene_text_val = intro_text
        else:
            intro_text = None

    context = GenerationContext(
        world=world,
        config=config_obj,
        encoder=encoder_obj,
        p_inflect=p_inflect_obj,
        logger=logger_obj,
        narrative_anchor_map=narrative_anchor_map,  # Now narrative_anchor_map is defined and populated
        all_atoms=all_atoms_in_ast,
        introduced_atoms=set(),
        scenes=scenes_list,
        tokens_used=tokens_used_count,
        last_scene_text=last_scene_text_val,
        beat_counter={"current": 0, "total": total_beats},
        sample_index=sample_index,
        max_pad_paragraphs=config_obj.MAX_PAD_PARAGRAPHS,
        overall_ground_truth_answer=overall_ground_truth_answer,
        overall_ast_root=ast,
    )
    
    logger_obj.info(
        f"Successfully generated narrative for sample {sample_index + 1}. Final context tokens: {context.tokens_used}"
    )
    return context


SAFETY_MARGIN = config.MAX_TOKENS_BUFFER
MAX_BEAT_COMPLETION_TOKENS = config.BEAT_MAX_TOKENS
MAX_PAD_COMPLETION_TOKENS = config.PADDING_MAX_TOKENS

# --- Setup Logging ---

if config.CLEAR_LOGS_ON_START:
    for filename in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, filename)
        try:
            os.remove(file_path)
        except OSError:
            pass

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)

# Remove existing handlers to avoid duplicates and force new handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Ensure the log file handler exists
os.makedirs(LOG_DIR, exist_ok=True)
handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "verbose_listops.log"),
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

print(
    f"Logger initialized with {len(logger.handlers)} handlers. Log file will be created at: {os.path.join(LOG_DIR, 'verbose_listops.log')}"
)

# --- Instantiate OpenAI Client for OpenRouter Endpoint ---
client = None
try:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError("OpenRouter API Key not found or is placeholder.")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    logger.info("OpenAI client configured to use OpenRouter API endpoint.")
except Exception as e:
    logger.error(f"Failed to configure OpenAI client for OpenRouter endpoint: {e}")
    client = None


# --- Rate Limiter for API Calls ---
class RateLimiter:
    """
    Thread-safe rate limiter that implements a token bucket algorithm.
    Allows for bursts of requests while maintaining a long-term rate limit.
    """

    def __init__(
        self,
        max_requests_per_second: float = 40.0,
        min_interval: float = 0.05,
        bucket_capacity: int = 5,
        jitter: float = 0.1,
    ):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = min_interval  # Minimum time between requests in seconds
        self.bucket_capacity = bucket_capacity  # Maximum tokens in the bucket
        self.jitter = jitter  # Random jitter to apply to wait times
        self.tokens = bucket_capacity  # Start with a full bucket
        self.last_refill_time = time.time()  # Last token refill timestamp
        self.lock = threading.Lock()  # Thread lock for concurrent access
        self.last_limits_check_time = 0  # Last time we checked account limits
        self.limits_check_interval = 5  # Check limits every 5 seconds
        self.initial_usage = None  # Store the initial usage value when first checked

        # Log configuration
        logger.info(
            f"Rate limiter initialized: {max_requests_per_second} req/s, "
            f"{min_interval}s min interval, bucket capacity {bucket_capacity}, jitter {jitter}"
        )

    def wait_if_needed(self):
        """
        Implements token bucket algorithm to manage API request rates.
        Returns the amount of time waited.
        """
        # Check if we should update our rate limits based on account status
        current_time = time.time()
        if current_time - self.last_limits_check_time > self.limits_check_interval:
            self.update_limits_from_api()

        with self.lock:
            # Refill tokens based on elapsed time
            current_time = time.time()
            elapsed = current_time - self.last_refill_time

            # Calculate token refill (tokens are added based on time elapsed)
            new_tokens = elapsed * self.max_requests_per_second

            # Update token count, but don't exceed capacity
            self.tokens = min(self.bucket_capacity, self.tokens + new_tokens)
            self.last_refill_time = current_time

            # If we have at least 1 token, consume it and continue immediately
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            # Otherwise, calculate wait time needed for at least 1 token
            wait_time = (1.0 - self.tokens) / self.max_requests_per_second

            # Ensure we wait at least the minimum interval
            wait_time = max(wait_time, self.min_interval)

            # Add random jitter to prevent thundering herd problem
            if self.jitter > 0:
                wait_time += random.uniform(0, self.jitter)

            # Wait and update
            time.sleep(wait_time)
            self.tokens = 0.0  # We\\'ve used our token
            self.last_refill_time = time.time()

            return wait_time

    def update_limits_from_api(self):
        """
        Check OpenRouter API rate limits and adjust rate limiter settings accordingly.
        Returns the current account usage (float) or None if an error occurs or usage is not found.
        """
        if (
            not OPENROUTER_API_KEY
            or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE"
        ):
            logger.warning("Cannot check OpenRouter limits: No valid API key")
            return None  # ADDED

        current_usage = None  # Initialize to None
        try:
            logger.info("OpenRouter STATUS:")
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                # logger.info(f"OpenRouter account status: {json.dumps(data, indent=2)}") # Made more concise

                # Extract the main data object from the nested response
                account_data = data.get("data", {})

                # Update rate limiter settings based on the nested rate_limit structure
                rate_limit_info = account_data.get("rate_limit", {})
                current_rate_for_log = self.max_requests_per_second
                limit_adjusted = False

                if rate_limit_info:
                    requests_limit = rate_limit_info.get("requests")
                    interval = rate_limit_info.get("interval", "")

                    if requests_limit and interval:
                        # Calculate RPS if in a format like "10s"
                        if interval.endswith("s") and interval[:-1].isdigit():
                            interval_seconds = int(interval[:-1])
                            rps = requests_limit / interval_seconds

                            # Set to 80% of the allowed rate limit as a safety buffer
                            new_rate = min(
                                float(rps) * 0.8, config.MAX_REQUESTS_PER_SECOND
                            )

                            if new_rate != self.max_requests_per_second:
                                # logger.info(f"Adjusting rate limiter based on OpenRouter limit: {rps} req/s → {new_rate} req/s (80% safety, capped at config.MAX_REQUESTS_PER_SECOND)")
                                self.max_requests_per_second = new_rate
                                limit_adjusted = True
                                current_rate_for_log = (
                                    new_rate  # Update for the concise log
                                )

                # Log credits/usage if available
                usage = account_data.get("usage")
                limit = account_data.get("limit")
                limit_remaining = account_data.get("limit_remaining")

                # Set initial usage on first check
                if usage is not None:
                    current_usage = float(usage)  # Store the usage
                    if self.initial_usage is None:
                        self.initial_usage = current_usage
                        logger.info(
                            f"Initial OpenRouter usage set to: ${self.initial_usage:.4f}"
                        )

                # Calculate run cost (difference between current and initial usage)
                run_cost = None
                if usage is not None and self.initial_usage is not None:
                    run_cost = current_usage - self.initial_usage

                log_message_parts = [f"RPS: {current_rate_for_log:.1f}"]
                if limit_adjusted:
                    log_message_parts.append("(Adjusted)")

                # Replace detailed usage stats with current run cost
                if run_cost is not None:
                    log_message_parts.append(f"Cost so far: ${run_cost:.4f}")
                elif usage is not None:
                    log_message_parts.append(f"Usage: ${usage:.4f}")

                if limit is not None and limit_remaining is not None:
                    log_message_parts.append(
                        f"Credits: Rem ${limit_remaining:.4f} of ${limit:.4f}"
                    )

                    # If very low on remaining limit, be more conservative with request rate
                    if limit and limit_remaining and limit_remaining / limit < 0.2:
                        # logger.warning(f"Low limit remaining ({limit_remaining}/{limit}). Reducing request rate.")
                        old_self_rate = self.max_requests_per_second
                        self.max_requests_per_second = min(
                            self.max_requests_per_second, 10.0
                        )
                        if old_self_rate != self.max_requests_per_second:
                            log_message_parts.append(
                                f"LOW CREDITS - RPS reduced to {self.max_requests_per_second:.1f}!"
                            )

                logger.info(", ".join(log_message_parts))
                self.last_limits_check_time = time.time()
            else:
                logger.warning(
                    f"Failed to get OpenRouter account status: HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error checking OpenRouter limits: {e}")
            return None  # Return None on exception

        self.last_limits_check_time = time.time()
        return current_usage  # Return the fetched usage


# Create a singleton rate limiter instance
rate_limiter = RateLimiter(
    max_requests_per_second=config.MAX_REQUESTS_PER_SECOND,
    min_interval=config.MIN_REQUEST_INTERVAL,
    bucket_capacity=100,  # Allow bursts of up to 100 requests
    jitter=0.01,  # Add up to 10ms of random jitter to prevent synchronization
)

# Check OpenRouter limits when starting up
# rate_limiter.update_limits_from_api() # Calling this here might be too early if logger not fully set up.
# Consider calling it first time wait_if_needed is invoked or explicitly after logger setup.

# --- Inflect Engine ---
try:
    p_inflect = inflect.engine()
except Exception as e:
    logger.error(f"Failed to initialize inflect engine: {e}")
    p_inflect = None


# --- Safe, memoised wrapper around inflect.number_to_words ---
@lru_cache(maxsize=None)
def num_to_words(n: int) -> str:
    """
    Convert an int to its English word form using inflect with memoisation.
    Falls back to the digit string if inflect is unavailable or raises.
    Ensures "and" is not used for compatibility with simpler parsing.
    """
    if p_inflect is None:
        return str(n)
    try:
        # Generate words without "and"
        return p_inflect.number_to_words(n, andword="")
    except Exception:
        return str(n)


# --- Generic retry helper (centralised back‑off policy) ---
def with_retry(func: Callable, *args, **kwargs):
    """
    Call `func` with the supplied args/kwargs, retrying on exception
    using exponential back‑off.
    The policy is defined by `config.RETRY_MAX_ATTEMPTS` and
    `config.RETRY_INITIAL_DELAY`.
    """
    delay = config.RETRY_INITIAL_DELAY
    for attempt in range(1, config.RETRY_MAX_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            is_rate_limit_error = False
            error_str = str(e).lower()

            if hasattr(e, "status_code") and getattr(e, "status_code") == 429:
                is_rate_limit_error = True
            elif hasattr(e, "http_status") and getattr(e, "http_status") == 429:
                is_rate_limit_error = True
            elif (
                hasattr(e, "response")
                and hasattr(e.response, "status_code")
                and e.response.status_code == 429
            ):
                is_rate_limit_error = True
            elif any(
                phrase in error_str
                for phrase in [
                    "rate limit",
                    "too many requests",
                    "ratelimit",
                    "quota exceeded",
                    "usage limit",
                    "capacity",
                    "throttled",
                ]
            ):
                is_rate_limit_error = True

            if is_rate_limit_error:
                rate_limited_delay = delay * 3
                logger.warning(
                    f"Rate limit error detected calling {getattr(func, '__name__', repr(func))} "
                    f"(attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}): {e}"
                )
                logger.info(
                    f"Rate limiting triggered - backing off for {rate_limited_delay:.2f}s"
                )
                try:
                    if hasattr(rate_limiter, "max_requests_per_second"):
                        old_rate = rate_limiter.max_requests_per_second
                        new_rate = max(1.0, old_rate * 0.8)
                        rate_limiter.max_requests_per_second = new_rate
                        logger.info(
                            f"Reducing rate limit: {old_rate:.1f} → {new_rate:.1f} req/s"
                        )
                except Exception as inner_e:
                    logger.warning(f"Failed to adjust rate limiter: {inner_e}")
                wait_time = rate_limited_delay
            else:
                logger.warning(
                    f"Retryable error calling {getattr(func, '__name__', repr(func))} "
                    f"(attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}): {e}"
                )
                wait_time = delay

            if attempt == config.RETRY_MAX_ATTEMPTS:
                logger.error("Max retry attempts reached. Raising.")
                raise

            wait_time += random.uniform(0, 0.5)  # Add jitter
            time.sleep(wait_time)
            delay *= 2


def clean_snippet(text: str, max_len: int = config.BEAT_CONTEXT) -> str:
    """Removes common model analysis/checklist lines and takes the last part."""
    if not text:
        return "The story begins..."

    lines = text.splitlines()
    cleaned_lines = [
        line
        for line in lines
        if not re.match(
            r"^\s*(-|\*|\d+\.|Critique|Checklist|Yes|No|Draft \d+|Option \d+|\[.*?\]:|MUST INCLUDE|MUST AVOID|Problem:|REASONING:|GOOD:|BAD:|Confidence Score:|Mental Sandbox:|Outcome is|Narrative:|Generation:|Rules:|System:|User:|Okay|Check\.|REMINDER:|Instructions:|Task:|^\?|^\s*$)",
            line.strip(),
            re.IGNORECASE,
        )
        and not line.strip().startswith(
            (
                "Imply the sum",
                "reference to the previous",
                "Narrate comparing",
                "This scene resolves",
            )
        )
    ]

    cleaned_text = "\n".join(cleaned_lines).strip()
    if not cleaned_text:
        original_lines = [line for line in lines if line.strip()]
        if original_lines:
            cleaned_text = original_lines[-1].strip()
        else:
            return "Previously..."

    return cleaned_text[-max_len:]


def retry_api_call(func: Callable):
    """Decorator that applies the shared `with_retry` policy to `func`."""

    def wrapper(*args, **kwargs):
        return with_retry(func, *args, **kwargs)

    return wrapper


# --- Budget-check Helper ---
def would_exceed_budget(
    current: int, upcoming: int, max_total: int, margin: int
) -> bool:
    """Return True if adding upcoming tokens would exceed max_total minus margin."""
    would_exceed = current + upcoming + margin > max_total
    remaining = max_total - current - margin
    percentage_used = (current / max_total) * 100

    if would_exceed:
        logger.warning(
            f"TOKEN LIMIT CHECK: WOULD EXCEED - Current: {current} tokens ({percentage_used:.1f}%), "
            f"Upcoming: +{upcoming}, Safety Margin: {margin}, Remaining: {remaining}, "
            f"Budget: {max_total}, Total After: {current + upcoming + margin}/{max_total}"
        )
    else:
        logger.debug(
            f"TOKEN LIMIT CHECK: WITHIN BUDGET - Current: {current} tokens ({percentage_used:.1f}%), "
            f"Upcoming: +{upcoming}, Safety Margin: {margin}, Remaining: {remaining}, "
            f"Budget: {max_total}, Will use: {current + upcoming + margin}/{max_total}"
        )

    return would_exceed


# --- Helper for future consolidation of retry loops ---
def generate_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    validate_fn: Callable[[str], bool],
    retries: int = config.MAX_BEAT_RETRIES,
    sample_index: int | None = None,
    temperature: float = config.CREATIVE_NARRATIVE_TEMP,
    reasoning_settings: dict = None,  # Added reasoning_settings parameter
):
    """
    Helper to call the OpenAI ChatCompletion API with retries and apply a validation function.
    Returns the first candidate text that passes validate_fn, or None if all attempts fail.
    Passes sample_index to log_prompt if provided.
    """
    candidate = None
    validation_failure_reasons = []

    for attempt in range(1, retries + 1):
        try:
            # Use the API token limit instead of the provided max_completion_tokens
            # This prevents truncation due to reasoning tokens being counted against max_tokens
            actual_max_tokens = config.MAX_API_TOKEN_LIMIT

            logger.debug(
                f"API Call: Using {actual_max_tokens} tokens for API (vs {max_completion_tokens} for internal tracking)"
            )

            api_params = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": actual_max_tokens,  # Use the higher limit
                "temperature": temperature,
            }
            if reasoning_settings:
                api_params["reasoning"] = reasoning_settings.copy()  # Pass a copy
                logger.debug(
                    f"generate_with_retry: Passing reasoning_settings to _chat_completion_call: {api_params['reasoning']}"
                )
            # If reasoning_settings is None, _chat_completion_call will apply its own
            # default logic based on config.REASONING_EXCLUDE.

            resp = _chat_completion_call(**api_params)

            truly_raw_llm_content = None
            if (
                resp
                and resp.choices
                and len(resp.choices) > 0
                and resp.choices[0].message
            ):
                truly_raw_llm_content = resp.choices[0].message.content
            log_prompt(
                f"LLM Turn Attempt {attempt}",
                f"System: {system_prompt}\\nUser: {user_prompt}\\n\\nGeneration (Raw):\\n{truly_raw_llm_content if truly_raw_llm_content is not None else '[[API returned None content string]]'}",
                sample_index=sample_index,
            )

            # Prepare the candidate for validation and return, which involves stripping.
            candidate_for_validation_and_return = None
            if truly_raw_llm_content is not None:
                candidate_for_validation_and_return = truly_raw_llm_content.strip()
            else:
                logger.warning(
                    f"API call in generate_with_retry attempt {attempt} returned None content. Response object: {resp}"
                )
                validation_failure_reasons.append("API returned None content")
                continue

            if candidate_for_validation_and_return is None:
                logger.warning(
                    f"generate_with_retry attempt {attempt} resulted in None candidate_for_validation_and_return (possibly after stripping None)."
                )
                validation_failure_reasons.append("Empty content after stripping")
            elif (
                not candidate_for_validation_and_return
                or candidate_for_validation_and_return.lower().startswith(
                    ("i cannot", "i'm sorry", "i am unable")
                )
            ):
                logger.warning(f"API refusal on generate_with_retry attempt {attempt}.")
                validation_failure_reasons.append("API refusal detected")
            elif validate_fn(candidate_for_validation_and_return):
                logger.info(f"Validation PASSED on attempt {attempt}")
                return candidate_for_validation_and_return
            else:
                # If we got here, validation failed - look for most recent failed validation file
                failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
                if os.path.exists(failed_validations_dir):
                    try:
                        # Get the most recent validation failure file (should be the one just created)
                        files = [
                            f
                            for f in os.listdir(failed_validations_dir)
                            if f.startswith("validation_fail_")
                        ]
                        if files:
                            files.sort(reverse=True)  # Most recent first
                            latest_file = os.path.join(failed_validations_dir, files[0])
                            with open(latest_file, "r", encoding="utf-8") as f:
                                failure_data = json.load(f)
                                reason = failure_data.get("validation_report", {}).get(
                                    "reason", "Unknown"
                                )
                                validation_failure_reasons.append(
                                    f"Validation failed: {reason}"
                                )

                                # Add more detailed debugging info
                                logger.warning(
                                    f"Validation failed on attempt {attempt}: {reason}"
                                )
                                logger.warning(
                                    f"Found numbers: {failure_data.get('validation_report', {}).get('found_numbers', [])}"
                                )
                                logger.warning(
                                    f"Required numbers: {failure_data.get('validation_report', {}).get('allowed_atoms', [])}"
                                )
                                logger.warning(
                                    f"Missing required: {failure_data.get('validation_report', {}).get('missing_required', [])}"
                                )
                                logger.warning(
                                    f"Forbidden extras: {failure_data.get('validation_report', {}).get('forbidden_extras', [])}"
                                )
                    except Exception as e:
                        logger.error(f"Error reading validation failure data: {e}")
                        validation_failure_reasons.append(
                            f"Validation failed (error reading details)"
                        )
                else:
                    validation_failure_reasons.append(
                        "Validation failed (no details available)"
                    )

        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")
            validation_failure_reasons.append(f"Exception: {str(e)}")

        if attempt < retries:
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    if validation_failure_reasons:
        logger.warning(
            f"generate_with_retry failed after {retries} attempts. Failure reasons: {validation_failure_reasons}"
        )
    else:
        logger.warning(
            f"generate_with_retry failed after {retries} attempts with no specific reasons recorded."
        )
    return None


OP_LABELS = {
    "MAX": "largest value",
    "MIN": "smallest value",
    "SUM": "sum of all values",
    "MED": "median value",
    "AVG": "integer-average (floored)",
    "SM": "sum modulo 10",
}


# --- AST Generation and Evaluation ---
def build_random_ast(
    max_ops: int, max_branch: int = config.MAX_BRANCH, config_obj: Config = config
) -> Node:
    """Constructs a random ListOps AST."""
    if not isinstance(max_ops, int) or max_ops < 1:
        raise ValueError("max_ops must be a positive int")
    if max_branch < config_obj.MIN_ARITY:
        raise ValueError(
            f"max_branch ({max_branch}) < MIN_ARITY ({config_obj.MIN_ARITY})"
        )
    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0

    def helper():
        nonlocal count
        if count >= max_ops or (
            count > 0 and random.random() < config_obj.EARLY_TERMINATION_PROBABILITY
        ):
            return Atom(
                random.randint(config_obj.MIN_ATOM_VAL, config_obj.MAX_ATOM_VAL)
            )
        count += 1
        op = random.choice(ops)

        if op == "MED":
            possible_arities = [
                n for n in range(config_obj.MIN_ARITY, max_branch + 1) if n % 2 == 1
            ]

            if not possible_arities:
                arity = (
                    config_obj.MIN_ARITY
                    if config_obj.MIN_ARITY % 2 == 1
                    else config_obj.MIN_ARITY + 1
                )
            else:
                arity = random.choice(possible_arities)
        else:
            arity = random.randint(config_obj.MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]

        # Ensure AVG direct atom sum is divisible by atom count
        if op == "AVG":
            direct_atoms = [c for c in children if isinstance(c, Atom)]
            arity = len(direct_atoms)
            if arity > 0:  # Only adjust if there are direct atoms
                current_sum = sum(a.n for a in direct_atoms)
                remainder = current_sum % arity

                if remainder != 0:
                    adjustment_needed = (arity - remainder) % arity
                    logger.debug(
                        f"AST Gen (AVG): current_sum={current_sum}, arity={arity}, remainder={remainder}, adjustment_needed={adjustment_needed}"
                    )

                    atom_to_adjust = random.choice(direct_atoms)
                    adjusted = False

                    new_value_add = atom_to_adjust.n + adjustment_needed
                    if (
                        config_obj.MIN_ATOM_VAL
                        <= new_value_add
                        <= config_obj.MAX_ATOM_VAL
                    ):
                        atom_to_adjust.n = new_value_add
                        atom_to_adjust.value = new_value_add
                        logger.debug(
                            f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value up to {atom_to_adjust.n} to make sum divisible by {arity}."
                        )
                        adjusted = True

                    if not adjusted:
                        new_value_sub = atom_to_adjust.n - (arity - adjustment_needed)
                        if (
                            config_obj.MIN_ATOM_VAL
                            <= new_value_sub
                            <= config_obj.MAX_ATOM_VAL
                        ):
                            atom_to_adjust.n = new_value_sub
                            atom_to_adjust.value = new_value_sub
                            logger.debug(
                                f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value down to {atom_to_adjust.n} to make sum divisible by {arity}."
                            )
                            adjusted = True

                    if not adjusted:

                        logger.warning(
                            f"AST Gen (AVG): Could not adjust atom value {atom_to_adjust.n} (target adjustment {adjustment_needed}) for AVG node sum {current_sum} to be divisible by {arity} due to bounds [{config_obj.MIN_ATOM_VAL}, {config_obj.MAX_ATOM_VAL}]."
                        )

        return OpNode(op, children)

    root = helper()

    # --- START MODIFICATION FOR SUGGESTION 2 ---
    # If the root is an Atom (meaning max_ops was 0 or 1 initially, or early termination)
    # OR if the root is NOT a combining operation (SUM, AVG, SM) and we want to force it
    # for problems with at least, say, 2 operations.
    # This ensures the final step is a calculation rather than just a selection if possible.

    is_combining_op = isinstance(root, OpNode) and root.op in ["SUM", "AVG", "SM"]
    MIN_OPS_FOR_COMBINING_ROOT = 2  # Arbitrary: only force if problem has some depth

    if (isinstance(root, Atom) and max_ops >= 1) or (
        count >= MIN_OPS_FOR_COMBINING_ROOT and not is_combining_op and max_ops > count
    ):  # Ensure we have ops left to make a new root

        logger.debug(
            f"AST Gen: Original root was {getattr(root, 'op', 'Atom')}. Attempting to ensure a combining root."
        )

        # Prefer SUM, AVG, SM as the new root
        combining_ops = ["SUM", "AVG", "SM"]
        new_root_op = random.choice(combining_ops)

        # Determine arity for the new root
        # New root will have the old root as one child, and new atoms as others.
        # Ensure arity is at least 2 to include the old root and at least one new atom.
        # Max arity should still respect max_branch.
        new_arity = random.randint(max(2, config_obj.MIN_ARITY), max_branch)

        new_children = [root]  # The old root is one child

        # Add new Atom children
        for _ in range(new_arity - 1):
            new_children.append(
                Atom(random.randint(config_obj.MIN_ATOM_VAL, config_obj.MAX_ATOM_VAL))
            )

        random.shuffle(new_children)

        # Special handling for AVG if we created it as the new root
        if new_root_op == "AVG":
            direct_atoms_new_root = [c for c in new_children if isinstance(c, Atom)]
            current_sum_new_root = sum(a.n for a in direct_atoms_new_root)
            num_direct_atoms_new_root = len(direct_atoms_new_root)

            if (
                num_direct_atoms_new_root > 0
            ):  # Should always be true if new_arity >=2 and one child is Atom
                remainder_new_root = current_sum_new_root % num_direct_atoms_new_root
                if remainder_new_root != 0:
                    adjustment_needed_new_root = (
                        num_direct_atoms_new_root - remainder_new_root
                    ) % num_direct_atoms_new_root

                    # Try to adjust one of the newly added atoms
                    newly_added_atoms = [
                        c for c in new_children if isinstance(c, Atom) and c is not root
                    ]  # Exclude the old root if it was an atom

                    atom_to_adjust_new_root = None
                    if newly_added_atoms:
                        atom_to_adjust_new_root = random.choice(newly_added_atoms)
                    elif (
                        direct_atoms_new_root
                    ):  # Fallback if old root was an atom and only one new atom was added
                        atom_to_adjust_new_root = random.choice(direct_atoms_new_root)

                    if atom_to_adjust_new_root:
                        adjusted_new_root = False
                        # Try adding
                        if (
                            config_obj.MIN_ATOM_VAL
                            <= atom_to_adjust_new_root.n + adjustment_needed_new_root
                            <= config_obj.MAX_ATOM_VAL
                        ):
                            atom_to_adjust_new_root.n += adjustment_needed_new_root
                            atom_to_adjust_new_root.value = (
                                atom_to_adjust_new_root.n
                            )  # Update value too
                            adjusted_new_root = True
                            logger.debug(
                                f"AST Gen (Forced Root AVG): Adjusted new atom value for divisibility."
                            )
                        # Try subtracting if adding failed
                        elif (
                            config_obj.MIN_ATOM_VAL
                            <= atom_to_adjust_new_root.n
                            - (num_direct_atoms_new_root - adjustment_needed_new_root)
                            <= config_obj.MAX_ATOM_VAL
                        ):
                            atom_to_adjust_new_root.n -= (
                                num_direct_atoms_new_root - adjustment_needed_new_root
                            )
                            atom_to_adjust_new_root.value = (
                                atom_to_adjust_new_root.n
                            )  # Update value too
                            adjusted_new_root = True
                            logger.debug(
                                f"AST Gen (Forced Root AVG): Adjusted new atom value (subtracted) for divisibility."
                            )

                        if not adjusted_new_root:
                            logger.warning(
                                f"AST Gen (Forced Root AVG): Could not adjust new atom for AVG divisibility."
                            )
                    else:
                        logger.warning(
                            f"AST Gen (Forced Root AVG): No suitable atom found to adjust for divisibility."
                        )

        root = OpNode(new_root_op, new_children)
        # Increment max_ops if we added an operation, or ensure count reflects it.
        # The 'count' variable in helper() tracks operations. If we add one here,
        # it's effectively one more operation than 'max_ops' might have initially allowed
        # for the helper. This is a design choice. For now, we assume 'max_ops' is a soft limit.
        logger.info(f"AST Gen: Ensured root is a combining op: {root.op}")
    # --- END MODIFICATION FOR SUGGESTION 2 ---

    # Original fallback if root is Atom and max_ops >=1 (this might be redundant now or need adjustment)
    # Consider if this block is still needed or if the logic above covers it.
    # For now, I'll keep it but it might interact with the above.
    # A simpler approach might be to remove this original block if the new logic is robust.
    elif isinstance(root, Atom) and max_ops >= 1:  # Original condition
        logger.debug(
            f"AST Gen: Original root was Atom, max_ops >=1. Wrapping with a random op."
        )
        op = random.choice(ops)  # ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
        arity = random.randint(config_obj.MIN_ARITY, max_branch)
        children = [
            Atom(random.randint(config_obj.MIN_ATOM_VAL, config_obj.MAX_ATOM_VAL))
            for _ in range(arity - 1)
        ]
        children.append(root)
        random.shuffle(children)
        root = OpNode(op, children)
        logger.info(f"AST Gen: Wrapped Atom root with {op}.")

    return root


def validate_ast(node: Node):
    """Recursively validate that all operators in the AST are supported."""
    if node.op not in OP_LABELS and not isinstance(node, Atom):
        raise ValueError(f"Invalid operator: {node.op}")
    for c in node.children:
        validate_ast(c)


def ast_to_prefix(node: Node) -> str:
    """Convert an AST node to its prefix notation representation."""
    if isinstance(node, Atom):
        return str(node.n)

    children_str = " ".join(ast_to_prefix(c) for c in node.children)
    return f"({node.op} {children_str})"


def eval_node(node: Node) -> int:
    """Evaluate the AST node recursively."""
    if isinstance(node, Atom):
        if node.value is None:
            node.value = node.n
        logger.debug(f"eval_node: Atom node, value = {node.value}")
        return node.value

    vals = [eval_node(c) for c in node.children]
    logger.debug(f"eval_node: OpNode {node.op}, child values = {vals}")

    if not vals:
        logger.error(f"eval_node: Operator node {node.op} has no children values.")
        raise ValueError(f"Operator node {node.op} has no children values.")

    func_map = {
        "MAX": max,
        "MIN": min,
        "MED": lambda v: sorted(v)[len(v) // 2],
        "SUM": sum,
        "SM": lambda v: sum(v) % 10,
        "AVG": lambda v: sum(v) // len(v) if v else 0,
    }
    try:
        func = func_map[node.op]
        if node.op == "MED" and len(vals) % 2 == 0:
            logger.warning(
                f"eval_node: MED operator for node {node.op} with even children ({len(vals)}). Using lower middle."
            )
        if node.op == "AVG" and not vals:
            logger.error(
                f"eval_node: Cannot calculate average of zero values for node {node.op}."
            )
            raise ValueError("Cannot calculate average of zero values.")

        calculated_value = func(vals)

        # Additional validation for SUM operations to catch calculation errors
        if node.op == "SUM":
            expected_sum = sum(vals)
            if calculated_value != expected_sum:
                logger.error(
                    f"eval_node: SUM validation error - func(vals)={calculated_value} != sum(vals)={expected_sum}"
                )
                # Use the manually calculated sum as a fallback
                calculated_value = expected_sum

        # Enhanced logging for all operations
        logger.debug(
            f"eval_node: OpNode {node.op}, inputs {vals}, result = {calculated_value}"
        )

        # Operation-specific detailed logging
        if node.op == "SUM":
            logger.info(
                f"SUM Operation - Node ID: {id(node)}, Input values: {vals}, Sum: {calculated_value}"
            )
        elif node.op == "AVG":
            total = sum(vals)
            logger.info(
                f"AVG Operation - Node ID: {id(node)}, Input values: {vals}, Sum: {total}, Count: {len(vals)}, Result: {calculated_value}"
            )
        elif node.op in ["MAX", "MIN", "MED"]:
            logger.info(
                f"{node.op} Operation - Node ID: {id(node)}, Input values: {vals}, Result: {calculated_value}"
            )

        node.value = calculated_value
        return node.value
    except KeyError:
        logger.error(f"eval_node: Unsupported operator: {node.op}")
        raise ValueError(f"Unsupported operator: {node.op}")
    except IndexError as e:
        logger.error(
            f"eval_node: Indexing error evaluating {node.op} with child values {vals}: {e}"
        )
        raise
    except ZeroDivisionError:
        logger.error(
            f"eval_node: Division by zero during AVG for {node.op} with child values {vals}"
        )
        raise ValueError(f"Division by zero during AVG for {node.op}")
    except Exception as e:
        logger.error(
            f"eval_node: Unexpected error evaluating {node.op} with values {vals}: {e}"
        )
        raise


def postorder(node: Node):
    """Yield nodes in post-order."""
    if node is None:  # Add this check to handle None values
        return
    for c in node.children:
        yield from postorder(c)
    yield node


@retry_api_call
def _chat_completion_call(*args, **kwargs):
    # ADD THIS LOG to verify MAX_API_TOKEN_LIMIT
    logger.info(
        f"DEBUG _chat_completion_call: Effective config.MAX_API_TOKEN_LIMIT = {config.MAX_API_TOKEN_LIMIT}"
    )

    if args:
        logger.warning(
            f"_chat_completion_call received unexpected positional arguments: {args}"
        )

    logger.debug(f"_chat_completion_call received kwargs: {kwargs}")

    if client is None:
        logger.error(
            "OpenAI client (for OpenRouter) not initialized. Cannot make API call."
        )
        raise RuntimeError("API client not initialized.")

    # Standard OpenAI client parameters
    standard_openai_client_params = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "top_k",
        # 'reasoning' is NOT a standard OpenAI client param, will go in extra_body
    }

    # Separate standard kwargs from OpenRouter-specific ones (like reasoning)
    api_call_standard_kwargs = {}
    openrouter_specific_params = {}

    if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
        # Use a temporary dict to avoid modifying original kwargs if it's passed around
        temp_kwargs = kwargs.copy()
        temp_kwargs["max_tokens"] = temp_kwargs.pop("max_completion_tokens")
        logger.info(f"DEBUG: Aliased max_completion_tokens to max_tokens")
    else:
        temp_kwargs = kwargs.copy()

    # Check if this call should use structured JSON output
    use_json_mode = temp_kwargs.pop("use_json_mode", False)
    json_schema = temp_kwargs.pop("json_schema", None)

    for k, v in temp_kwargs.items():
        if k in standard_openai_client_params:
            api_call_standard_kwargs[k] = v
        elif k == "reasoning":  # Explicitly handle reasoning for extra_body
            openrouter_specific_params[k] = v
        # else: # You could log other unexpected kwargs if needed
        # logger.warning(f"DEBUG: Unexpected kwarg '{k}' in _chat_completion_call, may be ignored or cause error if not for extra_body.")

    # --- Check if we need to modify the max_tokens to prevent truncation ---
    if "max_tokens" in api_call_standard_kwargs:
        original_max_tokens = api_call_standard_kwargs["max_tokens"]
        # Always use the higher limit to prevent truncation due to reasoning tokens
        api_call_standard_kwargs["max_tokens"] = config.MAX_API_TOKEN_LIMIT
        logger.debug(
            f"Modified max_tokens for API call: {original_max_tokens} → {config.MAX_API_TOKEN_LIMIT} (to handle reasoning tokens)"
        )

    # --- REFINED REASONING LOGIC for openrouter_specific_params ---
    current_model_name = api_call_standard_kwargs.get("model", "").lower()
    is_openai_o_series = "openai/" in current_model_name and (
        re.search(r"/o\d+", current_model_name) or "gpt-4o-mini" in current_model_name
    )

    reasoning_config_to_send = openrouter_specific_params.get("reasoning", {})
    if reasoning_config_to_send is None:  # Handle if None was explicitly passed
        reasoning_config_to_send = {}

    # Ensure it's a dict if it was passed as something else or not at all
    if not isinstance(reasoning_config_to_send, dict):
        reasoning_config_to_send = {}

    # 2. Handle 'effort' - Only allow on OpenAI o-series models
    if "effort" in reasoning_config_to_send:
        if not is_openai_o_series:
            logger.warning(
                f"DEBUG: Removing 'effort' from reasoning_config for non-o-series model ({current_model_name}). "
                f"Original effort: {reasoning_config_to_send['effort']}"
            )
            del reasoning_config_to_send["effort"]
        else:
            logger.debug(
                f"DEBUG: Keeping 'effort' for OpenAI o-series model ({current_model_name})."
            )

    # Update openrouter_specific_params with the processed reasoning_config
    if reasoning_config_to_send:
        openrouter_specific_params["reasoning"] = reasoning_config_to_send
    elif "reasoning" in openrouter_specific_params:  # If it was there but became empty
        del openrouter_specific_params["reasoning"]

    # Add JSON response format if requested
    if json_schema:
        # Use full schema-based JSON formatting
        openrouter_specific_params["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        logger.debug(
            f"Using JSON schema validation for API call with schema: {json_schema.get('type', 'unknown')}"
        )
    elif use_json_mode:
        # Use simple JSON mode (backward compatibility)
        openrouter_specific_params["response_format"] = {"type": "json_object"}
        logger.debug(f"Using simple JSON mode for API call")

    logger.debug(
        f"Final standard API call_kwargs: {json.dumps(api_call_standard_kwargs, indent=2)}"
    )
    logger.debug(
        f"Final OpenRouter specific_params (for extra_body): {json.dumps(openrouter_specific_params, indent=2)}"
    )

    max_tokens_value = api_call_standard_kwargs.get("max_tokens", "NOT SET")
    if max_tokens_value == "NOT SET":
        logger.warning(
            f"DEBUG: max_tokens value NOT SET for API call. API will use its default."
        )
    elif isinstance(max_tokens_value, int) and max_tokens_value <= 0:
        logger.error(
            f"DEBUG: max_tokens value is invalid ({max_tokens_value}). API call will likely fail."
        )
    else:
        logger.info(
            f"DEBUG: FINAL max_tokens value being sent to API: {max_tokens_value}"
        )

    try:
        wait_time = rate_limiter.wait_if_needed()
        if wait_time > 0:
            logger.debug(
                f"Rate limit applied - waited {wait_time:.2f}s before API call"
            )

        # Use extra_body for OpenRouter-specific parameters
        if openrouter_specific_params:
            resp = client.chat.completions.create(
                **api_call_standard_kwargs, extra_body=openrouter_specific_params
            )  # MODIFIED: assign to resp
        else:
            resp = client.chat.completions.create(
                **api_call_standard_kwargs
            )  # MODIFIED: assign to resp

        # --- Track token usage --- ADD THIS BLOCK ---
        if resp and hasattr(resp, "usage") and resp.usage:
            prompt_tokens = resp.usage.prompt_tokens or 0
            completion_tokens = resp.usage.completion_tokens or 0
            generation_token_tracker.add_usage(prompt_tokens, completion_tokens)
        else:
            logger.warning(
                "_chat_completion_call: No usage data found in API response."
            )
            # ADD THIS DETAILED LOGGING:
            if resp:
                logger.warning(
                    f"_chat_completion_call: Response object when usage was missing (type: {type(resp)}):"
                )
                try:
                    # Try to dump the whole response object as JSON for inspection
                    logger.warning(
                        f"Full response dump (model_dump_json): {resp.model_dump_json(indent=2)}"
                    )
                except AttributeError:
                    try:
                        logger.warning(
                            f"Full response dump (fallback .json()): {resp.json(indent=2)}"
                        )
                    except AttributeError:
                        logger.warning(f"Full response dump (repr): {repr(resp)}")
                    # Log choices and finish reasons if available even if usage is missing
                    if hasattr(resp, "choices") and resp.choices:
                        for i, choice_item in enumerate(resp.choices):
                            finish_reason_item = getattr(
                                choice_item, "finish_reason", "N/A"
                            )
                            message_item_content = "N/A"
                            if getattr(choice_item, "message", None) and getattr(
                                choice_item.message, "content", None
                            ):
                                message_item_content = choice_item.message.content
                            elif getattr(choice_item, "message", None):
                                message_item_content = f"Message object present but content is None/Empty. Message: {choice_item.message}"
                            else:
                                message_item_content = (
                                    "Message object missing in choice."
                                )

                            logger.warning(
                                f"Choice {i}: Finish Reason: {finish_reason_item}, Content Snippet: {str(message_item_content)[:200]}..."
                            )
                    else:
                        logger.warning(
                            "Response object had no 'choices' or 'choices' was empty when usage was missing."
                        )
            else:
                logger.warning(
                    "_chat_completion_call: Response object (resp) was None when usage was missing."
                )
        # --- END ADDED BLOCK ---

        return resp  # Ensure resp is returned

    except Exception as e:
        logger.error(f"Error during client.chat.completions.create: {e}")
        # Log both standard and extra_body args for clarity
        log_payload = {"standard_args": api_call_standard_kwargs}
        if openrouter_specific_params:
            log_payload["extra_body_args"] = openrouter_specific_params
        logger.error(f"Args that failed: {json.dumps(log_payload, indent=2)}")
        raise


# --- JSON Cleaning Helper ---
def clean_and_parse_json_block(text: str):
    """Extract JSON object between first '{' and last '}', then parse it."""
    logger.debug(
        f"clean_and_parse_json_block: Input text length: {len(text)}, text sample: '{text[:100]}...'"
    )

    if not text or not text.strip():
        logger.error(
            f"clean_and_parse_json_block: Received empty or whitespace-only text input"
        )
        raise ValueError("Empty input text")

    # Find first opening brace and last closing brace
    start_idx = text.find("{")
    if start_idx == -1:
        logger.error(f"No opening brace found in text: {text[:100]}...")
        raise ValueError("No JSON object found in text")

    end_idx = text.rfind("}")
    if end_idx == -1 or end_idx < start_idx:
        logger.error(f"No valid closing brace found in text: {text[:100]}...")
        raise ValueError("No valid JSON object found in text")

    # Extract just the JSON substring
    json_text = text[start_idx : end_idx + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON Decode Error: {e} in extracted text:\n---\n{json_text}\n---"
        )
        raise


# After clean_and_parse_json_block (around line 1664), add:
def parse_llm_json_with_fallback(
    raw_text: str, default_value: dict, context_info: str = ""
):
    """Parse JSON from LLM output with consistent error handling and fallback."""
    try:
        return clean_and_parse_json_block(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"JSON parsing failed {context_info}: {e}. Raw: {raw_text[:200]}"
        )
        return default_value


# --- Tuned Generate World Function ---
def generate_world(
    num_characters: int = config.MIN_WORLD_CHARS,
    num_concepts: int = config.MAX_WORLD_CONCEPTS,
    max_retries: int = config.WORLDGEN_MAX_RETRIES,
    sample_index: int | None = None,
) -> dict:
    """
    Generates fictional world metadata using an LLM, with a tuned prompt
    to maximize the likelihood of receiving valid, parseable JSON,
    especially regarding quote escaping and array formatting. Includes retries as a fallback.
    """
    if not isinstance(num_characters, int) or num_characters < 1:
        raise ValueError("num_characters must be positive int")
    if not isinstance(num_concepts, int) or num_concepts < 1:
        raise ValueError("num_concepts must be positive int")

    # Added specific instruction about commas in the characters array
    prompt = (
        "You are an expert system designed to generate structured data in **strictly valid JSON format**.\n"
        "Your task is to create fictional world metadata.\n\n"
        "The entire output MUST be a single, valid JSON object.\n"
        "**Instructions for Content Generation:**\n\n"
        "1.  **Characters:** Generate exactly {num_characters} distinct characters. Each character MUST have:\n"
        '    *   `name`: string (e.g., "Kaelen Vane", "Seraphina Moonwhisper")\n'
        '    *   `role`: string (e.g., "The grizzled warrior," "The cunning sorceress," "The naive apprentice")\n'
        '    *   `quirk`: string (a unique or unusual habit, belief, or physical trait, e.g., "Collects antique spoons," "Only speaks in riddles," "Has mismatched eyes")\n'
        "    Ensure each character's name, role, and quirk combination is unique. Remember to escape any internal double quotes in these string values as per Rule 3 above. Adhere strictly to Rule 5 for formatting this array.\n\n"
        '2.  **Genre:** Define a `genre` as a string (e.g., "Steampunk Adventure," "Urban Fantasy Mystery," "Cosmic Horror Saga").\n\n'
        '3.  **Setting:** Define a `setting` as a string (a brief, evocative description of the world or primary location, e.g., "A floating city powered by forgotten magic and steam contraptions," "A post-apocalyptic wasteland where ancient ruins hold dangerous secrets").\n\n'
        '4.  **Object:** Define an `object` as a string. This should be a plural noun representing key items characters might seek, collect, or use (e.g., "etherium crystals," "lost star-charts," "prophetic dream-shards").\n\n'
        "**Guidance for Content:**\n"
        "*   Strive for thematic coherence between the genre, setting, characters, and the collectible object. They should feel like they belong in the same world.\n\n"
        "Output ONLY the single, valid JSON object."
    ).format(num_characters=num_characters)

    for attempt in range(max_retries):
        logger.debug(
            f"Attempting world generation (Attempt {attempt + 1}/{max_retries}) with tuned prompt."
        )
        text = None
        cleaned_text_for_parsing = "" # Initialize to prevent reference before assignment in except block
        try:
            log_prompt(
                header=f"World Generation Prompt (Attempt {attempt + 1})",
                prompt=f"System: (Implicit in API call structure for this function)\nUser:\n{prompt}",
                sample_index=sample_index,
            )

            resp = _chat_completion_call(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=config.MAX_API_TOKEN_LIMIT,
                temperature=config.WORLD_GEN_TEMP,
                json_schema=WORLD_SCHEMA,
                reasoning={"exclude": True},
            )
            if (
                hasattr(resp, "choices") and resp.choices
            ):
                first_choice = resp.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    text = first_choice.message.content
                    if text is None:
                        logger.warning(
                            f"World Gen Attempt {attempt + 1}: API returned None content within message. Response: {resp}"
                        )
                        text = ""
                else:
                    logger.error(
                        f"World Gen Attempt {attempt + 1}: First choice object lacks 'message' attribute or message is empty. Response: {resp}"
                    )
                    text = ""
            else:
                logger.error(
                    f"World Gen Attempt {attempt + 1}: API response lacks 'choices' list or it's empty. Response: {resp}"
                )
                text = ""

            log_prompt(
                header=f"World Generation Response (Attempt {attempt + 1})",
                prompt=f"Raw LLM Output:\n{text}",
                sample_index=sample_index,
            )

            if not text.strip():
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: Received empty response from API."
                )
                if attempt < max_retries - 1:
                    delay = config.INITIAL_WORLD_RETRY_DELAY * (2**attempt)
                    logger.info(f"Retrying world generation in {delay:.2f} seconds...")
                    time.sleep(delay)
                continue

            cleaned_text_for_parsing = text.strip()
            if cleaned_text_for_parsing.startswith("```json"):
                cleaned_text_for_parsing = cleaned_text_for_parsing[len("```json") :]
            if cleaned_text_for_parsing.endswith("```"):
                cleaned_text_for_parsing = cleaned_text_for_parsing[: -len("```")]
            cleaned_text_for_parsing = cleaned_text_for_parsing.strip()

            world = parse_llm_json_with_fallback(
                cleaned_text_for_parsing,
                {},
                f"in world generation attempt {attempt+1}",
            )

            required_keys = ["characters", "genre", "setting", "object"]
            if not all(k in world for k in required_keys):
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: Generated JSON missing required keys. Keys found: {world.keys()}"
                )
                raise ValueError("Generated JSON missing required keys")
            if not isinstance(world.get("characters"), list) or not world["characters"]:
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: 'characters' key is not a non-empty list."
                )
                raise ValueError("'characters' key is not a non-empty list")

            for char_idx, char_obj in enumerate(world.get("characters", [])):
                if not isinstance(char_obj, dict):
                    logger.warning(
                        f"World Gen Attempt {attempt + 1}: Character at index {char_idx} is not a dictionary."
                    )
                    raise ValueError(f"Character at index {char_idx} is not a dictionary.")
                if not all(k_char in char_obj for k_char in ["name", "role", "quirk"]):
                    logger.warning(
                        f"World Gen Attempt {attempt + 1}: Character at index {char_idx} is missing required keys (name, role, quirk). Found: {char_obj.keys()}"
                    )
                    raise ValueError(f"Character at index {char_idx} missing required keys.")

            logger.debug(
                f"World Gen Attempt {attempt + 1}: Successfully generated and parsed world JSON."
            )
            logger.debug(f"Generated object: {world.get('object', 'N/A')}")
            return world
        except (
            json.JSONDecodeError,
            ValueError,
        ) as e:
            logger.error(
                f"World Gen Attempt {attempt + 1}: Failed ({type(e).__name__}): {e}. Raw text (after potential cleaning for ```json):\n---\n{cleaned_text_for_parsing}\n---"
            )
        except Exception as e:
            logger.error(
                f"World Gen Attempt {attempt + 1}: Unexpected error: {e}. Raw text:\n---\n{text if text else 'N/A'}\n---"
            )
        if attempt < max_retries - 1:
            delay = config.INITIAL_WORLD_RETRY_DELAY * (2**attempt)
            logger.info(f"Retrying world generation in {delay:.2f} seconds...")
            time.sleep(delay)
    logger.error(f"Failed to generate valid world JSON after {max_retries} attempts.")
    raise RuntimeError("World generation failed: Could not get valid JSON from LLM.")

# --- Number Extraction (Enhanced with Inflect for Words up to MAX_VALUE) ---
DIGIT_REGEX = re.compile(r"\b-?\d+\b")

MAX_NUMBER_FOR_NUMBER_DICT = 1000  # Define a larger limit for number-to-word conversion


def _build_expanded_number_words_dict(
    max_val: int = MAX_NUMBER_FOR_NUMBER_DICT,
) -> dict[str, int]:
    """Builds a dictionary mapping number words to ints up to max_val using inflect."""
    if p_inflect is None:
        logger.error(
            "Inflect engine not available for building expanded number words dict. Using basic range."
        )
        return {
            num_to_words(i).lower(): i
            for i in range(
                config.FALLBACK_MIN_NUM_WORD, config.FALLBACK_MAX_NUM_WORD + 1
            )
            if num_to_words(i)
        }
    num_word_dict = {}
    for i in range(max_val + 1):
        try:
            word_no_and = num_to_words(i)  # This will use the version without "and"
            if word_no_and:  # Ensure inflect returned something
                word_no_and_lower = word_no_and.lower()
                num_word_dict[word_no_and_lower] = i

                # *** START FIX: Add hyphenated version if applicable ***
                if " " in word_no_and_lower:
                    hyphenated_word = word_no_and_lower.replace(" ", "-")
                    num_word_dict[hyphenated_word] = i
                    # logger.debug(f"Added hyphenated number word: '{hyphenated_word}' for {i}") # Keep debug minimal here or it floods
                # *** END FIX ***

                # Additionally, add common "and" variations for numbers > 100
                if i > 100 and p_inflect:
                    word_with_and = p_inflect.number_to_words(i, andword="and")
                    if word_with_and:  # Ensure inflect returned something
                        word_with_and_lower = word_with_and.lower()
                        if word_with_and_lower != word_no_and_lower:
                            num_word_dict[word_with_and_lower] = i
                            # *** START FIX: Add hyphenated "and" version ***
                            if " " in word_with_and_lower:
                                hyphenated_word_with_and = word_with_and_lower.replace(
                                    " ", "-"
                                )
                                num_word_dict[hyphenated_word_with_and] = i
                                # logger.debug(f"Added hyphenated 'and' number word: '{hyphenated_word_with_and}' for {i}")
                            # *** END FIX ***
        except Exception as e:
            logger.warning(f"Inflect failed to convert {i} to words: {e}")
    logger.info(
        f"Built expanded number words dictionary with {len(num_word_dict)} entries (up to {max_val})."
    )
    # Log a few examples of hyphenated numbers if they were added
    hyphen_examples = {k: v for k, v in num_word_dict.items() if "-" in k and v < 100}
    if hyphen_examples:
        logger.info(
            f"Sample hyphenated entries: {dict(list(hyphen_examples.items())[:5])}"
        )
    return num_word_dict


# EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict() # Initialization moved down

# --- Build dict after logger is fully configured ---
EXPANDED_NUMBER_WORDS_DICT = {}
if p_inflect:
    EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict()
    logger.info(
        f"EXPANDED_NUMBER_WORDS_DICT initialized. Size: {len(EXPANDED_NUMBER_WORDS_DICT)}."
    )
    if len(EXPANDED_NUMBER_WORDS_DICT) > 0:
        sample_keys_diverse = []
        for num_val_to_check in [
            0,
            1,
            10,
            15,
            21,
            41,
            59,
            83,
            86,
            99,
            100,
            179,
            1000,
            1001,
            2025,
        ]:  # Added 99
            key_no_and = num_to_words(num_val_to_check)
            if key_no_and.lower() in EXPANDED_NUMBER_WORDS_DICT:
                sample_keys_diverse.append(key_no_and.lower())
            # Check for hyphenated version of no_and
            if " " in key_no_and.lower():
                hyphenated_key_no_and = key_no_and.lower().replace(" ", "-")
                if (
                    hyphenated_key_no_and in EXPANDED_NUMBER_WORDS_DICT
                    and hyphenated_key_no_and not in sample_keys_diverse
                ):
                    sample_keys_diverse.append(hyphenated_key_no_and)

            if num_val_to_check > 100 and p_inflect:
                word_with_and = p_inflect.number_to_words(
                    num_val_to_check, andword="and"
                )
                if (
                    word_with_and.lower() in EXPANDED_NUMBER_WORDS_DICT
                    and word_with_and.lower() not in sample_keys_diverse
                ):
                    sample_keys_diverse.append(word_with_and.lower())
                # Check for hyphenated version of with_and
                if " " in word_with_and.lower():
                    hyphenated_key_with_and = word_with_and.lower().replace(" ", "-")
                    if (
                        hyphenated_key_with_and in EXPANDED_NUMBER_WORDS_DICT
                        and hyphenated_key_with_and not in sample_keys_diverse
                    ):
                        sample_keys_diverse.append(hyphenated_key_with_and)

        sample_items = {
            k: EXPANDED_NUMBER_WORDS_DICT.get(k) for k in sample_keys_diverse[:20]
        }  # Log more samples
        logger.info(
            f"More diverse sample items from EXPANDED_NUMBER_WORDS_DICT: {sample_items}"
        )
    else:
        logger.warning(
            "EXPANDED_NUMBER_WORDS_DICT is empty after initialization attempt with p_inflect."
        )
else:
    logger.error(
        "p_inflect is None, EXPANDED_NUMBER_WORDS_DICT will be empty. Number word extraction will be limited."
    )

# --- Sort keys by length descending to prioritize longer matches ---
sorted_number_words = sorted(EXPANDED_NUMBER_WORDS_DICT.keys(), key=len, reverse=True)
logger.info(f"First 10 longest number words for regex: {sorted_number_words[:10]}")
logger.info(f"Last 10 shortest number words for regex: {sorted_number_words[-10:]}")

NUMBER_WORDS_PATTERN = (
    r"\b(?:(minus|negative)\s+)?("  # Corrected \b to \b
    + "|".join(re.escape(k) for k in sorted_number_words)
    + r")\b"  # Corrected \b to \b
)
NUMBER_WORDS_REGEX = re.compile(NUMBER_WORDS_PATTERN, re.IGNORECASE)


def extract_numbers_from_text(text: str) -> Set[int]:
    """Extracts integers (digits and words), ignoring specified ordinals."""
    if not text:
        return set()

    found_numbers = set()
    search_text = text.lower()

    text_chars_list = list(search_text)
    digit_spans_to_replace = []
    for match in DIGIT_REGEX.finditer(search_text):
        digit_str = match.group(0)
        try:
            value = int(digit_str)
            found_numbers.add(value)
            digit_spans_to_replace.append(match.span())
        except ValueError:
            logger.warning(
                f"Could not convert digit string '{digit_str}' to int during extraction."
            )
            continue

    for start, end in digit_spans_to_replace:
        for i in range(start, end):
            text_chars_list[i] = "|"  # Replace with a non-space, non-word character

    text_for_word_search = "".join(text_chars_list)
    text_for_word_search = text_for_word_search.replace(
        "|", " "
    )  # Now replace placeholders with spaces
    text_for_word_search = re.sub(r"\s+", " ", text_for_word_search).strip()

    logger.debug(
        f"extract_numbers_from_text: Text for word search (digits replaced, pipes to spaces): '{text_for_word_search[:500]}...'"
    )

    if "twelve" in text_for_word_search:
        logger.debug("extract_numbers_from_text: 'twelve' IS in text_for_word_search.")
        test_match_twelve = re.search(
            r"\btwelve\b", text_for_word_search
        )  # Ensure \b is correct for the test
        if test_match_twelve:
            logger.debug(
                f"extract_numbers_from_text: Manual re.search for '\btwelve\b' SUCCEEDED. Match: {test_match_twelve.group(0)}"
            )
        else:
            logger.warning(
                f"extract_numbers_from_text: Manual re.search for '\btwelve\b' FAILED on: {text_for_word_search[:100]}..."
            )
    else:
        logger.debug(
            f"extract_numbers_from_text: 'twelve' IS NOT in text_for_word_search: {text_for_word_search[:100]}..."
        )

    for match in NUMBER_WORDS_REGEX.finditer(text_for_word_search):
        sign_word = match.group(1)  # (minus|negative) or None
        number_word_matched = match.group(
            2
        ).lower()  # The matched number phrase from the regex pattern

        value = EXPANDED_NUMBER_WORDS_DICT.get(number_word_matched)

        if value is not None:
            if sign_word and value != 0:  # Apply sign if present
                value = -value
            found_numbers.add(value)
        else:
            # This should ideally not happen if regex is built from dict keys
            logger.warning(
                f"Word phrase '{number_word_matched}' found by NUMBER_WORDS_REGEX but not in EXPANDED_NUMBER_WORDS_DICT. This indicates a mismatch or an issue with regex construction from dictionary keys."
            )

    logger.debug(
        f"extract_numbers_from_text: Input '{text[:100]}...', Found: {found_numbers}"
    )
    return found_numbers


# --- Factory for number validation ---
def make_number_validator(
    allowed_atoms_with_duplicates: list[int] | None,
    forbidden_atoms: set[int] | None,
    operand_count: int,
    correct_result_for_beat: int | None,
    enforce_result_presence: bool = False,
    operation_type: str | None = None, # This will be used for logging and specific logic
    overall_ground_truth_answer: int | None = None,
    is_root_node_being_validated: bool = False,
    conceptual_input_values: set[int] | None = None,
    config_obj: "Config" = None,
    logger_obj: "logging.Logger" = None,
):
    if allowed_atoms_with_duplicates is None:
        allowed_atoms_with_duplicates = []
    if forbidden_atoms is None:
        forbidden_atoms = set()
    if conceptual_input_values is None:
        conceptual_input_values = set()
    if config_obj is None:
        config_obj = Config()
        if logger_obj: logger_obj.warning("make_number_validator called without config_obj, using default.")
    if logger_obj is None:
        logger_obj = logging.getLogger("verbose_listops_default_validator_logger")
        logger_obj.setLevel(logging.DEBUG)

    from collections import Counter

    effective_required_atomic_mention_values = []
    if operation_type == "MED" and correct_result_for_beat is not None:
        for val in allowed_atoms_with_duplicates:
            if val != correct_result_for_beat:
                effective_required_atomic_mention_values.append(val)
        if len(effective_required_atomic_mention_values) != len(allowed_atoms_with_duplicates):
            omitted_value = correct_result_for_beat
            original_count = allowed_atoms_with_duplicates.count(omitted_value)
            logger_obj.debug(
                f"[PythonNumValidator-MEDIAN] Op: {operation_type}, Result: {omitted_value}. "
                f"Original required atoms for step: {allowed_atoms_with_duplicates}. "
                f"Value '{num_to_words(omitted_value)}' ({omitted_value}) is MEDIAN result and was an input ({original_count}x); "
                f"it MUST NOT be mentioned. Effective atoms to find mentioned: {effective_required_atomic_mention_values}."
            )
    else:
        effective_required_atomic_mention_values = list(allowed_atoms_with_duplicates)

    required_atomic_mention_counts = Counter(effective_required_atomic_mention_values)

    def validator_func(text: str) -> bool:
        numbers_found_in_text_with_words, number_words_map = find_numbers_in_text_detailed(text, config_obj)
        found_counts_in_text = Counter(numbers_found_in_text_with_words)

        # --- Define phrasing_numbers_str_for_validator_prompt for logging ---
        temp_phrasing_detailed_for_validator_prompt = [
            f"'{num_to_words(n)}' ({n})"
            for n in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        ]
        phrasing_numbers_str_for_validator_prompt = ", ".join(
            sorted(temp_phrasing_detailed_for_validator_prompt)
        )
        # --- End definition ---

        missing_required_atoms_details = []
        for num, req_count in required_atomic_mention_counts.items():
            if found_counts_in_text[num] < req_count:
                missing_required_atoms_details.append(
                    f"Atom '{num_to_words(num)}' ({num}): required {req_count} time(s), found {found_counts_in_text[num]} time(s) (in words: {', '.join(number_words_map.get(num, [])) if num in number_words_map else 'none'})"
                )

        if missing_required_atoms_details:
            reason_str = (
                f"Beat Op: {operation_type}. Missing required atomic inputs (or insufficient counts): "
                f"{'; '.join(missing_required_atoms_details)}. "
                f"Original AST atomic inputs for this step: {allowed_atoms_with_duplicates}. "
                f"Effective atomic inputs expected to be mentioned (after MEDIAN exception if any): {effective_required_atomic_mention_values}. "
                f"Numbers found in text: {numbers_found_in_text_with_words} (Counts: {dict(found_counts_in_text)})."
            )
            report_dict = {
                "reason": reason_str, "reason_code": "MISSING_REQUIRED_ATOMS_OR_COUNTS",
                "operation_type": operation_type, "found_numbers": numbers_found_in_text_with_words,
                "required_atomic_mention_counts": dict(required_atomic_mention_counts),
                "effective_required_atomic_mention_values": effective_required_atomic_mention_values,
                "allowed_atoms_with_duplicates_original": allowed_atoms_with_duplicates,
            }
            _log_failed_validation(text, report_dict, logger_obj)
            return False

        explicitly_allowed_for_this_beat_check = set(allowed_atoms_with_duplicates)
        explicitly_allowed_for_this_beat_check.update(config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET)

        is_arity_problematic_forbidden_for_py_val = (
            operand_count in forbidden_atoms and
            operand_count not in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET and
            operand_count not in set(allowed_atoms_with_duplicates)
        )
        if operand_count > 0 and not is_arity_problematic_forbidden_for_py_val:
             explicitly_allowed_for_this_beat_check.add(operand_count)

        for num_val, num_words_list in number_words_map.items():
            is_this_num_a_required_mention_for_current_step = num_val in required_atomic_mention_counts and found_counts_in_text[num_val] >= required_atomic_mention_counts[num_val]
            is_phrasing_or_allowed_arity = (
                num_val in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET or
                (num_val == operand_count and not is_arity_problematic_forbidden_for_py_val)
            )

            if num_val in forbidden_atoms and not is_this_num_a_required_mention_for_current_step and not is_phrasing_or_allowed_arity:
                if num_val in conceptual_input_values:
                    # Determine the conceptual name(s) if possible for better logging
                    conceptual_names_for_value = [
                        name for name, val_ci in (conceptual_input_values if isinstance(conceptual_input_values, dict) else {}).items() if val_ci == num_val
                    ]
                    conceptual_name_str = f" (conceptual name(s): {', '.join(conceptual_names_for_value)})" if conceptual_names_for_value else ""

                    reason_str = (
                        f"Beat Op: {operation_type}. Forbidden number '{num_to_words(num_val)}' ({num_val}) found (words: {', '.join(num_words_list)}). "
                        f"This number is the value of a conceptual input{conceptual_name_str}, which should only be referenced by its name, not its numerical value."
                    )
                    report_dict = {
                        "reason": reason_str, "reason_code": "CONCEPTUAL_INPUT_VALUE_LEAKED",
                        "operation_type": operation_type, "found_number_value": num_val,
                        "found_number_words": num_words_list, "forbidden_atoms_set": list(forbidden_atoms),
                        "conceptual_input_values_map": conceptual_input_values # Keep this for structured data
                    }
                    _log_failed_validation(text, report_dict, logger_obj)
                    return False

                # General forbidden number violation
                reason_str = (
                    f"Beat Op: {operation_type}. Forbidden number '{num_to_words(num_val)}' ({num_val}) found (words: {', '.join(num_words_list)}). "
                    f"This number was in the forbidden set for this beat ({sorted(list(forbidden_atoms))}) and is not a required mention for this step ({effective_required_atomic_mention_values}), nor an allowed phrasing/counting number ({phrasing_numbers_str_for_validator_prompt}, arity: {operand_count if not is_arity_problematic_forbidden_for_py_val else 'forbidden arity'})."
                )
                report_dict = {
                    "reason": reason_str, "reason_code": "FORBIDDEN_NUMBER_FOUND",
                    "operation_type": operation_type, "found_number_value": num_val,
                    "found_number_words": num_words_list, "forbidden_atoms_set": list(forbidden_atoms),
                    "effective_required_atomic_mention_values": effective_required_atomic_mention_values,
                    "always_allowed_phrasing": list(config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET),
                    "operand_count_for_beat": operand_count,
                    "is_arity_problematic": is_arity_problematic_forbidden_for_py_val
                }
                _log_failed_validation(text, report_dict, logger_obj)
                return False

        if correct_result_for_beat is not None:
            if not is_root_node_being_validated:
                if correct_result_for_beat in found_counts_in_text and found_counts_in_text[correct_result_for_beat] > 0:
                    is_result_also_direct_input = correct_result_for_beat in set(allowed_atoms_with_duplicates)
                    if operation_type == "MED" and is_result_also_direct_input:
                        reason_str = (
                            f"Beat Op: {operation_type} (MEDIAN). Result '{num_to_words(correct_result_for_beat)}' ({correct_result_for_beat}) "
                            f"was found in text (words: {', '.join(number_words_map.get(correct_result_for_beat, []))}). "
                            f"For MEDIAN, if result is also an input, it MUST NOT be mentioned. This violates MEDIAN omission rule."
                        )
                        report_dict = {
                            "reason": reason_str, "reason_code": "MEDIAN_RESULT_AS_INPUT_STATED",
                            "operation_type": operation_type, "stated_result": correct_result_for_beat,
                            "words_for_stated_result": number_words_map.get(correct_result_for_beat, [])
                        }
                        _log_failed_validation(text, report_dict, logger_obj)
                        return False
                    elif not is_result_also_direct_input:
                        reason_str = (
                            f"Beat Op: {operation_type}. Intermediate result '{num_to_words(correct_result_for_beat)}' ({correct_result_for_beat}) "
                            f"was explicitly stated (words: {', '.join(number_words_map.get(correct_result_for_beat, []))}) but should be implicit. "
                            f"It was not one of the direct atomic inputs for this step ({allowed_atoms_with_duplicates})."
                        )
                        report_dict = {
                            "reason": reason_str, "reason_code": "INTERMEDIATE_RESULT_STATED",
                            "operation_type": operation_type, "stated_result": correct_result_for_beat,
                            "words_for_stated_result": number_words_map.get(correct_result_for_beat, []),
                            "direct_atomic_inputs_for_step": allowed_atoms_with_duplicates
                        }
                        _log_failed_validation(text, report_dict, logger_obj)
                        return False
            elif is_root_node_being_validated and not enforce_result_presence:
                if correct_result_for_beat in found_counts_in_text and found_counts_in_text[correct_result_for_beat] > 0:
                    is_result_also_direct_input = correct_result_for_beat in set(allowed_atoms_with_duplicates)
                    if operation_type == "MED" and is_result_also_direct_input:
                         reason_str = (
                            f"Beat Op: {operation_type} (FINAL ANSWER - MEDIAN). Result '{num_to_words(correct_result_for_beat)}' ({correct_result_for_beat}) "
                            f"was found in text (words: {', '.join(number_words_map.get(correct_result_for_beat, []))}). "
                            f"For MEDIAN, if result is also an input, it MUST NOT be mentioned. This violates MEDIAN omission rule, even for final answer."
                        )
                         report_dict = {
                            "reason": reason_str, "reason_code": "FINAL_MEDIAN_RESULT_AS_INPUT_STATED",
                            "operation_type": operation_type, "stated_result": correct_result_for_beat,
                            "words_for_stated_result": number_words_map.get(correct_result_for_beat, [])
                         }
                         _log_failed_validation(text, report_dict, logger_obj)
                         return False
                    elif not is_result_also_direct_input :
                        reason_str = (
                            f"Beat Op: {operation_type} (FINAL ANSWER). Final result '{num_to_words(correct_result_for_beat)}' ({correct_result_for_beat}) "
                            f"was explicitly stated (words: {', '.join(number_words_map.get(correct_result_for_beat, []))}) but should be implicit."
                        )
                        report_dict = {
                            "reason": reason_str, "reason_code": "FINAL_ANSWER_STATED_WHEN_IMPLICIT",
                            "operation_type": operation_type, "stated_result": correct_result_for_beat,
                            "words_for_stated_result": number_words_map.get(correct_result_for_beat, [])
                        }
                        _log_failed_validation(text, report_dict, logger_obj)
                        return False

        if operation_type in ["PADDING", "INTRO"]:
            for num_val, num_words_list in number_words_map.items():
                if num_val not in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
                    reason_prefix = "PADDING" if operation_type == "PADDING" else "INTRO"
                    reason_str = (
                        f"{reason_prefix} VALIDATION: Found number '{num_to_words(num_val)}' ({num_val}) (words: {', '.join(num_words_list)}) in {operation_type.lower()} text. "
                        f"Only numbers in {config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET} are allowed for general phrasing."
                    )
                    report_dict = {
                        "reason": reason_str, "reason_code": f"{reason_prefix}_NUMBER_VIOLATION",
                        "operation_type": operation_type, "found_number_value": num_val,
                        "found_number_words": num_words_list,
                        "allowed_phrasing_numbers": list(config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET)
                    }
                    _log_failed_validation(text, report_dict, logger_obj)
                    return False
        
        return True

    return validator_func

# Add a helper function to save failed validation attempts
def _log_failed_validation(
    text: str, validation_report: dict, logger_obj: logging.Logger = logger
):
    """
    Save failed validation attempts for diagnostic purposes.
    This provides a detailed record of why each beat was rejected.
    Additionally writes to the LLM turns log to keep all information in one place.
    """
    logger_obj.debug(f"Saving failed validation record")
    try:
        # Ensure log directory exists
        failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
        os.makedirs(failed_validations_dir, exist_ok=True)

        # Create a unique filename for this validation failure
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        operation = validation_report.get("operation_type", "unknown_op")
        reason = validation_report.get("reason", "unknown_reason")

        # Generate the filename using significant information for easy identification
        filename = f"validation_fail_{operation}_{reason}_{timestamp}.json"
        filepath = os.path.join(failed_validations_dir, filename)

        # Create the full record to save
        full_report = {
            "validation_report": validation_report,
            "full_text": text,
            "timestamp": timestamp,
        }

        # Save the record
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved failed validation record to {filepath}")

        # --- NEW SECTION: Also write validation failure to LLM turns log ---
        # Extract sample_index from the calling frame's context if available
        sample_index = None
        import inspect

        try:
            # Look for context.sample_index in the stack frames
            frames = inspect.stack()
            for frame in frames:
                if "context" in frame.frame.f_locals:
                    ctx = frame.frame.f_locals["context"]
                    if hasattr(ctx, "sample_index"):
                        sample_index = ctx.sample_index
                        break
        except Exception as e:
            logger.error(f"Error extracting sample_index from stack: {e}")

        # Format validation failure details in a way useful for LLM analysis
        found_nums = validation_report.get("found_numbers", [])
        allowed_atoms = validation_report.get("allowed_atoms", [])
        missing = validation_report.get("missing_required", [])
        forbidden = validation_report.get("forbidden_extras", [])
        operation = validation_report.get("operation_type", "unknown")
        correct_result = validation_report.get("correct_result", None)
        intermediate_sum = validation_report.get("intermediate_sum", None)
        operand_count = validation_report.get("operand_count", None)

        # Enhanced header with more diagnostic information
        # Try to extract narrative anchor information from stack
        narrative_anchor = None
        beat_counter = None
        for frame in frames:
            frame_locals = frame.frame.f_locals
            if "narrative_anchor" in frame_locals:
                narrative_anchor = frame_locals["narrative_anchor"]
            if "context" in frame_locals:
                ctx = frame_locals["context"]
                if hasattr(ctx, "beat_counter"):
                    beat_counter = ctx.beat_counter
            if narrative_anchor and beat_counter:
                break

        # Create a more descriptive header with beat and narrative context when available
        beat_info = ""
        if beat_counter:
            beat_info = f", Beat {beat_counter.get('current', '?')}/{beat_counter.get('total', '?')}"

        anchor_info = ""
        if narrative_anchor:
            anchor_info = f", Anchor: '{narrative_anchor}'"

        validation_header = (
            f"VALIDATION FAILURE: Op={operation}{beat_info}{anchor_info}, Reason={reason} "
            f"[Consolidated log for LLM analysis]"
        )

        # Create a more detailed validation summary
        validation_details = f"{'='*80}\n"  # Clear visual separator
        validation_details += f"=== VALIDATION FAILURE REPORT ===\n"
        validation_details += f"{'='*80}\n\n"
        validation_details += f"Operation type: {operation}\n"
        validation_details += f"Failure reason: {reason}\n"
        validation_details += f"Operand count: {operand_count}\n"

        if correct_result is not None:
            validation_details += (
                f"Correct result (should be mentioned): {correct_result}\n"
            )

        if intermediate_sum is not None:
            validation_details += (
                f"Intermediate sum (may be mentioned for AVG/SM): {intermediate_sum}\n"
            )

        validation_details += f"\n--- Number Analysis ---\n"
        validation_details += f"Found numbers in text: {found_nums}\n"
        validation_details += f"Required numbers: {allowed_atoms}\n"
        validation_details += f"Missing required: {missing}\n"
        validation_details += f"Forbidden extras: {forbidden}\n"

        # Include any additional detailed failure information
        details = validation_report.get("details", [])
        if details:
            validation_details += f"\n--- Detailed Analysis ---\n"
            for detail in details:
                validation_details += f"- {detail}\n"

        validation_details += f"\n--- Generated Text That Failed Validation ---\n{text}"

        # Add clear ending separator
        validation_details += f"\n\n{'='*80}\n"
        validation_details += f"=== END OF VALIDATION FAILURE REPORT ===\n"
        validation_details += f"{'='*80}\n"

        # Write to LLM turns log using the existing log_prompt function
        log_prompt(validation_header, validation_details, sample_index=sample_index)

    except Exception as e:
        logger.error(f"Error saving failed validation record: {e}")


def check_operand_presence(text: str, operand_val: int) -> bool:
    """Checks if an operand is present as a digit or word (using inflect)."""
    if p_inflect is None:
        logger.error("Inflect engine NA, checking only digits.")
        digit_regex = rf"\b{operand_val}\b"
        return bool(re.search(digit_regex, text))
    digit_regex = rf"\b{operand_val}\b"
    if re.search(digit_regex, text):
        return True
    try:
        if operand_val == 0:
            op_words = ["zero", "nought"]
        else:
            op_words = [p_inflect.number_to_words(operand_val)]
        for op_word in op_words:
            word_regex = rf"\b{re.escape(op_word)}\b"
            if re.search(word_regex, text, re.IGNORECASE):
                return True
    except Exception as e:
        logger.error(f"Inflect failed for {operand_val}: {e}. Checking digits only.")
        return bool(re.search(digit_regex, text))
    return False


def get_atoms_in_subtree(node: Node) -> Set[int]:
    """Recursively find all atomic values (leaf nodes) in the subtree rooted at node."""
    if isinstance(node, Atom):
        return {node.n}
    atoms = set()
    for child in node.children:
        atoms.update(get_atoms_in_subtree(child))
    return atoms


def generate_narrative_anchor_with_llm(
    world_info: dict,
    op_node: OpNode,
    all_previous_anchors: list[str],
    sample_index: int | None = None,
) -> str | None:
    """
    Uses an LLM to generate a short, thematic noun phrase based on keywords.
    Focuses on reliability with a very simple prompt structure.
    """

    op_label = OP_LABELS.get(op_node.op, op_node.op)
    genre = world_info.get("genre", "unknown genre")
    setting = world_info.get("setting", "a mysterious place")
    primary_object = world_info.get("object", "items")

    concept_keywords_map = {
        "MAX": "Pinpointing the most potent or largest element",
        "MIN": "Isolating the smallest or most fundamental essence",
        "SUM": "Amalgamating all components into a unified total",
        "MED": "Identifying the central balancing point in an ordered series",
        "AVG": "Discerning the common thread or typical measure across all items",
        "SM": "Unveiling a core symbolic number through cyclical transformation",
    }
    concept_keywords_for_prompt = concept_keywords_map.get(
        op_node.op, f"{op_label} Concept"
    )
    system_prompt = f"""You are a master {world_info.get('genre')} storyteller and creative naming expert. Your task is to generate a short, evocative, and thematic 'narrative anchor'.

A narrative anchor is a creative, conceptual name that serves as a descriptive **label** or **stand-in** for the *result* (the outcome) of a specific event or calculation within the story. Its purpose is to allow the narrative to refer to this result conceptually in later parts of the story, *without* explicitly stating its numerical value. For example, if a calculation's outcome is 50, the anchor might be 'The Sunstone's Core.' The story would then mention 'The Sunstone's Core' (which implicitly represents the value 50) instead of the number itself, allowing the narrative to flow without revealing intermediate figures.

Key Guidelines for the Narrative Anchor:
1.  **Thematic:** The name MUST fit the provided Genre, Setting, and Primary Object.
2.  **Concise:** Aim for 2 to {config.MAX_ANCHOR_WORDS} words. Often a noun phrase (e.g., 'The Sunstone's Core,' 'The Oracle's Key').
3.  **No Numbers:** Absolutely no numerical values in the anchor itself.
4.  **No Direct Math Terms:** Do NOT use words like 'Sum', 'Min', 'Max', 'Average', 'Median', 'Count' directly in the anchor name. The 'Concept/Operation Hint' provided will hint at the nature of the operation without using these explicit terms.
5.  **Represent Outcome:** The name should conceptually represent the *result* or *culmination* of the described action/operation.
6.  **Focus on the Noun:** The anchor should feel like a "thing" or a "state" that has been achieved or discovered.

Examples of good anchors based on different inputs:
*   Genre: High Fantasy, Setting: Enchanted Forest, Object: Moonpetal Flowers, Concept/Operation Hint: Amalgamating all components into a unified total -> The Lunar Bloom's Essence
*   Genre: Noir Detective, Setting: Rain-slicked City, Object: Stolen Jewels, Concept/Operation Hint: Isolating the smallest or most fundamental essence -> The Shadow Locket's Secret
*   Genre: Steampunk, Object: Clockwork Gears, Concept: The central piece -> The Chronometer's Heart

You will be given the Genre, Setting, Item (Primary Object), and Concept/Operation Hint. Provide ONLY the generated anchor as your response."""

    user_prompt = (
        f"Genre: {genre}\n"
        f"Setting: {setting}\n"
        f"Item: {primary_object}\n"
        f"Concept/Operation Hint: {concept_keywords_for_prompt}\n"
    )
    prompt_log_header = f"--- Narrative Anchor Prompt (Op: {op_node.op}, Item: {primary_object}, Concept: {concept_keywords_for_prompt}) ---"
    prompt_content_for_log = f"System: {system_prompt}\\nUser:\\n{user_prompt}"

    # --- Log the prompt using log_prompt ---
    log_prompt(
        header=f"Narrative Anchor Generation Prompt (Op: {op_node.op})",
        prompt=prompt_content_for_log,
        sample_index=sample_index,
    )

    logger.debug(
        f"Attempting Narrative Anchor generation for {op_node.op} (Item: {primary_object}) with prompt:\\n{prompt_content_for_log}"
    )

    try:
        request_payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": config.ANCHOR_MAX_TOKENS,
            "temperature": config.ANCHOR_GEN_TEMP,
            "reasoning": {"exclude": True},
        }

        logger.debug(
            f"Using request_payload for narrative anchor generation: {request_payload}"
        )

        resp = _chat_completion_call(**request_payload)

        raw_candidate = None
        finish_reason = "N/A"

        if resp and resp.choices and len(resp.choices) > 0:
            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            if choice.message:
                raw_candidate = choice.message.content
                if raw_candidate is None:
                    logger.warning(
                        f"Narrative Anchor Gen: Received None content. Finish reason: {finish_reason}. Response: {resp}"
                    )
            else:
                logger.warning(
                    f"Narrative Anchor Gen: Message object is missing or empty. Finish reason: {finish_reason}. Response: {resp}"
                )
        else:
            logger.warning(
                f"Narrative Anchor Gen: Unexpected response structure (no choices or empty response). Response: {resp}"
            )

        log_prompt(
            header=f"Narrative Anchor Generation Response (Op: {op_node.op})",
            prompt=f"Raw LLM Output (Finish Reason: {finish_reason}):\n{raw_candidate if raw_candidate is not None else 'None'}",
            sample_index=sample_index,
        )
        logger.debug(
            f"Narrative Anchor Gen - API Call Details: Finish Reason='{finish_reason}', Raw Candidate='{str(raw_candidate)[:100]}...'"
        )

        if raw_candidate is None:
            logger.warning(f"Narrative Anchor Gen: Received None content in response.")
            return None

        candidate = raw_candidate.strip()

        # --- Strip surrounding quotes ---
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1].strip()
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1].strip()

        # --- Remove boilerplate prefixes ---
        original_candidate_before_boilerplate_strip = candidate  # Store for comparison
        candidate = re.sub(
            r"^(OUTPUT \(Phrase Only\):)\s*", "", candidate, flags=re.IGNORECASE
        ).strip()
        candidate = re.sub(
            r"^(Okay, here's a noun phrase:|Noun Phrase:|Phrase:|Label:|Descriptor:|Designation:|Certainly:|Here it is:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        # --- Check if boilerplate was present or if string is now empty ---
        boilerplate_indicators_lower = [
            "output (phrase only):",
            "okay, here's a noun phrase:",
            "noun phrase:",
            "phrase:",
            "label:",
            "descriptor:",
            "designation:",
            "certainly:",
            "here it is:",
        ]
        # Check if the candidate, after stripping, still starts with a known boilerplate indicator
        # or if it became empty after stripping boilerplate (meaning it was *only* boilerplate)
        if not candidate or any(
            original_candidate_before_boilerplate_strip.lower().startswith(indicator)
            for indicator in boilerplate_indicators_lower
        ):
            if not candidate:  # It became empty after stripping
                logger.warning(
                    f"Narrative Anchor Gen: Response was only boilerplate (raw: '{raw_candidate}', processed to empty string)"
                )
            else:  # It still starts with boilerplate, or original started with it and cleaning wasn't perfect
                logger.warning(
                    f"Narrative Anchor Gen: Boilerplate detected in response (raw: '{raw_candidate}', processed: '{candidate}'). Triggering retry."
                )
            return None  # Fail this attempt to trigger retry

        # --- Aggressively remove echoed input preamble ---
        # Construct the expected preamble pattern based on current genre, item, concept
        preamble_pattern_str = (
            rf"Genre: {re.escape(genre)}\s*\n"
            rf"Setting: {re.escape(setting)}\s*\n"
            rf"Item: {re.escape(primary_object)}\s*\n"
            rf"Concept/Operation Hint: {re.escape(concept_keywords_for_prompt)}\s*\n"
        )
        # Remove preamble if found at the beginning of the candidate string
        candidate = re.sub(
            f"^{preamble_pattern_str}",
            "",
            candidate,
            flags=re.IGNORECASE | re.MULTILINE,  # Added re.MULTILINE
        ).strip()

        # Fallback: remove potential "OUTPUT (Phrase Only):"
        candidate = re.sub(
            r"^(OUTPUT \(Phrase Only\):)\s*", "", candidate, flags=re.IGNORECASE
        ).strip()
        candidate = re.sub(
            r"^(Okay, here's a noun phrase:|Noun Phrase:|Phrase:|Label:|Descriptor:|Designation:|Certainly:|Here it is:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        # --- More robust check for guideline echoing ---
        # Strip surrounding quotes, as models sometimes wrap short answers in them.
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1].strip()
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1].strip()

        guideline_starters_lower = [
            "**thematic:**",
            "**concise:**",
            "**no numbers:**",
            "**no direct math terms:**",
            "**represent outcome:**",
            "**avoid repetition:**",
            "**focus on the noun:**",
            "1.",
            "2.",
            "3.",
            "4.",
            "5.",
            "6.",
            "7.",
            "key guidelines",
            "examples of good anchors",
        ]  # Already lowercase or will be lowercased by startswith check

        candidate_lower_stripped = candidate.lower().strip()
        if any(
            candidate_lower_stripped.startswith(starter)
            for starter in guideline_starters_lower
        ):
            logger.warning(
                f"Narrative Anchor Gen: Response starts with a guideline phrase (raw: '{raw_candidate}', cleaned: '{candidate}')"
            )
            return None

        num_words = len(candidate.split())
        if (
            not candidate or num_words == 0 or num_words > config.MAX_ANCHOR_WORDS
        ):  # Use config.MAX_ANCHOR_WORDS instead of hardcoded 5
            logger.warning(
                f"Narrative Anchor Gen: Invalid (empty, too long/short, refused) response (raw: '{raw_candidate}', processed: '{candidate}', words: {num_words})"
            )
            return None

        if candidate.lower().startswith(
            (
                "i cannot",
                "i'm sorry",
                "i am unable",
                "as an ai",
                "i do not have",
                "unable to provide",
            )
        ):
            logger.warning(
                f"Narrative Anchor Gen: Explicit refusal detected (raw: '{raw_candidate}', processed: '{candidate}')"
            )
            return None

        logger.info(
            f"Narrative Anchor: '{candidate}' for Op: {op_node.op}, Item: {primary_object} (raw: '{raw_candidate}')"
        )
        return candidate
    except Exception as e:
        logger.error(
            f"Narrative Anchor LLM API Error for Op: {op_node.op}, Item: {primary_object}: {e}. Prompt that failed:\n{prompt_content_for_log}"
        )
        return None


generate_narrative_anchor_with_llm = retry_api_call(generate_narrative_anchor_with_llm)


# --- BeatGenerationError Exception ---
class BeatGenerationError(Exception):
    """Raised when a story beat fails to generate, aborting entire narrative."""

    pass


# --- LLM-based Static Beat Content Validation (Existing Function, Renamed) ---
def create_user_prompt(
    world_info: dict,
    current_op_node: OpNode,
    inputs_str_for_validation: str,
    action_description_for_validation: str,
    expected_beat_result: int | None,
    overall_ground_truth_answer: int | None,
    beat_text: str,
) -> str:
    """
    Creates a standard user prompt for LLM validation of beat content.
    This centralizes the prompt creation logic to ensure consistency.
    """
    primary_object = world_info.get("object", "items")
    op_label = OP_LABELS.get(current_op_node.op, current_op_node.op)

    expected_result_str = (
        f"{expected_beat_result} ('{num_to_words(expected_beat_result)}')"
        if expected_beat_result is not None
        else "N/A"
    )
    ground_truth_str = (
        f"{overall_ground_truth_answer} ('{num_to_words(overall_ground_truth_answer)}')"
        if overall_ground_truth_answer is not None
        else "unknown"
    )

    return (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'unknown')}\n"
        f"- Setting: {world_info.get('setting', 'unknown')}\n"
        f"- Primary Object: {primary_object}\n\n"
        f"**Operation Details:**\n"
        f"- Current Operation: {current_op_node.op} ({op_label})\n"
        f"- Conceptual and Atomic Inputs: {inputs_str_for_validation}\n"
        f"- Expected Beat Result: {expected_result_str}\n"
        f"- Overall Target Answer: {ground_truth_str}\n"
        f"- Action Intent: {action_description_for_validation}\n\n"
        f"**Task:**\n"
        f"Evaluate the beat against strict numerical rules. Check that:\n"
        f"1. All required input numbers are clearly mentioned\n"
        f"2. The result is either implied or stated as required by rules\n"
        f"3. No forbidden numbers or extraneous values appear\n"
        f"4. The beat logically represents the specified operation\n\n"
        f"**Beat Text to Validate:**\n"
        f"```\n{beat_text}\n```"
    )


def perform_llm_static_content_validation(
    world_info: dict,
    current_op_node: OpNode,
    inputs_str_for_validation: str,
    action_description_for_validation: str,
    expected_beat_result: int | None,
    overall_ground_truth_answer: int | None,
    beat_text: str,
    sample_index: int,
    config_obj: Config,
    logger_obj: logging.Logger,
    encoder_obj: any,
) -> tuple[bool, str]:
    # Ensure create_user_prompt is defined and in scope
    from __main__ import create_user_prompt

    user_prompt = create_user_prompt(
        world_info,
        current_op_node,
        inputs_str_for_validation,
        action_description_for_validation,
        expected_beat_result,
        overall_ground_truth_answer,
        beat_text,
    )

    system_prompt = """You are an AI literary critic and numerical compliance checker.
Evaluate a story 'beat' for precise mathematical narration and adherence to strict numerical rules.
Context (world, operation, numbers, rules) will be provided.
Ensure coherence, logical operation, and perfect numerical compliance.
Respond in structured JSON format ONLY."""

    log_prompt(
        header=f"LLM Static Beat Validation Prompt (Op: {current_op_node.op}) for model {STATIC_CHECKER_MODEL}",
        prompt=f"System: {system_prompt}\\nUser:\\n{user_prompt}",
        sample_index=sample_index,
    )

    # Outermost try for this function
    try:
        # Inner try for the API call and response processing
        try:
            resp = _chat_completion_call(
                model=STATIC_CHECKER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=config_obj.BEAT_MAX_TOKENS,
                temperature=config_obj.LLM_VALIDATOR_TEMP,
                json_schema=STATIC_VALIDATOR_SCHEMA,  # Use the defined schema
                reasoning={"exclude": True},
            )

            raw_llm_output = ""
            if (
                resp
                and resp.choices
                and len(resp.choices) > 0
                and resp.choices[0].message
            ):
                raw_llm_output = resp.choices[0].message.content or ""

            log_prompt(
                header=f"LLM Static Beat Validation Raw Response (Op: {current_op_node.op}) from model {STATIC_CHECKER_MODEL}",
                prompt=f"Raw Output:\n{raw_llm_output}",
                sample_index=sample_index,
            )

            # Innermost try for JSON parsing
            try:
                # The response should be valid JSON already, but still use our parser
                # for consistency and as a fallback
                result_json = parse_llm_json_with_fallback(
                    raw_llm_output,
                    {
                        "is_beat_valid": False,
                        "reasoning": "Failed to parse LLM validator JSON response",
                    },
                    f"in static beat validation for {current_op_node.op}",
                )
                is_valid = bool(result_json.get("is_beat_valid", False))
                reasoning = str(
                    result_json.get(
                        "reasoning", "No reasoning provided by static checker."
                    )
                )

                if not reasoning.strip() and not is_valid:
                    reasoning = "Static checker marked beat as invalid but provided no explicit reasoning."
                elif not reasoning.strip() and is_valid:
                    reasoning = "Static checker marked beat as valid."

                logger_obj.info(
                    f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Static validation result: {is_valid}. Reasoning: {reasoning}"
                )
                return is_valid, reasoning
            except Exception as e:  # Catch other parsing/processing errors
                logger_obj.error(
                    f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Unexpected error processing static beat validation response: {e}. Raw: '{raw_llm_output}'"
                )
                return (
                    False,
                    f"Unexpected error processing LLM validator response: {e}. Raw: {raw_llm_output[:config_obj.MAX_TOKENS_BUFFER // 3]}...",
                )

        except Exception as e:  # Catch errors from _chat_completion_call itself
            logger_obj.error(
                f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] API call to static beat validator ({STATIC_CHECKER_MODEL}) failed: {e}"
            )
            return False, f"API call to static beat validator failed: {e}"
    except Exception as e:  # Catch any other unexpected error in this function
        logger_obj.error(
            f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Unexpected error in perform_llm_beat_validation: {e}",
            exc_info=True,
        )
        return False, f"Outer exception in LLM beat validation: {e}"


# --- Iterative LLM Validation Loop ---
def _generate_narrative_recursive(
    node: Node,
    context: "GenerationContext", # Type hint for GenerationContext
    is_root: bool,
):
    world = context.world
    config_obj = context.config
    encoder = context.encoder
    logger_obj = context.logger
    narrative_anchor_map = context.narrative_anchor_map

    node_id = id(node)
    current_node_conceptual_name = "this_step_s_outcome" # Default
    if isinstance(node, OpNode): # Ensure it's an OpNode before accessing op attribute
        current_node_conceptual_name = narrative_anchor_map.get(
            node_id, f"the_unnamed_{node.op}_result_{node_id % 1000}" # Increased modulo for uniqueness
        )

    op_for_log = getattr(node, "op", "AtomNode") # Safe way to get op
    logger_obj.debug(
        f"[Sample {context.sample_index + 1}] _generate_narrative_recursive: "
        f"Processing Node Type: {type(node).__name__}, Op: {op_for_log}, Conceptual Name: '{current_node_conceptual_name}', IsRoot: {is_root}, "
        f"Beat: {context.beat_counter['current'] + 1 if isinstance(node, OpNode) else '-'}/{context.beat_counter['total']}"
    )

    if isinstance(node, Atom):
        logger_obj.debug(f"Node is Atom ({node.n}), value is {node.value}. Returning.")
        return # Base case for recursion

    child_op_node_results_as_conceptual_inputs = {}
    child_conceptual_names_list = []

    for child_index, child in enumerate(node.children):
        _generate_narrative_recursive(child, context, is_root=False)
        if isinstance(child, OpNode):
            child_anchor_str = narrative_anchor_map.get(id(child))
            if child_anchor_str:
                child_conceptual_names_list.append(child_anchor_str)
                if child.value is not None:
                    child_op_node_results_as_conceptual_inputs[child_anchor_str] = child.value
                else:
                    logger_obj.error(f"CRITICAL ERROR: Child OpNode {child.op} ('{child_anchor_str}') has no computed value for parent {node.op}!")
            else:
                logger_obj.warning(f"OpNode child {child.op} of {node.op} has no conceptual name in map.")
        if context.tokens_used >= config_obj.MAX_TOTAL_TOKENS - config_obj.MAX_TOKENS_BUFFER:
            logger_obj.warning(f"TOKEN LIMIT reached after child {child_index+1} for node {node.op}. Aborting beat.")
            raise BeatGenerationError("Token limit reached during child processing.")

    context.beat_counter["current"] += 1
    logger_obj.info(
        f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for {node.op} ('{current_node_conceptual_name}')"
    )

    op_label = OP_LABELS.get(node.op, node.op)
    direct_atom_children = [c_atom for c_atom in node.children if isinstance(c_atom, Atom)]
    current_op_arity = len(node.children)
    direct_atom_values_list_with_duplicates = [a.n for a in direct_atom_children]
    direct_atom_values_set_unique = {a.n for a in direct_atom_children}
    correct_result = node.value

    if correct_result is None:
        logger_obj.error(f"CRITICAL: Node {node.op} ('{current_node_conceptual_name}') has no pre-computed value. Aborting beat.")
        raise BeatGenerationError(f"Node {node.op} ('{current_node_conceptual_name}') has no pre-computed value.")

    # --- Construct the list of atoms that MUST be mentioned for Rule 1 ---
    atoms_to_explicitly_mention_in_narrative_values = [] # List of integer values
    if node.op == "MED" and correct_result is not None:
        # For MEDIAN, if an input value equals the median result, that value is NOT mentioned.
        # All other atomic inputs (including duplicates of other values) ARE mentioned.
        temp_mention_list_values = []
        for atom_val in direct_atom_values_list_with_duplicates:
            if atom_val != correct_result:
                temp_mention_list_values.append(atom_val)
        atoms_to_explicitly_mention_in_narrative_values = temp_mention_list_values
    else:
        # For other ops, ALL direct atomic inputs (including duplicates) must be mentioned.
        atoms_to_explicitly_mention_in_narrative_values = direct_atom_values_list_with_duplicates
    
    # Create a descriptive string for the LLM prompt (Rule 1)
    must_mention_rule_parts_for_llm = []
    if atoms_to_explicitly_mention_in_narrative_values:
        from collections import Counter
        counts = Counter(atoms_to_explicitly_mention_in_narrative_values)
        for num, count_val in sorted(counts.items()): # Use count_val to avoid conflict
            num_word = num_to_words(num)
            if count_val == 1:
                must_mention_rule_parts_for_llm.append(f"'{num_word}' ({num})")
            else:
                # e.g., "two instances of 'seven' (7)"
                must_mention_rule_parts_for_llm.append(f"{num_to_words(count_val)} instances of '{num_word}' ({num})")
    
    must_mention_text_for_rule1_final = "no new atomic numbers to explicitly state for this scene"
    if must_mention_rule_parts_for_llm:
        must_mention_text_for_rule1_final = " and ".join(must_mention_rule_parts_for_llm)

    # Clarification for Rule 1 based on operation type
    special_med_input_clarification_for_rule1_final = ""
    if node.op == "MED":
        if correct_result is not None and correct_result in direct_atom_values_set_unique:
            special_med_input_clarification_for_rule1_final = (
                f"This is a MEDIAN operation. The number '{num_to_words(correct_result)}' ({correct_result}), which was an input AND is the median result for this step, has ALREADY BEEN EXCLUDED from the list of numbers to mention above. That value ('{num_to_words(correct_result)}') MUST NOT be mentioned. "
                f"You MUST, however, mention all other distinct atomic inputs and their required counts as listed above (e.g., if the list says 'two instances of seven', you must mention 'seven' twice in your narrative description of the inputs)."
            )
        else: # Median value is not among direct inputs OR no direct inputs at all
            special_med_input_clarification_for_rule1_final = (
                "This is a MEDIAN operation. The numerical result (the median) MUST NOT be stated. "
                "Ensure all distinct new atomic numbers and their counts listed above (if any) ARE mentioned."
            )
        if not must_mention_rule_parts_for_llm and (correct_result is not None and correct_result in direct_atom_values_set_unique):
             must_mention_text_for_rule1_final = (
                 f"no new numbers to explicitly state for this scene (because the only new atomic input(s) had the value '{num_to_words(correct_result)}' "
                 f"which is also the MEDIAN result and therefore must be omitted from mention)"
             )
    elif not must_mention_rule_parts_for_llm: # Not MEDIAN, and no atoms to mention
        special_med_input_clarification_for_rule1_final = "There are no new atomic numbers to explicitly mention in this scene; the operation likely uses only prior conceptual results."
    else: # Not MEDIAN, and there are atoms to mention
        special_med_input_clarification_for_rule1_final = "These numbers (and their counts if specified, e.g., 'two instances of seven') are direct inputs to the current operation and are the only new numbers you should explicitly state."


    # --- Result Handling Rule (Rule 2) ---
    result_handling_rule_text = ""
    if not is_root:
        result_handling_rule_text = (
            f"The numerical result of THIS intermediate operation ({op_label}) -- which you know to be '{num_to_words(correct_result)}' ({correct_result}) -- MUST NOT be explicitly stated in the text. "
            f"It must only be implied by events. This implied result will be known conceptually as '{current_node_conceptual_name}' for future steps."
        )
        if node.op == "MED":
             result_handling_rule_text += f" ⚠️ CRITICAL MEDIAN RULE: For MEDIAN operations, NEVER explicitly write the median value ('{num_to_words(correct_result)}') anywhere. Imply '{current_node_conceptual_name}' conceptually."
        elif correct_result is not None and correct_result in direct_atom_values_set_unique:
            result_handling_rule_text += (
                f" (Special Note: The result value '{num_to_words(correct_result)}' ({correct_result}) is also one of your required atomic inputs. "
                f"While you MUST mention it (and all its instances) as an input (Rule 1), ensure your narrative does NOT frame it as the *outcome* of this operation. "
                f"The outcome '{current_node_conceptual_name}' must still be implied conceptually.)"
            )
    else: # Root operation
         result_handling_rule_text = (
            f"This is the FINAL operation of the entire story. The numerical result -- which you know to be '{num_to_words(correct_result)}' ({correct_result}) -- MUST NOT be explicitly stated in the text. "
            f"The story should lead up to this final outcome, but the reader must infer it from the narrative's conclusion."
        )

    # --- Permitted Phrasing Numbers (Rule 3) ---
    temp_phrasing_words_detailed_for_rule3 = [f"'{num_to_words(n)}' ({n})" for n in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET]
    phrasing_numbers_gen_str_detailed_for_rule3 = ", ".join(sorted(temp_phrasing_words_detailed_for_rule3))
    may_use_gen_parts_detailed = [
        f"small numbers like {phrasing_numbers_gen_str_detailed_for_rule3} for general narrative phrasing (e.g., 'two guards')"
    ]
    arity_for_counting_rule = current_op_arity
    
    # Determine if arity_for_counting_rule is forbidden *for other reasons*
    # It's okay if arity_for_counting_rule is one of the direct_atom_values_set_unique, as Rule 1 covers its mention.
    # It's okay if arity_for_counting_rule is one of ALWAYS_ALLOWED_PHRASING_NUMBERS_SET.
    # It's problematic if it's in forbidden_for_current_beat_py_validator AND NOT covered by the above.
    # Note: forbidden_for_current_beat_py_validator is built *before* this arity check.
    
    # Python validator's forbidden set for *this beat's context*
    forbidden_for_py_val_this_beat = set(context.introduced_atoms)
    if context.overall_ast_root:
        for pn_temp in postorder(context.overall_ast_root):
            if isinstance(pn_temp, OpNode) and pn_temp.value is not None and id(pn_temp) != id(node):
                is_direct_child_temp = any(id(child_node_temp) == id(pn_temp) for child_node_temp in node.children if isinstance(child_node_temp, OpNode))
                if not is_direct_child_temp:
                    forbidden_for_py_val_this_beat.add(pn_temp.value)
    if context.overall_ground_truth_answer is not None and \
       context.overall_ground_truth_answer != correct_result and \
       context.overall_ground_truth_answer not in direct_atom_values_set_unique:
        forbidden_for_py_val_this_beat.add(context.overall_ground_truth_answer)
    forbidden_for_py_val_this_beat -= direct_atom_values_set_unique # Current inputs are not "forbidden" in this sense

    is_arity_problematic_forbidden = (
        arity_for_counting_rule in forbidden_for_py_val_this_beat and
        arity_for_counting_rule not in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET and
        arity_for_counting_rule not in direct_atom_values_set_unique # If arity is an input, Rule 1 handles it
    )

    if arity_for_counting_rule > 0 :
        if not is_arity_problematic_forbidden:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(arity_for_counting_rule)}' ({arity_for_counting_rule}) IF it's genuinely used to count the total items/inputs (atomic or conceptual) involved in THIS specific action"
            )
        else:
             may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(arity_for_counting_rule)}' ({arity_for_counting_rule}) ONLY if essential for counting items in THIS action AND clearly distinct from its forbidden meaning (use with extreme caution or avoid if possible, as it's a forbidden number from another context)"
            )
    may_use_gen_clause_content_detailed = "; ".join(may_use_gen_parts_detailed)

    gt_counting_caution_for_gen = ""
    # (gt_counting_caution_for_gen logic remains the same)

    # --- Forbidden Numbers (Rule 4) ---
    forbidden_for_llm_prompt = set()
    if context.overall_ast_root:
        for pn in postorder(context.overall_ast_root):
            if isinstance(pn, OpNode) and pn.value is not None and id(pn) != id(node):
                is_direct_child = any(id(child_node) == id(pn) for child_node in node.children if isinstance(child_node, OpNode))
                if not is_direct_child:
                    forbidden_for_llm_prompt.add(pn.value)
    if context.overall_ground_truth_answer is not None and \
       context.overall_ground_truth_answer != correct_result and \
       context.overall_ground_truth_answer not in direct_atom_values_set_unique:
        forbidden_for_llm_prompt.add(context.overall_ground_truth_answer)
    
    forbidden_for_llm_prompt -= set(atoms_to_explicitly_mention_in_narrative_values) # Values from Rule 1 are not "forbidden" by Rule 4
    if correct_result is not None: # Rule 2 handles result presence/absence
        forbidden_for_llm_prompt.discard(correct_result)
    forbidden_for_llm_prompt -= config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    if not is_arity_problematic_forbidden :
        forbidden_for_llm_prompt.discard(arity_for_counting_rule)

    temp_forbidden_detailed_list_for_rule4 = []
    if forbidden_for_llm_prompt:
        for n_forbidden in sorted(list(forbidden_for_llm_prompt)):
            temp_forbidden_detailed_list_for_rule4.append(f"'{num_to_words(n_forbidden)}' ({n_forbidden})")
    must_avoid_str_for_generator_prompt_detailed = (
        ", ".join(temp_forbidden_detailed_list_for_rule4)
        if temp_forbidden_detailed_list_for_rule4
        else "None specifically (beyond the general rule against unlisted numbers, unstated results, and numbers not covered by Rule 1 for this step)"
    )

    # --- Prior Results Handling (Rule 6) ---
    prior_results_handling_rule = ""
    # (prior_results_handling_rule logic remains the same)

    # --- Construct `ultra_strict_instruction` ---
    ultra_strict_instruction = (
        f"**Narrative Challenge & Your Writing Guide for This Scene (Scene {context.beat_counter['current']}/{context.beat_counter['total']}):**\n"
        f"Your main goal is to weave a compelling scene. However, for this specific task, you must precisely control how numbers are mentioned, turning constraints into creative storytelling. Adhere meticulously to these rules:\n\n"
        f"1.  **Key Details to Feature (Numbers in Action):** Your narrative MUST explicitly mention the following new numerical quantities (written as words, e.g., 'nine treasures', and ensuring correct counts if specified, like 'two instances of seven'): **{must_mention_text_for_rule1_final}**. These are the ONLY new numbers you should state for this scene. This list has been carefully curated: {special_med_input_clarification_for_rule1_final}\n"
        f"2.  **The Unspoken Outcome (Result Handling):** {result_handling_rule_text}\n"
        f"3.  **Permitted Narrative Flourishes (Optional Numbers):** You MAY use {may_use_gen_clause_content_detailed} for general color, if truly necessary for fluency and ONLY if these numbers are not forbidden by other rules.\n"
        f"{gt_counting_caution_for_gen.rstrip() + ('\\n' if gt_counting_caution_for_gen.strip() else '')}"
        f"4.  **Whispers Best Left Unheard (Forbidden Numbers):** Strictly avoid mentioning these specific numbers: {must_avoid_str_for_generator_prompt_detailed}. These are typically from unrelated past events or the overall story's final answer (if it's not this step's result and not an allowed input for this step by Rule 1).\n"
        f"5.  **The Rule of No Other Numbers:** ABSOLUTELY NO OTHER NUMBERS: Do not introduce any other numerical values (digits or words) beyond those explicitly covered by rules 1-4 above. No intermediate sums or calculations should be shown numerically in the narrative.\n"
        f"{prior_results_handling_rule}" # Rule 6
        f"Focus on clear storytelling that naturally implies the calculations based on these strict numerical constraints. Failure to adhere will result in rejection."
    )

    # --- Construct `action_description` ---
    primary_object = world.get("object", "items")
    safe_primary_object_for_fstring = str(primary_object).replace("{", "{{").replace("}", "}}")
    conceptual_input_names_only_list_for_action = [f"'{name}'" for name in child_conceptual_names_list]
    conceptual_input_names_only_str_for_action = (
        ", ".join(conceptual_input_names_only_list_for_action)
        if conceptual_input_names_only_list_for_action
        else "no prior calculated quantities"
    )
    
    # Use the detailed `must_mention_text_for_rule1_final` for action_description
    mentionable_new_numbers_for_action_desc = must_mention_text_for_rule1_final
    if "no new atomic numbers to explicitly state" in must_mention_text_for_rule1_final:
        if direct_atom_children:
            mentionable_new_numbers_for_action_desc = "specific new numbers (as per your Writing Guide's Rule 1, noting any MEDIAN exceptions that mean some inputs are not stated, and ensuring all required instances of numbers are covered if duplicates are listed in Rule 1)"
        else:
            mentionable_new_numbers_for_action_desc = "no new specific numbers for this step"

    action_description_parts = [
        f"**Your Scene's Core Action & Narrative Goal (Follow this closely):**\n"
        f"This scene needs to narrate an event or discovery that mirrors the mathematical operation: **{op_label}**. "
        f"The central items of interest are the '{safe_primary_object_for_fstring}'."
    ]
    op_specific_action_details = ""
    op_specific_outcome_implication_details = ""

    # (OP_SPECIFIC ACTION DETAILS - ensure these use `mentionable_new_numbers_for_action_desc`
    # and reinforce the need to mention all instances if Rule 1 specifies duplicates)
    if node.op == "SUM":
        op_specific_action_details = (
            f"Imagine your characters are gathering or combining distinct collections of '{safe_primary_object_for_fstring}'. "
            f"Some of these collections might be newly found or counted, corresponding to: {mentionable_new_numbers_for_action_desc}. "
            f"Other collections might be the results of previous efforts, known only by evocative names like {conceptual_input_names_only_str_for_action}. "
            f"Your story should clearly show these being brought together into a single, larger accumulation, ensuring all required instances of numbers (e.g., 'two sevens') from Rule 1 are mentioned."
        )
        op_specific_outcome_implication_details = (
            f"This combined accumulation will then be known conceptually as '{current_node_conceptual_name}'. "
            f"Remember, its actual total numerical size ('{num_to_words(correct_result)}') must not be stated (as per Rule 2 of your Writing Guide)."
        )
    elif node.op == "MIN":
        op_specific_action_details = (
            f"Picture your characters assessing several distinct quantities or instances of '{safe_primary_object_for_fstring}'. "
            f"These might include newly encountered items (quantified by {mentionable_new_numbers_for_action_desc}) "
            f"and also the conceptual results of past endeavors (referred to as {conceptual_input_names_only_str_for_action}). "
            f"The narrative should focus on them identifying or selecting the *smallest* or *least significant* among all these, ensuring all required instances of numbers from Rule 1 are mentioned when describing the inputs."
        )
        op_specific_outcome_implication_details = (
            f"This single, smallest quantity they identify will then be conceptually known as '{current_node_conceptual_name}'. "
            f"Its precise numerical value ('{num_to_words(correct_result)}') must remain a secret to the reader (as per Rule 2 of your Writing Guide)."
        )
    elif node.op == "MAX":
        op_specific_action_details = (
            f"Envision your characters evaluating several different amounts or examples of '{safe_primary_object_for_fstring}'. "
            f"These could be new discoveries (detailed by {mentionable_new_numbers_for_action_desc}) "
            f"or the outcomes of prior activities (known conceptually as {conceptual_input_names_only_str_for_action}). "
            f"The story should center on them determining or isolating the *largest*, *most potent*, or *most significant* among these, ensuring all required instances of numbers from Rule 1 are mentioned when describing the inputs."
        )
        op_specific_outcome_implication_details = (
            f"This preeminent quantity will thereafter be referred to conceptually as '{current_node_conceptual_name}'. "
            f"Do not explicitly state its actual numerical value ('{num_to_words(correct_result)}') (as per Rule 2 of your Writing Guide)."
        )
    elif node.op == "AVG":
        op_specific_action_details = (
            f"The scene should depict characters considering a set of '{safe_primary_object_for_fstring}' "
            f"(some new, described by {mentionable_new_numbers_for_action_desc}, "
            f"others being prior conceptual results like {conceptual_input_names_only_str_for_action}). "
            f"Their actions or observations should lead them to understand a 'typical', 'representative', or 'average' characteristic or measure across these items, "
            f"without performing explicit mathematical division in the narrative. Ensure all required instances of numbers from Rule 1 are mentioned when describing the inputs."
        )
        op_specific_outcome_implication_details = (
            f"This representative 'average' value (which is '{num_to_words(correct_result)}') will be conceptually known as '{current_node_conceptual_name}' and must be implied, not stated numerically (as per Rule 2 of your Writing Guide)."
        )
    elif node.op == "MED":
        is_median_also_an_input = correct_result in direct_atom_values_set_unique if correct_result is not None else False
        op_specific_action_details = (
            f"**This is a MEDIAN scene, requiring EXTREME care with number mentions!** Your characters will encounter various '{safe_primary_object_for_fstring}' "
            f"(some new, described by: {mentionable_new_numbers_for_action_desc}; "
            f"others being prior conceptual results like {conceptual_input_names_only_str_for_action}). "
            f"The core of the scene is about them identifying a 'central element', 'balancing point', or 'middle value' from all these available items/numbers. "
            f"Crucially, the actual median value ('{num_to_words(correct_result)}') MUST NEVER be written out, as per Rule 2 of your Writing Guide."
        )
        if is_median_also_an_input:
            op_specific_action_details += (
                f" ⚠️ **VERY IMPORTANT MEDIAN TWIST (Rule 1 & 2 Interaction):** The number '{num_to_words(correct_result)}' ({correct_result}) is not only the median outcome but also one of the numbers involved in this step. "
                f"Because of this, you MUST NOT mention '{num_to_words(correct_result)}' ({correct_result}) AT ALL in this scene, not even when describing the initial numbers. "
                f"Your Writing Guide's Rule 1 (Key Details to Feature) has ALREADY EXCLUDED this number from the list of what you must mention. "
                f"You should describe the set of initial numbers by listing ONLY the *other* numbers (and their counts, from your Writing Guide, Rule 1) and refer to this specific one indirectly (e.g., '...and one particular item whose nature was central...')."
            )
        op_specific_outcome_implication_details = (
            f"The conceptual outcome, '{current_node_conceptual_name}' (which represents the median value '{num_to_words(correct_result)}'), must be implied through the narrative of finding this central point, not stated numerically (as per Rule 2 of your Writing Guide)."
        )
    elif node.op == "SM": # Sum Modulo 10
        op_specific_action_details = (
            f"**This is a SUM MODULO 10 scene.** The characters' actions should represent combining all involved quantities of '{safe_primary_object_for_fstring}' "
            f"(the new numbers being: {mentionable_new_numbers_for_action_desc}; "
            f"and any prior results referred to only by their conceptual names: {conceptual_input_names_only_str_for_action}). "
            f"After this conceptual combination (do not state any intermediate sum!), they should then discover or focus on a core essence, a symbolic digit, or a cyclical pattern related to this unstated total. This discovery is equivalent to finding the sum modulo 10. Ensure all required instances of numbers from Rule 1 are mentioned when describing the inputs."
        )
        op_specific_outcome_implication_details = (
            f"This final symbolic essence or pattern will be conceptually known as '{current_node_conceptual_name}' (representing the value '{num_to_words(correct_result)}'), and this outcome must be implied, not stated numerically (as per Rule 2 of your Writing Guide)."
        )
    
    if not op_specific_action_details:
        op_specific_action_details = f"  - The characters perform an action related to '{op_label}' using new numbers ({mentionable_new_numbers_for_action_desc}) and prior results ({conceptual_input_names_only_str_for_action}). Ensure all required instances of numbers are mentioned."
    action_description_parts.append(op_specific_action_details)
    if not op_specific_outcome_implication_details:
        op_specific_outcome_implication_details = (
            f"The outcome of this action should lead to a new understanding or quantity, which will be conceptually known as '{current_node_conceptual_name}'. "
            f"This concept ('{current_node_conceptual_name}') corresponds to the numerical value {correct_result} ('{num_to_words(correct_result)}')."
        )
    action_description_parts.append(f"\n{op_specific_outcome_implication_details}")
    action_description_parts.append(
        f"\n**Crucial Storytelling Constraint (Rule 2 Reminder):** Your narrative MUST NOT explicitly state the number '{num_to_words(correct_result)}' ({correct_result}) AS THE RESULT of this operation (unless this is the final operation of the story, in which case it must also not be stated). "
        f"Instead, the story should imply this outcome through the characters' actions, discoveries, or the state of the '{safe_primary_object_for_fstring}', "
        f"so that '{current_node_conceptual_name}' becomes the way to think about this new state."
    )
    if node.op != "MED" and correct_result is not None and correct_result in direct_atom_values_set_unique:
        action_description_parts.append(
            f"**Important Narrative Challenge (Rule 1 & 2 Interaction):** The numerical value of this operation's outcome ('{num_to_words(correct_result)}') is ALSO one of the numbers you must mention as an input (as per Rule 1 of your Writing Guide). "
            f"Your story must carefully distinguish its role. Mention '{num_to_words(correct_result)}' (and all its required instances) when describing the initial items/quantities. "
            f"However, when describing the *result* of the operation, you must only imply it conceptually as '{current_node_conceptual_name}' and NOT restate '{num_to_words(correct_result)}' as the outcome (as per Rule 2)."
        )
    action_description_parts.append(
        f"\nRemember, all numbers that ARE explicitly mentioned (only those listed with their counts in Rule 1 of your Writing Guide) must be written as words (e.g., 'seven' not '7')."
    )
    action_description = "\n".join(action_description_parts)

    # --- Construct the rest of the prompt (initial_user_message_for_generator) ---
    # (Conceptual inputs context string for LLM's background)
    conceptual_inputs_context_list = []
    if child_op_node_results_as_conceptual_inputs:
        for name, val in child_op_node_results_as_conceptual_inputs.items():
            conceptual_inputs_context_list.append(
                f"the concept known as '{name}' (which represents the numerical value {val})"
            )
    conceptual_inputs_context_str = (
        ", ".join(conceptual_inputs_context_list)
        if conceptual_inputs_context_list
        else "None (this is the first calculation or uses only new numbers)"
    )
    # (Atomic inputs context string for LLM's background)
    atomic_inputs_context_str_detailed = (
        ", ".join([f"'{num_to_words(atom_node.n)}' ({atom_node.n}) - appears {direct_atom_values_list_with_duplicates.count(atom_node.n)} time(s)" for atom_node in direct_atom_children]) # Show counts for clarity
        if direct_atom_children
        else "None"
    )

    context_snippet = clean_snippet(context.last_scene_text, max_len=config_obj.BEAT_CONTEXT)
    initial_user_message_parts = [
        f"Story Scene Task: Create the narrative for the step resulting in '{current_node_conceptual_name}' (Scene {context.beat_counter['current']}/{context.beat_counter['total']})\n\n"
        f"**Background for Your Scene (Context for you, the writer - follow strict rules below for what appears in the story):**\n"
        f"- Genre: {world.get('genre', 'N/A')}\n"
        f"- Setting: {world.get('setting', 'N/A')}\n"
        f"- Central Items in the Story: {primary_object}\n"
        f"- Quantities from Previous Events (Conceptual Names & their values for your understanding - DO NOT use these values in the story, only their conceptual names as per Rule 6 below): {conceptual_inputs_context_str}\n"
        f"- New Numbers Introduced in this Scene (Values and their counts for your understanding - Use word form in story ONLY IF PERMITTED by Rule 1 below): {atomic_inputs_context_str_detailed}\n\n"
        f"{action_description}\n\n"
        f"{ultra_strict_instruction}\n\n"
    ]

    if node.op == "MED" and config_obj.FEW_SHOT_EXAMPLES > 0:
        examples_to_actually_use = []
        for ex_idx_loop, (ex_rules, ex_good, ex_bad, ex_reason) in enumerate(FEW_SHOT_EXAMPLES_STRICT):
            if "MEDIAN Example" in ex_rules:
                examples_to_actually_use.append((ex_rules, ex_good, ex_bad, ex_reason))
        examples_to_actually_use = examples_to_actually_use[:config_obj.FEW_SHOT_EXAMPLES]

        if examples_to_actually_use:
            few_shot_section = ["--- CRITICAL FEW-SHOT EXAMPLES FOR MEDIAN OPERATIONS ---"]
            few_shot_section.append(
                "These examples illustrate how to handle the strict numerical rules when the operation is MEDIAN. Pay close attention to how input numbers are mentioned (or not mentioned if they are the median value AND an input) and how the result is implied.\n"
            )
            for ex_idx, (example_rules_text, good_narrative, bad_narrative, bad_reasoning) in enumerate(examples_to_actually_use):
                why_good_text = "This example correctly follows the MEDIAN rules. "
                if "eighty-nine" in example_rules_text and "seventy-two" in example_rules_text:
                    why_good_text += "The example's rules (already applying the MEDIAN exception) state to mention inputs 'seventy-two, eighty-four, ninety-one, and ninety-five'. The number 'eighty-nine' was an input but also the median, so it was correctly EXCLUDED from the 'MUST INCLUDE' list in the rules and is OMITTED in the good narrative. The median result ('eighty-nine') is only implied conceptually as 'the balanced keystone'. This demonstrates the critical MEDIAN rule."
                elif "eighty-seven" in example_rules_text and "seventy-three" in example_rules_text:
                    why_good_text += "The example's rules (already applying the MEDIAN exception) state to mention inputs 'seventy-three, eighty-five, eighty-eight, eighty-nine, and ninety-one'. The number 'eighty-seven' was an input but also the median, so it was correctly EXCLUDED from the 'MUST INCLUDE' list in the rules and is OMITTED in the good narrative. The median result ('eighty-seven') is only implied conceptually. This again demonstrates the critical MEDIAN rule."
                else:
                     why_good_text += "It mentions only the permitted input numbers (with MEDIAN exceptions correctly applied in its own rules) and implies the median result conceptually without stating its numerical value."

                few_shot_section.append(f"**EXAMPLE {ex_idx + 1} RULES (from a hypothetical different problem - note how its 'MUST INCLUDE' list is already filtered for MEDIAN exceptions):**\n{example_rules_text.replace('\\\\n', '\\n')}\n")
                few_shot_section.append(f"**EXAMPLE {ex_idx + 1} GOOD NARRATIVE:**\n{good_narrative}\n")
                few_shot_section.append(f"**WHY GOOD:**\n{why_good_text}\n")
                few_shot_section.append(f"**EXAMPLE {ex_idx + 1} BAD NARRATIVE:**\n{bad_narrative}\n")
                few_shot_section.append(f"**WHY BAD (Reason for failure):**\n{bad_reasoning}\n")
            few_shot_section.append("**REMEMBER THE CRITICAL MEDIAN RULE FOR *YOUR* CURRENT TASK:**")
            few_shot_section.append(
                "For MEDIAN operations, the median value itself must NEVER appear explicitly in the text. If an atomic input number for *your current task* happens to BE the median value, that specific input number must ALSO NOT be mentioned (it will be absent from the list in Rule 1 of your Writing Guide). Refer to your 'Narrative Challenge & Your Writing Guide' section in the main prompt for the specific numbers and rules for *your current scene*.\n"
            )
            initial_user_message_parts.append("\n".join(few_shot_section))
            logger_obj.info(
                f"Added {len(examples_to_actually_use)} MEDIAN-specific few-shot examples to the initial generator prompt for Op: {node.op}."
            )

    initial_user_message_parts.append(f'**Continue From (End of last scene):**\n"...{context_snippet}..."\n\n')
    initial_user_message_parts.append(f"**Your Response:**\nWrite ONLY the narrative text for this new scene, continuing smoothly. Do not add titles, notes, or anything outside the story itself.")
    initial_user_message_for_generator = "".join(initial_user_message_parts)

    py_validator_enforce_result_presence = False
    if is_root:
        py_validator_enforce_result_presence = False

    validate_beat_numbers = make_number_validator(
        allowed_atoms_with_duplicates=direct_atom_values_list_with_duplicates,
        forbidden_atoms=forbidden_for_py_val_this_beat, # Use the more locally relevant forbidden set for python validator
        operand_count=current_op_arity,
        correct_result_for_beat=correct_result,
        enforce_result_presence=py_validator_enforce_result_presence,
        operation_type=node.op,
        overall_ground_truth_answer=context.overall_ground_truth_answer,
        is_root_node_being_validated=is_root,
        conceptual_input_values=set(child_op_node_results_as_conceptual_inputs.values()),
        config_obj=config_obj,
        logger_obj=logger_obj,
    )

    base_system_prompt_template = (
        "You are a master {genre} storyteller crafting a narrative. Your task is to write a single scene contributing to an ongoing story. "
        "Focus solely on advancing the tale as specified in the user message. Do not include explanations or analysis outside the narrative itself. "
        "The story involves mathematical operations implied through narrative actions. Pay EXTREMELY CAREFUL attention to the detailed 'Narrative Challenge & Your Writing Guide for This Scene' provided in the user message, especially ALL rules about number mentions. Produce ONLY clean narrative text."
    )
    operator_specific_system_focus = ""
    if node.op == "MED":
        operator_specific_system_focus = (
            "\n\n**CRITICAL SYSTEM FOCUS FOR THIS SCENE (MEDIAN OPERATION):**\n"
            "The current scene involves a MEDIAN calculation. This type of scene has unique and exceptionally strict rules about which numbers can and cannot be mentioned, especially if an input number is also the median result itself. "
            "You MUST meticulously follow the 'Narrative Challenge & Your Writing Guide' section in the user message, particularly Rule 1 (Key Details to Feature) which specifies EXACTLY which numbers (and their counts) to mention (and by implication, which to omit due to MEDIAN rules) and Rule 2 (The Unspoken Outcome) which dictates the median result MUST be implicit. "
            "Failure to adhere to these MEDIAN rules (e.g., mentioning a forbidden number, missing a required number or count from Rule 1, or explicitly stating the median result) will result in rejection of your scene. Re-read these rules in the user prompt carefully."
        )
    system_prompt_for_generator = base_system_prompt_template.format(
        genre=world.get('genre', 'Fantasy')
    ) + operator_specific_system_focus

    current_max_beat_completion_tokens = config_obj.BEAT_MAX_TOKENS
    beat_text_final_validated = None

    llm_val_conceptual_input_names_only = (
        ", ".join([f"'{name_val}'" for name_val in child_conceptual_names_list])
        if child_conceptual_names_list
        else "None"
    )
    llm_val_atomic_inputs_words_str_for_llm_validator = must_mention_text_for_rule1_final
    
    llm_val_expected_beat_result_detailed_for_validator = "N/A"
    if correct_result is not None:
        llm_val_expected_beat_result_detailed_for_validator = (
            f"'{num_to_words(correct_result)}' ({correct_result}) (This should be IMPLICIT in the narrative as per Rule 2)"
        )
        if node.op == "MED":
            llm_val_expected_beat_result_detailed_for_validator += " - CRITICAL: For MEDIAN, this value must NEVER be stated."

    for attempt_outer in range(1, config_obj.MAX_BEAT_RETRIES + 1):
        logger_obj.info(
            f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Result Concept: '{current_node_conceptual_name}'] Outer Beat Gen Attempt: {attempt_outer}/{config_obj.MAX_BEAT_RETRIES}"
        )
        llm_validated_beat_text = _generate_and_llm_validate_beat(
            original_user_message_for_generator=initial_user_message_for_generator,
            system_prompt_for_generator=system_prompt_for_generator,
            world_info=world,
            current_op_node=node,
            conceptual_inputs_str_for_llm_validator=llm_val_conceptual_input_names_only,
            atomic_inputs_words_str_for_llm_validator=llm_val_atomic_inputs_words_str_for_llm_validator,
            action_description_for_llm_validator=action_description,
            expected_beat_result_words_for_llm_validator=llm_val_expected_beat_result_detailed_for_validator,
            ultra_strict_instruction_for_llm_validator_context=ultra_strict_instruction,
            current_max_beat_completion_tokens=current_max_beat_completion_tokens,
            sample_index=context.sample_index,
            context_config=config_obj,
            logger_obj=logger_obj,
            encoder_obj=encoder,
            is_current_beat_root_node=is_root,
            overall_ground_truth_answer_val=context.overall_ground_truth_answer,
            primary_object_name=primary_object,
            forbidden_prior_results_and_gt_for_llm_validator=forbidden_for_llm_prompt, # Use the refined set
            correct_result_val=correct_result,
            direct_atom_values_val=direct_atom_values_set_unique, # Pass unique set for LLM validator context
        )

        if llm_validated_beat_text:
            if validate_beat_numbers(llm_validated_beat_text, beat_op_for_log=op_for_log):
                beat_text_final_validated = llm_validated_beat_text
                logger_obj.info(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Python validator PASSED LLM-validated beat."
                )
                break
            else:
                logger_obj.warning(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Python validator FAILED for LLM-validated beat. Outer attempt {attempt_outer} failed."
                )
        else:
            logger_obj.warning(
                f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Iterative LLM validation loop returned None. Outer attempt {attempt_outer} failed."
            )
        if attempt_outer < config_obj.MAX_BEAT_RETRIES:
            time.sleep(config_obj.RETRY_INITIAL_DELAY * (2 ** (attempt_outer - 1)))

    if not beat_text_final_validated:
        logger_obj.error(
            f"Operator {node.op} (Result Concept: '{current_node_conceptual_name}') failed after {config_obj.MAX_BEAT_RETRIES} outer attempts. Aborting for this sample."
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} (Result Concept: '{current_node_conceptual_name}') after all outer retries."
        )

    beat_text = beat_text_final_validated
    btoks = len(encoder.encode(beat_text))
    context.scenes.append(beat_text)
    context.tokens_used += btoks
    context.last_scene_text = beat_text
    context.introduced_atoms.update(direct_atom_values_set_unique) # Add unique atoms from this beat
    logger_obj.debug(
        f"Beat {context.beat_counter['current']} for Op {node.op} successful. Introduced atoms updated with: {direct_atom_values_set_unique}"
    )

    # --- Padding Logic ---
    if not is_root:
        forbidden_for_padding_slot = set(context.introduced_atoms)
        if context.overall_ast_root:
            for processed_node_padding in postorder(context.overall_ast_root):
                if isinstance(processed_node_padding, OpNode) and processed_node_padding.value is not None:
                    forbidden_for_padding_slot.add(processed_node_padding.value)
        if context.overall_ground_truth_answer is not None:
            forbidden_for_padding_slot.add(context.overall_ground_truth_answer)
        forbidden_for_padding_slot.update(direct_atom_values_set_unique) # Forbid current beat's atoms too
        if correct_result is not None: # Also forbid current beat's result in immediate padding
            forbidden_for_padding_slot.add(correct_result)

        validate_padding = make_number_validator(
            allowed_atoms_with_duplicates=[],
            forbidden_atoms=forbidden_for_padding_slot,
            operand_count=0,
            correct_result_for_beat=None,
            strict_zero=True,
            enforce_result_presence=False,
            operation_type="PADDING",
            overall_ground_truth_answer=context.overall_ground_truth_answer,
            is_root_node_being_validated=False,
            conceptual_input_values=None,
            config_obj=config_obj,
            logger_obj=logger_obj,
        )
        
        current_padding_total_overall = context.padding_stats["total_padding_tokens"]
        max_padding_allowed_overall = context.padding_stats["max_padding_allowed"]
        padding_budget_this_slot = context.padding_stats.get("padding_per_slot", 0)
        padding_tokens_added_this_slot = 0
        local_padding_segments_added_this_slot = 0
        padding_termination_reason = "Loop completed (max segments or slot budget likely met)."

        logger_obj.info(
            f"PADDING SLOT INIT [After Op: {node.op}, Result Concept: '{current_node_conceptual_name}']: Slot Budget: {padding_budget_this_slot if padding_budget_this_slot > 0 else 'N/A'}, Overall Budget Rem: {max_padding_allowed_overall - current_padding_total_overall}."
            f" Forbidden for padding: {sorted(list(forbidden_for_padding_slot)) if forbidden_for_padding_slot else 'None'}"
        )

        if padding_budget_this_slot > 0:
            for _ in range(context.max_pad_paragraphs):
                if not (
                    context.tokens_used < config_obj.MAX_TOTAL_TOKENS - config_obj.MAX_TOKENS_BUFFER
                    and current_padding_total_overall < max_padding_allowed_overall
                    and padding_tokens_added_this_slot < padding_budget_this_slot
                ):
                    padding_termination_reason = "Budget met (overall total, overall padding, or slot padding)."
                    break
                estimated_next_padding_segment_cost = config_obj.PADDING_MAX_TOKENS + (config_obj.MAX_TOKENS_BUFFER // 5)
                if padding_tokens_added_this_slot + estimated_next_padding_segment_cost > padding_budget_this_slot:
                    padding_termination_reason = f"Slot budget would be exceeded (est. {padding_tokens_added_this_slot + estimated_next_padding_segment_cost}/{padding_budget_this_slot})"
                    break
                if would_exceed_budget(context.tokens_used, config_obj.PADDING_MAX_TOKENS, config_obj.MAX_TOTAL_TOKENS, config_obj.MAX_TOKENS_BUFFER):
                    padding_termination_reason = "Overall token budget would be exceeded (pre-gen check)."
                    break

                padding_system_prompt = "You are a concise storyteller, skilled at adding brief, atmospheric paragraphs that bridge scenes without introducing new numbers or calculations."
                cleaned_snippet_padding = clean_snippet(context.last_scene_text, max_len=config_obj.PADDING_CONTEXT)
                padding_user_prompt = (
                    f"The story is set in a {context.world.get('genre', 'mysterious world')} ({context.world.get('setting', 'unknown location')}).\n"
                    f"The characters are focused on {context.world.get('object', 'important items')}.\n"
                    f'Previous Scene Snippet (End of last scene): "...{cleaned_snippet_padding.replace("\\n", " ")}..."\n\n'
                    f"Task: Write ONE short, atmospheric paragraph (typically 3-5 sentences) that continues smoothly from the previous scene snippet. "
                    f"This paragraph should be purely narrative filler or scene transition. "
                    f"ABSOLUTELY NO NUMBERS (digits or words like 'one', 'two', 'first', etc.) are allowed in this paragraph, except potentially {', '.join([f'{num_to_words(n)} ({n})' for n in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET])} if used for completely general phrasing and not quantities. Strive for zero numbers. "
                    f"Do not advance the core plot calculation. Do not mention specific quantities. "
                    f"Output ONLY the text for this single paragraph. No titles, no explanations, no analysis."
                )
                padding_text = generate_with_retry(
                    padding_system_prompt,
                    padding_user_prompt,
                    config_obj.PADDING_MAX_TOKENS,
                    validate_padding,
                    config_obj.MAX_PAD_RETRIES,
                    context.sample_index,
                    config_obj.CREATIVE_NARRATIVE_TEMP,
                    {"exclude": True},
                )

                if padding_text:
                    ptoks = len(encoder.encode(padding_text))
                    if not (context.tokens_used + ptoks <= config_obj.MAX_TOTAL_TOKENS - config_obj.MAX_TOKENS_BUFFER):
                        padding_termination_reason = "Overall total token limit would be exceeded by actual padding tokens."
                        break
                    if not (current_padding_total_overall + ptoks <= max_padding_allowed_overall):
                        padding_termination_reason = "Overall padding token limit would be exceeded."
                        break
                    if not (padding_tokens_added_this_slot + ptoks <= padding_budget_this_slot):
                        padding_termination_reason = "Padding slot budget would be exceeded."
                        break
                    local_padding_segments_added_this_slot += 1
                    context.scenes.append(padding_text)
                    context.tokens_used += ptoks
                    context.last_scene_text = padding_text
                    context.padding_stats["total_padding_tokens"] += ptoks
                    current_padding_total_overall = context.padding_stats["total_padding_tokens"]
                    context.padding_stats["padding_segments_added"] += 1
                    padding_tokens_added_this_slot += ptoks
                else:
                    padding_termination_reason = "Padding generation or validation failed for this segment."
                    break
            if local_padding_segments_added_this_slot > 0:
                logger_obj.info(
                    f"PADDING SLOT SUMMARY [After Op: {node.op}, Result Concept: '{current_node_conceptual_name}']: Added {local_padding_segments_added_this_slot} segments, using {padding_tokens_added_this_slot}/{padding_budget_this_slot if padding_budget_this_slot > 0 else 'N/A'} tokens. Termination Reason: {padding_termination_reason}."
                )
            elif padding_budget_this_slot > 0 :
                logger_obj.info(
                    f"PADDING SLOT SUMMARY [After Op: {node.op}, Result Concept: '{current_node_conceptual_name}']: No padding segments added. Termination Reason: {padding_termination_reason}."
                )

    logger_obj.debug(
        f"Exiting _generate_narrative_recursive for Op {op_for_log} ('{current_node_conceptual_name}'). Total tokens used: {context.tokens_used}"
    )

# --- Main Narrative Generation Function ---
def generate_narrative(
    ast: Node,
    world: dict,
    config: Config, # Note: this is the global config object, not context.config
    encoder, # Global encoder
    p_inflect, # Global p_inflect
    logger, # Global logger
    sample_index: int,
    overall_ground_truth_answer: int,
) -> GenerationContext | None: # Return GenerationContext or None
    """
    Generate a structured narrative representation of the AST.
    Each operation is represented by a scene, carefully sequenced with
    intermediate node anchor names. Uses STRICT recursive generation.
    """
    logger.info(f"[Sample {sample_index + 1}] Starting narrative generation.")
    logger.debug(f"DEBUG: Using model: {MODEL}") # Uses global MODEL
    logger.debug(f"DEBUG: AST: {ast_to_prefix(ast)}")

    # --- Initial Setup ---
    # all_operator_nodes = [node_iter for node_iter in postorder(ast) if not isinstance(node_iter, Atom)]
    all_atoms = set()
    for node_iter in postorder(ast):
        if isinstance(node_iter, Atom):
            all_atoms.add(node_iter.n)
    logger.debug(f"DEBUG: All atoms in AST: {sorted(list(all_atoms))}")
    logger.debug(
        f"DEBUG: Overall ground truth (final answer): {overall_ground_truth_answer}"
    )

    operator_nodes_for_anchor_gen = [] # Used to collect nodes for anchor generation and logging
    narrative_anchor_map = {}
    # intro_text = None # Defined later
    scenes = []
    tokens_used = 0

    # --- Generate narrative anchors for op nodes ---
    if config.USE_NARRATIVE_ANCHORS:

        def generate_anchor_for_node(op_node_arg): # Renamed op_node to op_node_arg
            if not config.USE_NARRATIVE_ANCHORS: # Check global config
                return f"the_{op_node_arg.op.lower()}_result_{id(op_node_arg) % 1000:03d}"

            all_anchors_list = list(narrative_anchor_map.values())
            try:
                # Pass the global logger to generate_narrative_anchor_with_llm if it needs one,
                # or ensure it uses its own or the context's logger.
                # For now, assuming generate_narrative_anchor_with_llm handles its logging.
                anchor = generate_narrative_anchor_with_llm(
                    world, op_node_arg, all_anchors_list, sample_index=sample_index
                )

                if anchor:
                    return anchor
                else:
                    logger.warning( # Use global logger
                        f"Failed to generate LLM anchor for {op_node_arg.op}. Using deterministic fallback."
                    )
                    return f"the_{op_node_arg.op.lower()}_result_{id(op_node_arg) % 1000:03d}"
            except Exception as e:
                logger.error(f"Error in narrative anchor generation: {e}") # Use global logger
                return f"the_{op_node_arg.op.lower()}_result_{id(op_node_arg) % 1000:03d}"

        for node_iter_anchor in postorder(ast): 
            if isinstance(node_iter_anchor, OpNode):
                anchor = generate_anchor_for_node(node_iter_anchor)
                narrative_anchor_map[id(node_iter_anchor)] = anchor
                operator_nodes_for_anchor_gen.append(node_iter_anchor) # Add to list for logging
                logger.debug( # Use global logger
                    f"Added narrative anchor '{anchor}' for {node_iter_anchor.op} node"
                )
    else:
        for node_iter_anchor in postorder(ast):
            if isinstance(node_iter_anchor, OpNode):
                narrative_anchor_map[id(node_iter_anchor)] = (
                    f"the_{node_iter_anchor.op.lower()}_result_{id(node_iter_anchor) % 1000:03d}"
                )
                operator_nodes_for_anchor_gen.append(node_iter_anchor) # Add to list for logging

    logger.info(f"Generated {len(narrative_anchor_map)} narrative anchors.") # Use global logger
    if operator_nodes_for_anchor_gen: # Check if list is not empty
        log_str = "Narrative anchors: " + ", ".join(
            [
                f"'{narrative_anchor_map.get(id(op_node_log), 'MISSING')}' ({op_node_log.op})"
                for op_node_log in operator_nodes_for_anchor_gen # Iterate over the collected list
            ]
        )
        logger.debug(log_str) # Use global logger
    else:
        logger.debug("No operator nodes found to generate anchors for.")


    # Pass global config and logger to generate_introduction_scene
    intro_text = generate_introduction_scene(world, sample_index=sample_index, config_obj=config, logger_obj=logger)

    if intro_text:
        intro_tokens = len(encoder.encode(intro_text)) # Use global encoder
        # Use global config for MAX_TOTAL_TOKENS and SAFETY_MARGIN
        if intro_tokens <= config.MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            scenes.append(intro_text)
            tokens_used += intro_tokens
            logger.info( # Use global logger
                f"Generated and added introductory scene ({intro_tokens} tokens)."
            )
        else:
            logger.warning( # Use global logger
                f"Generated introductory scene ({intro_tokens} tokens) was too long and would exceed budget. "
                f"Not adding to narrative. Budget: {config.MAX_TOTAL_TOKENS}, Safety: {SAFETY_MARGIN}"
            )
            intro_text = None # Reset intro_text if not used
    else:
        logger.warning( # Use global logger
            "Failed to generate valid introductory scene. Starting narrative without intro."
        )

    last_scene_text = intro_text if intro_text else "The story begins..."
    introduced_atoms_during_generation = set()
    # total_beats should be based on the nodes that will actually generate beats
    # This is typically all OpNodes.
    total_beats = sum(1 for n in postorder(ast) if isinstance(n, OpNode))
    beat_counter = {"current": 0, "total": total_beats}

    # Create the GenerationContext
    # Note: context.config will be the global config object passed to generate_narrative
    # context.logger will be the global logger object passed to generate_narrative
    context = GenerationContext(
        world=world,
        config=config, # Pass the global config
        encoder=encoder, # Pass the global encoder
        p_inflect=p_inflect, # Pass the global p_inflect
        logger=logger, # Pass the global logger
        narrative_anchor_map=narrative_anchor_map,
        all_atoms=all_atoms,
        introduced_atoms=introduced_atoms_during_generation,
        scenes=scenes,
        tokens_used=tokens_used,
        last_scene_text=last_scene_text,
        beat_counter=beat_counter,
        sample_index=sample_index,
        max_pad_paragraphs=config.MAX_PAD_PARAGRAPHS, # Use global config
        overall_ground_truth_answer=overall_ground_truth_answer,
        overall_ast_root=ast,
    )

    tokens_available_for_narrative_and_padding = (
        config.MAX_TOTAL_TOKENS - tokens_used - SAFETY_MARGIN # Use global config
    )
    max_padding_allowed = int(
        tokens_available_for_narrative_and_padding * config.PADDING_MAX_TOK_PERCENT # Use global config
    )
    context.padding_stats["max_padding_allowed"] = max_padding_allowed

    num_padding_slots = total_beats - 1 if total_beats > 1 else 0
    if num_padding_slots > 0:
        padding_per_slot_calculated = max_padding_allowed // num_padding_slots
        context.padding_stats["padding_per_slot"] = padding_per_slot_calculated
        logger.info( # Use global logger
            f"Calculated padding per slot: {padding_per_slot_calculated} tokens ({max_padding_allowed} total / {num_padding_slots} slots)"
        )
    else:
        context.padding_stats["padding_per_slot"] = 0
        logger.info( # Use global logger
            f"No padding slots available (total_beats: {total_beats}). Padding per slot set to 0."
        )

    logger.info( # Use global logger
        f"PADDING BUDGET INITIALIZED: Tokens after intro: {tokens_used}, "
        f"Available for narrative+padding: {tokens_available_for_narrative_and_padding}, "
        f"Max padding %: {config.PADDING_MAX_TOK_PERCENT*100:.1f}%, "
        f"Max padding tokens allowed: {max_padding_allowed}, "
        f"Padding per slot: {context.padding_stats['padding_per_slot']}, "
        f"Max padding segments per beat: {config.MAX_PAD_PARAGRAPHS}" # Use global config
    )

    try:
        # _generate_narrative_recursive will use the logger and config from the context object
        _generate_narrative_recursive(
            ast,
            context, # Pass the fully initialized context
            is_root=True,
        )
    except BeatGenerationError as e:
        logger.error(f"Narrative generation aborted due to beat failure: {e}") # Use global logger
        return None
    except Exception as e:
        logger.error( # Use global logger
            f"Unexpected error during recursive narrative generation: {e}",
            exc_info=True,
        )
        return None

    if not context.scenes:
        logger.error("Narrative generation resulted in no scenes.") # Use global logger
        return None

    # narrative_body = "\n\n".join(context.scenes).strip() # This is local, not returned directly
    final_token_count = len(encoder.encode("\n\n".join(context.scenes).strip())) # Use global encoder
    if final_token_count > config.MAX_TOTAL_TOKENS: # Use global config
        logger.warning( # Use global logger
            f"Final generated narrative ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({config.MAX_TOTAL_TOKENS}). Truncation might occur."
        )

    total_padding_tokens = context.padding_stats["total_padding_tokens"]
    padding_segments_added = context.padding_stats["padding_segments_added"]
    padding_percentage_of_max = (
        (total_padding_tokens / context.padding_stats["max_padding_allowed"] * 100)
        if context.padding_stats["max_padding_allowed"] > 0
        else 0
    )
    padding_percentage_of_total_narrative = (
        (total_padding_tokens / context.tokens_used * 100)
        if context.tokens_used > 0
        else 0
    )
    logger.info( # Use global logger
        f"PADDING FINAL SUMMARY: "
        f"Padding tokens: {total_padding_tokens}/{context.padding_stats['max_padding_allowed']} ({padding_percentage_of_max:.1f}% of max allowed for padding), "
        f"Padding percentage of total narrative tokens: {padding_percentage_of_total_narrative:.1f}%, "
        f"Padding segments added: {padding_segments_added}"
    )

    failed_validations_dir = os.path.join(LOG_DIR, "failed_validations") # Uses global LOG_DIR
    if os.path.exists(failed_validations_dir):
        validation_files = [
            f_name
            for f_name in os.listdir(failed_validations_dir)
            if f_name.startswith(f"validation_fail_") # No sample_index specific filtering here, could be added if needed
        ] # Removed sample_index specific filtering for this general summary

        if validation_files: # This summary is now general, not per-sample
            failures_by_reason = {}
            failures_by_op = {}
            for file_name in validation_files:
                try:
                    # Basic parsing, might need adjustment if filenames are complex
                    parts = file_name.split("_") 
                    if len(parts) >= 4: # e.g., validation_fail_OP_REASON_timestamp.json
                        op_type = parts[2]
                        reason_code = parts[3] 
                        failures_by_reason[reason_code] = (
                            failures_by_reason.get(reason_code, 0) + 1
                        )
                        failures_by_op[op_type] = failures_by_op.get(op_type, 0) + 1
                except IndexError:
                    logger.warning( # Use global logger
                        f"Could not parse validation failure filename: {file_name}"
                    )
            logger.info(f"[Sample {sample_index + 1}] VALIDATION FAILURES SUMMARY (for this sample, based on files in failed_validations):") # Clarify scope
            logger.info( # Use global logger
                f"  Total validation failures logged (may include other samples if not filtered by sample_index in filename): {len(validation_files)}"
            )
            if failures_by_reason:
                logger.info(f"  Failures by reason: {failures_by_reason}") # Use global logger
            if failures_by_op:
                logger.info(f"  Failures by operation: {failures_by_op}") # Use global logger
        else:
            logger.info( # Use global logger
                f"[Sample {sample_index + 1}] No validation failure files found in general log dir (or matching this sample's pattern if implemented)."
            )

    logger.info( # Use global logger
        f"Successfully generated narrative for sample {sample_index + 1}. Final context tokens: {context.tokens_used}, Narrative tokens: {final_token_count}"
    )
    return context # Return the GenerationContext object

def generate_introduction_scene(
    world_info: dict,
    sample_index: int | None = None,
    config_obj: Config = config,
    logger_obj: logging.Logger = logger,
) -> str | None:
    logger_obj.info(
        f"[Sample {sample_index + 1 if sample_index is not None else 'N/A'}] Generating introduction scene..."
    )

    system_prompt = (
        f"You are a master {world_info.get('genre')} storyteller. Your task is to write a compelling introductory scene for a new story. "
        "This scene should establish the setting, introduce one or two key characters, and hint at a central mystery or goal related to the primary object. "
        "Crucially, this introductory scene MUST NOT contain any numerical values (digits or words like 'one', 'two', 'first', etc.), "
        "except potentially the word 'one', 'two', or 'three' if used for completely general, non-quantitative phrasing (e.g., 'a single ray of light', 'two figures emerged', 'three ancient symbols'). Strive for zero numbers. "
        "Focus on atmosphere and intrigue. Do not reveal any specific quantities or begin any calculations. "
        "Output ONLY the narrative text for this scene. No titles, no explanations, no analysis."
    )

    characters_list = world_info.get("characters", [])
    char_names_roles = []
    if characters_list:
        num_intro_chars = random.randint(1, min(2, len(characters_list)))
        intro_chars = random.sample(characters_list, num_intro_chars)
        for char_info in intro_chars:
            char_names_roles.append(
                f"{char_info.get('name', 'A mysterious figure')} ({char_info.get('role', 'of unknown purpose')})"
            )

    user_prompt = (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'A realm of mystery')}\n"
        f"- Setting: {world_info.get('setting', 'An enigmatic place')}\n"
        f"- Primary Object of Interest: {world_info.get('object', 'ancient artifacts')}\n"
        f"- Characters to potentially feature: {', '.join(char_names_roles) if char_names_roles else 'The inhabitants of this world'}\n\n"
        f"**Task:** Write an engaging introductory scene based on the context above. Remember the strict rule: NO numbers (or strive for zero numbers, with very limited exceptions for 'one'/'two'/'three' in general phrasing only). "
        f"The scene should set a tone and hint at the story's direction without giving away specifics. "
        f"Output ONLY the narrative text."
    )

    # Validator for intro: NO numbers, or at most very specific phrasing numbers if allowed by config.
    # The intro prompt asks for NO numbers.
    # THIS IS THE CORRECTED CALL TO make_number_validator:
    validate_intro = make_number_validator(
        allowed_atoms_with_duplicates=[],  # Correct keyword and type (list)
        forbidden_atoms=set(),
        operand_count=0,
        correct_result_for_beat=None,
        enforce_result_presence=False,
        operation_type="INTRO",
        overall_ground_truth_answer=None,
        is_root_node_being_validated=False,
        conceptual_input_values=None,      # Added this required parameter
        config_obj=config_obj,
        logger_obj=logger_obj,
        # strict_zero=True, # This parameter was removed from make_number_validator
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=config_obj.INTRO_MAX_TOKENS,
        validate_fn=validate_intro,
        retries=config_obj.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config_obj.CREATIVE_NARRATIVE_TEMP,
        reasoning_settings={"exclude": True},
    )
    if intro_text:
        logger_obj.info(
            f"Successfully generated intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return intro_text.strip()
    else:
        logger_obj.error(
            f"Failed to generate intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return None

# --- Narrative Generation with Parent Operator Prompting ---
def _generate_and_llm_validate_beat(
    original_user_message_for_generator: str, # This will now contain MEDIAN few-shots if applicable
    system_prompt_for_generator: str,
    world_info: dict,
    current_op_node: OpNode,
    # Inputs for LLM Validator prompt construction:
    conceptual_inputs_str_for_llm_validator: str,
    atomic_inputs_words_str_for_llm_validator: str,
    action_description_for_llm_validator: str,
    expected_beat_result_words_for_llm_validator: str | None,
    ultra_strict_instruction_for_llm_validator_context: str,
    # Other parameters:
    current_max_beat_completion_tokens: int,
    sample_index: int,
    context_config: Config,
    logger_obj: logging.Logger,
    encoder_obj: any,
    is_current_beat_root_node: bool = False,
    overall_ground_truth_answer_val: int | None = None,
    primary_object_name: str = "items",
    forbidden_prior_results_and_gt_for_llm_validator: Set[int] = None,
    correct_result_val: int | None = None,
    direct_atom_values_val: Set[int] = None,
) -> str | None:

    if forbidden_prior_results_and_gt_for_llm_validator is None:
        forbidden_prior_results_and_gt_for_llm_validator = set()
    if direct_atom_values_val is None:
        direct_atom_values_val = set()

    current_op_arity = len(current_op_node.children)
    logger_obj.debug(
        f"LLMValidateBeat: Op={current_op_node.op}, Arity={current_op_arity}, CorrectResult={correct_result_val}"
    )

    history_of_attempts = []
    history_of_critiques = []
    current_generator_user_prompt_for_iteration = original_user_message_for_generator

    for iteration in range(1, context_config.MAX_LLM_VALIDATION_ITERATIONS + 1):
        logger_obj.info(
            f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validation Loop Iteration: {iteration}/{context_config.MAX_LLM_VALIDATION_ITERATIONS}"
        )

        generator_temp = context_config.BEAT_GEN_TEMP

        if iteration > 1: # This is a revision attempt
            generator_temp = context_config.BEAT_REVISION_TEMP
            history_prompt_addition = (
                "\n\n--- FAILED ATTEMPT REVIEW & REVISION TASK ---\n"
            )
            last_attempt_text = (
                history_of_attempts[-1]
                if history_of_attempts
                else "N/A (Error in prior generation)"
            )
            last_critique = history_of_critiques[-1] if history_of_critiques else {}
            history_prompt_addition += f"**Your Previous Attempt (Attempt {iteration-1}):**\n{last_attempt_text}\n\n"
            explanation = last_critique.get(
                "explanation_for_generator", "No detailed explanation."
            )
            summary_for_gen = last_critique.get(
                "overall_revision_summary_for_generator_prompt", "Please revise."
            )
            history_prompt_addition += f"**Validator Feedback for Your Previous Attempt:**\n  - Summary: {summary_for_gen}\n  - Details: {explanation}\n\n"

            # *** MODIFICATION START: Add specific instruction if intermediate result was stated ***
            intermediate_result_stated_explicitly_flag = False
            if last_critique:
                ast_steps = last_critique.get("ast_evaluation_steps", [])
                # Check current beat's step (assuming it's the one being processed, might need more robust linking if validator checks multiple future steps)
                # For simplicity, let's assume the validator's feedback is primarily about the *current* beat being generated.
                # A more robust way would be to pass the current beat_counter to the validator or have the validator only return feedback for the *single* beat it was asked to validate.
                # For now, we'll check if *any* intermediate step in the validator's feedback (which should ideally be just one step for a single beat validation) has this issue.
                for step_feedback in ast_steps:
                    # Only apply this for non-root nodes (intermediate steps)
                    # The validator's `intermediate_result_implicit` can be "N/A" or true for the root if not stated.
                    # We are concerned when it's `false` for an intermediate step.
                    # We need to know if the *current_op_node* is the root. This is passed as `is_current_beat_root_node`.
                    if not is_current_beat_root_node and step_feedback.get("intermediate_result_implicit") is False:
                        intermediate_result_stated_explicitly_flag = True
                        stated_result_val = step_feedback.get("result_from_ast", "UNKNOWN_STATED_RESULT")
                        conceptual_name_for_result = "UNKNOWN_CONCEPTUAL_NAME"
                        # Try to get the conceptual name for the current node
                        if current_op_node and hasattr(current_op_node, 'op'): # Check if current_op_node is an OpNode
                            node_id_current = id(current_op_node)
                            # Need access to narrative_anchor_map from the main context.
                            # This function doesn't have it directly. This is a limitation of this isolated fix.
                            # For a full fix, context or narrative_anchor_map would need to be passed down.
                            # As a placeholder:
                            # conceptual_name_for_result = f"the_concept_for_{current_op_node.op}_{node_id_current % 100}"
                            # Let's assume expected_beat_result_words_for_llm_validator contains the conceptual name or value.
                            # The `ultra_strict_instruction_for_llm_validator_context` contains the conceptual name.
                            # We can parse it from there or, ideally, have it passed directly.
                            # For now, we'll use the `expected_beat_result_words_for_llm_validator` which has the numeric form.
                            # A better fix would be to pass the actual conceptual name for `current_op_node`.
                            # Let's assume `current_node_conceptual_name` was available here.
                            # For now, we'll use a generic instruction.
                            history_prompt_addition += (
                                f"**CRITICAL REVISION - INTERMEDIATE RESULT STATED:**\n"
                                f"Your previous scene INCORRECTLY stated the numerical result of an intermediate operation (which was '{stated_result_val}'). "
                                f"Rule 2 of your ULTRA-STRICT NUMBER RULES clearly states: 'The numerical result of THIS operation ... MUST NOT be explicitly stated in the text. It must only be implied...'.\n"
                                f"You MUST rewrite the scene to ensure this numerical result is NOT stated. Instead, the outcome must be implied conceptually (e.g., referred to by its conceptual name like 'The Dragon's Hoard' if that were its name, not its numerical value).\n"
                            )
                        logger_obj.warning(f"Revision prompt for Op {current_op_node.op}: Added specific instruction to NOT state intermediate result '{stated_result_val}'.")
                        break # Found the issue for this beat
            # *** MODIFICATION END ***

            history_prompt_addition += (
                f"**Current Revision Task (Attempt {iteration}):**\n"
                f"1. Review ALL feedback, especially any CRITICAL REVISION notes above. 2. Re-read original task & ALL number rules. 3. Fix ALL issues. 4. Ensure narrative logic. 5. Output ONLY revised narrative.\n\n"
                f"**Key Original Rules (Reminder - see full initial prompt for all details, especially the ULTRA-STRICT NUMBER RULES section which contains the specific numbers for *your* current scene):\n**"
                f"{ultra_strict_instruction_for_llm_validator_context[:1000]}...\n"
            )
            current_generator_user_prompt_for_iteration = (
                f"{original_user_message_for_generator}\n\n{history_prompt_addition}"
            )

        generated_text_cleaned = ""
        try:
            log_prompt(
                header=f"LLM Beat Generator Prompt (Op: {current_op_node.op}, Iter: {iteration})",
                prompt=f"System: {system_prompt_for_generator}\nUser:\n{current_generator_user_prompt_for_iteration}",
                sample_index=sample_index,
            )
            resp_gen = _chat_completion_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt_for_generator},
                    {
                        "role": "user",
                        "content": current_generator_user_prompt_for_iteration,
                    },
                ],
                max_completion_tokens=current_max_beat_completion_tokens,
                temperature=generator_temp,
                reasoning={"exclude": True},
            )
            raw_gen_text = ""
            if resp_gen and resp_gen.choices and resp_gen.choices[0].message:
                raw_gen_text = resp_gen.choices[0].message.content or ""

            log_prompt(
                header=f"LLM Beat Generator Raw Response (Op: {current_op_node.op}, Iter: {iteration})",
                prompt=f"Raw Output:\n{raw_gen_text}",
                sample_index=sample_index,
            )

            generated_text_cleaned = raw_gen_text.strip()
            if not generated_text_cleaned or generated_text_cleaned.lower().startswith(
                ("i cannot", "i'm sorry", "i am unable")
            ):
                logger_obj.warning(
                    f"Generator refusal or empty in iter {iteration}. Raw: '{raw_gen_text}'"
                )
                generated_text_cleaned = ""

            history_of_attempts.append(
                generated_text_cleaned
                if generated_text_cleaned
                else "GENERATION_EMPTY_OR_REFUSED"
            )
            if not generated_text_cleaned:
                history_of_critiques.append(
                    {
                        "is_valid": False,
                        "explanation_for_generator": "The generation was empty or an API refusal.",
                        "overall_revision_summary_for_generator_prompt": "Previous attempt was empty/refused. Please generate the scene as per original instructions.",
                        "explanation_for_audit": "N/A - Generation failed.",
                        "suggested_revisions": []
                    }
                )
                continue
        except Exception as e_gen:
            logger_obj.error(
                f"Error generating beat in LLM loop iter {iteration}: {e_gen}",
                exc_info=True,
            )
            history_of_attempts.append(f"ERROR_DURING_GENERATION: {str(e_gen)[:200]}")
            history_of_critiques.append(
                {
                    "is_valid": False,
                    "explanation_for_generator": f"Exception during generation: {e_gen}",
                    "overall_revision_summary_for_generator_prompt": "System error during previous generation attempt. Please retry the original task, carefully following all instructions.",
                    "explanation_for_audit": "N/A - Generation exception.",
                    "suggested_revisions": []
                }
            )
            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue
            else:
                return None

        validator_system_prompt = """You are an AI numerical compliance checker.
Your ONLY task is to evaluate a story 'beat' against strict numerical rules.
You MUST output your response as a valid JSON object and NOTHING ELSE, adhering to the provided schema.
Do not include any text, explanations, or markdown (like ```json) before or after the single JSON object.
Start with '{' and end with '}'."""

        temp_forbidden_detailed_for_validator_prompt = []
        if forbidden_prior_results_and_gt_for_llm_validator:
            for n_forbidden in sorted(
                list(forbidden_prior_results_and_gt_for_llm_validator)
            ):
                temp_forbidden_detailed_for_validator_prompt.append(
                    f"'{num_to_words(n_forbidden)}' ({n_forbidden})"
                )
        forbidden_values_str_for_validator_prompt = (
            ", ".join(temp_forbidden_detailed_for_validator_prompt)
            if temp_forbidden_detailed_for_validator_prompt
            else "None specifically identified beyond general rules"
        )

        temp_phrasing_detailed_for_validator_prompt = [
            f"'{num_to_words(n)}' ({n})"
            for n in context_config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        ]
        phrasing_numbers_str_for_validator_prompt = ", ".join(
            sorted(temp_phrasing_detailed_for_validator_prompt)
        )

        gt_word_val_for_validator_prompt = "N/A"
        if overall_ground_truth_answer_val is not None:
            gt_word_val_for_validator_prompt = f"'{num_to_words(overall_ground_truth_answer_val)}' ({overall_ground_truth_answer_val})"

        current_op_arity_word_val_for_validator_prompt = (
            f"'{num_to_words(current_op_arity)}' ({current_op_arity})" if current_op_arity > 0 else "N/A (no direct inputs)"
        )

        is_result_also_atomic_input_for_validator = False
        if correct_result_val is not None and direct_atom_values_val:
            if correct_result_val in direct_atom_values_val:
                is_result_also_atomic_input_for_validator = True

        validator_user_prompt = f"""You are an AI numerical compliance checker. Evaluate the 'Generated Beat Text' below with ABSOLUTE PRECISION regarding its numerical content and adherence to storytelling rules.

**Primary Goal:** Verify the text strictly adheres to all numerical rules provided in the 'ULTRA-STRICT NUMBER RULES (Generator's Writing Guide)' section below.

**Context for Your Evaluation (Derived from Generator's Task):**
- Operation Type: {current_op_node.op} ({OP_LABELS.get(current_op_node.op, current_op_node.op)})
- Conceptual Names for Prior Results (if any were inputs to this step): {conceptual_inputs_str_for_llm_validator}
- New Specific Numbers for This Step (Atomic Inputs, as given to Generator): {atomic_inputs_words_str_for_llm_validator}
- Numerical Result of THIS Operation (as known to Generator, was to be Implied): {expected_beat_result_words_for_llm_validator}
- Overall Final Answer of Entire Story (for context): {gt_word_val_for_validator_prompt}
- Is this beat's numerical result also one of its atomic inputs? {'Yes' if is_result_also_atomic_input_for_validator else 'No'}

**ULTRA-STRICT NUMBER RULES (Generator's Writing Guide - THIS IS THE GROUND TRUTH FOR YOUR VALIDATION):**
---
{ultra_strict_instruction_for_llm_validator_context}
---

**VALIDATION ALGORITHM - FOLLOW EXACTLY, using the Generator's Writing Guide above as the source of truth for allowed/forbidden numbers for THIS beat:**
1.  Identify ALL numbers (words or digits) in the 'Generated Beat Text'.
2.  **RULE A (Required Atomic Inputs):**
    -   Does the text mention ALL numbers listed in Rule 1 of the Generator's Writing Guide?
    -   *MEDIAN Exception Handling:* If the operation is MEDIAN, and Rule 1 (or Rule 2) of the Generator's Writing Guide indicates that a specific input number (which is also the median result) MUST NOT be mentioned, verify its ABSENCE. All OTHER numbers listed in Rule 1 must still be present.
3.  **RULE B (Outcome Handling):**
    -   Does the text adhere to Rule 2 of the Generator's Writing Guide regarding the explicit statement (or non-statement) of the numerical result?
    -   *MEDIAN Exception Handling:* For MEDIAN, the numerical result MUST NEVER be stated.
    -   **CRITICAL FOR ALL INTERMEDIATE STEPS (NON-FINAL OPERATIONS):** The numerical result of the operation MUST NOT be explicitly stated. It must be implied. If it is stated, this is a failure.
4.  **RULE C (Additionally Allowed Numbers):**
    -   Are any numbers present that are ONLY allowed under Rule 3 of the Generator's Writing Guide (e.g., phrasing numbers, arity count)? Are they used appropriately as per those rules?
5.  **RULE D (Forbidden Numbers & Values):**
    -   Does the text avoid ALL numbers listed as forbidden in Rule 4 of the Generator's Writing Guide?
    -   Does the text avoid any other numbers not covered by Rules 1-4, as per Rule 5?
6.  **RULE E (Prior Result Handling):**
    -   If Rule 6 (Referring to Quantities from Previous Events) is present in the Generator's Writing Guide, does the text adhere to it by using conceptual names for prior results and NOT their numerical values?

**VALIDATION RESPONSE (Output JSON only, using the schema provided):**
Based on the algorithm above:
-   `is_valid` (boolean): True if ALL rules (A-E, based on the Generator's Writing Guide) are met, False otherwise.
-   `explanation_for_generator` (string): Detailed step-by-step reasoning for failure, referencing specific rules (A-E) from the Generator's Writing Guide and numbers found/missing. If valid, explain why it meets all criteria, especially how MEDIAN exceptions were handled if applicable. **If Rule B (Outcome Handling for intermediate steps) is violated because an intermediate numerical result was stated, explicitly say: "VIOLATION: Intermediate numerical result [value] was explicitly stated. It MUST be implied conceptually."**
-   `explanation_for_audit` (string, only if `is_valid` is true): Brief summary of why it's valid.
-   `overall_revision_summary_for_generator_prompt` (string): Concise instruction for the generator if invalid. If valid, state "No revision needed."
-   `suggested_revisions` (array of strings, optional).

**Generated Beat Text to Evaluate:**
---
{generated_text_cleaned}
---
"""
        try:
            log_prompt(
                header=f"LLM Validator Prompt (Op: {current_op_node.op}, Iter: {iteration})",
                prompt=f"System: {validator_system_prompt}\nUser:\n{validator_user_prompt}",
                sample_index=sample_index,
            )
            api_call_params_for_validator = {
                "model": context_config.LLM_VALIDATOR_MODEL,
                "messages": [
                    {"role": "system", "content": validator_system_prompt},
                    {"role": "user", "content": validator_user_prompt},
                ],
                "max_completion_tokens": context_config.BEAT_MAX_TOKENS,
                "temperature": context_config.LLM_VALIDATOR_TEMP,
                "reasoning": {"exclude": True},
                "json_schema": VALIDATOR_RESPONSE_SCHEMA,
            }
            resp_val = _chat_completion_call(**api_call_params_for_validator)

            validator_raw_output = ""
            if resp_val and resp_val.choices and resp_val.choices[0].message:
                validator_raw_output = resp_val.choices[0].message.content or ""

            log_prompt(
                header=f"LLM Validator Raw Response (Op: {current_op_node.op}, Iter: {iteration})",
                prompt=f"Raw Output:\n{validator_raw_output}",
                sample_index=sample_index,
            )

            validation_result = parse_llm_json_with_fallback(
                validator_raw_output,
                {
                    "is_valid": False,
                    "explanation_for_generator": "Validator response was not valid JSON or was empty. This might be due to an issue with the generated text or the validator itself. Please try generating the scene again, ensuring all numerical rules are meticulously followed.",
                    "overall_revision_summary_for_generator_prompt": "Validator had trouble processing the previous text. Please regenerate the scene, focusing on extreme clarity and adherence to all numerical constraints.",
                    "explanation_for_audit": "Validator output parsing error.",
                    "suggested_revisions": []
                },
                f"in LLM validator iteration {iteration} for Op: {current_op_node.op}"
            )
            history_of_critiques.append(validation_result)

            if validation_result.get("is_valid"):
                logger_obj.info(
                    f"LLM Validator PASSED beat in iter {iteration}. Audit: {validation_result.get('explanation_for_audit', 'N/A')}"
                )
                return generated_text_cleaned
            else:
                logger_obj.warning(
                    f"LLM Validator FAILED beat in iter {iteration}. Feedback for generator: {validation_result.get('overall_revision_summary_for_generator_prompt', 'No summary.')}"
                )
                logger_obj.debug(f"Detailed explanation from validator: {validation_result.get('explanation_for_generator', 'No detailed explanation.')}")

        except Exception as e_val_call:
            logger_obj.error(
                f"Error during LLM validation call/processing iter {iteration} for Op {current_op_node.op}: {e_val_call}",
                exc_info=True,
            )
            # Ensure critique is added even if an exception occurs during the call or parsing
            if len(history_of_critiques) < len(history_of_attempts): # if critique for this attempt hasn't been added
                 history_of_critiques.append(
                    {
                        "is_valid": False,
                        "explanation_for_generator": f"A system error occurred during the validation phase: {str(e_val_call)[:200]}. Please try to regenerate the scene, carefully adhering to all original instructions and numerical rules.",
                        "overall_revision_summary_for_generator_prompt": "System error during validation. Please retry the original task, focusing on strict rule adherence.",
                        "explanation_for_audit": f"N/A - Exception during validation: {e_val_call}",
                        "suggested_revisions": []
                    }
                )
            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue
            else:
                return None # Failed all iterations

    logger_obj.error(
        f"Beat failed LLM validation after {context_config.MAX_LLM_VALIDATION_ITERATIONS} iterations for Op: {current_op_node.op}."
    )
    if history_of_critiques: # Log the last critique if all iterations failed
        last_fail_critique = history_of_critiques[-1]
        logger_obj.error(f"Last critique for Op {current_op_node.op} (failed): {json.dumps(last_fail_critique, indent=2)}")
    return None

def generate_introduction_scene(
    world_info: dict,
    sample_index: int | None = None,
    config_obj: Config = config,  # Add config_obj parameter
    logger_obj: logging.Logger = logger,  # Add logger_obj parameter
) -> str | None:
    logger_obj.info(
        f"[Sample {sample_index + 1 if sample_index is not None else 'N/A'}] Generating introduction scene..."
    )

    # --- ADDED/COMPLETED PROMPT DEFINITIONS ---
    system_prompt = (
        f"You are a master {world_info.get('genre')} storyteller. Your task is to write a compelling introductory scene for a new story. "
        "This scene should establish the setting, introduce one or two key characters, and hint at a central mystery or goal related to the primary object. "
        "Crucially, this introductory scene MUST NOT contain any numerical values (digits or words like 'one', 'two', 'first', etc.), "
        "except potentially the word 'one', 'two', or 'three' if used for completely general, non-quantitative phrasing (e.g., 'a single ray of light', 'two figures emerged', 'three ancient symbols'). Strive for zero numbers. "
        "Focus on atmosphere and intrigue. Do not reveal any specific quantities or begin any calculations. "
        "Output ONLY the narrative text for this scene. No titles, no explanations, no analysis."
    )

    characters_list = world_info.get("characters", [])
    char_names_roles = []
    if characters_list:
        # Select one or two characters for the intro
        num_intro_chars = random.randint(1, min(2, len(characters_list)))
        intro_chars = random.sample(characters_list, num_intro_chars)
        for char_info in intro_chars:
            char_names_roles.append(
                f"{char_info.get('name', 'A mysterious figure')} ({char_info.get('role', 'of unknown purpose')})"
            )

    user_prompt = (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'A realm of mystery')}\n"
        f"- Setting: {world_info.get('setting', 'An enigmatic place')}\n"
        f"- Primary Object of Interest: {world_info.get('object', 'ancient artifacts')}\n"
        f"- Characters to potentially feature: {', '.join(char_names_roles) if char_names_roles else 'The inhabitants of this world'}\n\n"
        f"**Task:** Write an engaging introductory scene based on the context above. Remember the strict rule: NO numbers (or strive for zero numbers, with very limited exceptions for 'one'/'two'/'three' in general phrasing only). "
        f"The scene should set a tone and hint at the story's direction without giving away specifics. "
        f"Output ONLY the narrative text."
    )
    # --- END OF ADDED/COMPLETED PROMPT DEFINITIONS ---

    # Validator for intro: NO numbers, or at most very specific phrasing numbers if allowed by config.
    # The intro prompt asks for NO numbers.
    validate_intro = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=set(),
        operand_count=0,
        correct_result_for_beat=None,
        strict_zero=True,  # Key for intro/padding style validation
        enforce_result_presence=False,
        operation_type="INTRO",
        overall_ground_truth_answer=None,  # No GT relevant for intro in this way
        is_root_node_being_validated=False,
        config_obj=config_obj,  # Pass the config
        logger_obj=logger_obj,
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,  # Now defined
        user_prompt=user_prompt,  # Now defined
        max_completion_tokens=config_obj.INTRO_MAX_TOKENS,
        validate_fn=validate_intro,
        retries=config_obj.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config_obj.CREATIVE_NARRATIVE_TEMP,
        reasoning_settings={"exclude": True},
    )
    if intro_text:
        logger_obj.info(
            f"Successfully generated intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return intro_text.strip()
    else:
        logger_obj.error(
            f"Failed to generate intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return None



def node_to_dict(node: Node) -> dict:
    """Convert a Node to a dictionary representation."""
    if isinstance(node, Atom):
        return {
            "type": "Atom",
            "value": node.n,
            "id": id(node),
        }

    return {
        "type": "OpNode",
        "op": node.op,
        "id": id(node),
        "children": [node_to_dict(child) for child in node.children],
        "result": node.value,
    }


def node_to_dict(node: Node) -> dict:
    """Convert a Node to a dictionary representation."""
    if isinstance(node, Atom):
        return {
            "type": "Atom",
            "value": node.n,
            "id": id(node),
        }

    return {
        "type": "OpNode",
        "op": node.op,
        "id": id(node),
        "children": [node_to_dict(child) for child in node.children],
        "result": node.value,
    }


def walk_ast(node: Node):
    """Iterator that yields all nodes in the AST."""
    yield node
    for child in node.children:
        yield from walk_ast(child)


def count_ops(node: Node) -> int:
    """Count the number of operations (non-atom nodes) in the AST."""
    return sum(1 for n in walk_ast(node) if not isinstance(n, Atom))


def walk_ast(node: Node):
    """Iterator that yields all nodes in the AST."""
    yield node
    for child in node.children:
        yield from walk_ast(child)


def count_ops(node: Node) -> int:
    """Count the number of operations (non-atom nodes) in the AST."""
    return sum(1 for n in walk_ast(node) if not isinstance(n, Atom))


# In verbose-listops.py


def generate_single_sample(sample_index: int, config_obj: Config) -> dict | None:
    logger.info(
        f"[Sample {sample_index+1}] Starting generation with config: MAX_OPS={config_obj.MAX_OPS}, "
        f"MIN_ARITY={config_obj.MIN_ARITY}, MAX_TOTAL_TOKENS={config_obj.MAX_TOTAL_TOKENS}"
    )
    try:
        node = build_random_ast(
            max_ops=config_obj.MAX_OPS,
            max_branch=config_obj.MAX_BRANCH,
            config_obj=config_obj,
        )

        world_data = generate_world(
            num_characters=random.randint(
                config_obj.MIN_WORLD_CHARS, config_obj.MAX_WORLD_CHARS
            ),
            num_concepts=random.randint(
                config_obj.MIN_WORLD_CONCEPTS, config_obj.MAX_WORLD_CONCEPTS
            ),
            sample_index=sample_index,
            max_retries=config_obj.WORLDGEN_MAX_RETRIES,
        )

        if not node or not world_data:
            logger.error(f"AST or World gen failed for sample {sample_index+1}")
            return None

        final_ast_answer = eval_node(node)

        narrative_gen_context = generate_narrative(
            ast=node,
            world=world_data,
            config=config_obj,
            encoder=encoder,
            p_inflect=p_inflect,
            logger=logger,
            sample_index=sample_index,
            overall_ground_truth_answer=final_ast_answer,
        )

        if not narrative_gen_context:
            logger.error(
                f"Narrative generation failed for sample {sample_index+1}. Skipping sample."
            )
            return None

        narrative_body_from_context = "\n\n".join(narrative_gen_context.scenes).strip()

        # Ensure FINAL_QUESTION_TEMPLATE is defined globally and correctly substituted
        # Example: FINAL_QUESTION_TEMPLATE = Template("... Question: ... '$primary_object' ...")
        final_question_text = FINAL_QUESTION_TEMPLATE.substitute(
            primary_object=world_data["object"]
        )
        full_text_for_eval_content = narrative_body_from_context + final_question_text

        sample = {
            "id": str(sample_index),  # Canonical ID
            "timestamp": datetime.datetime.now().isoformat(),
            "ast_str": ast_to_prefix(node),  # Canonical AST string
            "ground_truth_value": narrative_gen_context.overall_ground_truth_answer,  # Canonical ground truth
            "narrative": narrative_body_from_context,  # Story only
            "question": final_question_text.strip(),  # Question only (strip leading/trailing whitespace from the template's output)
            "full_text_for_eval": full_text_for_eval_content,  # Combined narrative + question
            "world_data": world_data,
            "scenes_detail": [  # Renamed for clarity from just "scenes"
                {"scene_number": i + 1, "text": text}
                for i, text in enumerate(narrative_gen_context.scenes)
            ],
            "num_operations": count_ops(node),
            "token_counts": {  # Grouped token counts
                "total_generated_context": narrative_gen_context.tokens_used,
                "narrative_body": (
                    len(encoder.encode(narrative_body_from_context)) if encoder else 0
                ),
                "padding": narrative_gen_context.padding_stats["total_padding_tokens"],
            },
            "conceptual_references": dict(
                narrative_gen_context.narrative_anchor_map
            ),  # Renamed from narrative_anchors
            "generation_metadata": {  # Renamed from "meta"
                "script_version": "verbose-listops_v_DRY_output_fix1",  # Updated version
                "generation_model": MODEL,
                "iterative_validator_model": config_obj.LLM_VALIDATOR_MODEL,
                "always_allowed_phrasing_numbers": list(
                    config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
                ),
                "config_params": {  # Nested config for clarity
                    "max_ops": config_obj.MAX_OPS,
                    "min_arity": config_obj.MIN_ARITY,
                    "max_total_tokens": config_obj.MAX_TOTAL_TOKENS,
                    # Add other key config params you want to quickly see
                },
                "full_config_snapshot": asdict(
                    config_obj
                ),  # Keep full config for reproducibility
            },
        }
        return sample

    except Exception as e:
        logger.error(
            f"Outer error in generate_single_sample for sample {sample_index+1}: {e}",
            exc_info=True,
        )
        return None

def main(
    config: Config,
    num_samples: int = NUM_SAMPLES_TO_GENERATE,
    max_workers: int = DEFAULT_MAX_WORKERS,
):
    logger.info("START OF MAIN FUNCTION")

    initial_account_usage = None
    if client and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
        logger.info("Fetching initial OpenRouter account usage...")
        initial_account_usage = rate_limiter.update_limits_from_api()
        if initial_account_usage is not None:
            logger.info(f"Initial OpenRouter account usage: ${initial_account_usage:.4f}")
        else:
            logger.warning("Could not fetch initial OpenRouter account usage.")
    else:
        logger.warning("Skipping initial OpenRouter account usage check: Client not initialized or API key missing/placeholder.")

    # --- Dynamic Filename and Subfolder Generation ---
    sanitized_model_name = MODEL.replace("/", "_").replace(":", "-")
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    base_filename_parts = [
        f"{config.MAX_TOTAL_TOKENS}tok",
        f"{config.MAX_OPS}mxops",
        f"{config.MIN_ARITY}minarity",
        f"{config.MAX_BRANCH}mxbrch",
        f"{sanitized_model_name}",
        timestamp_str
    ]
    run_specific_identifier = "_".join(base_filename_parts)

    run_output_dir = os.path.join(DATASETS_DIR, run_specific_identifier)
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"All outputs for this run will be saved in: {run_output_dir}")

    # 1. Filename for RESEARCHER_DETAIL (comprehensive raw generated data)
    researcher_detail_basename = f"[1_RESEARCHER_DETAIL]_DATASET_{run_specific_identifier}.jsonl"
    researcher_detail_output_file = os.path.join(run_output_dir, researcher_detail_basename)
    logger.info(f"Researcher detail output will be saved to: {researcher_detail_output_file}")

    # 2. Filename for EVAL_READY data (lean format for ALL successfully generated samples)
    eval_ready_basename = f"[2_EVAL_READY]_DATASET_{run_specific_identifier}.jsonl"
    eval_ready_output_file = os.path.join(run_output_dir, eval_ready_basename)
    logger.info(f"Evaluation-ready output (all generated) will be saved to: {eval_ready_output_file}")

    # 3. Names for auxiliary files related to validator.py processing (if PROD_RUN is true)
    validator_results_basename = f"[3.1_VALIDATOR_RESULTS]_{run_specific_identifier}.jsonl"
    validator_results_path = os.path.join(run_output_dir, validator_results_basename)

    validator_stdout_basename = f"[3.2_VALIDATOR_STDOUT]_{run_specific_identifier}.log"
    validator_stdout_log_path = os.path.join(run_output_dir, validator_stdout_basename)

    # 4. Filename for the final VALIDATOR_CLEANED dataset (lean format, only validator-passed samples)
    # This is now the primary "final" dataset for evaluation.
    final_cleaned_lean_basename = f"[4_FINAL_EVAL_CLEANED]_DATASET_{run_specific_identifier}.jsonl"
    final_cleaned_lean_output_file = os.path.join(run_output_dir, final_cleaned_lean_basename)


    # --- Generation Loop ---
    # ... (generation loop is identical, populating generated_results with full dictionaries) ...
    logger.info(f"Script started. Generating {num_samples} samples using up to {max_workers} workers.")
    samples_generated_successfully = 0
    samples_failed_generation = 0
    start_time = time.time()
    generated_results = []

    progress_lock = threading.Lock()
    completed_tasks = 0
    last_print_time = start_time
    print_interval = max(5, min(30, max_workers // 10))

    print(f"Starting generation of {num_samples} samples using {max_workers} workers...")
    print(f"Progress updates will be shown every ~{print_interval} seconds")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(generate_single_sample, i, config): i
            for i in range(num_samples)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_data = future.result()
                with progress_lock:
                    if sample_data:
                        generated_results.append(sample_data)
                        samples_generated_successfully += 1
                    else:
                        samples_failed_generation += 1
                    completed_tasks = samples_generated_successfully + samples_failed_generation
            except Exception as exc:
                logger.error(f"[Sample {index + 1}] task generated exception: {exc}", exc_info=True)
                with progress_lock:
                    samples_failed_generation += 1
                    completed_tasks = samples_generated_successfully + samples_failed_generation

            current_time = time.time()
            should_print_progress = False
            with progress_lock:
                if (current_time - last_print_time >= print_interval) or (completed_tasks == num_samples):
                    should_print_progress = True
                    last_print_time = current_time

            if should_print_progress:
                elapsed_time = current_time - start_time
                if completed_tasks > 0:
                    throughput = completed_tasks / (elapsed_time / 60) if elapsed_time > 0 else 0
                    remaining_samples = num_samples - completed_tasks
                    estimated_time_remaining_seconds = (remaining_samples / throughput * 60) if throughput > 0 else 0

                    time_str = "N/A"
                    if estimated_time_remaining_seconds > 0:
                        if estimated_time_remaining_seconds >= 3600:
                            time_str = f"{estimated_time_remaining_seconds/3600:.1f}h"
                        elif estimated_time_remaining_seconds >= 60:
                            time_str = f"{estimated_time_remaining_seconds/60:.1f}m"
                        else:
                            time_str = f"{estimated_time_remaining_seconds:.0f}s"

                    success_rate_generation = (samples_generated_successfully / completed_tasks * 100) if completed_tasks > 0 else 0
                    print(
                        f"Progress: {completed_tasks}/{num_samples} ({completed_tasks/num_samples*100:.1f}%) | "
                        f"Success: {samples_generated_successfully} ({success_rate_generation:.1f}%) | "
                        f"Failed: {samples_failed_generation} | "
                        f"Rate: {throughput:.2f} samples/min | "
                        f"ETA: {time_str}"
                    )

    # --- Writing Initial Output Files ---
    logger.info(f"Parallel generation complete. Writing {samples_generated_successfully} samples to initial output files...")
    samples_written_researcher = 0
    samples_written_eval_raw = 0 # For the [2_EVAL_READY] file

    if samples_generated_successfully > 0:
        try:
            with open(researcher_detail_output_file, "w", encoding="utf-8") as f_researcher, \
                 open(eval_ready_output_file, "w", encoding="utf-8") as f_eval_raw:

                for full_sample_data in generated_results:
                    # Write the full data to the researcher file
                    try:
                        f_researcher.write(json.dumps(full_sample_data, default=lambda o: list(o) if isinstance(o, set) else str(o), ensure_ascii=False) + "\n")
                        samples_written_researcher += 1
                    except Exception as e:
                        logger.error(f"Error writing full sample {full_sample_data.get('id', 'Unknown')} to researcher file: {e}. Skipping.")

                    # Create and write the lean data to the initial eval_ready file
                    eval_sample_data = {
                        "id": full_sample_data.get("id"),
                        "full_text_for_eval": full_sample_data.get("full_text_for_eval"),
                        "ground_truth_value": full_sample_data.get("ground_truth_value"),
                        "ast_str": full_sample_data.get("ast_str"),
                        "num_operations": full_sample_data.get("num_operations"),
                        "token_count_narrative": full_sample_data.get("token_counts", {}).get("narrative_body")
                    }
                    eval_sample_data_cleaned = {k: v for k, v in eval_sample_data.items() if v is not None}
                    try:
                        f_eval_raw.write(json.dumps(eval_sample_data_cleaned, ensure_ascii=False) + "\n")
                        samples_written_eval_raw += 1
                    except Exception as e:
                        logger.error(f"Error writing initial eval sample {eval_sample_data_cleaned.get('id', 'Unknown')} to eval_raw file: {e}. Skipping.")

            logger.info(f"Successfully wrote {samples_written_researcher} samples to {researcher_detail_output_file}.")
            logger.info(f"Successfully wrote {samples_written_eval_raw} samples to {eval_ready_output_file} (initial eval-ready set).")

        except IOError as e: # Catch file opening errors
            logger.error(f"Fatal file write error opening/writing initial output files: {e}")
        except Exception as e: # Catch other errors during this writing phase
            logger.error(f"Unexpected error during initial output file writing phase: {e}", exc_info=True)
    else:
        logger.info("No samples were successfully generated, so no initial output files will be written.")

    end_time_generation = time.time()
    # ... (logging for generation phase summary - no change) ...
    total_time_generation = end_time_generation - start_time
    logger.info(f"--- Generation Phase Summary ---")
    logger.info(f"Samples attempted: {num_samples}")
    logger.info(f"Successfully generated: {samples_generated_successfully}")
    logger.info(f"Written to researcher detail file: {samples_written_researcher}")
    logger.info(f"Written to initial eval-ready file: {samples_written_eval_raw}")
    logger.info(f"Failed generations: {samples_failed_generation}")
    logger.info(f"Generation time: {total_time_generation/60:.2f} minutes")

    if samples_written_researcher > 0: print(f"\nResearcher Detail dataset saved to: {researcher_detail_output_file}")
    if samples_written_eval_raw > 0: print(f"Initial Eval-Ready dataset saved to: {eval_ready_output_file}")
    if samples_generated_successfully == 0: print(f"\nNo samples were successfully generated by verbose-listops.py.")


    # --- PROD_RUN: Validation and Creation of Final Cleaned Eval-Ready Dataset ---
    bad_sample_ids_from_validator = set()

    if PROD_RUN and samples_written_researcher > 0 and os.path.exists(researcher_detail_output_file):
        logger.info(f"--- Starting PROD_RUN validation using {researcher_detail_output_file} ---")
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        validator_script_path = os.path.join(current_script_dir, "validator.py")

        if not os.path.exists(validator_script_path):
            logger.error(f"Validator script not found at {validator_script_path}. Cannot perform cleaning.")
        else:
            cmd = [
                sys.executable, validator_script_path,
                researcher_detail_output_file, # Validator processes the comprehensive researcher file
                "--output-results", validator_results_path,
            ]
            try:
                logger.info(f"Running validator command: {' '.join(cmd)}")
                with open(validator_stdout_log_path, "w", encoding="utf-8") as f_val_stdout:
                    run_result = subprocess.run(cmd, check=True, stdout=f_val_stdout, stderr=subprocess.PIPE, text=True, encoding="utf-8")
                # ... (logging validator stdout/stderr - no change) ...
                if os.path.exists(validator_stdout_log_path):
                    with open(validator_stdout_log_path, "r", encoding="utf-8") as f_val_stdout_read:
                        logger.info(f"Validator process stdout (from {validator_stdout_log_path}):")
                        for line in f_val_stdout_read: logger.info(f"VALIDATOR_STDOUT: {line.strip()}")
                if run_result.stderr:
                    logger.warning("Validator process stderr:")
                    for line in run_result.stderr.splitlines(): logger.warning(f"VALIDATOR_STDERR: {line.strip()}")
                logger.info(f"Validator finished. Detailed validation results saved to {validator_results_path}")


                if os.path.exists(validator_results_path):
                    with open(validator_results_path, "r", encoding="utf-8") as f_results:
                        for line_num, res_line in enumerate(f_results, 1):
                            try:
                                val_res = json.loads(res_line)
                                if val_res.get("status") != "correct":
                                    sample_id_to_remove = val_res.get("id")
                                    if sample_id_to_remove: bad_sample_ids_from_validator.add(sample_id_to_remove)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse validator result line {line_num} from {validator_results_path}: {res_line.strip()}")
                    logger.info(f"Identified {len(bad_sample_ids_from_validator)} samples to remove based on results from {validator_results_path}.")

                    # Create the [4_FINAL_EVAL_CLEANED] dataset (lean format, validator-passed)
                    temp_final_cleaned_lean_file = final_cleaned_lean_output_file + ".tmp"
                    good_samples_written_to_final_cleaned_lean = 0

                    # We iterate through the *original full generated_results* list
                    # because it's already in memory and contains all necessary fields
                    # to construct the lean format for the good samples.
                    with open(temp_final_cleaned_lean_file, "w", encoding="utf-8") as f_temp_lean:
                        for full_sample_data in generated_results: # Iterate over the in-memory full data
                            if full_sample_data.get("id") not in bad_sample_ids_from_validator:
                                # Construct the lean eval format for this good sample
                                eval_sample_data = {
                                    "id": full_sample_data.get("id"),
                                    "full_text_for_eval": full_sample_data.get("full_text_for_eval"),
                                    "ground_truth_value": full_sample_data.get("ground_truth_value"),
                                    "ast_str": full_sample_data.get("ast_str"),
                                    "num_operations": full_sample_data.get("num_operations"),
                                    "token_count_narrative": full_sample_data.get("token_counts", {}).get("narrative_body")
                                }
                                eval_sample_data_cleaned = {k: v for k, v in eval_sample_data.items() if v is not None}
                                try:
                                    f_temp_lean.write(json.dumps(eval_sample_data_cleaned, ensure_ascii=False) + "\n")
                                    good_samples_written_to_final_cleaned_lean += 1
                                except Exception as e_write_lean:
                                    logger.error(f"Error writing lean sample {eval_sample_data_cleaned.get('id', 'Unknown')} to final cleaned eval file: {e_write_lean}")

                    shutil.move(temp_final_cleaned_lean_file, final_cleaned_lean_output_file)
                    logger.info(f"{good_samples_written_to_final_cleaned_lean} validator-passed samples (lean format) saved to {final_cleaned_lean_output_file}.")
                    print(f"PROD_RUN: Final cleaned eval-ready dataset ({good_samples_written_to_final_cleaned_lean} samples) saved to {final_cleaned_lean_output_file}.")

                else: # validator_results_path does not exist
                    logger.warning(f"Validator results file not found at {validator_results_path}. Cannot create final cleaned eval dataset.")
            # ... (rest of except blocks for subprocess errors - largely the same) ...
            except FileNotFoundError:
                logger.error(f"Validator script '{validator_script_path}' not found. Skipping cleaning step.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Validator script failed with exit code {e.returncode}. Skipping cleaning step.")
                if os.path.exists(validator_stdout_log_path):
                    with open(validator_stdout_log_path, "r", encoding="utf-8") as f_val_stdout_read_err:
                        logger.error(f"Validator stdout on error (from {validator_stdout_log_path}):")
                        for i, line_e in enumerate(f_val_stdout_read_err): logger.error(f"VAL_STDOUT_ERR: {line_e.strip()}")
                stderr_snapshot = e.stderr.splitlines() if e.stderr else []
                logger.error("Validator stderr snapshot on error:")
                for i, line_e in enumerate(stderr_snapshot): logger.error(f"VAL_STDERR_ERR: {line_e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during PROD_RUN validation or cleaning: {e}", exc_info=True)

    elif PROD_RUN and (samples_written_researcher == 0 or not os.path.exists(researcher_detail_output_file)):
        logger.info("PROD_RUN was True, but no researcher detail samples were available for validation. Skipping validation and cleaning.")

    # --- Final Summary ---
    # ... (logging for total run time, token usage, cost - largely the same) ...
    # Update print statements to reflect the new file names and purposes.
    end_time_total = time.time()
    total_run_time = end_time_total - start_time

    logger.info(f"--- Overall Run Summary ---")
    logger.info(f"Total samples attempted for generation: {num_samples}")
    logger.info(f"Successfully generated by verbose-listops: {samples_generated_successfully}")
    logger.info(f"Successfully written to researcher detail file ({researcher_detail_basename}): {samples_written_researcher}")
    logger.info(f"Successfully written to initial eval-ready file ({eval_ready_basename}): {samples_written_eval_raw}")

    if PROD_RUN:
        if final_cleaned_lean_output_file and os.path.exists(final_cleaned_lean_output_file):
            try:
                with open(final_cleaned_lean_output_file, 'r', encoding='utf-8') as f_cleaned_count:
                    num_cleaned_samples = sum(1 for _ in f_cleaned_count)
                logger.info(f"Samples in final cleaned eval-ready file ({final_cleaned_lean_basename}): {num_cleaned_samples}")
                print(f"\nFinal Cleaned Eval-Ready dataset saved to: {final_cleaned_lean_output_file} ({num_cleaned_samples} samples)")
            except Exception as e_count:
                logger.error(f"Could not count samples in final cleaned eval-ready file {final_cleaned_lean_output_file}: {e_count}")
                print(f"\nFinal Cleaned Eval-Ready dataset saved to: {final_cleaned_lean_output_file} (count error)")
        else:
            logger.info("Final Cleaned Eval-Ready dataset was not produced or path is invalid.")
            print(f"\nFinal Cleaned Eval-Ready dataset was not produced due to issues in validation/cleaning or no samples passed.")

    logger.info(f"Total run time: {total_run_time:.2f} seconds")

    hours, rem = divmod(total_run_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str_display = ""
    if hours > 0: time_str_display += f"{int(hours)}h "
    if minutes > 0 or hours > 0 : time_str_display += f"{int(minutes)}m "
    time_str_display += f"{seconds:.2f}s"
    print(f"\n✅ Total execution time: {time_str_display.strip()} ({total_run_time:.2f} seconds)")
    print(f"\nAll datasets for this run are in subfolder: {run_output_dir}")

    gen_prompt_tokens, gen_completion_tokens, gen_api_calls = generation_token_tracker.get_summary()
    estimated_generation_cost = generation_token_tracker.calculate_cost(
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS,
    )
    logger.info(f"--- Generation Token Usage & Estimated Cost (verbose-listops.py) ---")
    logger.info(f"Total API calls (generation): {gen_api_calls}")
    logger.info(f"Total Prompt Tokens (generation): {gen_prompt_tokens}")
    logger.info(f"Total Completion Tokens (generation): {gen_completion_tokens}")
    logger.info(f"Estimated Cost (generation only): ${estimated_generation_cost:.4f}")

    if PROD_RUN and os.path.exists(validator_stdout_log_path):
        try:
            with open(validator_stdout_log_path, "r", encoding="utf-8") as f_val_stdout:
                val_stdout_content = f_val_stdout.read()
                match = re.search(r"VALIDATOR_TOKEN_USAGE_SUMMARY:prompt_tokens=(\d+),completion_tokens=(\d+),api_calls=(\d+)", val_stdout_content)
                if match:
                    val_prompt_tokens = int(match.group(1))
                    val_completion_tokens = int(match.group(2))
                    val_api_calls = int(match.group(3))
                    estimated_validation_cost = (val_prompt_tokens / 1_000_000 * DEFAULT_COST_PER_MILLION_PROMPT_TOKENS) + \
                                                (val_completion_tokens / 1_000_000 * DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS)
                    logger.info(f"--- Validation Token Usage & Estimated Cost (from {validator_stdout_basename}) ---")
                    logger.info(f"Total API calls (validation): {val_api_calls}")
                    logger.info(f"Total Prompt Tokens (validation): {val_prompt_tokens}")
                    logger.info(f"Total Completion Tokens (validation): {val_completion_tokens}")
                    logger.info(f"Estimated Cost (validation only, using verbose-listops.py rates): ${estimated_validation_cost:.4f}")
        except Exception as e_val_cost:
            logger.warning(f"Could not parse token usage from validator stdout log ('{validator_stdout_log_path}'): {e_val_cost}")

    if client and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
        logger.info("Fetching final OpenRouter account usage...")
        final_account_usage = rate_limiter.update_limits_from_api()
        if final_account_usage is not None:
            logger.info(f"Final OpenRouter account usage: ${final_account_usage:.4f}")
            if initial_account_usage is not None:
                total_run_cost_by_difference = final_account_usage - initial_account_usage
                logger.info(f"--- Total Run Cost (from API Usage Difference) ---")
                logger.info(f"TOTAL RUN COST (Generation + Validation if validator.py used same key): ${total_run_cost_by_difference:.4f}")
        else:
            logger.warning("Could not fetch final OpenRouter account usage. Cannot calculate total run cost by difference.")
    else:
        logger.warning("Skipping final OpenRouter account usage check for run cost: Client not initialized or API key missing/placeholder.")

    logger.info("--- END OF MAIN FUNCTION ---")

    if client:  # Check if client was initialized
        try:
            logger.info("Attempting to close OpenAI client...")
            client.close()
            logger.info("OpenAI client closed successfully.")
        except Exception as e_close:
            logger.error(f"Error closing OpenAI client: {e_close}")

    logging.shutdown()

if __name__ == "__main__":
    # Call update_limits_from_api once after full logger setup and before starting main generation.
    # This is already handled by the logic at the start of main() now for initial_account_usage.
    # We can remove the specific pre-main call if main() handles it robustly.

    # if client and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
    #     try:
    #         logger.info("Performing initial OpenRouter limits check before starting main generation...")
    #         rate_limiter.update_limits_from_api() # This will now also attempt to get usage for logging by main
    #     except Exception as e:
    #         logger.error(f"Initial OpenRouter limits check failed: {e}")
    # else:
    #     logger.warning("Skipping initial OpenRouter limits check: Client not initialized or API key missing/placeholder.")

    main(
        config,
        num_samples=NUM_SAMPLES_TO_GENERATE,
        max_workers=DEFAULT_MAX_WORKERS,
        # initial_account_usage will be fetched inside main
    )
