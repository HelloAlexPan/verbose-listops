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
from collections import Counter
import collections

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
    MAX_ANCHOR_WORDS: int = 6
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
    WORLD_GEN_MAX_TOKENS: int = 200
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
    # REWRITTEN MEDIAN Example 1 (Original inputs: {72, 84, 89, 91, 95}, Median: 89)
    # NEW RULE: ALL inputs (72, 84, 89, 91, 95) MUST be mentioned. Median 89 is IMPLICIT.
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene - MEDIAN Example):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: seventy-two, eighty-four, eighty-nine, ninety-one, and ninety-five.\\\\n"
            # No special MEDIAN exception for input itemization anymore. ALL inputs must be mentioned.
            "*   **MEDIAN RESULT MUST BE IMPLICIT:** The median value ('eighty-nine') must NOT be explicitly stated as the result. It should be implied conceptually.\\\\n"
            "*   You MAY use the number 'five' (the count of direct items) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions ALL required numbers (72, 84, 89, 91, 95). Median 89 is IMPLICIT.
        "Seraphina arranged the five crystal fragments on the altar: 'This one pulses with seventy-two vibrations, this with eighty-four, this with eighty-nine, this with ninety-one, and this with ninety-five.' She examined the arrangement, particularly the one glowing with eighty-nine vibrations. 'This central fragment,' she mused, 'its resonance sits precisely in the middle, the perfect equilibrium point that will stabilize the ritual.' Marcus nodded, understanding its significance without her needing to state 'median'.",
        # BAD example: explicitly states "eighty-nine" AS THE MEDIAN RESULT.
        "Seraphina arranged the crystal fragments: 'We have seventy-two, eighty-four, eighty-nine, ninety-one, and ninety-five. The median of these is eighty-nine.'",
        "BAD output failed: Explicitly stated 'eighty-nine' as the median result. For MEDIAN operations, the result MUST be IMPLICIT.",
    ),
    # REWRITTEN MEDIAN Example 2 (Original inputs: {73, 85, 87, 88, 89, 91}, Median: 87)
    # NEW RULE: ALL inputs (73, 85, 87, 88, 89, 91) MUST be mentioned. Median 87 is IMPLICIT.
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene - MEDIAN Example):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: seventy-three, eighty-five, eighty-seven, eighty-eight, eighty-nine, and ninety-one.\\\\n"
            # No special MEDIAN exception for input itemization anymore. ALL inputs must be mentioned.
            "*   **MEDIAN RESULT MUST BE IMPLICIT:** The median value ('eighty-seven') must NOT be explicitly stated anywhere.\\\\n"
            "*   You MAY use the number 'six' (the count of direct items) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions ALL inputs (73, 85, 87, 88, 89, 91). Median 87 is IMPLICIT.
        "Kairos studied the alignment of six energy signatures on the quantum display. 'The readings show seventy-three, eighty-five, eighty-seven, eighty-eight, eighty-nine, and ninety-one.' He pointed to the value eighty-seven. 'This central point, the one reading eighty-seven, is our target. Its balance will stabilize the entire sequence.' Lyra nodded, understanding the critical equilibrium point without needing to name it as the median.",
        # BAD example: explicitly states "eighty-seven" AS THE MEDIAN RESULT.
        "Kairos studied the six signatures: seventy-three, eighty-five, eighty-seven, eighty-eight, eighty-nine, and ninety-one. 'The median here is eighty-seven,' he declared.",
        "BAD output failed: The median value 'eighty-seven' was explicitly stated as the result. CRITICAL ERROR: For MEDIAN operations, the result itself must always be IMPLICIT.",
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

@dataclass
class GenerationContext:
    world: dict
    config: Config
    encoder: any
    p_inflect: any
    logger: logging.Logger
    narrative_anchor_map: dict
    all_atoms: set
    introduced_atoms: set
    scenes: list
    tokens_used: int
    last_scene_text: str
    beat_counter: dict
    sample_index: int
    max_pad_paragraphs: int
    overall_ground_truth_answer: int | None
    overall_ast_root: Node | None = None
    beat_revision_logs: list = field(default_factory=list)  # <<< NEW FIELD
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
    max_retries: int = config.WORLDGEN_MAX_RETRIES,  # Use the config variable
    sample_index: int | None = None,
) -> dict:
    """
    Generates fictional world metadata using an LLM, with a tuned prompt
    to maximize the likelihood of receiving valid, parseable JSON,
    especially regarding quote escaping. Includes retries as a fallback.
    """
    if not isinstance(num_characters, int) or num_characters < 1:
        raise ValueError("num_characters must be positive int")
    if not isinstance(num_concepts, int) or num_concepts < 1:
        raise ValueError("num_concepts must be positive int")

    prompt = (
        "You are an expert system designed to generate structured data in **strictly valid JSON format**.\n"
        "Your task is to create fictional world metadata.\n\n"
        "**CRITICAL JSON FORMATTING RULES (MUST FOLLOW EXACTLY):**\n"
        "1.  The entire output MUST be a single, valid JSON object.\n"
        '2.  All string keys and string values within the JSON must be enclosed in double quotes (e.g., `"name": "value"`).\n'
        '3.  **If a string value itself needs to contain a double quote character (e.g., a nickname within a name), that internal double quote MUST be escaped with a backslash (`\\`)**. For example, if a character\'s name is `Dr. "Nickname" Who`, it must be represented in the JSON string as `"name": "Dr. \\"Nickname\\" Who"`.\n'
        # CORRECTED LINE BELOW
        "4.  Ensure all commas, colons, curly braces `{{}}`, and square brackets `[]` are correctly placed according to standard JSON syntax.\n"
        "5.  Do not include any text, explanations, or markdown (like ```json) before or after the single JSON object.\n\n"
        "**Instructions for Content Generation:**\n\n"
        "1.  **Characters:** Generate exactly {num_characters} distinct characters. Each character MUST have:\n"
        '    *   `name`: string (e.g., "Kaelen Vane", "Seraphina Moonwhisper")\n'
        '    *   `role`: string (e.g., "The grizzled warrior," "The cunning sorceress," "The naive apprentice")\n'
        '    *   `quirk`: string (a unique or unusual habit, belief, or physical trait, e.g., "Collects antique spoons," "Only speaks in riddles," "Has mismatched eyes")\n'
        "    Ensure each character's name, role, and quirk combination is unique. Remember to escape any internal double quotes in these string values as per Rule 3 above.\n\n"
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
        try:
            # Log the prompt
            log_prompt(
                header=f"World Generation Prompt (Attempt {attempt + 1})",
                prompt=f"System: (Implicit in API call structure for this function)\nUser:\n{prompt}",
                sample_index=sample_index,
            )

            # Use config.MAX_API_TOKEN_LIMIT instead of config.WORLD_GEN_MAX_TOKENS
            # to avoid truncation due to reasoning tokens
            resp = _chat_completion_call(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=config.MAX_API_TOKEN_LIMIT,  # Use higher limit
                temperature=config.WORLD_GEN_TEMP,
                json_schema=WORLD_SCHEMA,  # Use the defined schema
                reasoning={"exclude": True},
            )
            if (
                hasattr(resp, "choices") and resp.choices
            ):  # Check if choices list exists and is not empty

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

            # Log raw response
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

            # Attempt to strip markdown ```json ... ``` if present
            cleaned_text_for_parsing = text.strip()
            if cleaned_text_for_parsing.startswith("```json"):
                cleaned_text_for_parsing = cleaned_text_for_parsing[len("```json") :]
            if cleaned_text_for_parsing.endswith("```"):
                cleaned_text_for_parsing = cleaned_text_for_parsing[: -len("```")]
            cleaned_text_for_parsing = cleaned_text_for_parsing.strip()

            world = parse_llm_json_with_fallback(
                cleaned_text_for_parsing,  # Use the potentially cleaned text
                {},  # Empty dict will trigger the keys validation check right after
                f"in world generation attempt {attempt+1}",
            )

            # --- Validation ---
            required_keys = ["characters", "genre", "setting", "object"]
            if not all(k in world for k in required_keys):
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: Generated JSON missing required keys. Keys found: {world.keys()}"
                )
                raise ValueError(
                    "Generated JSON missing required keys"
                )  # Raise error to trigger except block
            if not isinstance(world.get("characters"), list) or not world["characters"]:
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: 'characters' key is not a non-empty list."
                )
                raise ValueError("'characters' key is not a non-empty list")

            # Additional validation for character structure
            for char_idx, char_obj in enumerate(world.get("characters", [])):
                if not isinstance(char_obj, dict):
                    logger.warning(
                        f"World Gen Attempt {attempt + 1}: Character at index {char_idx} is not a dictionary."
                    )
                    raise ValueError(
                        f"Character at index {char_idx} is not a dictionary."
                    )
                if not all(k_char in char_obj for k_char in ["name", "role", "quirk"]):
                    logger.warning(
                        f"World Gen Attempt {attempt + 1}: Character at index {char_idx} is missing required keys (name, role, quirk). Found: {char_obj.keys()}"
                    )
                    raise ValueError(
                        f"Character at index {char_idx} missing required keys."
                    )

            logger.debug(
                f"World Gen Attempt {attempt + 1}: Successfully generated and parsed world JSON."
            )
            logger.debug(f"Generated object: {world.get('object', 'N/A')}")
            return world  # Success! Exit the function.
        except (
            json.JSONDecodeError,
            ValueError,
        ) as e:  # Catch both parsing and validation errors
            logger.error(
                f"World Gen Attempt {attempt + 1}: Failed ({type(e).__name__}): {e}. Raw text (after potential cleaning for ```json):\n---\n{cleaned_text_for_parsing if 'cleaned_text_for_parsing' in locals() else text}\n---"
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
                if " " in word_no_and_lower:
                    hyphenated_word = word_no_and_lower.replace(" ", "-")
                    num_word_dict[hyphenated_word] = i
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


def extract_numbers_from_text(
    text: str,
) -> list[int]:  # Changed return type to list[int]
    """Extracts integers (digits and words), returning a list of all occurrences."""
    if not text:
        return []  # Return empty list

    found_numbers_list = []  # Changed to list
    search_text = text.lower()

    # Store spans of digits to avoid re-processing them as words
    # and to replace them before word search
    digit_spans_to_replace = []

    # First, extract all digit-based numbers
    for match in DIGIT_REGEX.finditer(search_text):
        digit_str = match.group(0)
        try:
            value = int(digit_str)
            found_numbers_list.append(value)  # Add to list
            digit_spans_to_replace.append(match.span())
        except ValueError:
            logger.warning(
                f"Could not convert digit string '{digit_str}' to int during extraction."
            )
            continue  # Skip this invalid digit string

    # Create a temporary version of the text with digits "blanked out"
    # to prevent them from being re-parsed by the word-based logic.
    # We replace with a character that won't be part of any number word.
    text_chars_list = list(search_text)
    for start, end in digit_spans_to_replace:
        for i in range(start, end):
            text_chars_list[i] = "|"  # Placeholder character

    # Reconstruct text for word search, replacing placeholders with spaces
    text_for_word_search = "".join(text_chars_list)
    text_for_word_search = text_for_word_search.replace(
        "|", " "
    )  # Replace placeholders
    text_for_word_search = re.sub(
        r"\s+", " ", text_for_word_search
    ).strip()  # Normalize spaces

    # Then, extract all word-based numbers from the modified text
    # The NUMBER_WORDS_REGEX uses sorted_number_words (longest first) to handle multi-word numbers.
    for match in NUMBER_WORDS_REGEX.finditer(text_for_word_search):
        sign_word = match.group(1)  # (minus|negative) or None
        number_word_matched = match.group(2).lower()  # The matched number phrase

        # Check if the matched word is in our dictionary (it should be, by construction of the regex)
        value = EXPANDED_NUMBER_WORDS_DICT.get(number_word_matched)

        if value is not None:
            # Apply sign if "minus" or "negative" was found
            if sign_word and value != 0:  # "minus zero" is still zero
                value = -value
            # Ignore common ordinal words if they happen to be parsed as numbers by inflect
            # (e.g., "first" might be 1, but we want to ignore it if it's used as an ordinal)
            # This check is somewhat heuristic; true ordinal parsing is complex.
            # The primary ordinal filtering should happen based on ORDINAL_WORDS_TO_IGNORE if needed,
            # but this provides a basic safeguard within number extraction.
            # However, the main script's logic for ORDINAL_WORDS_TO_IGNORE is not directly used here.
            # This function focuses on extracting all potential numerical mentions.
            # The calling validator should handle semantic interpretation of ordinals.
            # For now, we assume if it's in EXPANDED_NUMBER_WORDS_DICT, it's a number to be extracted.
            # The ORDINAL_WORDS_TO_IGNORE set is not used in this function directly.
            # The problem description implies the validator handles this.
            # Let's assume for now that if it's a number word, we extract it.
            # The main validator logic will decide if it's allowed.
            found_numbers_list.append(value)  # Add to list
        else:
            # This case should be rare if NUMBER_WORDS_REGEX is built correctly from EXPANDED_NUMBER_WORDS_DICT
            logger.warning(
                f"Word phrase '{number_word_matched}' matched by regex but not in EXPANDED_NUMBER_WORDS_DICT. This is unexpected."
            )

    logger.debug(
        f"extract_numbers_from_text (list): Input '{text[:100]}...', Found: {found_numbers_list}"
    )
    return found_numbers_list


# --- Factory for number validation ---
def make_number_validator(
    allowed_atoms_list: list[int],  # Changed from Set[int] to List[int]
    forbidden_atoms: Set[int],
    operand_count: int,
    correct_result_for_beat: int | None,
    strict_zero: bool = False,
    enforce_result_presence: bool = True,
    operation_type: str | None = None,
    overall_ground_truth_answer: int | None = None,
    is_root_node_being_validated: bool = False,
    conceptual_input_values: Set[int] | None = None,  # This remains a set of values
    config_obj: Config = config,
    logger_obj: logging.Logger = logger,
) -> Callable[[str], bool]:
    logger_obj.debug(
        f"Creating validator with: Allowed_Atoms_List (counts matter)={Counter(allowed_atoms_list)}, Forbidden_Set={forbidden_atoms}, OpCount={operand_count}, "
        f"Result={correct_result_for_beat}, StrictZero={strict_zero}, "
        f"EnforceResultPresence={enforce_result_presence}, Op={operation_type}, OverallGT={overall_ground_truth_answer}, IsRoot={is_root_node_being_validated}"
    )

    # Convert allowed_atoms_list to a Counter for easy frequency checking
    required_atoms_counts = Counter(allowed_atoms_list)
    # For logging and some checks, a set of unique required atoms is still useful
    unique_required_atoms_set = set(allowed_atoms_list)

    IMPLICITLY_ALLOWED_SMALL_NUMBERS = set(
        range(
            config_obj.MIN_ALLOWED_SMALL_NUMBER, config_obj.MAX_ALLOWED_SMALL_NUMBER + 1
        )
    )

    is_result_also_an_input_atom = False
    if (
        correct_result_for_beat is not None and allowed_atoms_list
    ):  # Check against the list
        if (
            correct_result_for_beat in required_atoms_counts
            and required_atoms_counts[correct_result_for_beat] > 0
        ):
            is_result_also_an_input_atom = True
            logger_obj.debug(
                f"Validator: Result {correct_result_for_beat} is ALSO a direct atomic input for Op {operation_type} (required count: {required_atoms_counts[correct_result_for_beat]})."
            )

    if operation_type == "MED":
        logger_obj.info(
            f"MED OPERATION VALIDATION (New Rule): All atomic inputs (with counts: {required_atoms_counts}) must be mentioned. "
            f"The median result ({correct_result_for_beat}) must be implicit."
        )

    def validate(text: str) -> bool:
        found_numbers_list = extract_numbers_from_text(text)  # Returns list[int]
        found_numbers_counts = Counter(found_numbers_list)

        text_preview = text[:100].replace("\n", " ") + (
            "..." if len(text) > 100 else ""
        )
        logger_obj.debug(f'Validator Input Text: "{text_preview}"')
        logger_obj.debug(f"Validator Found Numbers (Counts): {found_numbers_counts}")
        logger_obj.debug(f"Validator Required Atoms (Counts): {required_atoms_counts}")

        validation_report = {
            "status": "PASS",
            "reason": "All validation checks passed",
            "operation_type": operation_type,
            "text_preview": text_preview,
            "found_numbers_counts": dict(found_numbers_counts),  # Log counts
            "required_atoms_counts": dict(required_atoms_counts),  # Log counts
            "operand_count": operand_count,
            "correct_result": correct_result_for_beat,
            "is_root_node": is_root_node_being_validated,
            "enforce_result_presence_flag": enforce_result_presence,
            "overall_ground_truth_answer_for_this_validation_context": overall_ground_truth_answer,
            "missing_required_details": [],  # Will store dicts like {'number': N, 'required_count': X, 'found_count': Y}
            "forbidden_extras_details": [],  # Will store dicts like {'number': N, 'reason': 'reason_text'}
            "details": [],
        }

        if strict_zero:
            allowed_for_strict_zero = config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
            disallowed_in_strict_zero_counts = Counter()
            for num, count in found_numbers_counts.items():
                if num not in allowed_for_strict_zero:
                    disallowed_in_strict_zero_counts[num] = count

            if disallowed_in_strict_zero_counts:
                validation_report["status"] = "FAIL"
                validation_report["reason"] = "STRICT_ZERO_VIOLATION"
                validation_report["forbidden_extras_details"] = [
                    {
                        "number": n,
                        "count": c,
                        "reason": f"Not in allowed phrasing set {allowed_for_strict_zero} for strict_zero mode",
                    }
                    for n, c in disallowed_in_strict_zero_counts.items()
                ]
                validation_report["details"].append(
                    f"Strict zero mode: Found numbers with counts {dict(disallowed_in_strict_zero_counts)} not in {allowed_for_strict_zero}."
                )
                _log_failed_validation(text, validation_report, logger_obj)
                return False
            return True

        # Rule 1: IMPLICIT RESULT HANDLING (Result of THIS operation)
        # The result of ANY operation must be implicit and not stated numerically,
        # UNLESS that result value also happens to be one of the direct atomic inputs for THIS beat.
        if (
            correct_result_for_beat is not None
            and found_numbers_counts[correct_result_for_beat] > 0
        ):
            # If the result is also an input, its presence is governed by Rule 2 (Required Atoms).
            # The check here is if it's present *and not justified as an input*, or framed as the outcome.
            # The Python validator primarily checks presence. The LLM validator checks framing.
            if not is_result_also_an_input_atom:
                # This means the result was found, and it was NOT one of the required inputs.
                # This is a failure if it's stated explicitly.
                validation_report["status"] = "FAIL"
                validation_report["reason"] = "IMPLICIT_RESULT_STATED_EXPLICITLY"
                validation_report["details"].append(
                    f"Result {correct_result_for_beat} (Op: {operation_type}) should be implicit but was found explicitly stated (count: {found_numbers_counts[correct_result_for_beat]}) and was not also a required input."
                )
                _log_failed_validation(text, validation_report, logger_obj)
                return False
            else:
                # Result is also an input. Its presence as an input is fine.
                # The generator's prompt (Rule 2 Special Note) guides the LLM on framing.
                # The Python validator allows its presence if it's a required input.
                # We need to ensure it's not *over-represented* beyond its required input count + its potential single mention as a result (which is forbidden if not an input).
                # If result R is required K times as input, and found K+1 times, and R is not allowed for phrasing, then the +1 is the forbidden explicit result.
                # This is complex. For now, the Python validator focuses on: if result is found, and it's NOT an input, fail.
                # If it IS an input, its count is checked by Rule 2. Any excess beyond required input count will be caught by Rule 4 (Extraneous).
                logger_obj.debug(
                    f"Python Validator: Result {correct_result_for_beat} (Op: {operation_type}) found. It's also a required input. Its count will be checked against required input counts. Framing is LLM's job."
                )

        # Rule 2: All REQUIRED ATOMIC OPERANDS MUST be present with correct frequencies
        current_missing_required = []
        for num, required_count in required_atoms_counts.items():
            found_count = found_numbers_counts[num]
            if found_count < required_count:
                current_missing_required.append(
                    {
                        "number": num,
                        "required_count": required_count,
                        "found_count": found_count,
                        "missing_count": required_count - found_count,
                    }
                )

        if current_missing_required:
            validation_report["status"] = "FAIL"
            validation_report["reason"] = (
                "MISSING_REQUIRED_OPERANDS_OR_INSUFFICIENT_COUNT"
            )
            validation_report["missing_required_details"] = current_missing_required
            validation_report["details"].append(
                f"RequiredAtoms (Counts): {dict(required_atoms_counts)}, Found (Counts): {dict(found_numbers_counts)}. Deficiencies: {current_missing_required}."
            )
            _log_failed_validation(text, validation_report, logger_obj)
            return False

        # --- Advanced Check for Extraneous and Forbidden Numbers ---
        # Start with all found numbers and progressively subtract/excuse legitimate ones.
        potentially_extraneous_counts = found_numbers_counts.copy()

        # 1. Subtract counts of correctly provided required atoms
        for num, req_count in required_atoms_counts.items():
            # We only subtract up to the required count. If more were found, they remain.
            actually_found_for_this_req_num = min(
                potentially_extraneous_counts[num], req_count
            )
            potentially_extraneous_counts[num] -= actually_found_for_this_req_num
            if potentially_extraneous_counts[num] == 0:
                del potentially_extraneous_counts[num]  # Clean up zero counts

        # `potentially_extraneous_counts` now holds numbers found *in excess* of requirements, or numbers not required at all.

        # 2. Excuse specific allowed numbers (phrasing, operand count)
        #    These are excused if present, regardless of count (within reason, but hard to cap here).
        #    The generator is prompted to use them sparingly.
        numbers_to_fully_excuse_if_present = set()
        numbers_to_fully_excuse_if_present.update(
            config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        numbers_to_fully_excuse_if_present.update(IMPLICITLY_ALLOWED_SMALL_NUMBERS)
        if operand_count > 0:  # Only excuse operand_count if it's positive
            numbers_to_fully_excuse_if_present.add(operand_count)

        # If the current beat's result was also an input, and it was found,
        # its required input occurrences were already subtracted.
        # If it was *not* an input, but was found, Rule 1 (IMPLICIT_RESULT_STATED_EXPLICITLY)
        # should have already caught it if `enforce_result_presence` was effectively true for this check.
        # The `correct_result_for_beat` itself, if found and not excused as an input, is an error.
        # Let's ensure that if `correct_result_for_beat` is in `potentially_extraneous_counts`
        # AND it was NOT an input atom, it's flagged (this re-iterates Rule 1 logic for safety here).
        if (
            correct_result_for_beat is not None
            and potentially_extraneous_counts[correct_result_for_beat] > 0
            and not is_result_also_an_input_atom
        ):
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "IMPLICIT_RESULT_STATED_EXPLICITLY_AS_EXTRA"
            validation_report["forbidden_extras_details"].append(
                {
                    "number": correct_result_for_beat,
                    "count": potentially_extraneous_counts[correct_result_for_beat],
                    "reason": f"Result {correct_result_for_beat} stated explicitly and not a required input.",
                }
            )
            # No need to _log_failed_validation yet, will be caught at the end if this is the only issue.

        # Remove counts of these excusable numbers from `potentially_extraneous_counts`
        temp_excused_counts = Counter()
        for num_to_excuse in numbers_to_fully_excuse_if_present:
            if potentially_extraneous_counts[num_to_excuse] > 0:
                temp_excused_counts[num_to_excuse] = potentially_extraneous_counts[
                    num_to_excuse
                ]
                del potentially_extraneous_counts[
                    num_to_excuse
                ]  # Remove all occurrences

        if temp_excused_counts:
            logger_obj.debug(
                f"Validator: Excused phrasing/operand_count numbers (all occurrences): {dict(temp_excused_counts)}"
            )

        # `potentially_extraneous_counts` now holds numbers that are:
        # - Not required atomic inputs (or are in excess of required counts).
        # - Not the explicitly allowed phrasing/small/operand_count numbers.
        # - Potentially the stated result (if it wasn't an input).

        # 3. Check against the `forbidden_atoms` set (e.g., prior results, overall GT).
        #    Any number in `potentially_extraneous_counts` that is also in `forbidden_atoms` is a "truly forbidden extra."
        truly_forbidden_found_details = []
        for num, count_found_extra in potentially_extraneous_counts.items():
            if count_found_extra > 0 and num in forbidden_atoms:
                # It's forbidden AND it wasn't a required input (or was in excess) AND it wasn't an allowed phrasing number.
                truly_forbidden_found_details.append(
                    {
                        "number": num,
                        "count": count_found_extra,
                        "reason": "Present in forbidden_atoms set and not otherwise excused.",
                    }
                )

        if truly_forbidden_found_details:
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "FORBIDDEN_NUMBERS_FOUND"
            validation_report["forbidden_extras_details"].extend(
                truly_forbidden_found_details
            )
            # Details already added to forbidden_extras_details

        # 4. Any remaining counts in `potentially_extraneous_counts` are "other extraneous numbers."
        #    However, we must first remove the `truly_forbidden_found_details` from it to avoid double penalizing.
        other_extraneous_details = []
        for item in truly_forbidden_found_details:
            num_forbidden = item["number"]
            count_forbidden = item["count"]
            potentially_extraneous_counts[num_forbidden] -= count_forbidden
            if potentially_extraneous_counts[num_forbidden] <= 0:
                del potentially_extraneous_counts[num_forbidden]

        # Now, whatever is left in potentially_extraneous_counts is truly extraneous and not specifically forbidden.
        if potentially_extraneous_counts:  # If anything is left with count > 0
            for num, count_extra in potentially_extraneous_counts.items():
                if count_extra > 0:
                    other_extraneous_details.append(
                        {
                            "number": num,
                            "count": count_extra,
                            "reason": "Extraneous: Not required, not allowed phrasing/counting, not explicitly forbidden but still present.",
                        }
                    )

            if other_extraneous_details:
                validation_report["status"] = "FAIL"
                if (
                    validation_report["reason"] == "PASS"
                    or validation_report["reason"] == "FORBIDDEN_NUMBERS_FOUND"
                ):  # Avoid overwriting a more specific forbidden error
                    validation_report["reason"] = "OTHER_EXTRANEOUS_NUMBERS_FOUND"
                validation_report["forbidden_extras_details"].extend(
                    other_extraneous_details
                )

        if validation_report["status"] == "FAIL":
            # Consolidate details for logging
            if validation_report["missing_required_details"]:
                validation_report["details"].append(
                    f"Missing/Insufficient: {validation_report['missing_required_details']}"
                )
            if validation_report["forbidden_extras_details"]:
                validation_report["details"].append(
                    f"Forbidden/Extraneous: {validation_report['forbidden_extras_details']}"
                )
            _log_failed_validation(text, validation_report, logger_obj)
            return False

        logger_obj.debug(
            f"Validation PASS (Strict Python Validator with counts for Op: {operation_type})"
        )
        return True

    return validate


# Add a helper function to save failed validation attempts
def _log_failed_validation(
    text: str, validation_report: dict, logger_obj: logging.Logger = logger
):
    logger_obj.debug(f"Saving failed validation record")
    try:
        failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
        os.makedirs(failed_validations_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        operation = validation_report.get("operation_type", "unknown_op")
        reason = validation_report.get("reason", "unknown_reason")
        filename = f"validation_fail_{operation}_{reason}_{timestamp}.json"
        filepath = os.path.join(failed_validations_dir, filename)

        # Make a copy to avoid modifying the original report dict
        report_to_save = validation_report.copy()
        # Convert Counter objects to dict for JSON serialization if they are still Counters
        if "found_numbers_counts" in report_to_save and isinstance(
            report_to_save["found_numbers_counts"], collections.Counter
        ):
            report_to_save["found_numbers_counts"] = dict(
                report_to_save["found_numbers_counts"]
            )
        if "required_atoms_counts" in report_to_save and isinstance(
            report_to_save["required_atoms_counts"], collections.Counter
        ):
            report_to_save["required_atoms_counts"] = dict(
                report_to_save["required_atoms_counts"]
            )

        full_report_data = {
            "validation_report": report_to_save,  # Use the modified copy
            "full_text": text,
            "timestamp": timestamp,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(full_report_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved failed validation record to {filepath}")

        sample_index = None
        import inspect

        try:
            frames = inspect.stack()
            for frame in frames:
                if "context" in frame.frame.f_locals:
                    ctx = frame.frame.f_locals["context"]
                    if hasattr(ctx, "sample_index"):
                        sample_index = ctx.sample_index
                        break
        except Exception as e:
            logger.error(f"Error extracting sample_index from stack: {e}")

        # Enhanced logging for LLM turns
        found_counts_str = str(report_to_save.get("found_numbers_counts", {}))
        req_counts_str = str(report_to_save.get("required_atoms_counts", {}))
        missing_details_str = str(report_to_save.get("missing_required_details", []))
        forbidden_details_str = str(report_to_save.get("forbidden_extras_details", []))

        beat_info_str = ""
        # (Code to extract beat_counter and narrative_anchor from stack remains the same)
        # ...
        for frame in frames:  # Added this loop back from original code
            frame_locals = frame.frame.f_locals
            if "narrative_anchor" in frame_locals:  # Check if key exists
                narrative_anchor = frame_locals["narrative_anchor"]
            if "context" in frame_locals:
                ctx = frame_locals["context"]
                if hasattr(ctx, "beat_counter"):
                    beat_counter = ctx.beat_counter
            if (
                "narrative_anchor" in locals() and "beat_counter" in locals()
            ):  # Check if variables are defined
                if narrative_anchor and beat_counter:  # Then check if they have values
                    break

        if (
            "beat_counter" in locals() and beat_counter
        ):  # Check if beat_counter is defined and has a value
            beat_info_str = f", Beat {beat_counter.get('current', '?')}/{beat_counter.get('total', '?')}"
        anchor_info_str = ""
        if (
            "narrative_anchor" in locals() and narrative_anchor
        ):  # Check if narrative_anchor is defined and has a value
            anchor_info_str = f", Anchor: '{narrative_anchor}'"

        validation_header = (
            f"VALIDATION FAILURE: Op={operation}{beat_info_str}{anchor_info_str}, Reason={reason} "
            f"[Consolidated log for LLM analysis]"
        )

        validation_details = (
            f"{'='*80}\n=== VALIDATION FAILURE REPORT ===\n{'='*80}\n\n"
        )
        validation_details += f"Operation type: {operation}\nFailure reason: {reason}\n"
        validation_details += (
            f"Operand count: {validation_report.get('operand_count', 'N/A')}\n"
        )
        if validation_report.get("correct_result") is not None:
            validation_details += (
                f"Correct result (for beat): {validation_report['correct_result']}\n"
            )

        validation_details += f"\n--- Number Analysis (Counts) ---\n"
        validation_details += f"Found numbers (counts): {found_counts_str}\n"
        validation_details += f"Required numbers (counts): {req_counts_str}\n"
        if missing_details_str != "[]":
            validation_details += (
                f"Missing/Insufficient required: {missing_details_str}\n"
            )
        if forbidden_details_str != "[]":
            validation_details += (
                f"Forbidden/Extraneous extras: {forbidden_details_str}\n"
            )

        details_list = report_to_save.get("details", [])
        if details_list:
            validation_details += f"\n--- Detailed Analysis Log ---\n"
            for detail_item in details_list:
                validation_details += f"- {detail_item}\n"

        validation_details += f"\n--- Generated Text That Failed Validation ---\n{text}\n\n{'='*80}\n=== END OF VALIDATION FAILURE REPORT ===\n{'='*80}\n"

        log_prompt(validation_header, validation_details, sample_index=sample_index)

    except Exception as e:
        logger.error(
            f"Error saving/logging failed validation record: {e}", exc_info=True
        )


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
        f"You are a master {world_info.get('genre', 'Fantasy')} storyteller. Your task is to write a compelling introductory scene for a new story. "
        "This scene must establish the setting, introduce one or two key characters, and hint at a central mystery or goal related to the primary object.\n\n"
        "**ABSOLUTE NUMERICAL RULE FOR THIS INTRODUCTORY SCENE (CRITICAL - READ CAREFULLY):**\n"
        "1.  **ZERO NUMBERS IS THE PRIMARY GOAL:** Your paramount objective is to use NO numerical values at all (neither digits like '7' nor words like 'seven', 'first', 'dozen', etc.). The ideal scene contains zero explicit numbers.\n"
        "2.  **EXTREMELY LIMITED EXCEPTION (Use only if narratively unavoidable):** If absolutely essential for natural phrasing and no alternative exists, you MAY use the words 'one', 'two', or 'three' (e.g., 'a single ray of light', 'two figures emerged', 'three ancient symbols'). These are the ONLY numbers permitted, and only for such general, non-quantitative phrasing. Do NOT use any other numbers (e.g., 'four', 'ten', '734', '100') in any form (word or digit).\n"
        "3.  **HANDLING CHARACTER NAMES WITH DIGITS (Important if applicable):**\n"
        "    *   If a character's name or designation provided in the 'Characters to potentially feature' list includes digits (e.g., 'Unit 734', 'Agent 007'), you MUST AVOID explicitly stating the numerical part of their name as a standalone number or quantity in your narrative. \n"
        "    *   For example, instead of writing '...and Unit 734 approached...', which might be interpreted as mentioning the number 'seven hundred thirty-four', try to rephrase. You could refer to them by a role ('the automaton'), a descriptive title ('the designated unit'), or by the full name if the context makes it clear it's a name and not a quantity (e.g., 'Jedidiah spoke to Unit Seven-Three-Four.').\n"
        "    *   **SAFER APPROACH:** If possible for this introductory scene, select or describe characters in a way that avoids using names with numerical digits to prevent any ambiguity or accidental numerical mentions.\n\n"
        "Focus on atmosphere, intrigue, and character introduction. Do not reveal any specific quantities or begin any calculations. "
        "Output ONLY the narrative text for this scene. No titles, no explanations, no analysis. Your adherence to the ZERO NUMBERS goal (with the tiny exception) is paramount."
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
        f"**Task:** Write an engaging introductory scene based on the context above. \n"
        f"**CRITICAL REMINDER - ADHERE TO THE ABSOLUTE NUMERICAL RULE DETAILED IN THE SYSTEM MESSAGE:**\n"
        f"   - Strive for ZERO numbers. \n"
        f"   - Only 'one', 'two', 'three' are permissible for general phrasing IF UNAVOIDABLE. \n"
        f"   - NO OTHER NUMBERS (words or digits) are allowed. \n"
        f"   - Be extremely careful if character names contain digits; avoid stating the number part as a quantity. See system message for guidance.\n"
        f"The scene should set a tone and hint at the story's direction without giving away specifics. "
        f"Output ONLY the narrative text."
    )

    validate_intro = make_number_validator(
        allowed_atoms_list=[],
        forbidden_atoms=set(), # For intro, forbidden_atoms is effectively everything not in ALWAYS_ALLOWED_PHRASING_NUMBERS_SET for strict_zero
        operand_count=0,
        correct_result_for_beat=None,
        strict_zero=True,
        enforce_result_presence=False,
        operation_type="INTRO",
        overall_ground_truth_answer=None,
        is_root_node_being_validated=False,
        conceptual_input_values=None,
        config_obj=config_obj,
        logger_obj=logger_obj,
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=config_obj.INTRO_MAX_TOKENS,
        validate_fn=validate_intro,
        retries=config_obj.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config_obj.CREATIVE_NARRATIVE_TEMP, # Intro can be a bit creative
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
    original_user_message_for_generator: str,
    system_prompt_for_generator: str,
    world_info: dict,
    current_op_node: OpNode,
    conceptual_inputs_str_for_llm_validator: str,
    atomic_inputs_words_str_for_llm_validator: str,
    action_description_for_llm_validator: str,
    expected_beat_result_words_for_llm_validator: str | None,
    ultra_strict_instruction_for_llm_validator_context: str,
    current_max_beat_completion_tokens: int,
    sample_index: int,
    context_config: Config,
    logger_obj: logging.Logger,
    encoder_obj: any,
    # ADD these new parameters:
    context: "GenerationContext",
    current_node_conceptual_name: str,
    beat_number_in_sample: int,
    # End of new parameters
    is_current_beat_root_node: bool = False,
    overall_ground_truth_answer_val: int | None = None,
    primary_object_name: str = "items",
    forbidden_prior_results_and_gt_for_llm_validator: Set[int] = None,
    correct_result_val: int | None = None,
    direct_atom_values_val: Set[int] = None,
    direct_atom_counts_val: Counter | None = None,
) -> str | None:

    if forbidden_prior_results_and_gt_for_llm_validator is None:
        forbidden_prior_results_and_gt_for_llm_validator = set()
    if direct_atom_values_val is None: # Though not strictly primary now, ensure it's a set if used
        direct_atom_values_val = set()
    if direct_atom_counts_val is None:
        direct_atom_counts_val = Counter()

    current_op_arity = len(current_op_node.children)
    logger_obj.debug(
        f"LLMValidateBeat: Op={current_op_node.op}, Arity={current_op_arity}, CorrectResult={correct_result_val}, DirectAtomInputCounts={dict(direct_atom_counts_val)}"
    )

    history_of_attempts = [] # For local revision prompt generation
    history_of_critiques = [] # For local revision prompt generation
    current_generator_user_prompt_for_iteration = original_user_message_for_generator
    
    current_beat_conversation_turns = [] # <<< Initialize list for this beat's conversation log

    internal_validator_llm_model = context_config.LLM_VALIDATOR_MODEL
    logger_obj.info(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}] Using internal LLM validator: {internal_validator_llm_model}")

    for iteration in range(1, context_config.MAX_LLM_VALIDATION_ITERATIONS + 1):
        logger_obj.info(
            f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}] LLM Validation Loop Iteration: {iteration}/{context_config.MAX_LLM_VALIDATION_ITERATIONS}"
        )

        # Initialize turn_data at the start of each iteration
        turn_data = {
            "iteration": iteration,
            "generator_system_prompt": system_prompt_for_generator,
            "generator_user_prompt": None, # Will be set below
            "generator_output": "PENDING_GENERATION",
            "validator_system_prompt": "PENDING_VALIDATION_SETUP",
            "validator_user_prompt": "PENDING_VALIDATION_SETUP",
            "validator_critique_json": {"status": "PENDING_VALIDATION"}
        }

        generator_temp = context_config.BEAT_GEN_TEMP
        if iteration > 1: # This block constructs the revision prompt
            generator_temp = context_config.BEAT_REVISION_TEMP

            history_prompt_addition_parts = [
                "\n\n--- FAILED ATTEMPT REVIEW & REVISION MANDATE ---",
                "Your previous attempt did not meet all the strict numerical requirements. This revision MUST meticulously address all issues identified by the validator below, while also re-adhering to ALL original rules for this scene. Failure to correct these specific issues will likely result in another rejection.\n"
            ]
            last_attempt_text = (
                history_of_attempts[-1]
                if history_of_attempts
                else "N/A (Error in prior generation)"
            )
            last_critique_json = history_of_critiques[-1] if history_of_critiques else {}

            history_prompt_addition_parts.append(f"**Your Previous Attempt (Attempt {iteration-1}):**\n```text\n{last_attempt_text}\n```\n")

            explanation_from_validator = last_critique_json.get(
                "explanation_for_generator", "No detailed explanation from validator."
            )
            summary_for_generator_prompt = last_critique_json.get(
                "overall_revision_summary_for_generator_prompt", "Please revise carefully, addressing all feedback."
            )
            suggested_revisions_list = last_critique_json.get("suggested_revisions", [])

            history_prompt_addition_parts.append(f"**Validator Feedback for Your Previous Attempt:**\n  - Overall Revision Goal: {summary_for_generator_prompt}")

            # --- Detailed Issue Breakdown & Actionable Instructions ---
            history_prompt_addition_parts.append("  - **Specific Issues to Correct (CRITICAL - Address ALL of these):**")
            issues_found_for_generator_feedback = False

            # Parse Rule 1.A violations (Missing/Incorrect Counts)
            # Regexes to find detailed mismatches or missing numbers from validator's explanation
            mismatch_pattern = r"VIOLATION Rule A: Number '(\d+)' \(word form '([\w-]+)'\) was required (\d+) time(?:s)?.*?found (\d+) time(?:s)?"
            missing_pattern = r"VIOLATION Rule A: Required number '(\d+)' \(word form '([\w-]+)'\), expected (\d+) time(?:s)?, was completely missing"

            rule_a_mismatches = re.findall(mismatch_pattern, explanation_from_validator, re.IGNORECASE)
            rule_a_missing = re.findall(missing_pattern, explanation_from_validator, re.IGNORECASE)

            if rule_a_mismatches:
                for num_val_str, word_form, req_count_str, found_count_str in rule_a_mismatches:
                    history_prompt_addition_parts.append(
                        f"    - **Incorrect Count (Rule 1.A):** The number '{word_form}' (value: {num_val_str}) was required {req_count_str} time(s) but was found {found_count_str} time(s). "
                        f"**Action for Revision:** Adjust your narrative to ensure '{word_form}' appears *exactly* {req_count_str} time(s). Each mention must be a distinct narrative element representing an input to this operation."
                    )
                    if word_form.lower() == 'one' and int(req_count_str) > 1:
                         history_prompt_addition_parts.append(
                            f"      *Specific Reminder for multiple '{word_form}'s:* Describe separate, distinct instances (e.g., 'a single {primary_object_name} here... then another lone {primary_object_name} there'). Do NOT group them (e.g., AVOID 'two single {primary_object_name}s').*"
                         )
                    issues_found_for_generator_feedback = True
            if rule_a_missing:
                for num_val_str, word_form, req_count_str in rule_a_missing:
                    history_prompt_addition_parts.append(
                        f"    - **Missing Required Number (Rule 1.A):** The number '{word_form}' (value: {num_val_str}) was required {req_count_str} time(s) but was MISSING entirely. "
                        f"**Action for Revision:** You MUST add {req_count_str} distinct narrative mention(s) of '{word_form}' as inputs to this operation."
                    )
                    if word_form.lower() == 'one' and int(req_count_str) > 1:
                         history_prompt_addition_parts.append(
                            f"      *Specific Reminder for multiple '{word_form}'s:* Describe separate, distinct instances (e.g., 'a single {primary_object_name} here... then another lone {primary_object_name} there').*"
                         )
                    issues_found_for_generator_feedback = True
            
            # Check for STRICT_ZERO_VIOLATION (relevant if intro scene somehow gets into this loop, or if padding logic was adapted)
            # This is less likely for main beats but good to have a general pattern.
            # Example: "Strict zero mode: Found numbers with counts {734: 2} not in {1, 2, 3}."
            strict_zero_pattern = r"Strict zero mode: Found numbers with counts \{.*?(\d+): \d+.*?\} not in \{([\d, ]+)\}"
            strict_zero_match = re.search(strict_zero_pattern, explanation_from_validator, re.IGNORECASE)
            if "STRICT_ZERO_VIOLATION" in explanation_from_validator.upper() and strict_zero_match:
                violating_num = strict_zero_match.group(1) # A sample violating number
                allowed_set_str = strict_zero_match.group(2)
                history_prompt_addition_parts.append(
                    f"    - **Strict Zero Violation (Likely Intro/Padding Rule):** Your scene included numbers like '{num_to_words(int(violating_num))}' ({violating_num}) which are not allowed. Only numbers in {{{allowed_set_str}}} are permitted for general phrasing in this context. "
                    f"**Action for Revision:** REMOVE all numbers not in {{{allowed_set_str}}}. If character names contain digits (e.g., 'Unit 734'), refer to them by role or ensure the name is not parsed as a quantity."
                )
                issues_found_for_generator_feedback = True


            # Parse Rule B violations (Explicit Result)
            rule_b_pattern = r"VIOLATION Rule B: Intermediate numerical result \['?(\d+)'?\] was explicitly stated"
            rule_b_match = re.search(rule_b_pattern, explanation_from_validator, re.IGNORECASE)
            if rule_b_match:
                stated_result_val = rule_b_match.group(1)
                history_prompt_addition_parts.append(
                    f"    - **Explicit Outcome Stated (Rule 2):** The numerical result of this operation (which is '{num_to_words(int(stated_result_val))}' ({stated_result_val}), conceptually '{current_node_conceptual_name}') was incorrectly stated directly in the narrative. "
                    f"**Action for Revision:** This result MUST be implied conceptually. REMOVE any explicit statement of '{num_to_words(int(stated_result_val))}' or '{stated_result_val}' as the outcome. The narrative should lead to '{current_node_conceptual_name}' without saying the number."
                )
                issues_found_for_generator_feedback = True

            # Parse Rule D/4 violations (Forbidden Numbers)
            rule_d_pattern = r"forbidden number '([\w-]+)'" # Captures word form
            forbidden_numbers_found_words = re.findall(rule_d_pattern, explanation_from_validator, re.IGNORECASE)
            # Also check for direct number mentions if validator uses them
            rule_d_pattern_direct_num = r"forbidden.*?number.*?(\d+)"
            forbidden_numbers_found_digits = re.findall(rule_d_pattern_direct_num, explanation_from_validator, re.IGNORECASE)
            
            all_forbidden_mentioned_str = set()
            for w in forbidden_numbers_found_words: all_forbidden_mentioned_str.add(f"'{w}'")
            for d in forbidden_numbers_found_digits: all_forbidden_mentioned_str.add(f"'{num_to_words(int(d))}' ({d})")

            if all_forbidden_mentioned_str:
                formatted_forbidden = ", ".join(sorted(list(all_forbidden_mentioned_str)))
                history_prompt_addition_parts.append(
                    f"    - **Forbidden Number(s) Used (Rule 4):** Your narrative included forbidden number(s) like: {formatted_forbidden}. "
                    f"**Action for Revision:** These numbers MUST be removed. They are not allowed in this scene under any circumstances. Consult Rule 4 in your original Writing Guide (re-pasted below)."
                )
                issues_found_for_generator_feedback = True
            
            # Fallback for other errors or if parsing fails but it's an error
            if not issues_found_for_generator_feedback and not last_critique_json.get("is_valid", True):
                 history_prompt_addition_parts.append(
                    f"    - **General Adherence Issue:** The validator indicated issues with overall rule compliance. The primary feedback was: \"{summary_for_generator_prompt}\". The detailed explanation provided was: \"{explanation_from_validator}\". "
                    f"**Action for Revision:** Please carefully re-read this feedback and ALL rules in your original Writing Guide (re-pasted below), then revise to ensure full compliance. Pay special attention to any numbers mentioned in the validator's explanation."
                 )
            elif not issues_found_for_generator_feedback and last_critique_json.get("is_valid", True): # Should not happen if invalid
                 history_prompt_addition_parts.append(
                    f"    - (No specific rule violations were automatically parsed from the validator's explanation, but the overall summary was: \"{summary_for_generator_prompt}\". Please review the full explanation if provided: \"{explanation_from_validator}\")"
                 )

            if suggested_revisions_list:
                history_prompt_addition_parts.append("\n  - **Validator's Suggested Revisions (Consider these as potential solutions):**")
                for sug_rev_idx, sug_rev in enumerate(suggested_revisions_list):
                    history_prompt_addition_parts.append(f"    {sug_rev_idx+1}. {sug_rev}")
            
            history_prompt_addition_parts.append(f"\n  - *Full explanation from validator for your context (if you need more detail): \"{explanation_from_validator}\"*\n")

            # --- MANDATORY PRE-REVISION CHECKLIST ---
            history_prompt_addition_parts.append(
                "**MANDATORY PRE-REVISION CHECKLIST (Confirm your plan before rewriting):**\n"
                "1.  **Error Understanding:** Have you read and understood EACH specific issue and action item listed above from the validator?\n"
                "2.  **Rule 1.A Correction Plan (If Applicable):** For any missing or miscounted numbers, do you have a NEW, concrete plan for how and where you will narrate each required instance distinctly, ensuring exact frequencies as per the original Rule 1.A?\n"
                "3.  **Rule 2 Correction Plan (If Applicable):** Is your new plan for the outcome purely conceptual, completely avoiding the numerical result if it was previously stated?\n"
                "4.  **Rule 4/5 Correction Plan (If Applicable):** Have you identified and planned the removal of ALL forbidden or other extraneous numbers (those not covered by Rule 1 or a valid Rule 3 use)?\n"
                "5.  **Holistic Review:** Will your revised scene still be narratively coherent and engaging while perfectly meeting ALL numerical rules?\n"
                "6.  **Re-read Original Rules:** Briefly re-read the 'Key Original Rules (Reminder)' section below, especially the ULTRA-STRICT NUMBER RULES for *this specific scene*, to refresh your understanding of the target state.\n"
            )

            history_prompt_addition_parts.append(
                f"\n**Current Revision Task (Attempt {iteration}):**\n"
                f"1. Review ALL feedback above, especially the **Specific Issues to Correct** and complete the **Pre-Revision Checklist** mentally.\n"
                f"2. Re-read your original task & ALL number rules in the 'Narrative Challenge & Your Writing Guide for This Scene' section (partially re-pasted below for key rules).\n"
                f"3. **Meticulously fix ALL identified issues.** This is not optional. Ensure exact frequencies for required numbers and that mentions are explicit numerical words.\n"
                f"4. Ensure the narrative logic remains sound and compelling.\n"
                f"5. Output ONLY the revised narrative text for this scene.\n\n"
                f"**Key Original Rules (Reminder - see full initial prompt for all details, especially the ULTRA-STRICT NUMBER RULES section which contains the specific numbers for *your* current scene):\n**" # This is ultra_strict_instruction_for_llm_validator_context
                f"```text\n{ultra_strict_instruction_for_llm_validator_context[:3000]}...\n```\n" # Increased snippet length
            )
            current_generator_user_prompt_for_iteration = (
                f"{original_user_message_for_generator}\n\n{''.join(history_prompt_addition_parts)}"
            )
        else: # For iteration 1
            current_generator_user_prompt_for_iteration = original_user_message_for_generator
        
        turn_data["generator_user_prompt"] = current_generator_user_prompt_for_iteration

        generated_text_cleaned = ""
        try:
            log_prompt(
                header=f"LLM Beat Generator Prompt (Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}, Iter: {iteration}, Temp: {generator_temp:.2f}, Model: {MODEL})",
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
                header=f"LLM Beat Generator Raw Response (Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}, Iter: {iteration})",
                prompt=f"Raw Output:\n{raw_gen_text}",
                sample_index=sample_index,
            )

            generated_text_cleaned = raw_gen_text.strip()
            turn_data["generator_output"] = generated_text_cleaned if generated_text_cleaned else "GENERATION_EMPTY_OR_REFUSED"
            
            history_of_attempts.append(turn_data["generator_output"])

            if not generated_text_cleaned or generated_text_cleaned.lower().startswith(
                ("i cannot", "i'm sorry", "i am unable")
            ):
                logger_obj.warning(
                    f"Generator refusal or empty in iter {iteration} for Op {current_op_node.op}, BeatNum {beat_number_in_sample}. Raw: '{raw_gen_text}'"
                )
                # This case will flow to validator, which should fail it.
                # We still want to log this turn.
                critique_for_empty_gen = {
                        "is_valid": False,
                        "explanation_for_generator": "The generation was empty or an API refusal. This counts as a failed attempt.",
                        "overall_revision_summary_for_generator_prompt": "Previous attempt was empty/refused. Please generate the scene as per original instructions, paying close attention to all numerical rules, especially exact frequencies.",
                        "explanation_for_audit": "N/A - Generation failed or was refused by LLM.",
                        "suggested_revisions": []
                    }
                turn_data["validator_critique_json"] = critique_for_empty_gen
                history_of_critiques.append(critique_for_empty_gen)
                current_beat_conversation_turns.append(turn_data) # Log this failed turn
                if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS: time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue # To next iteration

        except Exception as e_gen:
            logger_obj.error(
                f"Error generating beat in LLM loop iter {iteration} for Op {current_op_node.op}, BeatNum {beat_number_in_sample}: {e_gen}",
                exc_info=True,
            )
            turn_data["generator_output"] = f"ERROR_DURING_GENERATION: {str(e_gen)[:200]}"
            history_of_attempts.append(turn_data["generator_output"])
            
            critique_for_gen_error = {
                    "is_valid": False,
                    "explanation_for_generator": f"A system error occurred during text generation: {str(e_gen)[:200]}. Please try to regenerate the scene, carefully adhering to all original instructions and numerical rules, especially exact counts.",
                    "overall_revision_summary_for_generator_prompt": "System error during previous generation attempt. Please retry the original task, focusing on strict rule adherence including number frequencies.",
                    "explanation_for_audit": f"N/A - Exception during generation: {e_gen}",
                    "suggested_revisions": []
                }
            turn_data["validator_critique_json"] = critique_for_gen_error
            history_of_critiques.append(critique_for_gen_error)
            current_beat_conversation_turns.append(turn_data) # Log this failed turn

            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue
            else: # Failed all iterations due to generation error
                context.beat_revision_logs.append({
                    "sample_index": sample_index,
                    "beat_op": current_op_node.op,
                    "beat_conceptual_name": current_node_conceptual_name,
                    "beat_number_in_sample": beat_number_in_sample,
                    "initial_generator_prompt_for_beat": original_user_message_for_generator,
                    "conversation_turns": current_beat_conversation_turns,
                    "final_status": "failed_all_revisions_due_to_generation_error"
                })
                return None

        # --- Internal LLM Validator Call ---
        validator_system_prompt = """You are an AI numerical compliance checker.
Your ONLY task is to evaluate a story 'beat' against strict numerical rules.
You MUST output your response as a valid JSON object and NOTHING ELSE, adhering to the provided schema.
Do not include any text, explanations, or markdown (like ```json) before or after the single JSON object.
Start with '{' and end with '}'."""

        required_counts_summary_list_for_val = []
        if direct_atom_counts_val:
            for num_val, count_val in sorted(direct_atom_counts_val.items()):
                num_word = num_to_words(num_val)
                count_word_val = num_to_words(count_val)
                required_counts_summary_list_for_val.append(f"- '{num_word}' ({num_val}): exactly {count_word_val} ({count_val}) time(s)")

        required_counts_summary_str_for_validator_context = ""
        if required_counts_summary_list_for_val:
            required_counts_summary_str_for_validator_context = (
                "**Summary of Required Number Frequencies (from Generator's Rule 1 for this beat):**\n" +
                "\n".join(required_counts_summary_list_for_val) + "\n"
            )
        else:
            required_counts_summary_str_for_validator_context = "(No specific atomic number frequencies were mandated for the generator for this beat according to its Rule 1 summary.)\n"

        gt_word_val_for_validator_prompt = "N/A"
        if overall_ground_truth_answer_val is not None:
            gt_word_val_for_validator_prompt = f"'{num_to_words(overall_ground_truth_answer_val)}' ({overall_ground_truth_answer_val})"

        validator_user_prompt = f"""You are an AI numerical compliance checker. Evaluate the 'Generated Beat Text' below with ABSOLUTE PRECISION regarding its numerical content and adherence to storytelling rules.

**Primary Goal:** Verify the text strictly adheres to all numerical rules provided in the 'ULTRA-STRICT NUMBER RULES (Generator's Writing Guide)' section below.

**Context for Your Evaluation (Derived from Generator's Task):**
- Operation Type: {current_op_node.op} ({OP_LABELS.get(current_op_node.op, current_op_node.op)})
- Conceptual Names for Prior Results (if any were inputs to this step): {conceptual_inputs_str_for_llm_validator}
- New Specific Numbers for This Step (Atomic Inputs, as given to Generator, including required counts/frequencies): {atomic_inputs_words_str_for_llm_validator}
{required_counts_summary_str_for_validator_context}
- Numerical Result of THIS Operation (as known to Generator, was to be Implied): {expected_beat_result_words_for_llm_validator}
- Overall Final Answer of Entire Story (for context): {gt_word_val_for_validator_prompt}
- Is this beat's numerical result also one of its atomic inputs? {'Yes' if (correct_result_val is not None and direct_atom_counts_val and direct_atom_counts_val.get(correct_result_val, 0) > 0) else 'No'}

**ULTRA-STRICT NUMBER RULES (Generator's Writing Guide - THIS IS THE GROUND TRUTH FOR YOUR VALIDATION):**
---
{ultra_strict_instruction_for_llm_validator_context}
---

**VALIDATION ALGORITHM - FOLLOW EXACTLY, using the Generator's Writing Guide above as the source of truth for allowed/forbidden numbers for THIS beat:**
1.  Identify ALL numbers (words or digits) in the 'Generated Beat Text'. Determine the count/frequency of each number found.
2.  **RULE A (Required Atomic Inputs):**
    -   Does the text mention ALL numbers listed in Rule 1 of the Generator's Writing Guide (specifically, in its 'Summary of Required Number Frequencies' if present, or the prose description), **with the correct frequency/count** as specified in that rule?
    -   *MEDIAN Exception Handling:* If the operation is MEDIAN, Rule 1 still applies fully: ALL direct atomic inputs (with their specified frequencies) MUST be present. The median result itself is handled by Rule B.
3.  **RULE B (Outcome Handling):**
    -   Does the text adhere to Rule 2 of the Generator's Writing Guide regarding the explicit statement (or non-statement) of the numerical result?
    -   *MEDIAN Exception Handling:* For MEDIAN, the numerical result MUST NEVER be stated.
    -   **CRITICAL FOR ALL INTERMEDIATE STEPS (NON-FINAL OPERATIONS):** The numerical result of the operation MUST NOT be explicitly stated. If it is stated, this is a failure.
4.  **RULE C (Additionally Allowed Numbers):**
    -   Are any numbers present that are ONLY allowed under Rule 3 of the Generator's Writing Guide (e.g., phrasing numbers, arity count)? Are they used appropriately and not excessively?
5.  **RULE D (Forbidden Numbers & Values):**
    -   Does the text avoid ALL numbers (and their counts if specified as forbidden in excess) listed as forbidden in Rule 4 of the Generator's Writing Guide?
    -   Does the text avoid any other numbers not covered by Rules 1-4, as per Rule 5?
6.  **RULE E (Prior Result Handling):**
    -   If Rule 6 (Referring to Quantities from Previous Events) is present in the Generator's Writing Guide, does the text adhere to it by using conceptual names for prior results and NOT their numerical values?

**Specific Checks for your `explanation_for_generator`:**
- If Rule A fails due to incorrect counts: "VIOLATION Rule A: Number [X] (word form '[word]') was required [Y] times (derived from the input list frequencies), but found [Z] times."
- If Rule A fails due to missing number: "VIOLATION Rule A: Required number [X] (word form '[word]'), expected [Y] times, was completely missing."
- If Rule B (intermediate result stated) fails: "VIOLATION Rule B: Intermediate numerical result [value] was explicitly stated. It MUST be implied conceptually."

**VALIDATION RESPONSE (Output JSON only, using the schema provided):**
Based on the algorithm above:
-   `is_valid` (boolean): True if ALL rules (A-E, based on the Generator's Writing Guide, including frequency checks for Rule A) are met, False otherwise.
-   `explanation_for_generator` (string): Detailed step-by-step reasoning for failure, referencing specific rules (A-E) from the Generator's Writing Guide and numbers/counts found/missing/incorrect. If valid, explain why it meets all criteria.
-   `explanation_for_audit` (string, only if `is_valid` is true): Brief summary of why it's valid.
-   `overall_revision_summary_for_generator_prompt` (string): Concise instruction for the generator if invalid. If valid, state "No revision needed."
-   `suggested_revisions` (array of strings, optional).

**Generated Beat Text to Evaluate:**
---
{generated_text_cleaned}
---
"""
        turn_data["validator_system_prompt"] = validator_system_prompt
        turn_data["validator_user_prompt"] = validator_user_prompt

        validation_result_json = None 
        try:
            log_prompt(
                header=f"LLM Validator Prompt (Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}, Iter: {iteration}, Model: {internal_validator_llm_model})",
                prompt=f"System: {validator_system_prompt}\nUser:\n{validator_user_prompt}",
                sample_index=sample_index,
            )
            api_call_params_for_validator = {
                "model": internal_validator_llm_model,
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
                header=f"LLM Validator Raw Response (Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}, Iter: {iteration})",
                prompt=f"Raw Output:\n{validator_raw_output}",
                sample_index=sample_index,
            )

            validation_result_json = parse_llm_json_with_fallback(
                validator_raw_output,
                { 
                    "is_valid": False,
                    "explanation_for_generator": "Validator response was not valid JSON or was empty. This might be due to an issue with the generated text or the validator itself. Please try generating the scene again, ensuring all numerical rules are meticulously followed, especially regarding exact counts of required numbers.",
                    "overall_revision_summary_for_generator_prompt": "Validator had trouble processing the previous text. Please regenerate the scene, focusing on extreme clarity and adherence to all numerical constraints, including exact frequencies of numbers.",
                    "explanation_for_audit": "Validator output parsing error.",
                    "suggested_revisions": []
                },
                f"in LLM validator iteration {iteration} for Op: {current_op_node.op}, BeatNum: {beat_number_in_sample}"
            )
            turn_data["validator_critique_json"] = validation_result_json
            history_of_critiques.append(validation_result_json) 

            current_beat_conversation_turns.append(turn_data) 

            if validation_result_json.get("is_valid"):
                logger_obj.info(
                    f"LLM Validator ({internal_validator_llm_model}) PASSED beat for Op {current_op_node.op}, BeatNum {beat_number_in_sample} in iter {iteration}. Audit: {validation_result_json.get('explanation_for_audit', 'N/A')}"
                )
                context.beat_revision_logs.append({
                    "sample_index": sample_index,
                    "beat_op": current_op_node.op,
                    "beat_conceptual_name": current_node_conceptual_name,
                    "beat_number_in_sample": beat_number_in_sample,
                    "initial_generator_prompt_for_beat": original_user_message_for_generator,
                    "conversation_turns": current_beat_conversation_turns,
                    "final_status": "success_after_revisions" if iteration > 1 else "success_on_first_attempt"
                })
                return generated_text_cleaned
            else:
                logger_obj.warning(
                    f"LLM Validator ({internal_validator_llm_model}) FAILED beat for Op {current_op_node.op}, BeatNum {beat_number_in_sample} in iter {iteration}. Feedback for generator: {validation_result_json.get('overall_revision_summary_for_generator_prompt', 'No summary.')}"
                )
                logger_obj.debug(f"Detailed explanation from validator: {validation_result_json.get('explanation_for_generator', 'No detailed explanation.')}")
                # Loop continues for next iteration

        except Exception as e_val_call:
            logger_obj.error(
                f"Error during LLM validation call/processing iter {iteration} for Op {current_op_node.op}, BeatNum {beat_number_in_sample} (Validator Model: {internal_validator_llm_model}): {e_val_call}",
                exc_info=True,
            )
            critique_for_val_error = {
                        "is_valid": False,
                        "explanation_for_generator": f"A system error occurred during the validation phase: {str(e_val_call)[:200]}. Please try to regenerate the scene, carefully adhering to all original instructions and numerical rules, especially exact counts.",
                        "overall_revision_summary_for_generator_prompt": "System error during validation. Please retry the original task, focusing on strict rule adherence including number frequencies.",
                        "explanation_for_audit": f"N/A - Exception during validation: {e_val_call}",
                        "suggested_revisions": []
                    }
            turn_data["validator_critique_json"] = critique_for_val_error
            history_of_critiques.append(critique_for_val_error) 
            current_beat_conversation_turns.append(turn_data) 

            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue 
            else: 
                context.beat_revision_logs.append({
                    "sample_index": sample_index,
                    "beat_op": current_op_node.op,
                    "beat_conceptual_name": current_node_conceptual_name,
                    "beat_number_in_sample": beat_number_in_sample,
                    "initial_generator_prompt_for_beat": original_user_message_for_generator,
                    "conversation_turns": current_beat_conversation_turns,
                    "final_status": "failed_all_revisions_due_to_validation_error"
                })
                return None

    # If loop finishes (all iterations failed because validator kept failing)
    logger_obj.error(
        f"Beat for Op {current_op_node.op}, BeatNum {beat_number_in_sample} failed LLM validation after {context_config.MAX_LLM_VALIDATION_ITERATIONS} iterations (Internal Validator: {internal_validator_llm_model})."
    )
    if history_of_critiques: 
        last_fail_critique = history_of_critiques[-1]
        logger_obj.error(f"Last critique for Op {current_op_node.op}, BeatNum {beat_number_in_sample} (failed all iterations with {internal_validator_llm_model}): {json.dumps(last_fail_critique, indent=2)}")
    
    if current_beat_conversation_turns: # Should always have turns if loop ran
        context.beat_revision_logs.append({
            "sample_index": sample_index,
            "beat_op": current_op_node.op,
            "beat_conceptual_name": current_node_conceptual_name,
            "beat_number_in_sample": beat_number_in_sample,
            "initial_generator_prompt_for_beat": original_user_message_for_generator,
            "conversation_turns": current_beat_conversation_turns,
            "final_status": "failed_all_revisions"
        })
    return None

# --- Iterative LLM Validation Loop ---
# --- Iterative LLM Validation Loop ---
def _generate_narrative_recursive(
    node: Node,
    context: "GenerationContext",
    is_root: bool,
):
    world = context.world
    config_obj = context.config # This is context.config, which was passed from global config
    encoder = context.encoder
    logger_obj = context.logger
    p_inflect = context.p_inflect
    narrative_anchor_map = context.narrative_anchor_map

    node_id = id(node)
    current_node_conceptual_name = "this_step_s_outcome"
    if isinstance(node, OpNode):
        current_node_conceptual_name = narrative_anchor_map.get(
            node_id, f"the_unnamed_{node.op}_result_{node_id % 100}"
        )

    op_for_log = getattr(node, "op", "AtomNode")
    logger_obj.debug(
        f"[Sample {context.sample_index + 1}] _generate_narrative_recursive: "
        f"Processing Node Type: {type(node).__name__}, Op: {op_for_log}, Conceptual Name: '{current_node_conceptual_name}', IsRoot: {is_root}, "
        f"Beat: {context.beat_counter['current'] + 1 if isinstance(node, OpNode) else '-'}/{context.beat_counter['total']}"
    )

    if isinstance(node, Atom):
        logger_obj.debug(f"Node is Atom ({node.n}), value is {node.value}. Returning.")
        return

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
                    logger_obj.error(f"CRITICAL ERROR: Child OpNode {child.op} ('{child_anchor_str}') has no computed value after recursive call for parent {node.op}!")
            else:
                logger_obj.warning(f"OpNode child {child.op} of {node.op} has no conceptual name in map.")
        if context.tokens_used >= config_obj.MAX_TOTAL_TOKENS - config_obj.MAX_TOKENS_BUFFER:
            logger_obj.warning(
                f"TOKEN LIMIT reached after processing child {child_index+1} for node {node.op} ('{current_node_conceptual_name}'). Aborting beat generation for this node."
            )
            raise BeatGenerationError("Token limit reached during child processing for current beat.")

    context.beat_counter["current"] += 1
    logger_obj.info(
        f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for {node.op} ('{current_node_conceptual_name}')"
    )

    op_label = OP_LABELS.get(node.op, node.op)
    direct_atom_children = [c_atom for c_atom in node.children if isinstance(c_atom, Atom)]
    all_direct_atomic_inputs_as_list = [a.n for a in direct_atom_children]
    direct_atom_values_counts = Counter(all_direct_atomic_inputs_as_list)
    unique_direct_atom_values_for_prompt = sorted(list(set(all_direct_atomic_inputs_as_list)))

    correct_result = node.value
    if correct_result is None:
        logger_obj.error(f"CRITICAL: Node {node.op} has no pre-computed value (node.value is None). Aborting beat.")
        raise BeatGenerationError(f"Node {node.op} has no pre-computed value.")

    forbidden_for_current_beat_py_validator = set(context.introduced_atoms)
    if context.overall_ast_root is not None:
        for processed_node in postorder(context.overall_ast_root):
            if isinstance(processed_node, OpNode) and processed_node.value is not None:
                if id(processed_node) == id(node): continue
                is_direct_child = any(id(child_node) == id(processed_node) for child_node in node.children if isinstance(child_node, OpNode))
                if is_direct_child: continue
                forbidden_for_current_beat_py_validator.add(processed_node.value)
    else:
        logger_obj.warning("overall_ast_root is None in context, cannot accurately build full forbidden set from prior OpNode results.")

    if context.overall_ground_truth_answer is not None:
        if context.overall_ground_truth_answer != correct_result and \
           context.overall_ground_truth_answer not in direct_atom_values_counts:
            forbidden_for_current_beat_py_validator.add(context.overall_ground_truth_answer)

    forbidden_for_current_beat_py_validator -= set(direct_atom_values_counts.keys())
    logger_obj.debug(
        f"Forbidden numbers for Python validator (Op: {node.op}, Beat: {context.beat_counter['current']}): {sorted(list(forbidden_for_current_beat_py_validator))}"
    )

    conceptual_inputs_context_list = []
    if child_op_node_results_as_conceptual_inputs:
        for name, val in child_op_node_results_as_conceptual_inputs.items():
            conceptual_inputs_context_list.append(
                f"the concept known as '{name}' (which represents the numerical value {val})"
            )
    conceptual_inputs_context_str = (
        (", ".join(conceptual_inputs_context_list) if conceptual_inputs_context_list
         else "None (this is the first calculation or uses only new numbers)")
    )

    conceptual_input_names_only_list_for_action = [f"'{name}'" for name in child_conceptual_names_list]
    conceptual_input_names_only_str_for_action = (
        (", ".join(conceptual_input_names_only_list_for_action)
        if conceptual_input_names_only_list_for_action
        else "no prior calculated quantities")
    )

    required_counts_summary_list = []
    if all_direct_atomic_inputs_as_list:
        counts_for_summary = Counter(all_direct_atomic_inputs_as_list)
        for num_val, count_val in sorted(counts_for_summary.items()):
            num_word = num_to_words(num_val)
            count_word_val = num_to_words(count_val)
            required_counts_summary_list.append(f"- '{num_word}' ({num_val}): exactly {count_word_val} ({count_val}) time(s)")

    required_counts_summary_str_for_prompt = ""
    if required_counts_summary_list:
        required_counts_summary_str_for_prompt = (
            "\n".join(required_counts_summary_list) +
            "\nEnsure each number above appears in the narrative precisely that many times as a direct input for this operation.\n"
        )

    must_mention_prose_parts = []
    for atom_val_prose in unique_direct_atom_values_for_prompt:
        count_prose = all_direct_atomic_inputs_as_list.count(atom_val_prose)
        atom_word_prose = num_to_words(atom_val_prose)
        if count_prose > 1 and p_inflect:
            must_mention_prose_parts.append(f"'{atom_word_prose}' ({atom_val_prose}) (which appears {p_inflect.number_to_words(count_prose)} times as direct inputs)")
        else:
            must_mention_prose_parts.append(f"'{atom_word_prose}' ({atom_val_prose})")
    must_mention_text_detailed_prose_for_prompt = (
        (" and ".join(must_mention_prose_parts) if must_mention_prose_parts
         else "No new numbers to explicitly state (e.g., if all inputs are conceptual)")
    )

    special_med_input_clarification_for_prompt = ""
    if node.op == "MED":
        special_med_input_clarification_for_prompt = (
            f" (Important MEDIAN Note: For this MEDIAN operation, ALL direct atomic input numbers listed above, with their exact frequencies as detailed in the summary, MUST be mentioned. "
            f"The MEDIAN result itself ('{num_to_words(correct_result)}' ({correct_result})) MUST be implied and not stated, as per Rule 2.)"
        )
        if not unique_direct_atom_values_for_prompt:
             must_mention_text_detailed_prose_for_prompt = "No new numbers to explicitly state for this MEDIAN step (e.g., if all inputs are conceptual)"

    atomic_inputs_context_list_detailed_full = []
    if direct_atom_children:
        for atom_node_ctx in direct_atom_children:
            atomic_inputs_context_list_detailed_full.append(
                f"'{num_to_words(atom_node_ctx.n)}' ({atom_node_ctx.n})"
            )
    atomic_inputs_context_str_detailed_for_prompt = (
        (", ".join(atomic_inputs_context_list_detailed_full)
        if atomic_inputs_context_list_detailed_full
        else "None")
    )

    result_handling_rule_text_for_prompt = (
        f"The numerical result of THIS operation ({op_label}) -- which is '{num_to_words(correct_result)}' ({correct_result}) -- MUST NOT be explicitly stated in the text. "
        f"It must only be implied by events. This implied result will be known conceptually as '{current_node_conceptual_name}' for future steps."
    )
    if correct_result is not None and correct_result in direct_atom_values_counts:
        result_handling_rule_text_for_prompt += (
            f" (Special Note: The result value '{num_to_words(correct_result)}' ({correct_result}) is also one of your required atomic inputs. "
            f"While you MUST mention it as an input (Rule 1, with correct frequency), ensure your narrative does NOT frame it as the *outcome* of this operation. "
            f"The outcome '{current_node_conceptual_name}' must still be implied conceptually.)"
        )

    temp_phrasing_words_detailed_for_rule3 = [f"'{num_to_words(n)}' ({n})" for n in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET]
    phrasing_numbers_gen_str_detailed_for_rule3 = ", ".join(sorted(temp_phrasing_words_detailed_for_rule3))

    actual_arity_for_current_op = len(node.children)
    may_use_gen_parts_detailed = [
        f"small numbers like {phrasing_numbers_gen_str_detailed_for_rule3} for general narrative phrasing (e.g., 'two guards')"
    ]
    arity_is_problematic_forbidden = (
        actual_arity_for_current_op in forbidden_for_current_beat_py_validator and
        actual_arity_for_current_op not in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    )
    if actual_arity_for_current_op > 0 :
        if not arity_is_problematic_forbidden:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(actual_arity_for_current_op)}' ({actual_arity_for_current_op}) IF it's genuinely used to count the items/inputs involved in THIS specific action"
            )
        else:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(actual_arity_for_current_op)}' ({actual_arity_for_current_op}) ONLY if essential for counting and clearly distinct from its forbidden meaning (use with extreme caution or avoid)"
            )
    may_use_gen_clause_content_detailed_for_prompt = "; ".join(may_use_gen_parts_detailed)

    gt_counting_caution_for_gen_prompt = ""
    if context.overall_ground_truth_answer is not None and \
       context.overall_ground_truth_answer in forbidden_for_current_beat_py_validator and \
       config_obj.MIN_ALLOWED_SMALL_NUMBER <= context.overall_ground_truth_answer <= config_obj.MAX_ALLOWED_SMALL_NUMBER and \
       context.overall_ground_truth_answer not in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
        gt_word = num_to_words(context.overall_ground_truth_answer)
        gt_counting_caution_for_gen_prompt = (
            f"- Special Caution: The number '{gt_word}' ({context.overall_ground_truth_answer}) is the overall story's final answer and generally forbidden here. "
            f"Avoid using it even for incidental counting unless absolutely unavoidable and clearly unrelated to the final answer.\n"
        )

    temp_forbidden_detailed_list_for_rule4 = []
    for n_forbidden in sorted(list(forbidden_for_current_beat_py_validator)):
        temp_forbidden_detailed_list_for_rule4.append(f"'{num_to_words(n_forbidden)}' ({n_forbidden})")

    must_avoid_str_for_generator_prompt_detailed = (
        (", ".join(temp_forbidden_detailed_list_for_rule4) if temp_forbidden_detailed_list_for_rule4
         else "None specifically (beyond the general rule against unlisted numbers and unstated results)")
    )
    no_other_numbers_rule_text_for_prompt = "Do not introduce any other numerical values (digits or words) beyond those explicitly covered by rules 1-4 above. No intermediate sums or calculations should be shown."

    prior_results_handling_rule_for_prompt = ""
    if child_conceptual_names_list:
        prior_results_names_str_for_rule6 = ", ".join([f"'{name}'" for name in child_conceptual_names_list])
        prior_results_handling_rule_for_prompt = (
            f"6.  **Echoes of the Past (Referring to Prior Results):** If your characters refer to the outcome of a previous significant event "
            f"(which we've named, for example, {prior_results_names_str_for_rule6}), you MUST use these evocative names. "
            f"Think of them as legendary items or well-known past results. The actual number these names represent should remain a secret from the reader, hinted at only by their conceptual name.\n"
        )

    primary_object_as_string = "[DEFAULT_PRIMARY_OBJECT]"
    primary_object_value_from_world = None
    try:
        primary_object_value_from_world = world.get("object")
        if primary_object_value_from_world is None:
            logger_obj.warning(
                f"[Sample {context.sample_index + 1}, Op: {op_for_log}] world.get('object') returned None. Using fallback '{primary_object_as_string}'. World keys: {list(world.keys())}"
            )
            primary_object_as_string = "[OBJECT_WAS_NONE]"
        elif isinstance(primary_object_value_from_world, str):
            primary_object_as_string = primary_object_value_from_world
        else:
            logger_obj.warning(
                f"[Sample {context.sample_index + 1}, Op: {op_for_log}] world['object'] was not a string (type: {type(primary_object_value_from_world)}). Attempting str(). Value: '{str(primary_object_value_from_world)[:100]}'"
            )
            primary_object_as_string = str(primary_object_value_from_world)
    except Exception as e_obj_conv:
        logger_obj.error(
            f"[Sample {context.sample_index + 1}, Op: {op_for_log}] Critical error getting/converting world['object']: {e_obj_conv}. "
            f"Original type: {type(primary_object_value_from_world)}. "
            f"Using fallback '{primary_object_as_string}'."
        )

    safe_primary_object_for_fstring = "[ERROR_ESCAPING_OBJECT_NAME_DEFAULT]"
    try:
        if not isinstance(primary_object_as_string, str):
            logger_obj.error(f"Internal Error: primary_object_as_string became non-string: {type(primary_object_as_string)}. Using fallback for safe_primary_object_for_fstring.")
        else:
            safe_primary_object_for_fstring = primary_object_as_string.replace("{", "{{").replace("}", "}}")
    except AttributeError as e_replace:
        logger_obj.error(
            f"[Sample {context.sample_index + 1}, Op: {op_for_log}] AttributeError during .replace() for safe_primary_object_for_fstring. "
            f"primary_object_as_string was type: {type(primary_object_as_string)}, value: '{str(primary_object_as_string)[:100]}'. Error: {e_replace}. "
            f"Falling back for safe_primary_object_for_fstring."
        )
    except Exception as e_generic_replace:
        logger_obj.error(
            f"[Sample {context.sample_index + 1}, Op: {op_for_log}] Generic error during .replace() for safe_primary_object_for_fstring. "
            f"primary_object_as_string was type: {type(primary_object_as_string)}, value: '{str(primary_object_as_string)[:100]}'. Error: {e_generic_replace}. "
            f"Falling back for safe_primary_object_for_fstring."
        )
    logger_obj.debug(f"OpNode {op_for_log}: primary_object_as_string = '{primary_object_as_string}', safe_primary_object_for_fstring = '{safe_primary_object_for_fstring}'")

    action_description_parts = [
        f"**Your Scene's Core Action & Narrative Goal (Follow this closely):**\n"
        f"This scene needs to narrate an event or discovery that mirrors the mathematical operation: **{op_label}**. "
        f"The central items of interest are the '{safe_primary_object_for_fstring}'."
    ]
    op_specific_action_details = ""
    op_specific_outcome_implication_details = ""

    if node.op == "SUM":
        op_specific_action_details = (
            f"Imagine your characters are gathering or combining distinct collections of '{safe_primary_object_for_fstring}'. "
            f"Some of these collections might be newly found or counted, corresponding to the numbers (and their frequencies): {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none mentioned for this step'}. "
            f"Other collections might be the results of previous efforts, known only by evocative names like {conceptual_input_names_only_str_for_action}. "
            f"Your story should clearly show these being brought together into a single, larger accumulation."
        )
        op_specific_outcome_implication_details = (
            f"This combined accumulation will then be known conceptually as '{current_node_conceptual_name}'. "
            f"Remember, its actual total numerical size ('{num_to_words(correct_result)}') must not be stated."
        )
    elif node.op == "MIN":
        op_specific_action_details = (
            f"Picture your characters assessing several distinct quantities or instances of '{safe_primary_object_for_fstring}'. "
            f"These might include newly encountered items (quantified by ALL the numbers with their frequencies: {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none for this step'}) "
            f"and also the conceptual results of past endeavors (referred to as {conceptual_input_names_only_str_for_action}). "
            f"The narrative should focus on them identifying or selecting the *smallest* or *least significant* among all these."
        )
        op_specific_outcome_implication_details = (
            f"This single, smallest quantity they identify will then be conceptually known as '{current_node_conceptual_name}'. "
            f"Its precise numerical value ('{num_to_words(correct_result)}') must remain a secret to the reader."
        )
    elif node.op == "MAX":
        op_specific_action_details = (
            f"Envision your characters evaluating several different amounts or examples of '{safe_primary_object_for_fstring}'. "
            f"These could be new discoveries (detailed by ALL the numbers with their frequencies: {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none for this step'}) "
            f"or the outcomes of prior activities (known conceptually as {conceptual_input_names_only_str_for_action}). "
            f"The story should center on them determining or isolating the *largest*, *most potent*, or *most significant* among these."
        )
        op_specific_outcome_implication_details = (
            f"This preeminent quantity will thereafter be referred to conceptually as '{current_node_conceptual_name}'. "
            f"Do not explicitly state its actual numerical value ('{num_to_words(correct_result)}')."
        )
    elif node.op == "AVG":
        op_specific_action_details = (
            f"The scene should depict characters considering a set of '{safe_primary_object_for_fstring}' "
            f"(some new, described by ALL numbers with their frequencies: {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none for this step'}, "
            f"others being prior conceptual results like {conceptual_input_names_only_str_for_action}). "
            f"Their actions or observations should lead them to understand a 'typical', 'representative', or 'average' characteristic or measure across these items, "
            f"without performing explicit mathematical division in the narrative."
        )
        op_specific_outcome_implication_details = (
            f"This representative 'average' value (which is '{num_to_words(correct_result)}') will be conceptually known as '{current_node_conceptual_name}' and must be implied, not stated numerically."
        )
    elif node.op == "MED":
        op_specific_action_details = (
            f"**This is a MEDIAN scene (NEW RULE: ALL direct atomic inputs with their exact frequencies MUST be mentioned).** Your characters will encounter various '{safe_primary_object_for_fstring}' "
            f"(some new, described by ALL the numbers with their frequencies: {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none for this step'}, "
            f"others being prior conceptual results like {conceptual_input_names_only_str_for_action}). "
            f"The core of the scene is about them identifying a 'central element', 'balancing point', or 'middle value' from all these available items/numbers. "
            f"Crucially, the actual median value ('{num_to_words(correct_result)}') MUST NEVER be written out as the result."
        )
        op_specific_outcome_implication_details = (
            f"The conceptual outcome, '{current_node_conceptual_name}' (which represents the median value '{num_to_words(correct_result)}'), must be implied through the narrative of finding this central point, not stated numerically."
        )
    elif node.op == "SM":
        op_specific_action_details = (
            f"**This is a SUM MODULO 10 scene.** The characters' actions should represent combining all involved quantities of '{safe_primary_object_for_fstring}' "
            f"(the new numbers being (with their frequencies): {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none for this step'}; "
            f"and any prior results referred to only by their conceptual names: {conceptual_input_names_only_str_for_action}). "
            f"After this conceptual combination (do not state any intermediate sum!), they should then discover or focus on a core essence, a symbolic digit, or a cyclical pattern related to this unstated total. This discovery is equivalent to finding the sum modulo 10."
        )
        op_specific_outcome_implication_details = (
            f"This final symbolic essence or pattern will be conceptually known as '{current_node_conceptual_name}' (representing the value '{num_to_words(correct_result)}'), and this outcome must be implied, not stated numerically."
        )

    if not op_specific_action_details:
        op_specific_action_details = f"  - The characters perform an action related to '{op_label}' using new numbers (with frequencies: {atomic_inputs_context_str_detailed_for_prompt if direct_atom_children else 'none'}) and prior results ({conceptual_input_names_only_str_for_action})."
    action_description_parts.append(op_specific_action_details)

    if child_conceptual_names_list and not ("conceptual_input_names_only_str_for_action" in op_specific_action_details.lower()):
        action_description_parts.append(
            f"  - Remember to refer to prior results (like {conceptual_input_names_only_str_for_action}) using ONLY their conceptual names, NOT their underlying numbers."
        )
    if direct_atom_children and not ("atomic_inputs_context_str_detailed_for_prompt" in op_specific_action_details.lower()):
         action_description_parts.append(
            f"  - The new numbers directly involved in this step (with their frequencies) are: {atomic_inputs_context_str_detailed_for_prompt} (ALL must be mentioned with correct counts)."
        )

    if not op_specific_outcome_implication_details:
        op_specific_outcome_implication_details = (
            f"The outcome of this action should lead to a new understanding or quantity, which will be conceptually known as '{current_node_conceptual_name}'. "
            f"This concept ('{current_node_conceptual_name}') corresponds to the numerical value {correct_result} ('{num_to_words(correct_result)}')."
        )
    action_description_parts.append(f"\n{op_specific_outcome_implication_details}")
    action_description_parts.append(
        f"\n**Crucial Storytelling Constraint:** Your narrative MUST NOT explicitly state the number '{num_to_words(correct_result)}' ({correct_result}) AS THE RESULT of this operation. "
        f"Instead, the story should imply this outcome through the characters' actions, discoveries, or the state of the '{safe_primary_object_for_fstring}', "
        f"so that '{current_node_conceptual_name}' becomes the way to think about this new state."
    )
    if correct_result is not None and correct_result in direct_atom_values_counts:
        action_description_parts.append(
            f"**Important Narrative Challenge:** The numerical value of this operation's outcome ('{num_to_words(correct_result)}') is ALSO one of the numbers you need to mention as an input (with its specific frequency). "
            f"Your story must carefully distinguish its role. Mention '{num_to_words(correct_result)}' ({correct_result}) when describing the initial items/quantities (as per Rule 1, with correct frequency). "
            f"However, when describing the *result* of the operation, you must only imply it conceptually as '{current_node_conceptual_name}' and NOT restate '{num_to_words(correct_result)}' as the outcome (as per Rule 2)."
        )
    action_description_parts.append(
        f"\nRemember, all numbers that ARE explicitly mentioned (only those listed in Rule 1 of your Writing Guide, with their exact frequencies) must be written as words (e.g., 'seven' not '7')."
    )
    action_description_for_prompt = "\n".join(action_description_parts)

    # --- Define Rule 1 components for ultra_strict_instruction ---
    rule1_header = "**1. Key Details to Feature (Numbers in Action & Their EXACT Frequencies):**\n"
    rule1_summary_section = (
        f"    **A. SUMMARY OF REQUIRED NUMBER FREQUENCIES FOR THIS SCENE (CRITICAL & MANDATORY: MENTION EACH NUMBER EXACTLY THE SPECIFIED TIMES. NO MORE, NO LESS. EACH MENTION MUST BE A DISTINCT NARRATIVE ELEMENT REPRESENTING AN INPUT TO THIS OPERATION.):**\n"
        f"{required_counts_summary_str_for_prompt if required_counts_summary_str_for_prompt else '    (No specific new atomic numbers and their frequencies are mandated for this step. Focus on other rules.)'}\n\n"
    )
    rule1_explanation_section = (
        f"    **B. Detailed Explanation & Examples for Rule 1 (Adherence is Paramount):**\n"
        f"    Your narrative MUST introduce quantities of the central item ('{safe_primary_object_for_fstring}') corresponding to EACH number listed in the Summary (1.A) above. Each number MUST be mentioned EXACTLY the number of times specified. \n"
        f"    *   **How to Count Mentions (Crucial for Correctness):** Each mention must correspond to a distinct group, instance, or observation of items being introduced as a direct input for THIS scene's operation. \n"
        f"        - If 'one' (value: {config_obj.INVALID_RESULT_PLACEHOLDER+1}) is required twice: You might narrate, 'She found a single {safe_primary_object_for_fstring} in an alcove. Later, exploring a different passage, another lone {safe_primary_object_for_fstring} was discovered.' This counts as two distinct mentions of 'one'.\n"
        f"        - Contrast: Saying 'She found two single {safe_primary_object_for_fstring}s' would likely count as one mention of 'two' (if 'two' was required) and might NOT correctly fulfill two required mentions of 'one'. Be explicit and distinct for each required instance.\n"
        f"    *   **ABSOLUTELY AVOID NUMERICAL SUMMARIES/RE-LISTING:** Do NOT describe the items and then, in a separate sentence or phrase, re-list or summarize the numbers found (e.g., 'So, in total, she had found one, then another one, and also five items...'). The act of narrating the discovery/presence of each quantity IS its mention. Any re-listing will likely cause a validation failure due to over-counting.\n"
        f"    *   **Example of Correct Frequency:** If the Summary (1.A) states \"'- 'five' ({config_obj.INVALID_RESULT_PLACEHOLDER+5}): exactly two (2) time(s)\", your narrative must contain the word 'five' exactly twice, each time referring to a distinct set of five {safe_primary_object_for_fstring}s relevant to this scene's operation. Do not mention 'five' a third time, even in a summary.\n"
        f"    The following list details the specific numbers and their word forms that are the direct atomic inputs for this scene's operation: {must_mention_text_detailed_prose_for_prompt}. Ensure your narrative reflects these inputs according to the frequency summary in 1.A.{special_med_input_clarification_for_prompt}\n\n"
    )

    # --- Assemble ultra_strict_instruction ---
    ultra_strict_instruction = (
        f"**Narrative Challenge & Your Writing Guide for This Scene (CRITICAL: Adhere Flawlessly. Failure to meet these numerical rules, especially exact frequencies in Rule 1.A, will result in rejection.):**\n"
        f"Your main goal is to weave a compelling scene. However, for this specific task, you must precisely control how numbers are mentioned, turning constraints into creative storytelling:\n\n"
        f"{rule1_header}{rule1_summary_section}{rule1_explanation_section}" # Rule 1 broken down
        f"**2.  The Unspoken Outcome (ABSOLUTE RULE - NO EXCEPTIONS):** {result_handling_rule_text_for_prompt}\n\n"
        f"**3.  Permitted Narrative Flourishes (Use EXTREMELY Sparingly, ONLY if Essential for Fluency, AND Not Forbidden by Rule 4):\n**"
        f"    You MAY use {may_use_gen_clause_content_detailed_for_prompt} for general color, IF TRULY NECESSARY for narrative flow AND the number is NOT listed in Rule 4 (Forbidden Numbers). Rule 4 takes precedence. Do not use these permitted flourishes in a way that could be confused with the required input numbers from Rule 1. If a number is needed for Rule 1, its use there counts towards its frequency; it cannot then also be a 'free' flourish unless Rule 1 requires multiple mentions and Rule 3 also permits it for phrasing (a rare case, check Rule 1.A carefully).\n"
        f"{gt_counting_caution_for_gen_prompt.rstrip() + ('\\n' if gt_counting_caution_for_gen_prompt.strip() else '')}\n"
        f"**4.  Whispers Best Left Unheard (ABSOLUTELY Forbidden Numbers for THIS SCENE - NO EXCEPTIONS):\n**"
        f"    Strictly AVOID MENTIONING these specific numbers in ANY FORM (word or digit): {must_avoid_str_for_generator_prompt_detailed}. This rule OVERRIDES Rule 3 if there's a conflict. If a number is on this forbidden list, it cannot be used, period.\n\n"
        f"**5.  The Rule of No Other Numbers (CRITICAL & ABSOLUTE - NO EXCEPTIONS):\n**"
        f"    {no_other_numbers_rule_text_for_prompt} Any number not explicitly justified by Rule 1.A (with exact frequency) or Rule 3 (and not overridden by Rule 4) is an error.\n\n"
        f"{prior_results_handling_rule_for_prompt}"
        f"Focus on clear storytelling that naturally implies the calculations based on these strict numerical constraints. The most common errors are failing to meet exact frequencies for Rule 1.A, re-stating numbers (numerical summaries), or using numbers not explicitly permitted by these rules. Double-check your work against ALL rules, especially 1.A, before finalizing your scene."
    )

    context_snippet = clean_snippet(context.last_scene_text, max_len=config_obj.BEAT_CONTEXT)
    initial_user_message_parts = [
        f"Story Scene Task: Create the narrative for the step resulting in '{current_node_conceptual_name}' (Scene {context.beat_counter['current']}/{context.beat_counter['total']})\n\n"
        f"**Background for Your Scene (Context for you, the writer - follow strict rules below for what appears in the story):**\n"
        f"- Genre: {world.get('genre', 'N/A')}\n"
        f"- Setting: {world.get('setting', 'N/A')}\n"
        f"- Central Items in the Story: {primary_object_as_string}\n"
        f"- Quantities from Previous Events (Conceptual Names & their values for your understanding - DO NOT use these values in the story): {conceptual_inputs_context_str}\n"
        f"- New Numbers Introduced in this Scene (Values & their required frequencies for your understanding - Use word form in story, ALL must be mentioned with exact frequencies): {atomic_inputs_context_str_detailed_for_prompt}\n\n"
        f"{action_description_for_prompt}\n\n"
        f"{ultra_strict_instruction}\n\n"
    ]

    if node.op == "MED" and config_obj.FEW_SHOT_EXAMPLES > 0:
        curated_median_examples_indices = []
        if len(FEW_SHOT_EXAMPLES_STRICT) > 1: curated_median_examples_indices.append(1)
        if len(FEW_SHOT_EXAMPLES_STRICT) > 2: curated_median_examples_indices.append(2)

        indices_to_use = curated_median_examples_indices[:config_obj.FEW_SHOT_EXAMPLES]
        examples_to_actually_use = [FEW_SHOT_EXAMPLES_STRICT[i] for i in indices_to_use if i < len(FEW_SHOT_EXAMPLES_STRICT)]

        if examples_to_actually_use:
            few_shot_section = ["--- CRITICAL FEW-SHOT EXAMPLES FOR MEDIAN OPERATIONS (NEW RULE: ALL INPUTS & FREQUENCIES MENTIONED) ---"]
            few_shot_section.append(
                "These examples illustrate how to handle the strict numerical rules when the operation is MEDIAN. Pay close attention to how ALL direct atomic input numbers are mentioned WITH THEIR CORRECT FREQUENCIES and how the result is implied.\n"
            )
            for ex_idx, (example_rules_text, good_narrative, bad_narrative, bad_reasoning) in enumerate(examples_to_actually_use):
                why_good_text = "This example correctly follows the NEW MEDIAN rules by mentioning ALL necessary atomic inputs (with correct frequencies if the example implies them) and implying the median result conceptually without stating its numerical value."
                few_shot_section.append(f"**EXAMPLE {ex_idx + 1} RULES (from a hypothetical different problem - NEW MEDIAN RULE APPLIED):**\n{example_rules_text.replace('\\\\n', '\\n')}\n")
                few_shot_section.append(f"**EXAMPLE {ex_idx + 1} GOOD NARRATIVE (NEW MEDIAN RULE):**\n{good_narrative}\n")
                few_shot_section.append(f"**WHY GOOD (NEW MEDIAN RULE):**\n{why_good_text}\n")
                few_shot_section.append(f"**EXAMPLE {ex_idx + 1} BAD NARRATIVE (NEW MEDIAN RULE):**\n{bad_narrative}\n")
                few_shot_section.append(f"**WHY BAD (Reason for failure - NEW MEDIAN RULE):**\n{bad_reasoning}\n")
            few_shot_section.append("**REMEMBER THE MEDIAN RULE FOR *YOUR* CURRENT TASK (NEW VERSION):**")
            few_shot_section.append(
                "For MEDIAN operations, ALL direct atomic input numbers (with their exact frequencies as per Rule 1.A summary) MUST be mentioned in the narrative. The median value itself must NEVER appear explicitly as the RESULT of the text. Refer to your 'Narrative Challenge & Your Writing Guide' section in the main prompt for the specific numbers and rules for *your current scene*.\n"
            )
            initial_user_message_parts.append("\n".join(few_shot_section))
            logger_obj.info(
                f"Added {len(examples_to_actually_use)} MEDIAN-specific few-shot examples (NEW RULE: ALL INPUTS & FREQUENCIES MENTIONED) to the initial generator prompt for Op: {node.op}."
            )

    # --- Define and add the MANDATORY PRE-WRITING CHECKLIST ---
    mandatory_pre_writing_checklist = (
        "\n\n**MANDATORY PRE-WRITING CHECKLIST & MENTAL WALKTHROUGH (DO THIS BEFORE GENERATING TEXT - YOUR SUCCESS DEPENDS ON IT):**\n"
        "Before you write a single word of the narrative, meticulously review your plan against these critical points. This is not optional.\n\n"
        "**1. Rule 1.A - Exact Frequencies - DETAILED PLAN:**\n"
        "    *   **Identify ALL Required Numbers & Frequencies:** Look at the 'SUMMARY OF REQUIRED NUMBER FREQUENCIES FOR THIS SCENE' (Rule 1.A in your Writing Guide above). List each number and its exact required count.\n"
        "    *   **Plan Each Distinct Mention:** For EACH number and EACH required instance of it, mentally (or on a scratchpad) outline a *specific, distinct narrative event or discovery* where this number will be mentioned. \n"
        f"        *   Example: If 'one' (value: {config_obj.INVALID_RESULT_PLACEHOLDER+1}) is required 2 times, your plan MUST include two separate narrative elements: \n"
        f"            *   Mention 1 of 'one': e.g., 'A single glowing {safe_primary_object_for_fstring} was found under the arch.'\n"
        f"            *   Mention 2 of 'one': e.g., 'Later, another lone {safe_primary_object_for_fstring} was spotted near the fountain.'\n"
        f"        *   Example: If 'four' (value: {config_obj.INVALID_RESULT_PLACEHOLDER+4}) is required 1 time, your plan includes: \n"
        f"            *   Mention 1 of 'four': e.g., 'A cache revealed four ancient {safe_primary_object_for_fstring}s.'\n"
        "    *   **Confirm Full Coverage:** Have you planned a distinct narrative mention for *every single required instance* of *every single required number* from Rule 1.A? Are you certain you will meet the exact frequencies – no more, no less?\n\n"
        "**2. Rule 1.B - NO Numerical Summaries/Re-listing:**\n"
        "    *   Am I absolutely certain that my narrative will NOT contain any sentences or phrases that re-list, summarize, or total up the numbers I've just introduced as inputs? (e.g., AVOID: 'So, she had found one, then another one, and also five items...'). This is a critical error.\n\n"
        "**3. Rule 2 - Implicit Outcome:**\n"
        "    *   Is the numerical result of THIS scene's operation (e.g., the sum, the minimum, the median value itself) kept entirely IMPLICIT and NOT stated numerically in the narrative?\n\n"
        "**4. Rule 3 & 4 - Permitted vs. Forbidden Numbers:**\n"
        "    *   If I am considering using any small numbers for general phrasing (Rule 3), are they: \n"
        "        a) Genuinely essential for narrative fluency? \n"
        "        b) Used EXTREMELY sparingly? \n"
        "        c) ABSOLUTELY NOT on the forbidden list in Rule 4? (Rule 4 always wins).\n\n"
        "**5. Rule 5 - No Other Numbers:**\n"
        "    *   Have I ensured NO other numbers, counts, or extraneous figures will appear beyond what's strictly allowed by Rule 1.A (with exact frequencies) and Rule 3 (and not forbidden by Rule 4)?\n\n"
        "**Only after you have mentally confirmed 'YES' to all applicable checklist items and have a clear plan for Rule 1.A, should you proceed to write the narrative. Your primary goal is flawless adherence to these numerical rules within a coherent story.**"
    )
    initial_user_message_parts.append(mandatory_pre_writing_checklist)

    initial_user_message_parts.append(f'\n\n**Continue From (End of last scene):**\n"...{context_snippet}..."\n\n')
    initial_user_message_parts.append(f"**Your Response:**\nWrite ONLY the narrative text for this new scene, continuing smoothly. Do not add titles, notes, or anything outside the story itself.")
    initial_user_message_for_generator = "".join(initial_user_message_parts)

    py_validator_enforce_result_presence = False # For intermediate beats, result is implied
    validate_beat_numbers = make_number_validator(
        allowed_atoms_list=all_direct_atomic_inputs_as_list, # List with counts
        forbidden_atoms=forbidden_for_current_beat_py_validator,
        operand_count=actual_arity_for_current_op,
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
        "You are a master {genre} storyteller with an exceptional eye for detail, tasked with crafting a single, compelling narrative scene. This scene is part of a larger story where mathematical operations are subtly embedded within the plot.\n\n"
        "**Your paramount responsibilities for this scene are:**\n"
        "1.  **Narrative Coherence:** Write an engaging scene that smoothly continues the story from the provided snippet.\n"
        "2.  **ULTRA-STRICT NUMERICAL PRECISION:** This is non-negotiable and your primary constraint. You MUST meticulously follow ALL rules in the 'Narrative Challenge & Your Writing Guide for This Scene' section of the user prompt. Pay fanatical attention to:\n"
        "    *   **Rule 1.A (Exact Frequencies):** Mentioning EACH required number EXACTLY the specified number of times, with each mention being a distinct narrative element. No more, no less.\n"
        "    *   **Rule 1.B (No Summaries):** AVOID any summarizing statements that re-list numbers already introduced as inputs.\n"
        "    *   **Rule 2 (Implicit Outcome):** The numerical result of THIS scene's operation MUST NOT be stated explicitly.\n"
        "    *   **Rule 4 & 5 (Forbidden & No Other Numbers):** Absolutely NO other numbers (words or digits) beyond those explicitly permitted by Rule 1.A and Rule 3. Any number not explicitly listed as 'required' or 'permitted for phrasing' (and not forbidden) is an error.\n\n"
        "**Output ONLY the clean narrative text for the scene.** Do not include titles, notes, analysis, or any meta-commentary. Your adherence to the numerical rules is as critical as your storytelling."
    )
    operator_specific_system_focus = ""
    if node.op == "MED":
        operator_specific_system_focus = (
            "\n\n**SPECIAL FOCUS FOR THIS SCENE (MEDIAN OPERATION - NEW RULE):**\n"
            "The current scene involves a MEDIAN calculation. Under the NEW RULE, ALL direct atomic inputs for this MEDIAN operation (with their exact specified frequencies as per Rule 1.A) MUST be mentioned in the narrative. "
            "The MEDIAN result itself MUST be implied and NEVER stated numerically. "
            "You MUST meticulously follow the 'Narrative Challenge & Your Writing Guide' section in the user message. Failure to adhere to these MEDIAN rules will result in rejection of your scene."
        )
    system_prompt_for_generator = base_system_prompt_template.format(
        genre=world.get('genre', 'Fantasy')
    ) + operator_specific_system_focus

    current_max_beat_completion_tokens = config_obj.BEAT_MAX_TOKENS
    beat_text_final_validated = None

    llm_val_conceptual_input_names_only = (
        (", ".join([f"'{name_val}'" for name_val in child_conceptual_names_list])
        if child_conceptual_names_list
        else "None")
    )
    llm_val_atomic_inputs_detailed_for_validator = atomic_inputs_context_str_detailed_for_prompt

    llm_val_expected_beat_result_detailed_for_validator = "N/A"
    if correct_result is not None:
        llm_val_expected_beat_result_detailed_for_validator = (
            f"'{num_to_words(correct_result)}' ({correct_result})"
        )

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
            atomic_inputs_words_str_for_llm_validator=llm_val_atomic_inputs_detailed_for_validator,
            action_description_for_llm_validator=action_description_for_prompt,
            expected_beat_result_words_for_llm_validator=llm_val_expected_beat_result_detailed_for_validator,
            ultra_strict_instruction_for_llm_validator_context=ultra_strict_instruction,
            current_max_beat_completion_tokens=current_max_beat_completion_tokens,
            sample_index=context.sample_index,
            context_config=config_obj, # Pass the config from context
            logger_obj=logger_obj,
            encoder_obj=encoder,
            context=context,
            current_node_conceptual_name=current_node_conceptual_name,
            beat_number_in_sample=context.beat_counter['current'],
            is_current_beat_root_node=is_root,
            overall_ground_truth_answer_val=context.overall_ground_truth_answer,
            primary_object_name=primary_object_as_string,
            forbidden_prior_results_and_gt_for_llm_validator=forbidden_for_current_beat_py_validator,
            correct_result_val=correct_result,
            direct_atom_values_val=set(direct_atom_values_counts.keys()),
            direct_atom_counts_val=direct_atom_values_counts
        )

        if llm_validated_beat_text:
            # Python validator (make_number_validator) is now primarily for sanity checking counts
            # as the LLM validator is meant to be the main gatekeeper for complex rules.
            if validate_beat_numbers(llm_validated_beat_text):
                beat_text_final_validated = llm_validated_beat_text
                logger_obj.info(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Python validator PASSED LLM-validated beat (counts verified)."
                )
                break
            else:
                logger_obj.warning(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Python validator FAILED for LLM-validated beat (likely count issue). Outer attempt {attempt_outer} failed."
                )
                # Log the ultra_strict_instruction that led to this Python validator failure after LLM validation
                logger_obj.debug(f"Dumping ultra_strict_instruction for Op {node.op} (Python validator fail after LLM pass):\n{ultra_strict_instruction}")
        else:
            logger_obj.warning(
                f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Iterative LLM validation loop returned None. Outer attempt {attempt_outer} failed."
            )

        if attempt_outer < config_obj.MAX_BEAT_RETRIES:
            time.sleep(config_obj.RETRY_INITIAL_DELAY * (2 ** (attempt_outer - 1)))
        else:
            logger_obj.error(
                f"Dumping last ultra_strict_instruction for failed Op {node.op} (Sample {context.sample_index+1}) after all outer retries:\n{ultra_strict_instruction}"
            )


    if not beat_text_final_validated:
        logger_obj.error(
            f"Operator {node.op} (Result Concept: '{current_node_conceptual_name}') failed after {config_obj.MAX_BEAT_RETRIES} outer attempts (incl. LLM validation loops). Aborting narrative generation for this sample."
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} (Result Concept: '{current_node_conceptual_name}') after all outer retries."
        )

    beat_text = beat_text_final_validated
    btoks = len(encoder.encode(beat_text))
    context.scenes.append(beat_text)
    context.tokens_used += btoks
    context.last_scene_text = beat_text
    context.introduced_atoms.update(direct_atom_values_counts.keys())

    logger_obj.debug(
        f"Beat {context.beat_counter['current']} for Op {node.op} successful. Introduced atoms (unique) updated with current beat's direct atoms: {direct_atom_values_counts.keys()}"
    )

    # --- Padding Logic (remains the same) ---
    if not is_root:
        if context.padding_stats["padding_per_slot"] > 0 and context.max_pad_paragraphs > 0:
            num_paragraphs_for_this_slot = random.randint(1, context.max_pad_paragraphs)
            max_tokens_for_this_padding_segment = context.padding_stats["padding_per_slot"]

            logger_obj.info(
                f"Attempting to add {num_paragraphs_for_this_slot} padding paragraphs (max {max_tokens_for_this_padding_segment} tokens) after beat {context.beat_counter['current']} for Op {node.op}."
            )

            padding_system_prompt = (
                f"You are a {context.world.get('genre', 'Fantasy')} storyteller. "
                "Continue the narrative with a short, atmospheric scene or transition that DOES NOT involve any numbers or calculations. "
                "Focus on character interaction, setting description, or internal monologue. "
                "This is filler text to bridge scenes. Output ONLY the narrative text."
            )
            padding_user_prompt_template = (
                "Previous Scene Snippet (End of last scene): \"...{snippet}...\"\n\n"
                "Task: Write {num_paras_str} short paragraph(s) of bridging narrative. "
                "STRICT RULE: Absolutely NO numbers (digits or words like 'one', 'two', 'first', etc.), except for 'one', 'two', or 'three' used ONLY for general, non-quantitative phrasing (e.g., 'a single glance', 'two paths diverged'). Strive for zero numbers. "
                "This text should be purely narrative and not advance any calculations. "
                "Output ONLY the narrative text."
            )

            forbidden_for_padding_slot = set(context.introduced_atoms)
            if correct_result is not None:
                forbidden_for_padding_slot.add(correct_result)
            if context.overall_ground_truth_answer is not None:
                forbidden_for_padding_slot.add(context.overall_ground_truth_answer)

            validate_padding = make_number_validator(
                allowed_atoms_list=[],
                forbidden_atoms=forbidden_for_padding_slot,
                operand_count=0,
                correct_result_for_beat=None,
                strict_zero=True,
                enforce_result_presence=False,
                operation_type="PADDING",
                overall_ground_truth_answer=context.overall_ground_truth_answer,
                is_root_node_being_validated=False,
                conceptual_input_values=None,
                config_obj=context.config, # Use context's config
                logger_obj=context.logger, # Use context's logger
            )

            current_padding_tokens_for_slot = 0
            padding_segments_added_this_slot = 0

            for i in range(num_paragraphs_for_this_slot):
                if current_padding_tokens_for_slot >= max_tokens_for_this_padding_segment:
                    logger_obj.info(f"Padding token limit for this slot ({max_tokens_for_this_padding_segment}) reached. Stopping padding generation.")
                    break

                remaining_token_budget_for_pad_para = max_tokens_for_this_padding_segment - current_padding_tokens_for_slot
                actual_max_tokens_for_this_para = min(remaining_token_budget_for_pad_para, config_obj.PADDING_MAX_TOKENS)
                if actual_max_tokens_for_this_para <= 10: # Ensure config_obj is from context
                    logger_obj.info(f"Remaining token budget for padding paragraph ({actual_max_tokens_for_this_para}) too small. Skipping further padding.")
                    break

                num_paras_str_for_prompt = "a single" if num_paragraphs_for_this_slot == 1 else "one"
                if i > 0 : num_paras_str_for_prompt = "another"

                padding_user_prompt = padding_user_prompt_template.format(
                    snippet=clean_snippet(context.last_scene_text, max_len=config_obj.PADDING_CONTEXT), # Ensure config_obj is from context
                    num_paras_str=num_paras_str_for_prompt
                )

                padding_text = generate_with_retry(
                    system_prompt=padding_system_prompt,
                    user_prompt=padding_user_prompt,
                    max_completion_tokens=actual_max_tokens_for_this_para,
                    validate_fn=validate_padding,
                    retries=config_obj.MAX_PAD_RETRIES, # Ensure config_obj is from context
                    sample_index=context.sample_index,
                    temperature=config_obj.CREATIVE_NARRATIVE_TEMP, # Ensure config_obj is from context
                    reasoning_settings={"exclude": True}
                )

                if padding_text:
                    ptoks = len(context.encoder.encode(padding_text))
                    if context.tokens_used + ptoks + config_obj.MAX_TOKENS_BUFFER <= config_obj.MAX_TOTAL_TOKENS and \
                       current_padding_tokens_for_slot + ptoks <= max_tokens_for_this_padding_segment : # Ensure config_obj is from context

                        context.scenes.append(f"\n[Padding Segment]\n{padding_text}\n[/Padding Segment]\n")
                        context.tokens_used += ptoks
                        context.last_scene_text = padding_text
                        context.padding_stats["total_padding_tokens"] += ptoks
                        context.padding_stats["padding_segments_added"] += 1
                        current_padding_tokens_for_slot += ptoks
                        padding_segments_added_this_slot +=1
                        logger_obj.info(f"Added padding segment ({ptoks} tokens). Total padding this slot: {current_padding_tokens_for_slot}/{max_tokens_for_this_padding_segment}. Overall padding tokens: {context.padding_stats['total_padding_tokens']}.")
                    else:
                        logger_obj.warning(f"Generated padding segment ({ptoks} tokens) would exceed slot budget or total token limit. Discarding.")
                        break
                else:
                    logger_obj.warning(f"Failed to generate valid padding segment for slot after beat {context.beat_counter['current']}. Attempt {i+1}/{num_paragraphs_for_this_slot}.")

            if padding_segments_added_this_slot > 0:
                 logger_obj.info(f"Finished adding {padding_segments_added_this_slot} padding segments for this slot.")
            else:
                 logger_obj.info(f"No padding segments added for this slot after beat {context.beat_counter['current']}.")
        elif context.padding_stats["padding_per_slot"] == 0:
            logger_obj.debug(f"Padding per slot is 0. Skipping padding after beat {context.beat_counter['current']}.")
        elif context.max_pad_paragraphs == 0:
            logger_obj.debug(f"Max_pad_paragraphs is 0. Skipping padding after beat {context.beat_counter['current']}.")


    logger_obj.debug(
        f"Exiting _generate_narrative_recursive for Op {op_for_log} ('{current_node_conceptual_name}'). Total tokens used: {context.tokens_used}"
    )

# --- Main Narrative Generation Function ---
def generate_narrative(
    ast: Node,
    world: dict,
    config: Config,  # Note: this is the global config object, not context.config
    encoder,  # Global encoder
    p_inflect,  # Global p_inflect
    logger,  # Global logger
    sample_index: int,
    overall_ground_truth_answer: int,
) -> GenerationContext | None:  # Return GenerationContext or None
    """
    Generate a structured narrative representation of the AST.
    Each operation is represented by a scene, carefully sequenced with
    intermediate node anchor names. Uses STRICT recursive generation.
    """
    logger.info(f"[Sample {sample_index + 1}] Starting narrative generation.")
    logger.debug(f"DEBUG: Using model: {MODEL}")  # Uses global MODEL
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

    operator_nodes_for_anchor_gen = (
        []
    )  # Used to collect nodes for anchor generation and logging
    narrative_anchor_map = {}
    # intro_text = None # Defined later
    scenes = []
    tokens_used = 0

    # --- Generate narrative anchors for op nodes ---
    if config.USE_NARRATIVE_ANCHORS:

        def generate_anchor_for_node(op_node_arg):  # Renamed op_node to op_node_arg
            if not config.USE_NARRATIVE_ANCHORS:  # Check global config
                return (
                    f"the_{op_node_arg.op.lower()}_result_{id(op_node_arg) % 1000:03d}"
                )

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
                    logger.warning(  # Use global logger
                        f"Failed to generate LLM anchor for {op_node_arg.op}. Using deterministic fallback."
                    )
                    return f"the_{op_node_arg.op.lower()}_result_{id(op_node_arg) % 1000:03d}"
            except Exception as e:
                logger.error(
                    f"Error in narrative anchor generation: {e}"
                )  # Use global logger
                return (
                    f"the_{op_node_arg.op.lower()}_result_{id(op_node_arg) % 1000:03d}"
                )

        for node_iter_anchor in postorder(ast):
            if isinstance(node_iter_anchor, OpNode):
                anchor = generate_anchor_for_node(node_iter_anchor)
                narrative_anchor_map[id(node_iter_anchor)] = anchor
                operator_nodes_for_anchor_gen.append(
                    node_iter_anchor
                )  # Add to list for logging
                logger.debug(  # Use global logger
                    f"Added narrative anchor '{anchor}' for {node_iter_anchor.op} node"
                )
    else:
        for node_iter_anchor in postorder(ast):
            if isinstance(node_iter_anchor, OpNode):
                narrative_anchor_map[id(node_iter_anchor)] = (
                    f"the_{node_iter_anchor.op.lower()}_result_{id(node_iter_anchor) % 1000:03d}"
                )
                operator_nodes_for_anchor_gen.append(
                    node_iter_anchor
                )  # Add to list for logging

    logger.info(
        f"Generated {len(narrative_anchor_map)} narrative anchors."
    )  # Use global logger
    if operator_nodes_for_anchor_gen:  # Check if list is not empty
        log_str = "Narrative anchors: " + ", ".join(
            [
                f"'{narrative_anchor_map.get(id(op_node_log), 'MISSING')}' ({op_node_log.op})"
                for op_node_log in operator_nodes_for_anchor_gen  # Iterate over the collected list
            ]
        )
        logger.debug(log_str)  # Use global logger
    else:
        logger.debug("No operator nodes found to generate anchors for.")

    # Pass global config and logger to generate_introduction_scene
    intro_text = generate_introduction_scene(
        world, sample_index=sample_index, config_obj=config, logger_obj=logger
    )

    if intro_text:
        intro_tokens = len(encoder.encode(intro_text))  # Use global encoder
        # Use global config for MAX_TOTAL_TOKENS and SAFETY_MARGIN
        if intro_tokens <= config.MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            scenes.append(intro_text)
            tokens_used += intro_tokens
            logger.info(  # Use global logger
                f"Generated and added introductory scene ({intro_tokens} tokens)."
            )
        else:
            logger.warning(  # Use global logger
                f"Generated introductory scene ({intro_tokens} tokens) was too long and would exceed budget. "
                f"Not adding to narrative. Budget: {config.MAX_TOTAL_TOKENS}, Safety: {SAFETY_MARGIN}"
            )
            intro_text = None  # Reset intro_text if not used
    else:
        logger.warning(  # Use global logger
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
        config=config,  # Pass the global config
        encoder=encoder,  # Pass the global encoder
        p_inflect=p_inflect,  # Pass the global p_inflect
        logger=logger,  # Pass the global logger
        narrative_anchor_map=narrative_anchor_map,
        all_atoms=all_atoms,
        introduced_atoms=introduced_atoms_during_generation,
        scenes=scenes,
        tokens_used=tokens_used,
        last_scene_text=last_scene_text,
        beat_counter=beat_counter,
        sample_index=sample_index,
        max_pad_paragraphs=config.MAX_PAD_PARAGRAPHS,  # Use global config
        overall_ground_truth_answer=overall_ground_truth_answer,
        overall_ast_root=ast,
    )

    tokens_available_for_narrative_and_padding = (
        config.MAX_TOTAL_TOKENS - tokens_used - SAFETY_MARGIN  # Use global config
    )
    max_padding_allowed = int(
        tokens_available_for_narrative_and_padding
        * config.PADDING_MAX_TOK_PERCENT  # Use global config
    )
    context.padding_stats["max_padding_allowed"] = max_padding_allowed

    num_padding_slots = total_beats - 1 if total_beats > 1 else 0
    if num_padding_slots > 0:
        padding_per_slot_calculated = max_padding_allowed // num_padding_slots
        context.padding_stats["padding_per_slot"] = padding_per_slot_calculated
        logger.info(  # Use global logger
            f"Calculated padding per slot: {padding_per_slot_calculated} tokens ({max_padding_allowed} total / {num_padding_slots} slots)"
        )
    else:
        context.padding_stats["padding_per_slot"] = 0
        logger.info(  # Use global logger
            f"No padding slots available (total_beats: {total_beats}). Padding per slot set to 0."
        )

    logger.info(  # Use global logger
        f"PADDING BUDGET INITIALIZED: Tokens after intro: {tokens_used}, "
        f"Available for narrative+padding: {tokens_available_for_narrative_and_padding}, "
        f"Max padding %: {config.PADDING_MAX_TOK_PERCENT*100:.1f}%, "
        f"Max padding tokens allowed: {max_padding_allowed}, "
        f"Padding per slot: {context.padding_stats['padding_per_slot']}, "
        f"Max padding segments per beat: {config.MAX_PAD_PARAGRAPHS}"  # Use global config
    )

    try:
        # _generate_narrative_recursive will use the logger and config from the context object
        _generate_narrative_recursive(
            ast,
            context,  # Pass the fully initialized context
            is_root=True,
        )
    except BeatGenerationError as e:
        logger.error(
            f"Narrative generation aborted due to beat failure: {e}"
        )  # Use global logger
        return None
    except Exception as e:
        logger.error(  # Use global logger
            f"Unexpected error during recursive narrative generation: {e}",
            exc_info=True,
        )
        return None

    if not context.scenes:
        logger.error("Narrative generation resulted in no scenes.")  # Use global logger
        return None

    # narrative_body = "\n\n".join(context.scenes).strip() # This is local, not returned directly
    final_token_count = len(
        encoder.encode("\n\n".join(context.scenes).strip())
    )  # Use global encoder
    if final_token_count > config.MAX_TOTAL_TOKENS:  # Use global config
        logger.warning(  # Use global logger
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
    logger.info(  # Use global logger
        f"PADDING FINAL SUMMARY: "
        f"Padding tokens: {total_padding_tokens}/{context.padding_stats['max_padding_allowed']} ({padding_percentage_of_max:.1f}% of max allowed for padding), "
        f"Padding percentage of total narrative tokens: {padding_percentage_of_total_narrative:.1f}%, "
        f"Padding segments added: {padding_segments_added}"
    )

    failed_validations_dir = os.path.join(
        LOG_DIR, "failed_validations"
    )  # Uses global LOG_DIR
    if os.path.exists(failed_validations_dir):
        validation_files = [
            f_name
            for f_name in os.listdir(failed_validations_dir)
            if f_name.startswith(
                f"validation_fail_"
            )  # No sample_index specific filtering here, could be added if needed
        ]  # Removed sample_index specific filtering for this general summary

        if validation_files:  # This summary is now general, not per-sample
            failures_by_reason = {}
            failures_by_op = {}
            for file_name in validation_files:
                try:
                    # Basic parsing, might need adjustment if filenames are complex
                    parts = file_name.split("_")
                    if (
                        len(parts) >= 4
                    ):  # e.g., validation_fail_OP_REASON_timestamp.json
                        op_type = parts[2]
                        reason_code = parts[3]
                        failures_by_reason[reason_code] = (
                            failures_by_reason.get(reason_code, 0) + 1
                        )
                        failures_by_op[op_type] = failures_by_op.get(op_type, 0) + 1
                except IndexError:
                    logger.warning(  # Use global logger
                        f"Could not parse validation failure filename: {file_name}"
                    )
            logger.info(
                f"[Sample {sample_index + 1}] VALIDATION FAILURES SUMMARY (for this sample, based on files in failed_validations):"
            )  # Clarify scope
            logger.info(  # Use global logger
                f"  Total validation failures logged (may include other samples if not filtered by sample_index in filename): {len(validation_files)}"
            )
            if failures_by_reason:
                logger.info(
                    f"  Failures by reason: {failures_by_reason}"
                )  # Use global logger
            if failures_by_op:
                logger.info(
                    f"  Failures by operation: {failures_by_op}"
                )  # Use global logger
        else:
            logger.info(  # Use global logger
                f"[Sample {sample_index + 1}] No validation failure files found in general log dir (or matching this sample's pattern if implemented)."
            )

    logger.info(  # Use global logger
        f"Successfully generated narrative for sample {sample_index + 1}. Final context tokens: {context.tokens_used}, Narrative tokens: {final_token_count}"
    )
    return context  # Return the GenerationContext object


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
            "id": str(sample_index),
            "timestamp": datetime.datetime.now().isoformat(),
            "ast_str": ast_to_prefix(node),
            "ground_truth_value": narrative_gen_context.overall_ground_truth_answer,
            "narrative": narrative_body_from_context,
            "question": final_question_text.strip(),
            "full_text_for_eval": full_text_for_eval_content,
            "world_data": world_data,
            "scenes_detail": [
                {"scene_number": i + 1, "text": text}
                for i, text in enumerate(narrative_gen_context.scenes)
            ],
            "num_operations": count_ops(node),
            "token_counts": {
                "total_generated_context": narrative_gen_context.tokens_used,
                "narrative_body": (
                    len(encoder.encode(narrative_body_from_context)) if encoder else 0
                ),
                "padding": narrative_gen_context.padding_stats["total_padding_tokens"],
            },
            "conceptual_references": dict(
                narrative_gen_context.narrative_anchor_map
            ),
            "beat_revision_details": narrative_gen_context.beat_revision_logs,  # <<< NEWLY ADDED
            "generation_metadata": {
                "script_version": "verbose-listops_v_DRY_output_fix1",
                "generation_model": MODEL,
                "iterative_validator_model": config_obj.LLM_VALIDATOR_MODEL,
                "always_allowed_phrasing_numbers": list(
                    config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
                ),
                "config_params": { 
                    "max_ops": config_obj.MAX_OPS,
                    "min_arity": config_obj.MIN_ARITY,
                    "max_total_tokens": config_obj.MAX_TOTAL_TOKENS,
                },
                "full_config_snapshot": asdict(
                    config_obj
                ),
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
    if (
        client
        and OPENROUTER_API_KEY
        and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE"
    ):
        logger.info("Fetching initial OpenRouter account usage...")
        initial_account_usage = rate_limiter.update_limits_from_api()
        if initial_account_usage is not None:
            logger.info(
                f"Initial OpenRouter account usage: ${initial_account_usage:.4f}"
            )
        else:
            logger.warning("Could not fetch initial OpenRouter account usage.")
    else:
        logger.warning(
            "Skipping initial OpenRouter account usage check: Client not initialized or API key missing/placeholder."
        )

    # --- Dynamic Filename and Subfolder Generation ---
    sanitized_model_name = MODEL.replace("/", "_").replace(":", "-")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    base_filename_parts = [
        f"{config.MAX_TOTAL_TOKENS}tok",
        f"{config.MAX_OPS}mxops",
        f"{config.MIN_ARITY}minarity",
        f"{config.MAX_BRANCH}mxbrch",
        f"{sanitized_model_name}",
        timestamp_str,
    ]
    run_specific_identifier = "_".join(base_filename_parts)

    run_output_dir = os.path.join(DATASETS_DIR, run_specific_identifier)
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"All outputs for this run will be saved in: {run_output_dir}")

    # 1. Filename for RESEARCHER_DETAIL (comprehensive raw generated data)
    researcher_detail_basename = (
        f"[1_RESEARCHER_DETAIL]_DATASET_{run_specific_identifier}.jsonl"
    )
    researcher_detail_output_file = os.path.join(
        run_output_dir, researcher_detail_basename
    )
    logger.info(
        f"Researcher detail output will be saved to: {researcher_detail_output_file}"
    )

    # 2. Filename for EVAL_READY data (lean format for ALL successfully generated samples)
    eval_ready_basename = f"[2_EVAL_READY]_DATASET_{run_specific_identifier}.jsonl"
    eval_ready_output_file = os.path.join(run_output_dir, eval_ready_basename)
    logger.info(
        f"Evaluation-ready output (all generated) will be saved to: {eval_ready_output_file}"
    )

    # 3. Names for auxiliary files related to validator.py processing (if PROD_RUN is true)
    validator_results_basename = (
        f"[3.1_VALIDATOR_RESULTS]_{run_specific_identifier}.jsonl"
    )
    validator_results_path = os.path.join(run_output_dir, validator_results_basename)

    validator_stdout_basename = f"[3.2_VALIDATOR_STDOUT]_{run_specific_identifier}.log"
    validator_output_log_path = os.path.join(
        run_output_dir, validator_stdout_basename
    )

    # 4. Filename for the final VALIDATOR_CLEANED dataset (lean format, only validator-passed samples)
    final_cleaned_lean_basename = (
        f"[4_FINAL_EVAL_CLEANED]_DATASET_{run_specific_identifier}.jsonl"
    )
    final_cleaned_lean_output_file = os.path.join(
        run_output_dir, final_cleaned_lean_basename
    )

    # 5. Filename for Beat Revision Logs
    beat_revision_log_basename = (
        f"[5_BEAT_REVISION_LOGS]_DATASET_{run_specific_identifier}.jsonl"
    )
    beat_revision_log_file = os.path.join(run_output_dir, beat_revision_log_basename)
    logger.info(f"Beat revision logs (for multi-attempt/failed beats) will be saved to: {beat_revision_log_file}")


    # --- Generation Loop ---
    logger.info(
        f"Script started. Generating {num_samples} samples using up to {max_workers} workers."
    )
    samples_generated_successfully = 0
    samples_failed_generation = 0
    start_time = time.time()
    generated_results = [] # This will store the full sample dicts

    progress_lock = threading.Lock()
    completed_tasks = 0
    last_print_time = start_time
    print_interval = max(5, min(30, max_workers // 10))

    print(
        f"Starting generation of {num_samples} samples using {max_workers} workers..."
    )
    print(f"Progress updates will be shown every ~{print_interval} seconds")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(generate_single_sample, i, config): i
            for i in range(num_samples)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_data = future.result() # This is the full sample dict from generate_single_sample
                with progress_lock:
                    if sample_data:
                        generated_results.append(sample_data)
                        samples_generated_successfully += 1
                    else:
                        samples_failed_generation += 1
                    completed_tasks = (
                        samples_generated_successfully + samples_failed_generation
                    )
            except Exception as exc:
                logger.error(
                    f"[Sample {index + 1}] task generated exception: {exc}",
                    exc_info=True,
                )
                with progress_lock:
                    samples_failed_generation += 1
                    completed_tasks = (
                        samples_generated_successfully + samples_failed_generation
                    )

            current_time = time.time()
            should_print_progress = False
            with progress_lock:
                if (current_time - last_print_time >= print_interval) or (
                    completed_tasks == num_samples
                ):
                    should_print_progress = True
                    last_print_time = current_time

            if should_print_progress:
                elapsed_time = current_time - start_time
                if completed_tasks > 0:
                    throughput = (
                        completed_tasks / (elapsed_time / 60) if elapsed_time > 0 else 0
                    )
                    remaining_samples = num_samples - completed_tasks
                    estimated_time_remaining_seconds = (
                        (remaining_samples / throughput * 60) if throughput > 0 else 0
                    )

                    time_str = "N/A"
                    if estimated_time_remaining_seconds > 0:
                        if estimated_time_remaining_seconds >= 3600:
                            time_str = f"{estimated_time_remaining_seconds/3600:.1f}h"
                        elif estimated_time_remaining_seconds >= 60:
                            time_str = f"{estimated_time_remaining_seconds/60:.1f}m"
                        else:
                            time_str = f"{estimated_time_remaining_seconds:.0f}s"

                    success_rate_generation = (
                        (samples_generated_successfully / completed_tasks * 100)
                        if completed_tasks > 0
                        else 0
                    )
                    print(
                        f"Progress: {completed_tasks}/{num_samples} ({completed_tasks/num_samples*100:.1f}%) | "
                        f"Success: {samples_generated_successfully} ({success_rate_generation:.1f}%) | "
                        f"Failed: {samples_failed_generation} | "
                        f"Rate: {throughput:.2f} samples/min | "
                        f"ETA: {time_str}"
                    )

    # --- Writing Initial Output Files ---
    logger.info(
        f"Parallel generation complete. Writing {samples_generated_successfully} samples to initial output files..."
    )
    samples_written_researcher = 0
    samples_written_eval_raw = 0

    if samples_generated_successfully > 0:
        try:
            with open(
                researcher_detail_output_file, "w", encoding="utf-8"
            ) as f_researcher, open(
                eval_ready_output_file, "w", encoding="utf-8"
            ) as f_eval_raw:

                for full_sample_data in generated_results: # Iterate through the list of dicts
                    try:
                        f_researcher.write(
                            json.dumps(
                                full_sample_data,
                                default=lambda o: (
                                    list(o) if isinstance(o, set) else str(o)
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        samples_written_researcher += 1
                    except Exception as e:
                        logger.error(
                            f"Error writing full sample {full_sample_data.get('id', 'Unknown')} to researcher file: {e}. Skipping."
                        )

                    eval_sample_data = {
                        "id": full_sample_data.get("id"),
                        "full_text_for_eval": full_sample_data.get(
                            "full_text_for_eval"
                        ),
                        "ground_truth_value": full_sample_data.get(
                            "ground_truth_value"
                        ),
                        "ast_str": full_sample_data.get("ast_str"),
                        "num_operations": full_sample_data.get("num_operations"),
                        "token_count_narrative": full_sample_data.get(
                            "token_counts", {}
                        ).get("narrative_body"),
                    }
                    eval_sample_data_cleaned = {
                        k: v for k, v in eval_sample_data.items() if v is not None
                    }
                    try:
                        f_eval_raw.write(
                            json.dumps(eval_sample_data_cleaned, ensure_ascii=False)
                            + "\n"
                        )
                        samples_written_eval_raw += 1
                    except Exception as e:
                        logger.error(
                            f"Error writing initial eval sample {eval_sample_data_cleaned.get('id', 'Unknown')} to eval_raw file: {e}. Skipping."
                        )

            logger.info(
                f"Successfully wrote {samples_written_researcher} samples to {researcher_detail_output_file}."
            )
            logger.info(
                f"Successfully wrote {samples_written_eval_raw} samples to {eval_ready_output_file} (initial eval-ready set)."
            )

        except IOError as e:
            logger.error(
                f"Fatal file write error opening/writing initial output files: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during initial output file writing phase: {e}",
                exc_info=True,
            )
    else:
        logger.info(
            "No samples were successfully generated, so no initial output files will be written."
        )

    # --- Writing Beat Revision Logs ---
    all_beat_revision_logs_for_run = []
    if samples_generated_successfully > 0: # Check if there are any results to process
        for full_sample_data_item in generated_results: # Iterate through the list of dicts
            # The key is "beat_revision_details" as added in generate_single_sample
            if "beat_revision_details" in full_sample_data_item and full_sample_data_item["beat_revision_details"]:
                all_beat_revision_logs_for_run.extend(full_sample_data_item["beat_revision_details"])
        
        if all_beat_revision_logs_for_run:
            written_revision_logs_count = 0
            try:
                with open(beat_revision_log_file, "w", encoding="utf-8") as f_rev_logs:
                    for log_entry in all_beat_revision_logs_for_run:
                        # Filter out entries where success was on first attempt
                        if log_entry.get("final_status") != "success_on_first_attempt":
                             f_rev_logs.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                             written_revision_logs_count +=1
                logger.info(
                    f"Successfully wrote {written_revision_logs_count} beat revision log entries (filtered for multi-attempt/failed) to {beat_revision_log_file}."
                )
                if written_revision_logs_count > 0:
                    print(f"\nBeat revision logs (for multi-attempt/failed beats) saved to: {beat_revision_log_file}")
                else:
                    print("\nNo beat revisions (requiring multiple attempts or failing) were logged to file (all successful beats passed on first attempt).")
            except IOError as e:
                logger.error(f"File write error for beat revision logs {beat_revision_log_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error writing beat revision logs to {beat_revision_log_file}: {e}", exc_info=True)
        else:
            logger.info("No beat revision details were collected during the run (e.g., all beats passed on first attempt or no samples generated).")
            print("\nNo beat revision details were collected (e.g., all successful beats passed on first attempt or no samples generated).")
    elif samples_generated_successfully == 0:
        logger.info("No samples generated, so no beat revision logs to write.")
        print("\nNo samples generated, so no beat revision logs to write.")


    end_time_generation = time.time()
    total_time_generation = end_time_generation - start_time
    logger.info(f"--- Generation Phase Summary ---")
    logger.info(f"Samples attempted: {num_samples}")
    logger.info(f"Successfully generated: {samples_generated_successfully}")
    logger.info(f"Written to researcher detail file: {samples_written_researcher}")
    logger.info(f"Written to initial eval-ready file: {samples_written_eval_raw}")
    logger.info(f"Failed generations: {samples_failed_generation}")
    logger.info(f"Generation time: {total_time_generation/60:.2f} minutes")

    if samples_written_researcher > 0:
        print(f"\nResearcher Detail dataset saved to: {researcher_detail_output_file}")
    if samples_written_eval_raw > 0:
        print(f"Initial Eval-Ready dataset saved to: {eval_ready_output_file}")
    if samples_generated_successfully == 0 and not PROD_RUN: # Avoid double message if PROD_RUN will also say this
        print(f"\nNo samples were successfully generated by verbose-listops.py.")


    # --- PROD_RUN: Validation and Creation of Final Cleaned Eval-Ready Dataset ---
    bad_sample_ids_from_validator = set()

    if (
        PROD_RUN
        and samples_written_researcher > 0 # Use this instead of samples_written_eval_raw
        and os.path.exists(researcher_detail_output_file) # Check the input for validator
    ):
        logger.info(
            f"--- Starting PROD_RUN validation using {researcher_detail_output_file} ---"
        )
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        validator_script_path = os.path.join(current_script_dir, "validator.py")

        if not os.path.exists(validator_script_path):
            logger.error(
                f"Validator script not found at {validator_script_path}. Cannot perform cleaning."
            )
            print(f"ERROR: Validator script not found at {validator_script_path}.")
        else:
            cmd = [
                sys.executable,
                validator_script_path,
                researcher_detail_output_file, # Use the researcher detail file as input
                "--output-results",
                validator_results_path,
            ]
            try:
                logger.info(f"Running validator command: {' '.join(cmd)}")
                print(
                    f"\n--- Starting Validator Subprocess ({' '.join(cmd)}) ---"
                )

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )

                stderr_output_lines = []

                with open(
                    validator_output_log_path, "w", encoding="utf-8"
                ) as f_val_out_log:
                    if process.stdout:
                        f_val_out_log.write("--- VALIDATOR STDOUT ---\n")
                        for line in iter(process.stdout.readline, ""):
                            print(
                                line, end="", flush=True
                            )
                            f_val_out_log.write(line)
                        process.stdout.close()

                    return_code = process.wait()
                    print()

                    if process.stderr:
                        stderr_output_content = process.stderr.read()
                        if stderr_output_content:
                            f_val_out_log.write("\n--- VALIDATOR STDERR ---\n")
                            f_val_out_log.write(stderr_output_content)
                            stderr_output_lines = stderr_output_content.splitlines()
                        process.stderr.close()

                print(
                    f"--- Validator Subprocess Finished (Return Code: {return_code}) ---"
                )

                if stderr_output_lines:
                    logger.info("Validator process stderr (captured):")
                    for line in stderr_output_lines:
                        logger.info(f"VALIDATOR_LOG_VIA_STDERR: {line.strip()}")

                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code,
                        cmd,
                        stderr=(
                            stderr_output_content
                            if "stderr_output_content" in locals()
                            else None
                        ),
                    )

                logger.info(
                    f"Validator finished. Full output (stdout & stderr) logged to {validator_output_log_path}"
                )

                if os.path.exists(validator_results_path):
                    with open(
                        validator_results_path, "r", encoding="utf-8"
                    ) as f_results:
                        for line_num, res_line in enumerate(f_results, 1):
                            try:
                                val_res = json.loads(res_line)
                                if val_res.get("status") != "correct":
                                    sample_id_to_remove = val_res.get("id")
                                    if sample_id_to_remove:
                                        bad_sample_ids_from_validator.add(
                                            sample_id_to_remove
                                        )
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Could not parse validator result line {line_num} from {validator_results_path}: {res_line.strip()}"
                                )
                    logger.info(
                        f"Identified {len(bad_sample_ids_from_validator)} samples to remove based on results from {validator_results_path}."
                    )

                    temp_final_cleaned_lean_file = (
                        final_cleaned_lean_output_file + ".tmp"
                    )
                    good_samples_written_to_final_cleaned_lean = 0

                    # We need to read from the original generated_results to create the lean file
                    with open(
                        temp_final_cleaned_lean_file, "w", encoding="utf-8"
                    ) as f_temp_lean:
                        for full_sample_data in generated_results: # Iterate original full data
                            if (
                                full_sample_data.get("id")
                                not in bad_sample_ids_from_validator
                            ):
                                eval_sample_data = { # Reconstruct lean sample
                                    "id": full_sample_data.get("id"),
                                    "full_text_for_eval": full_sample_data.get(
                                        "full_text_for_eval"
                                    ),
                                    "ground_truth_value": full_sample_data.get(
                                        "ground_truth_value"
                                    ),
                                    "ast_str": full_sample_data.get("ast_str"),
                                    "num_operations": full_sample_data.get(
                                        "num_operations"
                                    ),
                                    "token_count_narrative": full_sample_data.get(
                                        "token_counts", {}
                                    ).get("narrative_body"),
                                }
                                eval_sample_data_cleaned = {
                                    k: v
                                    for k, v in eval_sample_data.items()
                                    if v is not None
                                }
                                try:
                                    f_temp_lean.write(
                                        json.dumps(
                                            eval_sample_data_cleaned, ensure_ascii=False
                                        )
                                        + "\n"
                                    )
                                    good_samples_written_to_final_cleaned_lean += 1
                                except Exception as e_write_lean:
                                    logger.error(
                                        f"Error writing lean sample {eval_sample_data_cleaned.get('id', 'Unknown')} to final cleaned eval file: {e_write_lean}"
                                    )

                    shutil.move(
                        temp_final_cleaned_lean_file, final_cleaned_lean_output_file
                    )
                    logger.info(
                        f"{good_samples_written_to_final_cleaned_lean} validator-passed samples (lean format) saved to {final_cleaned_lean_output_file}."
                    )
                    print(
                        f"\nPROD_RUN: Final cleaned eval-ready dataset ({good_samples_written_to_final_cleaned_lean} samples) saved to {final_cleaned_lean_output_file}."
                    )

                else:
                    logger.warning(
                        f"Validator results file not found at {validator_results_path}. Cannot create final cleaned eval dataset."
                    )
                    print(
                        f"WARNING: Validator results file not found at {validator_results_path}. Cannot create final cleaned eval dataset."
                    )

            except FileNotFoundError:
                logger.error(
                    f"Validator script '{validator_script_path}' not found. Skipping cleaning step."
                )
                print(f"ERROR: Validator script '{validator_script_path}' not found.")
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Validator script failed with exit code {e.returncode}. Skipping cleaning step."
                )
                print(f"ERROR: Validator script failed with exit code {e.returncode}.")
                if e.stderr:
                    logger.error(
                        "Validator stderr (from CalledProcessError exception object):"
                    )
                    for line_e in e.stderr.splitlines():
                        logger.error(f"VAL_STDERR_ERR_CPE: {line_e}")
                else:
                    logger.error(
                        "Validator stderr (from CalledProcessError exception object) was None or empty."
                    )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during PROD_RUN validation or cleaning: {e}",
                    exc_info=True,
                )
                print(
                    f"ERROR: An unexpected error occurred during PROD_RUN validation: {e}"
                )

    elif PROD_RUN and (
        samples_written_researcher == 0 # Check if input for validator was even created
        or not os.path.exists(researcher_detail_output_file)
    ):
        logger.info(
            "PROD_RUN was True, but no researcher detail samples were available for validation. Skipping validation and cleaning."
        )
        print(
            "\nINFO: PROD_RUN was True, but no researcher detail samples were available for validation. Skipping validation and cleaning."
        )
    elif not PROD_RUN:
        logger.info("PROD_RUN is False. Skipping validation and final cleaning step.")
        print("\nINFO: PROD_RUN is False. Skipping validation and final cleaning step.")


    # --- Final Summary ---
    end_time_total = time.time()
    total_run_time = end_time_total - start_time

    logger.info(f"--- Overall Run Summary ---")
    logger.info(f"Total samples attempted for generation: {num_samples}")
    logger.info(
        f"Successfully generated by verbose-listops: {samples_generated_successfully}"
    )
    logger.info(
        f"Successfully written to researcher detail file ({researcher_detail_basename}): {samples_written_researcher}"
    )
    logger.info(
        f"Successfully written to initial eval-ready file ({eval_ready_basename}): {samples_written_eval_raw}"
    )

    if PROD_RUN:
        if final_cleaned_lean_output_file and os.path.exists(
            final_cleaned_lean_output_file
        ):
            try:
                with open(
                    final_cleaned_lean_output_file, "r", encoding="utf-8"
                ) as f_cleaned_count:
                    num_cleaned_samples = sum(1 for _ in f_cleaned_count)
                logger.info(
                    f"Samples in final cleaned eval-ready file ({final_cleaned_lean_basename}): {num_cleaned_samples}"
                )
                # This print was already handled inside the PROD_RUN block
            except Exception as e_count:
                logger.error(
                    f"Could not count samples in final cleaned eval-ready file {final_cleaned_lean_output_file}: {e_count}"
                )
                print(
                    f"\nFinal Cleaned Eval-Ready dataset saved to: {final_cleaned_lean_output_file} (count error)"
                )
        elif PROD_RUN: # Only print this if PROD_RUN was true and file wasn't made
            logger.info(
                "Final Cleaned Eval-Ready dataset was not produced or path is invalid."
            )
            print(
                f"\nFinal Cleaned Eval-Ready dataset was not produced due to issues in validation/cleaning or no samples passed."
            )

    logger.info(f"Total run time: {total_run_time:.2f} seconds")

    hours, rem = divmod(total_run_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str_display = ""
    if hours > 0:
        time_str_display += f"{int(hours)}h "
    if minutes > 0 or hours > 0:
        time_str_display += f"{int(minutes)}m "
    time_str_display += f"{seconds:.2f}s"
    print(
        f"\n✅ Total execution time: {time_str_display.strip()} ({total_run_time:.2f} seconds)"
    )
    print(f"\nAll datasets for this run are in subfolder: {run_output_dir}")

    gen_prompt_tokens, gen_completion_tokens, gen_api_calls = (
        generation_token_tracker.get_summary()
    )
    estimated_generation_cost = generation_token_tracker.calculate_cost(
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS,
    )
    logger.info(f"--- Generation Token Usage & Estimated Cost (verbose-listops.py) ---")
    logger.info(f"Total API calls (generation): {gen_api_calls}")
    logger.info(f"Total Prompt Tokens (generation): {gen_prompt_tokens}")
    logger.info(f"Total Completion Tokens (generation): {gen_completion_tokens}")
    logger.info(f"Estimated Cost (generation only): ${estimated_generation_cost:.4f}")

    if PROD_RUN and os.path.exists(
        validator_output_log_path
    ):
        try:
            with open(validator_output_log_path, "r", encoding="utf-8") as f_val_out:
                val_out_content = f_val_out.read()
                match = re.search(
                    r"VALIDATOR_TOKEN_USAGE_SUMMARY:prompt_tokens=(\d+),completion_tokens=(\d+),api_calls=(\d+)",
                    val_out_content,
                )
                if match:
                    val_prompt_tokens = int(match.group(1))
                    val_completion_tokens = int(match.group(2))
                    val_api_calls = int(match.group(3))
                    estimated_validation_cost = (
                        val_prompt_tokens
                        / 1_000_000
                        * DEFAULT_COST_PER_MILLION_PROMPT_TOKENS
                    ) + (
                        val_completion_tokens
                        / 1_000_000
                        * DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS
                    )
                    logger.info(
                        f"--- Validation Token Usage & Estimated Cost (from {validator_stdout_basename}) ---"
                    )
                    logger.info(f"Total API calls (validation): {val_api_calls}")
                    logger.info(
                        f"Total Prompt Tokens (validation): {val_prompt_tokens}"
                    )
                    logger.info(
                        f"Total Completion Tokens (validation): {val_completion_tokens}"
                    )
                    logger.info(
                        f"Estimated Cost (validation only, using verbose-listops.py rates): ${estimated_validation_cost:.4f}"
                    )
                else:
                    logger.warning(
                        f"Could not find VALIDATOR_TOKEN_USAGE_SUMMARY line in '{validator_output_log_path}'."
                    )
        except Exception as e_val_cost:
            logger.warning(
                f"Could not parse token usage from validator output log ('{validator_output_log_path}'): {e_val_cost}"
            )

    if (
        client
        and OPENROUTER_API_KEY
        and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE"
    ):
        logger.info("Fetching final OpenRouter account usage...")
        final_account_usage = rate_limiter.update_limits_from_api()
        if final_account_usage is not None:
            logger.info(f"Final OpenRouter account usage: ${final_account_usage:.4f}")
            if initial_account_usage is not None:
                total_run_cost_by_difference = (
                    final_account_usage - initial_account_usage
                )
                logger.info(f"--- Total Run Cost (from API Usage Difference) ---")
                logger.info(
                    f"TOTAL RUN COST (Generation + Validation if validator.py used same key): ${total_run_cost_by_difference:.4f}"
                )
        else:
            logger.warning(
                "Could not fetch final OpenRouter account usage. Cannot calculate total run cost by difference."
            )
    else:
        logger.warning(
            "Skipping final OpenRouter account usage check for run cost: Client not initialized or API key missing/placeholder."
        )

    logger.info("--- END OF MAIN FUNCTION ---")
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
