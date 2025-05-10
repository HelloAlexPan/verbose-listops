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
from dataclasses import dataclass, field
import re
import concurrent.futures
import inflect
import tiktoken
from functools import lru_cache
from openai import OpenAI
import shutil  # Add shutil for rmtree
import threading  # Added for RateLimiter
import requests  # Added for RateLimiter
import subprocess  # Added for PROD_RUN
import sys  # Added for PROD_RUN

from dotenv import load_dotenv

load_dotenv()

# fmt: off
# --- Batch Settings ---
NUM_SAMPLES_TO_GENERATE = 5  # How many samples to generate in one run
DEFAULT_MAX_WORKERS = 50   # Number of parallel threads for batch generation
MODEL = "google/gemini-2.5-pro-preview"  # OpenRouter model for main generation
STATIC_CHECKER_MODEL = "qwen/qwq-32b" # Different model for static beat validation
DATASETS_DIR = "datasets"    # Directory for saving generated datasets
PROD_RUN: bool = True       # If True, run validator and clean dataset post-generation

# --- Generation Settings ---
"""
Modify the difficulty of the generated problems by changing the parameters below.
"""

@dataclass
class Config:
    # === Core Experiment Variables ===
    # --- 1. ListOps Problem Difficulty  ---
    MAX_OPS: int = 8                                # Max ListOps operations
    MAX_BRANCH: int = 6                             # Max numbers/sub-ops per operation
    MIN_ARITY: int = 3                              # Min numbers/sub-ops per operation
    MIN_ATOM_VAL: int = 1                           # Min value for atomic numbers
    MAX_ATOM_VAL: int = 100                         # Max value for atomic numbers
    MAX_TOTAL_TOKENS: int = 10000                   # Cleaned sample token budget
    EARLY_TERMINATION_PROBABILITY: float = 0.0      # Chance to end AST branch early
    PADDING_MAX_TOK_PERCENT: float = 0.60           # How much total tok budget can be padding
    ALLOW_IMPLICIT_INTERMEDIATE_RESULTS: bool = True # If True, intermediate results don't need to be explicitly stated

    # --- 2. Narrative Context Generation & Style ---
    USE_NARRATIVE_ANCHORS: bool = True              # Conceptual placeholders for intermediate results
    USE_LLM_NAMING: bool = True                     # Use LLM for creative anchor names
    MIN_WORLD_CHARS: int = 3                        # Min chars for randomized world gen
    MAX_WORLD_CHARS: int = 6                        # Max chars for randomized world gen
    MIN_WORLD_CONCEPTS: int = 3                     # Min concepts for randomized world gen
    MAX_WORLD_CONCEPTS: int = 5                    # Max concepts for randomized world gen
    BEAT_CONTEXT: int = 1000                        # Max previous scene chars for beat gen prompt
    PADDING_CONTEXT: int = 1500                     # Tokens of context for padding
    MAX_PAD_PARAGRAPHS: int = 20                    # Max padding segments per-beat

    # --- 3. Temperature ---
    WORLD_GEN_TEMP:  float = 0.9                    # Temp. for world gen
    BEAT_GEN_TEMP: float = 0.3                      # Temp. for generating narrative beats (Corrected to 0.3)
    CREATIVE_NARRATIVE_TEMP: float = 0.5            # Temp. for creative parts (intro, padding)
    ANCHOR_GEN_TEMP: float = 0.75                   # Temp. for narrative anchor generation

    # === Static Beat Validation (New) ===
    USE_LLM_STATIC_BEAT_VALIDATION: bool = True     # Whether to use an LLM to validate each beat's narrative logic
    STATIC_BEAT_VALIDATION_MAX_RETRIES: int = 5     # Max retries for a beat if LLM validation fails
    STATIC_BEAT_VALIDATION_TEMP: float = 0.05       # Temperature for the beat validator LLM

    # === Iterative LLM Beat Validation & Revision (New from User Plan) ===
    LLM_VALIDATOR_MODEL: str = "qwen/qwq-32b"       # Model for the iterative LLM beat validator
    LLM_VALIDATOR_TEMP: float = 0.05                # Temperature for the iterative LLM beat validator
    BEAT_REVISION_TEMP: float = 0.1                 # Temperature for generator LLM during beat revisions
    MAX_LLM_VALIDATION_ITERATIONS: int = 3          # Max attempts for the inner LLM validation loop
    MODEL_MAX_CONTEXT_TOKENS: int = 750000          # Max context for generator model (example, adjust for actual model, e.g. Gemini 2.5 Pro ~1M-2M, leave buffer)

    # === Other: Probably don't touch the below unless you know what you're doing ===
    # LLM Interaction & Prompting
    MAX_ANCHOR_WORDS: int = 4                       # Max words allowed in a narrative anchor name
    FEW_SHOT_EXAMPLES: int = 1                      # Few-shot examples for beat generation

    # Data Validation & Fallbacks
    FALLBACK_MIN_NUM_WORD: int = 0                  # Fallback range for num_to_words
    FALLBACK_MAX_NUM_WORD: int = 20                 # Fallback range for num_to_words
    MIN_ALLOWED_SMALL_NUMBER: int = 0               # Validator setting: min implicitly allowed small number
    MAX_ALLOWED_SMALL_NUMBER: int = 10              # Validator: max implicitly allowed small numbers
    INVALID_RESULT_PLACEHOLDER: int = -999          # Validator: placeholder for specific error cases
    PROBLEM_SMALL_NUMBERS_TO_CHECK: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) # Numbers that get special attention if forbidden

    #  API Configuration & Retries
    RETRY_MAX_ATTEMPTS: int = 5                     # Max retries for API calls
    RETRY_INITIAL_DELAY: float = 0.25                # Initial delay for exponential backoff
    MAX_BEAT_RETRIES: int = 5                       # Max retries for beat generation
    MAX_PAD_RETRIES: int = 5                        # Max retries for padding generation
    INTRO_MAX_RETRIES: int = 3                      # Max retries for intro scene generation
    WORLDGEN_MAX_RETRIES: int = 3                   # Max retries for world generation
    INITIAL_WORLD_RETRY_DELAY: float = 0.5          # Initial retry delay for world gen
    MAX_REQUESTS_PER_SECOND: float = 900.0          # Max requests/s to OpenRouter
    MIN_REQUEST_INTERVAL: float = 0.001             # Min time (seconds) between requests

    # Logging Configuration
    LOG_MAX_BYTES: int = 5 * 1024 * 1024            # Maximum log file size (5MB)
    LOG_BACKUP_COUNT: int = 3                       # Number of backup log files to keep
    CLEAR_LOGS_ON_START: bool = True                # If True delete existing logs on startup

    # Token & Budget Management !!! DONT FUCKING TOUCH THIS !!!
    MAX_TOKENS_BUFFER: int = 500                    # Safety buffer for overall budget
    
    # API Token limits - Set high values to avoid truncation due to reasoning tokens
    MAX_API_TOKEN_LIMIT: int = 32000                # High tok limit for reasoning tokens
    
    # Internal token budget tracking (these just track, not limit API calls)
    WORLD_GEN_MAX_TOKENS: int = 200                # World gen .json max tok (for tracking) 
    ANCHOR_MAX_TOKENS: int = 100                   # Anchor gen max tok (for tracking)
    INTRO_MAX_TOKENS: int = 100                    # Intro scene max tok (for tracking)
    BEAT_MAX_TOKENS: int = 400                     # Beat max tok (for tracking)
    PADDING_MAX_TOKENS: int = 200                  # Padding max tok (for tracking)

# fmt: on

config = Config()


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


# Instantiate the tracker globally
generation_token_tracker = GenerationTokenTracker()

# Define some placeholder costs (you should update these with actual model costs)
DEFAULT_COST_PER_MILLION_PROMPT_TOKENS = 0.50  # Example: $0.50 / 1M prompt tokens
DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS = (
    1.50  # Example: $1.50 / 1M completion tokens
)


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

# --- Prompt Templates ---
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
        # --- Example 1: Basic Success vs. Extraneous Number (>10) ---
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
]

# --- Meta-Instruction + Task-Solving Few-Shot Examples ---
META_INSTRUCTION = (
    "Here are examples demonstrating how to solve narrative math problems. For each problem: "
    "Read the entire story. Identify the quantity being tracked (e.g., coins, artifacts, energy units). "
    "Follow the narrative step-by-step, performing the calculation implied by the actions in each scene "
    "(e.g., finding items, combining, selecting the largest/smallest, averaging, reducing, resetting). "
    "Keep track of the current quantity as it changes. Finally, answer the question by providing only "
    "the single integer representing the final quantity based on the last relevant action described."
)

TASK_SOLVING_FEW_SHOTS = [
    """
*   **Calculation Trace:**
    *   Scene 1: Find [24, 92, 96]. Cue: 'sum of all'. Calc: SUM -> 212. Current: 212.
    *   Scene 2: Find [1, 27, 54, 88]. Cue: 'combining... average, rounded down'. Calc: FLOOR(AVG([212, 1, 27, 54, 88])) -> 76. Current: 76.
    *   Scene 3: Find [68, 76]. Cue: 'adding... integer average'. Calc: FLOOR(AVG([76, 68, 76])) -> 73. Current: 73.
    *   Scene 4: Find [15, 20]. Cue: 'combined... their sum'. Calc: SUM [73, 15, 20] -> 108. Current: 108.
    *   Scene 5: Choice [1, 69]. Cue: 'greatest bounty'. Calc: MAX [1, 69] -> 69. *Replaces* previous. Final: 69.

*   **Question:** Following the entire sequence of events described in the story, exactly how many Soulstones did the characters end up with? Provide only the final integer count.
*   **Answer:** 69
    """,
    """
*   **Narrative:**
    Prospectors started with 40 Glow-Shards. Found veins: 28, 9, 35 Shards. Map: 'Echoing Lock requires *smallest* yield.' Took only that amount. Later, found pouch with 15 Shards, added to collection.

*   **Calculation Trace:**
    *   Step 1: Start: 40.
    *   Step 2: Find [28, 9, 35]. Cue: MIN -> 9. *Take only 9*, replaces 40. Current: 9. (Irrelevant: '5 hours')
    *   Step 3: Find 15. Cue: 'added'. Calc: SUM [9, 15] -> 24. Current: 24.

*   **Question:** Following the entire sequence of events described in the story, exactly how many Glow-Shards did the prospectors end up with? Provide only the final integer count.
*   **Answer:** 24
    """,
    """
*   **Narrative:**
    Prof. A started artifact hunt. **Found 18 Fragments.** **Found 32 Fragments**, added them. Tremor, **satchel spilled contents into chasm.** Pressed on. **Found pedestal 'Contains exactly 7 Fragments'.** Took these.

*   **Calculation Trace (Linked to Highlights):**
    *   Find 18 -> Current: 18
    *   Find 32 -> Add: 18 + 32 = 50. Current: 50.
    *   Spilled contents -> Reset: Current: 0.
    *   Find 7 -> Take: Current: 7. (Irrelevant: 'another hour')

*   **Question:** Following the entire sequence of events described in the story, exactly how many Relic Fragments did Professor Armitage end up with? Provide only the final integer count.
*   **Answer:** 7
    """,
    """
*   **Narrative:**
    Alchemist Zosimos checked 3 conduits: 15 units, 8 units, 0 units. Calibration Matrix required sum, then integer average (floored) of the 3 readings for baseline resonance.

*   **Calculation Trace:**
    *   Step 1: Readings: [15, 8, 0].
    *   Step 2: Cue: 'sum'. SUM [15, 8, 0] = 23. (Intermediate sum allowed).
    *   Step 3: Cue: 'integer average (floored)'. Calc: FLOOR(23 / 3) -> 7.
    *   Step 4: Result 7 is baseline. Final: 7.

*   **Question:** Following the entire sequence of events described in the story, what was the final baseline resonance (in units) set for the Athanor? Provide only the final integer count.
*   **Answer:** 7
    """,
]

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# --- Setup Logging --- (This section needs to come AFTER LOG_DIR is defined and potentially cleared)

if config.CLEAR_LOGS_ON_START:
    if os.path.exists(LOG_DIR):
        try:
            shutil.rmtree(LOG_DIR)  # Remove the entire logs directory
            print(f"Removed existing log directory: {LOG_DIR}")
        except OSError as e:
            # Use logger if available, otherwise print
            if "logger" in globals() and logger:
                logger.error(f"Error removing log directory {LOG_DIR}: {e}")
            else:
                print(f"Error removing log directory {LOG_DIR}: {e}")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)  # Recreate the logs directory
        # print(f"Ensured log directory exists: {LOG_DIR}") # Optional: for very early debugging before logger is set up
    except OSError as e:
        # Use logger if available, otherwise print
        if "logger" in globals() and logger:  # Check if logger is initialized
            logger.error(f"Error creating log directory {LOG_DIR}: {e}")
        else:
            print(f"Error creating log directory {LOG_DIR}: {e}")

# Initialize logger AFTER log directory is confirmed to exist and is writable
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

# Now that logger is configured, we can safely log messages if clearing happened earlier.
if config.CLEAR_LOGS_ON_START and os.path.exists(LOG_DIR):
    logger.info(f"Log directory {LOG_DIR} cleared and recreated successfully.")

# --- Few-shot prompt examples ---
EXAMPLE_TEXTS = [
    (
        "Example 1:\\n"
        'Narrative: "Guild offered two contracts: one 9 silver, other 4. Kaelen chose lower (4). Received 5 silver bonus."\\n'
        "Implicit Calculation: MIN(9, 4) = 4. SUM(4, 5) = 9.\\n"
        "Answer: 9\\n"
    ),
    (
        "Example 2:\\n"
        "Narrative: Vault needed combined energy of four crystals (1, 1, 1, 1). Lock used final digit of total power.\\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Mod 10 -> 4.\\n"
        "Answer: 4\\n"
    ),
    (
        "Example 3:\\n"
        "Narrative: Three scouts reported patrol durations: 5, 5, 5 hours. Procedure: calculate average time (rounded down) for logbook.\\n"
        "Implicit Calculation: SUM(5, 5, 5) = 15. Count=3. AVG = 15/3 = 5. Floor(5) = 5.\\n"
        "Answer: 5"
    ),
]


# --- Prompt Logging Helper ---
def log_prompt(
    header: str,
    prompt: str,
    sample_index: int | None = None,
):
    """Append a timestamped prompt header and text to a sample-specific prompts log."""
    try:
        # Define the target directory for these specific LLM turn logs
        llm_turns_main_dir = os.path.join(LOG_DIR, "llm_turns")
        llm_turns_log_specific_dir = os.path.join(llm_turns_main_dir, "log")

        if sample_index is not None:
            log_filename = f"llm_turns_sample_{sample_index + 1}.log"
        else:
            # Fallback for any logs not associated with a specific sample
            log_filename = "llm_turns_general.log"

        # Ensure the specific log directory exists
        os.makedirs(llm_turns_log_specific_dir, exist_ok=True)

        # Construct the full path to the log file
        current_log_file_path = os.path.join(llm_turns_log_specific_dir, log_filename)

        timestamp = datetime.datetime.now().isoformat()

        log_header_text = (
            f"[Sample {sample_index + 1}] {header}"
            if sample_index is not None
            else header
        )
        with open(current_log_file_path, "a", encoding="utf-8") as f:
            f.write(f"--- Log Time: {timestamp} ---\\n")
            f.write(f"{log_header_text}\\n{prompt}\\n\\n---\\n\\n")
    except Exception as e:
        # Use logger for errors within log_prompt itself if possible,
        # otherwise print.
        if logger:
            logger.error(f"Error writing to LLM turn log file: {e}")
        else:
            print(f"Error writing to LLM turn log file: {e}")


FINAL_QUESTION_TEMPLATE = Template(
    "\n\n---\n\n**Question:** Considering the entire sequence of events described in the story, what is the final, precise quantity of $primary_object that the characters possess or have determined at the very end of their activities? Provide only the single integer representing this final amount."
)

# --- GenerationContext for recursive narrative state ---
from dataclasses import dataclass
import logging


@dataclass
class GenerationContext:
    """
    Container that groups together the mutable state shared across the
    recursive narrative‑generation calls.  Replaces the long positional
    parameter list that _generate_narrative_recursive previously required.
    """

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
    max_pad_paragraphs: (
        int  # No default value, must be set explicitly during instantiation
    )
    overall_ground_truth_answer: int | None = (
        None  # ADDED: Store the AST's final answer
    )
    # Add tracking for padding token statistics
    padding_stats: dict = field(
        default_factory=lambda: {
            "total_padding_tokens": 0,
            "padding_segments_added": 0,
            "max_padding_allowed": 0,  # Will be calculated during initialization
        }
    )


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
            logger.info("Checking OpenRouter rate limits and remaining credits...")
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

                log_message_parts = [
                    f"OR Limits: Current RPS: {current_rate_for_log:.1f}"
                ]
                if limit_adjusted:
                    log_message_parts.append("(Adjusted)")

                if usage is not None:
                    log_message_parts.append(f"Usage: ${usage:.4f}")
                    current_usage = float(usage)  # Store the usage

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


# --- AST Node Definitions ---
@dataclass
class Node:
    op: str
    children: list = field(default_factory=list)
    value: int = None


@dataclass
class Atom(Node):
    n: int = None

    def __init__(self, n: int):
        super().__init__(op="ATOM", children=[])
        self.n = n
        self.value = n


@dataclass
class OpNode(Node):
    def __init__(self, op: str, children: list):
        super().__init__(op=op, children=children)
        self.value = None


# --- AST Generation and Evaluation ---
def build_random_ast(max_ops: int, max_branch: int = config.MAX_BRANCH) -> Node:
    """Constructs a random ListOps AST."""
    if not isinstance(max_ops, int) or max_ops < 1:
        raise ValueError("max_ops must be a positive int")
    if max_branch < config.MIN_ARITY:
        raise ValueError(f"max_branch ({max_branch}) < MIN_ARITY ({config.MIN_ARITY})")
    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0

    def helper():
        nonlocal count
        if count >= max_ops or (
            count > 0 and random.random() < config.EARLY_TERMINATION_PROBABILITY
        ):
            return Atom(random.randint(config.MIN_ATOM_VAL, config.MAX_ATOM_VAL))
        count += 1
        op = random.choice(ops)

        if op == "MED":
            possible_arities = [
                n for n in range(config.MIN_ARITY, max_branch + 1) if n % 2 == 1
            ]

            if not possible_arities:
                arity = (
                    config.MIN_ARITY
                    if config.MIN_ARITY % 2 == 1
                    else config.MIN_ARITY + 1
                )
            else:
                arity = random.choice(possible_arities)
        else:
            arity = random.randint(config.MIN_ARITY, max_branch)
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
                    if config.MIN_ATOM_VAL <= new_value_add <= config.MAX_ATOM_VAL:
                        atom_to_adjust.n = new_value_add
                        atom_to_adjust.value = new_value_add
                        logger.debug(
                            f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value up to {atom_to_adjust.n} to make sum divisible by {arity}."
                        )
                        adjusted = True

                    if not adjusted:
                        new_value_sub = atom_to_adjust.n - (arity - adjustment_needed)
                        if config.MIN_ATOM_VAL <= new_value_sub <= config.MAX_ATOM_VAL:
                            atom_to_adjust.n = new_value_sub
                            atom_to_adjust.value = new_value_sub
                            logger.debug(
                                f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value down to {atom_to_adjust.n} to make sum divisible by {arity}."
                            )
                            adjusted = True

                    if not adjusted:

                        logger.warning(
                            f"AST Gen (AVG): Could not adjust atom value {atom_to_adjust.n} (target adjustment {adjustment_needed}) for AVG node sum {current_sum} to be divisible by {arity} due to bounds [{config.MIN_ATOM_VAL}, {config.MAX_ATOM_VAL}]."
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
        new_arity = random.randint(max(2, config.MIN_ARITY), max_branch)

        new_children = [root]  # The old root is one child

        # Add new Atom children
        for _ in range(new_arity - 1):
            new_children.append(
                Atom(random.randint(config.MIN_ATOM_VAL, config.MAX_ATOM_VAL))
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
                            config.MIN_ATOM_VAL
                            <= atom_to_adjust_new_root.n + adjustment_needed_new_root
                            <= config.MAX_ATOM_VAL
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
                            config.MIN_ATOM_VAL
                            <= atom_to_adjust_new_root.n
                            - (num_direct_atoms_new_root - adjustment_needed_new_root)
                            <= config.MAX_ATOM_VAL
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
        arity = random.randint(config.MIN_ARITY, max_branch)
        children = [
            Atom(random.randint(config.MIN_ATOM_VAL, config.MAX_ATOM_VAL))
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
    for c in node.children:
        yield from postorder(c)
    yield node


@retry_api_call
def _chat_completion_call(*args, **kwargs):
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

    # The global reasoning configuration check is removed - we now expect reasoning params to be explicitly passed

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
    """Strip Markdown code fences and parse JSON."""

    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e} in text:\n---\n{text}\n---")
        raise  # Re-raise after logging


# --- Tuned Generate World Function ---
def generate_world(
    num_characters: int = 5,
    num_concepts: int = 7,
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
        "**Instructions:**\n\n"
        "1.  **Characters:** Generate exactly {num_characters} distinct characters. Each character MUST have:\n"
        '    *   `name`: string (e.g., "Kaelen Vane", "Seraphina Moonwhisper")\n'
        '    *   `role`: string (e.g., "The grizzled warrior," "The cunning sorceress," "The naive apprentice")\n'
        '    *   `quirk`: string (a unique or unusual habit, belief, or physical trait, e.g., "Collects antique spoons," "Only speaks in riddles," "Has mismatched eyes")\n'
        "    Ensure each character's name, role, and quirk combination is unique.\n\n"
        '2.  **Genre:** Define a `genre` as a string (e.g., "Steampunk Adventure," "Urban Fantasy Mystery," "Cosmic Horror Saga").\n\n'
        '3.  **Setting:** Define a `setting` as a string (a brief, evocative description of the world or primary location, e.g., "A floating city powered by forgotten magic and steam contraptions," "A post-apocalyptic wasteland where ancient ruins hold dangerous secrets").\n\n'
        '4.  **Object:** Define an `object` as a string. This should be a plural noun representing key items characters might seek, collect, or use (e.g., "etherium crystals," "lost star-charts," "prophetic dream-shards").\n\n'
        "**Guidance for Content:**\n"
        "*   Strive for thematic coherence between the genre, setting, characters, and the collectible object. They should feel like they belong in the same world.\n\n"
        "**Output Format:**\n"
        "Output *ONLY* a single, raw, **strictly valid JSON object** adhering precisely to the following structure. Do NOT include ```json markdown fences or *any* other text before or after the JSON object.\n\n"
        "{{\n"
        f'  "characters": [{{"name": "string", "role": "string", "quirk": "string"}}, ...], // Exactly {num_characters} character objects\n'
        '  "genre": "string",\n'
        '  "setting": "string",\n'
        '  "object": "string"\n'
        "}}\n\n"  # Note: Double curly braces {{ and }} are used to escape them in an f-string for the JSON structure
        "**!!! CRITICAL JSON RULE: Escaping Double Quotes !!!**\n"
        "If any string value itself needs to contain double quotes (e.g., a nickname within a name, a quote in a setting description), these internal double quotes **MUST** be escaped with a backslash (`\\\\`).\n"  # Python `\\\\` -> prompt `\\`
        '   - **CORRECT Example:** `\\"name\\": \\"Bartholomew \\\\\\"Barty\\\\\\" Bumble\\"`\n'  # Python `\\\\\\\"` -> prompt `\\\"`
        '   - **INCORRECT Example:** `\\"name\\": \\"Bartholomew \\"Barty\\" Bumble\\"` (This will cause a parsing error!)\n'
        "Adhere strictly to all JSON syntax rules, including commas between elements and correct brace/bracket usage.\n\n"
        "**Final Reminder:**\n"
        "Your response must be *only* the JSON object. Do NOT include ```json markdown fences, explanations, introductory phrases, or *any* other text before or after the JSON object.\n\n"
        "Generate the JSON data now."
    )

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
                temperature=0.5,
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

            world = clean_and_parse_json_block(
                text
            )  # Uses your existing cleaning function

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
                f"World Gen Attempt {attempt + 1}: Failed ({type(e).__name__}): {e}. Raw text:\n---\n{text}\n---"
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

MAX_WORDS_FOR_NUMBER_DICT = 5000  # Define a larger limit for number-to-word conversion


def _build_expanded_number_words_dict(
    max_val: int = MAX_WORDS_FOR_NUMBER_DICT,
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
    allowed_atoms: Set[int],
    forbidden_atoms: Set[int],
    operand_count: int,
    correct_result_for_beat: int | None = None,
    intermediate_sum_allowed: int | None = None,
    strict_zero: bool = False,
    enforce_result_presence: bool = True,
    operation_type: str | None = None,
) -> Callable[[str], bool]:
    logger.debug(
        f"Creating validator with: Allowed_Atoms={allowed_atoms}, Forbidden={forbidden_atoms}, OpCount={operand_count}, Result={correct_result_for_beat}, InterSum={intermediate_sum_allowed}, StrictZero={strict_zero}, EnforceResult={enforce_result_presence}, Op={operation_type}"
    )

    IMPLICITLY_ALLOWED_SMALL_NUMBERS = set(
        range(config.MIN_ALLOWED_SMALL_NUMBER, config.MAX_ALLOWED_SMALL_NUMBER + 1)
    )

    # This set includes direct operands, the current beat's result, and its intermediate sum.
    # These are numbers that *should* or *can* be in the text for this specific beat.
    current_beat_explicitly_allowed_numbers = allowed_atoms.copy()
    if correct_result_for_beat is not None:
        current_beat_explicitly_allowed_numbers.add(correct_result_for_beat)
    if intermediate_sum_allowed is not None:
        current_beat_explicitly_allowed_numbers.add(intermediate_sum_allowed)

    def validate(text: str) -> bool:
        found_numbers = extract_numbers_from_text(text)

        # Get text for logging (truncated to avoid huge logs)
        text_preview = text[:100].replace("\n", " ")
        if len(text) > 100:
            text_preview += "..."

        logger.debug(f'Validator Input Text: "' + text_preview + '"')
        logger.debug(f"Validator Found Numbers: {found_numbers}")

        # Create a detailed validation report
        validation_report = {
            "status": "PASS",
            "reason": "All validation checks passed",
            "operation_type": operation_type,
            "text_preview": text_preview,
            "found_numbers": list(found_numbers),
            "allowed_atoms": list(allowed_atoms),
            "operand_count": operand_count,
            "correct_result": correct_result_for_beat,
            "intermediate_sum": intermediate_sum_allowed,
            "missing_required": [],
            "forbidden_extras": [],
            "details": [],
        }

        if strict_zero:
            # Check if numbers were found
            if found_numbers:
                # If numbers are found, check if it's ONLY the number 1
                if found_numbers == {1}:
                    logger.debug(
                        f"Validation PASS (Strict Zero context, but only '1' found, which is tolerated for padding/intro). Found: {found_numbers}"
                    )
                    # Even though it's a "pass" for this specific rule,
                    # we don't return True yet, as other rules might apply if strict_zero was False.
                    # However, for padding/intro, strict_zero IS True, and this is the only number check we care about.
                    # So, if we reach here, it means for padding/intro, finding only "1" is acceptable.
                    # The rest of the validator logic for non-strict_zero (operand checks etc.) is not relevant here.
                    # BUT, we need to ensure this doesn't bypass other checks if strict_zero was meant to be part of a larger validation.
                    # Given the current structure, if strict_zero is true, we ONLY care if non-{1} numbers are present.
                else:
                    # Numbers other than just {1} were found, or a mix including 1. This is a failure.
                    validation_report["status"] = "FAIL"
                    validation_report["reason"] = "STRICT_ZERO_VIOLATION"
                    validation_report["details"].append(
                        f"Found numbers {found_numbers} when only '1' (or zero numbers) were allowed in this strict context."
                    )
                    log_reason = f"Validation FAIL (Strict Zero context): Found numbers {found_numbers}. Expected zero numbers or only '1'."
                    logger.debug(log_reason)
                    _log_failed_validation(text, validation_report)
                    return False  # Fail because numbers other than just '1' were found
            # If found_numbers is empty, it's a pass for strict_zero
            logger.debug(f"Validation PASS (Strict Zero context: No numbers found).")
            # If we are in strict_zero mode and passed (either empty or just {1}),
            # for padding/intro, this is the end of validation.
            # We need to ensure this doesn't incorrectly proceed to other checks.
            # The original logic for strict_zero was to return True if no numbers.
            # Now, it should return True if no numbers OR only {1}.

            # Corrected logic for strict_zero:
            if not found_numbers or found_numbers == {1}:
                logger.debug(
                    f"Validation PASS (Strict Zero context): Found numbers: {found_numbers} (empty or only '1' is acceptable)."
                )
                return True  # This is the definitive pass for strict_zero sections (padding/intro)
            else:
                # This 'else' handles cases where found_numbers is not empty AND not equal to {1}
                validation_report["status"] = "FAIL"
                validation_report["reason"] = "STRICT_ZERO_VIOLATION"
                validation_report["details"].append(
                    f"Found numbers {found_numbers} when only '1' (or zero numbers) were allowed in this strict context."
                )
                log_reason = f"Validation FAIL (Strict Zero context): Found numbers {found_numbers}. Expected zero numbers or only '1'."
                logger.debug(log_reason)
                _log_failed_validation(text, validation_report)
                return False

        # Check if result is required and present
        if enforce_result_presence and correct_result_for_beat is not None:
            # Always check for result presence when enforce_result_presence is True
            if correct_result_for_beat not in found_numbers:
                validation_report["status"] = "FAIL"
                validation_report["reason"] = "MISSING_REQUIRED_RESULT"
                validation_report["details"].append(
                    f"Result {correct_result_for_beat} must be present. Found: {found_numbers}, Op: {operation_type}"
                )

                log_reason = f"Validation FAIL (Missing Required Result): Result {correct_result_for_beat} must be present. Found: {found_numbers}, Op: {operation_type}"
                logger.debug(log_reason)

                # Log the failed attempt with error code
                _log_failed_validation(text, validation_report)
                return False
            logger.debug(
                f"Validation INFO: Required result {correct_result_for_beat} is present for {operation_type if operation_type else 'unspecified'} operation"
            )

        missing_expected = allowed_atoms - found_numbers
        if missing_expected:
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "MISSING_REQUIRED_OPERANDS"
            validation_report["missing_required"] = list(missing_expected)
            validation_report["details"].append(
                f"RequiredOperands={allowed_atoms}, Missing={missing_expected}, FoundInText={found_numbers}."
            )

            log_reason = f"Validation FAIL (Rule 1: Missing Required Operands): RequiredOperands={allowed_atoms}, Missing={missing_expected}, FoundInText={found_numbers}."
            logger.debug(log_reason)

            # Log the failed attempt with error code
            _log_failed_validation(text, validation_report)
            return False

        # Rule 2: Check for numbers that are on the forbidden_atoms list (from prior beats)
        # but are NOT legitimately part of the current beat's explicit allowances.
        forbidden_and_found_in_text = found_numbers & forbidden_atoms

        violations_of_forbidden_rule = set()
        for num_in_question in forbidden_and_found_in_text:
            # If the number is a direct operand, current result, or current intermediate sum, it's NOT a violation here.
            if num_in_question in current_beat_explicitly_allowed_numbers:
                continue
            # Otherwise, it's a number from a past step, found in the text, and not allowed in current beat's explicit set.
            violations_of_forbidden_rule.add(num_in_question)

        if violations_of_forbidden_rule:
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "FORBIDDEN_NUMBERS_FOUND"
            validation_report["forbidden_extras"] = list(violations_of_forbidden_rule)
            validation_report["details"].append(
                f"ForbiddenSetFromPriorBeats={forbidden_atoms}, Violations={violations_of_forbidden_rule}, "
                f"CurrentBeatExplicitlyAllowed={current_beat_explicitly_allowed_numbers}, FoundAllInText={found_numbers}."
            )

            log_reason = (
                f"Validation FAIL (Rule 2: Found Forbidden Violation): "
                f"ForbiddenSetFromPriorBeats={forbidden_atoms}, Violations={violations_of_forbidden_rule}, "
                f"CurrentBeatExplicitlyAllowed={current_beat_explicitly_allowed_numbers}, FoundAllInText={found_numbers}."
            )
            logger.debug(log_reason)

            # Log the failed attempt with error code
            _log_failed_validation(text, validation_report)
            return False
        else:
            logger.debug(
                f"Validation INFO (Rule 2: Forbidden Check): No violations. ForbiddenAndFoundInText={forbidden_and_found_in_text}, CurrentBeatExplicitlyAllowed={current_beat_explicitly_allowed_numbers}"
            )

        # Numbers found in text that are NOT part of the current beat's explicit allowances (operands, result, intermediate sum).
        # These are candidates for being "truly disallowed extras" unless they are operand_count or allowed small_numbers.
        unexpected_found = found_numbers - current_beat_explicitly_allowed_numbers
        logger.debug(
            f"Validator: initial unexpected_found (found_numbers - current_beat_explicitly_allowed_numbers)={unexpected_found}"
        )
        truly_disallowed_extras = set()
        for extra_num in unexpected_found:
            # --- START MODIFICATION ---
            # If the extra number is 1, 2, or 3, consider it allowed for general phrasing
            # and skip further checks for it as a "truly disallowed extra".
            # This means even if 1, 2, or 3 were on the forbidden_atoms list (from a previous beat's result),
            # their use for general phrasing in the current beat is now tolerated.
            if extra_num in [1, 2, 3]:
                logger.debug(
                    f"Validator: Allowing {extra_num} as it's 1, 2, or 3, considered general phrasing. Skipping further checks for this number as an 'extra'."
                )
                continue
            # --- END MODIFICATION ---

            is_allowed_count = (
                extra_num == operand_count and extra_num not in forbidden_atoms
            )
            is_allowed_small = (
                extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS
                and extra_num not in forbidden_atoms
            )

            if not (is_allowed_count or is_allowed_small):  # Adjusted condition
                fail_reason_detail = []

                # Primary check - if it's in forbidden_atoms, that's a key reason
                if extra_num in forbidden_atoms:
                    fail_reason_detail.append(
                        "on forbidden_atoms list (from prior beat) and not otherwise allowed for current beat"
                    )

                # Add specific reason based on what check failed
                if extra_num == operand_count and extra_num in forbidden_atoms:
                    fail_reason_detail.append(
                        f"is operand_count ({operand_count}) but also in forbidden_atoms"
                    )
                elif (
                    extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS
                    and extra_num in forbidden_atoms
                ):
                    fail_reason_detail.append(
                        f"is small number ({extra_num}) but also in forbidden_atoms"
                    )
                # The case for extra_num == 1 is removed as is_allowed_one would always be true when extra_num == 1

                # If it's not any of the allowed categories (and not covered by specific reasons above)
                if not (
                    extra_num == 1
                    or extra_num == operand_count
                    or extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS
                ):
                    fail_reason_detail.append(
                        "not 1, not operand_count, and not an allowed small number"
                    )

                # Ensure we always have at least one reason
                if not fail_reason_detail:
                    fail_reason_detail.append(
                        "unexpected extraneous number not fitting any explicit or implicit allowance"
                    )

                truly_disallowed_extras.add(
                    (
                        extra_num,
                        (
                            ", ".join(fail_reason_detail)
                            if fail_reason_detail
                            else "unexpected extraneous"
                        ),
                    )
                )

        if truly_disallowed_extras:
            formatted_disallowed = ", ".join(
                [f"{n}({reason})" for n, reason in truly_disallowed_extras]
            )

            validation_report["status"] = "FAIL"
            validation_report["reason"] = "EXTRANEOUS_NUMBERS"
            validation_report["details"].append(
                f"Disallowed_Extras={{ {formatted_disallowed} }}. "
                f"CurrentBeatExplicitlyAllowed={current_beat_explicitly_allowed_numbers}, ForbiddenSet={forbidden_atoms}, "
                f"OperandCount={operand_count}, FoundInText={found_numbers}, UnexpectedRawAfterExplicit={unexpected_found}"
            )

            log_reason = (
                f"Validation FAIL (Strict Rule: Extraneous Numbers): Disallowed_Extras={{ {formatted_disallowed} }}. "
                f"CurrentBeatExplicitlyAllowed={current_beat_explicitly_allowed_numbers}, ForbiddenSet={forbidden_atoms}, "
                f"OperandCount={operand_count}, FoundInText={found_numbers}, UnexpectedRawAfterExplicit={unexpected_found}"
            )
            logger.debug(log_reason)

            # Log the failed attempt with error code
            _log_failed_validation(text, validation_report)
            return False

        logger.debug(f"Validation PASS (Strict)")
        logger.debug(
            f"--> Context: AllowedOperands={allowed_atoms}, ForbiddenSet={forbidden_atoms}, OperandCount={operand_count}, Result={correct_result_for_beat}, IntermediateSum={intermediate_sum_allowed}, FoundInText={found_numbers}"
        )
        return True

    return validate


# Add a helper function to save failed validation attempts
def _log_failed_validation(text: str, validation_report: dict):
    """
    Save failed validation attempts for diagnostic purposes.
    This provides a detailed record of why each beat was rejected.
    Additionally writes to the LLM turns log to keep all information in one place.
    """
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
    system_prompt = f"""You are a master storyteller and creative naming expert. Your task is to generate a short, evocative, and thematic 'narrative anchor'.

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
def perform_llm_static_content_validation(
    beat_text: str,
    current_op_node: OpNode,
    world_info: dict,
    inputs_str_for_validation: str,
    action_description_for_validation: str,
    overall_ground_truth_answer: int,
    expected_beat_result: int,
    sample_index: int | None = None,
    config_obj: Config = config,
    logger_obj: logging.Logger = logger,
) -> tuple[bool, str]:
    # DEPRECATED: This function is no longer called in the main flow as it has been replaced by the
    # iterative validation approach in _generate_and_llm_validate_beat. Keep for reference or potential future use.
    logger_obj.info(
        f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Performing LLM static content validation on beat text."
    )

    system_prompt = (
        "You are a meticulous AI assistant specialized in validating narrative logic within the Verbose ListOps benchmark. "
        "Your task is to determine if a given narrative 'beat' (a story segment) correctly and faithfully executes a specified mathematical operation on a given set of inputs, "
        "while adhering to strict rules about how those inputs (especially results from prior conceptual steps) are used. "
        "Focus solely on input fidelity, operational accuracy, result consistency, and overall adherence to instructions."
    )

    user_prompt_parts = [
        f"### World Context:\\n",
        f"- Genre: {world_info.get('genre', 'N/A')}\\n",
        f"- Setting: {world_info.get('setting', 'N/A')}\\n",
        f"- Primary Object: {world_info.get('object', 'items')}\\n\\n",
        f"### Current Operation Analysis:\\n",
        f"- Operation Type: {current_op_node.op} ({OP_LABELS.get(current_op_node.op, current_op_node.op)})\\n",
        f"- Expected Inputs (Narrative Anchors from prior steps + New Atomic Numbers for this step): {inputs_str_for_validation}\\n",
        f"  (IMPORTANT: Conceptual anchors like 'The Alpha Confluence' represent specific numerical results from previous steps and MUST be treated as those exact numbers in the current operation described by the narrative beat.)\\n",
        f"- Expected Numerical Result of THIS operation (if not final root op): {expected_beat_result}\\n",
        f"- Overall Ground Truth for the entire problem (should NOT be mentioned unless this is the *very final* operation and even then, only implied, never explicitly stated): {overall_ground_truth_answer}\\n\\n",
        f"### Task Description Given to Narrative Generator for this Beat:\\n",
        f'\\"\\"\\"{action_description_for_validation}\\"\\"\\"\\\\n\\\\n',  # Corrected escape
        f"### Generated Narrative Beat to Validate:\\n",
        f'\\"\\"\\"{beat_text}\\"\\"\\"\\\\n\\\\n',  # Corrected escape
        f"### Validation Task & Criteria:\\n",
        f"1. Input Fidelity: Does the 'Generated Narrative Beat' base its described actions/calculations faithfully and exclusively on the 'Expected Inputs' listed above? Specifically, were all conceptual anchors treated as their underlying numerical values and combined with any new atomic numbers for THIS operation? Were any expected inputs ignored, or unmentioned inputs invented by the narrative? This is the MOST CRITICAL check.\\n",
        f"2. Operational Accuracy: Does the narrative accurately portray the '{current_op_node.op}' operation being performed on the complete and correct set of inputs (as determined in step 1)?\\n",
        f"3. Result Consistency: If the operation's result is mentioned or implied in the narrative, is it consistent with the '{expected_beat_result}'? (Note: For the *final root operation* of the entire story, the numerical result should NOT be explicitly stated in the narrative beat itself. Its absence in that specific case is correct.)\\n",
        f"4. Adherence to Task: Does the beat strictly follow the 'Task Description Given to Narrative Generator' without deviating, introducing uninstructed numbers, or revealing forbidden numbers (like the overall_ground_truth_answer prematurely)?\\n\\n",
        f"### Your Response Format (Strict JSON Only):\\n",
        f"Provide your response as a SINGLE JSON object with two keys: 'is_beat_valid' (boolean: true ONLY if all checks above, especially Input Fidelity, pass; false otherwise) and 'reasoning' (string: a concise explanation, MANDATORY if false, detailing the primary failure, e.g., 'Narrative ignored anchored input X (value Y) and used invented number Z.'). Example:\\n",
        f'{{ \\"is_beat_valid\\": false, \\"reasoning\\": \\"Narrative ignored the anchored input \'The Alpha Confluence\' (value 25) and instead used an invented number 30. It also incorrectly stated the result as 50 instead of 45.\\" }}\\n',
        f"Output ONLY the JSON object. No other text, apologies, or explanations outside the JSON structure.",
    ]
    user_prompt = "".join(user_prompt_parts)

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
                max_completion_tokens=350,
                temperature=config_obj.STATIC_BEAT_VALIDATION_TEMP,
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
                cleaned_output = raw_llm_output.strip()
                match = re.search(r"\{\s*\"is_beat_valid\"", cleaned_output)
                if match:
                    cleaned_output = cleaned_output[match.start() :]
                if cleaned_output.startswith("{") and not cleaned_output.endswith("}"):
                    last_brace = cleaned_output.rfind("}")
                    if last_brace != -1:
                        cleaned_output = cleaned_output[: last_brace + 1]
                elif not cleaned_output.startswith("{") and cleaned_output.endswith(
                    "}"
                ):
                    first_brace = cleaned_output.find("{")
                    if first_brace != -1:
                        cleaned_output = cleaned_output[first_brace:]

                result_json = json.loads(cleaned_output)
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
            except json.JSONDecodeError as e:
                logger_obj.warning(
                    f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Failed to parse static beat validation JSON response: {e}. Raw: '{raw_llm_output}'"
                )
                return (
                    False,
                    f"Failed to parse LLM validator JSON response: {e}. Raw: {raw_llm_output[:150]}...",
                )
            except Exception as e:  # Catch other parsing/processing errors
                logger_obj.error(
                    f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Unexpected error processing static beat validation response: {e}. Raw: '{raw_llm_output}'"
                )
                return (
                    False,
                    f"Unexpected error processing LLM validator response: {e}. Raw: {raw_llm_output[:150]}...",
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


# --- NEW FUNCTION: _generate_and_llm_validate_beat (Iterative LLM Validation Loop) ---
def _generate_and_llm_validate_beat(
    original_user_message_for_generator: str, # The initial prompt for the beat generator
    system_prompt_for_generator: str,
    world_info: dict, # For validator prompt
    current_op_node: OpNode, # For validator prompt
    conceptual_inputs_str: str, # For validator prompt
    atomic_inputs_words_str: str, # For validator prompt
    action_description: str, # For validator prompt
    expected_beat_result_words: str | None, # For validator prompt
    current_max_beat_completion_tokens: int,
    sample_index: int,
    context_config: Config, # Pass the main config object
    logger_obj: logging.Logger,
    encoder_obj: any # Pass the tokenizer
) -> str | None: # Returns validated beat text or None
    
    # --- Accumulators for history ---
    history_of_attempts = [] # List of strings (generated beats)
    history_of_critiques = [] # List of dicts (parsed JSON feedback from validator)
    # ---

    for iteration in range(1, context_config.MAX_LLM_VALIDATION_ITERATIONS + 1):
        logger_obj.info(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validation Loop Iteration: {iteration}/{context_config.MAX_LLM_VALIDATION_ITERATIONS}")

        # 1. Prepare Generator Prompt with Full History
        current_generator_user_prompt = original_user_message_for_generator # Base prompt

        if iteration > 1:
            history_prompt_addition = "\n\n--- PREVIOUS ATTEMPTS AND FEEDBACK HISTORY ---\n"
            for i in range(len(history_of_attempts)):
                history_prompt_addition += f"\n**--- Round {i+1} ---**\n"
                attempt_text = history_of_attempts[i]
                if attempt_text.startswith("ERROR_DURING_GENERATION:"): # Handle cases where generation itself failed
                    history_prompt_addition += f"**Your Attempt {i+1} (failed during generation):**\n{attempt_text}\n\n"
                else:
                    history_prompt_addition += f"**Your Attempt {i+1}:**\n{attempt_text}\n\n"
                
                history_prompt_addition += f"**Validator Feedback for Attempt {i+1}:**\n"
                critique = history_of_critiques[i] # Should always exist if attempt exists
                history_prompt_addition += f"  Explanation: {critique.get('explanation_for_generator', 'N/A')}\n"
                history_prompt_addition += f"  Overall Summary for Revision: {critique.get('overall_revision_summary_for_generator_prompt', 'N/A')}\n"
                
                # Add suggested revisions if available and in a useful format
                if critique.get("suggested_revisions") and isinstance(critique.get("suggested_revisions"), list):
                    suggested_revisions = critique.get("suggested_revisions")
                    if suggested_revisions:
                        history_prompt_addition += f"  Suggested Revisions:\n"
                        for rev in suggested_revisions:
                            if isinstance(rev, dict):
                                rev_type = rev.get("type", "change")
                                rev_reason = rev.get("reason", "improve the narrative")
                                history_prompt_addition += f"    - {rev_type}: {rev_reason}\n"

            # Add the most recent critique again clearly for the current revision task
            most_recent_critique = history_of_critiques[-1]
            last_attempt_number = iteration - 1
            history_prompt_addition += (
                f"\n--- CURRENT TASK: REVISE ATTEMPT {last_attempt_number} (FROM ROUND {last_attempt_number} ABOVE) ---\n"
                f"**Based on Validator Feedback for Attempt {last_attempt_number}:** {most_recent_critique.get('overall_revision_summary_for_generator_prompt', 'Please revise based on issues found.')}\n\n"
                f"Please provide a new version (Attempt {iteration}) of the scene, incorporating ALL feedback received so far (especially the most recent) "
                f"to meet the original task requirements and number rules. Ensure the new scene continues from the 'Prior Scene Snippet' if one was part of the original task. "
                f"Output ONLY the revised narrative text."
            )
            current_generator_user_prompt += history_prompt_addition
            generator_temp = context_config.BEAT_REVISION_TEMP
        else:
            generator_temp = context_config.BEAT_GEN_TEMP
        
        # --- Token Count Check for Generator Prompt (Crucial with Full History) ---
        generator_prompt_tokens = 0
        if encoder_obj: # Check if encoder_obj is available
            try:
                generator_prompt_tokens = len(encoder_obj.encode(system_prompt_for_generator + current_generator_user_prompt))
            except Exception as e_encode:
                logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error encoding generator prompt for token count: {e_encode}. Assuming high token count.")
                generator_prompt_tokens = context_config.MODEL_MAX_CONTEXT_TOKENS // 2 # Assume half of max context to be safe
        else:
            logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Tokenizer (encoder_obj) not available. Cannot check prompt token count. Proceeding with caution.")
            # If no encoder, we can't accurately check, so we might skip or assume it fits.
            # For safety, if no encoder, we could potentially restrict history depth, but for now, it proceeds.

        # Using MAX_API_TOKEN_LIMIT for max_completion_tokens to avoid truncation due to reasoning/tool use,
        # but this is for the *API call itself*. The *actual content tokens* are what we budget for.
        # Let's use a more descriptive name for what the generator is expected to output.
        EXPECTED_MAX_BEAT_OUTPUT_TOKENS = context_config.BEAT_MAX_TOKENS 
        
        # Check against configured model context limit
        if generator_prompt_tokens + EXPECTED_MAX_BEAT_OUTPUT_TOKENS > context_config.MODEL_MAX_CONTEXT_TOKENS:
            logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Iteration {iteration}: Combined prompt ({generator_prompt_tokens} toks) "
                             f"and max expected output ({EXPECTED_MAX_BEAT_OUTPUT_TOKENS} toks) = {generator_prompt_tokens + EXPECTED_MAX_BEAT_OUTPUT_TOKENS} toks, "
                             f"which would exceed model context limit ({context_config.MODEL_MAX_CONTEXT_TOKENS}). Failing beat.")
            # If history causes overflow, we might need a strategy to shorten it. For now, it fails.
            # One simple strategy: if history_of_attempts has items, pop the oldest and retry this iteration's prompt construction?
            # This would be complex to manage within this loop without causing infinite sub-loops.
            # For now, if it's too long, it fails the beat.
            return None 
        # ---

        # 2. Generate Beat (or revision)
        try:
            log_prompt(
                header=f"LLM Beat Gen (Iterative Loop Attempt {iteration}) Op: {current_op_node.op}",
                prompt=f"System: {system_prompt_for_generator}\nUser: {current_generator_user_prompt}",
                sample_index=sample_index
            )
            resp_gen = _chat_completion_call(
                model=MODEL, # Main generator model
                messages=[
                    {"role": "system", "content": system_prompt_for_generator},
                    {"role": "user", "content": current_generator_user_prompt},
                ],
                max_completion_tokens=current_max_beat_completion_tokens,
                temperature=generator_temp,
                reasoning={"exclude": True} 
            )
            
            generated_text_raw = ""
            if resp_gen and resp_gen.choices and len(resp_gen.choices) > 0 and resp_gen.choices[0].message:
                _content = resp_gen.choices[0].message.content
                generated_text_raw = _content if _content is not None else ""
            
            # Apply cleaning logic (adapted from _generate_narrative_recursive)
            lines = generated_text_raw.splitlines()
            filtered_lines = []
            prompt_echoing_fragments = (
                "Imply the sum", "reference to the previous", "Narrate comparing", "This scene resolves",
                "The outcome MUST be", "Narrate an action", "Narrate an event", "The final quantity MUST become"
            )
            analysis_patterns_to_remove = [
                r"^\s*Critique:.*$", r"^\s*Checklist:.*$", r"^\s*Analysis:.*$", r"^\s*Rules check:.*$",
                r"^\s*MUST INCLUDE:.*$", r"^\s*MUST AVOID:.*$", r"^\s*Reasoning:.*$", r"^\s*Validation:.*$",
                r"^\s*Following the rules:.*$", r"^\s*Based on the instructions:.*$", r"^\s*The task is to.*$",
                r"^\s*The prompt asks.*$", r"^\s*Scene instructions:.*$", r"^\s*Instructions:.*$",
                r"^\s*Number rules:.*$", r"^\s*Rules:.*$", r"^\s*Confidence Score:.*$", r"^\s*Mental Sandbox:.*$",
                r"^\s*\d+\.\s*STRICT:.*$", r"^\s*\d+\.\s*MAY USE.*$", r"^\s*Output ONLY.*$", r"^\s*REMINDER:.*$",
                r"^\s*Okay.*$", r"^\s*Certainly.*$", r"^\s*```json.*$", r"^\s*```.*$", r"^\s*Generation:\s*",
                r"^\s*Narrative:\s*", r"^\s*Scene \d+:\s*", r"^\s*Beat \d+:\s*", r"^\s*Refinement \d+.*$",
                r"^\s*Yes\s*$", r"^\s*N/A\s*$", r"^\s*\[.*?\]\s*$", r"^\s*-\s.*$", r"^\s*\*\s.*$",
                r"^\s*Outcome is.*$", r"^\s*System:.*$", r"^\s*User:.*$", r"^\s*Check\..*$", r"^\s*Task:.*$",
                r"^\s*\?.*$"
            ]
            for line in lines:
                stripped_line = line.strip()
                is_analysis_or_echo = False
                for pattern in analysis_patterns_to_remove:
                    if re.match(pattern, stripped_line, re.IGNORECASE):
                        is_analysis_or_echo = True
                        # logger_obj.debug(f"Cleaning (iterative beat loop): Removing line matching pattern '{pattern}': '{line}'")
                        break
                if is_analysis_or_echo:
                    continue
                if any(stripped_line.lower().startswith(fragment.lower()) for fragment in prompt_echoing_fragments):
                    # logger_obj.debug(f"Cleaning (iterative beat loop): Removing line starting with prompt fragment: '{line}'")
                    is_analysis_or_echo = True
                if not is_analysis_or_echo:
                    filtered_lines.append(line)
            generated_text_cleaned = "\n".join(filtered_lines).strip()

            # Check for API refusals or empty content
            if not generated_text_cleaned or generated_text_cleaned.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Generator produced API refusal or empty text in iteration {iteration}.")
                generated_text_cleaned = ""  # Ensure it's empty for consistent handling
            
            log_prompt(
                header=f"LLM Beat Gen (Iterative Loop Attempt {iteration}, Cleaned) Op: {current_op_node.op}",
                prompt=f"Cleaned Generation:\n{generated_text_cleaned}",
                sample_index=sample_index
            )

            if not generated_text_cleaned:
                # Add a placeholder critique if generation is empty, so history structure is maintained
                history_of_attempts.append("") 
                history_of_critiques.append({
                    "is_valid": False,
                    "explanation_for_generator": "The generation was empty or contained only removable boilerplate.",
                    "overall_revision_summary_for_generator_prompt": "The previous generation was empty or boilerplate. Please generate the narrative scene as requested by the original task."
                })
                continue

            history_of_attempts.append(generated_text_cleaned) # Add successful generation to history

        except Exception as e_gen:
            logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error generating beat in LLM loop iteration {iteration}: {e_gen}")
            history_of_attempts.append(f"ERROR_DURING_GENERATION: {str(e_gen)[:500]}") # Log error as attempt, truncated
            history_of_critiques.append({
                "is_valid": False,
                "explanation_for_generator": f"An error occurred during generation: {str(e_gen)[:500]}",
                "overall_revision_summary_for_generator_prompt": "An error occurred during the previous generation attempt. Please try generating the scene again, focusing on the original task."
            })
            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(1)
                continue
            else:
                return None

        # 3. LLM Validate Beat
        # Construct validator user prompt
        validator_system_prompt = "You are an expert AI literary critic and logic checker. Your task is to evaluate a short story segment (a 'beat') that is supposed to narrate a specific mathematical operation. You must determine if the narrative accurately and coherently portrays this operation using the given inputs. Provide your feedback in a structured JSON format."
        
        validator_user_prompt = f"""World Context:
- Genre: {world_info.get("genre", "N/A")}
- Setting: {world_info.get("setting", "N/A")}
- Primary Object: {world_info.get("object", "items")}

Beat Task:
- Operation: {current_op_node.op} ({OP_LABELS.get(current_op_node.op, current_op_node.op)})
- Conceptual Inputs (from previous steps): {conceptual_inputs_str}
- New Atomic Inputs for this beat: {atomic_inputs_words_str}
- Intended Narrative Action: {action_description}
- Expected Numerical Result (for this beat, if not root): {expected_beat_result_words}

Generated Beat Text to Evaluate:
---
{generated_text_cleaned}
---

Validation Criteria:
1. Task Adherence: Does the narrative accurately reflect the specified 'Operation' and 'Intended Narrative Action'? (e.g., if SUM, does it describe combining things?)
2. Input Fidelity: Does the narrative correctly incorporate ALL 'Conceptual Inputs' and 'New Atomic Inputs'? Are any missing or ignored? Are any unmentioned inputs invented?
3. Narrative Coherence: Is the story segment logical and easy to understand in the context of the task?
4. Result Alignment (for non-root nodes if explicit mention is expected, or implicit alignment otherwise): Does the narrative lead to a situation consistent with the 'Expected Numerical Result'? (For root nodes, the result should NOT be stated).

Output Instructions:
Respond ONLY with a single JSON object.
If the beat is valid according to the criteria:
{{
  "is_valid": true,
  "explanation_for_audit": "Concise reason why it's valid."
}}
If the beat is NOT valid:
{{
  "is_valid": false,
  "explanation_for_generator": "Detailed explanation of the primary flaws for the generator LLM.",
  "overall_revision_summary_for_generator_prompt": "A concise (1-2 sentence) instruction for the generator on what to fix. E.g., 'Revise to ensure the narrative describes summing all inputs, including X and Y, instead of selecting the maximum.'",
  "suggested_revisions": [
    // Optional: array of specific suggestions if you can make them very concrete
    // {{ "type": "change_focus", "from": "problematic phrase", "to": "suggested focus/phrase", "reason": "..." }},
    // {{ "type": "ensure_mention", "item": "missing item", "reason": "..." }}
  ]
}}
"""  # Note: Double curly braces for JSON examples in f-string
        
        try:
            log_prompt(
                header=f"LLM Beat Validator Prompt (Iterative Loop Attempt {iteration}) Op: {current_op_node.op}",
                prompt=f"System: {validator_system_prompt}\nUser: {validator_user_prompt}",
                sample_index=sample_index
            )
            resp_val = _chat_completion_call(
                model=context_config.LLM_VALIDATOR_MODEL,
                messages=[
                    {"role": "system", "content": validator_system_prompt},
                    {"role": "user", "content": validator_user_prompt},
                ],
                max_completion_tokens=700, # Increased for potentially larger JSON feedback
                temperature=context_config.LLM_VALIDATOR_TEMP,
                reasoning={"exclude": True}
            )
            validator_raw_output = ""
            if resp_val and resp_val.choices and len(resp_val.choices) > 0 and resp_val.choices[0].message:
                 _content_val = resp_val.choices[0].message.content
                 validator_raw_output = _content_val if _content_val is not None else ""
            
            log_prompt(
                header=f"LLM Beat Validator Raw Response (Iterative Loop Attempt {iteration}) Op: {current_op_node.op}",
                prompt=f"Raw Output:\n{validator_raw_output}",
                sample_index=sample_index
            )

            try:
                # Using existing clean_and_parse_json_block for robust parsing
                validation_result = clean_and_parse_json_block(validator_raw_output) 
            except json.JSONDecodeError as e_json:
                logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Failed to parse LLM validator JSON in iteration {iteration}. Error: {e_json}. Raw: {validator_raw_output[:300]}...")
                validation_result = {
                    "is_valid": False,
                    "explanation_for_generator": "Validator response was not valid JSON or was empty.",
                    "overall_revision_summary_for_generator_prompt": "The validator's response was unparsable or empty. Please try generating the scene again, focusing on the original task and ensuring clear narrative structure."
                }
            
            history_of_critiques.append(validation_result) 

            if validation_result.get("is_valid"):
                logger_obj.info(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validator PASSED beat in iteration {iteration}. Audit: {validation_result.get('explanation_for_audit')}")
                return generated_text_cleaned # Success!
            else:
                logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validator FAILED beat in iteration {iteration}. Feedback: {validation_result.get('explanation_for_generator')}")
                # Loop continues for next iteration with this feedback
        
        except Exception as e_val:
            logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error during LLM validation call/processing in iteration {iteration}: {e_val}")
            # Ensure history_of_critiques has a corresponding entry if an exception occurred before it was added.
            # Check if len(history_of_critiques) is less than len(history_of_attempts)
            if len(history_of_critiques) < len(history_of_attempts):
                history_of_critiques.append({
                    "is_valid": False,
                    "explanation_for_generator": f"An error occurred during validation: {str(e_val)[:500]}",
                    "overall_revision_summary_for_generator_prompt": "An error occurred during the validation step. Please try generating the scene again, focusing on the original task."
                })
            elif history_of_critiques: # If it exists, update the last one's reason if it was a generic one before this error.
                 # This case might be less common if an exception during call means no prior critique was added for *this specific validation attempt*.
                 # The more likely scenario is handled by the len check above.
                 pass

            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(1)
                continue
            else:
                return None # Max iterations reached due to validation error

    logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Beat failed LLM validation after {context_config.MAX_LLM_VALIDATION_ITERATIONS} iterations.")
    return None # Failed all iterations


# --- Narrative Generation with Parent Operator Prompting ---
def _generate_narrative_recursive(  # Line ~1315
    node: Node,
    context: "GenerationContext",
    is_root: bool,
):
    """
    Recursive helper for POST-ORDER strict narrative generation using FEW-SHOT examples.
    Processes children first, then the current node.
    Modifies the context object directly.
    """
    world = context.world
    config = context.config
    encoder = context.encoder
    p_inflect = context.p_inflect
    logger = context.logger
    narrative_anchor_map = context.narrative_anchor_map

    node_id = id(node)
    narrative_anchor = narrative_anchor_map.get(
        node_id, f"the_unnamed_{node.op}_entity" if isinstance(node, OpNode) else "atom"
    )
    # logger.debug(
    #     f"_generate_narrative_recursive (POST-ORDER): processing node {getattr(node, 'op', 'Atom')} with narrative anchor '{narrative_anchor}'"
    # ) # Replaced by more specific log below for OpNodes

    # Log token budget at the start of processing this node
    token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
    token_remaining = config.MAX_TOTAL_TOKENS - context.tokens_used - SAFETY_MARGIN
    logger.debug(
        f"TOKEN BUDGET START [{getattr(node, 'op', 'Atom')}/{narrative_anchor}]: "
        f"{context.tokens_used}/{config.MAX_TOTAL_TOKENS} tokens used ({token_percentage:.1f}%), "
        f"Remaining: {token_remaining} (with {SAFETY_MARGIN} margin)"
    )

    if isinstance(node, Atom):
        logger.debug(f"Node is Atom ({node.n}), returning.")
        return

    # --- Add detailed OpNode processing log here ---
    if isinstance(node, OpNode):
        # This log replaces the one from the removed _generate_narrative_recursive_with_tracking
        # and provides context for the current operation being processed by this function instance.
        node_id_for_log = id(node)
        op_for_log = getattr(node, "op", "OpNode_No_Op_Attr")  # More robust getattr
        # The narrative_anchor variable defined a few lines above can be reused if its scope is appropriate,
        # otherwise, re-fetch or ensure it's correctly representing the current node's anchor.
        # Using the 'narrative_anchor' defined at the start of this function for consistency:
        logger.debug(
            f"[Sample {context.sample_index + 1}] _generate_narrative_recursive: "
            f"Processing OpNode: {op_for_log}, Anchor: '{narrative_anchor}', Root: {is_root}, "
            f"Beat: {context.beat_counter['current'] + 1}/{context.beat_counter['total']}"  # +1 as beat_counter increments after this
        )

    child_narrative_anchors = []
    for child_index, child in enumerate(node.children):
        logger.debug(
            f"Processing child {child_index+1}/{len(node.children)} of {node.op} ({narrative_anchor})"
        )

        _generate_narrative_recursive(
            child,
            context,
            is_root=False,
        )
        if isinstance(child, OpNode) and id(child) in narrative_anchor_map:
            child_narrative_anchors.append(narrative_anchor_map[id(child)])
        elif isinstance(child, OpNode):
            logger.warning(
                f"OpNode child {child.op} of parent {node.op} has no narrative anchor in map."
            )

        # Check token budget after processing each child
        token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
        token_remaining = config.MAX_TOTAL_TOKENS - context.tokens_used - SAFETY_MARGIN

        if context.tokens_used >= config.MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(
                f"⚠️ TOKEN LIMIT REACHED after child {child_index+1}/{len(node.children)} "
                f"of {node.op} ({narrative_anchor}) - Used: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} "
                f"tokens ({token_percentage:.1f}%), Safety margin: {SAFETY_MARGIN}, "
                f"Remaining: {token_remaining}. HALTING GENERATION FOR THIS BRANCH."
            )
            return
        else:
            logger.debug(
                f"TOKEN BUDGET AFTER CHILD {child_index+1}/{len(node.children)} of {node.op} ({narrative_anchor}): "
                f"{context.tokens_used}/{config.MAX_TOTAL_TOKENS} tokens ({token_percentage:.1f}%), "
                f"Remaining: {token_remaining} (with {SAFETY_MARGIN} margin)"
            )

    logger.debug(
        f"Finished processing children for operator {getattr(node, 'op', 'Atom')} ({narrative_anchor}). Now processing node itself."
    )
    if is_root:
        token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
        logger.info(
            f"ROOT NODE ({node.op}): Starting beat generation. Current tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} "
            f"({token_percentage:.1f}%), Remaining: {config.MAX_TOTAL_TOKENS - context.tokens_used - SAFETY_MARGIN} "
            f"(with {SAFETY_MARGIN} margin)"
        )

    context.beat_counter["current"] += 1
    logger.info(
        f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for operator {node.op} ({narrative_anchor})"
    )
    op_label = OP_LABELS.get(node.op, node.op)

    direct_atom_children = [c for c in node.children if isinstance(c, Atom)]
    operand_count = len(direct_atom_children)
    direct_atom_values = {a.n for a in direct_atom_children}

    # Get the correct result early to avoid UnboundLocalError
    correct_result = node.value

    required_atoms_for_beat = set(direct_atom_values)
    logger.debug(
        f"Required atoms for beat {node.op} ({narrative_anchor}): {required_atoms_for_beat}"
    )

    # --- Determine forbidden numbers for the current beat's prompt and validation ---
    # Start with atoms introduced in previous beats
    forbidden_for_current_beat_rules = context.introduced_atoms.copy()

    # Add the overall ground truth answer to this base forbidden set for every beat
    if context.overall_ground_truth_answer is not None:
        # We add it here. If the overall_ground_truth_answer happens to be
        # one of the required_atoms_for_beat OR the correct_result of THIS beat,
        # the validator's logic and the prompt construction will allow it specifically for this beat.
        # Otherwise, its appearance will be an error.
        forbidden_for_current_beat_rules.add(context.overall_ground_truth_answer)
        logger.debug(
            f"Added overall_ground_truth_answer ({context.overall_ground_truth_answer}) to forbidden set for beat {node.op} ({narrative_anchor}). Current forbidden set: {forbidden_for_current_beat_rules}"
        )

    # 'truly_forbidden_for_prompt' are numbers that should be listed in the "MUST AVOID" part of the prompt.
    # These are numbers from the `forbidden_for_current_beat_rules` set,
    # EXCLUDING any that are direct atomic inputs for the *current* beat.
    # The `correct_result` of the current beat is handled separately in prompt construction (must be included)
    # and by the validator (implicitly allowed if it's the step's result).
    truly_forbidden_for_prompt = (
        forbidden_for_current_beat_rules - required_atoms_for_beat
    )

    # --- REFINED HANDLING OF OVERALL_GROUND_TRUTH_ANSWER ---
    # For the root node, we don't want to list the final answer as "MUST AVOID" in the prompt
    # because it's the result of this node's calculation. Its absence is managed by:
    # 1. Not putting it in "MUST INCLUDE" for the root prompt
    # 2. The action_description telling the LLM not to state the final result
    # 3. Setting enforce_result_presence=False in the validator
    if (
        is_root
        and context.overall_ground_truth_answer is not None
        and context.overall_ground_truth_answer in truly_forbidden_for_prompt
    ):
        # Remove the overall_ground_truth_answer from "MUST AVOID" for the root node's prompt
        truly_forbidden_for_prompt.remove(context.overall_ground_truth_answer)
        logger.debug(
            f"ROOT NODE: Removed overall_ground_truth_answer ({context.overall_ground_truth_answer}) from truly_forbidden_for_prompt to avoid conflicting instructions."
        )

    # Create a specialized set for the validator that's more nuanced than the prompt's "MUST AVOID" list
    forbidden_atoms_for_validator = context.introduced_atoms.copy()
    # For non-root nodes, always include the overall_ground_truth_answer in the validator's forbidden set
    # For the root node, only include it if it's somehow different from the node's own calculated result
    # (which would be a safety check, as normally they should be the same)
    if context.overall_ground_truth_answer is not None:
        if not is_root:  # For intermediate nodes
            forbidden_atoms_for_validator.add(context.overall_ground_truth_answer)
            logger.debug(
                f"VALIDATOR: Added overall_ground_truth_answer ({context.overall_ground_truth_answer}) to validator's forbidden set for non-root node."
            )
        elif context.overall_ground_truth_answer != correct_result:
            # Edge case: For root node, only add if it's different from the calculated result
            forbidden_atoms_for_validator.add(context.overall_ground_truth_answer)
            logger.debug(
                f"VALIDATOR: Added overall_ground_truth_answer ({context.overall_ground_truth_answer}) to validator's forbidden set for root node because it differs from calculated result ({correct_result})."
            )

    # Remove the required atoms from the validator's forbidden set
    forbidden_atoms_for_validator = (
        forbidden_atoms_for_validator - required_atoms_for_beat
    )

    primary_object = world["object"]

    direct_atom_sum = None
    # Calculate direct_atom_sum for both AVG and SM if direct atoms exist
    if node.op in ["AVG", "SM"] and direct_atom_children:
        try:
            direct_atom_sum = sum(a.n for a in direct_atom_children)
            logger.debug(
                f"{node.op} Beat: Calculated direct atom sum for validation/prompt: {direct_atom_sum}"
            )
        except Exception as e:
            logger.error(
                f"Error calculating direct atom sum for {node.op} node {narrative_anchor}: {e}"
            )
            direct_atom_sum = None

    # --- MODIFIED SECTION TO INCLUDE THE RESULT IN "MUST INCLUDE" FOR THE PROMPT ---
    # This set will hold all numbers that the LLM should be explicitly told to include.
    # It starts with the direct atomic operands for the current beat.
    numbers_to_mention_in_prompt = set(required_atoms_for_beat)

    # Only add correct_result to "MUST INCLUDE" if:
    # 1. It's NOT the root node (we never want to mention the final result)
    # 2. AND for intermediate nodes, only if ALLOW_IMPLICIT_INTERMEDIATE_RESULTS is False (requiring explicit mention)
    if not is_root and correct_result is not None:
        if not context.config.ALLOW_IMPLICIT_INTERMEDIATE_RESULTS:
            # Only add the result if we're requiring explicit mention of intermediate results
            numbers_to_mention_in_prompt.add(correct_result)
            logger.debug(
                f"Added correct_result ({correct_result}) to numbers_to_mention_in_prompt because ALLOW_IMPLICIT_INTERMEDIATE_RESULTS=False"
            )
        else:
            logger.debug(
                f"Intermediate node beat: correct_result ({correct_result}) NOT added to numbers_to_mention_in_prompt because ALLOW_IMPLICIT_INTERMEDIATE_RESULTS=True"
            )
    elif is_root and correct_result is not None:
        logger.debug(
            f"Root node beat: correct_result ({correct_result}) will NOT be added to numbers_to_mention_in_prompt for the LLM."
        )

    # Now, create the sorted list of strings for the prompt.
    # Using a set first (numbers_to_mention_in_prompt) handles potential duplicates
    # (e.g., if the result happens to be one of the input atoms, like MIN(5,10,5) -> 5).
    must_include_list = []
    if numbers_to_mention_in_prompt:
        # Sort for consistent prompt generation
        sorted_numbers_to_mention = sorted(list(numbers_to_mention_in_prompt))
        must_include_list = [num_to_words(x) for x in sorted_numbers_to_mention]

    if not must_include_list:
        must_include_combined_str = (
            "None applicable for this step (only uses results from previous steps)"
        )
    elif len(must_include_list) == 1:
        must_include_combined_str = must_include_list[0]
    elif len(must_include_list) == 2:
        must_include_combined_str = " and ".join(must_include_list)
    else:
        must_include_combined_str = (
            ", ".join(must_include_list[:-1]) + ", and " + must_include_list[-1]
        )
    # --- END OF MODIFIED SECTION ---

    if truly_forbidden_for_prompt:  # This uses the updated `truly_forbidden_for_prompt`
        must_avoid_str = ", ".join(
            num_to_words(x) for x in sorted(truly_forbidden_for_prompt)
        )
    else:
        must_avoid_str = "None"

    may_use_parts = []
    operand_count_is_forbidden_for_prompt = operand_count in truly_forbidden_for_prompt
    if operand_count > 0 and operand_count not in truly_forbidden_for_prompt:
        operand_count_word = num_to_words(operand_count)
        may_use_parts.append(
            f"the number {operand_count} ('{operand_count_word}', the count of direct items being considered)"
        )
    if 1 not in truly_forbidden_for_prompt:
        may_use_parts.append("the number 1 ('one')")
    if may_use_parts:
        may_use_clause = f"*   You MAY use { ' and '.join(may_use_parts) } for natural narrative flow.\n"
    else:
        may_use_clause = ""

    ultra_strict_instruction = (
        f"**ULTRA-STRICT NUMBER RULES (THIS SCENE ONLY):**\n"
        f"- MUST INCLUDE: {must_include_combined_str}\n"
        f"{may_use_clause.replace('*   You MAY use', '- MAY USE:').strip()}\n"
        f"- NO OTHER NUMBERS ALLOWED{' (except intermediate sum ' + str(direct_atom_sum) + ' for AVG)' if node.op == 'AVG' and direct_atom_sum is not None else ''}."
    )

    object_list_str_for_preamble = ""
    if direct_atom_values:
        items = [num_to_words(x) for x in sorted(direct_atom_values)]
        if len(items) == 1:
            object_list_str_for_preamble = items[0]
        elif len(items) == 2:
            object_list_str_for_preamble = " and ".join(items)
        else:
            object_list_str_for_preamble = ", ".join(items[:-1]) + ", and " + items[-1]

    scene_preamble = ""
    if direct_atom_values:
        if node.op == "SUM":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                f"They collect all of these {primary_object}, combining their haul."
            )
        elif node.op == "MED":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                f"They need to select the one corresponding to the median (middle) value."
            )
        elif node.op == "MIN":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or storage areas containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                f"But for a reason you concoct, they can only access or retrieve from the area containing the smallest quantity (MIN) of {primary_object}."
            )
        elif node.op == "MAX":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or storage areas containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                f"But for a reason you concoct, they can only access or retrieve from the area containing the largest quantity (MAX) of {primary_object}."
            )
        elif node.op == "AVG":
            sum_str = (
                str(direct_atom_sum)
                if direct_atom_sum is not None
                else "calculated sum"
            )
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                f"These are to be averaged (integer floored). The sum of these direct items is {sum_str}."
            )
        elif node.op == "SM":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                f"They collect all these {primary_object}, combining their haul. The operation involves keeping only the final digit of this total."
            )

    has_operator_children = bool(child_narrative_anchors)
    has_direct_atom_children = bool(direct_atom_values)

    # Remove the duplicate assignment since we already assigned correct_result earlier
    # correct_result = node.value - This line is being removed

    formatted_child_names_str = "None"
    if has_operator_children:
        formatted_child_names_list = [f"'{name}'" for name in child_narrative_anchors]
        formatted_child_names_str = ", ".join(formatted_child_names_list)

    formatted_direct_values_str = "None"
    if has_direct_atom_children:
        formatted_direct_values_list = [
            num_to_words(v) for v in sorted(direct_atom_values)
        ]
        formatted_direct_values_str = ", ".join(formatted_direct_values_list)

    input_description_parts = []
    if has_operator_children:
        input_description_parts.append(
            f"the outcome(s) from previous step(s) conceptually known as {formatted_child_names_str}"
        )
    if has_direct_atom_children:
        input_description_parts.append(
            f"newly discovered quantities ({formatted_direct_values_str})"
        )

    if not input_description_parts:
        inputs_str = "inputs determined entirely by context (e.g., selecting between previous outcomes)"
    else:
        inputs_str = " and ".join(input_description_parts)

    action_description = ""
    # --- MODIFIED action_description based on is_root ---
    if is_root:
        # For the root node, describe the operation but DO NOT state the final numerical result.
        # The inputs_str will correctly list the direct operands for the final operation.
        if node.op == "SUM":
            action_description = (
                f"Narrate the final action (e.g., gathering all remaining resources, tallying the final count) involving {inputs_str}. "
                f"The story should conclude with the characters having performed the action that would lead to their final quantity of {primary_object}. "
                f"DO NOT state the numerical value of this final tally in this scene. The question at the end will ask for it."
            )
        elif node.op == "AVG":
            # For AVG, direct_atom_sum might be relevant if the root AVG combines new atoms with an anchor.
            # If the root AVG only averages previous anchors, direct_atom_sum would be None or based on 0 atoms.
            direct_atom_sum_words_for_root_avg = (
                num_to_words(direct_atom_sum)
                if direct_atom_sum is not None and direct_atom_children
                else "any new direct items"
            )
            action_description = (
                f"Narrate the final event (e.g., determining the final average, establishing the ultimate equilibrium) involving {inputs_str}. "
                f"The characters perform the necessary steps to determine the final average value of {primary_object}. "
                f"DO NOT state the numerical value of this final average in this scene. The question at the end will ask for it. "
                f"If relevant for clarity (e.g., if new items {formatted_direct_values_str} are part of this final average), you MAY mention their intermediate sum ({direct_atom_sum_words_for_root_avg}) before they are combined with prior results for averaging, but the final average itself must not be stated."
            )
        elif node.op == "SM":
            sm_intermediate_sum_words_for_root = "unknown total"
            try:
                # Ensure child values are evaluated if not already (should be by now)
                child_values_for_sm_root = [eval_node(c) for c in node.children]
                sm_intermediate_sum_for_root = sum(child_values_for_sm_root)
                sm_intermediate_sum_words_for_root = num_to_words(
                    sm_intermediate_sum_for_root
                )
            except Exception as e_sm_sum_root:
                logger.warning(
                    f"SM Root Beat: Could not calculate intermediate sum for prompt: {e_sm_sum_root}"
                )

            action_description = (
                f"Narrate the final action involving {inputs_str}. The characters combine these inputs (conceptually reaching a temporary total around {sm_intermediate_sum_words_for_root}). "
                f"Then, describe a plausible event that makes them keep only a quantity representing the final digit of that total. "
                f"The story should resolve with them having performed the action that results in this final quantity. DO NOT state its numerical value. "
                f"Examples of events: a magical lock consumes all but the final unit, a tax collector takes most, a reaction leaves only the core essence."
            )
        elif node.op == "MAX":
            action_description = (
                f"Narrate the final comparison of {inputs_str}. The characters identify the item or quantity with the largest value. "
                f"Describe their realization or decision-making process that identifies this largest quantity. "
                f"The scene should conclude with them having made this crucial identification. "
                f"DO NOT explicitly state that they take or end up with this largest quantity. DO NOT state its numerical value as their final possession. "
                f"The story implies they will act on this decision, but the narrative ends before that final acquisition."
            )
        elif node.op == "MIN":
            action_description = (
                f"Narrate the final comparison of {inputs_str}. The characters identify the item or quantity with the smallest value. "
                f"Describe their realization or decision-making process that identifies this smallest quantity. "
                f"The scene should conclude with them having made this crucial identification. "
                f"DO NOT explicitly state that they take or end up with this smallest quantity. DO NOT state its numerical value as their final possession. "
                f"The story implies they will act on this decision, but the narrative ends before that final acquisition."
            )
        elif node.op == "MED":
            action_description = (
                f"Narrate the final evaluation of {inputs_str} numerically. The characters select the item or quantity representing the middle value (when sorted). "
                f"Describe their realization or decision-making process that identifies this median quantity. "
                f"The scene should conclude with them having made this crucial identification. "
                f"DO NOT explicitly state that they take or end up with this median quantity. DO NOT state its numerical value as their final possession. "
                f"The story implies they will act on this decision, but the narrative ends before that final acquisition."
            )
        else:  # Fallback for any other ops if added later
            action_description = (
                f"Narrate the final application of '{op_label}' to {inputs_str}. The story should conclude with the characters having performed the action to reach their final result. "
                f"DO NOT state the numerical value of this result in this scene."
            )
    else:  # NOT is_root (existing logic for intermediate nodes)
        correct_result_words = num_to_words(correct_result)

        # Check if intermediate results must be explicitly mentioned
        if context.config.ALLOW_IMPLICIT_INTERMEDIATE_RESULTS:
            # Intermediate results are OPTIONAL - use more flexible phrasing
            if node.op == "SUM":
                action_description = (
                    f"Narrate an action (e.g., gathering, merging) involving {inputs_str} to calculate their sum. "
                    f"The narrative should clearly imply this summation occurred. You MAY explicitly state that the total quantity becomes {correct_result_words} {primary_object}, but it's not strictly required if the summation is clear from the action."
                )
            elif node.op == "AVG":
                direct_atom_sum_words = (
                    num_to_words(direct_atom_sum)
                    if direct_atom_sum is not None
                    else "calculated sum"
                )
                action_description = (
                    f"Narrate an event (e.g., balancing, averaging mechanism) involving {inputs_str} to determine their integer average (floored). "
                    f"The narrative should clearly imply this averaging. You MAY explicitly state the outcome is {correct_result_words} {primary_object}. "
                    f"You MAY also mention the intermediate sum ({direct_atom_sum_words} from direct inputs ({formatted_direct_values_str})) if it aids the narrative."
                )
            elif node.op == "SM":
                sm_intermediate_sum_words = "unknown"  # Default
                try:
                    child_values = [eval_node(c) for c in node.children]
                    sm_intermediate_sum = sum(child_values)
                    sm_intermediate_sum_words = num_to_words(sm_intermediate_sum)
                except Exception as e:
                    logger.warning(
                        f"SM Beat: Could not calculate intermediate sum for prompt explanation: {e}"
                    )
                    sm_intermediate_sum_words = "unknown"

                action_description = (
                    f"Narrate an action involving {inputs_str}. The characters combine these inputs (conceptually reaching a temporary total around {sm_intermediate_sum_words}). "
                    f"Then, describe an event that makes them keep only a quantity equal to the final digit of that total. "
                    f"The narrative should imply this sum-modulo-10 operation. You MAY explicitly state the final quantity becomes {correct_result_words} {primary_object}, but it's not strictly required."
                )
            elif node.op == "MAX":
                action_description = (
                    f"Narrate comparing {inputs_str} to find the largest value. Justify the choice. "
                    f"The narrative should clearly imply this selection. You MAY explicitly state the chosen quantity is {correct_result_words} {primary_object}, but it's not strictly required."
                )
            elif node.op == "MIN":
                action_description = (
                    f"Narrate comparing {inputs_str} to find the smallest value. Justify the choice. "
                    f"The narrative should clearly imply this selection. You MAY explicitly state the chosen quantity is {correct_result_words} {primary_object}, but it's not strictly required."
                )
            elif node.op == "MED":
                action_description = (
                    f"Narrate evaluating {inputs_str} numerically to select the median (middle) value. Justify the choice. "
                    f"The narrative should clearly imply this selection. You MAY explicitly state the chosen quantity is {correct_result_words} {primary_object}, but it's not strictly required."
                )
            else:
                action_description = (
                    f"Narrate applying '{op_label}' to {inputs_str}. The narrative should clearly imply this operation. "
                    f"You MAY explicitly state the outcome is {correct_result_words} {primary_object}, but it's not strictly required."
                )
        else:
            # Original strict behavior - intermediate results MUST be mentioned
            if node.op == "SUM":
                action_description = (
                    f"Narrate an action (e.g., gathering, merging) involving {inputs_str}. "
                    f"The outcome MUST be that the total quantity becomes exactly **{correct_result_words}** {primary_object}. Imply the sum through action."
                )
            elif node.op == "AVG":
                direct_atom_sum_words = (
                    num_to_words(direct_atom_sum)
                    if direct_atom_sum is not None
                    else "calculated sum"
                )
                action_description = (
                    f"Narrate an event (e.g., balancing, averaging mechanism) involving {inputs_str}. "
                    f"The outcome MUST be exactly **{correct_result_words}** {primary_object} (the floored average). "
                    f"You MAY mention the intermediate sum ({direct_atom_sum_words} from direct inputs ({formatted_direct_values_str})) if needed for the narrative, but the final result is key."
                )
            elif node.op == "SM":
                correct_result_words = num_to_words(correct_result)
                sm_intermediate_sum = "unknown"
                try:
                    child_values = [eval_node(c) for c in node.children]
                    sm_intermediate_sum = sum(child_values)
                    sm_intermediate_sum_words = num_to_words(sm_intermediate_sum)
                except Exception as e:
                    logger.warning(
                        f"SM Beat: Could not calculate intermediate sum for prompt explanation: {e}"
                    )
                    sm_intermediate_sum_words = "unknown"

                action_description = (
                    f"Narrate an action involving {inputs_str}. The characters combine these inputs (reaching a temporary total conceptually around {sm_intermediate_sum_words}). "
                    f"Then, describe an event that makes them keep only the final digit of that total. "
                    f"The narrative should explicitly state that they end up with exactly **{correct_result_words}** {primary_object}."
                )
            elif node.op == "MAX":
                action_description = f"Narrate comparing {inputs_str} to find the largest value. The characters MUST select and end up with exactly **{correct_result_words}** {primary_object}."
            elif node.op == "MIN":
                action_description = f"Narrate comparing {inputs_str} to find the smallest value. The characters MUST select and end up with exactly **{correct_result_words}** {primary_object}."
            elif node.op == "MED":
                action_description = f"Narrate evaluating {inputs_str} numerically. The characters MUST select and end up with exactly **{correct_result_words}** {primary_object}, the median."
            else:
                action_description = f"Narrate applying '{op_label}' to {inputs_str}. The final operation result MUST be exactly **{correct_result_words}** {primary_object}."

    # Determine if the result presence should be enforced for the validator
    enforce_validator_result_presence = False  # Default to False
    if is_root:
        # For root node: NEVER enforce the presence of the final answer
        enforce_validator_result_presence = False
        logger.debug(
            f"Root node: Setting enforce_validator_result_presence=False to allow omitting the final answer"
        )
    else:
        # For intermediate nodes: Only enforce if ALLOW_IMPLICIT_INTERMEDIATE_RESULTS is False
        if not context.config.ALLOW_IMPLICIT_INTERMEDIATE_RESULTS:
            enforce_validator_result_presence = True
            logger.debug(
                f"Intermediate node: Setting enforce_validator_result_presence=True because ALLOW_IMPLICIT_INTERMEDIATE_RESULTS=False"
            )
        else:
            logger.debug(
                f"Intermediate node: Setting enforce_validator_result_presence=False because ALLOW_IMPLICIT_INTERMEDIATE_RESULTS=True"
            )

    validate_beat_numbers = make_number_validator(
        allowed_atoms=required_atoms_for_beat,
        forbidden_atoms=forbidden_atoms_for_validator,  # Use our refined set instead of truly_forbidden_for_prompt
        operand_count=operand_count,
        correct_result_for_beat=correct_result,
        intermediate_sum_allowed=(
            direct_atom_sum if node.op in ["AVG", "SM"] else None
        ),
        enforce_result_presence=enforce_validator_result_presence,  # Use our dynamic setting
        operation_type=node.op,
    )
    forbidden_for_padding = context.introduced_atoms.union(required_atoms_for_beat)
    forbidden_for_padding.add(correct_result)
    logger.debug(
        f"Creating padding validator. correct_result={correct_result}, forbidden_for_padding={forbidden_for_padding}"
    )
    validate_padding = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=forbidden_for_padding,
        operand_count=0,
        correct_result_for_beat=config.INVALID_RESULT_PLACEHOLDER,
        strict_zero=True,
    )

    beat_text = None
    candidate_text = ""

    # Define the system prompt for narrative beat generation (used by initial generation and revisions)
    system_prompt_for_generator = """You are a master storyteller. Your task is to write a single scene that contributes to an ongoing narrative.
Focus solely on advancing the story as specified in the task. Do not include explanations or analysis of your work.
The story involves mathematical operations represented through narrative actions. Pay careful attention to the number rules.
Produce ONLY clean narrative text."""

    # Create the initial user message for the beat generator
    # This will be passed to _generate_and_llm_validate_beat and used for the first attempt
    context_snippet = clean_snippet(
        context.last_scene_text, max_len=config.BEAT_CONTEXT
    )
    initial_user_message_for_generator = (
        f"BEAT {context.beat_counter['current']}/{context.beat_counter['total']}\n\n"
        f"**Current Status:** {scene_preamble}\n\n"
        f"**Task:** {action_description}\n\n"
        f"{ultra_strict_instruction}\n\n"
        f"**Prior Scene Snippet:**\n{context_snippet}\n\n"
        f"**Write the next scene continuing directly from the prior scene with a focus on the TASK above.**"
    )

    # Set max completion tokens for the beat
    current_max_beat_completion_tokens = config.BEAT_MAX_TOKENS

    # --- MODIFIED BEAT GENERATION LOOP (Outer Retries) ---
    beat_text_final_validated = None  # Will store the beat if all validation passes

    for attempt in range(1, config.MAX_BEAT_RETRIES + 1):  # Outer retry loop
        logger.info(
            f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Anchor: {narrative_anchor}] Outer Beat Gen Attempt: {attempt}/{config.MAX_BEAT_RETRIES}"
        )

        # Prepare parameters for _generate_and_llm_validate_beat
        conceptual_inputs_str_for_val = (
            ", ".join([f"'{name}'" for name in child_narrative_anchors])
            if child_narrative_anchors
            else "None"
        )
        atomic_inputs_words_str_for_val = (
            ", ".join([num_to_words(v) for v in sorted(direct_atom_values)])
            if direct_atom_values
            else "None"
        )

        expected_beat_result_words_for_val = None
        if correct_result is not None:
            # Only provide expected result words if it's an intermediate node AND explicit results are required,
            # OR if implicit results are allowed (validator might check for alignment).
            # For root node, it should be None for the validator prompt to check for *absence* of explicit mention.
            if not is_root:
                expected_beat_result_words_for_val = num_to_words(correct_result)
            # else: for root node, keep it None - validator understands this means result shouldn't be stated.

        # Call the new iterative LLM validation function
        llm_validated_beat = _generate_and_llm_validate_beat(
            original_user_message_for_generator=initial_user_message_for_generator,
            system_prompt_for_generator=system_prompt_for_generator,
            world_info=world,
            current_op_node=node,  # op_node is current 'node' in this context
            conceptual_inputs_str=conceptual_inputs_str_for_val,
            atomic_inputs_words_str=atomic_inputs_words_str_for_val,
            action_description=action_description,
            expected_beat_result_words=expected_beat_result_words_for_val,
            current_max_beat_completion_tokens=current_max_beat_completion_tokens,
            sample_index=context.sample_index,
            context_config=config,
            logger_obj=logger,
            encoder_obj=encoder,  # Pass the encoder
        )

        if llm_validated_beat:
            logger.info(
                f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Anchor: {narrative_anchor}] Beat PASSED iterative LLM validation loop. Proceeding to static number validation."
            )
            # Now, perform static validation on the LLM-validated beat
            log_prompt(
                header=f"Static Validation Input (Op: {node.op}, Anchor: {narrative_anchor}, Outer Attempt {attempt})",
                prompt=f"Text for static validation (after LLM validation loop):\n{llm_validated_beat}",
                sample_index=context.sample_index,
            )
            if validate_beat_numbers(
                llm_validated_beat
            ):  # your existing static validator
                logger.info(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Anchor: {narrative_anchor}] Static number validation PASSED after LLM validation loop. Beat is successful."
                )
                beat_text_final_validated = llm_validated_beat
                break  # Success for this beat, break outer retry loop
            else:
                logger.warning(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Anchor: {narrative_anchor}] Static number validation FAILED for LLM-validated beat. Outer attempt {attempt} failed."
                )
                # _log_failed_validation is called inside validate_beat_numbers
        else:
            logger.warning(
                f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Anchor: {narrative_anchor}] Iterative LLM validation loop failed to produce a beat. Outer attempt {attempt} failed."
            )

        # If either LLM loop or static validation failed, and more retries are left in outer loop:
        if attempt < config.MAX_BEAT_RETRIES:
            delay = config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
            logger.info(
                f"Retrying beat generation for {node.op} (Outer Attempt {attempt+1}/{config.MAX_BEAT_RETRIES}) in {delay:.2f}s"
            )
            time.sleep(delay)

    # After the outer loop:
    if beat_text_final_validated:
        beat_text = beat_text_final_validated  # This is the successfully generated and validated beat
        # ... (rest of your existing logic: append to scenes, update tokens, etc. - this starts below)
    else:
        logger.error(
            f"Operator {node.op} ({narrative_anchor}) failed after {config.MAX_BEAT_RETRIES} outer attempts (incl. LLM validation loops). Aborting narrative generation. {'(ROOT NODE)' if is_root else ''}"
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} ({narrative_anchor}) after all outer retries."
        )
    # --- END OF MODIFIED BEAT GENERATION LOOP ---

    # This 'if beat_text:' block and its contents are original logic that processes a successful beat
    if (
        beat_text
    ):  # This condition is now met if beat_text_final_validated was successful
        btoks = len(encoder.encode(beat_text))
        context.scenes.append(beat_text)
        context.tokens_used += btoks
        context.last_scene_text = beat_text
        context.introduced_atoms.update(required_atoms_for_beat)

        token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
        logger.debug(
            f"Beat {context.beat_counter['current']} successful. Introduced atoms updated: {context.introduced_atoms}"
        )
        logger.info(
            f"Beat {context.beat_counter['current']} successful. Tokens used this beat: {btoks}, "
            f"Total tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
            f"Remaining: {config.MAX_TOTAL_TOKENS - context.tokens_used - SAFETY_MARGIN} (with {SAFETY_MARGIN} margin)"
        )
    else:
        logger.error(
            f"Operator {node.op} ({narrative_anchor}) failed after {config.MAX_BEAT_RETRIES} attempts. Aborting narrative generation. {'(ROOT NODE)' if is_root else ''}"
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} ({narrative_anchor})"
        )

    # --- Enhanced Intelligent Padding Loop ---
    if not is_root:
        # Calculate current padding statistics
        current_padding_total = context.padding_stats["total_padding_tokens"]
        max_padding_allowed = context.padding_stats["max_padding_allowed"]
        padding_budget_remaining = max_padding_allowed - current_padding_total
        padding_usage_percent = (
            (current_padding_total / max_padding_allowed * 100)
            if max_padding_allowed > 0
            else 0
        )

        token_percentage_before_padding = (
            context.tokens_used / config.MAX_TOTAL_TOKENS
        ) * 100
        logger.info(
            f"PADDING BUDGET CHECK [{node.op}/{narrative_anchor}]: "
            f"Current tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage_before_padding:.1f}%), "
            f"Padding used so far: {current_padding_total}/{max_padding_allowed} tokens ({padding_usage_percent:.1f}%), "
            f"Padding budget remaining: {padding_budget_remaining} tokens, "
            f"Segments added so far: {context.padding_stats['padding_segments_added']}"
        )

        local_padding_segments_added = 0

        # Modified loop conditions to include padding budget check
        while (
            # Condition 1: Global token budget allows for more content
            context.tokens_used < config.MAX_TOTAL_TOKENS - SAFETY_MARGIN
            # Condition 2: We haven't added too many padding segments *locally* for this beat.
            # context.max_pad_paragraphs (from config.MAX_PAD_PARAGRAPHS) acts as the per-beat cap.
            and local_padding_segments_added < context.max_pad_paragraphs
            # Condition 3: Check if we have padding budget remaining
            and current_padding_total < max_padding_allowed
        ):
            # Estimate the cost of the next padding segment (prompt + completion)
            estimated_next_padding_segment_cost = MAX_PAD_COMPLETION_TOKENS + 100

            # Check if adding this segment would exceed padding budget
            if (
                current_padding_total + estimated_next_padding_segment_cost
                > max_padding_allowed
            ):
                logger.warning(
                    f"PADDING BUDGET LIMIT [{node.op}/{narrative_anchor}]: "
                    f"Current padding: {current_padding_total}/{max_padding_allowed} tokens ({padding_usage_percent:.1f}%), "
                    f"Next segment est. cost: +{estimated_next_padding_segment_cost}, "
                    f"Would exceed max padding budget. Stopping padding for this beat."
                )
                break

            token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
            estimated_after_padding = (
                context.tokens_used + estimated_next_padding_segment_cost
            )
            estimated_percentage_after_padding = (
                estimated_after_padding / config.MAX_TOTAL_TOKENS
            ) * 100

            if (
                context.tokens_used + estimated_next_padding_segment_cost
                > config.MAX_TOTAL_TOKENS - SAFETY_MARGIN
            ):
                logger.debug(
                    f"PADDING ABORT [{node.op}/{narrative_anchor}]: "
                    f"Current: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
                    f"Next padding est. cost: +{estimated_next_padding_segment_cost}, "
                    f"Would reach: {estimated_after_padding}/{config.MAX_TOTAL_TOKENS} ({estimated_percentage_after_padding:.1f}%), "
                    f"Max allowed: {config.MAX_TOTAL_TOKENS - SAFETY_MARGIN}. Stopping padding for this beat."
                )
                break  # Break from this beat's padding loop if next segment likely too costly

            # Increment counter for this beat's padding *before* attempting generation
            local_padding_segments_added += 1

            logger.debug(
                f"PADDING ATTEMPT [{node.op}/{narrative_anchor}]: Segment {local_padding_segments_added}/{context.max_pad_paragraphs}, "
                f"Current tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
                f"Current padding: {current_padding_total}/{max_padding_allowed} tokens ({padding_usage_percent:.1f}%)"
            )

            padding_system_prompt = "You are a concise storyteller adding descriptive filler. FOLLOW THE USER'S RULES EXACTLY."
            cleaned_snippet_padding = clean_snippet(
                context.last_scene_text, max_len=config.PADDING_CONTEXT
            )

            # Corrected f-string and multi-line string handling for padding_user_prompt
            padding_user_prompt = (
                f'Previous Scene Snippet: "...{cleaned_snippet_padding.replace("\n", " ")}"...\n\n'
                f"Task: Write ONE short paragraph (3-5 sentences) continuing the story. "
                f"Describe atmosphere, character reactions, or minor transitions. "
                f"DO NOT mention ANY numbers (digits or words like 'one', 'two', 'first', etc.). "
                f"Output ONLY the padding paragraph."
            )

            log_prompt(
                f"Padding Prompt Segment {local_padding_segments_added} after {node.op} ({narrative_anchor})",
                f"System: {padding_system_prompt}\nUser: {padding_user_prompt}",
                sample_index=context.sample_index,
            )

            estimated_pad_prompt_tokens = len(
                encoder.encode(padding_system_prompt + padding_user_prompt)
            )

            # This check is slightly more precise for the immediate call to generate_with_retry
            if would_exceed_budget(
                context.tokens_used,
                MAX_PAD_COMPLETION_TOKENS,  # Only consider the potential completion against the budget
                config.MAX_TOTAL_TOKENS,
                SAFETY_MARGIN,
            ):
                logger.warning(
                    f"Padding after beat for {node.op} ({narrative_anchor}): "
                    f"Budget check indicates insufficient space for padding segment {local_padding_segments_added} "
                    f"(max_completion {MAX_PAD_COMPLETION_TOKENS}). Current COMPLETION tokens: {context.tokens_used}. "
                    f"STOPPING padding for this beat."
                )
                break  # Break from this beat's padding loop

            padding_text = generate_with_retry(
                system_prompt=padding_system_prompt,
                user_prompt=padding_user_prompt,
                max_completion_tokens=MAX_PAD_COMPLETION_TOKENS,
                validate_fn=validate_padding,
                retries=config.MAX_PAD_RETRIES,
                sample_index=context.sample_index,
                temperature=config.CREATIVE_NARRATIVE_TEMP,
                reasoning_settings={"exclude": True},  # Always exclude thinking tokens
            )

            if padding_text:
                ptoks = len(encoder.encode(padding_text))

                # Check if this would exceed padding budget
                if current_padding_total + ptoks > max_padding_allowed:
                    logger.warning(
                        f"PADDING BUDGET EXCEEDED [{node.op}/{narrative_anchor}]: Generated padding segment ({ptoks} tokens) would "
                        f"exceed total padding budget ({current_padding_total + ptoks}/{max_padding_allowed}). Discarding."
                    )
                    break

                # Re-check budget with actual token count before adding, though prior checks should catch most overflows
                if (
                    context.tokens_used + ptoks
                    <= config.MAX_TOTAL_TOKENS - SAFETY_MARGIN
                ):
                    context.scenes.append(padding_text)
                    context.tokens_used += ptoks
                    context.last_scene_text = padding_text

                    # Update padding tracking
                    context.padding_stats["total_padding_tokens"] += ptoks
                    current_padding_total = context.padding_stats[
                        "total_padding_tokens"
                    ]  # Update local var
                    context.padding_stats["padding_segments_added"] += 1

                    # Calculate new percentages
                    token_percentage_after = (
                        context.tokens_used / config.MAX_TOTAL_TOKENS
                    ) * 100
                    padding_percentage_after = (
                        current_padding_total / max_padding_allowed * 100
                    )

                    logger.debug(
                        f"PADDING SUCCESS [{node.op}/{narrative_anchor}]: Segment {local_padding_segments_added} added. "
                        f"Size: {ptoks} tokens. Total now: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage_after:.1f}%), "
                        f"Padding total: {current_padding_total}/{max_padding_allowed} ({padding_percentage_after:.1f}%), "
                        f"Total padding segments: {context.padding_stats['padding_segments_added']}"
                    )
                else:
                    token_percentage_would_be = (
                        (context.tokens_used + ptoks) / config.MAX_TOTAL_TOKENS
                    ) * 100
                    logger.warning(
                        f"PADDING DISCARD [{node.op}/{narrative_anchor}]: Generated padding segment {local_padding_segments_added} "
                        f"({ptoks} tokens) would exceed limit. Current: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
                        f"With padding: {context.tokens_used + ptoks}/{config.MAX_TOTAL_TOKENS} ({token_percentage_would_be:.1f}%). Discarding."
                    )
                    # Even if we discard, we should break as we are at the budget limit.
                    break
            else:
                logger.warning(
                    f"PADDING FAIL [{node.op}/{narrative_anchor}]: Segment {local_padding_segments_added} generation failed "
                    f"(validator failed or API error after retries). Current tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} "
                    f"({token_percentage:.1f}%). Stopping further padding attempts for this beat."
                )
                break  # Break from this beat's padding loop if generation fails

        # Log summary if no padding was added
        if local_padding_segments_added == 0:
            logger.debug(
                f"NO PADDING ADDED [{node.op}/{narrative_anchor}]: Either token limit reached, "
                f"padding budget exceeded ({current_padding_total}/{max_padding_allowed} tokens), "
                f"or max segments per beat ({context.max_pad_paragraphs}) reached."
            )

    # End of function token budget summary
    token_percentage_end = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
    padding_percentage = (
        (
            context.padding_stats["total_padding_tokens"]
            / context.padding_stats["max_padding_allowed"]
            * 100
        )
        if context.padding_stats["max_padding_allowed"] > 0
        else 0
    )
    logger.debug(
        f"TOKEN BUDGET END [{getattr(node, 'op', 'Atom')}/{narrative_anchor}]: "
        f"{context.tokens_used}/{config.MAX_TOTAL_TOKENS} tokens ({token_percentage_end:.1f}%), "
        f"Padding: {context.padding_stats['total_padding_tokens']}/{context.padding_stats['max_padding_allowed']} ({padding_percentage:.1f}%), "
        f"Remaining: {config.MAX_TOTAL_TOKENS - context.tokens_used - SAFETY_MARGIN} (with {SAFETY_MARGIN} margin)"
    )


def generate_introduction_scene(
    world_info: dict, sample_index: int | None = None
) -> str | None:
    """
    Generates an introductory scene for the narrative.
    Returns the generated scene text as a string, or None if generation fails.
    """
    logger.info(f"[Sample {sample_index + 1}] Generating introduction scene...")

    # Ensure world_info is a dictionary
    if not isinstance(world_info, dict):
        logger.error(
            f"[Sample {sample_index + 1}] Invalid world_info type: {type(world_info)}. Expected dict."
        )
        return None

    # Extract world details with fallbacks
    genre = world_info.get("genre", "a fantastical")
    setting_desc = world_info.get("setting", "a land of mystery and intrigue")
    primary_object = world_info.get("object", "ancient artifacts")

    # Safely get character names
    characters = world_info.get("characters", [])
    if isinstance(characters, list) and characters:
        char_names = [
            char.get("name", f"Character {i+1}") for i, char in enumerate(characters)
        ]
        # Make a nice list of names
        if len(char_names) == 1:
            character_list_str = char_names[0]
        elif len(char_names) == 2:
            character_list_str = " and ".join(char_names)
        else:
            character_list_str = ", ".join(char_names[:-1]) + ", and " + char_names[-1]
    else:
        character_list_str = "a group of intrepid adventurers"
        logger.warning(
            f"[Sample {sample_index + 1}] No character names found or invalid format in world_info. Using default."
        )

    system_prompt = (
        "You are a creative writer tasked with crafting an engaging opening scene for a story. "
        "Your response should be ONLY the narrative text for this scene. "
        "Adhere strictly to the rule: NO NUMERICAL VALUES (digits or words like 'one', 'two', 'first', etc.) are allowed in this introductory scene."
    )

    user_prompt = (
        f"Genre: {genre}\n"
        f"Setting: {setting_desc}\n"
        f"Characters: {character_list_str}\n"
        f"Primary Object of Interest: {primary_object}\n\n"
        f"Task: Write a brief introductory scene (1-2 paragraphs, approximately 50-150 words). "
        f"This scene should establish the setting and introduce the characters (or group). "
        f"It should clearly state that the characters are embarking on a quest or have a strong interest in finding/understanding the '{primary_object}'. "
        f"Crucially, this scene must NOT contain any numbers or numerical concepts (e.g., 'three days', 'two paths', 'first'). "
        f"Focus on descriptive language and setting the stage for an adventure. "
        f"Output ONLY the narrative text for the scene. Do not include titles, headers, or any explanations."
    )

    # Define the validator for the introduction scene (strict_zero means no numbers allowed, except '1' if not forbidden)
    # For intro, we forbid '1' as well by not adding it to allowed_atoms and keeping forbidden_atoms empty.
    validate_intro = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=set(),  # No specific atoms are forbidden beyond the general "no numbers" rule
        operand_count=0,
        correct_result_for_beat=None,  # No result to check for intro
        intermediate_sum_allowed=None,
        strict_zero=True,  # Enforces no numbers (or only '1' if it were allowed, but it's not here)
        enforce_result_presence=False,
        operation_type="INTRO",
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=config.INTRO_MAX_TOKENS,  # Use a config value
        validate_fn=validate_intro,
        retries=config.INTRO_MAX_RETRIES,  # Use a config value
        sample_index=sample_index,
        temperature=config.CREATIVE_NARRATIVE_TEMP,  # Use a config value
        reasoning_settings={"exclude": True},
    )

    if intro_text:
        logger.info(
            f"[Sample {sample_index + 1}] Successfully generated introduction scene."
        )
        # Log the final, validated intro text
        log_prompt(
            header=f"Validated Introduction Scene - Sample {sample_index + 1}",
            prompt=f"Final Intro Text:\n{intro_text}",
            sample_index=sample_index,
        )
        return intro_text.strip()
    else:
        logger.error(
            f"[Sample {sample_index + 1}] Failed to generate a valid introduction scene after retries."
        )
        # Log failure
        log_prompt(
            header=f"Failed Introduction Scene Generation - Sample {sample_index + 1}",
            prompt="Intro generation failed validation or API calls after all retries.",
            sample_index=sample_index,
        )
        return None


def generate_narrative(
    ast: Node,
    world: dict,
    config: Config,
    encoder,
    p_inflect,
    logger,
    sample_index: int,
    overall_ground_truth_answer: int,  # ADDED: Receive the overall ground truth
) -> str | None:
    """
    Generate a structured narrative representation of the AST.
    Each operation is represented by a scene, carefully sequenced with
    intermediate node anchor names. Uses STRICT recursive generation.
    """
    logger.info(f"[Sample {sample_index + 1}] Starting narrative generation.")
    logger.debug(f"DEBUG: Using model: {MODEL}")
    logger.debug(f"DEBUG: AST: {ast_to_prefix(ast)}")

    # --- Initial Setup ---
    all_operator_nodes = [node for node in postorder(ast) if not isinstance(node, Atom)]
    all_atoms = set()
    for node in postorder(ast):
        if isinstance(node, Atom):
            all_atoms.add(node.n)
    logger.debug(f"DEBUG: All atoms in AST: {sorted(list(all_atoms))}")
    # Add the final answer to the log for debugging
    logger.debug(
        f"DEBUG: Overall ground truth (final answer): {overall_ground_truth_answer}"
    )

    operator_nodes = []
    narrative_anchor_map = {}
    intro_text = None
    scenes = []
    tokens_used = 0

    # --- Generate narrative anchors for op nodes ---
    if config.USE_NARRATIVE_ANCHORS:

        def generate_anchor_for_node(op_node):
            """Helper to generate anchors for intermediate nodes."""
            if not config.USE_LLM_NAMING:
                # Use deterministic naming if not using LLM
                return f"the_{op_node.op.lower()}_result_{id(op_node) % 1000:03d}"

            all_anchors_list = list(narrative_anchor_map.values())
            try:
                anchor = generate_narrative_anchor_with_llm(
                    world, op_node, all_anchors_list, sample_index=sample_index
                )

                if anchor:
                    return anchor
                else:
                    logger.warning(
                        f"Failed to generate LLM anchor for {op_node.op}. Using deterministic fallback."
                    )
                    return f"the_{op_node.op.lower()}_result_{id(op_node) % 1000:03d}"
            except Exception as e:
                logger.error(f"Error in narrative anchor generation: {e}")
                return f"the_{op_node.op.lower()}_result_{id(op_node) % 1000:03d}"

        # Process nodes in postorder for anchors (bottom-up)
        for node in postorder(ast):
            if isinstance(node, OpNode):
                # Use helper to generate anchor names
                anchor = generate_anchor_for_node(node)
                narrative_anchor_map[id(node)] = anchor
                operator_nodes.append(node)
                logger.debug(f"Added narrative anchor '{anchor}' for {node.op} node")
    else:
        # Without narrative anchors, just use basic names
        for node in postorder(ast):
            if isinstance(node, OpNode):
                narrative_anchor_map[id(node)] = (
                    f"the_{node.op.lower()}_result_{id(node) % 1000:03d}"
                )
                operator_nodes.append(node)

    # Now we have anchors for all operator nodes
    # Note on operator_nodes ordering: Because we populated it during postorder
    # traversal, it's already in the correct execution order (leaves first, root last).
    logger.info(f"Generated {len(narrative_anchor_map)} narrative anchors.")
    log_str = "Narrative anchors: " + ", ".join(
        [
            f"'{anchor}' ({node.op})"
            for node, anchor in [
                (n, narrative_anchor_map.get(id(n), "MISSING")) for n in operator_nodes
            ]
        ]
    )
    logger.debug(log_str)

    # --- Generate introductory scene ---
    # This now calls the full generate_introduction_scene function
    intro_text = generate_introduction_scene(world, sample_index=sample_index)

    if intro_text:  # Check if intro_text is not None and not empty
        # Further check if intro_text would exceed budget BEFORE adding
        intro_tokens = len(encoder.encode(intro_text))
        if (
            intro_tokens <= config.MAX_TOTAL_TOKENS - SAFETY_MARGIN
        ):  # Check against overall budget
            scenes.append(intro_text)
            tokens_used += intro_tokens
            logger.info(
                f"Generated and added introductory scene ({intro_tokens} tokens)."
            )
        else:
            logger.warning(
                f"Generated introductory scene ({intro_tokens} tokens) was too long and would exceed budget. "
                f"Not adding to narrative. Budget: {config.MAX_TOTAL_TOKENS}, Safety: {SAFETY_MARGIN}"
            )
            intro_text = None  # Ensure it's None if not added
    else:
        logger.warning(
            "Failed to generate valid introductory scene. Starting narrative without intro."
        )
        # intro_text is already None or empty if generation failed

    last_scene_text = intro_text if intro_text else "The story begins..."
    introduced_atoms_during_generation = set()
    total_beats = len(
        operator_nodes
    )  # This should be correct as operator_nodes are OpNode from postorder
    beat_counter = {"current": 0, "total": total_beats}

    # --- Create GenerationContext instance for sharing state ---
    context = GenerationContext(
        world=world,
        config=config,
        encoder=encoder,
        p_inflect=p_inflect,
        logger=logger,
        narrative_anchor_map=narrative_anchor_map,
        all_atoms=all_atoms,
        introduced_atoms=introduced_atoms_during_generation,  # Starts empty
        scenes=scenes,  # Starts with potential intro
        tokens_used=tokens_used,  # Starts with potential intro tokens
        last_scene_text=last_scene_text,  # Starts with potential intro text
        beat_counter=beat_counter,
        sample_index=sample_index,
        max_pad_paragraphs=config.MAX_PAD_PARAGRAPHS,
        overall_ground_truth_answer=overall_ground_truth_answer,
    )

    # Initialize padding budget tracking
    # Calculate max_padding_allowed based on tokens *remaining after intro* if intro was successful
    # Or based on total budget if intro failed or was skipped.
    tokens_available_for_narrative_and_padding = (
        config.MAX_TOTAL_TOKENS - tokens_used - SAFETY_MARGIN
    )
    max_padding_allowed = int(
        tokens_available_for_narrative_and_padding * config.PADDING_MAX_TOK_PERCENT
    )

    context.padding_stats["max_padding_allowed"] = max_padding_allowed
    logger.info(
        f"PADDING BUDGET INITIALIZED: Tokens after intro: {tokens_used}, "
        f"Available for narrative+padding: {tokens_available_for_narrative_and_padding}, "
        f"Max padding %: {config.PADDING_MAX_TOK_PERCENT*100:.1f}%, "
        f"Max padding tokens allowed: {max_padding_allowed}, "
        f"Max padding segments per beat: {config.MAX_PAD_PARAGRAPHS}"
    )

    # --- Start the POST-ORDER recursive generation ---
    try:
        _generate_narrative_recursive(
            ast,
            context,  # Pass the mutable context
            is_root=True,
        )
    except BeatGenerationError as e:
        logger.error(f"Narrative generation aborted due to beat failure: {e}")
        return None  # Indicate failure
    except Exception as e:
        logger.error(
            f"Unexpected error during recursive narrative generation: {e}",
            exc_info=True,
        )
        return None  # Indicate failure

    # Final Assembly
    if not context.scenes:
        logger.error("Narrative generation resulted in no scenes.")
        return None

    narrative_body = "\n\n".join(context.scenes).strip()

    # Return just the narrative body without the question
    # This allows the validator to properly evaluate if the narrative avoids stating the final answer
    final_token_count = len(encoder.encode(narrative_body))
    if final_token_count > config.MAX_TOTAL_TOKENS:
        logger.warning(
            f"Final generated narrative ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({config.MAX_TOTAL_TOKENS}). Truncation might occur."
        )

    total_padding_tokens = context.padding_stats["total_padding_tokens"]
    # max_padding_allowed already calculated and stored in context
    padding_segments_added = context.padding_stats["padding_segments_added"]

    padding_percentage_of_max = (
        (total_padding_tokens / context.padding_stats["max_padding_allowed"] * 100)
        if context.padding_stats["max_padding_allowed"] > 0
        else 0
    )
    # Use context.tokens_used for total tokens in the narrative (including padding)
    padding_percentage_of_total_narrative = (
        (total_padding_tokens / context.tokens_used * 100)
        if context.tokens_used > 0
        else 0
    )

    logger.info(
        f"PADDING FINAL SUMMARY: "
        f"Padding tokens: {total_padding_tokens}/{context.padding_stats['max_padding_allowed']} ({padding_percentage_of_max:.1f}% of max allowed for padding), "
        f"Padding percentage of total narrative tokens: {padding_percentage_of_total_narrative:.1f}%, "
        f"Padding segments added: {padding_segments_added}"
    )

    failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
    if os.path.exists(failed_validations_dir):
        validation_files = [
            f
            for f in os.listdir(failed_validations_dir)
            if f.startswith(f"validation_fail_") and f"sample_{sample_index+1}" in f
        ]  # Filter for current sample

        if validation_files:
            failures_by_reason = {}
            failures_by_op = {}

            for file_name in validation_files:
                try:
                    # Basic parsing from filename, assuming format like "validation_fail_OP_REASON_timestamp.json"
                    parts = file_name.split("_")
                    if len(parts) >= 4:  # Check if there are enough parts
                        op_type = parts[2]  # validation_fail_OPERATION_REASON...
                        reason_code = parts[3]

                        failures_by_reason[reason_code] = (
                            failures_by_reason.get(reason_code, 0) + 1
                        )
                        failures_by_op[op_type] = failures_by_op.get(op_type, 0) + 1
                except IndexError:
                    logger.warning(
                        f"Could not parse validation failure filename: {file_name}"
                    )

            logger.info(f"[Sample {sample_index + 1}] VALIDATION FAILURES SUMMARY:")
            logger.info(
                f"  Total validation failures for this sample: {len(validation_files)}"
            )
            if failures_by_reason:
                logger.info(f"  Failures by reason: {failures_by_reason}")
            if failures_by_op:
                logger.info(f"  Failures by operation: {failures_by_op}")
        else:
            logger.info(
                f"[Sample {sample_index + 1}] No validation failure files found for this sample."
            )

    logger.info(
        f"Successfully generated narrative for sample {sample_index + 1}. Final context tokens: {context.tokens_used}, Narrative tokens: {final_token_count}"
    )
    return narrative_body.strip()


def main(
    config: Config,
    num_samples: int = NUM_SAMPLES_TO_GENERATE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    # REMOVE initial_account_usage: float | None = None # We'll fetch it inside
):
    """Generate samples with strict validation."""
    # Test log output to ensure logger is working properly
    logger.info(
        "START OF MAIN FUNCTION - THIS LOG SHOULD APPEAR IN verbose_listops.log"
    )

    # --- Fetch initial account usage --- ADD THIS BLOCK ---
    initial_account_usage = None  # Initialize
    if (
        client
        and OPENROUTER_API_KEY
        and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE"
    ):
        logger.info("Fetching initial OpenRouter account usage...")
        initial_account_usage = (
            rate_limiter.update_limits_from_api()
        )  # Call modified function
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
    # --- END ADDED BLOCK ---

    # --- Dynamic Filename Generation ---
    sanitized_model_name = MODEL.replace("/", "_").replace(":", "-")

    # Ensure datasets directory exists
    os.makedirs(DATASETS_DIR, exist_ok=True)

    output_file = os.path.join(
        DATASETS_DIR,
        f"DATASET_"
        f"{config.MAX_TOTAL_TOKENS}tok_"
        f"{config.MAX_OPS}-mxops_"
        f"{config.MIN_ARITY}-arity_"
        f"{config.MAX_BRANCH}-mxbrch_"
        f"{sanitized_model_name}_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        f".jsonl",
    )
    logger.info(f"Output filename (dynamic): {output_file}")
    logger.info(
        f"Script started. Generating {num_samples} samples using up to {max_workers} workers."
    )
    logger.info(
        f"Using {config.FEW_SHOT_EXAMPLES} few-shot examples for narrative generation."
    )

    samples_generated_successfully = 0
    samples_failed = 0
    start_time = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(generate_single_sample, i): i for i in range(num_samples)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_data = future.result()
                if sample_data:
                    results.append(sample_data)
                    samples_generated_successfully += 1
                else:
                    samples_failed += 1
            except Exception as exc:
                logger.error(
                    f"[Sample {index + 1}] task generated exception: {exc}",
                    exc_info=True,
                )  # Log stack trace for task exceptions
                samples_failed += 1
    logger.info(
        f"Parallel generation complete. Writing {samples_generated_successfully} samples to {output_file}..."
    )
    try:

        write_mode = "w"  # Or 'a' if you prefer appending
        logger.info(f"Opening {output_file} in '{write_mode}' mode.")
        with open(output_file, write_mode, encoding="utf-8") as f:
            for (
                sample_data
            ) in results:  # Iterate through successfully generated results
                try:
                    f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                except TypeError as e:

                    logger.error(
                        f"Serialization failed for sample {sample_data.get('id', 'Unknown')}: {e}. Skipping write for this sample."
                    )

                    samples_failed += 1
                    samples_generated_successfully -= (
                        1  # Decrement success as it wasn't written
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error writing sample {sample_data.get('id', 'Unknown')}: {e}. Skipping write."
                    )
                    samples_failed += 1
                    samples_generated_successfully -= 1  # Decrement success
    except IOError as e:
        logger.error(f"Fatal file write error opening/writing {output_file}: {e}")

        samples_failed += (
            samples_generated_successfully  # All successful generations failed to write
        )
        samples_generated_successfully = 0
    except Exception as e:
        logger.error(f"Unexpected error during file writing phase: {e}", exc_info=True)
        samples_failed += samples_generated_successfully
        samples_generated_successfully = 0

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"--- Batch generation complete ---")
    logger.info(f"Total samples attempted: {num_samples}")
    logger.info(f"Successfully generated and written: {samples_generated_successfully}")
    logger.info(f"Failed generations or writes: {samples_failed}")
    total_count = num_samples
    success_count = samples_generated_successfully
    success_rate = (success_count / total_count * 100) if total_count else 0
    logger.info(
        f"Overall success rate (generated AND written): {success_rate:.2f}% ({success_count}/{total_count})"
    )
    logger.info(f"Total time: {total_time:.2f} seconds")
    if samples_generated_successfully > 0:
        logger.info(f"Dataset output file: {output_file}")
        print(f"\nDataset saved to: {output_file}")
    else:
        logger.warning(
            f"No samples were successfully generated and written. Output file '{output_file}' may be empty or non-existent."
        )

    # Print datasets directory location for user reference
    print(f"\nDatasets are saved in: {os.path.abspath(DATASETS_DIR)}")

    # Add a clear console output showing total execution time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {seconds:.2f}s"
    else:
        time_str = f"{seconds:.2f}s"

    print(f"\n✅ Total execution time: {time_str} ({total_time:.2f} seconds)")

    # --- PROD_RUN: Validation and Cleaning Step ---
    if (
        PROD_RUN
        and samples_generated_successfully > 0
        and output_file
        and os.path.exists(output_file)
    ):
        logger.info(
            f"--- Starting PROD_RUN validation and cleaning for {output_file} ---"
        )
        # Assuming validator.py is in the same directory as verbose-listops.py
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        validator_script_path = os.path.join(current_script_dir, "validator.py")
        validator_results_path = output_file + ".validation_results.jsonl"

        if not os.path.exists(validator_script_path):
            logger.error(
                f"Validator script not found at {validator_script_path}. Cannot perform cleaning."
            )
        else:
            cmd = [
                sys.executable,
                validator_script_path,
                output_file,
                "--output-results",
                validator_results_path,
            ]
            try:
                logger.info(f"Running validator command: {' '.join(cmd)}")
                run_result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, encoding="utf-8"
                )
                logger.info("Validator process stdout:")
                for line in run_result.stdout.splitlines():
                    logger.info(f"VALIDATOR_STDOUT: {line}")
                if run_result.stderr:
                    logger.warning("Validator process stderr:")
                    for line in run_result.stderr.splitlines():
                        logger.warning(f"VALIDATOR_STDERR: {line}")
                logger.info(
                    f"Validator finished. Results expected in {validator_results_path}"
                )

                bad_sample_ids = set()
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
                                        bad_sample_ids.add(sample_id_to_remove)
                                    else:
                                        logger.warning(
                                            f"Validator result line {line_num} missing 'id': {res_line.strip()}"
                                        )
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Could not parse validator result line {line_num}: {res_line.strip()}"
                                )
                    logger.info(
                        f"Identified {len(bad_sample_ids)} samples to remove based on validator results."
                    )

                    if bad_sample_ids:
                        temp_cleaned_output_file = output_file + ".cleaned.tmp"
                        good_samples_written = 0
                        original_sample_count = 0

                        with open(output_file, "r", encoding="utf-8") as f_orig, open(
                            temp_cleaned_output_file, "w", encoding="utf-8"
                        ) as f_temp:
                            for line_num, dataset_line in enumerate(f_orig, 1):
                                original_sample_count += 1
                                try:
                                    sample_in_dataset = json.loads(dataset_line)

                                    # Make sure each sample has all fields expected by validator
                                    # Add these fields if missing but we have equivalent fields
                                    if (
                                        "narrative_with_question"
                                        not in sample_in_dataset
                                        and "full_prompt" in sample_in_dataset
                                    ):
                                        sample_in_dataset["narrative_with_question"] = (
                                            sample_in_dataset["full_prompt"]
                                        )

                                    if (
                                        "ast" not in sample_in_dataset
                                        and "ast_prefix" in sample_in_dataset
                                    ):
                                        sample_in_dataset["ast"] = sample_in_dataset[
                                            "ast_prefix"
                                        ]

                                    if (
                                        "ground_truth" not in sample_in_dataset
                                        and "ground_truth_answer" in sample_in_dataset
                                    ):
                                        sample_in_dataset["ground_truth"] = (
                                            sample_in_dataset["ground_truth_answer"]
                                        )

                                    # Check if this sample should be kept
                                    if (
                                        sample_in_dataset.get("id")
                                        not in bad_sample_ids
                                    ):
                                        # Write the sample with the added fields if needed
                                        f_temp.write(
                                            json.dumps(sample_in_dataset) + "\n"
                                        )
                                        good_samples_written += 1
                                    else:
                                        logger.debug(
                                            f"Removing sample {sample_in_dataset.get('id')} (from line {line_num}) due to validation status."
                                        )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Could not parse line {line_num} in original dataset '{output_file}' during filtering: {dataset_line.strip()}. Discarding this line."
                                    )

                        # Construct the new filename with [CLEAN] prefix
                        output_dir = os.path.dirname(output_file)
                        base_filename = os.path.basename(output_file)
                        new_base_filename = f"[CLEAN]{base_filename}"
                        new_output_file_path = os.path.join(
                            output_dir, new_base_filename
                        )

                        # Move the temporary cleaned file to the new path instead of overwriting
                        shutil.move(temp_cleaned_output_file, new_output_file_path)
                        deleted_count = original_sample_count - good_samples_written
                        logger.info(
                            f"Removed {deleted_count} bad samples. {good_samples_written} cleaned samples saved to {new_output_file_path}."
                        )
                        logger.info(
                            f"(Original dataset with all generated samples remains at: {output_file})"
                        )  # Add note about original file
                        print(
                            f"PROD_RUN: After validation and cleaning, {good_samples_written} samples saved to {new_output_file_path}."
                        )
                        # Update samples_generated_successfully for any final tally if needed, though new print is clearer
                        # samples_generated_successfully = good_samples_written
                    else:
                        logger.info(
                            "No samples identified for removal by the validator (all were 'correct' or no IDs matched)."
                        )
                else:
                    logger.warning(
                        f"Validator results file not found at {validator_results_path}. Skipping removal of bad samples."
                    )

            except FileNotFoundError:
                logger.error(
                    f"Validator script '{validator_script_path}' not found. Ensure it is in the correct directory. Skipping cleaning step."
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Validator script failed with exit code {e.returncode}. Skipping cleaning step."
                )
                logger.error("Validator stdout snapshot:")
                stdout_snapshot = e.stdout.splitlines()
                for i, line_e in enumerate(stdout_snapshot):
                    if i < 50:  # Log first 50 lines of stdout
                        logger.error(f"VALIDATOR_STDOUT_ERR: {line_e}")
                    elif i == 50:
                        logger.error(
                            f"VALIDATOR_STDOUT_ERR: ... (stdout truncated after 50 lines)"
                        )
                        break
                logger.error("Validator stderr snapshot:")
                stderr_snapshot = e.stderr.splitlines()
                for i, line_e in enumerate(stderr_snapshot):
                    if i < 50:  # Log first 50 lines of stderr
                        logger.error(f"VALIDATOR_STDERR_ERR: {line_e}")
                    elif i == 50:
                        logger.error(
                            f"VALIDATOR_STDERR_ERR: ... (stderr truncated after 50 lines)"
                        )
                        break
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during PROD_RUN validation or cleaning: {e}",
                    exc_info=True,
                )
    elif PROD_RUN and (
        samples_generated_successfully == 0
        or not output_file
        or not os.path.exists(output_file)
    ):
        logger.info(
            "PROD_RUN was True, but no samples were successfully generated, output file is missing, or path is invalid. Skipping validation and cleaning."
        )

    # --- Final Cost Calculation and Logging --- ADD THIS SECTION ---
    gen_prompt_tokens, gen_completion_tokens, gen_api_calls = (
        generation_token_tracker.get_summary()
    )
    estimated_generation_cost = generation_token_tracker.calculate_cost(
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS,
    )

    logger.info(f"--- Generation Token Usage & Estimated Cost ---")
    logger.info(f"Total API calls (generation): {gen_api_calls}")
    logger.info(f"Total Prompt Tokens (generation): {gen_prompt_tokens}")
    logger.info(f"Total Completion Tokens (generation): {gen_completion_tokens}")
    logger.info(
        f"Estimated Cost (generation only): ${estimated_generation_cost:.4f} (using placeholder rates)"
    )
    logger.info(
        f"Note: Costs are estimates. Actual costs depend on specific models and OpenRouter pricing."
    )

    # --- Calculate and Log Total Run Cost via Usage Difference ---
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
                logger.info(f"--- Total Run Cost (from Usage Difference) ---")
                logger.info(f"Initial Usage: ${initial_account_usage:.4f}")
                logger.info(f"Final Usage:   ${final_account_usage:.4f}")
                logger.info(
                    f"TOTAL RUN COST (Generation + Validation): ${total_run_cost_by_difference:.4f}"
                )
            else:
                logger.warning(
                    "Cannot calculate total run cost by difference: Initial account usage was not fetched."
                )
        else:
            logger.warning(
                "Could not fetch final OpenRouter account usage. Cannot calculate total run cost by difference."
            )
    else:
        logger.warning(
            "Skipping final OpenRouter account usage check for run cost: Client not initialized or API key missing/placeholder."
        )
    # --- END ADDED SECTION ---

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
