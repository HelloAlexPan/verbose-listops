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
import shutil # Add shutil for rmtree
import threading # Added for RateLimiter
import requests # Added for RateLimiter
import subprocess # Added for PROD_RUN
import sys # Added for PROD_RUN

from dotenv import load_dotenv

load_dotenv()

# --- Batch Settings ---
NUM_SAMPLES_TO_GENERATE = 1100  # How many samples to generate in one run
DEFAULT_MAX_WORKERS = 900   # Number of parallel threads for batch generation
MODEL = "google/gemini-2.5-pro-preview-03-25"  # OpenRouter model
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

    # --- 2. Narrative Context Generation & Style ---
    USE_NARRATIVE_ANCHORS: bool = True              # Conceptual placeholders for intermediate results
    USE_LLM_NAMING: bool = True                     # Use LLM for creative anchor names
    MIN_WORLD_CHARS: int = 3                        # Min chars for randomized world gen
    MAX_WORLD_CHARS: int = 6                        # Max chars for randomized world gen
    MIN_WORLD_CONCEPTS: int = 5                     # Min concepts for randomized world gen
    MAX_WORLD_CONCEPTS: int = 10                    # Max concepts for randomized world gen
    BEAT_CONTEXT: int = 1000                        # Max previous scene chars for beat gen prompt
    PADDING_CONTEXT: int = 1500                     # Tokens of context for padding
    MAX_PAD_PARAGRAPHS: int = 20                    # Max padding segments per-beat

    # --- 3. Temperature ---
    WORLD_GEN_TEMP:  float = 0.9                    # Temp. for world gen
    BEAT_GEN_TEMP: float = 0.1                      # Temp. for generating narrative beats
    CREATIVE_NARRATIVE_TEMP: float = 0.5            # Temp. for creative parts (intro, padding)
    ANCHOR_GEN_TEMP: float = 0.75                   # Temp. for narrative anchor generation

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

    # Production Mode
    # PROD_RUN: bool = False # Moved to Batch Settings

config = Config()


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
    "final"
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
            shutil.rmtree(LOG_DIR) # Remove the entire logs directory
            print(f"Removed existing log directory: {LOG_DIR}")
        except OSError as e:
            # Use logger if available, otherwise print
            if 'logger' in globals() and logger:
                 logger.error(f"Error removing log directory {LOG_DIR}: {e}")
            else:
                print(f"Error removing log directory {LOG_DIR}: {e}")
    try:
        os.makedirs(LOG_DIR, exist_ok=True) # Recreate the logs directory
        # print(f"Ensured log directory exists: {LOG_DIR}") # Optional: for very early debugging before logger is set up
    except OSError as e:
        # Use logger if available, otherwise print
        if 'logger' in globals() and logger: # Check if logger is initialized
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

print(f"Logger initialized with {len(logger.handlers)} handlers. Log file will be created at: {os.path.join(LOG_DIR, 'verbose_listops.log')}")

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
    "\n\n---\n\n**Question:** Following the entire sequence of events described in the story, exactly how many $primary_object did the characters end up with? Provide only the final integer count."
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
    max_pad_paragraphs: int  # No default value, must be set explicitly during instantiation
    # Add tracking for padding token statistics
    padding_stats: dict = field(default_factory=lambda: {
        "total_padding_tokens": 0,
        "padding_segments_added": 0,
        "max_padding_allowed": 0,  # Will be calculated during initialization
    })


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

print(f"Logger initialized with {len(logger.handlers)} handlers. Log file will be created at: {os.path.join(LOG_DIR, 'verbose_listops.log')}")

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
    def __init__(self, max_requests_per_second: float = 40.0,
                min_interval: float = 0.05,
                bucket_capacity: int = 5,
                jitter: float = 0.1):
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
        logger.info(f"Rate limiter initialized: {max_requests_per_second} req/s, "
                f"{min_interval}s min interval, bucket capacity {bucket_capacity}, jitter {jitter}")

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
        """
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
            logger.warning("Cannot check OpenRouter limits: No valid API key")
            return

        try:
            logger.info("Checking OpenRouter rate limits and remaining credits...")
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10
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
                            new_rate = min(float(rps) * 0.8, config.MAX_REQUESTS_PER_SECOND)
                            
                            if new_rate != self.max_requests_per_second:
                                # logger.info(f"Adjusting rate limiter based on OpenRouter limit: {rps} req/s → {new_rate} req/s (80% safety, capped at config.MAX_REQUESTS_PER_SECOND)")
                                self.max_requests_per_second = new_rate
                                limit_adjusted = True
                                current_rate_for_log = new_rate # Update for the concise log

                # Log credits/usage if available
                usage = account_data.get("usage")
                limit = account_data.get("limit")
                limit_remaining = account_data.get("limit_remaining")
                
                log_message_parts = [f"OR Limits: Current RPS: {current_rate_for_log:.1f}"]
                if limit_adjusted:
                    log_message_parts.append("(Adjusted)")

                if usage is not None:
                    log_message_parts.append(f"Usage: ${usage:.4f}")

                if limit is not None and limit_remaining is not None:
                    log_message_parts.append(f"Credits: Rem ${limit_remaining:.4f} of ${limit:.4f}")

                    # If very low on remaining limit, be more conservative with request rate
                    if limit and limit_remaining and limit_remaining / limit < 0.2:
                        # logger.warning(f"Low limit remaining ({limit_remaining}/{limit}). Reducing request rate.")
                        old_self_rate = self.max_requests_per_second
                        self.max_requests_per_second = min(self.max_requests_per_second, 10.0)
                        if old_self_rate != self.max_requests_per_second:
                             log_message_parts.append(f"LOW CREDITS - RPS reduced to {self.max_requests_per_second:.1f}!")


                logger.info(", ".join(log_message_parts))
                self.last_limits_check_time = time.time()
            else:
                logger.warning(f"Failed to get OpenRouter account status: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error checking OpenRouter limits: {e}")

        # Wait at least 5 minutes before checking again
        self.last_limits_check_time = time.time()

# Create a singleton rate limiter instance
rate_limiter = RateLimiter(
    max_requests_per_second=config.MAX_REQUESTS_PER_SECOND,
    min_interval=config.MIN_REQUEST_INTERVAL,
    bucket_capacity=100, # Allow bursts of up to 100 requests
    jitter=0.01  # Add up to 10ms of random jitter to prevent synchronization
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
    """
    if p_inflect is None:
        return str(n)
    try:
        return p_inflect.number_to_words(n)
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

            if hasattr(e, 'status_code') and getattr(e, 'status_code') == 429:
                is_rate_limit_error = True
            elif hasattr(e, 'http_status') and getattr(e, 'http_status') == 429:
                is_rate_limit_error = True
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                is_rate_limit_error = True
            elif any(phrase in error_str for phrase in [
                'rate limit', 
                'too many requests', 
                'ratelimit', 
                'quota exceeded',
                'usage limit',
                'capacity',
                'throttled'
            ]):
                is_rate_limit_error = True

            if is_rate_limit_error:
                rate_limited_delay = delay * 3
                logger.warning(
                    f"Rate limit error detected calling {getattr(func, '__name__', repr(func))} "
                    f"(attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}): {e}"
                )
                logger.info(f"Rate limiting triggered - backing off for {rate_limited_delay:.2f}s")
                try:
                    if hasattr(rate_limiter, 'max_requests_per_second'):
                        old_rate = rate_limiter.max_requests_per_second
                        new_rate = max(1.0, old_rate * 0.8)
                        rate_limiter.max_requests_per_second = new_rate
                        logger.info(f"Reducing rate limit: {old_rate:.1f} → {new_rate:.1f} req/s")
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
            
            wait_time += random.uniform(0, 0.5) # Add jitter
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
    reasoning_settings: dict = None, # Added reasoning_settings parameter
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
                api_params["reasoning"] = reasoning_settings.copy() # Pass a copy
                logger.debug(f"generate_with_retry: Passing reasoning_settings to _chat_completion_call: {api_params['reasoning']}")
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
                        files = [f for f in os.listdir(failed_validations_dir) if f.startswith("validation_fail_")]
                        if files:
                            files.sort(reverse=True)  # Most recent first
                            latest_file = os.path.join(failed_validations_dir, files[0])
                            with open(latest_file, 'r', encoding='utf-8') as f:
                                failure_data = json.load(f)
                                reason = failure_data.get("validation_report", {}).get("reason", "Unknown")
                                validation_failure_reasons.append(f"Validation failed: {reason}")
                                
                                # Add more detailed debugging info
                                logger.warning(f"Validation failed on attempt {attempt}: {reason}")
                                logger.warning(f"Found numbers: {failure_data.get('validation_report', {}).get('found_numbers', [])}")
                                logger.warning(f"Required numbers: {failure_data.get('validation_report', {}).get('allowed_atoms', [])}")
                                logger.warning(f"Missing required: {failure_data.get('validation_report', {}).get('missing_required', [])}")
                                logger.warning(f"Forbidden extras: {failure_data.get('validation_report', {}).get('forbidden_extras', [])}")
                    except Exception as e:
                        logger.error(f"Error reading validation failure data: {e}")
                        validation_failure_reasons.append(f"Validation failed (error reading details)")
                else:
                    validation_failure_reasons.append("Validation failed (no details available)")

        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")
            validation_failure_reasons.append(f"Exception: {str(e)}")

        if attempt < retries:
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    if validation_failure_reasons:
        logger.warning(f"generate_with_retry failed after {retries} attempts. Failure reasons: {validation_failure_reasons}")
    else:
        logger.warning(f"generate_with_retry failed after {retries} attempts with no specific reasons recorded.")
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
                        if (
                            config.MIN_ATOM_VAL
                            <= new_value_sub
                            <= config.MAX_ATOM_VAL
                        ):
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
    if isinstance(root, Atom) and max_ops >= 1:
        op = random.choice(ops)
        arity = random.randint(config.MIN_ARITY, max_branch)
        children = [
            Atom(random.randint(config.MIN_ATOM_VAL, config.MAX_ATOM_VAL))
            for _ in range(arity - 1)
        ]
        children.append(root)
        random.shuffle(children)
        root = OpNode(op, children)
    return root


def validate_ast(node: Node):
    """Recursively validate that all operators in the AST are supported."""
    if node.op not in OP_LABELS and not isinstance(node, Atom):
        raise ValueError(f"Invalid operator: {node.op}")
    for c in node.children:
        validate_ast(c)


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
            logger.error(f"eval_node: Cannot calculate average of zero values for node {node.op}.")
            raise ValueError("Cannot calculate average of zero values.")

        calculated_value = func(vals)
        
        # Additional validation for SUM operations to catch calculation errors
        if node.op == "SUM":
            expected_sum = sum(vals)
            if calculated_value != expected_sum:
                logger.error(f"eval_node: SUM validation error - func(vals)={calculated_value} != sum(vals)={expected_sum}")
                # Use the manually calculated sum as a fallback
                calculated_value = expected_sum
                
        # Enhanced logging for all operations    
        logger.debug(f"eval_node: OpNode {node.op}, inputs {vals}, result = {calculated_value}")
        
        # Operation-specific detailed logging
        if node.op == "SUM":
            logger.info(f"SUM Operation - Node ID: {id(node)}, Input values: {vals}, Sum: {calculated_value}")
        elif node.op == "AVG":
            total = sum(vals)
            logger.info(f"AVG Operation - Node ID: {id(node)}, Input values: {vals}, Sum: {total}, Count: {len(vals)}, Result: {calculated_value}")
        elif node.op in ["MAX", "MIN", "MED"]:
            logger.info(f"{node.op} Operation - Node ID: {id(node)}, Input values: {vals}, Result: {calculated_value}")
        
        node.value = calculated_value
        return node.value
    except KeyError:
        logger.error(f"eval_node: Unsupported operator: {node.op}")
        raise ValueError(f"Unsupported operator: {node.op}")
    except IndexError as e:
        logger.error(f"eval_node: Indexing error evaluating {node.op} with child values {vals}: {e}")
        raise
    except ZeroDivisionError:
        logger.error(f"eval_node: Division by zero during AVG for {node.op} with child values {vals}")
        raise ValueError(f"Division by zero during AVG for {node.op}")
    except Exception as e:
        logger.error(f"eval_node: Unexpected error evaluating {node.op} with values {vals}: {e}")
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
        "model", "messages", "max_tokens", "temperature", "top_p", "n",
        "stream", "stop", "presence_penalty", "frequency_penalty",
        "logit_bias", "user", "top_k"
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
        elif k == "reasoning": # Explicitly handle reasoning for extra_body
            openrouter_specific_params[k] = v
        # else: # You could log other unexpected kwargs if needed
            # logger.warning(f"DEBUG: Unexpected kwarg '{k}' in _chat_completion_call, may be ignored or cause error if not for extra_body.")

    # --- Check if we need to modify the max_tokens to prevent truncation ---
    if "max_tokens" in api_call_standard_kwargs:
        original_max_tokens = api_call_standard_kwargs["max_tokens"]
        # Always use the higher limit to prevent truncation due to reasoning tokens
        api_call_standard_kwargs["max_tokens"] = config.MAX_API_TOKEN_LIMIT
        logger.debug(f"Modified max_tokens for API call: {original_max_tokens} → {config.MAX_API_TOKEN_LIMIT} (to handle reasoning tokens)")
        
    # --- REFINED REASONING LOGIC for openrouter_specific_params ---
    current_model_name = api_call_standard_kwargs.get("model", "").lower()
    is_openai_o_series = ("openai/" in current_model_name and
                          (re.search(r"/o\d+", current_model_name) or "gpt-4o-mini" in current_model_name))

    reasoning_config_to_send = openrouter_specific_params.get("reasoning", {})
    if reasoning_config_to_send is None: # Handle if None was explicitly passed
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
            logger.debug(f"DEBUG: Keeping 'effort' for OpenAI o-series model ({current_model_name}).")
    
    # Update openrouter_specific_params with the processed reasoning_config
    if reasoning_config_to_send:
        openrouter_specific_params["reasoning"] = reasoning_config_to_send
    elif "reasoning" in openrouter_specific_params: # If it was there but became empty
        del openrouter_specific_params["reasoning"]

    logger.debug(f"Final standard API call_kwargs: {json.dumps(api_call_standard_kwargs, indent=2)}")
    logger.debug(f"Final OpenRouter specific_params (for extra_body): {json.dumps(openrouter_specific_params, indent=2)}")

    max_tokens_value = api_call_standard_kwargs.get('max_tokens', 'NOT SET')
    if max_tokens_value == 'NOT SET':
        logger.warning(f"DEBUG: max_tokens value NOT SET for API call. API will use its default.")
    elif isinstance(max_tokens_value, int) and max_tokens_value <= 0:
        logger.error(f"DEBUG: max_tokens value is invalid ({max_tokens_value}). API call will likely fail.")
    else:
        logger.info(f"DEBUG: FINAL max_tokens value being sent to API: {max_tokens_value}")

    try:
        wait_time = rate_limiter.wait_if_needed()
        if wait_time > 0:
            logger.debug(f"Rate limit applied - waited {wait_time:.2f}s before API call")

        # Use extra_body for OpenRouter-specific parameters
        if openrouter_specific_params:
            return client.chat.completions.create(**api_call_standard_kwargs, extra_body=openrouter_specific_params)
        else:
            return client.chat.completions.create(**api_call_standard_kwargs)

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
                reasoning={"exclude": True}
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

MAX_WORDS_FOR_NUMBER_DICT = 5000 # Define a larger limit for number-to-word conversion

def _build_expanded_number_words_dict(
    max_val: int = MAX_WORDS_FOR_NUMBER_DICT, # Use the new larger limit
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
            word = num_to_words(i)
            num_word_dict[word.lower()] = i
        except Exception as e:
            logger.warning(f"Inflect failed to convert {i} to words: {e}")
    logger.info(
        f"Built expanded number words dictionary with {len(num_word_dict)} entries (up to {max_val})."
    )
    return num_word_dict


# EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict()
# --- Build dict after logger is fully configured ---
EXPANDED_NUMBER_WORDS_DICT = {}
if __name__ == "__main__" or "pytest" in str(os.environ.get("_", "")): #Ensure this runs for main script and tests
    if p_inflect: # Check if p_inflect was initialized
        EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict()
    else:
        logger.error("p_inflect is None, EXPANDED_NUMBER_WORDS_DICT will be empty. Number word extraction will be limited.")


# --- Sort keys by length descending to prioritize longer matches ---
sorted_number_words = sorted(EXPANDED_NUMBER_WORDS_DICT.keys(), key=len, reverse=True)
NUMBER_WORDS_PATTERN = (
    r"\b(?:(minus|negative)\s+)?("
    + "|".join(re.escape(k) for k in sorted_number_words)
    + r")\b"
)
NUMBER_WORDS_REGEX = re.compile(NUMBER_WORDS_PATTERN, re.IGNORECASE)


def extract_numbers_from_text(text: str) -> Set[int]:
    """Extracts integers (digits and words), ignoring specified ordinals."""
    if not text:
        return set()

    found_numbers = set()

    for match in DIGIT_REGEX.finditer(text):
        digit_str = match.group(0)
        try:
            value = int(digit_str)
            found_numbers.add(value)
        except ValueError:
            continue

    for match in NUMBER_WORDS_REGEX.finditer(text):
        sign_word = match.group(1)
        number_word = match.group(2).lower()

        if number_word in ORDINAL_WORDS_TO_IGNORE:
            continue

        value = EXPANDED_NUMBER_WORDS_DICT.get(number_word)

        if value is not None:
            if sign_word and value != 0:
                value = -value
            found_numbers.add(value)
        else:
            logger.warning(f"    Word '{number_word}' found by regex but not in dict.")

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

    IMPLICITLY_ALLOWED_SMALL_NUMBERS = set(range(config.MIN_ALLOWED_SMALL_NUMBER, config.MAX_ALLOWED_SMALL_NUMBER + 1))

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
            
        logger.debug(
            f'Validator Input Text: "' + text_preview + '"'
        )
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
            "details": []
        }

        if strict_zero:
            # Check if numbers were found
            if found_numbers:
                # If numbers are found, check if it's ONLY the number 1
                if found_numbers == {1}:
                    logger.debug(f"Validation PASS (Strict Zero context, but only '1' found, which is tolerated for padding/intro). Found: {found_numbers}")
                    # Even though it's a "pass" for this specific rule,
                    # we don't return True yet, as other rules might apply if strict_zero was False.
                    # However, for padding/intro, strict_zero IS True, and this is the only number check we care about.
                    # So, if we reach here, it means for padding/intro, finding only "1" is acceptable.
                    # The rest of the validator logic for non-strict_zero (operand checks etc.) is not relevant here.
                    # BUT, we need to ensure this doesn't bypass other checks if strict_zero was meant to be part of a larger validation.
                    # Given the current structure, if strict_zero is true, this is the primary gate.
                    # Let's refine: if strict_zero is true, we ONLY care if non-{1} numbers are present.
                else:
                    # Numbers other than just {1} were found, or a mix including 1. This is a failure.
                    validation_report["status"] = "FAIL"
                    validation_report["reason"] = "STRICT_ZERO_VIOLATION"
                    validation_report["details"].append(f"Found numbers {found_numbers} when only '1' (or zero numbers) were allowed in this strict context.")
                    log_reason = f"Validation FAIL (Strict Zero context): Found numbers {found_numbers}. Expected zero numbers or only '1'."
                    logger.debug(log_reason)
                    _log_failed_validation(text, validation_report)
                    return False # Fail because numbers other than just '1' were found
            # If found_numbers is empty, it's a pass for strict_zero
            logger.debug(f"Validation PASS (Strict Zero context: No numbers found).")
            # If we are in strict_zero mode and passed (either empty or just {1}),
            # for padding/intro, this is the end of validation.
            # We need to ensure this doesn't incorrectly proceed to other checks.
            # The original logic for strict_zero was to return True if no numbers.
            # Now, it should return True if no numbers OR only {1}.

            # Corrected logic for strict_zero:
            if not found_numbers or found_numbers == {1}:
                logger.debug(f"Validation PASS (Strict Zero context): Found numbers: {found_numbers} (empty or only '1' is acceptable).")
                return True # This is the definitive pass for strict_zero sections (padding/intro)
            else:
                # This 'else' handles cases where found_numbers is not empty AND not equal to {1}
                validation_report["status"] = "FAIL"
                validation_report["reason"] = "STRICT_ZERO_VIOLATION"
                validation_report["details"].append(f"Found numbers {found_numbers} when only '1' (or zero numbers) were allowed in this strict context.")
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
                validation_report["details"].append(f"Result {correct_result_for_beat} must be present. Found: {found_numbers}, Op: {operation_type}")
                
                log_reason = f"Validation FAIL (Missing Required Result): Result {correct_result_for_beat} must be present. Found: {found_numbers}, Op: {operation_type}"
                logger.debug(log_reason)
                
                # Log the failed attempt with error code
                _log_failed_validation(text, validation_report)
                return False
            logger.debug(f"Validation INFO: Required result {correct_result_for_beat} is present for {operation_type if operation_type else 'unspecified'} operation")

        missing_expected = allowed_atoms - found_numbers
        if missing_expected:
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "MISSING_REQUIRED_OPERANDS"
            validation_report["missing_required"] = list(missing_expected)
            validation_report["details"].append(f"RequiredOperands={allowed_atoms}, Missing={missing_expected}, FoundInText={found_numbers}.")
            
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
                logger.debug(f"Validator: Allowing {extra_num} as it's 1, 2, or 3, considered general phrasing. Skipping further checks for this number as an 'extra'.")
                continue 
            # --- END MODIFICATION ---

            is_allowed_count = (
                extra_num == operand_count and extra_num not in forbidden_atoms
            )
            is_allowed_small = (
                extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS
                and extra_num not in forbidden_atoms
            )

            if not (is_allowed_count or is_allowed_small): # Adjusted condition
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
                elif extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS and extra_num in forbidden_atoms:
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
                    fail_reason_detail.append("unexpected extraneous number not fitting any explicit or implicit allowance")

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
            "timestamp": timestamp
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
                if 'context' in frame.frame.f_locals:
                    ctx = frame.frame.f_locals['context']
                    if hasattr(ctx, 'sample_index'):
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
            if 'narrative_anchor' in frame_locals:
                narrative_anchor = frame_locals['narrative_anchor']
            if 'context' in frame_locals:
                ctx = frame_locals['context']
                if hasattr(ctx, 'beat_counter'):
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
            validation_details += f"Correct result (should be mentioned): {correct_result}\n"
        
        if intermediate_sum is not None:
            validation_details += f"Intermediate sum (may be mentioned for AVG/SM): {intermediate_sum}\n"
            
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
        log_prompt(
            validation_header,
            validation_details,
            sample_index=sample_index
        )
        
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
    system_prompt = """You are a master storyteller and creative naming expert. Your task is to generate a short, evocative, and thematic 'narrative anchor'.

A narrative anchor is a creative, conceptual name that serves as a descriptive **label** or **stand-in** for the *result* (the outcome) of a specific event or calculation within the story. Its purpose is to allow the narrative to refer to this result conceptually in later parts of the story, *without* explicitly stating its numerical value. For example, if a calculation's outcome is 50, the anchor might be 'The Sunstone's Core.' The story would then mention 'The Sunstone's Core' (which implicitly represents the value 50) instead of the number itself, allowing the narrative to flow without revealing intermediate figures.

Key Guidelines for the Narrative Anchor:
1.  **Thematic:** The name MUST fit the provided Genre, Setting, and Primary Object.
2.  **Concise:** Aim for 2-5 words. Often a noun phrase (e.g., 'The Sunstone's Core,' 'The Oracle's Key').
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
        f"Genre: {genre}\\n"
        f"Setting: {setting}\\n"
        f"Item: {primary_object}\\n"
        f"Concept/Operation Hint: {concept_keywords_for_prompt}\\n"
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
            "reasoning": {"exclude": True}
        }
        
        logger.debug(f"Using request_payload for narrative anchor generation: {request_payload}")
        
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
            rf"Genre: {re.escape(genre)}\s*\\n"
            rf"Setting: {re.escape(setting)}\s*\\n"
            rf"Item: {re.escape(primary_object)}\s*\\n"
            rf"Concept/Operation Hint: {re.escape(concept_keywords_for_prompt)}\s*\\n"
        )
        # Remove preamble if found at the beginning of the candidate string
        candidate = re.sub(
            f"^{preamble_pattern_str}", "", candidate, flags=re.IGNORECASE
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


# --- Narrative Generation with Strict Checks ---
class BeatGenerationError(Exception):
    """Raised when a story beat fails to generate, aborting entire narrative."""

    pass


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
    logger.debug(
        f"_generate_narrative_recursive (POST-ORDER): processing node {getattr(node, 'op', 'Atom')} with narrative anchor '{narrative_anchor}'"
    )

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

    forbidden_atoms_for_prompt = context.introduced_atoms.copy()
    truly_forbidden_for_prompt = forbidden_atoms_for_prompt - required_atoms_for_beat
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

    # Add the calculated result of the current operation to this set.
    # The validator (make_number_validator) is already configured to expect this result
    # when enforce_result_presence is True. This change ensures the LLM is also told to include it.
    if correct_result is not None: # Should always be defined and not None at this stage
        numbers_to_mention_in_prompt.add(correct_result)
    else:
        logger.warning(f"DEV WARNING: correct_result is None for node {node.op} ({narrative_anchor}) when building must_include_list. This is unexpected.")

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

    if truly_forbidden_for_prompt:
        must_avoid_str = ", ".join(num_to_words(x) for x in sorted(truly_forbidden_for_prompt))
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
        formatted_direct_values_list = [num_to_words(v) for v in sorted(direct_atom_values)]
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
    if node.op == "SUM":
        correct_result_words = num_to_words(correct_result)
        action_description = (
            f"Narrate an action (e.g., gathering, merging) involving {inputs_str}. "
            f"The outcome MUST be that the total quantity becomes exactly **{correct_result_words}** {primary_object}. Imply the sum through action."
        )
    elif node.op == "AVG":
        correct_result_words = num_to_words(correct_result)
        direct_atom_sum_words = num_to_words(direct_atom_sum) if direct_atom_sum is not None else "calculated sum"
        action_description = (
            f"Narrate an event (e.g., balancing, averaging mechanism) involving {inputs_str}. "
            f"The outcome MUST be exactly **{correct_result_words}** {primary_object} (the floored average). "
            f"You MAY mention the intermediate sum ({direct_atom_sum_words} from direct inputs ({formatted_direct_values_str}) if needed for the narrative, but the final result is key."
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
            f"Then, describe a specific, plausible event that **forces them to keep only a quantity equal to the final digit of that total**. "
            f"Examples: \n"
            f"*   A magical lock clicks open, consuming all but the final unit of energy ({correct_result_words}).\\n"
            f"*   A mystical tax collector appears, taking all but the last {correct_result_words} {primary_object}.\\n"
            f"*   The combined items react, leaving only {correct_result_words} stable {primary_object}.\\n"
            f"The final quantity MUST become exactly **{correct_result_words}** {primary_object}. Do NOT explicitly state 'sum' or 'modulo'; the *event* causes the result."
        )
    elif node.op == "MAX":
        correct_result_words = num_to_words(correct_result)
        action_description = (
            f"Narrate comparing {inputs_str}. They MUST choose the item/quantity with the largest value. "
            f"The outcome MUST be exactly **{correct_result_words}** {primary_object}. Justify the choice."
        )
    elif node.op == "MIN":
        correct_result_words = num_to_words(correct_result)
        action_description = (
            f"Narrate comparing {inputs_str}. They MUST choose the item/quantity with the smallest value. "
            f"The outcome MUST be exactly **{correct_result_words}** {primary_object}. Justify the choice."
        )
    elif node.op == "MED":
        correct_result_words = num_to_words(correct_result)
        action_description = (
            f"Narrate evaluating {inputs_str} numerically. They MUST select the item/quantity with the middle value (when sorted). "
            f"The outcome MUST be exactly **{correct_result_words}** {primary_object}. Justify the choice."
        )
    else:
        correct_result_words = num_to_words(correct_result)
        action_description = f"Narrate applying '{op_label}' to {inputs_str}. Outcome must be {correct_result_words}."

    reminder = ""
    if child_narrative_anchors:
        reminder_names_str = ", ".join(f"'{name}'" for name in child_narrative_anchors)
        must_mention_str = (
            formatted_direct_values_str
            if has_direct_atom_children
            else "(none for this step, as it only involves prior conceptual results)"
        )
        reminder = (
            f"\n**REMINDER:** Do NOT mention the actual numeric results associated with previous conceptual steps ({reminder_names_str}) in your text. "
            f"Refer to them by their conceptual names (e.g., '{random.choice(child_narrative_anchors) if child_narrative_anchors else "a previous finding"}') ONLY. "
            f"This is crucial for maintaining narrative suspense and focusing the reader on the current step's numbers. "
            f"However, you MUST explicitly mention the newly discovered quantities for *this* step: {must_mention_str} (as words)."
        )

    operational_instruction = (
        f"This scene resolves the step named '{narrative_anchor}'.\n"
        f"{action_description}\n"
        f"{reminder}"
    )

    few_shot_section = ""
    num_shots = config.FEW_SHOT_EXAMPLES

    if num_shots == 1 and FEW_SHOT_EXAMPLES_STRICT:
        rules_str, good_narrative, _, _ = FEW_SHOT_EXAMPLES_STRICT[0]
        example_prompt_text = (
            f"--- Example of Strict Narrative Generation ---\n"
            f"Rules:\n{rules_str}\n"
            f"Good Narrative Output (Follows Rules):\n{good_narrative}\n"
            f"--- End Example ---"
        )
        few_shot_section = (
            "Here is an example of generating a narrative scene while strictly following number rules:\n\n"
            + example_prompt_text  # Renamed to avoid conflict with function name
            + "\n\n---\n\n"
        )
    elif num_shots > 0 and FEW_SHOT_EXAMPLES_STRICT:
        logger.warning(
            f"Configured for {num_shots} few-shot examples, but FEW_SHOT_EXAMPLES_STRICT list only has {len(FEW_SHOT_EXAMPLES_STRICT)} example(s) after modification. Using the first one."
        )
        rules_str, good_narrative, _, _ = FEW_SHOT_EXAMPLES_STRICT[0]
        example_prompt_text = (
            f"--- Example of Strict Narrative Generation ---\n"
            f"Rules:\n{rules_str}\n"
            f"Good Narrative Output (Follows Rules):\n{good_narrative}\n"
            f"--- End Example ---"
        )
        few_shot_section = (
            "Here is an example of generating a narrative scene while strictly following number rules:\n\n"
            + example_prompt_text
            + "\n\n---\n\n"
        )

    system_prompt = (
        "You are a fiction writer creating sequential story scenes. Your ONLY task is to write the *next* narrative scene text based on the user's instructions and a snippet of the previous scene. "
        "ABSOLUTELY FORBIDDEN: Any text other than the story scene. NO analysis, NO checklists, NO rule explanations, NO calculations, NO meta-commentary, NO greetings. "
        "Study the few shot examples of what to do. Adhere STRICTLY to ALL instructions, especially the number rules for the current scene."
    )

    cleaned_snippet = clean_snippet(context.last_scene_text, max_len=150)

    user_message_content = (
        f"**CONTEXT:**\\n"
        f"Genre: {world.get('genre', 'fantasy')}\\n"
        f"Setting: '{world.get('setting', 'a fictional world')}'\\n"
        f"Previous Scene Snippet: '...{cleaned_snippet}'\\n\\n"
    )

    # --- Few-Shot Example Section ---
    few_shot_section = ""
    num_shots = config.FEW_SHOT_EXAMPLES
    if num_shots == 1 and FEW_SHOT_EXAMPLES_STRICT:
        rules_str, good_narrative, _, _ = FEW_SHOT_EXAMPLES_STRICT[0]
        example_prompt_text = (
            f"Rules:\\n{rules_str}\\n"
            f"Good Narrative Output (Follows Rules):\\n{good_narrative}\\n"
        )
        few_shot_section = (
            f"**EXAMPLE (How to Follow Rules):**\\n"  # Clearer heading
            f"{example_prompt_text}"
            f"---\\n\\n"
        )
    # Add elif/else for more examples if FEW_SHOT_EXAMPLES_STRICT is expanded later
    elif num_shots > 0 and FEW_SHOT_EXAMPLES_STRICT:
        logger.warning(
            f"Configured for {num_shots} few-shot examples, but FEW_SHOT_EXAMPLES_STRICT list only has {len(FEW_SHOT_EXAMPLES_STRICT)} example(s). Using the first one."
        )
        rules_str, good_narrative, _, _ = FEW_SHOT_EXAMPLES_STRICT[0]
        example_prompt_text = (
            f"Rules:\\n{rules_str}\\n"
            f"Good Narrative Output (Follows Rules):\\n{good_narrative}\\n"
        )
        few_shot_section = (
            f"**EXAMPLE (How to Follow Rules):**\\n"
            f"{example_prompt_text}"
            f"---\\n\\n"
        )

    if few_shot_section:
        user_message_content += few_shot_section

    # --- Combined Task & Scene Requirements ---
    task_header = "Reaching the conclusion" if is_root else "Continuing on the journey"
    current_scene_instructions = (
        f"**YOUR TASK: Write ONLY the narrative text for the next scene.**\\n"
    )
    current_scene_instructions += (
        f"Continue directly from the snippet: '...{cleaned_snippet}'\\n\\n"
    )

    current_scene_instructions += (
        f"**SCENE REQUIREMENTS ({task_header} for '{narrative_anchor}'):**\\n"
    )
    if scene_preamble:
        current_scene_instructions += f"*   Discovery: {scene_preamble}\\n"
    current_scene_instructions += f"*   Action: {action_description}\\n"
    if reminder:
        current_scene_instructions += (
            f"*   Reminder: {reminder.replace('**REMINDER:**','').strip()}\\n"
        )

    user_message_content += (
        f"{current_scene_instructions}\\n"  # Add the combined requirements
    )

    # --- Strengthened Number Rules Section --- (and explicit formatting instruction)
    user_message_content += (
        f"**!!! CRITICAL NUMBER RULES & FORMATTING !!!**\\n"
        f"Your writing MUST explicitly state the numbers for the current calculation (listed as 'MUST INCLUDE' below) using written word forms (e.g., 'forty-two', 'one hundred and seven').\\n"
        f"For other numbers you are allowed to use (like counts or the number 'one'), also write them out as words (e.g., 'three', 'one').\\n"
        f"Do NOT use digits (like '42' or '107') in your story text for ANY number.\\n"
        f"Do NOT use the 'word (digit)' format like 'seven (7)' in your story text for ANY number.\\n"
        f"--- Current Step's Numbers ---\\n"
        f"- MUST INCLUDE: {must_include_combined_str} (as written words)\\n"
    )
    
    # New explicit construction for "You may optionally use..."
    optional_use_parts_list = []
    # Special instruction if operand_count itself is forbidden
    if operand_count > 0 and operand_count_is_forbidden_for_prompt:
        operand_count_word = num_to_words(operand_count)
        user_message_content += (
            f"- **SPECIAL NARRATIVE CHALLENGE FOR THIS SCENE:** While there are {operand_count_word} new groups of {primary_object} to be described, "
            f"the number '{operand_count_word}' itself was critically important in a previous event and is now TEMPORARILY FORBIDDEN for direct mention as a count. "
            f"Therefore, for THIS SCENE ONLY, you MUST describe the discovery of these {operand_count_word} groups "
            f"WITHOUT using the word '{operand_count_word}' to state their count. "
            f"Instead, use phrasing like 'They found several new caches...' or 'Another set of discoveries was made...' or 'The next locations revealed...' "
            f"and then proceed to list their contents (which are in the 'MUST INCLUDE' list: {must_include_combined_str}). "
            f"This tests your ability to narrate around a specific, temporarily forbidden number while still conveying all necessary new information.\n"
        )
    elif operand_count > 0: # operand_count is not forbidden, so it can be optionally used
        operand_count_word = num_to_words(operand_count)
        # primary_object is defined earlier in this function scope
        optional_use_parts_list.append(
            f"{operand_count_word}, as this is the number of groups of {primary_object}s the characters will find"
        )

    # No longer explicitly mentioning 'one' as optionally usable
    # The numbers 'one', 'two', 'three', as well as common ordinals are now generally allowed for narrative fluency
    # as indicated in the no_other_numbers_instruction

    # --- CORRECTED LOGIC for mentioning intermediate sum in "optional use" ---
    # This `direct_atom_sum` is the sum of direct atomic children of the current node.
    intermediate_sum_mention_in_optional_rules = ""
    if (
        direct_atom_sum is not None
    ):  # If there are direct atoms and their sum is calculated
        if node.op == "AVG":
            direct_atom_sum_words = num_to_words(direct_atom_sum)
            intermediate_sum_mention_in_optional_rules = f"the sum of new items ({direct_atom_sum_words}) which contributes to the average calculation"
        elif node.op == "SM":
            # For SM, direct_atom_sum is the sum of new atomic items for this step.
            # The overall sum for SM (sm_intermediate_sum) is mentioned in the action_description.
            # Allowing direct_atom_sum here is consistent with the validator.
            direct_atom_sum_words = num_to_words(direct_atom_sum)
            intermediate_sum_mention_in_optional_rules = f"the sum of new items ({direct_atom_sum_words}) before they are combined with other values for the final digit operation"
        # Add other conditions here if other ops have a specific intermediate sum from direct atoms that can be mentioned.

    # Build the "You may optionally use" line
    full_optional_statement_parts = []
    if optional_use_parts_list:
        # Join the existing optional parts (like operand count and 'one')
        full_optional_statement_parts.append(" and ".join(optional_use_parts_list))

    if (
        intermediate_sum_mention_in_optional_rules
    ):  # If there's a relevant intermediate sum to mention
        full_optional_statement_parts.append(intermediate_sum_mention_in_optional_rules)

    if full_optional_statement_parts:
        user_message_content += f"- You may optionally use: { ' and '.join(full_optional_statement_parts) }.\n"

    # Now, resume the original f-string concatenation for the remaining rules
    # --- Refined "NO OTHER NUMBERS ALLOWED" instruction with narrative context ---
    # `numbers_to_mention_in_prompt` is the set of current beat's direct operands + its result
    # `truly_forbidden_for_prompt` is (all_previously_introduced_atoms - current_beat_direct_operands)

    primary_object = world.get("object", "items") # Get the primary object name

    # Use dynamic problematic numbers: base set from config + current operand_count if it's forbidden
    dynamic_problematic_numbers_to_check = config.PROBLEM_SMALL_NUMBERS_TO_CHECK.copy()
    if operand_count_is_forbidden_for_prompt: # If the operand count itself is forbidden
        dynamic_problematic_numbers_to_check.add(operand_count)

    special_attention_clauses = []
    
    for num_val in sorted(list(dynamic_problematic_numbers_to_check)):
        # Do not create special warnings for 1, 2, 3 as they are now generally allowed by validator for fluency
        if num_val in [1, 2, 3]:
            continue 

        if (num_val in truly_forbidden_for_prompt) and \
           (num_val not in numbers_to_mention_in_prompt):
            
            num_word = num_to_words(num_val)
            object_form = p_inflect.singular_noun(primary_object) if num_val == 1 and p_inflect else primary_object
            
            is_current_operand_count_clause = ""
            if num_val == operand_count and operand_count_is_forbidden_for_prompt:
                is_current_operand_count_clause = (f" (Note: even though there are {num_word} new groups of {object_form} in *this* scene, "
                                                   f"the word '{num_word}' is forbidden due to its prior significance - see SPECIAL NARRATIVE CHALLENGE above if provided)")

            clause = (
                f"The number **'{num_word}'** (referring to '{num_word} {object_form}' or just the quantity '{num_word}') "
                f"was significant in a prior event. For THIS SCENE, you MUST NOT use the word '{num_word}'{is_current_operand_count_clause}, unless '{num_word}' is explicitly in 'MUST INCLUDE' for this step."
            )
            special_attention_clauses.append(clause)

    # Start with the general rule
    no_other_numbers_instruction = (
        "- **NO OTHER NUMBERS ALLOWED** beyond those in 'MUST INCLUDE' or 'You may optionally use' (if any are listed).\n"
        "  This means you should avoid introducing other numerical quantities (e.g., 'seven items', 'after twelve minutes') unless they are part of the core calculation for this scene.\n"
        "  The numbers 'one', 'two', and 'three', as well as common ordinal numbers (first, second, third, etc.), are generally acceptable for natural narrative flow and are not considered part of this restriction, even if not explicitly listed as 'optional'.\n"
    )

    if special_attention_clauses:
        # If there are specific warnings, add a header and then list them.
        # This makes the prompt structure clearer.
        no_other_numbers_instruction += (
            "  **ADDITIONAL CLARIFICATIONS ON PREVIOUSLY MENTIONED QUANTITIES (VERY IMPORTANT FOR THIS SCENE):**\n"
        )
        for clause_text in special_attention_clauses:
            no_other_numbers_instruction += f"    *   {clause_text}\n"

    user_message_content += no_other_numbers_instruction
    # The "Adherence to these number rules is MANDATORY" line should follow immediately
    user_message_content += (
        f"**Adherence to these number rules is MANDATORY.**\n\n"
    )

    # --- Final Reminder ---
    user_message_content += (
        f"REMEMBER: OUTPUT ONLY THE NARRATIVE TEXT FOR THIS SCENE. NO EXTRA TEXT."
    )

    log_prompt(
        f"{{'=== FINAL' if is_root else '=== Intermediate'}} Operator Beat Prompt (Op: {node.op}, Narrative Anchor: {narrative_anchor})",
        f"System: {system_prompt}\\n---\\nUser:\\n{user_message_content}",
        sample_index=context.sample_index,
    )

    estimated_prompt_tokens = 0
    try:
        estimated_prompt_tokens = len(
            encoder.encode(system_prompt + user_message_content)
        )
    except Exception as e:
        logger.error(f"Error encoding prompt for token estimation: {e}")

    current_max_beat_completion_tokens = (
        config.BEAT_MAX_TOKENS
    )  # Renamed variable

    # Enhanced logging for token budget before API call
    token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
    expected_after_call = context.tokens_used + estimated_prompt_tokens + current_max_beat_completion_tokens
    expected_percentage = (expected_after_call / config.MAX_TOTAL_TOKENS) * 100
    will_exceed = would_exceed_budget(
        context.tokens_used,
        estimated_prompt_tokens + current_max_beat_completion_tokens,
        config.MAX_TOTAL_TOKENS,
        SAFETY_MARGIN,
    )
    
    logger.info(
        f"TOKEN BUDGET PRE-CALL [{node.op}/{narrative_anchor}]: "
        f"Current: {context.tokens_used} ({token_percentage:.1f}%), "
        f"This call: +{estimated_prompt_tokens} prompt, +{current_max_beat_completion_tokens} max completion = "
        f"+{estimated_prompt_tokens + current_max_beat_completion_tokens} total, "
        f"Expected after: {expected_after_call}/{config.MAX_TOTAL_TOKENS} ({expected_percentage:.1f}%), "
        f"Will exceed: {will_exceed}"
    )

    # Original token budget check with enhanced error message
    if would_exceed_budget(
        context.tokens_used,
        current_max_beat_completion_tokens, # Only consider the potential completion against the budget
        config.MAX_TOTAL_TOKENS,
        SAFETY_MARGIN,
    ):
        logger.warning(
            f"❌ TOKEN LIMIT ABORT (Completions Only): Cannot generate beat for {node.op} ({narrative_anchor}). "
            f"Current COMPLETION tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
            f"Required max COMPLETION for this beat: +{current_max_beat_completion_tokens}, "
            f"Total potential COMPLETION tokens: {context.tokens_used + current_max_beat_completion_tokens}, "
            f"Max allowed COMPLETION tokens: {config.MAX_TOTAL_TOKENS - SAFETY_MARGIN} (with margin). "
            f"STOPPING GENERATION."
        )
        raise BeatGenerationError(
            f"Token budget exceeded before generating beat for {node.op}"
        )

    validate_beat_numbers = make_number_validator(
        allowed_atoms=required_atoms_for_beat,
        forbidden_atoms=truly_forbidden_for_prompt,
        operand_count=operand_count,
        correct_result_for_beat=correct_result,
        intermediate_sum_allowed=(
            direct_atom_sum if node.op in ["AVG", "SM"] else None
        ),
        enforce_result_presence=True,
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

    for attempt in range(1, config.MAX_BEAT_RETRIES + 1):
        reason = None
        try:
            resp = _chat_completion_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content},
                ],
                max_completion_tokens=current_max_beat_completion_tokens,
                temperature=config.BEAT_GEN_TEMP,
                reasoning={"exclude": True}
            )

            # Get the raw content from LLM response, without any stripping yet
            truly_raw_llm_content = ""  # Initialize to empty string
            if (
                resp
                and resp.choices
                and len(resp.choices) > 0
                and resp.choices[0].message
            ):
                # Assign content if it exists, otherwise it remains an empty string
                # This stores the actual string from the API, or None if that's what was in message.content
                _content = resp.choices[0].message.content
                truly_raw_llm_content = _content if _content is not None else ""
            else:
                logger.warning(
                    f"Beat {context.beat_counter['current']} attempt {attempt}: Invalid response structure or no message content from API: {resp}"
                )
            # truly_raw_llm_content is now the truly raw string from the API, or an empty string if content was None or structure invalid

            # Log the TRULY raw output to llm_turns.log
            log_prompt(
                f"LLM Beat Generation Attempt {attempt} for operator {node.op} ({narrative_anchor})",
                f"System: {system_prompt}\\nUser: {user_message_content}\\n\\nGeneration (Raw):\\n{truly_raw_llm_content}",  # Logging the untouched content
                sample_index=context.sample_index,
            )

            # --- Aggressive Cleaning Step ---
            # The cleaning process will take this raw content and perform its own stripping and filtering.
            # Start the cleaning process with the truly_raw_llm_content.
            # The variable 'cleaned_candidate_text' will be the result of this cleaning.
            cleaned_candidate_text = (
                truly_raw_llm_content  # Pass raw content to be cleaned
            )
            lines = cleaned_candidate_text.splitlines()
            filtered_lines = []
            prompt_echoing_fragments = (
                "Imply the sum",
                "reference to the previous",
                "Narrate comparing",
                "This scene resolves",
                "The outcome MUST be",
                "Narrate an action",
                "Narrate an event",
                "The final quantity MUST become",
            )
            analysis_patterns_to_remove = (
                [  # This list was inside the 'if raw_candidate_text:' block
                    r"^\s*Critique:.*$",
                    r"^\s*Checklist:.*$",
                    r"^\s*Analysis:.*$",
                    r"^\s*Rules check:.*$",
                    r"^\s*MUST INCLUDE:.*$",
                    r"^\s*MUST AVOID:.*$",
                    r"^\s*Reasoning:.*$",
                    r"^\s*Validation:.*$",
                    r"^\s*Following the rules:.*$",
                    r"^\s*Based on the instructions:.*$",
                    r"^\s*The task is to.*$",
                    r"^\s*The prompt asks.*$",
                    r"^\s*Scene instructions:.*$",
                    r"^\s*Instructions:.*$",
                    r"^\s*Number rules:.*$",
                    r"^\s*Rules:.*$",
                    r"^\s*Confidence Score:.*$",
                    r"^\s*Mental Sandbox:.*$",
                    r"^\s*\d+\.\s*STRICT:.*$",
                    r"^\s*\d+\.\s*MAY USE.*$",
                    r"^\s*Output ONLY.*$",
                    r"^\s*REMINDER:.*$",
                    r"^\s*Okay.*$",
                    r"^\s*Certainly.*$",
                    r"^\s*```json.*$",
                    r"^\s*```.*$",
                    r"^\s*Generation:\s*",
                    r"^\s*Narrative:\s*",
                    r"^\s*Scene \d+:\s*",
                    r"^\s*Beat \d+:\s*",
                    r"^\s*Refinement \d+.*$",
                    r"^\s*Yes\s*$",
                    r"^\s*N/A\s*$",
                    r"^\s*\[.*?\]\s*$",
                    r"^\s*-\s.*$",
                    r"^\s*\*\s.*$",
                    r"^\s*Outcome is.*$",
                    r"^\s*System:.*$",
                    r"^\s*User:.*$",
                    r"^\s*Check\..*$",
                    r"^\s*Task:.*$",
                    r"^\s*\?.*$",
                ]
            )
            for line in lines:
                stripped_line = line.strip()  # Stripping happens inside cleaning
                is_analysis_or_echo = False
                # Check regex patterns
                for pattern in analysis_patterns_to_remove:
                    if re.match(pattern, stripped_line, re.IGNORECASE):
                        is_analysis_or_echo = True
                        logger.debug(
                            f"Cleaning (beat): Removing line matching pattern '{pattern}': '{line}'"
                        )
                        break
                if is_analysis_or_echo:
                    continue

                # Check for prompt echoing fragments
                # Using lower() for startswith check to be case-insensitive for these fragments
                if any(
                    stripped_line.lower().startswith(fragment.lower())
                    for fragment in prompt_echoing_fragments
                ):
                    logger.debug(
                        f"Cleaning (beat): Removing line starting with prompt fragment: '{line}'"
                    )
                    is_analysis_or_echo = True

                if not is_analysis_or_echo:
                    filtered_lines.append(line)  # Append original line
            cleaned_candidate_text = "\\n".join(
                filtered_lines
            ).strip()  # Final strip is part of cleaning

            log_prompt(  # Log the cleaned version too
                f"LLM Beat Generation Attempt {attempt} for operator {node.op} ({narrative_anchor}) - Cleaned",
                f"Cleaned Generation:\\n{cleaned_candidate_text}",
                sample_index=context.sample_index,
            )

            if not cleaned_candidate_text or cleaned_candidate_text.lower().startswith(
                ("i cannot", "i'm sorry", "i am unable")
            ):
                reason = "empty or API refusal (after cleaning)"
            # --- Use cleaned_candidate_text for validation ---
            elif not validate_beat_numbers(cleaned_candidate_text):
                reason = (
                    "number validation failed (see validator logs for cleaned text)"
                )
            else:
                beat_text = cleaned_candidate_text  # Assign the cleaned text if valid
                break

        except Exception as e:
            reason = f"exception: {e}"
            logger.error(
                f"Exception during beat generation attempt {attempt}: {e}",
                exc_info=True,
            )

        if beat_text is None:
            logger.warning(
                f"Beat {context.beat_counter['current']}/{context.beat_counter['total']} retry {attempt}/{config.MAX_BEAT_RETRIES} for operator {node.op} ({narrative_anchor}) failed: {reason}"
            )
            if attempt < config.MAX_BEAT_RETRIES:
                time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    if beat_text:
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
        padding_usage_percent = (current_padding_total / max_padding_allowed * 100) if max_padding_allowed > 0 else 0
        
        token_percentage_before_padding = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
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
            if current_padding_total + estimated_next_padding_segment_cost > max_padding_allowed:
                logger.warning(
                    f"PADDING BUDGET LIMIT [{node.op}/{narrative_anchor}]: "
                    f"Current padding: {current_padding_total}/{max_padding_allowed} tokens ({padding_usage_percent:.1f}%), "
                    f"Next segment est. cost: +{estimated_next_padding_segment_cost}, "
                    f"Would exceed max padding budget. Stopping padding for this beat."
                )
                break
            
            token_percentage = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
            estimated_after_padding = context.tokens_used + estimated_next_padding_segment_cost
            estimated_percentage_after_padding = (estimated_after_padding / config.MAX_TOTAL_TOKENS) * 100
            
            if context.tokens_used + estimated_next_padding_segment_cost > config.MAX_TOTAL_TOKENS - SAFETY_MARGIN:
                logger.debug(
                    f"PADDING ABORT [{node.op}/{narrative_anchor}]: "
                    f"Current: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
                    f"Next padding est. cost: +{estimated_next_padding_segment_cost}, "
                    f"Would reach: {estimated_after_padding}/{config.MAX_TOTAL_TOKENS} ({estimated_percentage_after_padding:.1f}%), "
                    f"Max allowed: {config.MAX_TOTAL_TOKENS - SAFETY_MARGIN}. Stopping padding for this beat."
                )
                break # Break from this beat's padding loop if next segment likely too costly

            # Increment counter for this beat's padding *before* attempting generation
            local_padding_segments_added += 1
            
            logger.debug(
                f"PADDING ATTEMPT [{node.op}/{narrative_anchor}]: Segment {local_padding_segments_added}/{context.max_pad_paragraphs}, "
                f"Current tokens: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage:.1f}%), "
                f"Current padding: {current_padding_total}/{max_padding_allowed} tokens ({padding_usage_percent:.1f}%)"
            )

            padding_system_prompt = "You are a concise storyteller adding descriptive filler. FOLLOW THE USER'S RULES EXACTLY."
            cleaned_snippet_padding = clean_snippet(context.last_scene_text, max_len=config.PADDING_CONTEXT)

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
                MAX_PAD_COMPLETION_TOKENS, # Only consider the potential completion against the budget
                config.MAX_TOTAL_TOKENS,
                SAFETY_MARGIN,
            ):
                logger.warning(
                    f"Padding after beat for {node.op} ({narrative_anchor}): "
                    f"Budget check indicates insufficient space for padding segment {local_padding_segments_added} "
                    f"(max_completion {MAX_PAD_COMPLETION_TOKENS}). Current COMPLETION tokens: {context.tokens_used}. "
                    f"STOPPING padding for this beat."
                )
                break # Break from this beat's padding loop

            padding_text = generate_with_retry(
                system_prompt=padding_system_prompt,
                user_prompt=padding_user_prompt,
                max_completion_tokens=MAX_PAD_COMPLETION_TOKENS,
                validate_fn=validate_padding,
                retries=config.MAX_PAD_RETRIES,
                sample_index=context.sample_index,
                temperature=config.CREATIVE_NARRATIVE_TEMP,
                reasoning_settings={"exclude": True}  # Always exclude thinking tokens
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
                    current_padding_total = context.padding_stats["total_padding_tokens"] # Update local var
                    context.padding_stats["padding_segments_added"] += 1
                    
                    # Calculate new percentages
                    token_percentage_after = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
                    padding_percentage_after = (current_padding_total / max_padding_allowed * 100)
                    
                    logger.debug(
                        f"PADDING SUCCESS [{node.op}/{narrative_anchor}]: Segment {local_padding_segments_added} added. "
                        f"Size: {ptoks} tokens. Total now: {context.tokens_used}/{config.MAX_TOTAL_TOKENS} ({token_percentage_after:.1f}%), "
                        f"Padding total: {current_padding_total}/{max_padding_allowed} ({padding_percentage_after:.1f}%), "
                        f"Total padding segments: {context.padding_stats['padding_segments_added']}"
                    )
                else:
                    token_percentage_would_be = ((context.tokens_used + ptoks) / config.MAX_TOTAL_TOKENS) * 100
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
                break # Break from this beat's padding loop if generation fails
        
        # Log summary if no padding was added
        if local_padding_segments_added == 0:
            logger.debug(
                f"NO PADDING ADDED [{node.op}/{narrative_anchor}]: Either token limit reached, "
                f"padding budget exceeded ({current_padding_total}/{max_padding_allowed} tokens), "
                f"or max segments per beat ({context.max_pad_paragraphs}) reached."
            )
    
    # End of function token budget summary
    token_percentage_end = (context.tokens_used / config.MAX_TOTAL_TOKENS) * 100
    padding_percentage = (context.padding_stats["total_padding_tokens"] / context.padding_stats["max_padding_allowed"] * 100) if context.padding_stats["max_padding_allowed"] > 0 else 0
    logger.debug(
        f"TOKEN BUDGET END [{getattr(node, 'op', 'Atom')}/{narrative_anchor}]: "
        f"{context.tokens_used}/{config.MAX_TOTAL_TOKENS} tokens ({token_percentage_end:.1f}%), "
        f"Padding: {context.padding_stats['total_padding_tokens']}/{context.padding_stats['max_padding_allowed']} ({padding_percentage:.1f}%), "
        f"Remaining: {config.MAX_TOTAL_TOKENS - context.tokens_used - SAFETY_MARGIN} (with {SAFETY_MARGIN} margin)"
    )


def generate_narrative(
    ast: Node,
    world: dict,
    config: Config,
    encoder,
    p_inflect,
    logger,
    sample_index: int,
) -> str | None:
    """
    Post-Order Strict Validation: Generate a narrative for a ListOps AST, ensuring
    original atomic operands are mentioned incrementally with their parent operation.
    Post-order traversal: children scenes precede their parent's scene.
    """

    if not isinstance(ast, Node):
        raise ValueError("ast must be an instance of Node")
    if not isinstance(world, dict):
        raise ValueError("world must be a dict")
    required_keys = ("characters", "genre", "setting", "object")
    if not all(k in world for k in required_keys):
        logger.error(f"World info missing required keys: {world.keys()}")
        raise ValueError("world missing required key(s)")
    if encoder is None:
        raise RuntimeError("Tokenizer not initialized.")
    if p_inflect is None:
        raise RuntimeError("Inflect engine not initialized.")

    # Pre-calculate node values
    logger.debug("Pre-calculating all AST node values...")
    try:
        eval_node(ast)  # Evaluate the whole tree to populate .value attributes
        logger.debug("AST node values pre-calculation complete.")
    except Exception as e:
        logger.error(f"Error during AST pre-evaluation: {e}")
        raise RuntimeError("Failed to pre-evaluate AST values.") from e

    all_atoms = get_atoms_in_subtree(ast)

    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    narrative_anchor_map = {}
    
    if config.USE_NARRATIVE_ANCHORS and operator_nodes:
        use_llm = config.USE_LLM_NAMING
        characters = world.get("characters", [])
        
        # Create a function for ThreadPoolExecutor to run
        def generate_anchor_for_node(op_node):
            if not isinstance(op_node, OpNode):
                return (id(op_node), None)
                
            node_id = id(op_node)
            narrative_anchor = None
            
            if use_llm:
                # Note: We're no longer using all_anchors_so_far as that would 
                # create a dependency between parallel executions
                try:
                    narrative_anchor = generate_narrative_anchor_with_llm(
                        world_info=world,
                        op_node=op_node,
                        all_previous_anchors=[],  # Empty list since we're parallelizing
                        sample_index=sample_index,
                    )
                except Exception as e:
                    logger.error(f"LLM Naming failed for OpNode {op_node.op}: {e}")
            
            # Return the node_id and anchor (or None if failed)
            return (node_id, narrative_anchor)
        
        # Submit all jobs to ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all jobs
            future_to_node = {
                executor.submit(generate_anchor_for_node, op_node): op_node 
                for op_node in operator_nodes
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_node):
                op_node = future_to_node[future]
                try:
                    node_id, narrative_anchor = future.result()
                    
                    # Apply fallback naming if needed
                    if not narrative_anchor:
                        primary_object = world["object"]
                        op_index = operator_nodes.index(op_node)
                        
                        if characters:
                            char_name = random.choice(characters).get("name", "Someone")
                            possessive = (
                                f"{char_name}'"
                                if char_name.endswith("s")
                                else f"{char_name}'s"
                            )
                            narrative_anchor = f"{possessive} {op_node.op} Result ({op_index+1})"
                        else:
                            narrative_anchor = f"the {primary_object} ({op_node.op} #{op_index+1})"
                    
                    # Update the map
                    narrative_anchor_map[node_id] = narrative_anchor
                    logger.debug(f"Mapped narrative anchor for node {op_node.op}: '{narrative_anchor}'")
                    
                except Exception as e:
                    logger.error(f"Error in parallel anchor generation for {getattr(op_node, 'op', 'unknown')}: {e}")
                    # Apply emergency fallback
                    node_id = id(op_node)
                    narrative_anchor_map[node_id] = f"emergency_fallback_{getattr(op_node, 'op', 'node')}_{node_id}"
        
        logger.info(f"Generated {len(narrative_anchor_map)} narrative anchors in parallel")
    
    scenes = []
    tokens_used = 0
    intro_system_prompt = (
        f"You are a fiction writer specializing in writing opening scenes for {world.get('genre', 'Unknown')} novels. Your ONLY job is to produce an introductory scene for a story with the context I provide you. "
        "In your response, do not include ANY text other than the introductory scene. NO greetings, analysis, checklists, reasoning, or meta-commentary. "
        "Additionally, you must not use ANY numbers in your opening scene. NO numbers are allowed (neither digits like '1', '2' nor words like 'one', 'two', 'first', 'second')."
    )
    intro_user_prompt = (
        f"Write a short, introductory narrative story scene for the following context (2-4 sentences) about '{world.get('setting', 'Unknown')}'. This story has these characters:\\n"
        f"{json.dumps(world.get('characters', []))}\\n"
        "REMEMBER: Output ONLY the narrative text for the scene. Nothing else."
    )

    log_prompt(  # Log the intro prompt
        "Intro Scene Generation Prompt",
        f"System: {intro_system_prompt}\nUser: {intro_user_prompt}",
        sample_index=sample_index,
    )

    intro_text = generate_with_retry(
        system_prompt=intro_system_prompt,
        user_prompt=intro_user_prompt,
        max_completion_tokens=config.INTRO_MAX_TOKENS,
        validate_fn=make_number_validator(
            allowed_atoms=set(),
            forbidden_atoms=set(),
            operand_count=0,
            correct_result_for_beat=config.INVALID_RESULT_PLACEHOLDER,
            strict_zero=True,
        ),
        retries=config.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config.CREATIVE_NARRATIVE_TEMP,
        reasoning_settings={"exclude": True}  # Always exclude thinking tokens
    )

    # --- ADD EXPLICIT LOGGING FOR THE RESULT OF INTRO GENERATION ---
    log_prompt(
        header="Intro Scene LLM Result",
        prompt=f"Final Generated Intro Text (after retries and validation):\n{intro_text if intro_text else 'None (generation failed, was invalid, or API returned no content)'}",
        sample_index=sample_index,
    )

    if (
        intro_text
        and len(encoder.encode(intro_text)) <= config.MAX_TOTAL_TOKENS
    ):
        scenes.append(intro_text)
        tokens_used += len(encoder.encode(intro_text))
        logger.info("Generated introductory scene.")
    else:
        logger.warning(
            "Failed to generate valid introductory scene or it was too long. Starting narrative without intro."
        )
        intro_text = None  # Ensure it's None if failed

    last_scene_text = intro_text if intro_text else "The story begins..."
    introduced_atoms_during_generation = set()
    total_beats = len(operator_nodes)
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
        max_pad_paragraphs=config.MAX_PAD_PARAGRAPHS,  # Or get from config
    )
    
    # Initialize padding budget tracking
    max_padding_allowed = int(config.MAX_TOTAL_TOKENS * config.PADDING_MAX_TOK_PERCENT)
    context.padding_stats["max_padding_allowed"] = max_padding_allowed
    logger.info(
        f"PADDING BUDGET INITIALIZED: Max total tokens: {config.MAX_TOTAL_TOKENS}, "
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
    if (
        not context.scenes
    ):  # Check if any scenes were generated (intro might have failed too)
        logger.error("Narrative generation resulted in no scenes.")
        return None

    narrative_body = "\n\n".join(context.scenes).strip()
    primary_object = world.get("object", "items")
    question = FINAL_QUESTION_TEMPLATE.substitute(primary_object=primary_object)
    final_prompt = narrative_body + question

    final_token_count = len(encoder.encode(final_prompt))
    if final_token_count > config.MAX_TOTAL_TOKENS:
        logger.warning(
            f"Final generated prompt ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({config.MAX_TOTAL_TOKENS}). Truncation might occur."
        )

    # Add padding statistics summary
    total_padding_tokens = context.padding_stats["total_padding_tokens"]
    max_padding_allowed = context.padding_stats["max_padding_allowed"]
    padding_segments_added = context.padding_stats["padding_segments_added"]
    
    padding_percentage_of_max = (total_padding_tokens / max_padding_allowed * 100) if max_padding_allowed > 0 else 0
    padding_percentage_of_total = (total_padding_tokens / context.tokens_used * 100) if context.tokens_used > 0 else 0
    
    logger.info(
        f"PADDING FINAL SUMMARY: "
        f"Padding tokens: {total_padding_tokens}/{max_padding_allowed} ({padding_percentage_of_max:.1f}% of max allowed), "
        f"Padding percentage of total tokens: {padding_percentage_of_total:.1f}%, "
        f"Padding segments added: {padding_segments_added}, "
        f"Config max padding: {config.PADDING_MAX_TOK_PERCENT*100:.1f}% of total tokens"
    )
    
    # Add validation diagnostic summary
    failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
    if os.path.exists(failed_validations_dir):
        validation_files = [f for f in os.listdir(failed_validations_dir) if f.startswith("validation_fail_")]
        if validation_files:
            # Count failures by reason and operation
            failures_by_reason = {}
            failures_by_op = {}
            
            for file in validation_files:
                parts = file.split('_')
                if len(parts) >= 4:
                    op = parts[2]
                    reason = parts[3]
                    
                    failures_by_reason[reason] = failures_by_reason.get(reason, 0) + 1
                    failures_by_op[op] = failures_by_op.get(op, 0) + 1
            
            logger.info(f"VALIDATION FAILURES SUMMARY:")
            logger.info(f"Total validation failures: {len(validation_files)}")
            logger.info(f"Failures by reason: {failures_by_reason}")
            logger.info(f"Failures by operation: {failures_by_op}")
            
            # List the most recent failures for quick reference
            recent_files = sorted(validation_files, reverse=True)[:5]  # Get 5 most recent
            logger.info(f"Most recent validation failures:")
            for file in recent_files:
                logger.info(f"  - {file}")
    
    logger.info(
        f"Successfully generated narrative prompt. Final estimated tokens: {context.tokens_used} (body), {final_token_count} (full prompt)"
    )
    return final_prompt.strip()


def ast_to_prefix(node: Node) -> str:
    """Convert an AST to a prefix notation string."""
    if isinstance(node, Atom):
        return str(node.n)
    parts = [node.op] + [ast_to_prefix(child) for child in node.children]
    return "(" + " ".join(parts) + ")"


# --- HELPER FOR SINGLE SAMPLE GENERATION ---
def generate_single_sample(sample_index: int) -> dict | None:
    """Generate one sample with strict validation."""
    # Properly declare as global to modify it
    global _generate_narrative_recursive
    
    logger.info(f"--- Starting generation for sample {sample_index + 1} ---")
    sample_start_time = time.time()

    # Define paths for the per-sample LLM turn log
    llm_turns_main_dir = os.path.join(LOG_DIR, "llm_turns")
    llm_turns_log_specific_dir = os.path.join(llm_turns_main_dir, "log")
    log_filename_base = f"llm_turns_sample_{sample_index + 1}.log"
    original_log_path = os.path.join(llm_turns_log_specific_dir, log_filename_base)
    
    # Track current operation and narrative anchor for more informative log filenames
    current_op = "unknown"
    current_anchor = "unknown"

    try:
        if encoder is None or p_inflect is None:
            logger.error(
                f"[Sample {sample_index + 1}] Missing tokenizer or inflect engine. Aborting."
            )
            # Attempt to rename log even for this early failure, though few LLM turns might exist
            if os.path.exists(original_log_path):
                failed_log_path = os.path.join(llm_turns_log_specific_dir, f"[FAIL_INIT] {log_filename_base}")
                try:
                    os.rename(original_log_path, failed_log_path)
                    logger.info(f"Renamed LLM turn log for (early) failed sample {sample_index + 1} to: {failed_log_path}")
                except OSError as e:
                    logger.error(f"Error renaming log file {original_log_path} to {failed_log_path}: {e}")
            return None

        logger.info(f"[Sample {sample_index + 1}] Building random AST...")
        ast = build_random_ast(
            max_ops=config.MAX_OPS, max_branch=config.MAX_BRANCH
        )
        validate_ast(ast)
        ast_prefix_string = ast_to_prefix(ast)
        logger.debug(f"[Sample {sample_index + 1}] Generated AST: {ast_prefix_string}")

        logger.info(f"[Sample {sample_index + 1}] Evaluating AST...")
        ground_truth_answer = eval_node(ast)
        logger.info(
            f"[Sample {sample_index + 1}] AST evaluation complete. Ground Truth: {ground_truth_answer}"
        )

        logger.info(f"[Sample {sample_index + 1}] Generating world metadata...")
        world_info = generate_world(
            num_characters=random.randint(config.MIN_WORLD_CHARS, config.MAX_WORLD_CHARS),
            num_concepts=random.randint(config.MIN_WORLD_CONCEPTS, config.MAX_WORLD_CONCEPTS),
            sample_index=sample_index,
        )
        logger.info(f"[Sample {sample_index + 1}] World metadata generated.")
        logger.debug(f"[Sample {sample_index + 1}] World Info: {world_info}")

        # Store the original function reference
        original_function = _generate_narrative_recursive
        
        # Create tracking wrapper
        def _generate_narrative_recursive_with_tracking(node: Node, context: "GenerationContext", is_root: bool):
            nonlocal current_op, current_anchor
            node_id = id(node)
            current_op = getattr(node, 'op', 'UNKNOWN_OP')
            current_anchor = context.narrative_anchor_map.get(node_id, f"unnamed_{current_op}")
            # Use sanitized anchor name for logs
            sanitized_anchor = str(current_anchor).replace(" ", "_").replace("'", "").replace('"', "")
            logger.debug(f"[Sample {sample_index + 1}] Now processing: OP={current_op}, Anchor={sanitized_anchor}")
            return original_function(node, context, is_root)
        
        # Replace the global function temporarily
        _generate_narrative_recursive = _generate_narrative_recursive_with_tracking

        logger.info(
            f"[Sample {sample_index + 1}] Starting narrative rendering with post-order strict validation..."
        )
        try:
            narrative_prompt = generate_narrative(
                ast,
                world_info,
                config,
                encoder,
                p_inflect,
                logger,
                sample_index,
            )
        finally:
            # Restore the original function
            _generate_narrative_recursive = original_function
            
        if not narrative_prompt:
            logger.error(
                f"[Sample {sample_index + 1}] Narrative generation failed or returned None. Aborting sample."
            )
            # Rename log for narrative failure
            if os.path.exists(original_log_path):
                failed_log_path = os.path.join(llm_turns_log_specific_dir, f"[FAIL_NARRATIVE_GEN_{current_op}_{current_anchor}] {log_filename_base}")
                try:
                    os.rename(original_log_path, failed_log_path)
                    logger.info(f"Renamed LLM turn log for failed sample {sample_index + 1} to: {failed_log_path} (narrative gen failed)")
                except OSError as e_rename:
                    logger.error(f"Error renaming log file {original_log_path} to {failed_log_path}: {e_rename}")
            return None

        # Construct the final sample dictionary
        # Create a serializable version of config
        config_dict_to_store = {}
        try:
            # Attempt to create a serializable version of config
            for field_name, field_value in config.__dict__.items():
                if isinstance(field_value, (int, float, str, bool, list, dict, tuple)) or field_value is None:
                    config_dict_to_store[field_name] = field_value
                else:
                    config_dict_to_store[field_name] = str(field_value)  # Convert non-basic types to string
            logger.debug(f"[Sample {sample_index + 1}] Successfully created config_dict_to_store.")
        except Exception as config_err:
            logger.error(f"[Sample {sample_index + 1}] Error converting config to dict: {config_err}", exc_info=True)
            # Raise a new error to be caught by the outer exception handler
            raise RuntimeError(f"Failed to serialize config for sample {sample_index + 1}: {config_err}")

        sample_data = {
            "id": f"verbose_listops_sample_{sample_index + 1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "ast_prefix": ast_prefix_string,
            "ground_truth_answer": ground_truth_answer,
            "world_metadata": world_info,
            "narrative_prompt": narrative_prompt,
            "model_used": MODEL,
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "config_params": config_dict_to_store,  # Use the sanitized version
        }

        sample_end_time = time.time()
        logger.info(
            f"--- Successfully generated sample {sample_index + 1} in {sample_end_time - sample_start_time:.2f}s ---"
        )
        # Rename log for success
        if os.path.exists(original_log_path):
            success_log_path = os.path.join(llm_turns_log_specific_dir, f"[SUCCESS] {log_filename_base}")
            try:
                os.rename(original_log_path, success_log_path)
                logger.info(f"Renamed LLM turn log for successful sample {sample_index + 1} to: {success_log_path}")
            except OSError as e_rename:
                logger.error(f"Error renaming log file {original_log_path} to {success_log_path}: {e_rename}")
        return sample_data

    except Exception as e_caught:  # Catch any exception from the try block with a distinct variable name
        sample_end_time = time.time()  # Calculate time early
        actual_error_type = type(e_caught).__name__

        # Log with logger.exception FIRST to maximize chance of getting stack trace
        logger.exception(
            f"[Sample {sample_index + 1}] Unexpected error during sample data construction or finalization. Type: {actual_error_type}, Error: {e_caught}"
        )

        # Also capture error details for inspection as an additional backup
        import traceback
        error_stack = traceback.format_exc()
        logger.error(f"[Sample {sample_index + 1}] Stack trace:\n{error_stack}")

        # Your custom summary log
        logger.error(
            f"--- Failed sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f}s (Exception: {actual_error_type} at {current_op}/{current_anchor}) ---"
        )

        # Log renaming logic with the actual error type
        if os.path.exists(original_log_path):
            safe_anchor = str(current_anchor).replace(" ", "_").replace("'", "").replace('"', "")
            # Use actual_error_type in the filename
            failed_log_path = os.path.join(llm_turns_log_specific_dir, f"[FAIL_{actual_error_type}_{current_op}_{safe_anchor}] {log_filename_base}")
            try:
                os.rename(original_log_path, failed_log_path)
                logger.info(f"Renamed LLM turn log for failed sample {sample_index + 1} to: {failed_log_path} (due to exception: {actual_error_type})")
            except OSError as e_rename_fail:
                logger.error(f"Error renaming log file {original_log_path} to {failed_log_path}: {e_rename_fail}")
        return None


def main(
    config: Config,
    num_samples: int = NUM_SAMPLES_TO_GENERATE,  # Use global constant instead
    max_workers: int = DEFAULT_MAX_WORKERS,      # Use global constant instead
):
    """Generate samples with strict validation."""
    # Test log output to ensure logger is working properly
    logger.info("START OF MAIN FUNCTION - THIS LOG SHOULD APPEAR IN verbose_listops.log")
    
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
        f".jsonl"
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
    if PROD_RUN and samples_generated_successfully > 0 and output_file and os.path.exists(output_file):
        logger.info(f"--- Starting PROD_RUN validation and cleaning for {output_file} ---")
        # Assuming validator.py is in the same directory as verbose-listops.py
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        validator_script_path = os.path.join(current_script_dir, "validator.py")
        validator_results_path = output_file + ".validation_results.jsonl"

        if not os.path.exists(validator_script_path):
            logger.error(f"Validator script not found at {validator_script_path}. Cannot perform cleaning.")
        else:
            cmd = [
                sys.executable, 
                validator_script_path, 
                output_file, 
                "--output-results", 
                validator_results_path
            ]
            try:
                logger.info(f"Running validator command: {' '.join(cmd)}")
                run_result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                logger.info("Validator process stdout:")
                for line in run_result.stdout.splitlines():
                    logger.info(f"VALIDATOR_STDOUT: {line}")
                if run_result.stderr:
                    logger.warning("Validator process stderr:")
                    for line in run_result.stderr.splitlines():
                        logger.warning(f"VALIDATOR_STDERR: {line}")
                logger.info(f"Validator finished. Results expected in {validator_results_path}")

                bad_sample_ids = set()
                if os.path.exists(validator_results_path):
                    with open(validator_results_path, 'r', encoding='utf-8') as f_results:
                        for line_num, res_line in enumerate(f_results, 1):
                            try:
                                val_res = json.loads(res_line)
                                if val_res.get("status") != "correct":
                                    sample_id_to_remove = val_res.get("id")
                                    if sample_id_to_remove:
                                        bad_sample_ids.add(sample_id_to_remove)
                                    else:
                                        logger.warning(f"Validator result line {line_num} missing 'id': {res_line.strip()}")
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse validator result line {line_num}: {res_line.strip()}")
                    logger.info(f"Identified {len(bad_sample_ids)} samples to remove based on validator results.")

                    if bad_sample_ids:
                        temp_cleaned_output_file = output_file + ".cleaned.tmp"
                        good_samples_written = 0
                        original_sample_count = 0
                        
                        with open(output_file, 'r', encoding='utf-8') as f_orig, \
                             open(temp_cleaned_output_file, 'w', encoding='utf-8') as f_temp:
                            for line_num, dataset_line in enumerate(f_orig, 1):
                                original_sample_count += 1
                                try:
                                    sample_in_dataset = json.loads(dataset_line)
                                    if sample_in_dataset.get("id") not in bad_sample_ids:
                                        f_temp.write(dataset_line) # Write original line as it is good
                                        good_samples_written += 1
                                    else:
                                        logger.debug(f"Removing sample {sample_in_dataset.get('id')} (from line {line_num}) due to validation status.")
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not parse line {line_num} in original dataset '{output_file}' during filtering: {dataset_line.strip()}. Discarding this line.")
                        
                        # Replace original file with cleaned one
                        shutil.move(temp_cleaned_output_file, output_file)
                        deleted_count = original_sample_count - good_samples_written
                        logger.info(f"Removed {deleted_count} bad samples. {good_samples_written} samples remain in {output_file}.")
                        print(f"PROD_RUN: After validation and cleaning, {good_samples_written} samples remain in {output_file}.")
                        # Update samples_generated_successfully for any final tally if needed, though new print is clearer
                        # samples_generated_successfully = good_samples_written
                    else:
                        logger.info("No samples identified for removal by the validator (all were 'correct' or no IDs matched).")
                else:
                    logger.warning(f"Validator results file not found at {validator_results_path}. Skipping removal of bad samples.")

            except FileNotFoundError:
                logger.error(f"Validator script '{validator_script_path}' not found. Ensure it is in the correct directory. Skipping cleaning step.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Validator script failed with exit code {e.returncode}. Skipping cleaning step.")
                logger.error("Validator stdout snapshot:")
                stdout_snapshot = e.stdout.splitlines()
                for i, line_e in enumerate(stdout_snapshot):
                    if i < 50: # Log first 50 lines of stdout
                        logger.error(f"VALIDATOR_STDOUT_ERR: {line_e}")
                    elif i == 50:
                        logger.error(f"VALIDATOR_STDOUT_ERR: ... (stdout truncated after 50 lines)")
                        break
                logger.error("Validator stderr snapshot:")
                stderr_snapshot = e.stderr.splitlines()
                for i, line_e in enumerate(stderr_snapshot):
                    if i < 50: # Log first 50 lines of stderr
                        logger.error(f"VALIDATOR_STDERR_ERR: {line_e}")
                    elif i == 50:
                        logger.error(f"VALIDATOR_STDERR_ERR: ... (stderr truncated after 50 lines)")
                        break
            except Exception as e:
                logger.error(f"An unexpected error occurred during PROD_RUN validation or cleaning: {e}", exc_info=True)
    elif PROD_RUN and (samples_generated_successfully == 0 or not output_file or not os.path.exists(output_file)):
        logger.info("PROD_RUN was True, but no samples were successfully generated, output file is missing, or path is invalid. Skipping validation and cleaning.")

    logging.shutdown()


if __name__ == "__main__":
    # Call update_limits_from_api once after full logger setup and before starting main generation.
    if client and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
        try:
            logger.info("Performing initial OpenRouter limits check before starting main generation...")
            rate_limiter.update_limits_from_api()
        except Exception as e:
            logger.error(f"Initial OpenRouter limits check failed: {e}")
    else:
        logger.warning("Skipping initial OpenRouter limits check: Client not initialized or API key missing/placeholder.")

    main(
        config,
        num_samples=NUM_SAMPLES_TO_GENERATE,
        max_workers=DEFAULT_MAX_WORKERS,
    )
