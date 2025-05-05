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
from openai import OpenAI # Add or ensure this line exists

from dotenv import load_dotenv
load_dotenv()

# Ordinals to ignore when extracting numbers
ORDINAL_WORDS_TO_IGNORE = {
    "first", "second", "third",
}


# --- OpenAI API Key and Tokenizer Initialization ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Use new variable name
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. Using placeholder.") # Updated warning
    GOOGLE_API_KEY = "YOUR_GOOGLE_AI_API_KEY_HERE" # Updated placeholder
try:
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"Failed to initialize tokenizer: {e}")
    encoder = None
from string import Template

# --- Prompt Templates ---
BASE_BEAT_TEMPLATE = Template(
    "You are a $beat_mode storyteller writing the next sequential scene in an ongoing narrative.\n" # Added "sequential"
    "Characters: $characters\n"
    "Setting: $setting\n"
    'Previous Scene Snippet (End of last scene): "...$snippet"\n\n'
    "--- $task_header ---\n"
    "$task_body\n\n" # This will contain the refined ownership_instruction_detail
    "$ultra_strict_instruction\n\n" # This will contain the refined number rules
    "Output only the narrative text for this new scene, continuing from the snippet. Do not include titles, headings, or explanations." # Clarified output expectation
)
#  Configuration Constants 

# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 20 # How many samples to generate in one run
DEFAULT_MAX_WORKERS = 20  # Default number of parallel threads for batch generation

# --- COMPREHENSIVE FEW-SHOT EXAMPLES (Illustrating Success & Failure) ---
# Each tuple: (example_rules_string, good_output, bad_output, failure_reason)
# Assumes "Option A" validator logic (0-10 allowed unless forbidden/required)

FEW_SHOT_EXAMPLES_STRICT = [
    (
        # --- Example 1: Basic Success vs. Extraneous Number (>10) ---
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\n"
            "*   **MUST INCLUDE:** ... mention ... numbers (use digits): thirty-nine (39), ninety (90), and ninety-three (93).\n"
            "*   **MUST AVOID (FORBIDDEN):** Do NOT mention ...: five (5).\n"
            "*   You MAY use the number 3 ('three', the count of direct items...) and the number 1 ('one').\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD Narrative Output (Follows Rules)
        "Felix examined the three newly opened caches. \"Right then,\" he declared, pointing, \"this one holds 93 relics, that one 90 relics, and the last contains 39 relics.\" Liora consulted the Cipher Wheel's concept. \"We need the smallest. That means the cache of 39.\"",
        # BAD Narrative Output (Failure: Includes disallowed extra '12')
        "Felix examined the three newly opened caches. \"Right then,\" he declared, pointing, \"this one holds 93 relics, that one 90 relics, and the last contains 39 relics.\" Liora consulted the Cipher Wheel's concept. \"We need the smallest. That means the cache of 39. It took them 12 minutes to decide.\"",
        # REASONING FOR FAILURE
        "The BAD output failed because it included the number 12. Rule Analysis: 12 was not in MUST INCLUDE {39, 90, 93}, not in MUST AVOID {5}, not the allowed operand count (3), and not an implicitly allowed small number (0-10). It violates the 'ABSOLUTELY NO OTHER NUMBERS' rule."
    ),
    (
        # --- Example 2: Success vs. Missing Required Number ---
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\n"
            "*   **MUST INCLUDE:** ... mention ... numbers (use digits): twenty-eight (28), fifty-five (55), and ninety-four (94).\n"
            "*   **MUST AVOID (FORBIDDEN):** Do NOT mention ...: seven (7).\n"
            "*   You MAY use the number 3 ('three', the count of direct items...) and the number 1 ('one').\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD Narrative Output (Follows Rules)
        "Fizzwick gathered the caches. \"Okay, we have 28 gears, 55 gears, and 94 gears.\" Kelvin nodded. \"The Ninth Gear Dial works on their combined essence, using only the final digit. Let's activate it.\"",
        # BAD Narrative Output (Failure: Missing required '55')
        "Fizzwick gathered the caches. \"Okay, we have 28 gears and 94 gears.\" Kelvin nodded. \"The Ninth Gear Dial works on their combined essence, using only the final digit. Let's activate it.\"",
        # REASONING FOR FAILURE
        "The BAD output failed because it omitted a required number. Rule Analysis: It failed to include 55 from the MUST INCLUDE set {28, 55, 94}."
    ),
    (
        # --- Example 3: Success vs. Including Forbidden Number ---
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\n"
            "*   **MUST INCLUDE:** ... mention ... numbers (use digits): thirty-nine (39), ninety (90), and ninety-three (93).\n"
            "*   **MUST AVOID (FORBIDDEN):** Do NOT mention ...: five (5), twenty-eight (28).\n" # Added another forbidden
            "*   You MAY use the number 3 ('three', the count of direct items...) and the number 1 ('one').\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD Narrative Output (Follows Rules)
        "Felix examined the three newly opened caches. \"Right then,\" he declared, pointing, \"this one holds 93 relics, that one 90 relics, and the last contains 39 relics.\" Liora consulted the Cipher Wheel's concept. \"We need the smallest. That means the cache of 39.\"",
        # BAD Narrative Output (Failure: Includes forbidden '28')
        "Felix examined the three newly opened caches. \"Right then,\" he declared, pointing, \"this one holds 93 relics, that one 90 relics, and the last contains 39 relics.\" Liora consulted the Cipher Wheel's concept. \"We need the smallest, unlike the 28 relics before. That means the cache of 39.\"",
        # REASONING FOR FAILURE
        "The BAD output failed because it included a forbidden number. Rule Analysis: It included 28, which is explicitly listed in the MUST AVOID (FORBIDDEN) set {5, 28}."
    ),
     (
        # --- Example 4: Success vs. Including Disallowed Small Number (that was forbidden) ---
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\n"
            "*   **MUST INCLUDE:** ... mention ... numbers (use digits): forty-eight (48), twenty-seven (27), eighty (80).\n"
            "*   **MUST AVOID (FORBIDDEN):** Do NOT mention ...: five (5), ninety (90).\n" # 5 is forbidden
            "*   You MAY use the number 3 ('three', the count of direct items...) and the number 1 ('one').\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD Narrative Output
        "Kelvin pointed to the three pressure valves. \"Readings are 48, 27, and 80.\" Rynna checked her notes. \"The mechanism requires the middle reading, which is 48.\"",
        # BAD Narrative Output (Failure: Includes forbidden '5', even though it's small)
         "Kelvin pointed to the three pressure valves. \"Readings are 48, 27, and 80.\" Rynna checked her notes. \"The mechanism requires the middle reading, which is 48. We only need 5 more units.\"",
        # REASONING FOR FAILURE
        "The BAD output failed because it included a forbidden number. Rule Analysis: It included 5. Although 5 is normally an allowed small number (0-10), it was explicitly listed in the MUST AVOID (FORBIDDEN) set {5, 90} for this specific beat, making it disallowed."
    ),
    (
        # --- Example 5: Success vs. Including Extraneous Small Number (0-10) ---
        # This shows failure when an *unnecessary* small number is added alongside other numbers.
        # Note: If *only* an allowed small number was added (and no other errors), the Option A validator would pass it.
        # This example demonstrates failure when *other* numbers outside the allowed set are present.
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\n"
            "*   **MUST INCLUDE:** ... mention ... numbers (use digits): forty (40), sixty-one (61).\n"
            "*   **MUST AVOID (FORBIDDEN):** Do NOT mention ...: ninety-nine (99).\n"
            "*   You MAY use the number 2 ('two', the count of direct items...) and the number 1 ('one').\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD Narrative Output
        "They found two chests. The first held 40 gears, the second contained 61 gears. They took both.",
        # BAD Narrative Output (Failure: Includes extraneous '7' and '15')
        "They found two chests. The first held 40 gears, the second contained 61 gears. It took 7 minutes to open them. They took both, leaving 15 behind.",
        # REASONING FOR FAILURE
        "The BAD output failed because it included numbers 7 and 15. Rule Analysis: 7 is an allowed small number (0-10) and not forbidden, BUT 15 is not in MUST INCLUDE {40, 61}, not in MUST AVOID {99}, not the allowed operand count (2), and not an allowed small number. Because 15 violates the 'ABSOLUTELY NO OTHER NUMBERS' rule, the entire output fails validation (even though 7 might have been okay on its own)."
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
    # Example 1 (Soulstone Story - Summarized Narrative + Detailed Trace)
    """
*   **Narrative Summary & Calculation Trace:**
    *   **Scene 1 (Initial Find & Sum):** Characters find three caches of Soulstones: 24, 92, and 96. Narrative cue: "The Unified Measure requires the sum of all." They combine them.
        *   *Calculation:* SUM [24, 92, 96] = 212
        *   *Current Soulstones:* 212
    *   **Scene 2 (New Finds & Average):** They find four more patches: 1, 27, 54, and 88. Narrative cue: "combining it with the Soulstones representing The Unified Measure" (the 212). Then, collection "reduced, leaving them with only... the average, rounded down".
        *   *Calculation:* Values for average are [212, 1, 27, 54, 88]. Total = 382. Count = 5. FLOOR(AVERAGE(382 / 5)) = 76.
        *   *Current Soulstones:* 76
    *   **Scene 3 (More Finds & Average):** They find 68 and 76 more. Narrative cue: "adding them to the Soulstones representing Balanced Convergence" (the 76). Combined amount then shrinks, leaving the "integer average".
        *   *Calculation:* Values for average are [76, 68, 76]. Total = 220. Count = 3. FLOOR(AVERAGE(220 / 3)) = 73.
        *   *Current Soulstones:* 73
    *   **Scene 4 (Final Finds & Sum):** They find 15 and 20 more. Narrative cue: Anya "combined these two newfound sets with the Soulstones linked to the Convergence Sigil" (the 73), stating "their sum we must bind!"
        *   *Calculation:* SUM [73, 15, 20] = 108.
        *   *Current Soulstones:* 108
    *   **Scene 5 (Final Choice - Max):** Presented with portals showing 1 and 69 Soulstones. Narrative cue: "The Apex Lode requires the greatest bounty." They choose the larger cache. This action *determines* the final amount.
        *   *Calculation:* Choose MAX [1, 69] = 69. This selection *replaces* the previous total.
        *   *Final Soulstones:* 69

*   **Question:** Following the entire sequence of events described in the story, exactly how many Soulstones did the characters end up with? Provide only the final integer count.
*   **Answer:** 69
    """,
    # Example 2 (Dwarf Story - MIN Operation, Irrelevant Number, Step Format Trace)
    """
*   **Narrative:**
    The dwarven prospectors, Borin and Grung, started their expedition into the Crystal Caves with 40 Glow-Shards for light. Deep inside, after 5 hours of careful searching, they found three veins. The first yielded 28 Shards, the second a meager 9 Shards, and the third held 35 Shards. Borin consulted the ancient map, "The Echoing Lock requires the *smallest* yield to attune its frequency. We can only take that amount forward." They carefully chipped out the required Shards and left the rest. Later, they found a small pouch containing another 15 Shards, which they added to their collection.

*   **Calculation Trace:**
    *   **Step 1:** Start with 40 Glow-Shards.
    *   **Step 2:** Find veins yielding [28, 9, 35]. Narrative requires MIN ("smallest yield") for the lock. MIN [28, 9, 35] = 9. They *take only this amount*, replacing the initial 40. Current Shards: 9. (Note: "5 hours" is irrelevant to Shard count).
    *   **Step 3:** Find pouch with 15 Shards. Narrative: "added them to their collection". SUM [9, 15] = 24. Current Shards: 24.

*   **Question:** Following the entire sequence of events described in the story, exactly how many Glow-Shards did the prospectors end up with? Provide only the final integer count.
*   **Answer:** 24
    """,
    # Example 3 (Professor Story - Reset/Replacement, Highlighting Format Trace)
    """
*   **Narrative:**
    Professor Armitage began his artifact hunt with high hopes. **He first discovered a cache of 18 Relic Fragments.** Encouraged, **he located another site yielding 32 Fragments**, carefully adding them to his satchel. As he navigated a narrow passage, **a sudden tremor caused him to stumble, his satchel tearing open and spilling its entire contents into a bottomless chasm.** Dejected but determined, he pressed on for another hour. Just as he was about to give up, **he found a small, intact pedestal explicitly marked 'Contains exactly 7 Fragments'.** He took these precious few items.

*   **Calculation Trace (Linked to Highlights):**
    *   Initial Find: **"...cache of 18 Relic Fragments."** -> Current: 18
    *   Second Find: **"...another site yielding 32 Fragments..."** -> Action: Add to current. Calculation: 18 + 32 = 50. Current: 50.
    *   Accident: **"...spilling its entire contents..."** -> Action: Reset based on narrative. Current: 0.
    *   Final Find: **"...pedestal explicitly marked 'Contains exactly 7 Fragments'."** -> Action: Take this specific amount. Current: 7. (Note: "another hour" is irrelevant).

*   **Question:** Following the entire sequence of events described in the story, exactly how many Relic Fragments did Professor Armitage end up with? Provide only the final integer count.
*   **Answer:** 7
    """
]
# --- END NEW FEW-SHOT SECTION ---


# --- Base configurations ---
# Directory for all log output within the project
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# --- Few-shot prompt examples ---
EXAMPLE_TEXTS = [
    (
        "Example 1:\n"
        'Narrative: "The guild offered two contracts: one paying 9 silver pieces, the other only 4. Kaelen chose the lower-paying contract to avoid scrutiny. He then received a standard 5 silver piece bonus for completing the task quickly."\n'
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n"
    ),
    (
        "Example 2:\n"
        "Narrative: To unlock the ancient vault, the combined energy signature of four power crystals (reading 1, 1, 1, and 1) was required. The locking mechanism, however, only used the final digit of their total combined power.\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Modulo 10 (final digit) is 4.\n"
        "Answer: 4\n"
    ),
    (
        "Example 3:\n"
        "Narrative: Three scouts reported patrol durations of 5, 5, and 5 hours. Standard procedure required calculating their average patrol time, rounded down to the nearest whole hour, for the official logbook entry.\n"
        "Implicit Calculation: SUM(5, 5, 5) = 15. Count = 3. Average = 15 / 3 = 5. Floor(5) = 5.\n"
        "Answer: 5"
    ),
]

# --- AST Random ListOps problem gen params ---
DEFAULT_MAX_OPS = 5  # Max operations (e.g. max, min ,etc.) in a problem
MIN_ARITY = 3  # Min numbers in an operation
DEFAULT_MAX_BRANCH = 5  # Max operations / numbers in an operation


# --- Prompt Logging Helper ---
def log_prompt(
    header: str,
    prompt: str,
    sample_index: int | None = None, # <-- Add optional sample_index argument
    path: str = os.path.join(LOG_DIR, "llm_turns.log"),
):
    """Append a timestamped prompt header and text to the prompts log."""
    timestamp = datetime.datetime.now().isoformat()
    # Prepend sample info to header if available
    log_header = f"[Sample {sample_index + 1}] {header}" if sample_index is not None else header
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"--- Log Time: {timestamp} ---\n")
        # Use the modified log_header
        f.write(f"{log_header}\n{prompt}\n\n---\n\n")

# --- API retry + logging config ---

LOG_MAX_BYTES = 5 * 1024 * 1024  # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3  # Number of backup log files to keep
CLEAR_LOGS_ON_START = True  # If True, delete existing logs in LOG_DIR on startup

FINAL_QUESTION_TEMPLATE = Template( # Note: $primary_object is no longer used in this version
    "\n\n---\n\n**Question:** Following the entire sequence of events described in the story, exactly how many $primary_object did the characters end up with? Provide only the final integer count."
)

# --- Dataclasses ---
@dataclass
class Config:
    NUM_SAMPLES_TO_GENERATE: int = NUM_SAMPLES_TO_GENERATE
    DEFAULT_MAX_WORKERS: int = DEFAULT_MAX_WORKERS
    DEFAULT_MAX_TOTAL_TOKENS: int = 8000
    DEFAULT_MAX_BEAT_TOKENS: int = 450
    DEFAULT_MAX_PAD_TOKENS: int = 450
    MAX_TOKENS_BUFFER: int = 500
    RETRY_MAX_ATTEMPTS: int = 5
    RETRY_INITIAL_DELAY: int = 0.5
    MAX_BEAT_RETRIES: int = 5
    MAX_PAD_RETRIES: int = 5
    ATOM_MIN_VALUE: int = 1 # currently only supports p+ integers
    ATOM_MAX_VALUE: int = 100
    USE_OWNERSHIP_NARRATIVE: bool = True  # Master switch for ownership feature
    USE_LLM_NAMING: bool = (
        True  # If True, use LLM for owner names; else use thematic fallback
    )
    NUM_FEW_SHOT_EXAMPLES: int = 2 # 0-3 allowed


config = Config()


# --- GenerationContext dataclass for shared mutable state in recursive narrative generation ---
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
    owner_map: dict
    all_atoms: set
    introduced_atoms: set
    scenes: list
    tokens_used: int
    last_scene_text: str
    beat_counter: dict
    sample_index: int
    max_pad_paragraphs: int = 2

MODEL = "gemini-1.5-pro-latest"
SAFETY_MARGIN = config.MAX_TOKENS_BUFFER
MAX_BEAT_TOKENS = config.DEFAULT_MAX_BEAT_TOKENS
MAX_PAD_TOKENS = config.DEFAULT_MAX_PAD_TOKENS

# --- Setup Logging ---

os.makedirs(LOG_DIR, exist_ok=True)
if CLEAR_LOGS_ON_START:
    for filename in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, filename)
        try:
            os.remove(file_path)
        except OSError:
            pass

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)
# Prevent duplicate handlers if script is run multiple times in same session
if not logger.handlers:
    handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "verbose_listops.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # also log to console for RT feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Instantiate OpenAI Client for Google Endpoint ---
client = None # Initialize client variable
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY_HERE":
         raise ValueError("Google API Key not found or is placeholder.")

    client = OpenAI(
        api_key=GOOGLE_API_KEY,
        # *** Verify this URL in Google's documentation for their OpenAI-compatible endpoint ***
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    logger.info("OpenAI client configured to use Google Generative Language endpoint (Corrected Base URL).") # Updated log
except Exception as e:
    logger.error(f"Failed to configure OpenAI client for Google endpoint: {e}")
    client = None
    # Optional: exit if client fails? sys.exit("API Client setup failed.")

# --- Inflect Engine ---
try:
    p_inflect = inflect.engine()
except Exception as e:
    logger.error(f"Failed to initialize inflect engine: {e}")
    p_inflect = None


# --- Safe, memoised wrapper around inflect.number_to_words -----------------
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
# --------------------------------------------------------------------------

# --- Generic retry helper (centralised back‑off policy) -------------------
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
            logger.warning(
                f"Retryable error calling {getattr(func, '__name__', repr(func))} "
                f"(attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}): {e}"
            )
            if attempt == config.RETRY_MAX_ATTEMPTS:
                logger.error("Max retry attempts reached. Raising.")
                raise
            time.sleep(delay)
            delay *= 2
# --------------------------------------------------------------------------



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
    return current + upcoming + margin > max_total


# --- Helper for future consolidation of retry loops ---
def generate_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    validate_fn: Callable[[str], bool],
    retries: int = config.MAX_BEAT_RETRIES,
    sample_index: int | None = None, # <-- Add optional sample_index argument
):
    """
    Helper to call the OpenAI ChatCompletion API with retries and apply a validation function.
    Returns the first candidate text that passes validate_fn, or None if all attempts fail.
    Passes sample_index to log_prompt if provided.
    """
    candidate = None # Initialize candidate outside the loop
    for attempt in range(1, retries + 1):
        try:
            resp = _chat_completion_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                temperature=0.7,
            )
            raw_content = None
            # Safely access the content
            if resp and resp.choices and len(resp.choices) > 0 and resp.choices[0].message:
                raw_content = resp.choices[0].message.content

            if raw_content is not None:
                candidate = raw_content.strip() # Strip only if content is not None
            else:
                # Log that content was None, even if structure was okay
                logger.warning(f"API call in generate_with_retry attempt {attempt} returned None content. Response object: {resp}")
                candidate = None # Ensure candidate is None

            # Log this LLM turn: prompt and generation, passing sample_index
            log_prompt(
                f"LLM Turn Attempt {attempt}",
                f"System: {system_prompt}\nUser: {user_prompt}\n\nGeneration:\n{candidate}",
                sample_index=sample_index # <-- Pass sample_index here
            )

            if candidate is None: # Check if candidate is None after potential stripping or if raw_content was None
                logger.warning(f"generate_with_retry attempt {attempt} resulted in None candidate.")
            elif not candidate or candidate.lower().startswith(
                ("i cannot", "i'm sorry", "i am unable")
            ):
                logger.warning(f"API refusal on generate_with_retry attempt {attempt}.")
            elif validate_fn(candidate):
                return candidate # Return valid candidate
            # else: validation failed, loop continues

        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")

        # Only sleep if not the last attempt and validation failed or error occurred
        if attempt < retries:
             time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    # If loop finishes without returning, it means all attempts failed
    logger.warning(f"generate_with_retry failed after {retries} attempts.")
    return None # Return None explicitly if all retries fail


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
def build_random_ast(max_ops: int, max_branch: int = DEFAULT_MAX_BRANCH) -> Node:
    """Constructs a random ListOps AST."""
    if not isinstance(max_ops, int) or max_ops < 1:
        raise ValueError("max_ops must be a positive int")
    if max_branch < MIN_ARITY:
        raise ValueError(f"max_branch ({max_branch}) < MIN_ARITY ({MIN_ARITY})")
    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0

    def helper():
        nonlocal count
        if count >= max_ops or (count > 0 and random.random() < 0.1):
            return Atom(random.randint(config.ATOM_MIN_VALUE, config.ATOM_MAX_VALUE))
        count += 1
        op = random.choice(ops)
        # Enforce odd arity for MED operator to avoid even-child median ambiguity
        if op == "MED":
            # Build a list of possible odd arities within range
            possible_arities = [
                n for n in range(MIN_ARITY, max_branch + 1) if n % 2 == 1
            ]
            # Fallback to MIN_ARITY adjusted to odd if necessary
            if not possible_arities:
                arity = MIN_ARITY if MIN_ARITY % 2 == 1 else MIN_ARITY + 1
            else:
                arity = random.choice(possible_arities)
        else:
            arity = random.randint(MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]
        return OpNode(op, children)

    root = helper()
    if isinstance(root, Atom) and max_ops >= 1:
        op = random.choice(ops)
        arity = random.randint(MIN_ARITY, max_branch)
        children = [
            Atom(random.randint(config.ATOM_MIN_VALUE, config.ATOM_MAX_VALUE))
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
        return node.value
    vals = [eval_node(c) for c in node.children]
    if not vals:
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
                f"MED operator with even children ({len(vals)}). Using lower middle."
            )
        if node.op == "AVG" and not vals:
            raise ValueError("Cannot calculate average of zero values.")
        node.value = func(vals)
        return node.value
    except KeyError:
        raise ValueError(f"Unsupported operator: {node.op}")
    except IndexError as e:
        logger.error(f"Indexing error eval {node.op} with {vals}: {e}")
        raise
    except ZeroDivisionError:
        raise ValueError(f"Division by zero during AVG for {node.op}")


def postorder(node: Node):
    """Yield nodes in post-order."""
    for c in node.children:
        yield from postorder(c)
    yield node


@retry_api_call
def _chat_completion_call(*args, **kwargs):
    if args:
        logger.warning(f"_chat_completion_call received unexpected positional arguments: {args}")

    # Add check for client initialization
    if client is None:
        logger.error("OpenAI client (for Google) not initialized. Cannot make API call.")
        raise RuntimeError("API client not initialized.")

    # Filter kwargs to pass only standard OpenAI parameters
    # (Adjust list if needed based on OpenAI library version/spec)
    standard_params = {"model", "messages", "max_tokens", "temperature", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user"}
    standard_kwargs = {k: v for k, v in kwargs.items() if k in standard_params}

    # Map max_completion_tokens if used by calling code
    if "max_completion_tokens" in kwargs and "max_tokens" not in standard_kwargs:
        standard_kwargs["max_tokens"] = kwargs["max_completion_tokens"]

    logger.debug(f"DEBUG: Final args for client.chat.completions.create: {standard_kwargs}")
    logger.debug(f"Calling client.chat.completions.create with args: {standard_kwargs}")
    try:
        # Use the client instance and filtered arguments
        return client.chat.completions.create(**standard_kwargs)
    except Exception as e:
        logger.error(f"Error during client.chat.completions.create: {e}")
        logger.error(f"Args that failed: {standard_kwargs}")
        raise # Re-raise the exception

    # --- END TEMPORARY ---



# --- JSON Cleaning Helper ---
def clean_and_parse_json_block(text: str):
    """Strip Markdown code fences and parse JSON."""
    # Handle potential markdown fences and leading/trailing whitespace
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e} in text:\n---\n{text}\n---")
        raise  # Re-raise after logging


# --- Tuned Generate World Function ---
def generate_world(num_characters: int = 5, num_concepts: int = 7, max_retries: int = 3) -> dict:
    """
    Generates fictional world metadata using an LLM, with a tuned prompt
    to maximize the likelihood of receiving valid, parseable JSON,
    especially regarding quote escaping. Includes retries as a fallback.
    """
    if not isinstance(num_characters, int) or num_characters < 1:
        raise ValueError("num_characters must be positive int")
    if not isinstance(num_concepts, int) or num_concepts < 1:
        raise ValueError("num_concepts must be positive int")

    # --- TUNED PROMPT ---
    prompt = (
        "You are an expert system designed to generate structured data in **strictly valid JSON format**.\n"
        "Your task is to create fictional world metadata.\n\n"
        "**Instructions:**\n"
        f"1. Generate exactly {num_characters} distinct characters. Each character MUST have a 'name', 'role', and 'quirk' field, all as strings.\n"
        "2. Define a 'genre' (string).\n"
        "3. Define a 'setting' (string).\n"
        "4. Define an 'object' (string, plural noun for items characters collect).\n\n"
        "**Output Format:**\n"
        "Output *ONLY* a single, raw, **strictly valid JSON object** adhering precisely to the following structure. Do NOT include ```json markdown fences or *any* other text before or after the JSON object.\n"
        "{\n"
        '  "characters": [{"name": "string", "role": "string", "quirk": "string"}, ...],\n'
        '  "genre": "string",\n'
        '  "setting": "string",\n'
        '  "object": "string"\n'
        "}\n\n"
        "**!!! CRITICAL JSON RULE !!!**\n"
        "If any string value itself needs to contain double quotes (e.g., a nickname within a name, a quote in a setting description), these internal double quotes **MUST** be escaped with a backslash (`\\`).\n"
        "   - **CORRECT Example:** `\"name\": \"Bartholomew \\\"Barty\\\" Bumble\"`\n"
        "   - **INCORRECT Example:** `\"name\": \"Bartholomew \"Barty\" Bumble\"` (This will cause a parsing error!)\n"
        "Adhere strictly to all JSON syntax rules, including commas between elements and correct brace/bracket usage.\n\n"
        "Generate the JSON data now."
    )
    # --- END TUNED PROMPT ---

    for attempt in range(max_retries):
        logger.debug(f"Attempting world generation (Attempt {attempt + 1}/{max_retries}) with tuned prompt.")
        text = None
        try:
            # !!! Ensure _chat_completion_call, MODEL, logger exist in your script !!!
            resp = _chat_completion_call(
                model=MODEL, # Uses the MODEL constant defined in your script
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000, # Adjust if needed
                temperature=0.7, # Slightly lower temp might help consistency
            )
            if hasattr(resp, "choices") and resp.choices: # Check if choices list exists and is not empty
                # Access the *first* choice object in the list
                first_choice = resp.choices[0]
                # Check if the first choice has a 'message' attribute and it's not None
                if hasattr(first_choice, "message") and first_choice.message:
                    # Access content from the message within the first choice
                    text = first_choice.message.content
                    if text is None:
                        # Log if the content *within* the message is None
                        logger.warning(f"World Gen Attempt {attempt + 1}: API returned None content within message. Response: {resp}")
                        text = "" # Treat as empty string
                else:
                    # Log if the first choice object itself lacks a 'message' attribute
                    logger.error(f"World Gen Attempt {attempt + 1}: First choice object lacks 'message' attribute or message is empty. Response: {resp}")
                    text = "" # Treat as error / empty string
            else:
                # Log if the response lacks the 'choices' list entirely or it's empty
                logger.error(f"World Gen Attempt {attempt + 1}: API response lacks 'choices' list or it's empty. Response: {resp}")
                text = "" # Treat as error / empty string
            if not text.strip():
                logger.warning(f"World Gen Attempt {attempt + 1}: Received empty response from API.")
                # Optional: Add a small delay before retrying
                if attempt < max_retries - 1:
                    time.sleep(1) # Uses the time module
                continue # Go to the next attempt

            # !!! Ensure clean_and_parse_json_block exists in your script !!!
            world = clean_and_parse_json_block(text) # Uses your existing cleaning function

            # --- Validation ---
            required_keys = ["characters", "genre", "setting", "object"]
            if not all(k in world for k in required_keys):
                 logger.warning(f"World Gen Attempt {attempt + 1}: Generated JSON missing required keys. Keys found: {world.keys()}")
                 raise ValueError("Generated JSON missing required keys") # Raise error to trigger except block

            if not isinstance(world.get("characters"), list) or not world["characters"]:
                 logger.warning(f"World Gen Attempt {attempt + 1}: 'characters' key is not a non-empty list.")
                 raise ValueError("'characters' key is not a non-empty list")

            logger.debug(f"World Gen Attempt {attempt + 1}: Successfully generated and parsed world JSON.")
            logger.debug(f"Generated object: {world.get('object', 'N/A')}")
            return world # Success! Exit the function.

        except (json.JSONDecodeError, ValueError) as e: # Catch both parsing and validation errors
            logger.error(f"World Gen Attempt {attempt + 1}: Failed ({type(e).__name__}): {e}. Raw text:\n---\n{text}\n---")
            # Fall through to retry
        except Exception as e:
            # Catch other potential errors during API call or processing
            logger.error(f"World Gen Attempt {attempt + 1}: Unexpected error: {e}. Raw text:\n---\n{text if text else 'N/A'}\n---")
            # Fall through to retry

        # If parsing or validation failed and retries remain, wait before next attempt
        if attempt < max_retries - 1:
            # !!! Ensure config exists or adapt this delay calculation !!!
            # Option 1: Use your config object (Recommended if config is accessible here)
            # delay = config.RETRY_INITIAL_DELAY * (2 ** attempt)

            # Option 2: Use a simple hardcoded delay if config isn't easily available
            delay = 1.0 * (2 ** attempt) # Simple exponential backoff starting at 1 second

            logger.info(f"Retrying world generation in {delay:.2f} seconds...")
            time.sleep(delay) # Uses the time module

    # If loop finishes without returning, all retries failed
    logger.error(f"Failed to generate valid world JSON after {max_retries} attempts.")
    raise RuntimeError("World generation failed: Could not get valid JSON from LLM.") # Raise an error to stop the sample generation

# --- Number Extraction (Enhanced with Inflect for Words up to MAX_VALUE) ---
DIGIT_REGEX = re.compile(r"\b-?\d+\b")


def _build_expanded_number_words_dict(
    max_val: int = config.ATOM_MAX_VALUE,
) -> dict[str, int]:
    """Builds a dictionary mapping number words to ints up to max_val using inflect."""
    if p_inflect is None:
        logger.error(
            "Inflect engine not available for building expanded number words dict. Using basic 0-20."
        )
        # Fallback to basic dictionary if inflect failed
        return {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
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


EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict()

# --- NEW: Sort keys by length descending to prioritize longer matches ---
sorted_number_words = sorted(EXPANDED_NUMBER_WORDS_DICT.keys(), key=len, reverse=True)
NUMBER_WORDS_PATTERN = (
    r"\b(?:(minus|negative)\s+)?("
    + "|".join(re.escape(k) for k in sorted_number_words) # Use sorted keys
    + r")\b"
)
# --- END NEW ---
NUMBER_WORDS_REGEX = re.compile(NUMBER_WORDS_PATTERN, re.IGNORECASE)


def extract_numbers_from_text(text: str) -> Set[int]:
    """Extracts integers (digits and words), ignoring specified ordinals."""
    # --- DEBUG START ---
    logger.debug(f"--- extract_numbers_from_text START ---")
    logger.debug(f"Input text (first 200 chars): '{text[:200]}...'")
    # --- DEBUG END ---

    if not text:
        return set()

    found_numbers = set()

    # --- DEBUG START ---
    logger.debug("Checking for digits...")
    # --- DEBUG END ---
    for match in DIGIT_REGEX.finditer(text):
        digit_str = match.group(0)
        # --- DEBUG START ---
        logger.debug(f"  Found digit string: '{digit_str}'")
        # --- DEBUG END ---
        try:
            value = int(digit_str)
            # --- DEBUG START ---
            logger.debug(f"    Parsed as int: {value}")
            # --- DEBUG END ---
            # Only allow 'one' in word form; skip digit '1' - KEEPING THIS RULE
            if value == 1:
                # --- DEBUG START ---
                logger.debug(f"    Skipping digit 1 (only word 'one' allowed).")
                # --- DEBUG END ---
                continue
            found_numbers.add(value)
            # --- DEBUG START ---
            logger.debug(f"    Added {value} to found_numbers.")
            # --- DEBUG END ---
        except ValueError:
            # --- DEBUG START ---
            logger.debug(f"    Failed to parse '{digit_str}' as int.")
            # --- DEBUG END ---
            continue

    # --- DEBUG START ---
    logger.debug("Checking for number words...")
    # --- DEBUG END ---
    for match in NUMBER_WORDS_REGEX.finditer(text):
        sign_word = match.group(1)
        number_word = match.group(2).lower()
        # --- DEBUG START ---
        logger.debug(f"  Found potential number word: '{match.group(0)}' -> word: '{number_word}', sign: '{sign_word}'")
        # --- DEBUG END ---

        # Ordinal Check
        if number_word in ORDINAL_WORDS_TO_IGNORE:
            # --- DEBUG START ---
            logger.debug(f"    Ignoring ordinal word: '{number_word}'")
            # --- DEBUG END ---
            continue

        # Dictionary Lookup
        value = EXPANDED_NUMBER_WORDS_DICT.get(number_word)
        # --- DEBUG START ---
        logger.debug(f"    Dictionary lookup for '{number_word}': {value}")
        # --- DEBUG END ---

        if value is not None:
            if sign_word and value != 0:
                value = -value
                # --- DEBUG START ---
                logger.debug(f"    Applied negative sign: {value}")
                # --- DEBUG END ---
            found_numbers.add(value)
            # --- DEBUG START ---
            logger.debug(f"    Added {value} to found_numbers.")
            # --- DEBUG END ---
        else:
            # This case should ideally not happen if regex is built from dict keys
            logger.warning(f"    Word '{number_word}' found by regex but not in dict.")

    # --- DEBUG START ---
    logger.debug(f"Final found_numbers: {found_numbers}")
    logger.debug(f"--- extract_numbers_from_text END ---")
    # --- DEBUG END ---
    return found_numbers


# --- Factory for number validation ---
def make_number_validator(
    allowed_atoms: Set[int],
    forbidden_atoms: Set[int],
    operand_count: int
) -> Callable[[str], bool]:
    """
    Return a validator function based on new rules, including operand count.
    Using STRICT validation (e.g., original or Option A allowing 0-10 unless forbidden).
    """
    logger.debug(f"Creating validator (STRICT for Few-Shot) with: Allowed={allowed_atoms}, Forbidden={forbidden_atoms}, OperandCount={operand_count}")

    # Define the set of implicitly allowed small numbers (IF USING OPTION A VALIDATOR)
    IMPLICITLY_ALLOWED_SMALL_NUMBERS = set(range(0, 11)) # Allow 0 through 10

    def validate(text: str) -> bool:
        found_numbers = extract_numbers_from_text(text)
        logger.debug(f"Validator Input Text: \"{text[:100]}...\"")
        logger.debug(f"Validator Found Numbers: {found_numbers}")

        # Rule 1: Check if all required atoms are present
        missing_expected = allowed_atoms - found_numbers
        if missing_expected:
            # --- MODIFIED LOG ---
            logger.debug(f"Validation FAIL (Rule 1): Missing required numbers: {missing_expected}. Required={allowed_atoms}, Found={found_numbers}")
            return False

        # Rule 2: Check if any forbidden atoms are present
        found_forbidden = found_numbers & forbidden_atoms
        if found_forbidden:
            truly_forbidden_found = found_forbidden - allowed_atoms
            if truly_forbidden_found:
                # --- MODIFIED LOG ---
                logger.debug(f"Validation FAIL (Rule 2): Found forbidden numbers: {truly_forbidden_found}. Forbidden={forbidden_atoms}, Found={found_numbers}")
                return False
            else:
                logger.debug(f"Validator INFO: Found number(s) {found_forbidden} that were technically forbidden but also required. Allowing.")

        # Identify numbers found that were NOT explicitly required for this beat
        unexpected_found = found_numbers - allowed_atoms
        # logger.debug(f"Validator Unexpected Found (Before Rule 3/4/5): {unexpected_found}") # Keep this if you want

        # Check unexpected numbers against conditional allowances
        truly_disallowed_extras = set()
        for extra_num in unexpected_found:
            is_allowed_one = (extra_num == 1)
            is_allowed_count = (extra_num == operand_count and extra_num not in forbidden_atoms)
            is_allowed_small = (extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS and extra_num not in forbidden_atoms)

            if not (is_allowed_one or is_allowed_count or is_allowed_small):
                if extra_num not in forbidden_atoms:
                    truly_disallowed_extras.add(extra_num)
                # --- ADD LOGGING HERE TO SEE WHY AN EXTRA IS DISALLOWED ---
                else:
                    logger.debug(f"Validator DEBUG: Extra number {extra_num} was in unexpected_found but also in forbidden_atoms, so not added to truly_disallowed_extras here.")
            # --- ADD LOGGING HERE TO SEE WHY AN EXTRA IS ALLOWED ---
            # else:
            #      logger.debug(f"Validator DEBUG: Extra number {extra_num} was allowed. is_allowed_one={is_allowed_one}, is_allowed_count={is_allowed_count}, is_allowed_small={is_allowed_small}")


        # Final check for disallowed extras
        if truly_disallowed_extras:
            # --- MODIFIED LOG ---
            logger.debug(f"Validation FAIL (Strict Rule): Found unexpected/disallowed numbers: {truly_disallowed_extras}. AllowedSet={allowed_atoms}, ForbiddenSet={forbidden_atoms}, OperandCount={operand_count}, Found={found_numbers}, UnexpectedRaw={unexpected_found}")
            return False

        # If all checks pass
        logger.debug(f"Validation PASS (Strict)")
        logger.debug(f"--> Context: Allowed={allowed_atoms}, Forbidden={forbidden_atoms}, OperandCount={operand_count}, Found={found_numbers}")
        return True

    return validate

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
        # Use the original 'n' value
        return {node.n}
    atoms = set()
    for child in node.children:
        atoms.update(get_atoms_in_subtree(child))
    return atoms

# @retry_api_call # Apply decorator outside the function definition if needed
def generate_owner_name_with_llm(
    world_info: dict,
    op_node: OpNode,
    child_owner_names: list[str],
    max_name_tokens: int = 50,
) -> str | None:
    """
    Uses an LLM call to generate a creative, thematic, and NARRATIVELY USEFUL name
    for an element related to an OpNode's step. Prioritizes concrete elements.
    Relies on @retry_api_call for retries.
    Returns the generated name or None if generation fails after retries.
    """

    op_label = OP_LABELS.get(op_node.op, op_node.op)
    genre = world_info.get("genre", "unknown genre")
    setting = world_info.get("setting", "unknown setting")

    # --- World Info (Concise) ---
    characters_sample = world_info.get("characters", [])[:3]
    characters_str = json.dumps(
        [{"name": c.get("name", "N/A")} for c in characters_sample]
    )
    concepts_sample = world_info.get("entity_concepts", [])[:5] # Assuming you might add this back later
    concepts_str = json.dumps(concepts_sample if concepts_sample else ["generic items", "hidden mechanisms"]) # Add fallback concepts if empty
    # --- End World Info ---

    # --- Child Context (Concise) ---
    children_context = ""
    if child_owner_names:
        display_child_names = child_owner_names[:3]
        children_context = (
            f"This step might build upon previous outcomes conceptually known as: " # Keep conceptual
            f"{', '.join(f'{name}' for name in display_child_names)}"
            f"{' and others...' if len(child_owner_names) > 3 else '.'}"
        )
    # --- End Child Context ---

    # --- Create a more narrative description of the operation ---
    # Helps steer away from pure function
    op_narrative_desc = {
        "MAX": "selecting the largest amount or most significant item discovered",
        "MIN": "choosing the smallest amount or least significant item discovered",
        "SUM": "combining several discovered amounts or items into a total",
        "MED": "finding the middle or balancing value among several options",
        "AVG": "calculating a representative average value from several amounts",
        "SM": "determining a final code or quantity based on the last digit of a sum",
    }.get(op_node.op, f"performing an operation related to '{op_label}'") # Fallback remains
    # --- End narrative description ---


    # --- REVISED LLM Naming Prompt ---
    system_prompt = (
        "You are a creative assistant specializing in generating SHORT, CONCISE, and NARRATIVELY USEFUL names " # Emphasize usefulness
        "for elements within a fictional story. Be concise. Output only the name."
    )
    user_prompt = (
        f"Fictional World Context:\n"
        f"- Genre: {genre}\n"
        f"- Setting: {setting}\n"
        f"- Sample Characters: {characters_str}\n"
        f"- Sample Thematic Concepts: {concepts_str}\n\n"
        f"Task: Generate a SHORT (2-4 words), CONCISE, and THEMATIC name for a specific element relevant to the *current narrative step*. "
        f"This element should represent the focus or result of this step and MUST be something characters could plausibly refer to or interact with.\n"
        f"**Prioritize names for:**\n" # Make it more concrete with examples
        f"*   **Physical objects:** 'The Sunstone Cache', 'Barnaby's Combined Gadget', 'The Seven Echo Stones'.\n"
        f"*   **Specific locations:** 'The Echoing Chamber', 'Mirabel's Calculation Point', 'The Median Platform'.\n"
        f"*   **Documents/Readings:** 'The Final Tally Sheet', 'Gearsmith's Rhyming Scroll', 'The Cogsworth Ledger'.\n"
        f"*   **Significant character actions/decisions:** 'Kaelen's Low Bid', 'The Professor's Risky Calculation', 'Mirabel's Final Selection'.\n\n"
        f"Details about the current narrative step:\n"
        f"- It involves the characters {op_narrative_desc}.\n" # Use narrative description
        f"- {children_context}\n\n" # Include child context if available
        f"Instructions:\n"
        f"- The name MUST fit the {genre} genre and {setting} setting.\n"
        f"- Make it sound like something the characters could naturally refer to in dialogue or narration (e.g., 'Let's check the Final Tally Sheet', 'They approached the Median Platform').\n"
        f"- **Strongly prefer names linked to tangible objects, places, or specific character actions over abstract concepts.**\n" # Explicit prioritization
        f"- AVOID using the exact operation word (like '{op_node.op}') or generic terms like 'Result', 'Value', 'Calculation', 'Step', 'Process', 'Assembly', 'Equilibrium', 'Median', 'Average', 'Modulo'.\n" # Add more forbidden generics
        f"- Keep it concise (ideally 2-4 words).\n"
        f"- Output ONLY the generated name itself, with no quotes, labels, explanations, or introductory phrases."
    )

    prompt_log_header = f"--- LLM Owner Naming Prompt (Op: {op_node.op}, Attempting Call - Revised for Narrative Use) ---"
    prompt_content_for_log = f"System: {system_prompt}\nUser: {user_prompt}"
    logger.debug(
        f"Attempting LLM naming call for {op_node.op} with prompt:\n{prompt_content_for_log}"
    )
    logger.debug(f"Prompt length: {len(prompt_content_for_log)}")
    logger.debug(f"LLM Naming: Requesting max_tokens={max_name_tokens}")

    try:
        # Apply retry logic here if not using decorator
        resp = _chat_completion_call(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # --- CHANGE THIS LINE ---
            # max_completion_tokens=max_name_tokens, # Old way
            max_tokens=max_name_tokens, # New way - pass directly
            # --- END CHANGE ---
            temperature=0.75, # Keep some creativity
        )
        raw_candidate = None # Initialize
        if resp and resp.choices and resp.choices[0] and resp.choices[0].message:
            raw_candidate = resp.choices[0].message.content # Assign only if structure is valid
            if raw_candidate is None:
                logger.warning(f"LLM Naming: Received None content. Full response object: {resp}") 
        else:
            logger.warning(f"LLM Naming: Received unexpected response structure: {resp}")
            # Decide how to handle - maybe return None or raise specific error
            return None # Or raise BeatGenerationError("Invalid response in LLM Naming")

        # Check if content itself is None or empty *after* confirming structure
        if raw_candidate is None:
            logger.warning(f"LLM Naming: Received None content in response.")
            return None

        # --- Post-processing (Keep existing logic) ---
        stop_chars = ["\n", ".", ","]
        first_stop_index = len(raw_candidate)
        for char in stop_chars:
            idx = raw_candidate.find(char)
            if idx != -1 and idx < first_stop_index:
                first_stop_index = idx
        processed_candidate = raw_candidate[:first_stop_index]
        candidate = processed_candidate.strip()
        candidate = re.sub(
            r"^(?:Here is a name:|Name:|Entity Name:|\"|\')",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = re.sub(r"(?:\"|\')$", "", candidate).strip()
        # --- End Post-processing ---


        if not candidate or candidate.lower().startswith(
            ("i cannot", "i'm sorry", "i am unable")
        ):
            logger.warning(
                f"LLM Naming: Invalid/refusal response content (raw: '{raw_candidate}')"
            )
            return None

        # Add a check for overly generic terms that might slip through
        generic_terms_lower = {"result", "value", "calculation", "step", "process", "assembly", "equilibrium", "median", "average", "modulo", "outcome", "determination"}
        if any(term in candidate.lower().split() for term in generic_terms_lower):
            logger.warning(f"LLM Naming: Generated name '{candidate}' contains a generic term. Rejecting.")
            return None # Reject overly generic names

        if len(candidate.split()) > 5: # Stricter word count check
            logger.warning(
                f"LLM Naming: Name potentially too long ({len(candidate.split())} words): '{candidate}'"
            )
            # Optionally reject long names, or just log the warning
            # return None

        logger.debug(
            f"LLM generated owner name: '{candidate}' (processed from raw: '{raw_candidate}')"
        )
        return candidate
    except Exception as e:
        logger.error(
            f"LLM Naming API Error: {e}. Prompt that failed:\n{prompt_content_for_log}"
        )
        # Decide whether to raise or return None based on desired robustness
        # raise # Option 1: Fail fast
        return None # Option 2: Allow fallback naming


# Ensure the decorator is applied if needed
generate_owner_name_with_llm = retry_api_call(generate_owner_name_with_llm)


# --- END PHASE 4b Function ---

# --- Narrative Generation with Strict Checks ---
class BeatGenerationError(Exception):
    """Raised when a story beat fails to generate, aborting entire narrative."""
    pass

# --- Narrative Generation with Parent Operator Prompting ---

def _generate_narrative_recursive(
    node: Node,
    context: "GenerationContext",
    is_root: bool,  # Flag to know if this is the root node call
):
    """
    Recursive helper for strict narrative generation.
    Processes children first, then the current node.
    Handles nested operators with specific prompt instructions.
    Modifies the context object directly.
    """
    # --- Unpack shared context (read-only access mostly, modification happens via context object) ---
    world = context.world
    config = context.config
    encoder = context.encoder
    p_inflect = context.p_inflect
    logger = context.logger
    owner_map = context.owner_map
    all_atoms = context.all_atoms
    # introduced_atoms, scenes, tokens_used, last_scene_text are accessed/modified via context

    node_id = id(node)
    owner_name = owner_map.get(node_id, f"the_unnamed_{node.op}_entity" if isinstance(node, OpNode) else "atom")
    logger.debug(f"_generate_narrative_recursive (POST-ORDER): processing node {getattr(node, 'op', 'Atom')} with owner '{owner_name}'")

    # --- Base case: Atom ---
    if isinstance(node, Atom):
        logger.debug(f"Node is Atom ({node.n}), returning.")
        # No scene generation or state change needed for atoms in post-order
        return # State is managed in context

    # --- Recursive Step: Process Children First (Post-Order) ---
    child_owner_names = []  # Collect owner names of direct OpNode children
    for child in node.children:
        # Recursively call for the child. State updates happen within the context object.
        _generate_narrative_recursive(
            child,
            context, # Pass the mutable context down
            is_root=False,
        )

        # Collect owner names ONLY for OpNode children that have names AFTER the child call returns
        if isinstance(child, OpNode) and id(child) in owner_map:
             child_owner_names.append(owner_map[id(child)])
        elif isinstance(child, OpNode):
             # Log if an OpNode child somehow doesn't have a name mapped
             logger.warning(f"OpNode child {child.op} of parent {node.op} has no owner name in map.")

        # Check token budget *after* each child call returns and updates context.tokens_used
        if context.tokens_used >= config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(f"Token limit reached after processing child of operator {getattr(node, 'op', 'Atom')}. Stopping further generation for this branch.")
            return # Stop processing further children or the parent node

    # --- Process Current Operator Node (After All Children Have Been Processed) ---
    logger.debug(f"Finished processing children for operator {getattr(node, 'op', 'Atom')} ({owner_name}). Now processing node itself.")
    if is_root:
        logger.info(f"ROOT NODE ({node.op}): Starting beat generation. Current tokens: {context.tokens_used}/{MAX_TOTAL_TOKENS}")
    context.beat_counter["current"] += 1
    logger.info(f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for operator {node.op} ({owner_name})")
    op_label = OP_LABELS.get(node.op, node.op)

    # Identify direct atom children and calculate operand count
    direct_atom_children = [c for c in node.children if isinstance(c, Atom)]
    operand_count = len(direct_atom_children)
    logger.debug(f"Calculated operand_count (direct atoms) for node {node.op}: {operand_count}")
    direct_atom_values = {a.n for a in direct_atom_children}

    # Define the set of STRICTLY REQUIRED atoms for this beat
def _generate_narrative_recursive(
    node: Node,
    context: "GenerationContext",
    is_root: bool,
):
    """
    Recursive helper for POST-ORDER strict narrative generation using FEW-SHOT examples.
    Processes children first, then the current node.
    Modifies the context object directly.
    """
    # --- Unpack shared context (...) ---
    world = context.world
    config = context.config
    encoder = context.encoder
    p_inflect = context.p_inflect
    logger = context.logger
    owner_map = context.owner_map
    all_atoms = context.all_atoms

    node_id = id(node)
    owner_name = owner_map.get(node_id, f"the_unnamed_{node.op}_entity" if isinstance(node, OpNode) else "atom")
    logger.debug(f"_generate_narrative_recursive (POST-ORDER): processing node {getattr(node, 'op', 'Atom')} with owner '{owner_name}'")

    # --- Base case: Atom ---
    if isinstance(node, Atom):
        logger.debug(f"Node is Atom ({node.n}), returning.")
        return

    # --- Recursive Step: Process Children First (Post-Order) ---
    child_owner_names = []
    for child in node.children:
        _generate_narrative_recursive(
            child,
            context,
            is_root=False,
        )
        if isinstance(child, OpNode) and id(child) in owner_map:
             child_owner_names.append(owner_map[id(child)])
        elif isinstance(child, OpNode):
             logger.warning(f"OpNode child {child.op} of parent {node.op} has no owner name in map.")
        if context.tokens_used >= config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(f"Token limit reached after processing child of operator {getattr(node, 'op', 'Atom')}. Stopping further generation for this branch.")
            return

    # --- Process Current Operator Node (After All Children Have Been Processed) ---
    logger.debug(f"Finished processing children for operator {getattr(node, 'op', 'Atom')} ({owner_name}). Now processing node itself.")
    if is_root:
        logger.info(f"ROOT NODE ({node.op}): Starting beat generation. Current tokens: {context.tokens_used}/{config.DEFAULT_MAX_TOTAL_TOKENS}")

    context.beat_counter["current"] += 1
    logger.info(f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for operator {node.op} ({owner_name})")
    op_label = OP_LABELS.get(node.op, node.op)

    # --- Calculations for required/forbidden atoms, prompt building ---
    direct_atom_children = [c for c in node.children if isinstance(c, Atom)]
    operand_count = len(direct_atom_children)
    direct_atom_values = {a.n for a in direct_atom_children}

    # Required atoms (ONLY direct atoms, per previous fix)
    required_atoms_for_beat = set(direct_atom_values)
    logger.debug(f"Required atoms for beat {node.op} ({owner_name}): {required_atoms_for_beat}")

    # Forbidden atoms
    forbidden_atoms_for_prompt = context.introduced_atoms.copy()
    truly_forbidden_for_prompt = forbidden_atoms_for_prompt - required_atoms_for_beat
    # --- Semantic Layer: primary object concept ---
    primary_object = world["object"]

    # --- Build the scene preamble (explicit quantities) ---
    # Use object_list_str which contains "word (digit)" format for direct atoms
    object_list_str_for_preamble = ""
    if direct_atom_values:
        items = [f"{num_to_words(x)} ({x})" for x in sorted(direct_atom_values)]
        if len(items) == 1:
            object_list_str_for_preamble = items[0]
        elif len(items) == 2:
            object_list_str_for_preamble = " and ".join(items)
        else:
            object_list_str_for_preamble = ", ".join(items[:-1]) + ", and " + items[-1]

    scene_preamble = "" # Default empty
    # Only generate preambles if there are direct atoms to describe
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
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # REVISED REASON CONSTRAINT:
                f"They collect all these {primary_object}, but for a reason you concoct **that mathematically results in the integer average (floored)**, they are forced to give away their haul so that they are left only with a quantity equal to the average of the initial amounts."
            )
        elif node.op == "SM":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # REVISED REASON CONSTRAINT:
                f"They collect all these {primary_object}, combining their haul. However, for a reason you concoct **that mathematically results in keeping only the final digit of the total**, "
                f"they are forced to give away most of their collection, leaving them only with a quantity equal to the final digit of the total number gathered."
            )

    # Build strings for prompt instructions (MUST INCLUDE, MUST AVOID, MAY USE)
    # ... (Keep the logic from the previous fix where only direct atoms are in must_include_combined_str) ...
    must_include_list = []
    if required_atoms_for_beat:
        items = [f"{num_to_words(x)} ({x})" for x in sorted(required_atoms_for_beat)]
        must_include_list.extend(items)

    if not must_include_list:
        must_include_combined_str = "None applicable for this step (only uses results from previous steps)"
    elif len(must_include_list) == 1:
        must_include_combined_str = must_include_list[0]
    elif len(must_include_list) == 2:
        must_include_combined_str = " and ".join(must_include_list)
    else:
        must_include_combined_str = ", ".join(must_include_list[:-1]) + ", and " + must_include_list[-1]

    if truly_forbidden_for_prompt:
        must_avoid_str = ", ".join(f"{num_to_words(x)} ({x})" for x in sorted(truly_forbidden_for_prompt))
    else:
        must_avoid_str = "None"

    may_use_parts = []
    if operand_count > 0 and operand_count not in truly_forbidden_for_prompt:
        operand_count_word = num_to_words(operand_count)
        may_use_parts.append(f"the number {operand_count} ('{operand_count_word}', the count of direct items being considered)")
    if 1 not in truly_forbidden_for_prompt:
        may_use_parts.append("the number 1 ('one')")
    if may_use_parts:
        may_use_clause = f"*   You MAY use { ' and '.join(may_use_parts) } for natural narrative flow.\n"
    else:
        may_use_clause = ""

    # Build the ultra_strict_instruction string (keep it strict for few-shot)
    ultra_strict_instruction = (
        "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\n"
        f"*   **MUST INCLUDE:** The narrative for *this specific scene* must explicitly mention the following numbers (use digits): {must_include_combined_str}. These represent the direct inputs discovered or used *in this step only*.\n"
        f"*   **MUST AVOID (FORBIDDEN):** Do NOT mention any numbers from this list: {must_avoid_str}. These numbers belong to *other, previously completed* steps or unrelated parts of the story.\n"
        f"{may_use_clause}"
        "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values, intermediate calculations, counts (unless explicitly allowed in MAY USE), or unrelated figures into this scene's text.\n"
        "**Adhere strictly to these rules for this scene only.**"
    )

    # --- Build the scene preamble (explicit quantities) ---
    # (This logic remains the same, depends only on direct_atom_values)
    object_list_str_for_preamble = ""
    if direct_atom_values:
        items = [f"{num_to_words(x)} ({x})" for x in sorted(direct_atom_values)]
        if len(items) == 1:
            object_list_str_for_preamble = items[0]
        elif len(items) == 2:
            object_list_str_for_preamble = " and ".join(items)
        else:
            object_list_str_for_preamble = ""
    if direct_atom_values:
        items = [f"{num_to_words(x)} ({x})" for x in sorted(direct_atom_values)]
        if len(items) == 1:
            object_list_str_for_preamble = items[0]
        elif len(items) == 2:
            object_list_str_for_preamble = " and ".join(items)
        else:
            object_list_str_for_preamble = ", ".join(items[:-1]) + ", and " + items[-1]

    scene_preamble = "" # Default empty
    # Only generate preambles if there are direct atoms to describe
    if direct_atom_values:
        # --- Keep the existing if/elif block for SUM, MED, MIN, MAX, AVG, SM ---
        # (This block describes the *discovery* of new items)
        if node.op == "SUM":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # Removed the second sentence to focus preamble on discovery only
            )
        elif node.op == "MED":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # Removed the second sentence
            )
        elif node.op == "MIN":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or storage areas containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # Removed the second sentence
            )
        elif node.op == "MAX":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or storage areas containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # Removed the second sentence
            )
        elif node.op == "AVG":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # Removed the second sentence
            )
        elif node.op == "SM":
            scene_preamble = (
                f"In this stage, the characters discover separate caches or groups containing "
                f"{object_list_str_for_preamble} {primary_object} respectively. "
                # Removed the second sentence
            )
    # --- End Preamble Generation ---


    # --- REVISED: Build Operational Instruction Detail ---
    has_operator_children = bool(child_owner_names)
    has_direct_atom_children = bool(direct_atom_values)
    operational_instruction = ""
    input_description_parts = []
    input_names_for_reminder = []

    num_inputs = len(node.children)
    correct_result = node.value

    # Describe inputs conceptually
    if has_operator_children:
        child_names_str = ', '.join(f"'{name}'" for name in child_owner_names)
        input_description_parts.append(f"the outcome(s) from previous step(s) known as {child_names_str}")
        input_names_for_reminder.extend(child_owner_names) # Add names for reminder
    if has_direct_atom_children:
        direct_atom_words = [num_to_words(a) for a in sorted(direct_atom_values)]
        direct_values_str = ', '.join(f"{w} ({v})" for w, v in zip(direct_atom_words, sorted(direct_atom_values)))
        input_description_parts.append(f"newly discovered quantities ({direct_values_str})")

    if not input_description_parts:
        inputs_str = "inputs determined entirely by context (e.g., selecting between previous outcomes)"
    else:
        inputs_str = " and ".join(input_description_parts)

    # Describe the required action based on the operator
    action_description = ""
    if node.op == "SUM":
        action_description = (
            f"Your narrative MUST describe the characters **combining all {num_inputs} relevant inputs** ({inputs_str}). "
            f"The narrative action should logically lead to the total sum **({correct_result}) being the quantity now associated with '{owner_name}'**."
        )
    elif node.op == "AVG":
        action_description = (
            f"Your narrative MUST describe the characters **combining all {num_inputs} relevant inputs** ({inputs_str}) "
            f"and then determining the **integer average (rounded down)** of those inputs. "
            f"The narrative action (e.g., a magical reduction, distributing evenly) must clearly lead to this average value **({correct_result}) being the quantity now associated with '{owner_name}'**."
        )
    elif node.op == "SM":
        action_description = (
            f"Your narrative MUST describe the characters **combining all {num_inputs} relevant inputs** ({inputs_str}) "
            f"and then determining the **final digit of the sum**. "
            f"The narrative action (e.g., using a cipher, magical reduction) must clearly lead to this single digit **({correct_result}) being the quantity now associated with '{owner_name}'**."
        )
    elif node.op == "MAX":
        action_description = (
            f"Your narrative MUST describe the characters **evaluating or comparing all {num_inputs} relevant inputs** ({inputs_str}) "
            f"and **selecting ONLY the item/quantity corresponding to the largest value ({correct_result})**. "
            f"The narrative action must clearly show them choosing the maximum."
        )
    elif node.op == "MIN":
        action_description = (
            f"Your narrative MUST describe the characters **evaluating or comparing all {num_inputs} relevant inputs** ({inputs_str}) "
            f"and **selecting ONLY the item/quantity corresponding to the smallest value ({correct_result})**. "
            f"The narrative action must clearly show them choosing the minimum."
        )
    elif node.op == "MED":
        action_description = (
            f"Your narrative MUST describe the characters **evaluating all {num_inputs} relevant inputs ({inputs_str}) based purely on their NUMERICAL VALUES**. "
            f"It must be clear they are identifying the **middle NUMBER VALUE** after conceptually **sorting all {num_inputs} input values from smallest to largest** (disregarding order of discovery/physical arrangement). "
            f"The narrative action must clearly show them **identifying and choosing ONLY the item/quantity corresponding to this numerically middle value ({correct_result})**."
        )
    else:
        action_description = (
            f"Your narrative MUST clearly describe the characters applying the '{op_label}' rule to all {num_inputs} inputs ({inputs_str}). The outcome should correspond to {correct_result}."
        )


    # Add reminder about number usage
    reminder = ""
    if input_names_for_reminder:
        reminder_names_str = ', '.join(f"'{name}'" for name in input_names_for_reminder)
        reminder = (
            f"\n**REMINDER:** Do NOT mention the actual numeric results associated with previous steps ({reminder_names_str}) in your text. Refer to them by name or conceptually only. "
             f"However, you MUST explicitly mention the newly discovered quantities for *this* step: {must_include_combined_str}."
        )

    # Assemble the operational instruction
    operational_instruction = (
        f"This scene resolves the step named '{owner_name}'.\n"
        f"{action_description}\n"
        f"The final quantity associated with '{owner_name}' after this action is {correct_result}."
        f"{reminder}"
    )


    # --- Assemble Task Body ---
    task_body_parts = []
    if scene_preamble:
        task_body_parts.append(f"**Item Discovery:** {scene_preamble}")
    task_body_parts.append(f"**Action/Calculation:** {operational_instruction}")
    task_body = "\n\n".join(task_body_parts)


    # Decide header and mode (Keep existing logic)
    task_header = "Final Discovery" if is_root else "Discovery Step"
    beat_mode = (
        f"{world['genre']}, {'concluding' if is_root else 'continuous'} scene about "
        f"{primary_object}"
    )

    # --- Conditionally Add Meta-Instruction and Task-Solving Few-Shots ---
    few_shot_section = ""
    num_shots = config.NUM_FEW_SHOT_EXAMPLES # Get from config
    if 0 < num_shots <= len(TASK_SOLVING_FEW_SHOTS):
        few_shot_section += META_INSTRUCTION + "\n\n---\n\n" # Add meta-instruction
        # Select the requested number of examples
        selected_examples = TASK_SOLVING_FEW_SHOTS[:num_shots]
        # Format them
        few_shot_section += "\n\n---\n\n".join(selected_examples)
        few_shot_section += "\n\n---\n\n" # Separator after examples
    elif num_shots > len(TASK_SOLVING_FEW_SHOTS):
        logger.warning(f"Requested {num_shots} few-shot examples, but only {len(TASK_SOLVING_FEW_SHOTS)} are available. Using all available.")
        few_shot_section += META_INSTRUCTION + "\n\n---\n\n" # Add meta-instruction
        selected_examples = TASK_SOLVING_FEW_SHOTS # Use all
        few_shot_section += "\n\n---\n\n".join(selected_examples)
        few_shot_section += "\n\n---\n\n" # Separator after examples
    # If num_shots is 0, few_shot_section remains empty, correctly skipping meta-instruction and examples.
    # --- END NEW SECTION ---


    # --- Build the final prompt ---
    follow_example_instruction = (
        "YOUR TASK:\n"
        "Generate the *next* narrative scene based on the 'Previous Scene Snippet' and the 'Discovery Step' / 'Final Discovery' instructions provided below.\n"
        "**CRITICAL:** You MUST follow the specific **ULTRA-STRICT NUMBER RULES** (provided below under 'Your Current Task Starts Below') for THIS scene *exactly*. \n"
        "*   Ensure ALL required numbers (from the 'MUST INCLUDE' list) are present in your generated text.\n"
        "*   Ensure NO forbidden numbers (from the 'MUST AVOID' list) are present.\n"
        "*   Ensure ABSOLUTELY NO OTHER numbers are included unless explicitly allowed by the 'MAY use' clause or are small numbers (0-10) that are NOT forbidden.\n"
        "Failure to follow these number rules precisely will result in an invalid output.\n\n"
        "--- Your Current Task Starts Below ---\n\n"
    )

# Assemble the final prompt (remains the same structure)

    # Assemble the final prompt (remains the same structure)
    beat_prompt = (
        few_shot_section
        + follow_example_instruction 
        + BASE_BEAT_TEMPLATE.substitute(
            beat_mode=beat_mode,
            characters=json.dumps(world["characters"]),
            setting=world["setting"],
            snippet=context.last_scene_text[-100:],
            task_header=task_header,
            task_body=task_body,
            ultra_strict_instruction=ultra_strict_instruction, # The rules for the *current* beat
        )
    )


    log_prompt(
        f"{'=== FINAL' if is_root else '=== Intermediate'} Operator Beat Prompt (Op: {node.op}, Owner: {owner_name})",
        beat_prompt,
        sample_index=context.sample_index # <-- ADD THIS ARGUMENT
    )

    # --- Token Budget Check ---
    estimated_prompt_tokens = len(encoder.encode(beat_prompt))
    if would_exceed_budget(
        context.tokens_used, # This includes tokens from ALL children/padding
        estimated_prompt_tokens + MAX_BEAT_TOKENS,
        config.DEFAULT_MAX_TOTAL_TOKENS,
        SAFETY_MARGIN,
    ):
        logger.warning(f"Approaching token limit before generating operator {node.op} ({owner_name}). Stopping. {'(ROOT NODE)' if is_root else ''}")
        return # <--- If this happens for the root node, its scene is never generated

    # --- Create Validator ---
    # IMPORTANT: Use the strict validator you want the LLM to learn (e.g., Option A or original)
    validate_beat_numbers = make_number_validator(
        allowed_atoms=required_atoms_for_beat,
        forbidden_atoms=truly_forbidden_for_prompt,
        operand_count=operand_count
        # Make sure this validator implements the rules you want enforced,
        # ideally the one corresponding to the examples (e.g., Option A)
    )
    # Padding validator needs to forbid everything introduced so far
    validate_padding = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=context.introduced_atoms.union(required_atoms_for_beat),
        operand_count=0
    )


    # --- LLM Call and Validation Loop ---
    system_prompt = "You are a storyteller focused on narrative flow. FOLLOW THE USER'S NUMBER RULES EXACTLY. Use numbers rather than word forms."
    beat_text = None
    for attempt in range(1, config.MAX_BEAT_RETRIES + 1):
        reason = None
        try:
            resp = _chat_completion_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": beat_prompt},
                ],
                max_completion_tokens=MAX_BEAT_TOKENS,
                temperature=0.4,
            )
            candidate_text = resp.choices[0].message.content.strip()
            log_prompt(
                f"LLM Beat Generation Attempt {attempt} for operator {node.op} ({owner_name})",
                f"System: {system_prompt}\nUser: {beat_prompt}\n\nGeneration:\n{candidate_text}",
                sample_index=context.sample_index # <-- ADD THIS ARGUMENT
            )

            if not candidate_text or candidate_text.lower().startswith(("i cannot", "i'm sorry")):
                reason = "empty or API refusal"
            elif not validate_beat_numbers(candidate_text):
                reason = "number validation failed" # Updated reason
            else:
                beat_text = candidate_text
                break
        except Exception as e:
            reason = f"exception: {e}"
        if attempt < config.MAX_BEAT_RETRIES:
            logger.warning(f"Beat {context.beat_counter['current']}/{context.beat_counter['total']} retry {attempt}/{config.MAX_BEAT_RETRIES} for operator {node.op} ({owner_name}): {reason}")
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))
        if not validate_beat_numbers(candidate_text):
            reason = "number validation failed" # <--- Could fail repeatedly for root


    # --- Process Successful Generation or Raise Error ---
    if beat_text:
        btoks = len(encoder.encode(beat_text))
        context.scenes.append(beat_text)
        context.tokens_used += btoks
        context.last_scene_text = beat_text
        # Update introduced atoms *in the context* with the numbers required for THIS beat
        context.introduced_atoms.update(required_atoms_for_beat)
        logger.debug(f"Beat {context.beat_counter['current']} successful. Introduced atoms updated: {context.introduced_atoms}")
    else: # If generation failed after all retries
        logger.error(
            f"Operator {node.op} ({owner_name}) failed after {config.MAX_BEAT_RETRIES} attempts. Aborting narrative generation. {'(ROOT NODE)' if is_root else ''}"
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} ({owner_name})"
        )

    # --- Padding Generation Loop (existing logic, uses context) ---
    pad_count = 0
    add_padding = not is_root # No padding after the final root node beat
    while add_padding and context.tokens_used < config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < context.max_pad_paragraphs:
        pad_count += 1
        logger.debug(f"Attempting padding {pad_count}/{context.max_pad_paragraphs} after beat for {node.op} ({owner_name})")

        # --- Padding Prompt ---
        padding_prompt_template = Template(
            "You are a $beat_mode storyteller adding brief filler between narrative steps.\n"
            "Characters: $characters\n"
            "Setting: $setting\n"
            'Previous Scene Snippet (End of last scene): "...$snippet"\n\n'
            "--- Task: Add Narrative Padding ---\n"
            "Continue the story *immediately* following the previous snippet. Write one short paragraph (3-5 sentences) "
            "describing atmosphere, character thoughts/reactions, or a minor, non-quantitative action/transition. "
            "DO NOT advance the main plot involving calculations or obtaining specific quantities.\n\n"
            "**!!! CRITICAL RULE: ZERO NUMBERS !!!**\n"
            "*   This padding text MUST NOT contain ANY numbers whatsoever (no digits, no number words like 'one', 'two', 'first', 'second', etc.).\n"
            "*   Forbidden numbers from previous steps include: $forbidden_list_str\n\n" # Keep list for context, but main rule is ZERO numbers
            "Output only the narrative text for this padding paragraph."
        )
        # ... rest of padding generation logic ...

        # Update forbidden list for padding prompt display
        current_forbidden_for_padding = context.introduced_atoms # Everything introduced so far is forbidden
        forbidden_list_str_padding = "None"
        if current_forbidden_for_padding:
            forbidden_list_str_padding = ", ".join(f"{num_to_words(x)} ({x})" for x in sorted(current_forbidden_for_padding))


        padding_prompt = padding_prompt_template.substitute(
            beat_mode=f"{world['genre']} descriptive padding",
            characters=json.dumps(world["characters"]),
            setting=world["setting"],
            snippet=context.last_scene_text[-100:], # Use context's last scene text
            forbidden_list_str=forbidden_list_str_padding
        )

        log_prompt(f"Padding Prompt {pad_count} after {node.op} ({owner_name})", padding_prompt, sample_index=context.sample_index)

        # --- Token Budget Check for Padding ---
        estimated_pad_prompt_tokens = len(encoder.encode(padding_prompt))
        if would_exceed_budget(
            context.tokens_used,
            estimated_pad_prompt_tokens + MAX_PAD_TOKENS,
            config.DEFAULT_MAX_TOTAL_TOKENS,
            SAFETY_MARGIN,
        ):
            logger.warning(f"Approaching token limit before generating padding {pad_count}. Stopping padding.")
            add_padding = False # Prevent further padding attempts
            break # Exit padding loop

        # --- Generate Padding with Retry ---
        # Use the stricter padding validator created earlier
        padding_text = generate_with_retry(
            system_prompt="You are a storyteller adding descriptive filler. FOLLOW THE USER'S NUMBER RULES EXACTLY.",
            user_prompt=padding_prompt,
            max_tokens=MAX_PAD_TOKENS,
            validate_fn=validate_padding, # Use the correct validator
            retries=config.MAX_PAD_RETRIES,
            sample_index=context.sample_index # <-- ADD THIS ARGUMENT
        )


        if padding_text:
            ptoks = len(encoder.encode(padding_text))
            if context.tokens_used + ptoks <= config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN:
                context.scenes.append(padding_text) # Modify context
                context.tokens_used += ptoks      # Modify context
                context.last_scene_text = padding_text # Modify context
                logger.debug(f"Padding {pad_count} successful.")
            else:
                logger.warning(f"Generated padding {pad_count} would exceed token limit. Discarding.")
                add_padding = False # Prevent further padding attempts
                break # Exit padding loop
        else:
            logger.warning(f"Padding generation {pad_count} failed after retries. Stopping padding.")
            add_padding = False # Prevent further padding attempts
            break # Exit padding loop

    # --- No explicit return needed as context is modified in place ---
    return


# --- generate_narrative function remains largely the same ---
# It sets up the context and makes the initial call to the modified _generate_narrative_recursive

def generate_narrative(
    ast: Node, world: dict, config: Config, encoder, p_inflect, logger, sample_index: int # <-- ADDED sample_index
) -> str | None:
    """
    Post-Order Strict Validation: Generate a narrative for a ListOps AST, ensuring
    original atomic operands are mentioned incrementally with their parent operation.
    Post-order traversal: children scenes precede their parent's scene.
    """
    # Initial validation (unchanged)...
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

    # --- Pre-calculate node values ---
    # Ensure all node values are computed *before* generation starts
    # This is crucial as post-order relies on child values being ready
    logger.debug("Pre-calculating all AST node values...")
    try:
        eval_node(ast) # Evaluate the whole tree to populate .value attributes
        logger.debug("AST node values pre-calculation complete.")
    except Exception as e:
        logger.error(f"Error during AST pre-evaluation: {e}")
        raise RuntimeError("Failed to pre-evaluate AST values.") from e


    all_atoms = get_atoms_in_subtree(ast)
    # Owner mapping via postorder...
    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    owner_map = {}
    if config.USE_OWNERSHIP_NARRATIVE:
        if operator_nodes:
            use_llm = config.USE_LLM_NAMING
            characters = world.get("characters", [])
            # Iterate in post-order to ensure child names are available for parent LLM calls
            for op_node in operator_nodes: # Already in post-order thanks to postorder() generator
                if not isinstance(op_node, OpNode):
                    continue
                node_id = id(op_node)
                owner_name = None
                if use_llm:
                    child_owner_names = []
                    for child in op_node.children:
                        # Child names *should* be in the map already due to post-order iteration
                        if isinstance(child, OpNode) and id(child) in owner_map:
                            child_owner_names.append(owner_map[id(child)])
                        elif isinstance(child, OpNode):
                            logger.warning(f"LLM Naming: Child OpNode {child.op} of parent {op_node.op} has no owner name in map during parent naming.")

                    try:
                        owner_name = generate_owner_name_with_llm(
                            world, op_node, child_owner_names
                        )
                    except Exception as e:
                        logger.error(f"LLM Naming failed for OpNode {op_node.op}: {e}")
                        owner_name = None # Fallback handled below

                if not owner_name: # Fallback naming
                    primary_object = world["object"]
                    op_index = operator_nodes.index(op_node) # Find index for fallback naming
                    if characters:
                        char_name = random.choice(characters).get("name", "Someone")
                        possessive = (
                            f"{char_name}'"
                            if char_name.endswith("s")
                            else f"{char_name}'s"
                        )
                        # Make fallback slightly more descriptive
                        owner_name = f"{possessive} {op_node.op} Result ({op_index+1})"
                    else:
                        owner_name = f"the {primary_object} ({op_node.op} #{op_index+1})"
                owner_map[node_id] = owner_name
                logger.debug(f"Mapped owner name for node {op_node.op}: '{owner_name}'")

    # Intro scene generation...
    scenes = []
    tokens_used = 0
    intro_prompt = (
        "You are a creative storyteller setting a scene.\n"
        f"World Info: Genre={world.get('genre', 'Unknown')}, Setting={world.get('setting', 'Unknown')}.\n"
        f"Characters: {json.dumps(world.get('characters', []))}\n"
        "Task: Write a short introductory scene (2-4 sentences) establishing the setting and introducing the main characters or their initial situation. "
        "**ULTRA-STRICT NUMBER RULE:** This introduction MUST NOT contain ANY numbers (digits or words like 'one', 'two', etc.)."
    )
    intro_text = generate_with_retry(
        system_prompt="You are a storyteller. FOLLOW THE USER'S NUMBER RULES EXACTLY.",
        user_prompt=intro_prompt,
        max_tokens=250,
        # Validator ensures no numbers are present
        validate_fn=make_number_validator(allowed_atoms=set(), forbidden_atoms=all_atoms.union({0,1}), operand_count=0),
        retries=3, # Increase retries slightly for this specific task
    )

    if intro_text and len(encoder.encode(intro_text)) <= config.DEFAULT_MAX_TOTAL_TOKENS:
        scenes.append(intro_text)
        tokens_used += len(encoder.encode(intro_text))
        logger.info("Generated introductory scene.")
    else:
        logger.warning("Failed to generate valid introductory scene or it was too long. Starting narrative without intro.")
        intro_text = None # Ensure it's None if failed

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
        owner_map=owner_map,
        all_atoms=all_atoms,
        introduced_atoms=introduced_atoms_during_generation, # Starts empty
        scenes=scenes, # Starts with potential intro
        tokens_used=tokens_used, # Starts with potential intro tokens
        last_scene_text=last_scene_text, # Starts with potential intro text
        beat_counter=beat_counter,
        sample_index=sample_index,
        max_pad_paragraphs=2, # Or get from config
    )
    # --- Start the POST-ORDER recursive generation ---
    try:
        # Initial call to the recursive function for the root node
        _generate_narrative_recursive(
            ast,
            context, # Pass the mutable context
            is_root=True,
        )
    except BeatGenerationError as e:
        logger.error(f"Narrative generation aborted due to beat failure: {e}")
        return None # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error during recursive narrative generation: {e}", exc_info=True)
        return None # Indicate failure

    # --- Final Assembly (using the modified context) ---
    if not context.scenes: # Check if any scenes were generated (intro might have failed too)
        logger.error("Narrative generation resulted in no scenes.")
        return None

    narrative_body = "\n\n".join(context.scenes).strip()
    primary_object = world.get("object", "items")
    question = FINAL_QUESTION_TEMPLATE.substitute(primary_object=primary_object)
    final_prompt = narrative_body + question

    # Final validation check (optional but recommended)
    final_token_count = len(encoder.encode(final_prompt))
    if final_token_count > config.DEFAULT_MAX_TOTAL_TOKENS:
        logger.warning(f"Final generated prompt ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({config.DEFAULT_MAX_TOTAL_TOKENS}). Truncation might occur.")
        # Depending on requirements, you might return None or the truncated prompt

    logger.info(f"Successfully generated narrative prompt. Final estimated tokens: {context.tokens_used} (body), {final_token_count} (full prompt)")
    return final_prompt.strip()

# --- Other functions (AST generation, evaluation, world gen, main loop, etc.) remain unchanged ---
# Make sure to update the `validation_mode` string in `generate_single_sample`'s metadata
# if you want to track that this version uses post-order. E.g.:
# "validation_mode": "postorder_llm_ownership_v1_strict_validate" -> "postorder_llm_ownership_v2_strict_validate" (or similar)


def ast_to_prefix(node: Node) -> str:
    """Convert an AST to a prefix notation string."""
    if isinstance(node, Atom):
        return str(node.n)
    parts = [node.op] + [ast_to_prefix(child) for child in node.children]
    return "(" + " ".join(parts) + ")"


# --- HELPER FOR SINGLE SAMPLE GENERATION ---
def generate_single_sample(sample_index: int) -> dict | None:
    """Generate one sample with strict validation."""
    logger.info(f"--- Starting generation for sample {sample_index + 1} ---")
    sample_start_time = time.time()
    try:
        if encoder is None or p_inflect is None:
            logger.error(
                f"[Sample {sample_index + 1}] Missing tokenizer or inflect engine. Aborting."
            )
            return None

        logger.info(f"[Sample {sample_index + 1}] Building random AST...")
        ast = build_random_ast(max_ops=DEFAULT_MAX_OPS, max_branch=DEFAULT_MAX_BRANCH)
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
            num_characters=random.randint(3, 6), num_concepts=random.randint(5, 10)
        )
        logger.info(f"[Sample {sample_index + 1}] World metadata generated.")
        logger.debug(f"[Sample {sample_index + 1}] World Info: {world_info}")

        logger.info(
            f"[Sample {sample_index + 1}] Starting narrative rendering with pre-order strict validation (v5c)..."
        )
        narrative_prompt = generate_narrative(
            ast, world_info, config, encoder, p_inflect, logger, sample_index # <-- PASS sample_index here
        )
        if narrative_prompt is None:
            logger.error(
                f"[Sample {sample_index + 1}] Narrative generation failed pre-order strict validation. Skipping."
            )
            sample_end_time = time.time()
            logger.error(
                f"--- Failed sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f}s (Narrative Validation Failure) ---"
            )
            return None
        logger.info(
            f"[Sample {sample_index + 1}] Narrative rendering and validation complete."
        )

        all_atoms = get_atoms_in_subtree(ast)
        found_in_final_body = extract_numbers_from_text(narrative_prompt)
        introduced_atoms = found_in_final_body - all_atoms
        missing = all_atoms - found_in_final_body

        sample_data = {
            "id": f"verbose_listop_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{sample_index + 1}",
            "ast_prefix": ast_prefix_string,
            "ground_truth": ground_truth_answer,
            "world_info": world_info,
            "narrative_prompt": narrative_prompt,
            "metadata": {
                "generation_timestamp": datetime.datetime.now().isoformat(),
                "model_used": MODEL,
                "max_ops": DEFAULT_MAX_OPS,
                "max_branch": DEFAULT_MAX_BRANCH,
                "atom_min_value": config.ATOM_MIN_VALUE,
                "atom_max_value": config.ATOM_MAX_VALUE,
                "max_beat_retries": config.MAX_BEAT_RETRIES,
                "max_pad_retries": config.MAX_PAD_RETRIES,
                "validation_mode": (
                    "postorder_llm_ownership_v1_strict_validate"
                    if (config.USE_OWNERSHIP_NARRATIVE and config.USE_LLM_NAMING)
                    else (
                        "postorder_thematic_ownership_v1_strict_validate"
                        if config.USE_OWNERSHIP_NARRATIVE
                        else "postorder_strict_validation_v1_strict_validate"
                    )
                ),
            },
        }
        sample_end_time = time.time()
        logger.info(
            f"--- Successfully generated sample {sample_index + 1} in {sample_end_time - sample_start_time:.2f} seconds ---"
        )
        return sample_data

    except ValueError as e:
        logger.error(f"[Sample {sample_index + 1}] Data validation error: {e}")
    except RuntimeError as e:
        logger.error(f"[Sample {sample_index + 1}] Runtime error: {e}")
    except Exception as e:
        logger.exception(f"[Sample {sample_index + 1}] Unexpected error: {e}")

    sample_end_time = time.time()
    logger.error(
        f"--- Failed sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f}s (Exception) ---"
    )
    return None


def main(
    config: Config,
    num_samples: int = config.NUM_SAMPLES_TO_GENERATE,
    max_workers: int = config.DEFAULT_MAX_WORKERS
):
    """Generate samples with strict validation."""
    # --- Dynamic Filename Generation ---
    sanitized_model_name = MODEL.replace("/", "_").replace(":", "-")
    output_file = (
        f"DATASET_"
        f"{config.NUM_FEW_SHOT_EXAMPLES}shot_"
        f"{config.DEFAULT_MAX_TOTAL_TOKENS}-tok_"
        f"{DEFAULT_MAX_OPS}-mxops_"
        f"{MIN_ARITY}-arity_"
        f"{DEFAULT_MAX_BRANCH}-mxbrch"
        f"{sanitized_model_name}_"
        f".jsonl"
    )
    logger.info(f"Output filename (dynamic): {output_file}")
    logger.info(
        f"Script started. Generating {num_samples} samples using up to {max_workers} workers."
    )

    samples_generated_successfully = 0
    samples_failed = 0
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
                logger.error(f"[Sample {index + 1}] task generated exception: {exc}")
                samples_failed += 1
    logger.info(
        f"Parallel generation complete. Writing {samples_generated_successfully} samples to {output_file}..."
    )
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            for sample_data in results:
                try:
                    f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                except TypeError as e:
                    logger.error(
                        f"Serialization failed: {e}. Sample: {sample_data.get('id', 'Unknown')}"
                    )
                    samples_failed += 1
                    samples_generated_successfully -= 1
                except Exception as e:
                    logger.error(
                        f"Write error sample {sample_data.get('id', 'Unknown')}: {e}"
                    )
                    samples_failed += 1
                    samples_generated_successfully -= 1
    except IOError as e:
        logger.error(f"Fatal file write error {output_file}: {e}")
        samples_failed = num_samples
        samples_generated_successfully = 0
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"--- Batch generation complete ---")
    logger.info(f"Total samples attempted: {num_samples}")
    logger.info(f"Successfully generated and written: {samples_generated_successfully}")
    logger.info(f"Failed generations or writes: {samples_failed}")
    total_count   = num_samples # Use the function argument
    success_count = samples_generated_successfully # Use the counter
    success_rate  = (success_count / total_count * 100) if total_count else 0
    logger.info(f"Generation success rate: {success_rate:.2f}% ({success_count}/{total_count})")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Dataset output file: {output_file}")
    logging.shutdown()


if __name__ == "__main__":
    main(
        config,
        num_samples=config.NUM_SAMPLES_TO_GENERATE,
        max_workers=config.DEFAULT_MAX_WORKERS
    )