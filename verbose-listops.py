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
import openai

# Ordinals to ignore when extracting numbers
ORDINAL_WORDS_TO_IGNORE = {
    "first", "second", "third",
}


# --- OpenAI API Key and Tokenizer Initialization ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. Using placeholder.")
    OPENAI_API_KEY = "YOUR_API_KEY_HERE"

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
OUTPUT_FILENAME = "4o-mini-verbose_listops_dataset_ultra_strict_v4.jsonl"  # Output file for the dataset
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
    path: str = os.path.join(LOG_DIR, "llm_turns.log"),
):
    """Append a timestamped prompt header and text to the prompts log."""
    timestamp = datetime.datetime.now().isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"--- Log Time: {timestamp} ---\n")
        f.write(f"{header}\n{prompt}\n\n---\n\n")


# --- API retry + logging config ---

LOG_MAX_BYTES = 5 * 1024 * 1024  # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3  # Number of backup log files to keep
CLEAR_LOGS_ON_START = False  # If True, delete existing logs in LOG_DIR on startup

FINAL_QUESTION_TEMPLATE = Template( # Note: $primary_object is no longer used in this version
    "\n\n---\n\n**Final Question:** Carefully follow the main sequence of calculations described throughout the *entire narrative* above, focusing on the primary objective the characters are pursuing. Identify the single, concluding calculation performed as the **final step of that primary objective**. What is the **single integer** value that results *exclusively* from this final, top-level calculation related to the main task? Ignore any unrelated side-calculations or estimations mentioned incidentally, especially if they occur late in the narrative but are not part of the core sequence."
)

JUDGE_INSTRUCTIONS = ( # Renamed constant for clarity
    "\n\n**Instructions for Answering:**\n1. Read the entire narrative carefully from beginning to end.\n2. Identify the **primary objective** or main sequence of operations the characters are performing throughout the story.\n3. Trace this main sequence step-by-step, noting the numbers involved and performing calculations *only for this primary sequence*.\n4. **Crucially, disregard any incidental calculations, estimations, or quantitative details mentioned that are clearly separate from the main operational sequence (e.g., calculating fuel, counting unrelated items mentioned in passing).**\n5. Identify the *final operation* within the **main sequence** that concludes the primary objective.\n6. Determine the single integer result produced *only* by that concluding operation, based on its inputs as established by the preceding steps *within the main sequence*.\n7. Output **only** that single, final integer answer. No other text."
)# --- Dataclasses ---
@dataclass
class Config:
    NUM_SAMPLES_TO_GENERATE: int = NUM_SAMPLES_TO_GENERATE
    DEFAULT_MAX_WORKERS: int = DEFAULT_MAX_WORKERS
    DEFAULT_MAX_TOTAL_TOKENS: int = 8000
    DEFAULT_MAX_BEAT_TOKENS: int = 450
    DEFAULT_MAX_PAD_TOKENS: int = 450
    MAX_TOKENS_BUFFER: int = 500
    PROMPT_SHOT_COUNT: int = 3
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
    max_pad_paragraphs: int = 2


MODEL = "gpt-4o"
MAX_TOTAL_TOKENS = config.DEFAULT_MAX_TOTAL_TOKENS
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

    # Initialize OpenAI client now that logger and API key are defined
    try:
        openai.api_key = OPENAI_API_KEY
        logger.info("OpenAI client configured.")
    except Exception as e:
        logger.error(f"Failed to configure OpenAI client: {e}")



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
):
    """
    Helper to call the OpenAI ChatCompletion API with retries and apply a validation function.
    Returns the first candidate text that passes validate_fn, or None if all attempts fail.
    """
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
            candidate = resp.choices[0].message.content.strip()
            # Log this LLM turn: prompt and generation
            log_prompt(
                f"LLM Turn Attempt {attempt}",
                f"System: {system_prompt}\nUser: {user_prompt}\n\nGeneration:\n{candidate}"
            )
            if not candidate or candidate.lower().startswith(
                ("i cannot", "i'm sorry", "i am unable")
            ):
                logger.warning(f"API refusal on generate_with_retry attempt {attempt}.")
            elif validate_fn(candidate):
                return candidate
        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")
        time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))
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
def _chat_completion_call(**kwargs):
    return openai.chat.completions.create(**kwargs)


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


def generate_world(num_characters: int = 5, num_concepts: int = 7) -> dict:
    """Generates fictional world metadata including entity concepts."""
    if not isinstance(num_characters, int) or num_characters < 1:
        raise ValueError("num_characters must be positive int")
    if not isinstance(num_concepts, int) or num_concepts < 1:
        raise ValueError("num_concepts must be positive int")

    prompt = (
        "You are a creative world-builder.\n"
        f"Generate {num_characters} distinct characters (name, role, quirk). Define a genre and setting.\n"
        "Also, name a category of objects that characters will search for and collect, in plural form (e.g., 'coins').\n"
        "Output *only* a valid JSON object with NO extra text before or after the JSON structure:\n"
        "{\n"
        '  "characters": [{"name": "string", "role": "string", "quirk": "string"}, ...],\n'
        '  "genre": "string",\n'
        '  "setting": "string",\n'
        '  "object": "string"\n'
        "}"
    )

    try:
        resp = _chat_completion_call(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            temperature=0.8,
        )
        text = resp.choices[0].message.content.strip()
        cleaned = clean_and_parse_json_block(text)
        world = cleaned

        required_keys = ["characters", "genre", "setting", "object"]
        if not all(k in world for k in required_keys):
            missing_keys = [k for k in required_keys if k not in world]
            logger.error(
                f"Generated world JSON missing keys: {missing_keys}. Raw text: {text}"
            )
            raise RuntimeError(
                f"World JSON validation failed: Missing keys {missing_keys}"
            )

        if (
            not isinstance(world["characters"], list)
            or len(world["characters"]) != num_characters
        ):
            logger.error(
                f"Generated world JSON 'characters' structure error or wrong count. Expected {num_characters}. Raw text: {text}"
            )
            raise RuntimeError(
                "World JSON validation failed: Invalid 'characters' structure."
            )
        if not isinstance(world["object"], str) or not world["object"].endswith("s"):
            raise RuntimeError("World JSON 'object' must be a plural noun string.")
        logger.debug(f"Generated object: {world['object']}")
        return world

    except json.JSONDecodeError:
        raise RuntimeError("JSON parse failed")
    except Exception as e:
        logger.error(f"Error processing generated world JSON: {e}. Raw text: {text}")
        raise RuntimeError("World JSON processing failed.") from e


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
NUMBER_WORDS_PATTERN = (
    r"\b(?:(minus|negative)\s+)?("
    + "|".join(re.escape(k) for k in EXPANDED_NUMBER_WORDS_DICT.keys())
    + r")\b"
)
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
            logger.debug(f"Validation FAIL (Rule 1): Missing required numbers: {missing_expected}")
            return False

        # Rule 2: Check if any forbidden atoms are present
        found_forbidden = found_numbers & forbidden_atoms
        if found_forbidden:
            truly_forbidden_found = found_forbidden - allowed_atoms
            if truly_forbidden_found:
                 logger.debug(f"Validation FAIL (Rule 2): Found forbidden numbers: {truly_forbidden_found}")
                 return False
            else:
                 logger.debug(f"Validator INFO: Found number(s) {found_forbidden} that were technically forbidden but also required. Allowing.")

        # Identify numbers found that were NOT explicitly required for this beat
        unexpected_found = found_numbers - allowed_atoms
        logger.debug(f"Validator Unexpected Found (Before Rule 3/4/5): {unexpected_found}")

        # Check unexpected numbers against conditional allowances
        truly_disallowed_extras = set()
        for extra_num in unexpected_found:
            is_allowed_one = (extra_num == 1)
            is_allowed_count = (extra_num == operand_count and extra_num not in forbidden_atoms)
            # --- OPTION A Check (Include if using Option A validator) ---
            is_allowed_small = (extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS and extra_num not in forbidden_atoms)
            # --- END OPTION A Check ---

            # Adjust the condition based on whether you include is_allowed_small
            # If using Option A validator:
            if not (is_allowed_one or is_allowed_count or is_allowed_small):
            # If using original strict validator (no implicit small numbers):
            # if not (is_allowed_one or is_allowed_count):
                 if extra_num not in forbidden_atoms:
                     truly_disallowed_extras.add(extra_num)

        # Final check for disallowed extras
        if truly_disallowed_extras:
            logger.debug(f"Validation FAIL (Strict Rule): Found unexpected/disallowed numbers: {truly_disallowed_extras}")
            logger.debug(f"--> Context: Allowed={allowed_atoms}, Forbidden={forbidden_atoms}, OperandCount={operand_count}, Found={found_numbers}")
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
    max_name_tokens: int = 30,
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

    try:
        # Apply retry logic here if not using decorator
        resp = _chat_completion_call(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_name_tokens,
            temperature=0.75, # Keep some creativity
        )
        raw_candidate = resp.choices[0].message.content

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

# --- Narrative Generation with REVISED-REVISED Strict Checks ---
class BeatGenerationError(Exception):
    """Raised when a story beat fails to generate, aborting entire narrative."""
    pass

# --- Narrative Generation with REVISED Parent Operator Prompting ---

# --- Narrative Generation with REVISED Parent Operator Prompting ---

def _generate_narrative_recursive(
    node: Node,
    context: "GenerationContext",
    is_root: bool,  # Flag to know if this is the root node call
):
    """
    Recursive helper for POST-ORDER strict narrative generation.
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
        if context.tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
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
        if context.tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(f"Token limit reached after processing child of operator {getattr(node, 'op', 'Atom')}. Stopping further generation for this branch.")
            return

    # --- Process Current Operator Node (After All Children Have Been Processed) ---
    logger.debug(f"Finished processing children for operator {getattr(node, 'op', 'Atom')} ({owner_name}). Now processing node itself.")
    if is_root:
        logger.info(f"ROOT NODE ({node.op}): Starting beat generation. Current tokens: {context.tokens_used}/{MAX_TOTAL_TOKENS}")

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
                 f"But for a reason you concoct, they can only collect the cache containing the middle quantity (median) of {primary_object}s."
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
                 f"They collect all these {primary_object}, but for some reason you concoct, they are forced to give away their haul so that they are left only with a quantity equal to the average of the initial amounts (e.g., they find groups of 3, 4, and 5 {primary_object}, but can only walk away with 4 {primary_object})."
             )
        elif node.op == "SM":
             scene_preamble = (
                 f"In this stage, the characters discover separate caches or groups containing "
                 f"{object_list_str_for_preamble} {primary_object} respectively. "
                 f"They collect all these {primary_object}, combining their haul. However, for a reason you concoct, "
                 f"they are forced to give away most of their collection, leaving them only with a quantity equal to the final digit of the total number gathered (e.g., they collect a total of 27 {primary_object}, and can only walk away with 7 {primary_object})."
             )

    # --- REFINED: Dynamic Ownership Instruction Detail for Post-Order ---

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
            object_list_str_for_preamble = ", ".join(items[:-1]) + ", and " + items[-1]

    scene_preamble = "" # Default empty
    if direct_atom_values:
        # (Keep the existing if/elif block for SUM, MED, MIN, MAX, AVG, SM based on node.op)
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
                 f"But for a reason you concoct, they can only collect the cache containing the middle quantity (median) of {primary_object}s."
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
                 f"They collect all these {primary_object}, but for some reason you concoct, they are forced to give away their haul so that they are left only with a quantity equal to the average of the initial amounts (e.g., they find groups of 3, 4, and 5 {primary_object}, but can only walk away with 4 {primary_object})."
             )
        elif node.op == "SM":
             scene_preamble = (
                 f"In this stage, the characters discover separate caches or groups containing "
                 f"{object_list_str_for_preamble} {primary_object} respectively. "
                 f"They collect all these {primary_object}, combining their haul. However, for a reason you concoct, "
                 f"they are forced to give away most of their collection, leaving them only with a quantity equal to the final digit of the total number gathered (e.g., they collect a total of 27 {primary_object}, and can only walk away with 7 {primary_object})."
             )

    # (This logic remains the same, depends on child_owner_names collected earlier and direct_atom_values)
    has_operator_children = bool(child_owner_names)
    has_direct_atom_children = bool(direct_atom_values)
    ownership_instruction_detail = ""
    input_description_parts = [] # Describe the inputs for this step

    # Describe inputs from previous steps (child operations)
    if has_operator_children:
        child_names_str = ', '.join(f"'{name}'" for name in child_owner_names)
        input_description_parts.append(f"the outcome(s) from previous step(s) represented conceptually by {child_names_str}") # Now works
        ownership_instruction_detail += (
            f"REMEMBER: The specific numeric values resulting from {child_names_str} were determined in *earlier* scenes and are FORBIDDEN here. " # Line 1325 (Now works)
            f"Refer to them conceptually or by name only.\n"
        )

    # Describe inputs from direct atoms discovered in *this* step
    if has_direct_atom_children:
        direct_atom_words = [num_to_words(a) for a in sorted(direct_atom_values)]
        direct_values_str = ', '.join(f"{w} ({v})" for w, v in zip(direct_atom_words, sorted(direct_atom_values)))
        input_description_parts.append(f"newly discovered quantities: {direct_values_str}")
    # Combine input descriptions
    if not input_description_parts:
        inputs_str = "inputs determined entirely by context (e.g., selecting between previous outcomes)" # Fallback for MIN/MAX/MED on OpNodes only
    elif len(input_description_parts) == 1:
        inputs_str = input_description_parts[0]
    else:
        inputs_str = " and ".join(input_description_parts)

    # Construct the main instruction
    ownership_instruction_detail = ( # Use '=' to initialize the string for this beat
        f"This scene focuses on resolving the step named '{owner_name}'. "
    )

    # STEP 2: Append specific instruction based on condition using if/else
    if node.op in ("SUM", "AVG", "SM") and has_operator_children and has_direct_atom_children:
         ownership_instruction_detail += ( # Use '+=' to append to the string
             f"Narratively describe the action of **combining** {inputs_str} "
             f"and then applying the specific rule '{op_label}' to that combined set "
             f"to determine the final state, value, or significance associated with '{owner_name}'. "
             f"Make it clear that both the previous outcome ('{child_owner_names[0]}') and the new quantities ({direct_values_str}) are part of the input for the '{op_label}' rule."
         )
    else:
        # Default instruction for other ops or single input types
        ownership_instruction_detail += ( # Use '+=' to append
             f"Narratively describe how applying the rule '{op_label}' to {inputs_str} "
             f"determines the final state, value, or significance associated with '{owner_name}'. "
             f"Focus on the *action* of applying the rule in this specific scene."
        )

    # STEP 3: Append the reminder about forbidden values (now correctly placed after the if/else)
    if has_operator_children:
        child_names_str = ', '.join(f"'{name}'" for name in child_owner_names)
        ownership_instruction_detail += ( # Use '+=' to append
            f"\nREMEMBER: The specific numeric values resulting from {child_names_str} were determined in *earlier* scenes and are FORBIDDEN here. "
            f"Refer to them conceptually or by name only when describing the combination."
        )

    # Decide header and mode
    task_header = "Final Discovery" if is_root else "Discovery Step"
    beat_mode = (
        f"{world['genre']}, {'concluding' if is_root else 'continuous'} scene about "
        f"{primary_object}"
    )

    # Decide header and mode
    task_header = "Final Discovery" if is_root else "Discovery Step"
    beat_mode = (
        f"{world['genre']}, {'concluding' if is_root else 'continuous'} scene about "
        f"{primary_object}"
    )

    # Build task_body using the potentially empty scene_preamble and the dynamic ownership detail
    task_body_parts = []
    if scene_preamble:
        task_body_parts.append(scene_preamble)
    task_body_parts.append(ownership_instruction_detail)
    task_body = "\n\n".join(task_body_parts)

    # --- Prepend Comprehensive Few-Shot Examples ---
    formatted_examples = "--- Examples of How to Follow (and Fail) the Number Rules ---\n\n"
    # Select N examples (e.g., 1 or 2 to save tokens)
    num_examples_to_use = min(len(FEW_SHOT_EXAMPLES_STRICT), 3) # Use up to 3 examples now
    selected_examples = random.sample(FEW_SHOT_EXAMPLES_STRICT, num_examples_to_use) # Use the new list name

    for i, (rules, good_output, bad_output, reason) in enumerate(selected_examples): # Unpack all four parts
        formatted_examples += f"--- Example {i+1} ---\n"
        formatted_examples += f"Hypothetical Rules Provided:\n{rules}\n\n"
        formatted_examples += f"GOOD Narrative Output (Follows Rules):\n{good_output}\n\n"
        formatted_examples += f"BAD Narrative Output (Violates Rules):\n{bad_output}\n\n" # Include the bad output
        formatted_examples += f"REASONING FOR BAD Output FAILURE:\n{reason}\n" # Include the reasoning
        formatted_examples += f"--- End Example {i+1} ---\n\n"

    # --- Build the final prompt ---
    follow_example_instruction = (
        "YOUR TASK:\n"
        "Generate the *next* narrative scene based on the 'Previous Scene Snippet' and the 'Discovery Step' / 'Final Discovery' instructions below.\n"
        "**CRITICAL:** You MUST follow the specific **ULTRA-STRICT NUMBER RULES** provided below for THIS scene *exactly* like the GOOD Narrative Output examples above. Avoid making the mistakes shown in the BAD Narrative Output examples. Ensure ALL required numbers are present, NO forbidden numbers are present, and ABSOLUTELY NO OTHER numbers are included unless explicitly allowed by the 'MAY use' clause or are small numbers (0-10) that are NOT forbidden.\n\n" # Strengthened instruction
        "--- Your Current Task Starts Below ---\n\n"
    )

    # Assemble the final prompt (remains the same structure)
    beat_prompt = (
        formatted_examples # Add the examples first
        + follow_example_instruction # Add the instruction to follow them
        + BASE_BEAT_TEMPLATE.substitute( # Then add the main template for the *current* task
            beat_mode=beat_mode, # <<< ADD THIS LINE BACK
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
        beat_prompt
    )

    # --- Token Budget Check ---
    estimated_prompt_tokens = len(encoder.encode(beat_prompt))
    if would_exceed_budget(
        context.tokens_used, # This includes tokens from ALL children/padding
        estimated_prompt_tokens + MAX_BEAT_TOKENS,
        MAX_TOTAL_TOKENS,
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
                f"System: {system_prompt}\nUser: {beat_prompt}\n\nGeneration:\n{candidate_text}"
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
    while add_padding and context.tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < context.max_pad_paragraphs:
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

        log_prompt(f"Padding Prompt {pad_count} after {node.op} ({owner_name})", padding_prompt)

        # --- Token Budget Check for Padding ---
        estimated_pad_prompt_tokens = len(encoder.encode(padding_prompt))
        if would_exceed_budget(
            context.tokens_used,
            estimated_pad_prompt_tokens + MAX_PAD_TOKENS,
            MAX_TOTAL_TOKENS,
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
        )

        if padding_text:
            ptoks = len(encoder.encode(padding_text))
            if context.tokens_used + ptoks <= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
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
    ast: Node, world: dict, config: Config, encoder, p_inflect, logger
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
    # This is crucial because post-order relies on child values being ready
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

    if intro_text and len(encoder.encode(intro_text)) <= MAX_TOTAL_TOKENS:
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
    judge_instructions = JUDGE_INSTRUCTIONS
    final_prompt = narrative_body + question + judge_instructions

    # Final validation check (optional but recommended)
    final_token_count = len(encoder.encode(final_prompt))
    if final_token_count > MAX_TOTAL_TOKENS:
         logger.warning(f"Final generated prompt ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({MAX_TOTAL_TOKENS}). Truncation might occur.")
         # Depending on requirements, you might return None or the truncated prompt

    logger.info(f"Successfully generated narrative prompt using POST-ORDER traversal. Final estimated tokens: {context.tokens_used} (body), {final_token_count} (full prompt)")
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
            ast, world_info, config, encoder, p_inflect, logger
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
                "prompt_shot_count": config.PROMPT_SHOT_COUNT,
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
    num_samples: int = config.NUM_SAMPLES_TO_GENERATE,
    output_file: str = OUTPUT_FILENAME,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
):
    """Generate samples with strict validation."""
    logger.info(
        f"Script started. Generating {num_samples} samples using up to {max_workers} workers with Pre-Order Strict Validation (v5g - Stricter Validator)."
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
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Dataset output file: {output_file}")
    logging.shutdown()


if __name__ == "__main__":
    main(
        num_samples=config.NUM_SAMPLES_TO_GENERATE, # Pass config explicitly if desired
        output_file=OUTPUT_FILENAME,               # Or rely on defaults in main() signature
        max_workers=config.DEFAULT_MAX_WORKERS
    )

    total_count   = config.NUM_SAMPLES_TO_GENERATE
    success_count = len(samples)
    success_rate  = (success_count / total_count * 100) if total_count else 0
    logger.info(f"Generation success rate: {success_rate:.2f}% ({success_count}/{total_count})")
