class BeatGenerationError(Exception):
    """Raised when a story beat fails to generate, aborting entire narrative."""
    pass
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

# Ordinals to ignore when extracting numbers
ORDINAL_WORDS_TO_IGNORE = {
    "first", "second", "third",
}


import tiktoken
import openai


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
    "You are a $beat_mode storyteller.\n"
    "Characters: $characters\n"
    "Setting: $setting\n"
    'Previous Scene Snippet: "...$snippet"\n\n'
    "--- $task_header ---\n"
    "$task_body\n\n"
    "$ultra_strict_instruction\n\n"
    "Output only the narrative text for this scene, without titles or headings."
)

#  Configuration Constants 

# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 2  # How many samples to generate in one run
OUTPUT_FILENAME = "gpt4.5-verbose_listops_dataset_ultra_strict_v4.jsonl"  # Output file for the dataset
DEFAULT_MAX_WORKERS = 8  # Default number of parallel threads for batch generation

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

SHOT_EXAMPLES = {0: ""}
for idx, txt in enumerate(EXAMPLE_TEXTS, start=1):
    SHOT_EXAMPLES[idx] = f"<Prompt Shot>\n{txt}\n</Prompt Shot>\n"

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
CLEAR_LOGS_ON_START = True  # If True, delete existing logs in LOG_DIR on startup


# --- Dataclasses ---
@dataclass
class Config:
    NUM_SAMPLES_TO_GENERATE: int = NUM_SAMPLES_TO_GENERATE
    DEFAULT_MAX_WORKERS: int = DEFAULT_MAX_WORKERS
    DEFAULT_MAX_TOTAL_TOKENS: int = 5000
    DEFAULT_MAX_BEAT_TOKENS: int = 500
    DEFAULT_MAX_PAD_TOKENS: int = 500
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


MODEL = "gpt-4.1"
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


def retry_api_call(func: Callable):
    """Decorator to retry OpenAI API calls on failure with exponential backoff."""

    def wrapper(*args, **kwargs):
        delay = config.RETRY_INITIAL_DELAY
        for attempt in range(1, config.RETRY_MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"API call error attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}: {e}"
                )
            if attempt == config.RETRY_MAX_ATTEMPTS:
                logger.error("Max retry attempts reached.")
                raise
            time.sleep(delay)
            delay *= 2

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
                max_tokens=max_tokens,
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
            max_tokens=2000,
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
    except Exception as e:
        logger.error(f"World gen API call or processing error: {e}")
        raise RuntimeError("World gen failed") from e


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
            word = p_inflect.number_to_words(i)
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
    """Extracts integers (digits and basic words, handles 'minus'/'negative')."""
    if not text:
        return set()
    found_numbers = set()
    for match in DIGIT_REGEX.finditer(text):
        try:
            value = int(match.group(0))
        except ValueError:
            continue
        # Only allow 'one' in word form; skip digit '1'
        if value == 1:
            continue
        found_numbers.add(value)
    # Verification Note: Hyphenated numbers (e.g., "twenty-three") are handled
    # correctly as inflect generates them with hyphens, and the regex is built
    # from these hyphenated keys in EXPANDED_NUMBER_WORDS_DICT.
    for match in NUMBER_WORDS_REGEX.finditer(text):
        sign_word = match.group(1)
        number_word = match.group(2).lower()
        # <<<--- START ORDINAL CHECK ---<<<
        if number_word in ORDINAL_WORDS_TO_IGNORE:
            logger.debug(f"Ignoring ordinal word: '{number_word}'")
            continue  # Skip this match for ordinals
        # >>>--- END ORDINAL CHECK --->>>
        value = EXPANDED_NUMBER_WORDS_DICT.get(
            number_word
        )  # Use the expanded dictionary
        if value is not None:
            if sign_word and value != 0:
                value = -value
            found_numbers.add(value)
        else:
            logger.warning(f"Word '{number_word}' found by regex but not in dict.")
    return found_numbers


# --- Factory for number validation ---
def make_number_validator(
    allowed_atoms: Set[int],
    forbidden_atoms: Set[int],
    operand_count: int # <-- ADD THIS ARGUMENT
) -> Callable[[str], bool]:
    """
    Return a validator function based on new rules, including operand count.
    """
    logger.debug(f"Creating validator with: Allowed={allowed_atoms}, Forbidden={forbidden_atoms}, OperandCount={operand_count}")

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
        logger.debug(f"Validation FAIL (Rule 2): Found forbidden numbers: {found_forbidden}")
        return False

    # Identify numbers found that were NOT explicitly required
    unexpected_found = found_numbers - allowed_atoms
    logger.debug(f"Validator Unexpected Found (Before Rule 3/4): {unexpected_found}")

    # Check unexpected numbers against conditional allowances (Rules 3 & 4)
    truly_disallowed_extras = set()
    for extra_num in unexpected_found:
        # Rule 4 Check: Is it the number 1?
        is_allowed_one = (extra_num == 1)

        # Rule 3 Check: Is it the operand count?
        # Ensure operand_count itself isn't forbidden (Rule 2 already checked this, but good practice)
        is_allowed_count = (extra_num == operand_count and extra_num not in forbidden_atoms)

        # If it's NOT allowed by Rule 3 or 4, add to disallowed set
        if not (is_allowed_one or is_allowed_count):
             # Double-check it wasn't forbidden (Rule 2 check is primary, this is belt-and-suspenders)
             if extra_num not in forbidden_atoms:
                 truly_disallowed_extras.add(extra_num)
             # If it was forbidden, Rule 2 already caught it, no need to add here.

    # Rule 5: If any truly disallowed extras remain, fail
    if truly_disallowed_extras:
        logger.debug(f"Validation FAIL (Rule 5): Found unexpected/disallowed numbers: {truly_disallowed_extras}")
        logger.debug(f"--> Context: Allowed={allowed_atoms}, Forbidden={forbidden_atoms}, OperandCount={operand_count}, Found={found_numbers}")
        return False

    # If all checks pass
    logger.debug(f"Validation PASS")
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


# --- ADDED FOR PHASE 4b: LLM Naming Function ---
@retry_api_call
def generate_owner_name_with_llm(
    world_info: dict,
    op_node: OpNode,
    child_owner_names: list[str],
    max_name_tokens: int = 30,
) -> str | None:
    """
    Uses an LLM call to generate a creative, thematic name for an OpNode's owner entity.
    Relies on @retry_api_call for retries.
    Returns the generated name or None if generation fails after retries.
    --- REMOVED stop_sequences from API call, added post-processing ---
    """

    op_label = OP_LABELS.get(op_node.op, op_node.op)
    genre = world_info.get("genre", "unknown genre")
    setting = world_info.get("setting", "unknown setting")

    # --- Restore World Info - Keep it Concise ---
    characters_sample = world_info.get("characters", [])[:3]
    characters_str = json.dumps(
        [{"name": c.get("name", "N/A")} for c in characters_sample]
    )
    concepts_sample = world_info.get("entity_concepts", [])[:5]
    concepts_str = json.dumps(concepts_sample)
    # --- End World Info ---

    # --- Restore Child Context - Keep it Concise ---
    children_context = ""
    if child_owner_names:
        display_child_names = child_owner_names[:3]
        children_context = (
            f"This entity is formed from components related to: "
            f"{', '.join(display_child_names)}"
            f"{' and others...' if len(child_owner_names) > 3 else '.'}"
        )
    # --- End Child Context ---

    # --- Restore Detailed Prompt ---
    system_prompt = (
        "You are a creative assistant specializing in generating short, evocative, thematic names "
        "for concepts within a fictional narrative. Be concise. Output only the name."
    )
    user_prompt = (
        f"Fictional World Context:\n"
        f"- Genre: {genre}\n"
        f"- Setting: {setting}\n"
        f"- Sample Characters: {characters_str}\n"
        f"- Sample Thematic Concepts: {concepts_str}\n\n"
        f"Task: Generate a short (2-5 words), creative, and thematic name for an entity, "
        f"collection, process, or concept within this world.\n\n"
        f"Details about the entity to name:\n"
        f"- It represents the outcome of an operation conceptually similar to finding the '{op_label}'.\n"
        f"- {children_context}\n\n"
        f"Instructions:\n"
        f"- The name should fit the {genre} genre and {setting} setting.\n"
        f"- Make it sound like a specific thing/idea in the story (e.g., 'The Oracle's Final Whisper', 'Sector Gamma Scan Results', 'Kaelen's Calculated Risk').\n"
        f"- AVOID using the exact operation word (like '{op_node.op}' or '{op_label}').\n"
        f"- Output only the generated name itself, with no quotes, labels, explanations, or introductory phrases like 'Here is a name:'."
    )
    # --- End Detailed Prompt ---

    prompt_log_header = f"--- LLM Owner Naming Prompt (Op: {op_node.op}, Attempting Call - No Stop Sequences) ---"
    prompt_content_for_log = f"System: {system_prompt}\nUser: {user_prompt}"
    logger.debug(
        f"Attempting LLM naming call for {op_node.op} with prompt:\n{prompt_content_for_log}"
    )
    logger.debug(f"Prompt length: {len(prompt_content_for_log)}")

    try:
        resp = _chat_completion_call(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_name_tokens,
            temperature=0.75,
        )
        raw_candidate = resp.choices[0].message.content

        # --- Post-processing to simulate stop sequences ---
        stop_chars = ["\n", ".", ","]
        first_stop_index = len(raw_candidate)
        for char in stop_chars:
            idx = raw_candidate.find(char)
            if idx != -1 and idx < first_stop_index:
                first_stop_index = idx
        processed_candidate = raw_candidate[:first_stop_index]
        # --- End Post-processing ---

        candidate = processed_candidate.strip()
        candidate = re.sub(
            r"^(?:Here is a name:|Name:|Entity Name:|\"|\')",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = re.sub(r"(?:\"|\')$", "", candidate).strip()

        if not candidate or candidate.lower().startswith(
            ("i cannot", "i'm sorry", "i am unable")
        ):
            logger.warning(
                f"LLM Naming: Invalid/refusal response content (raw: '{raw_candidate}')"
            )
            return None
        if len(candidate) > max_name_tokens * 6:
            logger.warning(
                f"LLM Naming: Name potentially too long after processing: '{candidate}' (raw: '{raw_candidate}')"
            )
            return None

        logger.debug(
            f"LLM generated owner name: '{candidate}' (processed from raw: '{raw_candidate}')"
        )
        return candidate

    except Exception as e:
        logger.error(
            f"LLM Naming API Error: {e}. Prompt that failed:\n{prompt_content_for_log}"
        )
        raise


# --- END PHASE 4b Function ---

# --- Narrative Generation with REVISED-REVISED Strict Checks ---


def _generate_narrative_recursive(
    node: Node,
    world: dict,
    config: Config,
    encoder,
    p_inflect,
    logger,
    owner_map: dict,
    all_atoms: set,
    introduced_atoms: Set[int],
    scenes: list,
    tokens_used: int,
    last_scene_text: str,
    beat_counter: dict,
    is_root: bool, # Flag to know if this is the root node call
    max_pad_paragraphs: int = 2,
):
    """
    Recursive helper for POST-ORDER strict narrative generation.
    Processes children first, then the current node.
    Returns (scenes, tokens_used, last_scene_text).
    """
    node_id = id(node)
    owner_name = owner_map.get(node_id, f"the_unnamed_{node.op}_entity" if isinstance(node, OpNode) else "atom")
    logger.debug(f"_generate_narrative_recursive (POST-ORDER): processing node {getattr(node, 'op', 'Atom')} with owner '{owner_name}'")

    # --- Base case: Atom ---
    if isinstance(node, Atom):
        # Atoms don't generate narrative beats themselves in post-order.
        # Their values are introduced by their parent OpNode.
        logger.debug(f"Node is Atom ({node.n}), returning.")
        return scenes, tokens_used, last_scene_text

    # --- Recursive Step: Process Children First (Post-Order) ---
    child_owner_names = [] # Collect owner names of direct OpNode children
    for child in node.children:
        scenes, tokens_used, last_scene_text = _generate_narrative_recursive(
            child,
            world,
            config,
            encoder,
            p_inflect,
            logger,
            owner_map,
            all_atoms,
            introduced_atoms,
            scenes,
            tokens_used,
            last_scene_text,
            beat_counter,
            is_root=False,
            max_pad_paragraphs=max_pad_paragraphs,
        )
        if isinstance(child, OpNode):
            child_owner_names.append(owner_map.get(id(child), f"unnamed_{child.op}_result"))
        if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(f"Token limit reached after processing child of operator {getattr(node, 'op', 'Atom')}. Stopping further generation for this branch.")
            return scenes, tokens_used, last_scene_text

    # --- Process Current Operator Node (After Children) ---
    logger.debug(f"Finished processing children for operator {getattr(node, 'op', 'Atom')}. Now processing node itself.")
    # Track and log which beat out of total is being generated
    beat_counter["current"] += 1
    logger.info(f"Generating beat {beat_counter['current']}/{beat_counter['total']} for operator {node.op} ({owner_name})")
    op_label = OP_LABELS.get(node.op, node.op)
    # Identify direct atom children and atoms to introduce in *this* beat
    direct_atom_children = [c for c in node.children if isinstance(c, Atom)]
    operand_count = len(direct_atom_children)
    logger.debug(f"Calculated operand_count for node {node.op}: {operand_count}")
    direct_atom_values = {a.n for a in direct_atom_children}

    # Allow descriptive counts (e.g., “three caches”, “four compartments”) and list-start “one”
    count_val = len(direct_atom_values)

    # For SUM, AVG, and SM, include computed result, the count, and “one”
    if node.op in ("SUM", "AVG", "SM"):
        result_val = node.value if getattr(node, "value", None) is not None else eval_node(node)
        allowed_values = direct_atom_values | {result_val, count_val, 1}
        # <<<--- START VERIFICATION LOG ---<<<
        logger.debug(f"Verification (Op: {node.op}): Result {result_val} included in allowed_values: {allowed_values}")
        # >>>--- END VERIFICATION LOG --->>>
        # Verification Note: Confirmed that for SUM, AVG, SM nodes, the calculated
        # node.value (result_val) is correctly added to the set of allowed numbers
        # for the current beat's validation.
    else:
        # For MED, MIN, MAX, include direct values plus the count and “one”
        allowed_values = direct_atom_values | {count_val, 1}

    atoms_to_introduce_this_beat = allowed_values - introduced_atoms
    forbidden_atoms_for_prompt = introduced_atoms
    # --- Semantic Layer: primary object concept ---
    primary_object = world["object"]

    # Prepare operand descriptions
    operand_desc_parts = []
    if child_owner_names:
        operand_desc_parts.append(f"results from: {', '.join(child_owner_names)}")
    if direct_atom_values:
        atom_words = [p_inflect.number_to_words(a) for a in sorted(direct_atom_values)]
        operand_desc_parts.append(f"direct values: {', '.join(atom_words)}")
    operand_context_str = "; ".join(operand_desc_parts) if operand_desc_parts else "previously established context"

    # --- Demarcate object counts vs computed result for inclusion ---
    if direct_atom_values:
        items = [f"{p_inflect.number_to_words(x)} ({x})" for x in sorted(direct_atom_values)]
        if len(items) == 1:
            object_list_str = items[0]
        elif len(items) == 2:
            object_list_str = " and ".join(items)
        else:
            object_list_str = ", ".join(items[:-1]) + ", and " + items[-1]
    else:
        object_list_str = "None"
    operation_result_list_str = None
    if node.op in ("SUM", "AVG", "SM"):
        result_val = node.value if getattr(node, "value", None) is not None else eval_node(node)
        operation_result_list_str = f"{p_inflect.number_to_words(result_val)} ({result_val})"
    # Prepare number rule strings
    if forbidden_atoms_for_prompt:
        must_avoid_str = ", ".join(f"{p_inflect.number_to_words(x)} ({x})" for x in sorted(forbidden_atoms_for_prompt))
    else:
        must_avoid_str = "None"

    # For inclusion rules, objects first, then result if present
    if object_list_str:
        objects_rule = f"Numbers you MUST include: {object_list_str}"
    else:
        raise BeatGenerationError(f"No operand atoms available for operator {node.op}")
    if operation_result_list_str:
        result_rule = f", and {operation_result_list_str}, which is the number of {object_list_str} they end up with."
    else:
        result_rule = ""
    try:
        operand_count_word = p_inflect.number_to_words(operand_count) if p_inflect else str(operand_count)
    except Exception: # Handle potential inflect errors
         operand_count_word = str(operand_count)

    # Combine required operands and result (if applicable) into one string for the prompt
    must_include_combined = f"{object_list_str}{result_rule}" # result_rule is empty if no result needed

    ultra_strict_instruction = (
        "**STRICT NUMBER RULE:**\n"
        f"*   You MUST include the following numbers (use digits): {must_include_combined}.\n"
        f"*   You MUST NOT mention any numbers from this forbidden list: {must_avoid_str}.\n"
        f"*   You MAY use the number {operand_count} ('{operand_count_word}', the count of items being considered) and the number 1 ('one') for natural narrative flow, *unless* they are in the forbidden list above.\n"
        "*   NO OTHER numbers besides these are allowed (no intermediate calculations, no unrelated values)."
    )

    is_final_op = is_root

    ownership_instruction_detail = (
         f"This part of the story resolves the entity or concept known as '{owner_name}'. "
         f"Its final state or significance is determined by applying a rule ({op_label}) to its constituent parts: {operand_context_str}. "
         f"Describe narratively how '{owner_name}' is finalized, evaluated, or what consequences arise from its state. "
         f"Refer to the constituent parts implicitly or by their names if applicable."
    )

    # --- Operator-specific semantic templates ---
    if node.op == "SUM":
        scene_preamble = (
            f"In this stage, the characters discover "
            f"{object_list_str} {primary_object} in separate locations, "
            f"and they collect all of them."
        )
    elif node.op == "MED":
        scene_preamble = (
            f"In this stage, the characters discover {object_list_str} {primary_object}. "
            f"But for a reason you concoct, they can only collect the middle amount (median) of {primary_object}s."
        )
    elif node.op == "MIN":
        scene_preamble = (
            f"In this stage, the characters discover {object_list_str} {primary_object} lying or stored in different areas. "
            f"But for a reason you concoct, they can only access or retrieve from the area or storage space that contains the smallest quantity (MIN) of {primary_object} for collection."
        )
    elif node.op == "MAX":
        scene_preamble = (
            f"In this stage, the characters discover {object_list_str} {primary_object} lying or stored in different areas. "
            f"But for a reason you concoct, they can only access or retrieve from the area or storage space that contains the largest quantity (MAX) of {primary_object} for collection."
        )
    elif node.op == "AVG":
        scene_preamble = (
            f"In this stage, the characters discover "
            f"{object_list_str} {primary_object} in separate locations, "
            f"They collect them all, but for some reason you concoct, they are forced to give away their haul so that they are left only with the average of these objects. (e.g., they collect 3, 4, and 5 {primary_object}, but can only walk away with 4)."
        )
    elif node.op == "SM":
        scene_preamble = (
            f"In this stage, the characters discover "
            f"{object_list_str} {primary_object} in separate locations, "
            f"They collect them all, but for some reason you concoct, they are forced to give away their haul so that they are left only with its final digit (e.g., they collect a total of 27, and can only walk away with 7)."
        )
    # Decide header and mode
    task_header = "Final Discovery" if is_final_op else "Discovery Step"
    beat_mode = (
        f"{world['genre']}, {'concluding' if is_final_op else 'continuous'} scene about "
        f"{primary_object}"
    )
    # Build task_body
    task_body = (
        f"{scene_preamble}\n\n"
        f"{ownership_instruction_detail}"
    )

    prompt_examples = (
         "Example 1:\n" +
         "**ULTRA-STRICT NUMBER RULE:**\n" +
         "*   MUST ONLY INCLUDE: Only narratively include the following numbers: [eight (8), two (2)].\n" +
         "Valid Example:\n"
         "\"In the moonlit chamber, a warrior discovered a tablet etched with eight symbols and another with two and claimed it.\"\n"
         "Invalid Example:\n"
         "\"In the moonlit chamber, two warriors discovered a tablet etched with eight symbols and another with two and claimed it.\"\n"

         "Example 2:\n" +
         "**ULTRA-STRICT NUMBER RULE:**\n" +
         "*   This padding text MUST contain ZERO numbers. Output only the narrative text for this padding, without titles or headings..\n" +
         "Valid Example:\n"
         "\"The grand archive lay dormant in silence, no carvings marked its ancient doors.\"\n"
         "Invalid Example:\n"
         "\"The grand archive lay dormant, its doors sealed by nine enigmatic glyphs.\"\n"

         "Example 3:\n" +
         "**ULTRA-STRICT NUMBER RULE:**\n" +
         "*   MUST ONLY INCLUDE: Only narratively include the following numbers: [fifteen (15), twenty-five (25), thirty-five (35)].\n" +
         "Valid Example:\n"
         "\"At dawn, fifteen torches blazed along the ramparts, followed by twenty-five banners and thirty-five drums heralding the city’s awakening.\"\n"
         "Invalid Example:\n"
         "\"At dawn, fifteen torches blazed along the ramparts, followed by twenty-five banners, thirty-five drums, and three carrots heralding the city’s awakening.\"\n"
 
         "Example 4 (Failure Case):\n"
         "**REVISED STRICT NUMBER RULE:**\n"
         "*   MUST ONLY INCLUDE: [twenty (20), twenty-four (24)].\n"
         "*   FORBIDDEN: [ten (10)].\n"
         "*   MAY USE: 2 ('two', item count), 1 ('one').\n"
         "*   NO OTHER numbers allowed.\n"
         "Invalid Example Text:\n"
         "\"They found the cache of 20 crystals and the other with 24. Kaelen took three steps back.\" <-- FAIL\n"
         "Reasoning: The number 'three' (3) was mentioned. It was not required (20, 24), not forbidden (10), not the allowed count (2), and not the allowed number one (1). Therefore, it violates the 'NO OTHER numbers' rule.\n"
     )

    beat_prompt = prompt_examples + BASE_BEAT_TEMPLATE.substitute(
        beat_mode=beat_mode,
        characters=json.dumps(world["characters"]),
        setting=world["setting"],
        snippet=last_scene_text[-100:],
        task_header=task_header,
        task_body=task_body,
        ultra_strict_instruction=ultra_strict_instruction,
    )

    log_prompt(
        f"{'=== FINAL' if is_final_op else '=== Intermediate'} Operator Beat Prompt (Op: {node.op}, Owner: {owner_name})",
        beat_prompt
    )

    estimated_prompt_tokens = len(encoder.encode(beat_prompt))
    if would_exceed_budget(
        tokens_used,
        estimated_prompt_tokens + MAX_BEAT_TOKENS,
        MAX_TOTAL_TOKENS,
        SAFETY_MARGIN,
    ):
        logger.warning(f"Approaching token limit before generating operator {node.op} ({owner_name}). Stopping.")
        return scenes, tokens_used, last_scene_text

    validate_beat_numbers = make_number_validator(
        allowed_atoms=atoms_to_introduce_this_beat, # Or 'allowed_values' depending on your variable name
        forbidden_atoms=forbidden_atoms_for_prompt,
        operand_count=operand_count # <-- PASS THE CALCULATED COUNT
    )
    # Also update the call for validate_padding if needed (operand_count would likely be 0)
    

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
                max_tokens=MAX_BEAT_TOKENS,
                temperature=0.7,
            )
            candidate_text = resp.choices[0].message.content.strip()
            log_prompt(
                f"LLM Beat Generation Attempt {attempt} for operator {node.op} ({owner_name})",
                f"System: {system_prompt}\nUser: {beat_prompt}\n\nGeneration:\n{candidate_text}"
            )
            if not candidate_text or candidate_text.lower().startswith(("i cannot", "i'm sorry")):
                reason = "empty or API refusal"
            elif not validate_beat_numbers(candidate_text):
                reason = "inflect validation failed"
            elif atoms_to_introduce_this_beat and not all(
                check_operand_presence(candidate_text, op) for op in atoms_to_introduce_this_beat
            ):
                reason = "operand presence check failed"
            else:
                beat_text = candidate_text
                break
        except Exception as e:
            reason = f"exception: {e}"
        if attempt < config.MAX_BEAT_RETRIES:
            logger.warning(f"Beat {beat_counter['current']}/{beat_counter['total']} retry {attempt}/{config.MAX_BEAT_RETRIES} for operator {node.op} ({owner_name}): {reason}")
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    if not beat_text:
        logger.error(
            f"Operator {node.op} ({owner_name}) failed after {config.MAX_BEAT_RETRIES} attempts. Aborting narrative generation."
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} ({owner_name})"
        )

    btoks = len(encoder.encode(beat_text))
    scenes.append(beat_text)
    tokens_used += btoks
    last_scene_text = beat_text
    introduced_atoms.update(atoms_to_introduce_this_beat)

    pad_count = 0
    add_padding = not is_final_op
    validate_padding = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=introduced_atoms, # Use the up-to-date set
        operand_count=0 # Padding has no operands
    )
    while add_padding and tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
        if would_exceed_budget(tokens_used, 200 + MAX_PAD_TOKENS, MAX_TOTAL_TOKENS, SAFETY_MARGIN):
            break
        padding_prompt = (
            f"You are a {world['genre']} storyteller, writing filler text between scenes.\n"
            f"Characters: {json.dumps(world['characters'])}\nSetting: {world['setting']}\n"
            f'Previous Scene Snippet: "...{last_scene_text[-100:]}"\n\n'
            "--- Task ---\n"
            "Write 1-2 paragraphs of narrative filler or transition.\n"
            "**ULTRA-STRICT ABSOLUTE RULE:** This padding text MUST contain ZERO numbers.\n\n"
            "Output only the narrative text for this padding, without titles or headings."
        )
        pad_result = None
        for pad_attempt in range(1, config.MAX_PAD_RETRIES + 1):
            try:
                resp_pad = _chat_completion_call(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "DO NOT INCLUDE ANY NUMBERS IN YOUR STORY."},
                        {"role": "user", "content": padding_prompt},
                    ],
                    max_tokens=MAX_PAD_TOKENS,
                    temperature=0.7,
                )
                candidate_pad = resp_pad.choices[0].message.content.strip()
                if not candidate_pad or not validate_padding(candidate_pad):
                    continue
                pad_result = candidate_pad
                break
            except Exception:
                pass
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (pad_attempt - 1)))
        if not pad_result:
            break
        ptoks = len(encoder.encode(pad_result))
        if tokens_used + ptoks + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
            break
        scenes.append(pad_result)
        tokens_used += ptoks
        last_scene_text = pad_result
        pad_count += 1

    return scenes, tokens_used, last_scene_text


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

    all_atoms = get_atoms_in_subtree(ast)
    # Owner mapping via postorder...
    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    owner_map = {}
    if config.USE_OWNERSHIP_NARRATIVE:
        if operator_nodes:
            use_llm = config.USE_LLM_NAMING
            characters = world.get("characters", [])
            for i, op_node in enumerate(operator_nodes):
                if not isinstance(op_node, OpNode):
                    continue
                node_id = id(op_node)
                owner_name = None
                if use_llm:
                    child_owner_names = []
                    for child in op_node.children:
                        if isinstance(child, OpNode) and id(child) in owner_map:
                            child_owner_names.append(owner_map[id(child)])
                    try:
                        owner_name = generate_owner_name_with_llm(
                            world, op_node, child_owner_names
                        )
                    except Exception as e:
                        logger.error(f"LLM Naming failed for OpNode {op_node.op}: {e}")
                        owner_name = None
                if not owner_name:
                    primary_object = world["object"]
                    if characters:
                        char_name = random.choice(characters).get("name", "Someone")
                        possessive = (
                            f"{char_name}'"
                            if char_name.endswith("s")
                            else f"{char_name}'s"
                        )
                        owner_name = f"{possessive} {primary_object}"
                    else:
                        owner_name = f"the {primary_object} ({op_node.op} #{i+1})"
                owner_map[node_id] = owner_name
    # Intro scene generation...
    scenes = []
    tokens_used = 0
    intro_prompt = "...ZERO NUMBERS ALLOWED..."
    intro_text = generate_with_retry(
        "You are a storyteller. Do not mention any numbers.",
        "Write a short introductory scene for the story's world, setting, and characters. ZERO NUMBERS ALLOWED.",
        250,
        lambda txt: len(extract_numbers_from_text(txt)) == 0,
        retries=2,
    )
    if intro_text and tokens_used + len(encoder.encode(intro_text)) <= MAX_TOTAL_TOKENS:
        scenes.append(intro_text)
        tokens_used += len(encoder.encode(intro_text))
    last_scene_text = intro_text if intro_text else "The story begins..."
    introduced_atoms_during_generation = set()
    total_beats = len(operator_nodes)
    beat_counter = {"current": 0, "total": total_beats}
    try:
        scenes, tokens_used, last_scene_text = _generate_narrative_recursive(
            ast,
            world,
            config,
            encoder,
            p_inflect,
            logger,
            owner_map,
            all_atoms,
            introduced_atoms_during_generation,
            scenes,
            tokens_used,
            last_scene_text,
            beat_counter,
            is_root=True,
            max_pad_paragraphs=2,
        )
    except BeatGenerationError as e:
        logger.error(f"Narrative generation aborted: {e}")
        return None
    if not scenes:
        return None
    narrative_body = "\n\n".join(scenes).strip()
    found_in_final_body = extract_numbers_from_text(narrative_body)
    unexpected_numbers = found_in_final_body - all_atoms
    missing_atoms = all_atoms - introduced_atoms_during_generation
    if unexpected_numbers or missing_atoms:
        required_numbers = all_atoms
        extras = unexpected_numbers
        missing = missing_atoms
        parts = []
        if extras:
            parts.append(f"Extras: {extras}")
        if missing:
            parts.append(f"Missing: {missing}")
        logger.error(f"Narrative validation FAILED. Required numbers: {required_numbers}. {'; '.join(parts)}")
        logger.error(f"Narrative body: {narrative_body}")
    else:
        logger.info(f"Narrative validation PASSED. Required numbers: {all_atoms}")
    question = "...\n\n---\n\n..."
    judge_instructions = """
    ...
    """
    final_prompt = narrative_body + question + judge_instructions
    logger.info(f"Successfully generated and validated narrative prompt. Total tokens used (estimated): {tokens_used}")
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
    main(max_workers=config.DEFAULT_MAX_WORKERS)
