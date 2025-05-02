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
    "Previous Scene Snippet: \"...$snippet\"\n\n"
    "--- $task_header ---\n"
    "$task_body\n\n"
    "$ultra_strict_instruction\n\n"
    "Output only the narrative text for this scene, without titles or headings."
)

# ─── Configuration Constants ────────────────────────────────────────────────────────────────────────────

# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 2 # How many samples to generate in one run
OUTPUT_FILENAME = "verbose_listops_dataset_ultra_strict_v4.jsonl" # Output file for the dataset
DEFAULT_MAX_WORKERS = 8  # Default number of parallel threads for batch generation

# --- Base configurations ---
# Output Configuration
LOG_DIR = os.path.expanduser("~/verbose_listops_logs")

# --- Few-shot prompt examples ---
EXAMPLE_TEXTS = [
    (
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts: one paying 9 silver pieces, the other only 4. Kaelen chose the lower-paying contract to avoid scrutiny. He then received a standard 5 silver piece bonus for completing the task quickly.\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9"
    ),
    (
        "Example 2:\n"
        "Narrative: “To unlock the ancient vault, the combined energy signature of four power crystals (reading 1, 1, 1, and 1) was required. The locking mechanism, however, only used the final digit of their total combined power.”\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Modulo 10 (final digit) is 4.\n"
        "Answer: 4"
    ),
    (
        "Example 3:\n"
        "Narrative: “Three scouts reported patrol durations of 5, 5, and 5 hours. Standard procedure required calculating their average patrol time, rounded down to the nearest whole hour, for the official logbook entry.”\n"
        "Implicit Calculation: SUM(5, 5, 5) = 15. Count = 3. Average = 15 / 3 = 5. Floor(5) = 5.\n"
        "Answer: 5"
    ),
]

SHOT_EXAMPLES = {0: ""}
for idx, txt in enumerate(EXAMPLE_TEXTS, start=1):
    SHOT_EXAMPLES[idx] = f"<Prompt Shot>\n{txt}\n</Prompt Shot>\n"

# --- AST Random ListOps problem gen params ---
DEFAULT_MAX_OPS = 10 # Max operations (e.g. max, min ,etc.) in a problem
MIN_ARITY = 3 # Min numbers in an operation
DEFAULT_MAX_BRANCH = 6 # Max operations / numbers in an operation

# --- Prompt Logging Helper ---
def log_prompt(header: str, prompt: str, path: str = os.path.join(LOG_DIR, "verbose_listops_prompts.log")):
    """Append a prompt header and text to the prompts log."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{header}\n{prompt}\n\n")
        f.flush()



# --- API retry + logging config ---

LOG_MAX_BYTES = 5 * 1024 * 1024 # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3 # Number of backup log files to keep


# --- Dataclasses ---
@dataclass
class Config:
    NUM_SAMPLES_TO_GENERATE: int = NUM_SAMPLES_TO_GENERATE
    DEFAULT_MAX_WORKERS: int = DEFAULT_MAX_WORKERS
    DEFAULT_MAX_TOTAL_TOKENS: int = 10000
    DEFAULT_MAX_BEAT_TOKENS: int = 750
    DEFAULT_MAX_PAD_TOKENS: int = 750
    MAX_TOKENS_BUFFER: int = 1000
    PROMPT_SHOT_COUNT: int = 3
    RETRY_MAX_ATTEMPTS: int = 5
    RETRY_INITIAL_DELAY: int = 1
    MAX_BEAT_RETRIES: int = 5
    MAX_PAD_RETRIES: int = 3
    ATOM_MIN_VALUE: int = 1
    ATOM_MAX_VALUE: int = 100
    USE_OWNERSHIP_NARRATIVE: bool = True  # Master switch for ownership feature
    USE_LLM_NAMING: bool = True  # If True, use LLM for owner names; else use thematic fallback
config = Config()


MODEL = "gpt-4.5-preview"
MAX_TOTAL_TOKENS = config.DEFAULT_MAX_TOTAL_TOKENS
SAFETY_MARGIN = config.MAX_TOKENS_BUFFER
MAX_BEAT_TOKENS = config.DEFAULT_MAX_BEAT_TOKENS
MAX_PAD_TOKENS = config.DEFAULT_MAX_PAD_TOKENS

# --- Setup Logging ---
os.makedirs(LOG_DIR, exist_ok=True)

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
                logger.warning(f"API call error attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}: {e}")
            if attempt == config.RETRY_MAX_ATTEMPTS:
                logger.error("Max retry attempts reached.")
                raise
            time.sleep(delay)
            delay *= 2
    return wrapper

# --- Budget-check Helper ---
def would_exceed_budget(current: int, upcoming: int, max_total: int, margin: int) -> bool:
    """Return True if adding upcoming tokens would exceed max_total minus margin."""
    return current + upcoming + margin > max_total


# --- Helper for future consolidation of retry loops ---
def generate_with_retry(system_prompt: str, user_prompt: str, max_tokens: int, validate_fn: Callable[[str], bool], retries: int = config.MAX_BEAT_RETRIES):
    """
    Helper to call the Anthropic API with retries and apply a validation function.
    Returns the first candidate text that passes validate_fn, or None if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = _chat_completion_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            candidate = resp.choices[0].message.content.strip()
            if not candidate or candidate.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                logger.warning(f"API refusal on generate_with_retry attempt {attempt}.")
            elif validate_fn(candidate):
                return candidate
        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")
        time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))
    return None


OP_LABELS = {
    "MAX": "largest value", "MIN": "smallest value", "SUM": "sum of all values",
    "MED": "median value", "AVG": "integer-average (floored)", "SM": "sum modulo 10",
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
        super().__init__(op="ATOM", children=[]); self.n = n; self.value = n

@dataclass
class OpNode(Node):
    def __init__(self, op: str, children: list):
        super().__init__(op=op, children=children); self.value = None


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
        arity = random.randint(MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]
        return OpNode(op, children)
    root = helper()
    if isinstance(root, Atom) and max_ops >= 1:
        op = random.choice(ops)
        arity = random.randint(MIN_ARITY, max_branch)
        children = [Atom(random.randint(config.ATOM_MIN_VALUE, config.ATOM_MAX_VALUE)) for _ in range(arity-1)]
        children.append(root)
        random.shuffle(children)
        root = OpNode(op, children)
    return root

def validate_ast(node: Node):
    """Recursively validate that all operators in the AST are supported."""
    if node.op not in OP_LABELS and not isinstance(node, Atom): raise ValueError(f"Invalid operator: {node.op}")
    for c in node.children: validate_ast(c)

def eval_node(node: Node) -> int:
    """Evaluate the AST node recursively."""
    if isinstance(node, Atom):
        if node.value is None: node.value = node.n
        return node.value
    vals = [eval_node(c) for c in node.children]
    if not vals: raise ValueError(f"Operator node {node.op} has no children values.")
    func_map = {
        "MAX": max, "MIN": min, "MED": lambda v: sorted(v)[len(v) // 2],
        "SUM": sum, "SM": lambda v: sum(v) % 10, "AVG": lambda v: sum(v) // len(v) if v else 0,
    }
    try:
        func = func_map[node.op]
        if node.op == "MED" and len(vals) % 2 == 0: logger.warning(f"MED operator with even children ({len(vals)}). Using lower middle.")
        if node.op == "AVG" and not vals: raise ValueError("Cannot calculate average of zero values.")
        node.value = func(vals); return node.value
    except KeyError: raise ValueError(f"Unsupported operator: {node.op}")
    except IndexError as e: logger.error(f"Indexing error eval {node.op} with {vals}: {e}"); raise
    except ZeroDivisionError: raise ValueError(f"Division by zero during AVG for {node.op}")

def postorder(node: Node):
    """Yield nodes in post-order."""
    for c in node.children: yield from postorder(c)
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
    if not isinstance(num_characters, int) or num_characters < 1: raise ValueError("num_characters must be positive int")
    if not isinstance(num_concepts, int) or num_concepts < 1: raise ValueError("num_concepts must be positive int")

    prompt = (
        "You are a creative world-builder.\n"
        f"Generate {num_characters} distinct characters (name, role, quirk). Define a genre and setting.\n"
        f"Also, provide a list of {num_concepts} thematic concepts for groups, collections, or abstract entities relevant to this world (e.g., 'spell scrolls', 'cargo manifests', 'secret dossiers', 'market fluctuations', 'guild ranks'). Keep concepts concise (1-3 words).\n"
        "Output *only* a valid JSON object with NO extra text before or after the JSON structure:\n"
        "{\n"
        "  \"characters\": [{\"name\": \"string\", \"role\": \"string\", \"quirk\": \"string\"}, ...],\n"
        "  \"genre\": \"string\",\n"
        "  \"setting\": \"string\",\n"
        "  \"entity_concepts\": [\"string\", \"string\", ...]\n"
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

        required_keys = ["characters", "genre", "setting", "entity_concepts"]
        if not all(k in world for k in required_keys):
            missing_keys = [k for k in required_keys if k not in world]
            logger.error(f"Generated world JSON missing keys: {missing_keys}. Raw text: {text}")
            raise RuntimeError(f"World JSON validation failed: Missing keys {missing_keys}")

        if not isinstance(world["characters"], list) or len(world["characters"]) != num_characters:
            logger.error(f"Generated world JSON 'characters' structure error or wrong count. Expected {num_characters}. Raw text: {text}")
            raise RuntimeError("World JSON validation failed: Invalid 'characters' structure.")

        if not isinstance(world["entity_concepts"], list) or not world["entity_concepts"]:
            logger.error(f"Generated world JSON 'entity_concepts' structure error or empty list. Raw text: {text}")
            raise RuntimeError("World JSON validation failed: Invalid 'entity_concepts' structure.")

        logger.debug(f"Generated entity concepts: {world['entity_concepts']}")
        return world

    except json.JSONDecodeError:
        raise RuntimeError("JSON parse failed")
    except Exception as e:
        logger.error(f"Error processing generated world JSON: {e}. Raw text: {text}")
        raise RuntimeError("World JSON processing failed.") from e
    except Exception as e:
        logger.error(f"World gen API call or processing error: {e}")
        raise RuntimeError("World gen failed") from e


DIGIT_REGEX = re.compile(r'\b-?\d+\b')
NUMBER_WORDS_DICT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
NUMBER_WORDS_PATTERN = r'\b(?:(minus|negative)\s+)?(' + '|'.join(NUMBER_WORDS_DICT.keys()) + r')\b'
NUMBER_WORDS_REGEX = re.compile(NUMBER_WORDS_PATTERN, re.IGNORECASE)

def extract_numbers_from_text(text: str) -> Set[int]:
    """Extracts integers (digits and basic words, handles 'minus'/'negative')."""
    if not text: return set()
    found_numbers = set()
    for match in DIGIT_REGEX.finditer(text):
        try: found_numbers.add(int(match.group(0)))
        except ValueError: pass
    for match in NUMBER_WORDS_REGEX.finditer(text):
        sign_word = match.group(1); number_word = match.group(2).lower()
        value = NUMBER_WORDS_DICT.get(number_word)
        if value is not None:
            if sign_word and value != 0: value = -value
            found_numbers.add(value)
        else: logger.warning(f"Word '{number_word}' found by regex but not in dict.")
    return found_numbers

# --- Factory for number validation ---
def make_number_validator(allowed_atoms: Set[int]) -> Callable[[str], bool]:
    """
    Return a validator function that returns True if the text contains
    no numbers outside the allowed_atoms set.
    """
    def validate(text: str) -> bool:
        found = extract_numbers_from_text(text)
        return not (found - allowed_atoms)
    return validate

def check_operand_presence(text: str, operand_val: int) -> bool:
    """Checks if an operand is present as a digit or word (using inflect)."""
    if p_inflect is None:
        logger.error("Inflect engine NA, checking only digits."); digit_regex = rf"\b{operand_val}\b"; return bool(re.search(digit_regex, text))
    digit_regex = rf"\b{operand_val}\b"
    if re.search(digit_regex, text): return True
    try:
        if operand_val == 0: op_words = ["zero", "nought"]
        else: op_words = [p_inflect.number_to_words(operand_val)]
        for op_word in op_words:
             word_regex = rf"\b{re.escape(op_word)}\b"
             if re.search(word_regex, text, re.IGNORECASE): return True
    except Exception as e: logger.error(f"Inflect failed for {operand_val}: {e}. Checking digits only."); return bool(re.search(digit_regex, text))
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
    characters_str = json.dumps([{"name": c.get("name", "N/A")} for c in characters_sample])
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
    logger.debug(f"Attempting LLM naming call for {op_node.op} with prompt:\n{prompt_content_for_log}")
    logger.debug(f"Prompt length: {len(prompt_content_for_log)}")

    try:
        resp = _chat_completion_call(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_name_tokens,
            temperature=0.75,
        )
        raw_candidate = resp.choices[0].message.content

        # --- Post-processing to simulate stop sequences ---
        stop_chars = ['\n', '.', ',']
        first_stop_index = len(raw_candidate)
        for char in stop_chars:
            idx = raw_candidate.find(char)
            if idx != -1 and idx < first_stop_index:
                first_stop_index = idx
        processed_candidate = raw_candidate[:first_stop_index]
        # --- End Post-processing ---

        candidate = processed_candidate.strip()
        candidate = re.sub(r'^(?:Here is a name:|Name:|Entity Name:|\"|\')', '', candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r'(?:\"|\')$', '', candidate).strip()

        if not candidate or candidate.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
            logger.warning(f"LLM Naming: Invalid/refusal response content (raw: '{raw_candidate}')")
            return None
        if len(candidate) > max_name_tokens * 6:
            logger.warning(f"LLM Naming: Name potentially too long after processing: '{candidate}' (raw: '{raw_candidate}')")
            return None

        logger.debug(f"LLM generated owner name: '{candidate}' (processed from raw: '{raw_candidate}')")
        return candidate

    except Exception as e:
        logger.error(
            f"LLM Naming API Error: {e}. Prompt that failed:\n{prompt_content_for_log}"
        )
        raise

# --- END PHASE 4b Function ---

# --- Narrative Generation with REVISED-REVISED Strict Checks ---





# --- PHASE 2: generate_narrative (explicit retry loop, two-step validation, NO in-prompt examples) ---
def generate_narrative(ast: Node, world: dict) -> str | None:
    """
    Generate a narrative for a ListOps AST, ensuring only original atomic operands are mentioned,
    with strict validation and world/owner narrative integration. Uses explicit retry loop with two-phase validation.
    Does NOT include in-prompt few-shot examples.
    """
    # --- Initial validation ---
    if not isinstance(ast, Node):
        raise ValueError("ast must be an instance of Node")
    if not isinstance(world, dict):
        raise ValueError("world must be a dict")
    required_keys = ("characters", "genre", "setting", "entity_concepts")
    if not all(k in world for k in required_keys):
        logger.error(f"World info missing required keys: {world.keys()}")
        raise ValueError("world missing required key(s)")
    if encoder is None:
        raise RuntimeError("Tokenizer not initialized.")
    if p_inflect is None:
        raise RuntimeError("Inflect engine not initialized.")

    scenes = []
    tokens_used = 0
    all_atoms = get_atoms_in_subtree(ast)
    logger.debug(f"All atomic operands: {all_atoms}")
    atoms_so_far = set()
    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    max_pad_paragraphs = 2
    if not operator_nodes and isinstance(ast, Atom):
        operator_nodes = [OpNode("SUM", [ast])]
        max_pad_paragraphs = 0

    # --- Owner mapping ---
    owner_map = {}
    if config.USE_OWNERSHIP_NARRATIVE:
        if operator_nodes:
            use_llm = config.USE_LLM_NAMING
            concepts = world.get("entity_concepts", [])
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
                        owner_name = generate_owner_name_with_llm(world, op_node, child_owner_names)
                    except Exception as e:
                        logger.error(f"LLM Naming failed for OpNode {op_node.op}: {e}")
                        owner_name = None
                if not owner_name:
                    if concepts:
                        chosen_concept = random.choice(concepts)
                        if characters:
                            char_name = random.choice(characters).get("name", "Someone")
                            possessive = f"{char_name}'" if char_name.endswith("s") else f"{char_name}'s"
                            owner_name = f"{possessive} {chosen_concept}"
                        else:
                            owner_name = f"the {chosen_concept} ({op_node.op} #{i+1})"
                    else:
                        owner_name = f"the_{op_node.op}_entity_{i+1}"
                owner_map[node_id] = owner_name
    # --- End owner mapping ---

    total_ops = len(operator_nodes)
    logger.info(f"Starting narrative generation: {total_ops} operator beats.")
    last_scene_text = "The story begins..."

    for idx, node in enumerate(operator_nodes, start=1):
        is_final = (node is ast)
        # Atoms from child subtrees
        atoms_from_children = set()
        for c in node.children:
            atoms_from_children.update(get_atoms_in_subtree(c))
        allowed_atoms = atoms_so_far.union(atoms_from_children)
        result = node.value
        if result is None:
            logger.error(f"Node {node.op} has None value.")
            return None

        op_label = OP_LABELS.get(node.op, node.op)
        ultra_strict_instruction = (
            "**ULTRA-STRICT NUMBER RULE:**\n"
            "*   You MUST NOT include ANY numbers (digits or words) in your response UNLESS they are original input numbers that have appeared previously in the story AND are essential for narrative context.\n"
            "*   DO NOT mention any intermediate calculation results.\n"
            "*   DO NOT introduce any new numbers that were not part of the original inputs.\n"
            "*   Focus ONLY on describing the process or consequences narratively."
        )
        operand_list_str = ", ".join(str(x) for x in sorted(atoms_from_children))
        current_task_body = ""
        final_task_body = ""
        ownership_instruction_detail = ""
        if config.USE_OWNERSHIP_NARRATIVE:
            node_id = id(node)
            owner_name = owner_map.get(node_id, f"the_unnamed_{node.op}_entity")
            ownership_instruction_detail = (
                f"This part of the story revolves around an entity or concept known as '{owner_name}'. "
                f"Its final state or significance is determined by applying a rule ({op_label}) to its constituent parts or related elements (which have been mentioned or established previously). "
                f"Describe narratively how '{owner_name}' is formed, evaluated, or what consequences arise from its state. "
                f"Refer to the constituent parts implicitly as belonging to or defining '{owner_name}'. Avoid explicitly stating the operation (like 'taking the max'). Focus on the story."
            )
            current_task_body = (
                f"Continue the story for 1-2 paragraphs. This scene focuses on the process or intermediate state involving '{owner_name}'. "
                f"Its current relevance or form is shaped by applying the **{op_label}** rule to its constituent parts: [{operand_list_str}]. "
                f"Describe character actions, thoughts, or plot developments related to '{owner_name}' at this stage.\n\n"
                f"{ownership_instruction_detail}"
            )
            final_task_body = (
                f"Write the concluding scene (1-2 paragraphs). This scene reveals the final significance or consequence related to '{owner_name}', "
                f"whose nature was determined by the **{op_label}** rule applied to its components: [{operand_list_str}] (as described in the narrative flow). "
                f"Focus on the ultimate outcome, character reactions, or plot resolution stemming from '{owner_name}'.\n\n"
                f"{ownership_instruction_detail}"
            )
            if is_final:
                task_header = "Final Task (Ownership Narrative)"
                beat_mode = f"{world['genre']}, writing the concluding scene about '{owner_name}'"
            else:
                task_header = "Current Task (Ownership Narrative)"
                beat_mode = f"{world['genre']}, writing a continuous narrative involving '{owner_name}'"
        else:
            current_task_body = (
                "Continue the story for 1-2 paragraphs. This part involves a logical step related to determining "
                f"the **{op_label}** based on elements or values described previously: [{operand_list_str}]. "
                "Focus entirely on the characters' actions, thoughts, and the unfolding plot. "
                "Describe the *process* or *consequence* of this logical step."
            )
            final_task_body = (
                "Write the concluding scene (1-2 paragraphs). This scene incorporates the *final* logical step, "
                f"related to determining the **{op_label}** based on elements described previously: [{operand_list_str}]. "
                "Focus on the final outcome, consequences, and character reactions resulting from this last step."
            )
            if is_final:
                task_header = "Final Task"
                beat_mode = f"{world['genre']}, writing the concluding scene"
            else:
                task_header = "Current Task"
                beat_mode = f"{world['genre']}, writing a continuous narrative"

        is_final_beat = is_final
        actual_task_body = final_task_body if is_final_beat else current_task_body

        beat_prompt = BASE_BEAT_TEMPLATE.substitute(
            beat_mode=beat_mode,
            characters=json.dumps(world["characters"]),
            setting=world["setting"],
            snippet=last_scene_text[-150:],
            task_header=task_header,
            task_body=actual_task_body,
            ultra_strict_instruction=ultra_strict_instruction
        )
        prompt_log_header = (
            f"=== FINAL Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="
            if is_final else
            f"=== Intermediate Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="
        )
        log_prompt(prompt_log_header, beat_prompt)
        estimated_prompt_tokens = len(encoder.encode(beat_prompt))
        if would_exceed_budget(tokens_used, estimated_prompt_tokens + MAX_BEAT_TOKENS, MAX_TOTAL_TOKENS, SAFETY_MARGIN):
            logger.warning(f"Approaching token limit before generating beat {idx}. Stopping.")
            return None
        validate_forbidden_numbers = make_number_validator(allowed_atoms)
        system_prompt = "You are a storyteller focused on narrative flow. FOLLOW THE USER'S NUMBER RULES EXACTLY. No calculations, no results, NO FORBIDDEN NUMBERS."
        # --- Explicit retry loop with two-step validation ---
        beat_text = None
        for attempt in range(1, config.MAX_BEAT_RETRIES + 1):
            try:
                resp = _chat_completion_call(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": beat_prompt}
                    ],
                    max_tokens=MAX_BEAT_TOKENS,
                    temperature=0.7,
                )
                candidate = resp.choices[0].message.content.strip()
                # Refusal detection
                if not candidate or candidate.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                    logger.warning(f"API refusal on beat attempt {attempt}.")
                    continue
                # Step 1: Forbidden number check
                if not validate_forbidden_numbers(candidate):
                    logger.warning(f"Beat {idx} attempt {attempt}: Forbidden numbers found.")
                    continue
                # Step 2: Operand presence check
                # For each atomic operand in atoms_from_children, ensure at least one is present
                if not atoms_from_children:
                    logger.warning(f"Beat {idx} attempt {attempt}: No operands to check presence for.")
                    continue
                found_operand = any(check_operand_presence(candidate, operand_val) for operand_val in atoms_from_children)
                if not found_operand:
                    logger.warning(f"Beat {idx} attempt {attempt}: No required operand present.")
                    continue
                # Passed both validations
                beat_text = candidate
                break
            except Exception as e:
                logger.warning(f"Error on beat {idx} attempt {attempt}: {e}")
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))
        if not beat_text:
            logger.error(
                f"Beat {idx} ({node.op}) failed after {config.MAX_BEAT_RETRIES} attempts. "
                "Failure could be due to forbidden numbers, missing operands, API errors, or refusals."
            )
            return None
        btoks = len(encoder.encode(beat_text))
        atoms_so_far.update(atoms_from_children)
        scenes.append(beat_text)
        tokens_used += btoks
        last_scene_text = beat_text
        logger.info(f"Appended beat {idx}, tokens used: {btoks}, total: {tokens_used}")

        # Padding (zero numbers)
        pad_count = 0
        add_padding = not is_final
        validate_padding = make_number_validator(set())
        while add_padding and tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
            estimated_pad_prompt_tokens = 200
            if would_exceed_budget(tokens_used, estimated_pad_prompt_tokens + MAX_PAD_TOKENS, MAX_TOTAL_TOKENS, SAFETY_MARGIN):
                logger.warning(f"Skipping padding after beat {idx}: Approaching token limit.")
                break
            padding_prompt = (
                f"You are a {world['genre']} storyteller, writing filler text between scenes.\n"
                f"Characters: {json.dumps(world['characters'])}\nSetting: {world['setting']}\n"
                f"Previous Scene Snippet: \"...{last_scene_text[-150:]}\"\n\n"
                "--- Task ---\n"
                "Write 1-2 paragraphs of narrative filler or transition.\n"
                "**ULTRA-STRICT ABSOLUTE RULE:** This padding text MUST contain ZERO numbers. NO digits (e.g., 1, 2, 3). NO number words (e.g., one, two, three). NONE AT ALL.\n\n"
                "Output only the narrative text for this padding, without titles or headings."
            )
            pad_result = None
            for pad_attempt in range(1, config.MAX_PAD_RETRIES + 1):
                try:
                    resp_pad = _chat_completion_call(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": "You are a storyteller writing transitional text. FOLLOW THE USER'S NUMBER RULES EXACTLY. ZERO NUMBERS ALLOWED."},
                            {"role": "user", "content": padding_prompt}
                        ],
                        max_tokens=MAX_PAD_TOKENS,
                        temperature=0.7,
                    )
                    candidate_pad = resp_pad.choices[0].message.content.strip()
                    if not candidate_pad or candidate_pad.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                        logger.warning(f"Padding refusal on attempt {pad_attempt}.")
                        continue
                    if not validate_padding(candidate_pad):
                        logger.warning(f"Padding attempt {pad_attempt}: Found numbers.")
                        continue
                    pad_result = candidate_pad
                    break
                except Exception as e:
                    logger.warning(f"Padding error on attempt {pad_attempt}: {e}")
                time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (pad_attempt - 1)))
            if not pad_result:
                logger.error("Padding generation failed.")
                break
            ptoks = len(encoder.encode(pad_result))
            if tokens_used + ptoks + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
                logger.warning(f"Generated padding too long ({ptoks} tokens). Discarding.")
                break
            scenes.append(pad_result)
            tokens_used += ptoks
            last_scene_text = pad_result
            pad_count += 1
        if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning("Token limit reached; stopping operator processing.")
            break

    # --- Final Narrative Construction and Validation ---
    if not scenes:
        logger.error("No scenes generated.")
        return None
    narrative_body = "\n\n".join(scenes).strip()
    found_in_final_body = extract_numbers_from_text(narrative_body)
    unexpected = found_in_final_body - all_atoms
    if unexpected:
        logger.error(f"FINAL VALIDATION FAILED: Unexpected numbers found: {unexpected}")
        return None
    question = (
        f"\n\n---\n\n"
        f"Analyze the narrative above to identify and perform the sequence of calculations embedded within it.  "
        f"What is the single, final numerical result?"
    )
    judge_instructions = """
---
**Task:**

1.	Read the narrative carefully. A multi-step calculation involving operations like finding the maximum, minimum, median, sum, sum modulo 10, or average (integer/floored) of groups of numbers is embedded within the story’s events and descriptions. The narrative should primarily contain only the original numbers involved in the calculations.
2.	Determine the single, final numerical result of this entire calculation based on the narrative’s progression.

**Output:** Provide *only* the final integer result. Do not include explanations, reasoning, or calculations in your final answer. Just the number.

**Final Answer:**
"""
    final_prompt = narrative_body + question + judge_instructions
    logger.info(f"Successfully generated and validated narrative prompt. Total tokens used (estimated): {tokens_used}")
    return final_prompt.strip()


def ast_to_prefix(node: Node) -> str:
    """Convert an AST to a prefix notation string."""
    if isinstance(node, Atom): return str(node.n)
    parts = [node.op] + [ast_to_prefix(child) for child in node.children]
    return "(" + " ".join(parts) + ")"


# --- HELPER FOR SINGLE SAMPLE GENERATION ---
def generate_single_sample(sample_index: int) -> dict | None:
    """Generate one sample with strict validation."""
    logger.info(f"--- Starting generation for sample {sample_index + 1} ---")
    sample_start_time = time.time()
    try:
        if encoder is None or p_inflect is None:
             logger.error(f"[Sample {sample_index + 1}] Missing tokenizer or inflect engine. Aborting.")
             return None

        logger.info(f"[Sample {sample_index + 1}] Building random AST...")
        ast = build_random_ast(max_ops=DEFAULT_MAX_OPS, max_branch=DEFAULT_MAX_BRANCH)
        validate_ast(ast)
        ast_prefix_string = ast_to_prefix(ast)
        logger.debug(f"[Sample {sample_index + 1}] Generated AST: {ast_prefix_string}")

        logger.info(f"[Sample {sample_index + 1}] Evaluating AST...")
        ground_truth_answer = eval_node(ast)
        logger.info(f"[Sample {sample_index + 1}] AST evaluation complete. Ground Truth: {ground_truth_answer}")

        logger.info(f"[Sample {sample_index + 1}] Generating world metadata...")
        world_info = generate_world(num_characters=random.randint(3, 6), num_concepts=random.randint(5, 10))
        logger.info(f"[Sample {sample_index + 1}] World metadata generated.")
        logger.debug(f"[Sample {sample_index + 1}] World Info: {world_info}")

        logger.info(f"[Sample {sample_index + 1}] Starting narrative rendering with revised-revised strict validation...")
        narrative_prompt = generate_narrative(ast, world_info) # Uses the updated function
        if narrative_prompt is None:
            logger.error(f"[Sample {sample_index + 1}] Narrative generation failed revised-revised validation. Skipping.")
            sample_end_time = time.time()
            logger.error(f"--- Failed sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f}s (Narrative Validation Failure) ---")
            return None
        logger.info(f"[Sample {sample_index + 1}] Narrative rendering and validation complete.")

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
                     "ultra_strict_prompts_llm_ownership_v4b" if (config.USE_OWNERSHIP_NARRATIVE and config.USE_LLM_NAMING)
                     else "ultra_strict_prompts_thematic_ownership_v4a" if config.USE_OWNERSHIP_NARRATIVE
                     else "ultra_strict_prompts_revised_revised_validation"
                 ),
            }
        }
        sample_end_time = time.time()
        logger.info(f"--- Successfully generated sample {sample_index + 1} in {sample_end_time - sample_start_time:.2f} seconds ---")
        return sample_data

    except ValueError as e: logger.error(f"[Sample {sample_index + 1}] Data validation error: {e}")
    except RuntimeError as e: logger.error(f"[Sample {sample_index + 1}] Runtime error: {e}")
    except Exception as e: logger.exception(f"[Sample {sample_index + 1}] Unexpected error: {e}")

    sample_end_time = time.time()
    logger.error(f"--- Failed sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f}s (Exception) ---")
    return None

def main(num_samples: int = config.NUM_SAMPLES_TO_GENERATE, output_file: str = OUTPUT_FILENAME, max_workers: int = config.DEFAULT_MAX_WORKERS):
    """Generate samples with strict validation."""
    logger.info(f"Script started. Generating {num_samples} samples using up to {max_workers} workers with REVISED-REVISED STRICT validation.")
    samples_generated_successfully = 0; samples_failed = 0
    output_dir = os.path.dirname(output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    start_time = time.time(); results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(generate_single_sample, i): i for i in range(num_samples)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_data = future.result()
                if sample_data: results.append(sample_data); samples_generated_successfully += 1
                else: samples_failed += 1
            except Exception as exc: logger.error(f"[Sample {index + 1}] task generated exception: {exc}"); samples_failed += 1
    logger.info(f"Parallel generation complete. Writing {samples_generated_successfully} samples to {output_file}...")
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for sample_data in results:
                try: f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
                except TypeError as e: logger.error(f"Serialization failed: {e}. Sample: {sample_data.get('id', 'Unknown')}"); samples_failed += 1; samples_generated_successfully -= 1
                except Exception as e: logger.error(f"Write error sample {sample_data.get('id', 'Unknown')}: {e}"); samples_failed += 1; samples_generated_successfully -= 1
    except IOError as e: logger.error(f"Fatal file write error {output_file}: {e}"); samples_failed = num_samples; samples_generated_successfully = 0
    end_time = time.time(); total_time = end_time - start_time
    logger.info(f"--- Batch generation complete ---")
    logger.info(f"Total samples attempted: {num_samples}")
    logger.info(f"Successfully generated and written: {samples_generated_successfully}")
    logger.info(f"Failed generations or writes: {samples_failed}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Dataset output file: {output_file}")
    logging.shutdown()

if __name__ == "__main__":
    main(max_workers=config.DEFAULT_MAX_WORKERS)