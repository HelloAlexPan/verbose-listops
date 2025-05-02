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

import tiktoken
import anthropic
from anthropic import Anthropic
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
DEFAULT_MAX_OPS = 20 # Max operations (e.g. max, min ,etc.) in a problem
MIN_ARITY = 5 # Min numbers in an operation
DEFAULT_MAX_BRANCH = 8 # Max operations / numbers in an operation



import concurrent.futures
import inflect


# --- Prompt Logging Helper ---
def log_prompt(header: str, prompt: str, path: str = os.path.join(LOG_DIR, "verbose_listops_prompts.log")):
    """Append a prompt header and text to the prompts log."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{header}\n{prompt}\n\n")
        f.flush()



# --- API retry + logging config ---
 
LOG_MAX_BYTES = 5 * 1024 * 1024 # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3 # Number of backup log files to keep

# --- Anthropic Client Factory ---
def init_anthropic(api_key: str):
    """Initialize and return Anthropic client and tokenizer."""
    client = Anthropic(api_key=api_key)
    encoder = tiktoken.get_encoding("cl100k_base")
    return client, encoder

# --- Config Dataclass Grouping ---
@dataclass
class Config:
    NUM_SAMPLES_TO_GENERATE: int = NUM_SAMPLES_TO_GENERATE
    DEFAULT_MAX_WORKERS: int = DEFAULT_MAX_WORKERS
    DEFAULT_MAX_TOTAL_TOKENS: int = 20000
    DEFAULT_MAX_BEAT_TOKENS: int = 1000
    DEFAULT_MAX_PAD_TOKENS: int = 1000
    MAX_TOKENS_BUFFER: int = 1000
    PROMPT_SHOT_COUNT: int = 3
    RETRY_MAX_ATTEMPTS: int = 5
    RETRY_INITIAL_DELAY: int = 1
    MAX_BEAT_RETRIES: int = 5
    MAX_PAD_RETRIES: int = 3
    ATOM_MIN_VALUE: int = 1
    ATOM_MAX_VALUE: int = 100
config = Config()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder.")
    API_KEY = "YOUR_API_KEY_HERE" # Placeholder

MODEL = "claude-3-haiku-20240307" # Use Haiku for potentially faster/cheaper generation during testing strict rules
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


# --- Anthropic Client and Tokenizer ---
try:
    client, encoder = init_anthropic(API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client or tokenizer: {e}")
    client = None
    encoder = None

# --- Inflect Engine ---
try:
    p_inflect = inflect.engine()
except Exception as e:
    logger.error(f"Failed to initialize inflect engine: {e}")
    p_inflect = None


def retry_api_call(func: Callable):
    """Decorator to retry Anthropic API calls on failure with exponential backoff."""
    def wrapper(*args, **kwargs):
        if client is None:
            raise RuntimeError("Anthropic client not initialized.")
        delay = config.RETRY_INITIAL_DELAY
        for attempt in range(1, config.RETRY_MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except anthropic.APIConnectionError as e:
                logger.warning(f"API connection error attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}: {e}")
            except anthropic.RateLimitError as e:
                logger.warning(f"API rate limit error attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}: {e}")
            except anthropic.APIStatusError as e:
                logger.warning(f"API status error attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}: {e.status_code} - {e.response}")
            except Exception as e:
                logger.warning(f"API call failed attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}: {e}")
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
            resp = _client_create(
                model=MODEL,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            candidate = resp.content[0].text.strip()
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
def _client_create(**kwargs): return client.messages.create(**kwargs)


# --- JSON Cleaning Helper ---
def clean_and_parse_json_block(text: str):
    """Strip Markdown code fences and parse JSON."""
    text = re.sub(r"^```(?:json)?", "", text).strip()
    if text.endswith("```"):
        text = text[: -3].strip()
    return json.loads(text)

def generate_world(num_characters: int = 5) -> dict:
    """Generates fictional world metadata."""
    if not isinstance(num_characters, int) or num_characters < 1: raise ValueError("num_characters must be positive int")
    prompt = (
        "You are a creative world-builder.\n"
        f"Generate {num_characters} distinct characters (name, role, quirk). Define a genre and setting.\n"
        "Output *only* a valid JSON object: {\"characters\": [{\"name\": ..., \"role\": ..., \"quirk\": ...}, ...], \"genre\": ..., \"setting\": ...}\n"
    )
    try:
        resp = _client_create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.8)
        text = resp.content[0].text.strip()
        try:
            cleaned = clean_and_parse_json_block(text)
            world = cleaned
            if not all(k in world for k in ["characters", "genre", "setting"]) or \
               not isinstance(world["characters"], list) or len(world["characters"]) != num_characters:
                 logger.error(f"Generated world JSON structure error: {text}"); raise RuntimeError("World JSON validation failed.")
            return world
        except json.JSONDecodeError as e: logger.error(f"Failed to parse world JSON: {text}. Error: {e}"); raise RuntimeError("JSON parse failed") from e
    except Exception as e: logger.error(f"World gen API error: {e}"); raise RuntimeError("World gen failed") from e


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

# --- Narrative Generation with REVISED-REVISED Strict Checks ---
def generate_narrative(ast: Node, world: dict) -> str | None:
    """Generate narrative for each AST operator with strict number validation."""
    # --- Initial checks ---
    if not isinstance(ast, Node): raise ValueError("ast must be an instance of Node")
    if not isinstance(world, dict): raise ValueError("world must be a dict")
    if not all(k in world for k in ("characters", "genre", "setting")): raise ValueError("world missing required key")
    if encoder is None: raise RuntimeError("Tokenizer not initialized.")
    if p_inflect is None: raise RuntimeError("Inflect engine not initialized.")

    scenes = []
    tokens_used = 0
    log_file_path = os.path.join(LOG_DIR, "verbose_listops_prompts.log")

    # --- Get ALL original atomic values from the entire AST ---
    all_atomic_operands_in_ast = get_atoms_in_subtree(ast)
    logger.debug(f"All original atomic operands in AST: {all_atomic_operands_in_ast}")

    # --- Track atoms processed so far ---
    atoms_processed_so_far = set() # Atoms from subtrees whose beats have been generated

    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    # Default padding paragraphs (override for single-atom case below)
    max_pad_paragraphs = 2

    # --- Unify single-atom case by synthesizing dummy operator ---
    if not operator_nodes and isinstance(ast, Atom):
        operator_nodes = [OpNode("SUM", [ast])]
        # disable padding for single-atom case
        max_pad_paragraphs = 0

    # --- Process Operator Beats ---
    total_ops = len(operator_nodes)
    logger.info(f"Starting narrative generation: {total_ops} operator beats to process (post-order).")
    last_scene_text = "The story begins..."

    for idx, node in enumerate(operator_nodes, start=1):
        is_final_beat = (node is ast)

        # --- Get atoms associated *only* with the children of this node ---
        atoms_from_children_subtrees = set()
        for child_node in node.children:
            atoms_from_children_subtrees.update(get_atoms_in_subtree(child_node))

        # The set of all atoms allowed to appear in this beat's text:
        # Those already processed + those belonging to the direct children's subtrees.
        allowed_atoms_in_this_beat = atoms_processed_so_far.union(atoms_from_children_subtrees)

        # We still need the node's value for the prompt's hidden result info
        result = node.value
        if result is None: logger.error(f"Node {node.op} has None value."); return None

        logger.info(
            f"Processing operator {idx}/{total_ops}: {node.op}. "
            f"Atoms from children: {atoms_from_children_subtrees}. "
            f"Atoms processed before: {atoms_processed_so_far}. "
            f"Allowed atoms now: {allowed_atoms_in_this_beat}."
            f"{' (FINAL BEAT)' if is_final_beat else ''}"
        )

        # --- ULTRA-STRICT prompt construction ---
        operation_concept = OP_LABELS.get(node.op, node.op)
        # --- ULTRA-STRICT NUMBER RULE ---
        ultra_strict_instruction = (
            "**ULTRA-STRICT NUMBER RULE:**\n"
            "*   You MUST NOT include ANY numbers (digits or words) in your response UNLESS they are original input numbers that have appeared previously in the story AND are essential for narrative context.\n"
            "*   DO NOT mention any intermediate calculation results.\n"
            "*   DO NOT introduce any new numbers that were not part of the original inputs.\n"
            "*   Focus ONLY on describing the process or consequences narratively."
        )

        if is_final_beat:
            task_header = "Final Task"
            task_body = (
                "Write the concluding scene (1-2 paragraphs). This scene incorporates the *final* logical step, "
                "related to determining the **$operation_concept** based on elements described previously. "
                "Focus on the final outcome, consequences, and character reactions resulting from this last step."
            )
            beat_mode = f"{world['genre']}, writing the concluding scene"
        else:
            task_header = "Current Task"
            task_body = (
                "Continue the story for 1-2 paragraphs. This part involves a logical step related to determining "
                "the **$operation_concept** based on elements or values described previously. "
                "Focus entirely on the characters' actions, thoughts, and the unfolding plot. "
                "Describe the *process* or *consequence* of this logical step."
            )
            beat_mode = f"{world['genre']}, writing a continuous narrative"

        beat_prompt = BASE_BEAT_TEMPLATE.substitute(
            beat_mode=beat_mode,
            characters=json.dumps(world["characters"]),
            setting=world["setting"],
            snippet=last_scene_text[-150:],
            task_header=task_header,
            task_body=task_body.replace("$operation_concept", operation_concept),
            ultra_strict_instruction=ultra_strict_instruction
        )
        if is_final_beat:
            prompt_log_header = f"=== FINAL Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="
        else:
            prompt_log_header = f"=== Intermediate Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="

        log_prompt(prompt_log_header, beat_prompt)

        estimated_prompt_tokens = len(encoder.encode(beat_prompt))
        if would_exceed_budget(tokens_used, estimated_prompt_tokens + MAX_BEAT_TOKENS, MAX_TOTAL_TOKENS, SAFETY_MARGIN):
            logger.warning(f"Approaching token limit before generating beat {idx}. Stopping.")
            return None

        # ----------- Helper function for validating a candidate beat -----------
        validate_beat = make_number_validator(allowed_atoms_in_this_beat)

        # Use the retry helper
        system_prompt = "You are a storyteller focused on narrative flow. FOLLOW THE USER'S NUMBER RULES EXACTLY. No calculations, no results, NO FORBIDDEN NUMBERS."
        beat_text = generate_with_retry(system_prompt, beat_prompt, MAX_BEAT_TOKENS, validate_beat, retries=config.MAX_BEAT_RETRIES)
        if not beat_text:
            logger.error(f"Beat {idx} ({node.op}) failed after {config.MAX_BEAT_RETRIES} attempts.")
            return None
        btoks = len(encoder.encode(beat_text))
        atoms_processed_so_far.update(atoms_from_children_subtrees)
        scenes.append(beat_text)
        tokens_used += btoks
        last_scene_text = beat_text
        logger.info(f"Appended validated beat {idx}, tokens used: {btoks}, total: {tokens_used}")

        # ---------- Padding Generation (Ultra-strict: ZERO numbers) ----------
        pad_count = 0
        add_padding = not is_final_beat

        # --- Helper for validating padding ---
        validate_padding = make_number_validator(set())

        while add_padding and tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
            estimated_pad_prompt_tokens = 200
            if would_exceed_budget(tokens_used, estimated_pad_prompt_tokens + MAX_PAD_TOKENS, MAX_TOTAL_TOKENS, SAFETY_MARGIN):
                logger.warning(f"Skipping padding after beat {idx}: Approaching token limit.")
                add_padding = False
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

            # Use the generate_with_retry helper for padding
            result = generate_with_retry(
                system_prompt="You are a storyteller writing transitional text. FOLLOW THE USER'S NUMBER RULES EXACTLY. ZERO NUMBERS ALLOWED.",
                user_prompt=padding_prompt,
                max_tokens=MAX_PAD_TOKENS,
                validate_fn=validate_padding,
                retries=config.MAX_PAD_RETRIES,
            )
            if result is None:
                logger.error("Padding generation failed.")
                add_padding = False
            else:
                padding_text = result
                ptoks = len(encoder.encode(padding_text))
                if tokens_used + ptoks + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
                    logger.warning(f"Generated padding too long ({ptoks} tokens). Discarding.")
                    add_padding = False
                else:
                    scenes.append(padding_text)
                    tokens_used += ptoks
                    last_scene_text = padding_text
                    pad_count += 1
                    logger.info(f"Appended strictly validated padding {pad_count}, tokens used: {ptoks}, total: {tokens_used}")
            if pad_count >= max_pad_paragraphs:
                add_padding = False
            if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
                logger.warning("Token limit reached during padding.")
                add_padding = False
                break
        if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN: logger.warning("Token limit reached; stopping operator processing."); break
    # --- End of Operator Loop ---

    # --- Final Narrative Construction and Validation ---
    if not scenes: logger.error("No scenes generated."); return None
    narrative_body = "\n\n".join(scenes).strip()

    # --- Final Overall Number Check ---
    logger.info("Performing final check on the entire narrative body...")
    logger.debug(f"Expected atoms overall (from original AST): {all_atomic_operands_in_ast}")
    found_in_final_body = extract_numbers_from_text(narrative_body)
    logger.debug(f"Numbers found in final body: {found_in_final_body}")

    unexpected_in_final = found_in_final_body - all_atomic_operands_in_ast
    if unexpected_in_final:
        logger.error(f"FINAL VALIDATION FAILED: Unexpected numbers found in final narrative: {unexpected_in_final}. Not original atoms.")
        return None
    else:
        logger.info("Final narrative validation passed.")

    # --- Construct Final Prompt ---
    question = (
        f"\n\n---\n\n"
        f"Analyze the narrative above to identify and perform the sequence of calculations embedded within it.  "
        f"What is the single, final numerical result?"
    )
    judge_instructions = f"""
---
**Task:**

1.	Read the narrative carefully. A multi-step calculation involving operations like finding the maximum, minimum, median, sum, sum modulo 10, or average (integer/floored) of groups of numbers is embedded within the story’s events and descriptions. The narrative should primarily contain only the original numbers involved in the calculations.
2.	Determine the single, final numerical result of this entire calculation based on the narrative’s progression.

**Output:** Provide *only* the final integer result. Do not include explanations, reasoning, or calculations in your final answer. Just the number.

**Final Answer:**
"""
    few_shot_examples = SHOT_EXAMPLES.get(config.PROMPT_SHOT_COUNT, "")
    final_prompt = few_shot_examples + narrative_body + question + judge_instructions
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
        if client is None or encoder is None or p_inflect is None:
             logger.error(f"[Sample {sample_index + 1}] Missing critical component. Aborting.")
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
        world_info = generate_world(num_characters=random.randint(3, 6))
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
                 "validation_mode": "ultra_strict_prompts_revised_revised_validation"
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