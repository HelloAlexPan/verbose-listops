"""
verbose-listops.py

1. Generates a complex ListOps problem as an Abstract Syntax Tree (AST).
2. Evaluates the AST to get the ground truth result.
3. Generates fictional world metadata (characters, genre, setting) via the Anthropic API.
4. Renders a narrative where each step of the ListOps calculation is a story 'beat',
   interspersed with optional 'padding' paragraphs, using the Anthropic API.
   **Includes REVISED-REVISED strict checks:**
     - Beats must ONLY introduce numbers that are atomic operands relevant to the
       current operation's children OR atoms already seen. NO intermediate results allowed.
     - Padding must contain NO numbers.
     - Final narrative must contain ONLY numbers that were original atoms.
5. Constructs a final prompt including the narrative and instructions for a 'judge' LLM
   to solve the original ListOps problem based *only* on the narrative.
6. Supports batch generation and saving samples to a JSONL file.

Supports logging, API retry logic with exponential backoff, and AST validation.
Allows configuration via constants.
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
import concurrent.futures
import threading
import inflect # <-- ADDED for number words

# ─── Configuration Constants ────────────────────────────────────────────────────────────────────────────

# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 4 # How many samples to generate in one run
OUTPUT_FILENAME = "verbose_listops_dataset_strict_v3.jsonl" # Output file for the dataset (updated name)
DEFAULT_MAX_WORKERS = 8  # Default number of parallel threads for batch generation

# --- Base configurations ---
# Output Configuration
LOG_DIR = os.path.expanduser("~/verbose_listops_logs")
DEFAULT_MAX_TOTAL_TOKENS = 20000 # Overall token cap for narrative generation
DEFAULT_MAX_BEAT_TOKENS = 1000 # Maximum tokens used for each story 'beat' (listops paragraph)
DEFAULT_MAX_PAD_TOKENS = 1000 # Maximum tokens used for each story 'padding' section
MAX_TOKENS_BUFFER = 1000 # Safety buffer to prevent exceeding token limits
PROMPT_SHOT_COUNT = 3 # Number of few-shot examples to include in generated narrative prompt problem (0, 1, 2, or 3)
SHOT_EXAMPLES = {
    0: "",
    1: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts: one paying 9 silver pieces, the other only 4. Kaelen chose the lower-paying contract to avoid scrutiny. He then received a standard 5 silver piece bonus for completing the task quickly.\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n\n"
        "</Prompt Shot>\n"
    ),
    2: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts: one paying 9 silver pieces, the other only 4. Kaelen chose the lower-paying contract to avoid scrutiny. He then received a standard 5 silver piece bonus for completing the task quickly.\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n\n"
        "Example 2:\n"
        "Narrative: “To unlock the ancient vault, the combined energy signature of four power crystals (reading 1, 1, 1, and 1) was required. The locking mechanism, however, only used the final digit of their total combined power.”\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Modulo 10 (final digit) is 4.\n"
        "Answer: 4\n\n"
        "</Prompt Shot>\n"
    ),
    3: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts: one paying 9 silver pieces, the other only 4. Kaelen chose the lower-paying contract to avoid scrutiny. He then received a standard 5 silver piece bonus for completing the task quickly.\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n\n"
        "Example 2:\n"
        "Narrative: “To unlock the ancient vault, the combined energy signature of four power crystals (reading 1, 1, 1, and 1) was required. The locking mechanism, however, only used the final digit of their total combined power.”\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Modulo 10 (final digit) is 4.\n"
        "Answer: 4\n\n"
        "Example 3:\n"
        "Narrative: “Three scouts reported patrol durations of 5, 5, and 5 hours. Standard procedure required calculating their average patrol time, rounded down to the nearest whole hour, for the official logbook entry.”\n"
        "Implicit Calculation: SUM(5, 5, 5) = 15. Count = 3. Average = 15 / 3 = 5. Floor(5) = 5.\n"
        "Answer: 5\n"
        "</Prompt Shot>\n"
    ),
}

# --- AST Random ListOps problem gen params ---
DEFAULT_MAX_OPS = 20 # Max operations (e.g. max, min ,etc.) in a problem
MIN_ARITY = 5 # Min numbers in an operation
DEFAULT_MAX_BRANCH = 8 # Max operations / numbers in an operation
ATOM_MIN_VALUE = 1 # Minimum number
ATOM_MAX_VALUE = 100 # Maximum number


 # --- API retry + logging config ---
RETRY_MAX_ATTEMPTS = 5 # Maximum number of retry attempts for API calls
RETRY_INITIAL_DELAY = 1 # Initial delay between retries in seconds (doubles with each attempt)
MAX_BEAT_RETRIES = 5  # Increased retries for stricter validation
MAX_PAD_RETRIES = 3 # Retries for generating valid padding
LOG_MAX_BYTES = 5 * 1024 * 1024 # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3 # Number of backup log files to keep

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder.")
    API_KEY = "YOUR_API_KEY_HERE" # Placeholder

MODEL = "claude-3-haiku-20240307" # Use Haiku for potentially faster/cheaper generation during testing strict rules
MAX_TOTAL_TOKENS = DEFAULT_MAX_TOTAL_TOKENS
SAFETY_MARGIN = MAX_TOKENS_BUFFER
MAX_BEAT_TOKENS = DEFAULT_MAX_BEAT_TOKENS
MAX_PAD_TOKENS = DEFAULT_MAX_PAD_TOKENS

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
    client = Anthropic(api_key=API_KEY)
    encoder = tiktoken.get_encoding("cl100k_base")
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
        if client is None: raise RuntimeError("Anthropic client not initialized.")
        delay = RETRY_INITIAL_DELAY
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except anthropic.APIConnectionError as e: logger.warning(f"API connection error attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")
            except anthropic.RateLimitError as e: logger.warning(f"API rate limit error attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")
            except anthropic.APIStatusError as e: logger.warning(f"API status error attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e.status_code} - {e.response}")
            except Exception as e: logger.warning(f"API call failed attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")
            if attempt == RETRY_MAX_ATTEMPTS: logger.error("Max retry attempts reached."); raise
            time.sleep(delay); delay *= 2
    return wrapper


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
    if not isinstance(max_ops, int) or max_ops < 1: raise ValueError("max_ops must be a positive int")
    if max_branch < MIN_ARITY: raise ValueError(f"max_branch ({max_branch}) < MIN_ARITY ({MIN_ARITY})")
    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]; count = 0
    def helper():
        nonlocal count
        if count >= max_ops or (count > 0 and random.random() < 0.1): return Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE))
        count += 1; op = random.choice(ops); arity = random.randint(MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]; return OpNode(op, children)
    root = helper()
    if isinstance(root, Atom) and max_ops >= 1:
         op = random.choice(ops); arity = random.randint(MIN_ARITY, max_branch)
         children = [Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE)) for _ in range(arity-1)]
         children.append(root); random.shuffle(children); root = OpNode(op, children)
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

def preorder(node: Node):
    """Yield nodes in pre-order."""
    yield node
    for c in node.children: yield from preorder(c)

@retry_api_call
def _client_create(**kwargs): return client.messages.create(**kwargs)

# --- World Generation (Keep as is) ---
def generate_world(num_characters: int = 5) -> dict:
    """Generates fictional world metadata."""
    if not isinstance(num_characters, int) or num_characters < 1: raise ValueError("num_characters must be positive int")
    prompt = (
        "You are a creative world-builder.\n"
        f"Generate {num_characters} distinct characters (name, role, quirk). Define a genre and setting.\n"
        "Output *only* a valid JSON object: {\"characters\": [{\"name\": ..., \"role\": ..., \"quirk\": ...}, ...], \"genre\": ..., \"setting\": ...}\n"
    ) # Simplified prompt for brevity
    try:
        resp = _client_create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.8)
        text = resp.content[0].text.strip()
        try:
            if text.startswith("```json"): text = text[len("```json"):].strip()
            if text.startswith("```"): text = text[len("```"):].strip()
            if text.endswith("```"): text = text[:-len("```")].strip()
            world = json.loads(text)
            if not all(k in world for k in ["characters", "genre", "setting"]) or \
               not isinstance(world["characters"], list) or len(world["characters"]) != num_characters:
                 logger.error(f"Generated world JSON structure error: {text}"); raise RuntimeError("World JSON validation failed.")
            return world
        except json.JSONDecodeError as e: logger.error(f"Failed to parse world JSON: {text}. Error: {e}"); raise RuntimeError("JSON parse failed") from e
    except Exception as e: logger.error(f"World gen API error: {e}"); raise RuntimeError("World gen failed") from e


# --- Number Extraction Helper (Using corrected version for negative words) ---
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

# --- Helper to get atoms --- ADD THIS FUNCTION ---
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
    """
    Render a narrative for each operator in the AST using the Anthropic API.
    Includes REVISED-REVISED strict checks:
     - Beats must ONLY introduce numbers that are atomic operands relevant to the
       current operation's children OR atoms already seen. NO intermediate results allowed.
     - Padding must contain NO numbers.
     - Final narrative must contain ONLY numbers that were original atoms.
    """
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

    # --- Handle single atom case (no operators) ---
    if not operator_nodes and isinstance(ast, Atom):
         single_atom_prompt = (
              f"You are a {world['genre']} storyteller.\n"
              f"Characters: {json.dumps(world['characters'])}\n"
              f"Setting: {world['setting']}\n\n"
              f"Write a very short scene (1-2 paragraphs) where a character encounters or notes the number {ast.n}. " # Use ast.n
              "Focus on the story and characters. Do NOT include any other numbers."
         )
         with open(log_file_path, "a", encoding="utf-8") as prompts_log:
              prompts_log.write(f"=== Single Atom Prompt ===\n{single_atom_prompt}\n\n")
              prompts_log.flush()

         required_operand_atom = {ast.n} # Use ast.n
         atom_text = ""
         atom_generation_failed = False
         for attempt in range(MAX_BEAT_RETRIES):
             try:
                 resp = _client_create(
                      model=MODEL,
                      system="You are a storyteller. Write plain narrative paragraphs without markdown. Avoid unnecessary numbers.",
                      messages=[{"role": "user", "content": single_atom_prompt}],
                      max_tokens=MAX_BEAT_TOKENS,
                 )
                 candidate_text = resp.content[0].text.strip()

                 if not candidate_text or candidate_text.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                     logger.warning(f"API refusal on single atom generation (Attempt {attempt+1}).")
                     if attempt == MAX_BEAT_RETRIES - 1: atom_generation_failed = True; break
                     time.sleep(1)
                     continue

                 # --- Atom Validation ---
                 # 1. Check presence of the required atom value (using ast.n)
                 if not check_operand_presence(candidate_text, ast.n):
                     logger.warning(f"Missing required operand {ast.n} in single atom narrative (Attempt {attempt+1}).")
                     if attempt == MAX_BEAT_RETRIES - 1: atom_generation_failed = True; break
                     time.sleep(1)
                     continue

                 # 2. Check for extra numbers (any number not equal to the required one)
                 found_numbers = extract_numbers_from_text(candidate_text)
                 extra_numbers = found_numbers - required_operand_atom
                 if extra_numbers:
                     logger.warning(f"Found unexpected extra numbers {extra_numbers} in single atom narrative (Attempt {attempt+1}). Required: {required_operand_atom}, Found: {found_numbers}")
                     if attempt == MAX_BEAT_RETRIES - 1: atom_generation_failed = True; break
                     time.sleep(1)
                     continue

                 # Success
                 atom_text = candidate_text
                 atoms_processed_so_far.add(ast.n) # Mark atom as processed (use ast.n)
                 break
             except Exception as e:
                 logger.error(f"Error on single atom attempt {attempt+1}: {e}")
                 if attempt == MAX_BEAT_RETRIES - 1: atom_generation_failed = True; break
                 time.sleep(RETRY_INITIAL_DELAY * (2 ** attempt))

         if atom_generation_failed or not atom_text:
             logger.error("Single atom narrative generation ultimately failed validation.")
             return None

         scenes.append(atom_text)
         tokens_used += len(encoder.encode(scenes[-1]))
         # --- End single atom case ---

    # --- Process Operator Beats ---
    max_pad_paragraphs = 2
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

        # ----------------- prompt construction (REMOVE OPERANDS LIST) -----------------
        strict_instruction = (f"Ensure the narrative for this scene naturally reflects the process of performing a '{node.op}' operation "
                              f"(finding the {OP_LABELS.get(node.op, node.op)}) based on the outcomes or values described in the preceding narrative steps. "
                              "Avoid introducing any new numerical values unless they were part of the original inputs mentioned earlier in the story. "
                              "Do NOT explicitly state the numerical result of this step.")

        if is_final_beat:
             beat_prompt = (
                 f"You are a creative {world['genre']} storyteller, writing the concluding scene...\n"
                 f"Characters: {json.dumps(world['characters'])}\nSetting: {world['setting']}\n"
                 f"Previous Scene Snippet: \"...{last_scene_text[-150:]}\"\n\n"
                 "--- Final Task ---\n"
                 "Integrate the final logical step. This step involves:\n"
                 f"* Operation Type: {node.op} (finding the {OP_LABELS.get(node.op, node.op)}) based on results from previous steps.\n"
                 f"* Implicit Final Result: {result} (DO NOT MENTION THIS EXPLICITLY!)\n\n"
                 "--- Instructions ---\n"
                 "Write 1 concluding scene (2‑5 short paragraphs). Weave the *process* subtly. Describe the consequence/outcome.\n"
                 f"**NUMBER RULE:** {strict_instruction}\n"
                 "\nOutput only the narrative text..."
             )
             prompt_log_header = f"=== FINAL Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="
        else:
             beat_prompt = (
                 f"You are a creative {world['genre']} storyteller, writing a continuous narrative...\n"
                 f"Characters: {json.dumps(world['characters'])}\nSetting: {world['setting']}\n"
                 f"Previous Scene Snippet: \"...{last_scene_text[-150:]}\"\n\n"
                 "--- Current Task ---\n"
                 "Integrate the following logical step. This step involves:\n"
                 f"* Operation Type: {node.op} (finding the {OP_LABELS.get(node.op, node.op)}) based on results from previous steps.\n"
                 f"* Result: {result} (Do NOT state this number explicitly!)\n\n"
                 "--- Instructions ---\n"
                 "Write 1 scene (2‑5 short paragraphs). Weave the process into the story.\n"
                 f"**NUMBER RULE:** {strict_instruction}\n"
                 "\nOutput only the narrative text..."
             )
             prompt_log_header = f"=== Intermediate Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="

        with open(log_file_path, "a", encoding="utf-8") as prompts_log:
            prompts_log.write(prompt_log_header + "\n")
            prompts_log.write(beat_prompt + "\n\n")
            prompts_log.flush()

        estimated_prompt_tokens = len(encoder.encode(beat_prompt))
        if tokens_used + estimated_prompt_tokens + MAX_BEAT_TOKENS + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
            logger.warning(f"Approaching token limit before generating beat {idx}. Stopping.")
            return None

        # -------- retry loop with REVISED-REVISED validation ----------
        beat_text = ""
        btoks = 0
        beat_generation_failed = False
        for attempt in range(MAX_BEAT_RETRIES):
            logger.info(f"Attempt {attempt+1}/{MAX_BEAT_RETRIES} for beat {idx} ({node.op}). Allowed atoms: {allowed_atoms_in_this_beat}")
            try:
                resp = _client_create(
                    model=MODEL,
                    system="You are a storyteller. Write plain narrative paragraphs following the user's instructions precisely, especially the rules about numbers. Do not include markdown headings or section titles.",
                    messages=[{"role": "user", "content": beat_prompt}],
                    max_tokens=MAX_BEAT_TOKENS,
                    temperature=0.7,
                )
                candidate_text = resp.content[0].text.strip()

                if not candidate_text or candidate_text.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                    logger.warning(f"API refusal on beat {idx} generation (Attempt {attempt+1}).")
                    if attempt == MAX_BEAT_RETRIES - 1: beat_generation_failed = True; break
                    time.sleep(RETRY_INITIAL_DELAY * (2**attempt))
                    continue

                # --- Revised-Revised Number Validation ---
                validation_passed = True
                found_numbers_in_beat = extract_numbers_from_text(candidate_text)
                forbidden_numbers = found_numbers_in_beat - allowed_atoms_in_this_beat

                if forbidden_numbers:
                    intermediate_results_stated = {f for f in forbidden_numbers if abs(f) > ATOM_MAX_VALUE}
                    new_low_value_atoms = forbidden_numbers - intermediate_results_stated
                    if intermediate_results_stated: logger.warning(f"Beat {idx} Attempt {attempt+1} FAILED: Found forbidden intermediate results: {intermediate_results_stated}. (Allowed: {allowed_atoms_in_this_beat})")
                    if new_low_value_atoms: logger.warning(f"Beat {idx} Attempt {attempt+1} FAILED: Found new forbidden atomic-value numbers: {new_low_value_atoms}. (Allowed: {allowed_atoms_in_this_beat})")
                    validation_passed = False

                if validation_passed:
                    logger.info(f"Beat {idx} Attempt {attempt+1} PASSED validation. Found: {found_numbers_in_beat}")
                    beat_text = candidate_text
                    btoks = len(encoder.encode(beat_text))
                    # --- Update processed atoms ---
                    atoms_processed_so_far.update(atoms_from_children_subtrees)
                    logger.debug(f"Updated atoms_processed_so_far: {atoms_processed_so_far}")
                    break
                else:
                    if attempt == MAX_BEAT_RETRIES - 1: beat_generation_failed = True; break
                    time.sleep(RETRY_INITIAL_DELAY * (2**attempt))
                    continue

            except Exception as e:
                logger.error(f"Error during API call/processing for beat {idx} attempt {attempt+1}: {e}")
                if attempt == MAX_BEAT_RETRIES - 1: beat_generation_failed = True; break
                time.sleep(RETRY_INITIAL_DELAY * (2 ** attempt))

        if beat_generation_failed or not beat_text:
            logger.error(f"Beat {idx} ({node.op}) generation ultimately failed revised-revised validation after {MAX_BEAT_RETRIES} attempts.")
            return None

        scenes.append(beat_text)
        tokens_used += btoks
        last_scene_text = beat_text
        logger.info(f"Appended validated beat {idx}, tokens used: {btoks}, total: {tokens_used}")

        # ---------- Padding Generation (Keep Strict Check: NO numbers) ----------
        pad_count = 0
        add_padding = not is_final_beat
        while add_padding and tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
            estimated_pad_prompt_tokens = 200
            if tokens_used + estimated_pad_prompt_tokens + MAX_PAD_TOKENS + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
                logger.warning(f"Skipping padding after beat {idx}: Approaching token limit.")
                add_padding = False; break

            padding_prompt = (
                f"You are a {world['genre']} storyteller, writing filler text between scenes.\n"
                f"Characters: {json.dumps(world['characters'])}\nSetting: {world['setting']}\n"
                f"Previous Scene Snippet: \"...{last_scene_text[-150:]}\"\n\n"
                "--- Task ---\n"
                "Write 1-2 paragraphs of narrative filler or transition.\n"
                "**VERY STRICT RULE:** This padding text must NOT contain ANY numbers (digits or words).\n"
                "\nOutput only the narrative text..."
            )
            padding_text = ""; ptoks = 0; padding_generation_failed = False
            for pad_attempt in range(MAX_PAD_RETRIES):
                logger.info(f"Padding attempt {pad_attempt + 1}/{MAX_PAD_RETRIES} after beat {idx}.")
                try:
                    resp = _client_create(
                        model=MODEL,
                        system="You are a storyteller. Write plain narrative paragraphs following the user's instructions precisely, especially the strict rule about NO numbers.",
                        messages=[{"role": "user", "content": padding_prompt}],
                        max_tokens=MAX_PAD_TOKENS, temperature=0.7,
                    )
                    candidate_padding = resp.content[0].text.strip()
                    if not candidate_padding or candidate_padding.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                         logger.warning(f"API refusal on padding generation (Attempt {pad_attempt+1}).")
                         if pad_attempt == MAX_PAD_RETRIES - 1: padding_generation_failed = True; break
                         time.sleep(RETRY_INITIAL_DELAY * (2**pad_attempt)); continue

                    found_numbers_in_padding = extract_numbers_from_text(candidate_padding)
                    if found_numbers_in_padding:
                        logger.warning(f"Padding Attempt {pad_attempt+1} FAILED: Found forbidden numbers: {found_numbers_in_padding}")
                        if pad_attempt == MAX_PAD_RETRIES - 1: padding_generation_failed = True; break
                        time.sleep(RETRY_INITIAL_DELAY * (2**pad_attempt)); continue
                    else:
                        logger.info(f"Padding Attempt {pad_attempt+1} PASSED validation.")
                        padding_text = candidate_padding; ptoks = len(encoder.encode(padding_text))
                        if tokens_used + ptoks + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
                             logger.warning(f"Generated padding too long ({ptoks} tokens). Discarding."); padding_text = ""; add_padding = False
                        break
                except Exception as e:
                    logger.error(f"Error during padding attempt {pad_attempt+1}: {e}")
                    if pad_attempt == MAX_PAD_RETRIES - 1: padding_generation_failed = True; break
                    time.sleep(RETRY_INITIAL_DELAY * (2 ** pad_attempt))

            if padding_generation_failed: logger.error("Padding generation failed."); add_padding = False
            elif padding_text:
                scenes.append(padding_text); tokens_used += ptoks; last_scene_text = padding_text; pad_count += 1
                logger.info(f"Appended strictly validated padding {pad_count}, tokens used: {ptoks}, total: {tokens_used}")
            else: logger.info("No valid padding added."); add_padding = False
            if pad_count >= max_pad_paragraphs: add_padding = False
            if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN: logger.warning("Token limit reached during padding."); add_padding = False; break
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
        # logger.debug(f"Failed Narrative Body:\n---\n{narrative_body}\n---") # Uncomment to debug
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
    few_shot_examples = SHOT_EXAMPLES.get(PROMPT_SHOT_COUNT, "")
    final_prompt = few_shot_examples + narrative_body + question + judge_instructions
    logger.info(f"Successfully generated and validated narrative prompt. Total tokens used (estimated): {tokens_used}")
    return final_prompt.strip()


# --- ast_to_prefix (Keep as is) ---
def ast_to_prefix(node: Node) -> str:
    """Convert an AST to a prefix notation string."""
    if isinstance(node, Atom): return str(node.n)
    parts = [node.op] + [ast_to_prefix(child) for child in node.children]
    return "(" + " ".join(parts) + ")"


# --- HELPER FOR SINGLE SAMPLE GENERATION ---
def generate_single_sample(sample_index: int) -> dict | None:
    """Generates a single Verbose ListOps sample with REVISED-REVISED strict validation."""
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
                 "atom_min_value": ATOM_MIN_VALUE, # Add atom range to metadata
                 "atom_max_value": ATOM_MAX_VALUE,
                 "prompt_shot_count": PROMPT_SHOT_COUNT,
                 "max_beat_retries": MAX_BEAT_RETRIES,
                 "max_pad_retries": MAX_PAD_RETRIES,
                 "validation_mode": "revised_revised_strict_beat_padding_final" # Update mode description
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

# --- MAIN FUNCTION (Keep as is) ---
def main(num_samples: int = NUM_SAMPLES_TO_GENERATE, output_file: str = OUTPUT_FILENAME, max_workers: int = DEFAULT_MAX_WORKERS):
    """Orchestrates generation with REVISED-REVISED strict validation."""
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

# --- __main__ block (Keep as is) ---
if __name__ == "__main__":
    main(max_workers=DEFAULT_MAX_WORKERS)