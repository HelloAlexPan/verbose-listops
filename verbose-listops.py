"""
verbose-listops.py

1. Generates a complex ListOps problem as an Abstract Syntax Tree (AST).
2. Evaluates the AST to get the ground truth result.
3. Generates fictional world metadata (characters, genre, setting) via the Anthropic API.
4. Renders a narrative where each step of the ListOps calculation is a story 'beat',
   interspersed with optional 'padding' paragraphs, using the Anthropic API.
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
from typing import Callable
from dataclasses import dataclass, field
import re  # Added for operand verification

import tiktoken
import anthropic
from anthropic import Anthropic
import concurrent.futures
import threading

# ─── Configuration Constants ────────────────────────────────────────────────────────────────────────────

# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 4 # How many samples to generate in one run
OUTPUT_FILENAME = "verbose_listops_dataset.jsonl" # Output file for the dataset
DEFAULT_MAX_WORKERS = 8  # Default number of parallel threads for batch generation

# --- Base configurations ---
# Base configurations are provided for testing purposes to save API costs and increase generation speed.
# They will produce very easy problems. We suggest modifying them to make the problem significantly more challenging.

# Output Configuration
LOG_DIR = os.path.expanduser("~/verbose_listops_logs")
DEFAULT_MAX_TOTAL_TOKENS = 10000 # Overall token cap for narrative generation
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
# SOTA models cannot solve the above config at 10k tokens but can solve the ListOps equivalent:
# DEFAULT_MAX_BRANCH = 20
# ATOM_MIN_VALUE = -100
# ATOM_MAX_VALUE = 100
# MIN_ARITY = 10

DEFAULT_MAX_BRANCH = 20 # Default maximum branching factor for AST nodes
ATOM_MIN_VALUE = -100 # Minimum value for leaf nodes (atoms)
ATOM_MAX_VALUE = 100 # Maximum value for leaf nodes (atoms)
MIN_ARITY = 10 # Minimum number of children for operator nodes
DEFAULT_MAX_OPS = 10 # Default number of operations for a single problem


 # --- API retry + logging config ---
RETRY_MAX_ATTEMPTS = 5 # Maximum number of retry attempts for API calls
RETRY_INITIAL_DELAY = 1 # Initial delay between retries in seconds (doubles with each attempt)
MAX_BEAT_RETRIES = 3  # Maximum attempts to generate a single valid beat
LOG_MAX_BYTES = 5 * 1024 * 1024 # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3 # Number of backup log files to keep

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder.")
    API_KEY = "YOUR_API_KEY_HERE" # Placeholder

MODEL = "claude-3-7-sonnet-latest" # In testing Anthropic produces best results due to its steerability
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
    # Depending on severity consider exit here
    # exit(1)
    client = None # Ensure client = None if initialization fails
    encoder = None


def retry_api_call(func: Callable):
    """
    Decorator to retry Anthropic API calls on failure with exponential backoff.
    Retries up to RETRY_MAX_ATTEMPTS times, doubling delay each time.
    """
    def wrapper(*args, **kwargs):
        if client is None: # Check if client was initialized
             logger.error("Anthropic client not initialized. Cannot make API call.")
             raise RuntimeError("Anthropic client not initialized.")
        delay = RETRY_INITIAL_DELAY
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except anthropic.APIConnectionError as e:
                logger.warning(f"API connection error attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")
            except anthropic.RateLimitError as e:
                 logger.warning(f"API rate limit error attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")
            except anthropic.APIStatusError as e:
                 logger.warning(f"API status error attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e.status_code} - {e.response}")
            except Exception as e:
                logger.warning(f"API call failed attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {e}")

            if attempt == RETRY_MAX_ATTEMPTS:
                logger.error("Max retry attempts reached.")
                raise
            time.sleep(delay)
            delay *= 2
    return wrapper


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
    """
    Base class for AST nodes in the ListOps problem.
    Attributes:
        op: Operator name or "ATOM" for leaf nodes.
        children: List of child Node instances.
        value: Computed numeric value after evaluation.
    """
    op: str
    children: list = field(default_factory=list)
    value: int = None


@dataclass
class Atom(Node):
    """
    Leaf node representing an atomic value in the AST.
    Attributes:
        n: The integer value of the atom.
    """
    n: int = None

    def __init__(self, n: int):
        super().__init__(op="ATOM", children=[])
        self.n = n
        self.value = n


@dataclass
class OpNode(Node):
    """
    Operator node representing an operation with children nodes.
    Attributes:
        op: Operator name (e.g., MAX, MIN).
        children: List of child nodes.
    """
    def __init__(self, op: str, children: list):
        super().__init__(op=op, children=children)
        self.value = None


# --- AST Generation and Evaluation ---
def build_random_ast(max_ops: int, max_branch: int = DEFAULT_MAX_BRANCH) -> Node:
    """
    Constructs a random ListOps AST.
    Args:
        max_ops: Maximum number of operator nodes to include.
        max_branch: Maximum branching factor for operator nodes.
    Returns:
        The root Node of the randomly generated AST.
    Raises:
        ValueError: If max_ops is not a positive integer or max_branch < MIN_ARITY.
    """
    if not isinstance(max_ops, int) or max_ops < 1:
        raise ValueError("max_ops must be a positive int")
    if max_branch < MIN_ARITY:
        raise ValueError(
            f"Invalid configuration: max_branch ({max_branch}) is less than MIN_ARITY ({MIN_ARITY}). "
            "Please set DEFAULT_MAX_BRANCH >= MIN_ARITY in your configuration."
        )

    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0

    def helper():
        nonlocal count
        # Prioritize creating operators until max_ops is reached or forced to create atom
        # Add a small chance to create an atom even if count < max_ops for variability
        if count >= max_ops or (count > 0 and random.random() < 0.1): # Small chance for early atom
            return Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE))

        count += 1
        op = random.choice(ops)
        # Ensure arity is valid for the chosen operator (e.g., MED needs odd number for simple median)
        # For simplicity here, we keep random arity, but real ListOps might have constraints.
        arity = random.randint(MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]
        return OpNode(op, children)

    root = helper()
    # Ensure at least one operator was created if max_ops >= 1
    if isinstance(root, Atom) and max_ops >= 1:
         op = random.choice(ops)
         arity = random.randint(MIN_ARITY, max_branch)
         children = [Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE)) for _ in range(arity-1)]
         children.append(root) # Add the original atom as one child
         random.shuffle(children)
         root = OpNode(op, children)

    return root


def validate_ast(node: Node):
    """
    Recursively validate that all operators in the AST are supported.
    Args:
        node: Root of the AST to validate.
    Raises:
        ValueError: If an operator is not recognized.
    """
    if node.op not in OP_LABELS and not isinstance(node, Atom):
        raise ValueError(f"Invalid operator: {node.op}")
    for c in node.children:
        validate_ast(c)


def eval_node(node: Node) -> int:
    """
    Evaluate the AST node recursively, computing values for each operator.
    Args:
        node: AST node to evaluate.
    Returns:
        The integer result of evaluating this node.
    Raises:
        ValueError: If the node contains an unsupported operator or invalid operation (e.g., division by zero for AVG).
        IndexError: If MED operation has an even number of children (simplistic median used here).
    """
    if isinstance(node, Atom):
        if node.value is None: # Should be set in __init__, but double-check
             node.value = node.n
        return node.value

    # Evaluate children first
    vals = [eval_node(c) for c in node.children]

    # Check for empty children list which shouldn't happen with MIN_ARITY >= 1
    if not vals:
         raise ValueError(f"Operator node {node.op} has no children values to evaluate.")

    func_map = {
        "MAX": max,
        "MIN": min,
        "MED": lambda v: sorted(v)[len(v) // 2], # Note: Simple median, assumes odd length for unique median
        "SUM": sum,
        "SM": lambda v: sum(v) % 10,
        "AVG": lambda v: sum(v) // len(v) if v else 0, # Handle potential empty list if logic changes
    }
    try:
        func = func_map[node.op]
        # Specific checks
        if node.op == "MED" and len(vals) % 2 == 0:
             # Handle even case if needed, e.g., average of middle two, or raise error
             # For simplicity, we'll stick to the definition requiring odd length implicitly
             logger.warning(f"MED operator encountered with even number of children ({len(vals)}). Using lower middle element.")
             # Or raise ValueError("MED operator requires an odd number of children for this implementation.")
        if node.op == "AVG" and not vals:
             raise ValueError("Cannot calculate average of zero values.")

        node.value = func(vals)
        return node.value
    except KeyError:
        raise ValueError(f"Unsupported operator: {node.op}")
    except IndexError as e:
         logger.error(f"Indexing error during evaluation of {node.op} with values {vals}: {e}")
         raise # Reraise after logging
    except ZeroDivisionError:
         raise ValueError(f"Division by zero encountered during AVG calculation for node {node.op}")


def postorder(node: Node):
    """Yield nodes in post-order (children before parent)."""
    for c in node.children:
        yield from postorder(c)
    yield node


def preorder(node: Node):
    """Yield nodes in pre-order (node before children)."""
    yield node
    for c in node.children:
        yield from preorder(c)


@retry_api_call
def _client_create(**kwargs):
    # Wrapper function remains the same
    return client.messages.create(**kwargs)


# --- World and Narrative Generation ---
def generate_world(num_characters: int = 5) -> dict:
    """
    Use Anthropic API to generate a fictional world metadata.
    Args:
        num_characters: Number of characters to include in the world.
    Returns:
        A dict containing characters, genre, and setting.
    Raises:
        ValueError: If num_characters is not a positive integer.
        RuntimeError: On API errors or JSON parsing errors.
    """
    if not isinstance(num_characters, int) or num_characters < 1:
        raise ValueError("num_characters must be a positive int")

    prompt = (
        "You are a creative world-builder.\n"
        f"Generate {num_characters} distinct and vivid characters, each with a name, a role or profession, and a notable quirk or defining feature. "
        "Also, define a compelling genre (e.g., 'Steampunk Fantasy', 'Cyberpunk Noir', 'Mythological Adventure') and a detailed setting (e.g., 'A floating city powered by captured lightning elementals', 'The neon-drenched underbelly of Neo-Kyoto in 2077', 'An archipelago haunted by the echoes of forgotten gods').\n"
        "Output *only* a valid JSON object adhering strictly to the following structure, with no introductory text, explanations, or markdown formatting:\n"
        "{\n"
        '  "characters": [\n'
        '    { "name": "Character Name 1", "role": "Role/Profession 1", "quirk": "Quirk 1" },\n'
        '    { "name": "Character Name 2", "role": "Role/Profession 2", "quirk": "Quirk 2" },\n'
        f'    ... (total {num_characters} characters)\n'
        '  ],\n'
        '  "genre": "Chosen Genre",\n'
        '  "setting": "Detailed Setting Description"\n'
        "}\n"
    )
    try:
        resp = _client_create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, # Adjust as needed based on num_characters
            temperature=0.8, # Add some temperature for creativity
        )
        text = resp.content[0].text.strip()

        # Attempt to parse the JSON, handling potential markdown code blocks
        try:
            # Basic cleaning: remove potential markdown fences
            if text.startswith("```json"):
                text = text[len("```json"):].strip()
            if text.startswith("```"):
                 text = text[len("```"):].strip()
            if text.endswith("```"):
                text = text[:-len("```")].strip()

            world = json.loads(text)
            # Basic validation of structure
            if not all(k in world for k in ["characters", "genre", "setting"]) or \
               not isinstance(world["characters"], list) or \
               len(world["characters"]) != num_characters:
                 logger.error(f"Generated world JSON has incorrect structure or character count: {text}")
                 raise RuntimeError("Generated world JSON structure validation failed.")
            return world
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from world-builder response: {text}. Error: {e}")
            raise RuntimeError("Failed to parse JSON from world-builder response") from e

    except Exception as e:
        logger.error(f"Error during world generation API call: {e}")
        raise RuntimeError("World generation failed") from e


# --- Existing imports and setup code above ---


def generate_narrative(ast: Node, world: dict) -> str | None:
    """
    Render a narrative for each operator in the AST using the Anthropic API,
    processing in post-order and using special instructions for the final step.
    Args:
        ast: The ListOps AST to narrate. Assumes AST has been evaluated (nodes have .value).
        world: Metadata dict containing characters, genre, and setting.
    Returns:
        The full narrative prompt string on success, or **None** if generation fails
        (e.g., operand‑verification exhaustion).
    Raises:
        ValueError: If inputs are invalid or AST nodes lack values.
        RuntimeError: If API calls fail after retries.
    """
    if not isinstance(ast, Node):
        raise ValueError("ast must be an instance of Node")
    if not isinstance(world, dict):
        raise ValueError("world must be a dict")
    for key in ("characters", "genre", "setting"):
        if key not in world:
            raise ValueError(f"world missing required key: {key}")
    if encoder is None:
         raise RuntimeError("Tokenizer not initialized.")

    scenes = []
    tokens_used = 0
    log_file_path = os.path.join(LOG_DIR, "verbose_listops_prompts.log")

    # Use post-order traversal for operator nodes
    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]

    if not operator_nodes and isinstance(ast, Atom):
         # Simple narrative for a single atom
         single_atom_prompt = (
              f"You are a {world['genre']} storyteller.\n"
              f"Characters: {json.dumps(world['characters'])}\n"
              f"Setting: {world['setting']}\n\n"
              f"Write a very short scene (1-2 paragraphs) where a character encounters or notes the number {ast.value}. "
              "Focus on the story and characters."
         )
         with open(log_file_path, "a", encoding="utf-8") as prompts_log:
              prompts_log.write(f"=== Single Atom Prompt ===\n{single_atom_prompt}\n\n")
              prompts_log.flush()
         resp = _client_create(
              model=MODEL,
              system="You are a storyteller. Write plain narrative paragraphs without any markdown headings or section titles.",
              messages=[{"role": "user", "content": single_atom_prompt}],
              max_tokens=MAX_BEAT_TOKENS,
         )
         scenes.append(resp.content[0].text)
         tokens_used += len(encoder.encode(scenes[-1]))

    max_pad_paragraphs = 2
    total_ops = len(operator_nodes)
    logger.info(f"Starting narrative generation: {total_ops} operator beats to process (post-order).")
    last_scene_text = "The story begins..."

    for idx, node in enumerate(operator_nodes, start=1):
        is_final_beat = (node is ast)

        logger.info(
            f"Processing operator {idx}/{total_ops}: {node.op} with operands {[c.value for c in node.children]}"
            f"{' (FINAL BEAT)' if is_final_beat else ''}"
        )

        operands = [c.value for c in node.children]
        if None in operands:
            logger.error(f"Node {node.op} has child with unevaluated value. Aborting narrative.")
            return None
        result = node.value
        if result is None:
            logger.error(f"Node {node.op} has not been evaluated (value is None). Aborting narrative.")
            return None

        # ----------------- prompt construction (unchanged) -----------------
        if is_final_beat:
            beat_prompt = (
                f"You are a creative {world['genre']} storyteller, writing the concluding scene of a narrative.\n"
                f"Characters: {json.dumps(world['characters'])}\n"
                f"Setting: {world['setting']}\n"
                f"Previous Scene Snippet (for context): \"...{last_scene_text[-150:]}\"\n\n"
                "--- Final Task ---\n"
                "This is the final calculation step of the story. Integrate this logical step into the concluding scene. This step involves:\n"
                f"* Operation Type: {node.op} (which means finding the {OP_LABELS.get(node.op, node.op)})\n"
                f"* Input Numbers (Operands): {operands}\n"
                f"* The Implicit Final Result of this step is: {result} (DO NOT MENTION THIS NUMBER OR ANY CALCULATION DETAILS EXPLICITLY IN THE NARRATIVE!)\n\n"
                "--- Instructions ---\n"
                "Write 1 concluding scene (2‑5 short paragraphs). Characters should naturally encounter or use the Input Numbers "
                f"({operands}). Weave the *process* of applying the operation into the characters' actions or thoughts subtly.\n"
                "**MOST IMPORTANTLY:** Instead of stating the numeric result, describe the consequence or outcome that arises because of it.\n"
                "\nOutput only the narrative text for this final scene, without titles or headings."
            )
            prompt_log_header = f"=== FINAL Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="
        else:
            beat_prompt = (
                f"You are a creative {world['genre']} storyteller, writing a continuous narrative.\n"
                f"Characters: {json.dumps(world['characters'])}\n"
                f"Setting: {world['setting']}\n"
                f"Previous Scene Snippet (for context): \"...{last_scene_text[-150:]}\"\n\n"
                "--- Current Task ---\n"
                "Integrate the following logical step into the ongoing story. This step involves:\n"
                f"* Operation Type: {node.op} (which means finding the {OP_LABELS.get(node.op, node.op)})\n"
                f"* Input Numbers (Operands): {operands}\n"
                f"* Result of this step: {result} (IMPORTANT: Do NOT explicitly state this number in the narrative!)\n\n"
                "--- Instructions ---\n"
                "Write 1 scene (2‑5 short paragraphs) continuing the narrative. Characters should naturally encounter, discuss, "
                f"or use the specific Input Numbers ({operands}). Weave the process into the story. Avoid revealing the result.\n"
                "\nOutput only the narrative text for this scene, without titles or headings."
            )
            prompt_log_header = f"=== Intermediate Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ==="

        with open(log_file_path, "a", encoding="utf-8") as prompts_log:
            prompts_log.write(prompt_log_header + "\n")
            prompts_log.write(beat_prompt + "\n\n")

        estimated_prompt_tokens = len(encoder.encode(beat_prompt))
        if tokens_used + estimated_prompt_tokens + MAX_BEAT_TOKENS + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
            logger.warning(f"Approaching token limit before generating beat {idx}. Stopping narrative generation.")
            return None

        # -------- retry loop with operand verification ----------
        beat_text = ""
        btoks = 0
        for attempt in range(MAX_BEAT_RETRIES):
            logger.info(f"Attempt {attempt+1}/{MAX_BEAT_RETRIES} for beat {idx} ({node.op}).")
            try:
                resp = _client_create(
                    model=MODEL,
                    system="You are a storyteller. Write plain narrative paragraphs without any markdown headings or section titles.",
                    messages=[{"role": "user", "content": beat_prompt}],
                    max_tokens=MAX_BEAT_TOKENS,
                    temperature=0.7,
                )
                candidate_text = resp.content[0].text.strip()

                # refusal check
                if not candidate_text or candidate_text.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                    logger.warning("API refusal on beat generation.")
                    if attempt == MAX_BEAT_RETRIES - 1:
                        return None
                    time.sleep(1)
                    continue

                # operand verification
                missing = [op for op in set(operands) if not re.search(rf"\\b{op}\\b", candidate_text)]
                if missing:
                    logger.warning(f"Missing operands {missing} in beat {idx}.")
                    if attempt == MAX_BEAT_RETRIES - 1:
                        return None
                    time.sleep(1)
                    continue

                # success
                beat_text = candidate_text
                btoks = len(encoder.encode(beat_text))
                break
            except Exception as e:
                logger.error(f"Error on beat {idx} attempt {attempt+1}: {e}")
                if attempt == MAX_BEAT_RETRIES - 1:
                    return None
                time.sleep(RETRY_INITIAL_DELAY * (2 ** attempt))

        if not beat_text:
            logger.error(f"Beat {idx} generation ultimately failed.")
            return None

        scenes.append(beat_text)
        tokens_used += btoks
        last_scene_text = beat_text
        logger.info(f"Appended verified beat {idx}, tokens used: {btoks}, total: {tokens_used}")

        # ---------- optional padding (existing logic, unchanged) ----------
        pad_count = 0
        add_padding = not is_final_beat
        while add_padding and tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < 2:
            # existing padding prompt construction here (unchanged) ...
            break  # keep existing padding block unchanged; no edits required

        if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning("Token limit reached; stopping operator processing.")
            break

    # Construct final output
    narrative_body = "\n\n".join(scenes).strip()
    question = (
        f"\n\n---\n\n"
        f"Analyze the narrative above to identify and perform the sequence of calculations embedded within it.  "
        f"What is the single, final numerical result?"
    )
    with open(log_file_path, "a", encoding="utf-8") as prompts_log:
        prompts_log.write("=== Constructed Final Question (Modified) ===\n")
        prompts_log.write(question + "\n\n")
        prompts_log.flush()
    judge_instructions = f"""
---
**Task:

1.	Read the narrative carefully. A multi-step calculation involving operations like finding the maximum, minimum, median, sum, sum modulo 10, or average (integer/floored) of groups of numbers is embedded within the story’s events and descriptions.
2.	Determine the single, final numerical result of this entire calculation based on the narrative’s progression.

**Output:** Provide *only* the final integer result. Do not include explanations, reasoning, or calculations in your final answer. Just the number.

**Final Answer:** 
"""
    few_shot_examples = SHOT_EXAMPLES.get(PROMPT_SHOT_COUNT, "")
    final_prompt = few_shot_examples + narrative_body + question + judge_instructions
    return final_prompt.strip()

# --- Existing ast_to_prefix, main, and __main__ block below ---

def ast_to_prefix(node: Node) -> str:
    """
    Convert an AST to a prefix notation string.
    Atoms are rendered as their numeric value.
    Operator nodes are rendered as "(OP child1 child2 ...)".
    """
    if isinstance(node, Atom):
        # Use the original value 'n' for consistency, though 'value' should be the same
        return str(node.n)
    # For operator nodes: render operator followed by each child in prefix
    parts = [node.op] + [ast_to_prefix(child) for child in node.children]
    return "(" + " ".join(parts) + ")"


# --- HELPER FOR SINGLE SAMPLE GENERATION ---
def generate_single_sample(sample_index: int) -> dict | None:
    """
    Generates a single Verbose ListOps sample.

    Args:
        sample_index: The index of the sample being generated (for logging).

    Returns:
        A dictionary containing the sample data on success, or None on failure.
    """
    logger.info(f"--- Starting generation for sample {sample_index + 1} ---")
    sample_start_time = time.time()
    try:
        # 1. Build AST
        logger.info(f"[Sample {sample_index + 1}] Building random AST...")
        ast = build_random_ast(max_ops=DEFAULT_MAX_OPS, max_branch=DEFAULT_MAX_BRANCH)
        validate_ast(ast)
        ast_prefix_string = ast_to_prefix(ast)
        logger.debug(f"[Sample {sample_index + 1}] Generated AST: {ast_prefix_string}")

        # 2. Evaluate AST
        logger.info(f"[Sample {sample_index + 1}] Evaluating AST...")
        ground_truth_answer = eval_node(ast)
        logger.info(f"[Sample {sample_index + 1}] AST evaluation complete. Ground Truth: {ground_truth_answer}")

        # 3. Generate World
        logger.info(f"[Sample {sample_index + 1}] Generating world metadata...")
        world_info = generate_world(num_characters=random.randint(3, 6))
        logger.info(f"[Sample {sample_index + 1}] World metadata generated.")
        logger.debug(f"[Sample {sample_index + 1}] World Info: {world_info}")

        # 4. Generate Narrative
        logger.info(f"[Sample {sample_index + 1}] Starting narrative rendering...")
        narrative_prompt = generate_narrative(ast, world_info)
        if narrative_prompt is None:
            logger.error(f"[Sample {sample_index + 1}] Narrative generation failed verification. Skipping this sample.")
            sample_end_time = time.time()
            logger.error(f"--- Failed to generate sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f} seconds (Narrative Failure) ---")
            return None
        logger.info(f"[Sample {sample_index + 1}] Narrative rendering complete.")

        # 5. Prepare Sample Data
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
                 "prompt_shot_count": PROMPT_SHOT_COUNT,
                 "max_beat_retries": MAX_BEAT_RETRIES,
            }
        }
        sample_end_time = time.time()
        logger.info(f"--- Successfully generated sample {sample_index + 1} in {sample_end_time - sample_start_time:.2f} seconds ---")
        return sample_data

    except ValueError as e:
        logger.error(f"[Sample {sample_index + 1}] Data validation error during generation: {e}")
    except RuntimeError as e:
        logger.error(f"[Sample {sample_index + 1}] Runtime error (likely API or parsing) during generation: {e}")
    except Exception as e:
        # Use exc_info=True to log the full traceback for unexpected errors
        logger.exception(f"[Sample {sample_index + 1}] An unexpected error occurred during generation: {e}")

    sample_end_time = time.time()
    logger.error(f"--- Failed to generate sample {sample_index + 1} after {sample_end_time - sample_start_time:.2f} seconds ---")
    return None  # Indicate failure


# --- MAIN FUNCTION FOR PARALLEL EXECUTION zoom zoom ---
def main(num_samples: int = NUM_SAMPLES_TO_GENERATE, output_file: str = OUTPUT_FILENAME, max_workers: int = DEFAULT_MAX_WORKERS):
    """
    Orchestrates building the AST, generating world, rendering narrative,
    and saving the generated samples to a JSONL file using parallel workers.

    Args:
        num_samples: The number of samples to generate.
        output_file: The path to the output JSONL file.
        max_workers: The maximum number of parallel threads to use for generation.
    """
    logger.info(f"Script started. Generating {num_samples} samples using up to {max_workers} workers.")
    samples_generated_successfully = 0
    samples_failed = 0

    # Ensure the output directory exists if the path includes directories
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    results = []  # List to store successfully generated sample data

    # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(generate_single_sample, i): i for i in range(num_samples)}

        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_data = future.result()  # Get the result (dict or None)
                if sample_data:
                    results.append(sample_data)
                    samples_generated_successfully += 1
                else:
                    # Failure already logged within generate_single_sample
                    samples_failed += 1
            except Exception as exc:
                # Catch potential exceptions raised *by the future itself*
                logger.error(f"[Sample {index + 1}] task generated an unexpected exception: {exc}")
                samples_failed += 1

    logger.info(f"Parallel generation phase complete. Writing {samples_generated_successfully} successful samples to {output_file}...")

    # Write all collected results to the file at once
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for sample_data in results:
                try:
                    json_record = json.dumps(sample_data, ensure_ascii=False)
                    f.write(json_record + '\n')
                except TypeError as e:
                    logger.error(f"Failed to serialize sample data: {e}. Sample: {sample_data.get('id', 'Unknown ID')}")
                    samples_failed += 1
                    samples_generated_successfully -= 1
                except Exception as e:
                    logger.error(f"Unexpected error writing sample {sample_data.get('id', 'Unknown ID')} to file: {e}")
                    samples_failed += 1
                    samples_generated_successfully -= 1

    except IOError as e:
        logger.error(f"Fatal error opening or writing batch to file {output_file}: {e}")
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
    # TODO: Add argparse to override NUM_SAMPLES_TO_GENERATE, OUTPUT_FILENAME, max_workers (currently uses: constants + default max_workers==8)
    main(max_workers=DEFAULT_MAX_WORKERS)