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

import tiktoken
import anthropic
from anthropic import Anthropic

# ─── Configuration Constants ────────────────────────────────────────────────────────────────────────────

# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 2 # <<< How many samples to generate in one run
OUTPUT_FILENAME = "verbose_listops_dataset.jsonl" # <<< Output file for the dataset

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
        "Example:\n"
        "Narrative: “The recipe called for the median weight of three ingredients weighing 7, 2, and 5 grams. The alchemist needed to select the one with that middle weight.”\n"
        "Answer: 5\n"
        "</Prompt Shot>\n"
    ),
    2: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: “Two energy readings were displayed: 9 units and 4 units. The system required activating the conduit with the higher reading.”\n"
        "Answer: 9\n\n"
        "Example 2:\n"
        "Narrative: “The guard collected tolls from the three wagons: 3 silver, 1 silver, and 5 silver pieces. He needed to report the total amount collected.”\n"
        "Answer: 9\n"
        "</Prompt Shot>\n"
    ),
    3: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: “Two security codes were presented: 81 and 27. Access was granted using the code with the smaller value.”\n"
        "Answer: 27\n\n"
        "Example 2:\n"
        "Narrative: “Four sensor pings registered distances of 6, 8, 1, and 3 meters. The protocol involved summing these distances and using only the final digit for the lock combination.”\n"
        "Answer: 8\n\n" # 6+8+1+3 = 18. Sum Mod 10 is 8.
        "Example 3:\n"
        "Narrative: “The three trainees completed the obstacle course in 10, 12, and 11 minutes. Their squad leader calculated their average time, rounding down to the nearest whole minute.”\n"
        "Answer: 11\n" # (10+12+11)/3 = 33/3 = 11.
        "</Prompt Shot>\n"
    ),
}

# --- AST Random ListOps problem gen params ---
# SOTA models cannot solve the below config at 10k tokens but can solve the ListOps equivalent:
# DEFAULT_MAX_BRANCH = 20
# ATOM_MIN_VALUE = -100
# ATOM_MAX_VALUE = 100
# MIN_ARITY = 10

DEFAULT_MAX_BRANCH = 3 # Default maximum branching factor for AST nodes
ATOM_MIN_VALUE = 0 # Minimum value for leaf nodes (atoms)
ATOM_MAX_VALUE = 9 # Maximum value for leaf nodes (atoms)
MIN_ARITY = 2 # Minimum number of children for operator nodes
DEFAULT_MAX_OPS = 10 # Default number of operations for a single problem


# --- API retry + logging config ---
RETRY_MAX_ATTEMPTS = 5 # Maximum number of retry attempts for API calls
RETRY_INITIAL_DELAY = 1 # Initial delay between retries in seconds (doubles with each attempt)
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

def generate_narrative(ast: Node, world: dict) -> str:
    """
    Render a narrative for each operator in the AST using the Anthropic API, then ask a follow-up question.
    Args:
        ast: The ListOps AST to narrate. Assumes AST has been evaluated (nodes have .value).
        world: Metadata dict containing characters, genre, and setting.
    Returns:
        The full narrative plus final question and judge instructions as a single string.
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
    log_file_path = os.path.join(LOG_DIR, "verbose_listops_prompts.log") # Log prompts separately

    # Get evaluated operator nodes in processing order (pre-order seems more narrative-friendly)
    operator_nodes = [n for n in preorder(ast) if not isinstance(n, Atom)]
    if not operator_nodes:
         logger.warning("AST contains no operator nodes. Narrative will be minimal.")
         if isinstance(ast, Atom):
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

    max_pad_paragraphs = 2 # Max padding sections *between* operator beats
    total_ops = len(operator_nodes)
    logger.info(f"Starting narrative generation: {total_ops} operator beats to process.")

    last_scene_text = "The story begins..." # Initial context

    for idx, node in enumerate(operator_nodes, start=1):
        logger.info(
            f"Processing operator {idx}/{total_ops}: {node.op} with operands {[c.value for c in node.children]}"
        )

        try:
            operands = [c.value for c in node.children]
            if None in operands:
                 raise ValueError(f"Node {node.op} has child with unevaluated value.")
            result = node.value
            if result is None:
                 raise ValueError(f"Node {node.op} has not been evaluated (value is None).")
        except AttributeError:
             raise ValueError("AST nodes seem malformed or not evaluated (missing .value)")
        except Exception as e:
             logger.error(f"Error accessing node values for {node.op}: {e}")
             raise

        # --- Generate Operator Beat ---
        # Consider adding a specific instruction for the *final* beat (when node == ast)
        # to encourage a natural narrative conclusion implying the operation.
        # For now, using the same prompt structure for all beats.
        beat_prompt = (
            f"You are a creative {world['genre']} storyteller, writing a continuous narrative.\n"
            f"Characters: {json.dumps(world['characters'])}\n"
            f"Setting: {world['setting']}\n"
            f"Previous Scene Snippet (for context): \"...{last_scene_text[-150:]}\"\n\n"
            "--- Current Task ---\n"
            "Integrate the following logical step into the ongoing story. This step involves:\n"
            f"* Operation Type: {node.op} (which means finding the {OP_LABELS.get(node.op, node.op)})\n"
            f"* Input Numbers (Operands): {operands}\n"
            f"* Result of this step: {result} (IMPORTANT: Do NOT explicitly state this number '{result}' in the narrative!)\n\n"
            "--- Instructions ---\n"
            "Write 1 scene (2-5 short paragraphs) continuing the narrative. Characters should naturally encounter, discuss, or use the specific Input Numbers ({operands}). Weave the *process* of applying the operation (e.g., finding the median, summing then taking modulo 10, identifying the smallest) into the characters' actions, thoughts, or dialogue. The numbers should appear organically within the story's context (e.g., 'readings of {operands[0]} and {operands[1]}', 'they considered the {len(operands)} data points: {', '.join(map(str, operands))}'). Focus on vivid storytelling and character interaction, embedding the calculation logic subtly. Avoid mathematical jargon unless it fits the genre/character. Do NOT reveal the numeric result ({result})."
            "\nOutput only the narrative text for this scene, without titles or headings."
        )

        with open(log_file_path, "a", encoding="utf-8") as prompts_log:
            prompts_log.write(f"=== Operator Beat Prompt {idx}/{total_ops} (Op: {node.op}) ===\n")
            prompts_log.write(beat_prompt + "\n\n")
            prompts_log.flush()

        estimated_prompt_tokens = len(encoder.encode(beat_prompt))
        if tokens_used + estimated_prompt_tokens + MAX_BEAT_TOKENS + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
             logger.warning(f"Approaching token limit before generating beat {idx}. Stopping narrative generation.")
             break

        resp = _client_create(
            model=MODEL,
            system="You are a storyteller. Write plain narrative paragraphs without any markdown headings or section titles.",
            messages=[{"role": "user", "content": beat_prompt}],
            max_tokens=MAX_BEAT_TOKENS,
            temperature=0.7,
        )
        beat_text = resp.content[0].text.strip()
        btoks = len(encoder.encode(beat_text))

        if not beat_text or beat_text.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
             logger.warning(f"API refused to generate beat {idx} for op {node.op}. Stopping narrative generation.")
             break

        scenes.append(beat_text)
        tokens_used += btoks
        last_scene_text = beat_text
        logger.info(f"Generated beat {idx} ({node.op}), tokens used: {btoks}, total tokens: {tokens_used}")

        # --- Optional Padding ---
        pad_count = 0
        add_padding = (idx < total_ops)

        while add_padding and tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
            pad_prompt = (
                f"You are the same {world['genre']} storyteller, continuing the narrative smoothly.\n"
                f"Characters: {json.dumps(world['characters'])}\n"
                f"Setting: {world['setting']}\n"
                f"Last Scene Snippet: \"...{last_scene_text[-200:]}\"\n\n"
                "--- Current Task ---\n"
                "Write 1-3 short paragraphs of 'padding' that continue the story from the last scene. This could involve:\n"
                "*   Character reflection or dialogue unrelated to the main calculation.\n"
                "*   Description of the environment or atmosphere.\n"
                "*   A minor, unrelated event or observation.\n"
                "*   A hint of a side-plot or mystery.\n\n"
                "--- IMPORTANT ---\n"
                "Do NOT introduce new numbers or calculations. Do NOT mention the previous operation ({node.op}) or its inputs/results. This is purely transitional or world-building content.\n"
                "Output only the narrative text for this padding, without titles or headings."
            )

            with open(log_file_path, "a", encoding="utf-8") as prompts_log:
                prompts_log.write(f"=== Padding Prompt {pad_count+1} (after Op: {node.op}) ===\n")
                prompts_log.write(pad_prompt + "\n\n")
                prompts_log.flush()

            estimated_pad_prompt_tokens = len(encoder.encode(pad_prompt))
            if tokens_used + estimated_pad_prompt_tokens + MAX_PAD_TOKENS + SAFETY_MARGIN > MAX_TOTAL_TOKENS:
                logger.warning(f"Approaching token limit before generating padding {pad_count+1}. Skipping padding.")
                break

            pad_resp = _client_create(
                model=MODEL,
                system="Continue the narrative with transitional or world-building content. Write plain paragraphs without headings.",
                messages=[{"role": "user", "content": pad_prompt}],
                max_tokens=MAX_PAD_TOKENS,
                temperature=0.6,
            )
            pad_text = pad_resp.content[0].text.strip()
            ptoks = len(encoder.encode(pad_text))

            if not pad_text or pad_text.lower().startswith(("i cannot", "i'm sorry", "i am unable", "based on the previous", "to continue the calculation")):
                logger.info(f"Padding refused or irrelevant after operator {idx}, stopping padding for this beat.")
                break

            if tokens_used + ptoks > MAX_TOTAL_TOKENS - SAFETY_MARGIN:
                logger.warning(f"Padding {pad_count+1} exceeds token limit. Discarding padding.")
                break

            scenes.append(pad_text)
            tokens_used += ptoks
            last_scene_text = pad_text
            pad_count += 1
            logger.info(f"Generated padding {pad_count}, tokens used: {ptoks}, total tokens: {tokens_used}")

        if tokens_used >= MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(f"Reached token limit after processing operator {idx}. Stopping narrative generation.")
            break

    # --- Construct Final Output String ---
    narrative_body = "\n\n".join(scenes).strip()

    # --- MODIFIED QUESTION ---
    # Removed the sentence specifying the final operation type ({label})
    question = (
        f"\n\n---\n\n"
        f"Considering the entire narrative above, what single final number represents the ultimate result "
        f"of the main calculation woven into the story?" # <<< CHANGED LINE (removed final op hint)
    )

    with open(log_file_path, "a", encoding="utf-8") as prompts_log:
        prompts_log.write("=== Constructed Final Question (Modified) ===\n")
        prompts_log.write(question + "\n\n")
        prompts_log.flush()

    # --- MODIFIED JUDGE INSTRUCTIONS ---
    # Removed the specific mention of the final operation type ({label}, {top_op}) in step 5
    judge_instructions = f"""
---
**Instructions for Analysis:**

1.  **Goal:** Your task is to determine the single numerical result of the multi-step calculation embedded within the narrative above.
2.  **Identify Operations:** Read the story carefully to find mentions of calculations or comparisons involving groups of numbers. Look for keywords or descriptions related to:
    *   Maximum / Largest value (MAX)
    *   Minimum / Smallest value (MIN)
    *   Median / Middle value (MED)
    *   Sum / Total value (SUM)
    *   Sum Modulo 10 (SM)
    *   Average value (AVG, integer/floored)
3.  **Extract Numbers:** Note the specific numbers associated with each operation described.
4.  **Determine Structure:** Figure out how these operations are nested or sequenced based on the story's progression. The narrative follows a structure where results of earlier operations often feed into later ones.
5.  **Calculate Final Result:** Perform the calculations following the narrative's inferred hierarchy to determine the single, final numerical result. # <<< CHANGED LINE (removed final op hint)
6.  **Output:** Provide *only* the final single-digit integer result (or the final multi-digit integer if the result is > 9). Do not include explanations, reasoning, or calculations in your final answer. Just the number.

**Final Answer:** """

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


def main(num_samples: int = NUM_SAMPLES_TO_GENERATE, output_file: str = OUTPUT_FILENAME):
    """
    Orchestrates building the AST, generating world, rendering narrative,
    and saving the generated samples to a JSONL file.

    Args:
        num_samples: The number of samples to generate.
        output_file: The path to the output JSONL file.
    """
    logger.info(f"Script started. Generating {num_samples} samples.")
    samples_generated = 0
    samples_failed = 0

    # Ensure the output directory exists if the path includes directories
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    for i in range(num_samples):
        sample_start_time = time.time()
        logger.info(f"--- Generating sample {i+1}/{num_samples} ---")
        try:
            # 1. Build AST
            logger.info("Building random AST...")
            # Adjust max_ops dynamically or keep fixed? Let's keep fixed for now.
            ast = build_random_ast(max_ops=DEFAULT_MAX_OPS, max_branch=DEFAULT_MAX_BRANCH)
            validate_ast(ast)
            ast_prefix_string = ast_to_prefix(ast) # Get prefix string before evaluation modifies node state if needed
            logger.debug(f"Generated AST: {ast_prefix_string}")

            # 2. Evaluate AST
            logger.info("Evaluating AST...")
            ground_truth_answer = eval_node(ast)
            logger.info(f"AST evaluation complete. Ground Truth: {ground_truth_answer}")

            # 3. Generate World
            logger.info("Generating world metadata...")
            # Generate a new world for each sample for diversity
            world_info = generate_world(num_characters=random.randint(3, 6)) # Randomize characters slightly
            logger.info("World metadata generated.")
            logger.debug(f"World Info: {world_info}")

            # 4. Generate Narrative
            logger.info("Starting narrative rendering...")
            narrative_prompt = generate_narrative(ast, world_info)
            logger.info("Narrative rendering complete.")
            # logger.debug(f"Generated Narrative/Prompt:\n{narrative_prompt}") # Log full narrative only if needed

            # 5. Prepare Sample Data
            sample_data = {
                "id": f"verbose_listop_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{i+1}",
                "ast_prefix": ast_prefix_string,
                "ground_truth": ground_truth_answer,
                "world_info": world_info,
                "narrative_prompt": narrative_prompt, # This includes narrative, question, and instructions
                "metadata": {
                     "generation_timestamp": datetime.datetime.now().isoformat(),
                     "model_used": MODEL,
                     "max_ops": DEFAULT_MAX_OPS,
                     "max_branch": DEFAULT_MAX_BRANCH,
                     "prompt_shot_count": PROMPT_SHOT_COUNT,
                }
            }

            # 6. Save Sample to JSONL file
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    json_record = json.dumps(sample_data, ensure_ascii=False)
                    f.write(json_record + '\n')
                samples_generated += 1
                logger.info(f"Successfully generated and saved sample {i+1}/{num_samples}.")
            except IOError as e:
                 logger.error(f"Failed to write sample {i+1} to file {output_file}: {e}")
                 samples_failed += 1
            except Exception as e:
                 logger.error(f"Failed to serialize sample {i+1} data: {e}")
                 samples_failed += 1


        except ValueError as e:
            logger.error(f"Data validation error during generation of sample {i+1}: {e}")
            samples_failed += 1
        except RuntimeError as e:
            logger.error(f"Runtime error (likely API or parsing) during generation of sample {i+1}: {e}")
            samples_failed += 1
        except Exception as e:
            logger.exception(f"An unexpected error occurred during generation of sample {i+1}: {e}") # Use exc_info=True
            samples_failed += 1
        finally:
             sample_end_time = time.time()
             logger.info(f"Sample {i+1} processing took {sample_end_time - sample_start_time:.2f} seconds.")


    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"--- Batch generation complete ---")
    logger.info(f"Total samples attempted: {num_samples}")
    logger.info(f"Successfully generated: {samples_generated}")
    logger.info(f"Failed generations: {samples_failed}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Dataset saved to: {output_file}")

    logging.shutdown()


if __name__ == "__main__":
    # Maybe add command-line argument parsing using argparse to override NUM_SAMPLES_TO_GENERATE, OUTPUT_FILENAME, etc.
    # Right now it uses constants defined at the top.
    main()