"""
verbose-listops.py

1. Generates a complex ListOps problem
2. Renders a narrative for each operator via the Anthropic API, and 
3. constructs a follow-up question for a judge LLM. 

Supports logging, retry logic, and AST validation.
"""

# ─── Configuration Constants ─────────────────────────────────────────────────────

import os
import json
import random
import datetime
import logging
import logging.handlers
import time
from typing import Callable

import tiktoken
import anthropic
from anthropic import Anthropic

# Token and Output Configuration
LOG_DIR = os.path.expanduser("~/verbose_listops_logs")
DEFAULT_MAX_TOTAL_TOKENS = 10000  # Overall token cap for narrative generation 
DEFAULT_MAX_BEAT_TOKENS = 1000  # Maximum tokens used for each story 'beat' (listops paragraph)
DEFAULT_MAX_PAD_TOKENS = 1000  # Maximum tokens used for each story 'padding' section
MAX_TOKENS_BUFFER = 1000  # Safety buffer to prevent exceeding token limits

# AST Random ListOps problem gen params
DEFAULT_MAX_BRANCH = 3  # Default maximum branching factor for AST nodes
ATOM_MIN_VALUE = 0  # Minimum value for leaf nodes (atoms)
ATOM_MAX_VALUE = 9  # Maximum value for leaf nodes (atoms)
MIN_ARITY = 2  # Minimum number of children for operator nodes

# API retry + logging config
RETRY_MAX_ATTEMPTS = 5  # Maximum number of retry attempts for API calls
RETRY_INITIAL_DELAY = 1  # Initial delay between retries in seconds (doubles with each attempt)
LOG_MAX_BYTES = 5 * 1024 * 1024  # Maximum log file size (5MB)
LOG_BACKUP_COUNT = 3  # Number of backup log files to keep

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
MODEL = "claude-3-7-sonnet-latest"
MAX_TOTAL_TOKENS = DEFAULT_MAX_TOTAL_TOKENS
SAFETY_MARGIN = MAX_TOKENS_BUFFER
MAX_BEAT_TOKENS = DEFAULT_MAX_BEAT_TOKENS
MAX_PAD_TOKENS = DEFAULT_MAX_PAD_TOKENS

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)
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


def retry_api_call(func: Callable):
    """
    Decorator to retry Anthropic API calls on failure with exponential backoff.
    Retries up to RETRY_MAX_ATTEMPTS times, doubling delay each time.
    """
    def wrapper(*args, **kwargs):
        delay = RETRY_INITIAL_DELAY
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed attempt {attempt}: {e}")
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
FEW_SHOT_EXAMPLES = """
Example 1:
Story ends after you applied MAX.
Question: What was the largest value at the top level of the story above?
Answer:

Example 2:
Story ends after you applied SUM.
Question: What was the sum of all values at the top level of the story above?
Answer:

Example 3:
Story ends after you applied SM.
Question: What was the sum modulo 10 at the top level of the story above?
Answer:
"""

client = Anthropic(api_key=API_KEY)
encoder = tiktoken.get_encoding("cl100k_base")


from dataclasses import dataclass, field


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


def build_random_ast(max_ops: int, max_branch: int = DEFAULT_MAX_BRANCH):
    """
    Constructs a random ListOps AST.
    Args:
        max_ops: Maximum number of operator nodes to include.
        max_branch: Maximum branching factor for operator nodes.
    Returns:
        The root Node of the randomly generated AST.
    Raises:
        ValueError: If max_ops is not a positive integer.
    """
    if not isinstance(max_ops, int) or max_ops < 1:
        raise ValueError("max_ops must be a positive int")

    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0

    def helper():
        nonlocal count
        if count >= max_ops:
            return Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE))
        count += 1
        op = random.choice(ops)
        arity = random.randint(MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]
        return OpNode(op, children)

    return helper()


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
        ValueError: If the node contains an unsupported operator.
    """
    if isinstance(node, Atom):
        return node.n
    vals = [eval_node(c) for c in node.children]
    func_map = {
        "MAX": max,
        "MIN": min,
        "MED": lambda v: sorted(v)[len(v) // 2],
        "SUM": sum,
        "SM": lambda v: sum(v) % 10,
        "AVG": lambda v: sum(v) // len(v),
    }
    try:
        func = func_map[node.op]
    except KeyError:
        raise ValueError(f"Unsupported operator: {node.op}")
    node.value = func(vals)
    return node.value


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
    return client.messages.create(**kwargs)


def generate_world(num_characters: int = 5) -> dict:
    """
    Use Anthropic API to generate a fictional world metadata.
    Args:
        num_characters: Number of characters to include in the world.
    Returns:
        A dict containing characters, genre, and setting.
    Raises:
        ValueError: If num_characters is not a positive integer.
        RuntimeError: On JSON parsing errors.
    """
    if not isinstance(num_characters, int) or num_characters < 1:
        raise ValueError("num_characters must be a positive int")

    prompt = (
        "You are a world-builder.\n"
        f"Create {num_characters} vivid characters (name, role, quirk), choose a genre and setting.\n"
        "Output as JSON in exactly this shape. Respond with *only* the JSON object, no extra text or markdown:\n"
        "{\n"
        '  "characters": [ { "name": "...", "role": "...", "quirk": "..." }, … ],\n'
        '  "genre": "...",\n'
        '  "setting": "..." \n'
        "}\n"
    )
    resp = _client_create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    text = resp.content[0].text.strip()
    try:
        world = json.loads(text)
    except json.JSONDecodeError as e:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                world = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON substring from world-builder response: {text}"
                )
                raise RuntimeError(
                    "Failed to parse JSON substring from world-builder response"
                ) from e
        else:
            logger.error(f"Failed to parse JSON from world-builder response: {text}")
            raise RuntimeError("Failed to parse JSON from world-builder response") from e

    with open(os.path.join(LOG_DIR, "world.json"), "w", encoding="utf-8") as f:
        json.dump(world, f, ensure_ascii=False, indent=2)

    return world


def load_world():
    """
    Load the previously generated world metadata from disk.
    Returns:
        A dict parsed from world.json.
    Raises:
        FileNotFoundError: If world.json does not exist.
    """
    path = os.path.join(LOG_DIR, "world.json")
    if not os.path.exists(path):
        raise FileNotFoundError("world.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_narrative(ast: Node, world: dict) -> str:
    """
    Render a narrative for each operator in the AST using the Anthropic API, then ask a follow-up question.
    Args:
        ast: The ListOps AST to narrate.
        world: Metadata dict containing characters, genre, and setting.
    Returns:
        The full narrative plus final question as a single string.
    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(ast, Node):
        raise ValueError("ast must be an instance of Node")
    if not isinstance(world, dict):
        raise ValueError("world must be a dict")
    for key in ("characters", "genre", "setting"):
        if key not in world:
            raise ValueError(f"world missing required key: {key}")

    scenes = []
    tokens_used = 0

    operator_nodes = [n for n in preorder(ast) if not isinstance(n, Atom)]
    max_pad_paragraphs = 2
    total_ops = len(operator_nodes)
    logger.info(f"Starting narrative generation: {total_ops} operator beats to process")

    for idx, node in enumerate(operator_nodes, start=1):
        logger.info(
            f"Processing operator {idx}/{total_ops}: {node.op} with operands {[c.value for c in node.children]}"
        )

        if isinstance(node, Atom):
            continue

        operands = [c.value for c in node.children]
        beat_prompt = (
            f"You are a {world['genre']} storyteller.\n"
            f"Characters: {json.dumps(world['characters'])}\n"
            f"Setting: {world['setting']}\n\n"
            "Here is a logical step in the story:\n"
            f"Operation Type: {node.op} (meaning: find the {OP_LABELS.get(node.op, node.op)})\n"
            f"Input Numbers (Operands): {operands}\n"
            f"Result of this step (DO NOT REVEAL THIS NUMBER DIRECTLY): {node.value}\n\n"
            "Write 1 scene (2–5 short paragraphs) where the characters naturally encounter or use these specific "
            "Input Numbers. Weave the *process* of applying the operation (e.g., finding the median, sum modulo 10, "
            "smallest value) to these numbers into the narrative actions or dialogue. For example, mention the numbers "
            f"like 'readings of {operands[0]} and {operands[1]} flashed', or 'they considered the {len(operands)} data "
            f"points: {', '.join(map(str, operands))}' within the story context. Focus on the story and characters, "
            "embedding the calculation logic subtly. Do NOT state the numeric result directly."
        )
        with open(
            os.path.join(LOG_DIR, "verbose_listops.log"), "a", encoding="utf-8"
        ) as prompts_log:
            prompts_log.write(f"=== Operator Beat Prompt (operator: {node.op}) ===\n")
            prompts_log.write(beat_prompt + "\n\n")
            prompts_log.flush()
        resp = _client_create(
            model=MODEL,
            system="You are a storyteller. Write plain narrative paragraphs without any markdown headings or section titles.",
            messages=[{"role": "user", "content": beat_prompt}],
            max_tokens=MAX_BEAT_TOKENS,
        )
        beat = resp.content[0].text
        last_scene = beat
        btoks = len(encoder.encode(beat))
        scenes.append(beat)
        tokens_used += btoks

        pad_prompt = (
            "You are the same storyteller.\n"
            "Continue the last scene below in 2–3 short paragraphs—introduce side-quests, mysteries, or random asides.\n"
            "Last scene:\n"
            f"{last_scene}\n\n"
            "Do NOT change any established facts or operator logic. This is pure padding."
        )
        pad_count = 0
        while tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
            with open(
                os.path.join(LOG_DIR, "verbose_listops.log"), "a", encoding="utf-8"
            ) as prompts_log:
                prompts_log.write(f"=== Padding Prompt (operator: {node.op}) ===\n")
                prompts_log.write(pad_prompt + "\n\n")
                prompts_log.flush()
            pad_resp = _client_create(
                model=MODEL,
                system="Continue the narrative without adding any headings or titles, just plain paragraphs.",
                messages=[{"role": "user", "content": pad_prompt}],
                max_tokens=MAX_PAD_TOKENS,
            )
            pad = pad_resp.content[0].text
            if (
                pad.strip().lower().startswith("i do not have enough context")
                or pad.strip().lower().startswith("i'm sorry")
            ):
                logger.info(f"Padding refused at operator {idx}, stopping padding.")
                break
            ptoks = len(encoder.encode(pad))
            if tokens_used + ptoks > MAX_TOTAL_TOKENS:
                break
            scenes.append(pad)
            tokens_used += ptoks
            pad_count += 1

        if tokens_used >= MAX_TOTAL_TOKENS:
            break

    # Deterministic final question construction
    top_op = ast.op
    label = OP_LABELS.get(top_op, top_op)

    question = (
        f"Considering the entire narrative above, what single digit represents the final "
        f"result of the primary calculation? The story's main logical thread culminates in an operation "
        f"to find the {label}."
    )
    # Log the constructed question
    with open(os.path.join(LOG_DIR, "verbose_listops.log"), "a", encoding="utf-8") as prompts_log:
        prompts_log.write("=== Constructed Final Question ===\n")
        prompts_log.write(question + "\n\n")
        prompts_log.flush()
    scenes.append(question)

    # Append judge instructions
    judge_instructions = f"""
---
**Instructions for Analysis:**

1.  **Goal:** Your task is to determine the single numerical result of the multi-step calculation embedded within the narrative above.
2.  **Identify Operations:** Read the story carefully to find mentions of calculations or comparisons involving groups of single-digit numbers (0-9). Look for keywords or descriptions related to:
    * Maximum / Largest value (MAX)
    * Minimum / Smallest value (MIN)
    * Median / Middle value (MED)
    * Sum / Total value (SUM)
    * Sum Modulo 10 (SM)
    * Average value (AVG, integer/floored)
3.  **Extract Numbers:** Note the specific single-digit numbers associated with each operation described.
4.  **Determine Structure:** Figure out how these operations are nested or sequenced based on the story's progression. The narrative follows a structure where results of earlier operations feed into later ones.
5.  **Calculate Final Result:** Perform the calculations following the narrative's hierarchy. The overall goal culminates in finding the '{label}' ({top_op}).
6.  **Output:** Provide *only* the final single-digit integer result. Do not include explanations or calculations in your final answer.

**Final Answer:** """
    scenes.append(judge_instructions)

    return "\n\n".join(scenes)


def main():
    """Orchestrate building the AST, generating world, rendering narrative, and outputting final result."""
    try:
        logger.info("Script started")
        logger.info("Building random AST...")
        ast = build_random_ast(max_ops=10, max_branch=5)
        validate_ast(ast)
        logger.info("Evaluating AST...")
        eval_node(ast)
        logger.info("AST evaluation complete.")

        logger.info("Generating world metadata...")
        world = generate_world(num_characters=5)
        logger.info("World metadata generated.")
        logger.info("Starting narrative rendering...")

        narrative = generate_narrative(ast, world)
        logger.info("Narrative rendering complete.")

        logger.info("Narrative output:\n%s", narrative)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()