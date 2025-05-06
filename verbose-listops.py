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


ORDINAL_WORDS_TO_IGNORE = {
    "first", "second", "third",
}


# --- OpenAI API Key and Tokenizer Initialization ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") # Use OpenRouter variable name
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set. Using placeholder.") # Updated warning
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE" # Updated placeholder
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
    "$task_body\n\n"
    "$ultra_strict_instruction\n\n" # This will contain the refined number rules
    "Output only the narrative text for this new scene, continuing from the snippet. Do not include titles, headings, or explanations." # Clarified output expectation
)


# --- Batch Generation & Output ---
NUM_SAMPLES_TO_GENERATE = 1 # How many samples to generate in one run
DEFAULT_MAX_WORKERS = 20  # Default number of parallel threads for batch generation

FEW_SHOT_EXAMPLES_STRICT = [
    (
        # --- Example 1: Basic Success vs. Extraneous Number (>10) ---
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers (use digits): thirty-nine (39), ninety (90), and ninety-three (93).\\n"
            "*   **MUST AVOID (FORBIDDEN):** Do NOT mention ...: five (5).\\n"
            "*   You MAY use the number 3 ('three', the count of direct items...) and the number 1 ('one').\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),

        "Felix examined the three caches. 'This one has 93 relics, that one 90, and the last 39,' he said. Liora checked the Cipher Wheel. 'We need the smallest: 39.'", # Concise good narrative

        "Felix examined the three caches. 'This one has 93 relics, that one 90, and the last 39,' he said. Liora checked the Cipher Wheel. 'We need the smallest: 39. It took 12 minutes.'", # Concise bad narrative

        "BAD output failed: Included 12. Rule Analysis: 12 not in MUST INCLUDE {39, 90, 93}, not MUST AVOID {5}, not operand count (3), not allowed small num (0-10). Violates 'NO OTHER NUMBERS'." # Concise reasoning
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
    """
]
# --- END NEW FEW-SHOT SECTION ---


# --- Base configurations ---

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Create logs directory at startup

# --- Few-shot prompt examples ---
EXAMPLE_TEXTS = [
    (
        "Example 1:\\n"
        'Narrative: "Guild offered two contracts: one 9 silver, other 4. Kaelen chose lower (4). Received 5 silver bonus."\\n' # Concise narrative
        "Implicit Calculation: MIN(9, 4) = 4. SUM(4, 5) = 9.\\n"
        "Answer: 9\\n"
    ),
    (
        "Example 2:\\n"
        "Narrative: Vault needed combined energy of four crystals (1, 1, 1, 1). Lock used final digit of total power.\\n" # Concise narrative
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Mod 10 -> 4.\\n"
        "Answer: 4\\n"
    ),
    (
        "Example 3:\\n"
        "Narrative: Three scouts reported patrol durations: 5, 5, 5 hours. Procedure: calculate average time (rounded down) for logbook.\\n" # Concise narrative
        "Implicit Calculation: SUM(5, 5, 5) = 15. Count=3. AVG = 15/3 = 5. Floor(5) = 5.\\n"
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
    sample_index: int | None = None,
    path: str = os.path.join(LOG_DIR, "llm_turns.log"),
):
    """Append a timestamped prompt header and text to the prompts log."""
    try:

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        timestamp = datetime.datetime.now().isoformat()

        log_header = f"[Sample {sample_index + 1}] {header}" if sample_index is not None else header
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- Log Time: {timestamp} ---\n")
            f.write(f"{log_header}\n{prompt}\n\n---\n\n")
    except Exception as e:
        print(f"Error writing to log file {path}: {e}")  # Fallback to console if logging fails

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
    DEFAULT_MAX_TOTAL_TOKENS: int = 10000
    DEFAULT_MAX_BEAT_TOKENS: int = 1500
    DEFAULT_MAX_PAD_TOKENS: int = 1000
    MAX_TOKENS_BUFFER: int = 1000
    RETRY_MAX_ATTEMPTS: int = 5
    RETRY_INITIAL_DELAY: int = 0.5
    MAX_BEAT_RETRIES: int = 5
    MAX_PAD_RETRIES: int = 5
    ATOM_MIN_VALUE: int = 1 # currently only supports p+ integers
    ATOM_MAX_VALUE: int = 100
    USE_NARRATIVE_ANCHORS: bool = True
    USE_LLM_NAMING: bool = (
        True  # If True, use LLM for narrative anchors; else use thematic fallback
    )
    NUM_FEW_SHOT_EXAMPLES: int = 3 # 0-3


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
    narrative_anchor_map: dict
    all_atoms: set
    introduced_atoms: set
    scenes: list
    tokens_used: int
    last_scene_text: str
    beat_counter: dict
    sample_index: int
    max_pad_paragraphs: int = 2

MODEL = "google/gemini-2.5-pro-preview-03-25"
SAFETY_MARGIN = config.MAX_TOKENS_BUFFER
MAX_BEAT_TOKENS = config.DEFAULT_MAX_BEAT_TOKENS
MAX_PAD_TOKENS = config.DEFAULT_MAX_PAD_TOKENS

# --- Setup Logging ---

if CLEAR_LOGS_ON_START:
    for filename in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, filename)
        try:
            os.remove(file_path)
        except OSError:
            pass

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)

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


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Changed back from logging.ERROR
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Instantiate OpenAI Client for OpenRouter Endpoint --- # Updated comment
client = None # Initialize client variable
try:

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
         raise ValueError("OpenRouter API Key not found or is placeholder.") # Updated error message

    client = OpenAI(
        api_key=OPENROUTER_API_KEY, # Use OpenRouter key

        base_url="https://openrouter.ai/api/v1" # OpenRouter URL
    )
    logger.info("OpenAI client configured to use OpenRouter API endpoint.") # Updated log message
except Exception as e:
    logger.error(f"Failed to configure OpenAI client for OpenRouter endpoint: {e}") # Updated error log
    client = None


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

def clean_snippet(text: str, max_len: int = 150) -> str:
    """Removes common model analysis/checklist lines and takes the last part."""
    if not text:
        return "The story begins..."


    lines = text.splitlines()
    cleaned_lines = [
        line for line in lines
        if not re.match(
            r"^\s*(-|\*|\d+\.|Critique|Checklist|Yes|No|Draft \d+|Option \d+|\[.*?\]:|MUST INCLUDE|MUST AVOID|Problem:|REASONING:|GOOD:|BAD:|Confidence Score:|Mental Sandbox:|Outcome is|Narrative:|Generation:|Rules:|System:|User:|Okay|Check\.|REMINDER:|Instructions:|Task:|^\?|^\s*$)", # Added more patterns
            line.strip(),
            re.IGNORECASE
        )
        # Add a check for lines that seem like prompt echoing
        and not line.strip().startswith(('Imply the sum', 'reference to the previous', 'Narrate comparing', 'This scene resolves')) # Add more prompt fragments if needed
    ]


    cleaned_text = "\n".join(cleaned_lines).strip()
    if not cleaned_text: # Handle case where cleaning removed everything

        original_lines = [line for line in lines if line.strip()]
        if original_lines:
            cleaned_text = original_lines[-1].strip()
        else: # If original was also effectively empty
            return "Previously..." # Or some other placeholder


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
            
            truly_raw_llm_content = None # Initialize
            if resp and resp.choices and len(resp.choices) > 0 and resp.choices[0].message:
                truly_raw_llm_content = resp.choices[0].message.content # This is the truly raw content, could be None

            # Log the TRULY raw content (or a placeholder if it's None)
            # The 'Generation (Raw):' part of the message indicates its nature.
            log_prompt(
                f"LLM Turn Attempt {attempt}",
                f"System: {system_prompt}\\nUser: {user_prompt}\\n\\nGeneration (Raw):\\n{truly_raw_llm_content if truly_raw_llm_content is not None else '[[API returned None content string]]'}",
                sample_index=sample_index
            )

            # Now, prepare the candidate for validation and return, which involves stripping.
            candidate_for_validation_and_return = None
            if truly_raw_llm_content is not None:
                candidate_for_validation_and_return = truly_raw_llm_content.strip() # Strip for validation and further use
            else:
                logger.warning(f"API call in generate_with_retry attempt {attempt} returned None content. Response object: {resp}")
                # candidate_for_validation_and_return remains None
            
            # The rest of the function uses candidate_for_validation_and_return
            if candidate_for_validation_and_return is None:
                logger.warning(f"generate_with_retry attempt {attempt} resulted in None candidate_for_validation_and_return (possibly after stripping None).")
            elif not candidate_for_validation_and_return or candidate_for_validation_and_return.lower().startswith(
                ("i cannot", "i'm sorry", "i am unable")
            ):
                logger.warning(f"API refusal on generate_with_retry attempt {attempt}.")
            elif validate_fn(candidate_for_validation_and_return): # Validate the stripped version
                return candidate_for_validation_and_return # Return stripped version

        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")


        if attempt < retries:
             time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))


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

        if op == "MED":

            possible_arities = [
                n for n in range(MIN_ARITY, max_branch + 1) if n % 2 == 1
            ]

            if not possible_arities:
                arity = MIN_ARITY if MIN_ARITY % 2 == 1 else MIN_ARITY + 1
            else:
                arity = random.choice(possible_arities)
        else:
            arity = random.randint(MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]

        # --- NEW: Ensure AVG direct atom sum is divisible by atom count ---
        if op == "AVG":
            direct_atoms = [c for c in children if isinstance(c, Atom)]
            arity = len(direct_atoms)
            if arity > 0: # Only adjust if there are direct atoms
                current_sum = sum(a.n for a in direct_atoms)
                remainder = current_sum % arity

                if remainder != 0:

                    adjustment_needed = (arity - remainder) % arity
                    logger.debug(f"AST Gen (AVG): current_sum={current_sum}, arity={arity}, remainder={remainder}, adjustment_needed={adjustment_needed}")


                    atom_to_adjust = random.choice(direct_atoms)
                    adjusted = False


                    new_value_add = atom_to_adjust.n + adjustment_needed
                    if config.ATOM_MIN_VALUE <= new_value_add <= config.ATOM_MAX_VALUE:
                        atom_to_adjust.n = new_value_add
                        atom_to_adjust.value = new_value_add # Update value too
                        logger.debug(f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value up to {atom_to_adjust.n} to make sum divisible by {arity}.")
                        adjusted = True


                    if not adjusted:

                        new_value_sub = atom_to_adjust.n - (arity - adjustment_needed)
                        if config.ATOM_MIN_VALUE <= new_value_sub <= config.ATOM_MAX_VALUE:
                            atom_to_adjust.n = new_value_sub
                            atom_to_adjust.value = new_value_sub # Update value too
                            logger.debug(f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value down to {atom_to_adjust.n} to make sum divisible by {arity}.")
                            adjusted = True

                    if not adjusted:

                        logger.warning(f"AST Gen (AVG): Could not adjust atom value {atom_to_adjust.n} (target adjustment {adjustment_needed}) for AVG node sum {current_sum} to be divisible by {arity} due to bounds [{config.ATOM_MIN_VALUE}, {config.ATOM_MAX_VALUE}].")
        # --- END NEW ---

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

    logger.debug(f"DEBUG: _chat_completion_call received kwargs: {kwargs}") # <--- ADD THIS LINE

    if client is None:
        logger.error("OpenAI client (for OpenRouter) not initialized. Cannot make API call.")
        raise RuntimeError("API client not initialized.")



    standard_params = {"model", "messages", "max_tokens", "temperature", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user", "top_k"} # Ensure top_p and top_k are here if used
    standard_kwargs = {k: v for k, v in kwargs.items() if k in standard_params}


    if "max_completion_tokens" in kwargs and "max_tokens" not in standard_kwargs:
        standard_kwargs["max_tokens"] = kwargs["max_completion_tokens"]

    logger.debug(f"DEBUG: Final args for client.chat.completions.create: {standard_kwargs}")
    logger.debug(f"Calling client.chat.completions.create with args: {standard_kwargs}")
    try:

        return client.chat.completions.create(**standard_kwargs)
    except Exception as e:
        logger.error(f"Error during client.chat.completions.create: {e}")
        logger.error(f"Args that failed: {standard_kwargs}")
        raise # Re-raise the exception

    # --- END TEMPORARY ---



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
def generate_world(num_characters: int = 5, num_concepts: int = 7, max_retries: int = 3, sample_index: int | None = None) -> dict:
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
        "    *   `name`: string (e.g., \"Kaelen Vane\", \"Seraphina Moonwhisper\")\n"
        "    *   `role`: string (e.g., \"The grizzled warrior,\" \"The cunning sorceress,\" \"The naive apprentice\")\n"
        "    *   `quirk`: string (a unique or unusual habit, belief, or physical trait, e.g., \"Collects antique spoons,\" \"Only speaks in riddles,\" \"Has mismatched eyes\")\n"
        "    Ensure each character's name, role, and quirk combination is unique.\n\n"
        "2.  **Genre:** Define a `genre` as a string (e.g., \"Steampunk Adventure,\" \"Urban Fantasy Mystery,\" \"Cosmic Horror Saga\").\n\n"
        "3.  **Setting:** Define a `setting` as a string (a brief, evocative description of the world or primary location, e.g., \"A floating city powered by forgotten magic and steam contraptions,\" \"A post-apocalyptic wasteland where ancient ruins hold dangerous secrets\").\n\n"
        "4.  **Object:** Define an `object` as a string. This should be a plural noun representing key items characters might seek, collect, or use (e.g., \"etherium crystals,\" \"lost star-charts,\" \"prophetic dream-shards\").\n\n"
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
        "If any string value itself needs to contain double quotes (e.g., a nickname within a name, a quote in a setting description), these internal double quotes **MUST** be escaped with a backslash (`\\\\`).\n" # Python `\\\\` -> prompt `\\`
        "   - **CORRECT Example:** `\\\"name\\\": \\\"Bartholomew \\\\\\\"Barty\\\\\\\" Bumble\\\"`\n" # Python `\\\\\\\"` -> prompt `\\\"`
        "   - **INCORRECT Example:** `\\\"name\\\": \\\"Bartholomew \\\"Barty\\\" Bumble\\\"` (This will cause a parsing error!)\n"
        "Adhere strictly to all JSON syntax rules, including commas between elements and correct brace/bracket usage.\n\n"
        "**Final Reminder:**\n"
        "Your response must be *only* the JSON object. Do NOT include ```json markdown fences, explanations, introductory phrases, or *any* other text before or after the JSON object.\n\n"
        "Generate the JSON data now."
    )

    for attempt in range(max_retries):
        logger.debug(f"Attempting world generation (Attempt {attempt + 1}/{max_retries}) with tuned prompt.")
        text = None
        try:
            # --- Log the prompt ---
            log_prompt(
                header=f"World Generation Prompt (Attempt {attempt + 1})",
                prompt=f"System: (Implicit in API call structure for this function)\nUser:\n{prompt}", # Assuming prompt var holds the user part
                sample_index=sample_index
            )
            # --- End log prompt ---

            resp = _chat_completion_call(
                model=MODEL, # Uses the MODEL constant defined in your script
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2500, # Changed from 2000 to 2500
                temperature=0.5, # Changed from 0.7 to 0.5
            )
            if hasattr(resp, "choices") and resp.choices: # Check if choices list exists and is not empty

                first_choice = resp.choices[0]

                if hasattr(first_choice, "message") and first_choice.message:

                    text = first_choice.message.content
                    if text is None:

                        logger.warning(f"World Gen Attempt {attempt + 1}: API returned None content within message. Response: {resp}")
                        text = "" # Treat as empty string
                else:

                    logger.error(f"World Gen Attempt {attempt + 1}: First choice object lacks 'message' attribute or message is empty. Response: {resp}")
                    text = "" # Treat as error / empty string
            else:

                logger.error(f"World Gen Attempt {attempt + 1}: API response lacks 'choices' list or it's empty. Response: {resp}")
                text = "" # Treat as error / empty string
            
            # --- Log the raw response ---
            log_prompt(
                header=f"World Generation Response (Attempt {attempt + 1})",
                prompt=f"Raw LLM Output:\n{text}",
                sample_index=sample_index
            )
            # --- End log response ---

            if not text.strip():
                logger.warning(f"World Gen Attempt {attempt + 1}: Received empty response from API.")

                if attempt < max_retries - 1:
                    time.sleep(1) # Uses the time module
                continue # Go to the next attempt


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

        except Exception as e:

            logger.error(f"World Gen Attempt {attempt + 1}: Unexpected error: {e}. Raw text:\n---\n{text if text else 'N/A'}\n---")



        if attempt < max_retries - 1:





            delay = 1.0 * (2 ** attempt) # Simple exponential backoff starting at 1 second

            logger.info(f"Retrying world generation in {delay:.2f} seconds...")
            time.sleep(delay) # Uses the time module


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


        if number_word in ORDINAL_WORDS_TO_IGNORE:
            # --- DEBUG START ---
            logger.debug(f"    Ignoring ordinal word: '{number_word}'")
            # --- DEBUG END ---
            continue


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
    operand_count: int,
    correct_result_for_beat: int | None = None, # <<< ADD DEFAULT VALUE
    intermediate_sum_allowed: int | None = None,
    strict_zero: bool = False
) -> Callable[[str], bool]:
    """
    Return a validator function based on new rules.
    Allows the correct_result_for_beat (if not None), intermediate_sum (if provided),
    operand_count (if not forbidden), and small numbers (0-10, if not forbidden).
    If strict_zero is True, ALL numbers are forbidden.
    """
    logger.debug(f"Creating validator with: Allowed_Atoms={allowed_atoms}, Forbidden={forbidden_atoms}, OpCount={operand_count}, Result={correct_result_for_beat}, InterSum={intermediate_sum_allowed}, StrictZero={strict_zero}")


    IMPLICITLY_ALLOWED_SMALL_NUMBERS = set(range(0, 11)) # Allow 0 through 10

    # --- Combine all potentially allowed numbers for checking extras --- 
    explicitly_allowed_set = allowed_atoms.copy()
    if correct_result_for_beat is not None: # <<< CHECK IF NOT NONE
        explicitly_allowed_set.add(correct_result_for_beat) # Allow the result
    if intermediate_sum_allowed is not None:
        explicitly_allowed_set.add(intermediate_sum_allowed) # Allow intermediate sum if applicable
    # --- END NEW --- 

    def validate(text: str) -> bool:
        found_numbers = extract_numbers_from_text(text)
        logger.debug(f"Validator Input Text: \"" + text[:100].replace('\n', ' ') + "...\"") # Log cleaned text
        logger.debug(f"Validator Found Numbers: {found_numbers}")

        # --- Strict Zero Check --- 
        if strict_zero:
            if found_numbers:
                log_reason = f"Validation FAIL (Strict Zero): Found numbers {found_numbers} when ZERO were allowed."
                logger.debug(log_reason)
                return False
            else:
                logger.debug(f"Validation PASS (Strict Zero)")
                return True
        # --- End Strict Zero Check --- 


        missing_expected = allowed_atoms - found_numbers
        if missing_expected:
            # --- MODIFIED LOG --- # Detailed Failure Logging
            log_reason = f"Validation FAIL (Rule 1: Missing Required): Required={allowed_atoms}, Missing={missing_expected}, Found={found_numbers}."
            logger.debug(log_reason)
            return False


        found_forbidden = found_numbers & forbidden_atoms
        if found_forbidden:
            truly_forbidden_found = found_forbidden - allowed_atoms
            if truly_forbidden_found:
                # --- MODIFIED LOG --- # Detailed Failure Logging
                log_reason = f"Validation FAIL (Rule 2: Found Forbidden): Forbidden={forbidden_atoms}, Found_Forbidden={truly_forbidden_found}, Allowed={allowed_atoms}, Found_All={found_numbers}."
                logger.debug(log_reason)
                return False
            else:
                logger.debug(f"Validator INFO: Found number(s) {found_forbidden} that were technically forbidden but also required. Allowing.")


        unexpected_found = found_numbers - allowed_atoms

        # --- NEW: Whitelist the intermediate sum if provided ---
        if intermediate_sum_allowed is not None and intermediate_sum_allowed in unexpected_found:
            logger.debug(f"Validator INFO: Allowing explicitly permitted intermediate sum: {intermediate_sum_allowed}")
            unexpected_found.remove(intermediate_sum_allowed)
        # --- END NEW ---


        truly_disallowed_extras = set()
        for extra_num in unexpected_found:
            is_allowed_one = (extra_num == 1)

            is_allowed_count = (extra_num == operand_count and extra_num not in forbidden_atoms) 
            is_allowed_small = (extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS and extra_num not in forbidden_atoms)

            if not (is_allowed_one or is_allowed_count or is_allowed_small):

                fail_reason_detail = []
                if extra_num in forbidden_atoms:
                    fail_reason_detail.append("explicitly forbidden")

                if extra_num > 10 and extra_num != operand_count:
                     fail_reason_detail.append("neither small (<=10) nor operand_count")
                elif extra_num <=10 and extra_num not in IMPLICITLY_ALLOWED_SMALL_NUMBERS: # Should not happen if set is correct
                     fail_reason_detail.append("small but not in allowed small set") 
                elif extra_num <=10 and extra_num in forbidden_atoms:
                     fail_reason_detail.append("small but forbidden")
                elif extra_num == operand_count and extra_num in forbidden_atoms:
                     fail_reason_detail.append("operand_count but forbidden")
                elif extra_num == 1 and extra_num in forbidden_atoms: # Check if 'one' was forbidden
                     fail_reason_detail.append("'one' but forbidden")
                

                truly_disallowed_extras.add((extra_num, ", ".join(fail_reason_detail) if fail_reason_detail else "unexpected extraneous"))





        if truly_disallowed_extras:
            # --- MODIFIED LOG --- # Detailed Failure Logging

            formatted_disallowed = ', '.join([f'{n}({reason})' for n, reason in truly_disallowed_extras])
            log_reason = f"Validation FAIL (Strict Rule: Extraneous Numbers): Disallowed_Extras={{ {formatted_disallowed} }}. AllowedSet={allowed_atoms}, ForbiddenSet={forbidden_atoms}, OperandCount={operand_count}, Found={found_numbers}, UnexpectedRaw={unexpected_found}"
            logger.debug(log_reason)
            return False


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

        return {node.n}
    atoms = set()
    for child in node.children:
        atoms.update(get_atoms_in_subtree(child))
    return atoms


def generate_narrative_anchor_with_llm(
    world_info: dict,
    op_node: OpNode,
    all_previous_anchors: list[str],   # MODIFIED: Was child_narrative_anchors, now all_previous_anchors
    sample_index: int | None = None,
) -> str | None:
    """
    Uses an LLM to generate a short, thematic noun phrase based on keywords.
    Focuses on reliability with a very simple prompt structure.
    """

    op_label = OP_LABELS.get(op_node.op, op_node.op) # Still useful for fallback or logging
    genre = world_info.get("genre", "unknown genre")
    setting = world_info.get("setting", "a mysterious place") # ADDED: Get setting
    primary_object = world_info.get("object", "items")

    # --- NEW: Simplified Concept Keywords ---
    concept_keywords_map = {
        "MAX": "Pinpointing the most potent or largest element",
        "MIN": "Isolating the smallest or most fundamental essence",
        "SUM": "Amalgamating all components into a unified total",
        "MED": "Identifying the central balancing point in an ordered series",
        "AVG": "Discerning the common thread or typical measure across all items",
        "SM":  "Unveiling a core symbolic digit through cyclical transformation", # Or "Coded Essence"
    }
    concept_keywords_for_prompt = concept_keywords_map.get(op_node.op, f"{op_label} Concept") # Fallback

    # MODIFIED System Prompt's final instruction line
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

# You can then use this system_prompt variable in your LLM calls.
# For example:
# print(system_prompt)

    # MODIFIED User Prompt to include Setting and Previously Used Anchors
    user_prompt = (
        f"Genre: {genre}\\n"
        f"Setting: {setting}\\n"
        f"Item: {primary_object}\\n"
        f"Concept/Operation Hint: {concept_keywords_for_prompt}\\n"
    )
    # --- END NEW Ultra-Simple Prompt ---

    prompt_log_header = f"--- Narrative Anchor Prompt (Op: {op_node.op}, Item: {primary_object}, Concept: {concept_keywords_for_prompt}) ---" # Updated log header
    prompt_content_for_log = f"System: {system_prompt}\\nUser:\\n{user_prompt}" # Adjusted user prompt variable name for clarity
    
    # --- Log the prompt using log_prompt ---
    log_prompt(
        header=f"Narrative Anchor Generation Prompt (Op: {op_node.op})",
        prompt=prompt_content_for_log,
        sample_index=sample_index
    )
    # --- End log prompt ---
    
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
            "temperature": 0.55,
            # If you decide to add top_p later, include it here too:
            # "top_p": 0.95,
        }
        logger.debug(f"--- LLM Anchor Gen: EXACT REQUEST PAYLOAD ---")
        logger.debug(json.dumps(request_payload, indent=2)) # Make sure you have `import json` at the top of your file
        logger.debug(f"--- END EXACT REQUEST PAYLOAD ---")

        # Modify the existing call to use the payload dictionary:
        resp = _chat_completion_call(**request_payload)

        raw_candidate = None
        finish_reason = "N/A"

        if resp and resp.choices and len(resp.choices) > 0:
            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            if choice.message:
                raw_candidate = choice.message.content
                if raw_candidate is None:
                    logger.warning(f"Narrative Anchor Gen: Received None content. Finish reason: {finish_reason}. Response: {resp}")
            else:
                logger.warning(f"Narrative Anchor Gen: Message object is missing or empty. Finish reason: {finish_reason}. Response: {resp}")
        else:
            logger.warning(f"Narrative Anchor Gen: Unexpected response structure (no choices or empty response). Response: {resp}")

        log_prompt(
            header=f"Narrative Anchor Generation Response (Op: {op_node.op})",
            prompt=f"Raw LLM Output (Finish Reason: {finish_reason}):\n{raw_candidate if raw_candidate is not None else 'None'}",
            sample_index=sample_index
        )
        logger.debug(f"Narrative Anchor Gen - API Call Details: Finish Reason='{finish_reason}', Raw Candidate='{str(raw_candidate)[:100]}...'" )

        if raw_candidate is None:
            logger.warning(f"Narrative Anchor Gen: Received None content in response.")
            return None

        candidate = raw_candidate.strip()

        # --- Strip surrounding quotes --- reinstated
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1].strip()
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1].strip()
        # --- End strip surrounding quotes ---

        # --- Remove boilerplate prefixes --- reinstated
        original_candidate_before_boilerplate_strip = candidate # Store for comparison
        candidate = re.sub(r"^(OUTPUT \(Phrase Only\):)\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"^(Okay, here's a noun phrase:|Noun Phrase:|Phrase:|Label:|Descriptor:|Designation:|Certainly:|Here it is:)\s*", "", candidate, flags=re.IGNORECASE).strip()
        # --- End boilerplate removal ---

        # --- Check if boilerplate was present or if string is now empty --- reinstated
        boilerplate_indicators_lower = [
            "output (phrase only):", "okay, here's a noun phrase:", "noun phrase:", 
            "phrase:", "label:", "descriptor:", "designation:", "certainly:", "here it is:"
        ]
        # Check if the candidate, after stripping, still starts with a known boilerplate indicator
        # or if it became empty after stripping boilerplate (meaning it was *only* boilerplate)
        if not candidate or any(original_candidate_before_boilerplate_strip.lower().startswith(indicator) for indicator in boilerplate_indicators_lower):
            if not candidate: # It became empty after stripping
                 logger.warning(
                    f"Narrative Anchor Gen: Response was only boilerplate (raw: '{raw_candidate}', processed to empty string)"
                )
            else: # It still starts with boilerplate, or original started with it and cleaning wasn't perfect
                 logger.warning(
                    f"Narrative Anchor Gen: Boilerplate detected in response (raw: '{raw_candidate}', processed: '{candidate}'). Triggering retry."
                )
            return None # Fail this attempt to trigger retry
        # --- End boilerplate check ---

        # --- NEW: Aggressively remove echoed input preamble ---
        # Construct the expected preamble pattern based on current genre, item, concept
        # MODIFIED: Preamble removal pattern
        preamble_pattern_str = (
            rf"Genre: {re.escape(genre)}\s*\\n"
            rf"Setting: {re.escape(setting)}\s*\\n"
            rf"Item: {re.escape(primary_object)}\s*\\n"
            rf"Concept/Operation Hint: {re.escape(concept_keywords_for_prompt)}\s*\\n"
        )
        # Remove the preamble if found at the beginning of the candidate string
        candidate = re.sub(f"^{preamble_pattern_str}", "", candidate, flags=re.IGNORECASE).strip()
        # --- END NEW ---

        # Remove potential "OUTPUT (Phrase Only):" if the model echoes it (keeping this as a fallback)
        candidate = re.sub(r"^(OUTPUT \(Phrase Only\):)\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"^(Okay, here's a noun phrase:|Noun Phrase:|Phrase:|Label:|Descriptor:|Designation:|Certainly:|Here it is:)\s*", "", candidate, flags=re.IGNORECASE).strip()

        # --- NEW: More robust check for guideline echoing ---
        # Strip surrounding quotes, as models sometimes wrap short answers in them.
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1].strip()
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1].strip()

        guideline_starters_lower = [
            "**thematic:**", "**concise:**", "**no numbers:**", 
            "**no direct math terms:**", "**represent outcome:**", 
            "**avoid repetition:**", "**focus on the noun:**",
            "1.", "2.", "3.", "4.", "5.", "6.", "7.",
            "key guidelines", "examples of good anchors"
        ] # Already lowercase or will be lowercased by startswith check

        candidate_lower_stripped = candidate.lower().strip()
        if any(candidate_lower_stripped.startswith(starter) for starter in guideline_starters_lower):
            logger.warning(
                f"Narrative Anchor Gen: Response starts with a guideline phrase (raw: '{raw_candidate}', cleaned: '{candidate}')"
            )
            return None
        # --- END NEW GUIDELINE CHECK ---

        num_words = len(candidate.split())
        if not candidate or num_words == 0 or num_words > 5: # Relaxed upper to 5, but hoping for 2-4
            logger.warning(
                f"Narrative Anchor Gen: Invalid (empty, too long/short, refused) response (raw: '{raw_candidate}', processed: '{candidate}', words: {num_words})"
            )
            return None

        if candidate.lower().startswith(
            ("i cannot", "i'm sorry", "i am unable", "as an ai", "i do not have", "unable to provide")
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

# --- END PHASE 4b Function ---

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
    narrative_anchor = narrative_anchor_map.get(node_id, f"the_unnamed_{node.op}_entity" if isinstance(node, OpNode) else "atom")
    logger.debug(f"_generate_narrative_recursive (POST-ORDER): processing node {getattr(node, 'op', 'Atom')} with narrative anchor '{narrative_anchor}'")

    if isinstance(node, Atom):
        logger.debug(f"Node is Atom ({node.n}), returning.")
        return

    child_narrative_anchors = []
    for child in node.children:
        _generate_narrative_recursive(
            child,
            context,
            is_root=False,
        )
        if isinstance(child, OpNode) and id(child) in narrative_anchor_map:
            child_narrative_anchors.append(narrative_anchor_map[id(child)])
        elif isinstance(child, OpNode):
            logger.warning(f"OpNode child {child.op} of parent {node.op} has no narrative anchor in map.")
        if context.tokens_used >= config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            logger.warning(f"Token limit reached after processing child of operator {getattr(node, 'op', 'Atom')}. Stopping further generation for this branch.")


            return

    logger.debug(f"Finished processing children for operator {getattr(node, 'op', 'Atom')} ({narrative_anchor}). Now processing node itself.")
    if is_root:
        logger.info(f"ROOT NODE ({node.op}): Starting beat generation. Current tokens: {context.tokens_used}/{config.DEFAULT_MAX_TOTAL_TOKENS}")

    context.beat_counter["current"] += 1
    logger.info(f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for operator {node.op} ({narrative_anchor})")
    op_label = OP_LABELS.get(node.op, node.op)

    direct_atom_children = [c for c in node.children if isinstance(c, Atom)]
    operand_count = len(direct_atom_children)
    direct_atom_values = {a.n for a in direct_atom_children}

    required_atoms_for_beat = set(direct_atom_values)
    logger.debug(f"Required atoms for beat {node.op} ({narrative_anchor}): {required_atoms_for_beat}")

    forbidden_atoms_for_prompt = context.introduced_atoms.copy()
    truly_forbidden_for_prompt = forbidden_atoms_for_prompt - required_atoms_for_beat
    primary_object = world["object"]

    direct_atom_sum = None
    # Calculate direct_atom_sum for both AVG and SM if direct atoms exist
    if node.op in ["AVG", "SM"] and direct_atom_children:
        try:
            direct_atom_sum = sum(a.n for a in direct_atom_children)
            logger.debug(f"{node.op} Beat: Calculated direct atom sum for validation/prompt: {direct_atom_sum}")
        except Exception as e:
            logger.error(f"Error calculating direct atom sum for {node.op} node {narrative_anchor}: {e}")
            direct_atom_sum = None

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

    ultra_strict_instruction = (
        f"**ULTRA-STRICT NUMBER RULES (THIS SCENE ONLY):**\n"
        f"- MUST INCLUDE: {must_include_combined_str}\n"
        f"- MUST AVOID: {must_avoid_str}\n"
        f"{may_use_clause.replace('*   You MAY use', '- MAY USE:').strip()}\n"
        f"- NO OTHER NUMBERS ALLOWED{' (except intermediate sum ' + str(direct_atom_sum) + ' for AVG)' if node.op == 'AVG' and direct_atom_sum is not None else ''}."
    )

    object_list_str_for_preamble = ""
    if direct_atom_values:
        items = [f"{num_to_words(x)} ({x})" for x in sorted(direct_atom_values)]
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
            sum_str = str(direct_atom_sum) if direct_atom_sum is not None else 'calculated sum'
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

    correct_result = node.value

    formatted_child_names_str = "None"
    if has_operator_children:
        formatted_child_names_list = [f"'{name}'" for name in child_narrative_anchors]
        formatted_child_names_str = ', '.join(formatted_child_names_list)

    formatted_direct_values_str = "None"
    if has_direct_atom_children:
        direct_atom_words = [num_to_words(a) for a in sorted(direct_atom_values)]
        formatted_direct_values_list = [f"{w} ({v})" for w, v in zip(direct_atom_words, sorted(direct_atom_values))]
        formatted_direct_values_str = ', '.join(formatted_direct_values_list)

    input_description_parts = []
    if has_operator_children:
        input_description_parts.append(f"the outcome(s) from previous step(s) conceptually known as {formatted_child_names_str}")
    if has_direct_atom_children:
        input_description_parts.append(f"newly discovered quantities ({formatted_direct_values_str})")

    if not input_description_parts:
        inputs_str = "inputs determined entirely by context (e.g., selecting between previous outcomes)"
    else:
        inputs_str = " and ".join(input_description_parts)

    action_description = ""
    if node.op == "SUM":
        action_description = (
            f"Narrate an action (e.g., gathering, merging) involving {inputs_str}. "
            f"The outcome MUST be that the total quantity becomes exactly **{correct_result}** {primary_object}. Imply the sum through action."
        )
    elif node.op == "AVG":
        action_description = (
            f"Narrate an event (e.g., balancing, averaging mechanism) involving {inputs_str}. "
            f"The outcome MUST be exactly **{correct_result}** {primary_object} (the floored average). "
            f"You MAY mention the intermediate sum ({direct_atom_sum if direct_atom_sum is not None else 'calculated sum'}) from direct inputs ({formatted_direct_values_str}) if needed for the narrative, but the final result is key."
        )
    elif node.op == "SM":
        sm_intermediate_sum = "unknown"
        try:
             child_values = [eval_node(c) for c in node.children]
             sm_intermediate_sum = sum(child_values)
        except Exception as e:
             logger.warning(f"SM Beat: Could not calculate intermediate sum for prompt explanation: {e}")

        action_description = (
            f"Narrate an action involving {inputs_str}. The characters combine these inputs (reaching a temporary total conceptually around {sm_intermediate_sum}). "
            f"Then, describe a specific, plausible event that **forces them to keep only a quantity equal to the final digit of that total**. "
            f"Examples: \n"
            f"*   A magical lock clicks open, consuming all but the final unit of energy ({correct_result}).\n"
            f"*   A mystical tax collector appears, taking all but the last {correct_result} {primary_object}.\n"
            f"*   The combined items react, leaving only {correct_result} stable {primary_object}.\n"
            f"The final quantity MUST become exactly **{correct_result}** {primary_object}. Do NOT explicitly state 'sum' or 'modulo'; the *event* causes the result."
        )
    elif node.op == "MAX":
        action_description = (
            f"Narrate comparing {inputs_str}. They MUST choose the item/quantity with the largest value. "
            f"The outcome MUST be exactly **{correct_result}** {primary_object}. Justify the choice."
        )
    elif node.op == "MIN":
        action_description = (
            f"Narrate comparing {inputs_str}. They MUST choose the item/quantity with the smallest value. "
            f"The outcome MUST be exactly **{correct_result}** {primary_object}. Justify the choice."
        )
    elif node.op == "MED":
        action_description = (
            f"Narrate evaluating {inputs_str} numerically. They MUST select the item/quantity with the middle value (when sorted). "
            f"The outcome MUST be exactly **{correct_result}** {primary_object}. Justify the choice."
        )
    else:
        action_description = f"Narrate applying '{op_label}' to {inputs_str}. Outcome must be {correct_result}."

    reminder = ""
    if child_narrative_anchors:
        reminder_names_str = ', '.join(f"'{name}'" for name in child_narrative_anchors)
        must_mention_str = formatted_direct_values_str if has_direct_atom_children else "None for this step"
        reminder = (
            f"\n**REMINDER:** Do NOT mention the actual numeric results associated with previous conceptual steps ({reminder_names_str}) in your text. Refer to them by name or conceptually only. "
            f"However, you MUST explicitly mention the newly discovered quantities for *this* step: {must_mention_str}."
        )

    operational_instruction = (
        f"This scene resolves the step named '{narrative_anchor}'.\n"
        f"{action_description}\n"
        f"{reminder}"
    )

    few_shot_section = ""
    num_shots = config.NUM_FEW_SHOT_EXAMPLES

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
             + example_prompt_text # Renamed to avoid conflict with function name
             + "\n\n---\n\n"
        )
    elif num_shots > 0 and FEW_SHOT_EXAMPLES_STRICT:
         logger.warning(f"Configured for {num_shots} few-shot examples, but FEW_SHOT_EXAMPLES_STRICT list only has {len(FEW_SHOT_EXAMPLES_STRICT)} example(s) after modification. Using the first one.")
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


    # --- REFINED SYSTEM PROMPT ---
    system_prompt = (
        "You are a fiction writer creating sequential story scenes. Your ONLY task is to write the *next* narrative scene text based on the user's instructions and a snippet of the previous scene. "
        "ABSOLUTELY FORBIDDEN: Any text other than the story scene. NO analysis, NO checklists, NO rule explanations, NO calculations, NO meta-commentary, NO greetings. "
        "Study the few shot examples of what to do. Adhere STRICTLY to ALL instructions, especially the number rules for the current scene."
    )

    # --- REFINED USER PROMPT ---
    cleaned_snippet = clean_snippet(context.last_scene_text, max_len=150)

    user_message_content = (
        f"**CONTEXT:**\\n"
        f"Genre: {world.get('genre', 'fantasy')}\\n"
        f"Setting: '{world.get('setting', 'a fictional world')}'\\n"
        f"Previous Scene Snippet: '...{cleaned_snippet}'\\n\\n"
    )

    # --- Few-Shot Example Section (Remains the same logic) ---
    few_shot_section = ""
    num_shots = config.NUM_FEW_SHOT_EXAMPLES
    if num_shots == 1 and FEW_SHOT_EXAMPLES_STRICT:
        rules_str, good_narrative, _, _ = FEW_SHOT_EXAMPLES_STRICT[0]
        example_prompt_text = (
            f"Rules:\\n{rules_str}\\n"
            f"Good Narrative Output (Follows Rules):\\n{good_narrative}\\n"
        )
        few_shot_section = (
             f"**EXAMPLE (How to Follow Rules):**\\n" # Clearer heading
             f"{example_prompt_text}"
             f"---\\n\\n"
        )
    # Add elif/else for more examples if FEW_SHOT_EXAMPLES_STRICT is expanded later
    elif num_shots > 0 and FEW_SHOT_EXAMPLES_STRICT:
         logger.warning(f"Configured for {num_shots} few-shot examples, but FEW_SHOT_EXAMPLES_STRICT list only has {len(FEW_SHOT_EXAMPLES_STRICT)} example(s). Using the first one.")
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
    # --- End Few-Shot Section ---

    if few_shot_section:
        user_message_content += few_shot_section

    # --- Combined Task & Scene Requirements ---
    task_header = "Reaching the conclusion" if is_root else "Continuing on the journey"
    current_scene_instructions = f"**YOUR TASK: Write ONLY the narrative text for the next scene.**\\n"
    current_scene_instructions += f"Continue directly from the snippet: '...{cleaned_snippet}'\\n\\n"

    current_scene_instructions += f"**SCENE REQUIREMENTS ({task_header} for '{narrative_anchor}'):**\\n"
    # Combine discovery preamble and operational instruction logic here
    if scene_preamble: # scene_preamble describes the discovery
        current_scene_instructions += f"*   Discovery: {scene_preamble}\\n" # Use bullet point
    # Add the operational instruction (action_description + reminder)
    current_scene_instructions += f"*   Action: {action_description}\\n" # Use bullet point
    if reminder:
        current_scene_instructions += f"*   Reminder: {reminder.replace('**REMINDER:**','').strip()}\\n" # Integrate reminder

    user_message_content += f"{current_scene_instructions}\\n" # Add the combined requirements

    # --- Strengthened Number Rules Section ---
    user_message_content += (
        f"**!!! CRITICAL NUMBER RULES !!!** Your writing MUST\n"
        f"- Explicitly state these numbers for the calculation the characters are about to perform: {must_include_combined_str}\n"
    ) 

    # New explicit construction for "You may optionally use..."
    optional_use_parts_list = []
    if operand_count > 0 and operand_count not in truly_forbidden_for_prompt:
        operand_count_word = num_to_words(operand_count)
        # primary_object is defined earlier in this function scope
        optional_use_parts_list.append(f"{operand_count_word} ({operand_count}), as this is the number of groups of {primary_object}s the characters will find")
    if 1 not in truly_forbidden_for_prompt:
        optional_use_parts_list.append(f"'one' (1) to let you use more natural phrasing in your narrative (e.g. \"He picked up one more...\")")

    # --- CORRECTED LOGIC for mentioning intermediate sum in "optional use" ---
    # This `direct_atom_sum` is the sum of direct atomic children of the current node.
    intermediate_sum_mention_in_optional_rules = ""
    if direct_atom_sum is not None: # If there are direct atoms and their sum is calculated
        if node.op == 'AVG':
            intermediate_sum_mention_in_optional_rules = f"the sum of new items ({str(direct_atom_sum)}) which contributes to the average calculation"
        elif node.op == 'SM':
            # For SM, direct_atom_sum is the sum of new atomic items for this step.
            # The overall sum for SM (sm_intermediate_sum) is mentioned in the action_description.
            # Allowing direct_atom_sum here is consistent with the validator.
            intermediate_sum_mention_in_optional_rules = f"the sum of new items ({str(direct_atom_sum)}) before they are combined with other values for the final digit operation"
        # Add other conditions here if other ops have a specific intermediate sum from direct atoms that can be mentioned.

    # Build the "You may optionally use" line
    full_optional_statement_parts = []
    if optional_use_parts_list:
        # Join the existing optional parts (like operand count and 'one')
        full_optional_statement_parts.append(" and ".join(optional_use_parts_list))
    
    if intermediate_sum_mention_in_optional_rules: # If there's a relevant intermediate sum to mention
        full_optional_statement_parts.append(intermediate_sum_mention_in_optional_rules)

    if full_optional_statement_parts:
        user_message_content += f"- You may optionally use: { ' and '.join(full_optional_statement_parts) }.\n"
    # --- END CORRECTION ---
    
    # Now, resume the original f-string concatenation for the remaining rules
    user_message_content += (        f"- NO OTHER NUMBERS ALLOWED.\n"
        f"**Adherence to these number rules is MANDATORY.**\n\n" # Added mandatory note
    )

    # --- Final Reminder ---
    user_message_content += f"REMEMBER: OUTPUT ONLY THE NARRATIVE TEXT FOR THIS SCENE. NO EXTRA TEXT."
    # --- END REFINED USER PROMPT ---

    log_prompt(
        f"{'=== FINAL' if is_root else '=== Intermediate'} Operator Beat Prompt (Op: {node.op}, Narrative Anchor: {narrative_anchor})",
        f"System: {system_prompt}\n---\nUser:\n{user_message_content}",
        sample_index=context.sample_index
    )

    estimated_prompt_tokens = 0
    try:
        estimated_prompt_tokens = len(encoder.encode(system_prompt + user_message_content))
    except Exception as e:
        logger.error(f"Error encoding prompt for token estimation: {e}")




    current_max_beat_tokens = config.DEFAULT_MAX_BEAT_TOKENS # Use the config value

    logger.info(f"Beat Gen Pre-Call Tokens: Prompt Est={estimated_prompt_tokens}, Max Comp={current_max_beat_tokens}, Current Total={context.tokens_used}, Budget={config.DEFAULT_MAX_TOTAL_TOKENS}")

    if would_exceed_budget(
        context.tokens_used,
        estimated_prompt_tokens + current_max_beat_tokens,
        config.DEFAULT_MAX_TOTAL_TOKENS,
        SAFETY_MARGIN,
    ):
        logger.warning(f"Approaching token limit BEFORE generating operator {node.op} ({narrative_anchor}). Est Prompt {estimated_prompt_tokens} + Max Comp {current_max_beat_tokens} vs Remaining {config.DEFAULT_MAX_TOTAL_TOKENS - context.tokens_used}. Stopping.")
        raise BeatGenerationError(f"Token budget exceeded before generating beat for {node.op}")

    validate_beat_numbers = make_number_validator(
        allowed_atoms=required_atoms_for_beat,
        forbidden_atoms=truly_forbidden_for_prompt,
        operand_count=operand_count,
        correct_result_for_beat=correct_result,
        intermediate_sum_allowed=(direct_atom_sum if node.op in ["AVG", "SM"] else None)
    )
    forbidden_for_padding = context.introduced_atoms.union(required_atoms_for_beat)
    forbidden_for_padding.add(correct_result)
    logger.debug(f"Creating padding validator. correct_result={correct_result}, forbidden_for_padding={forbidden_for_padding}")
    validate_padding = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=forbidden_for_padding,
        operand_count=0,
        correct_result_for_beat=-999,
        strict_zero=True
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
                max_completion_tokens=current_max_beat_tokens, # Use the config value
                temperature=0.6,
            )

            # Get the raw content from LLM response, without any stripping yet
            truly_raw_llm_content = "" # Initialize to empty string
            if resp and resp.choices and len(resp.choices) > 0 and resp.choices[0].message:
                # Assign content if it exists, otherwise it remains an empty string
                # This stores the actual string from the API, or None if that's what was in message.content
                _content = resp.choices[0].message.content
                truly_raw_llm_content = _content if _content is not None else ""
            else:
                 logger.warning(f"Beat {context.beat_counter['current']} attempt {attempt}: Invalid response structure or no message content from API: {resp}")
            # truly_raw_llm_content is now the truly raw string from the API, or an empty string if content was None or structure invalid

            # Log the TRULY raw output to llm_turns.log
            log_prompt(
                f"LLM Beat Generation Attempt {attempt} for operator {node.op} ({narrative_anchor})",
                f"System: {system_prompt}\\nUser: {user_message_content}\\n\\nGeneration (Raw):\\n{truly_raw_llm_content}", # Logging the untouched content
                sample_index=context.sample_index
            )

            # --- NEW: Aggressive Cleaning Step ---
            # The cleaning process will take this raw content and perform its own stripping and filtering.
            # Start the cleaning process with the truly_raw_llm_content.
            # The variable 'cleaned_candidate_text' will be the result of this cleaning.
            cleaned_candidate_text = truly_raw_llm_content # Pass the raw content to be cleaned

            # ... [Your existing aggressive cleaning logic that operates on cleaned_candidate_text] ...
            # Example:
            lines = cleaned_candidate_text.splitlines() # This was 'raw_candidate_text.splitlines()'
            filtered_lines = []
            prompt_echoing_fragments = (
                'Imply the sum', 'reference to the previous', 'Narrate comparing', 'This scene resolves',
                'The outcome MUST be', 'Narrate an action', 'Narrate an event', 'The final quantity MUST become'
            )
            analysis_patterns_to_remove = [ # This list was inside the 'if raw_candidate_text:' block
                r"^\s*Critique:.*$", r"^\s*Checklist:.*$", r"^\s*Analysis:.*$", r"^\s*Rules check:.*$",
                r"^\s*MUST INCLUDE:.*$", r"^\s*MUST AVOID:.*$", r"^\s*Reasoning:.*$", r"^\s*Validation:.*$",
                r"^\s*Following the rules:.*$", r"^\s*Based on the instructions:.*$",
                r"^\s*The task is to.*$", r"^\s*The prompt asks.*$", r"^\s*Scene instructions:.*$", r"^\s*Instructions:.*$",
                r"^\s*Number rules:.*$", r"^\s*Rules:.*$",
                r"^\s*Confidence Score:.*$", r"^\s*Mental Sandbox:.*$",
                r"^\s*\d+\.\s*STRICT:.*$", r"^\s*\d+\.\s*MAY USE.*$", r"^\s*Output ONLY.*$",
                r"^\s*REMINDER:.*$", r"^\s*Okay.*$", r"^\s*Certainly.*$",
                r"^\s*```json.*$", r"^\s*```.*$",
                r"^\s*Generation:\s*", r"^\s*Narrative:\s*",
                r"^\s*Scene \d+:\s*", r"^\s*Beat \d+:\s*",
                r"^\s*Refinement \d+.*$",
                r"^\s*Yes\s*$", r"^\s*N/A\s*$",
                r"^\s*\[.*?\]\s*$",
                r"^\s*-\s.*$", r"^\s*\*\s.*$",
                r"^\s*Outcome is.*$",
                r"^\s*System:.*$",
                r"^\s*User:.*$",
                r"^\s*Check\..*$",
                r"^\s*Task:.*$",
                r"^\s*\?.*$",
            ]
            for line in lines:
                stripped_line = line.strip() # Stripping happens inside cleaning
                is_analysis_or_echo = False
                # Check regex patterns
                for pattern in analysis_patterns_to_remove:
                    if re.match(pattern, stripped_line, re.IGNORECASE):
                        is_analysis_or_echo = True
                        logger.debug(f"Cleaning (beat): Removing line matching pattern '{pattern}': '{line}'")
                        break
                if is_analysis_or_echo:
                    continue

                # Check for prompt echoing fragments
                # Using lower() for startswith check to be case-insensitive for these fragments
                if any(stripped_line.lower().startswith(fragment.lower()) for fragment in prompt_echoing_fragments):
                    logger.debug(f"Cleaning (beat): Removing line starting with prompt fragment: '{line}'")
                    is_analysis_or_echo = True
                
                if not is_analysis_or_echo:
                    filtered_lines.append(line) # Append original line
            cleaned_candidate_text = "\\n".join(filtered_lines).strip() # Final strip is part of cleaning
            # --- END NEW CLEANING ---

            log_prompt( # Log the cleaned version too
                f"LLM Beat Generation Attempt {attempt} for operator {node.op} ({narrative_anchor}) - Cleaned",
                f"Cleaned Generation:\\n{cleaned_candidate_text}",
                sample_index=context.sample_index
            )

            if not cleaned_candidate_text or cleaned_candidate_text.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                reason = "empty or API refusal (after cleaning)"
            # --- Use cleaned_candidate_text for validation ---
            elif not validate_beat_numbers(cleaned_candidate_text):
                reason = "number validation failed (see validator logs for cleaned text)"
            else:
                beat_text = cleaned_candidate_text # Assign the cleaned text if valid
                break

        except Exception as e:
            reason = f"exception: {e}"
            logger.error(f"Exception during beat generation attempt {attempt}: {e}", exc_info=True)

        if beat_text is None:
             logger.warning(f"Beat {context.beat_counter['current']}/{context.beat_counter['total']} retry {attempt}/{config.MAX_BEAT_RETRIES} for operator {node.op} ({narrative_anchor}) failed: {reason}")
             if attempt < config.MAX_BEAT_RETRIES:
                 time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    if beat_text:
        btoks = len(encoder.encode(beat_text))
        context.scenes.append(beat_text)
        context.tokens_used += btoks
        context.last_scene_text = beat_text
        context.introduced_atoms.update(required_atoms_for_beat)
        logger.debug(f"Beat {context.beat_counter['current']} successful. Introduced atoms updated: {context.introduced_atoms}")
        logger.info(f"Beat {context.beat_counter['current']} successful. Tokens used this beat: {btoks}. Total tokens: {context.tokens_used}")
    else:
        logger.error(f"Operator {node.op} ({narrative_anchor}) failed after {config.MAX_BEAT_RETRIES} attempts. Aborting narrative generation. {'(ROOT NODE)' if is_root else ''}")
        raise BeatGenerationError(f"Failed to generate narrative beat for operator {node.op} ({narrative_anchor})")

    pad_count = 0
    add_padding = not is_root
    while add_padding and context.tokens_used < config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < context.max_pad_paragraphs:
        pad_count += 1
        logger.debug(f"Attempting padding {pad_count}/{context.max_pad_paragraphs} after beat for {node.op} ({narrative_anchor})")

        padding_system_prompt = "You are a concise storyteller adding descriptive filler. FOLLOW THE USER'S RULES EXACTLY."
        cleaned_snippet_padding = clean_snippet(context.last_scene_text, max_len=100)

        padding_user_prompt = (
            f'Previous Scene Snippet: "...{cleaned_snippet_padding.replace("\n", " ")}"\n\n'
            f"Task: Write ONE short paragraph (3-5 sentences) continuing the story. Describe atmosphere, character reactions, or minor transitions. DO NOT mention ANY numbers (digits or words like 'one', 'two', 'first', etc.). Output ONLY the padding paragraph."
        )

        log_prompt(f"Padding Prompt {pad_count} after {node.op} ({narrative_anchor})",
                   f"System: {padding_system_prompt}\nUser: {padding_user_prompt}",
                   sample_index=context.sample_index)

        estimated_pad_prompt_tokens = len(encoder.encode(padding_system_prompt + padding_user_prompt))

        if would_exceed_budget(
            context.tokens_used,
            estimated_pad_prompt_tokens + MAX_PAD_TOKENS, # Global MAX_PAD_TOKENS
            config.DEFAULT_MAX_TOTAL_TOKENS,
            SAFETY_MARGIN,
        ):
            logger.warning(f"Approaching token limit before generating padding {pad_count}. Stopping padding.")
            add_padding = False
            break

        padding_text = generate_with_retry(
            system_prompt=padding_system_prompt,
            user_prompt=padding_user_prompt,
            max_tokens=MAX_PAD_TOKENS, # Global MAX_PAD_TOKENS
            validate_fn=validate_padding,
            retries=config.MAX_PAD_RETRIES,
            sample_index=context.sample_index
        )

        if padding_text:
            ptoks = len(encoder.encode(padding_text))
            if context.tokens_used + ptoks <= config.DEFAULT_MAX_TOTAL_TOKENS - SAFETY_MARGIN:
                context.scenes.append(padding_text)
                context.tokens_used += ptoks
                context.last_scene_text = padding_text
                logger.debug(f"Padding {pad_count} successful.")
            else:
                logger.warning(f"Generated padding {pad_count} would exceed token limit. Discarding.")
                add_padding = False
                break
        else:
            logger.warning(f"Padding generation {pad_count} failed after retries. Stopping padding.")
            add_padding = False
            break

    # --- No explicit return needed as context is modified in place ---



# --- generate_narrative function remains largely the same ---


def generate_narrative(
    ast: Node, world: dict, config: Config, encoder, p_inflect, logger, sample_index: int # <-- ADDED sample_index
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

    # --- Pre-calculate node values ---


    logger.debug("Pre-calculating all AST node values...")
    try:
        eval_node(ast) # Evaluate the whole tree to populate .value attributes
        logger.debug("AST node values pre-calculation complete.")
    except Exception as e:
        logger.error(f"Error during AST pre-evaluation: {e}")
        raise RuntimeError("Failed to pre-evaluate AST values.") from e


    all_atoms = get_atoms_in_subtree(ast)

    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    narrative_anchor_map = {}
    if config.USE_NARRATIVE_ANCHORS:
        if operator_nodes:
            use_llm = config.USE_LLM_NAMING
            characters = world.get("characters", [])

            for op_node in operator_nodes: # Already in post-order thanks to postorder() generator
                if not isinstance(op_node, OpNode):
                    continue
                node_id = id(op_node)
                narrative_anchor = None
                if use_llm:
                    # child_narrative_anchors = [] # This was for a different prompt design
                    # for child in op_node.children:

                    #     if isinstance(child, OpNode) and id(child) in narrative_anchor_map:
                    #         child_narrative_anchors.append(narrative_anchor_map[id(child)])
                    #     elif isinstance(child, OpNode):
                    #         logger.warning(f"LLM Naming: Child OpNode {child.op} of parent {op_node.op} has no narrative anchor in map during parent naming.")
                    
                    # Get all anchors generated so far for *this sample*
                    all_anchors_so_far = list(narrative_anchor_map.values()) # These are the anchors generated for previous ops in post-order

                    try:
                        narrative_anchor = generate_narrative_anchor_with_llm(
                            world_info=world, # Pass the whole world_info dict
                            op_node=op_node,
                            all_previous_anchors=all_anchors_so_far, # MODIFIED: Pass all anchors generated so far
                            sample_index=sample_index
                        )
                    except Exception as e:
                        logger.error(f"LLM Naming failed for OpNode {op_node.op}: {e}")
                        narrative_anchor = None # Fallback handled below

                if not narrative_anchor: # Fallback naming
                    primary_object = world["object"]
                    op_index = operator_nodes.index(op_node) # Find index for fallback naming
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
                narrative_anchor_map[node_id] = narrative_anchor
                logger.debug(f"Mapped narrative anchor for node {op_node.op}: '{narrative_anchor}'")


    scenes = []
    tokens_used = 0
    
    # --- SIMPLIFIED AND STRENGTHENED INTRO PROMPT ---
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
    # --- END SIMPLIFIED INTRO PROMPT ---

    log_prompt( # Log the intro prompt
        "Intro Scene Generation Prompt",
        f"System: {intro_system_prompt}\nUser: {intro_user_prompt}",
        sample_index=sample_index
    )

    intro_text = generate_with_retry(
        system_prompt=intro_system_prompt,
        user_prompt=intro_user_prompt,
        max_tokens=250,

        validate_fn=make_number_validator( # <<< UPDATE THIS CALL
            allowed_atoms=set(), 
            forbidden_atoms=set(), 
            operand_count=0, 
            correct_result_for_beat=-999, # <<< ADDED PLACEHOLDER
            strict_zero=True 
        ),
        retries=3, 
        sample_index=sample_index
    )
    # --- END SIMPLIFIED INTRO CALL ---

    # --- ADD EXPLICIT LOGGING FOR THE RESULT OF INTRO GENERATION ---
    log_prompt(
        header="Intro Scene LLM Result",
        prompt=f"Final Generated Intro Text (after retries and validation):\n{intro_text if intro_text else 'None (generation failed, was invalid, or API returned no content)'}",
        sample_index=sample_index
    )
    # --- END ADDED LOGGING ---

    if intro_text and len(encoder.encode(intro_text)) <= config.DEFAULT_MAX_TOTAL_TOKENS:
        # --- REMOVE OLD VALIDATOR/RETRY LINES --- 



        # --- END REMOVAL --- 
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
        narrative_anchor_map=narrative_anchor_map,
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


    final_token_count = len(encoder.encode(final_prompt))
    if final_token_count > config.DEFAULT_MAX_TOTAL_TOKENS:
        logger.warning(f"Final generated prompt ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({config.DEFAULT_MAX_TOTAL_TOKENS}). Truncation might occur.")


    logger.info(f"Successfully generated narrative prompt. Final estimated tokens: {context.tokens_used} (body), {final_token_count} (full prompt)")
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
            num_characters=random.randint(3, 6), num_concepts=random.randint(5, 10), sample_index=sample_index
        )
        logger.info(f"[Sample {sample_index + 1}] World metadata generated.")
        logger.debug(f"[Sample {sample_index + 1}] World Info: {world_info}")

        logger.info(
            f"[Sample {sample_index + 1}] Starting narrative rendering with post-order strict validation..." # Updated log message
        )
        narrative_prompt = generate_narrative(
            ast, world_info, config, encoder, p_inflect, logger, sample_index # <-- PASS sample_index here
        )
        if narrative_prompt is None:
            logger.error(
                f"[Sample {sample_index + 1}] Narrative generation failed post-order strict validation. Skipping." # Updated log message
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
                "validation_mode": ( # Updated validation mode string
                    "postorder_llm_anchor_v2_strict_validate"
                    if (config.USE_NARRATIVE_ANCHORS and config.USE_LLM_NAMING)
                    else (
                        "postorder_thematic_anchors_v2_strict_validate" # Changed v1 to v2 or similar
                        if config.USE_NARRATIVE_ANCHORS
                        else "postorder_strict_validation_v2_strict_validate" # Changed v1 to v2 or similar
                    )
                ),
                 "num_few_shot_examples": config.NUM_FEW_SHOT_EXAMPLES, # Add few-shot count
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
        f"{config.DEFAULT_MAX_BEAT_TOKENS}btok_"
        f"{config.DEFAULT_MAX_TOTAL_TOKENS}-ttok_"
        f"{DEFAULT_MAX_OPS}-mxops_"
        f"{MIN_ARITY}-arity_"
        f"{DEFAULT_MAX_BRANCH}-mxbrch_"
        f"{sanitized_model_name}_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        f".jsonl"
    )
    logger.info(f"Output filename (dynamic): {output_file}")
    logger.info(
        f"Script started. Generating {num_samples} samples using up to {max_workers} workers."
    )
    logger.info(f"Using {config.NUM_FEW_SHOT_EXAMPLES} few-shot examples for narrative generation.") # Log few-shot count

    samples_generated_successfully = 0
    samples_failed = 0
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir): # Create dir only if it doesn't exist
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
                logger.error(f"[Sample {index + 1}] task generated exception: {exc}", exc_info=True) # Log stack trace for task exceptions
                samples_failed += 1
    logger.info(
        f"Parallel generation complete. Writing {samples_generated_successfully} samples to {output_file}..."
    )
    try:

        write_mode = 'w' # Or 'a' if you prefer appending
        logger.info(f"Opening {output_file} in '{write_mode}' mode.")
        with open(output_file, write_mode, encoding="utf-8") as f:
            for sample_data in results: # Iterate through successfully generated results
                try:
                    f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                except TypeError as e:

                    logger.error(
                        f"Serialization failed for sample {sample_data.get('id', 'Unknown')}: {e}. Skipping write for this sample."
                    )

                    samples_failed += 1
                    samples_generated_successfully -= 1 # Decrement success as it wasn't written
                except Exception as e:
                    logger.error(
                        f"Unexpected error writing sample {sample_data.get('id', 'Unknown')}: {e}. Skipping write."
                    )
                    samples_failed += 1
                    samples_generated_successfully -= 1 # Decrement success
    except IOError as e:
        logger.error(f"Fatal file write error opening/writing {output_file}: {e}")

        samples_failed += samples_generated_successfully # All successful generations failed to write
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
    total_count   = num_samples # Use the function argument
    success_count = samples_generated_successfully # Use the counter for written samples
    success_rate  = (success_count / total_count * 100) if total_count else 0
    logger.info(f"Overall success rate (generated AND written): {success_rate:.2f}% ({success_count}/{total_count})")
    logger.info(f"Total time: {total_time:.2f} seconds")
    if samples_generated_successfully > 0:
        logger.info(f"Dataset output file: {output_file}")
    else:
        logger.warning(f"No samples were successfully generated and written. Output file '{output_file}' may be empty or non-existent.")
    logging.shutdown()



def clean_snippet(text: str, max_len: int = 150) -> str:
    """Removes common model analysis/checklist lines and takes the last part."""
    if not text:
        return "The story begins..."


    lines = text.splitlines()
    cleaned_lines = [
        line for line in lines
        if not re.match(
            r"^\s*(-|\*|\d+\.|Critique|Checklist|Yes|No|Draft \d+|Option \d+|\[.*?\]:|MUST INCLUDE|MUST AVOID|Problem:|REASONING:|GOOD:|BAD:|Confidence Score:|Mental Sandbox:|Outcome is|Narrative:|Generation:|Rules:|System:|User:|Okay|Check\.|REMINDER:|Instructions:|Task:|^\?|^\s*$)", # Added more patterns
            line.strip(),
            re.IGNORECASE
        )
        # Add a check for lines that seem like prompt echoing
        and not line.strip().startswith(('Imply the sum', 'reference to the previous', 'Narrate comparing', 'This scene resolves')) # Add more prompt fragments if needed
    ]


    cleaned_text = "\n".join(cleaned_lines).strip()
    if not cleaned_text: # Handle case where cleaning removed everything

        original_lines = [line for line in lines if line.strip()]
        if original_lines:
            cleaned_text = original_lines[-1].strip()
        else: # If original was also effectively empty
            return "Previously..." # Or some other placeholder


    return cleaned_text[-max_len:]


if __name__ == "__main__":



    main(
        config,
        num_samples=config.NUM_SAMPLES_TO_GENERATE,
        max_workers=config.DEFAULT_MAX_WORKERS
    )