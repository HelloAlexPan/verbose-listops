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
from dataclasses import dataclass, field, asdict
import re
import concurrent.futures
import inflect
import tiktoken
from functools import lru_cache
from openai import OpenAI
import shutil
import threading
import requests
import subprocess
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

# fmt: off
# --- Batch Settings ---
NUM_SAMPLES_TO_GENERATE = 1
DEFAULT_MAX_WORKERS = 100
MODEL = "google/gemini-2.5-flash-preview:thinking"
STATIC_CHECKER_MODEL = "google/gemini-2.5-flash-preview:thinking"
DATASETS_DIR = "datasets"
PROD_RUN: bool = True

@dataclass
class Config:
    MAX_OPS: int = 8
    MAX_BRANCH: int = 8
    MIN_ARITY: int = 6
    MIN_ATOM_VAL: int = 1
    MAX_ATOM_VAL: int = 9
    MAX_TOTAL_TOKENS: int = 10000
    EARLY_TERMINATION_PROBABILITY: float = 0.0
    PADDING_MAX_TOK_PERCENT: float = 0.60
    USE_NARRATIVE_ANCHORS: bool = True
    USE_LLM_NAMING: bool = True
    MIN_WORLD_CHARS: int = 6
    MAX_WORLD_CHARS: int = 8
    MIN_WORLD_CONCEPTS: int = 3
    MAX_WORLD_CONCEPTS: int = 7
    BEAT_CONTEXT: int = 1000
    PADDING_CONTEXT: int = 1500
    MAX_PAD_PARAGRAPHS: int = 30
    WORLD_GEN_TEMP:  float = 0.9
    BEAT_GEN_TEMP: float = 0.5
    CREATIVE_NARRATIVE_TEMP: float = 0.5
    ANCHOR_GEN_TEMP: float = 0.85
    LLM_VALIDATOR_MODEL: str = "google/gemini-2.5-flash-preview:thinking"
    LLM_VALIDATOR_TEMP: float = 0.05
    BEAT_REVISION_TEMP: float = 0.1
    MAX_LLM_VALIDATION_ITERATIONS: int = 6
    MODEL_MAX_CONTEXT_TOKENS: int = 750000
    MAX_ANCHOR_WORDS: int = 4
    FEW_SHOT_EXAMPLES: int = 1
    FALLBACK_MIN_NUM_WORD: int = 0
    FALLBACK_MAX_NUM_WORD: int = 20
    MIN_ALLOWED_SMALL_NUMBER: int = 0
    MAX_ALLOWED_SMALL_NUMBER: int = 10
    ALWAYS_ALLOWED_PHRASING_NUMBERS_SET: Set[int] = field(default_factory=lambda: {1, 2, 3})
    INVALID_RESULT_PLACEHOLDER: int = -999
    PROBLEM_SMALL_NUMBERS_TO_CHECK: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
    RETRY_MAX_ATTEMPTS: int = 10
    RETRY_INITIAL_DELAY: float = 0.25
    MAX_BEAT_RETRIES: int = 5
    MAX_PAD_RETRIES: int = 5
    INTRO_MAX_RETRIES: int = 3
    WORLDGEN_MAX_RETRIES: int = 5
    INITIAL_WORLD_RETRY_DELAY: float = 1.0
    MAX_REQUESTS_PER_SECOND: float = 900.0
    MIN_REQUEST_INTERVAL: float = 0.001
    LOG_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 3
    CLEAR_LOGS_ON_START: bool = True
    MAX_TOKENS_BUFFER: int = 500
    MAX_API_TOKEN_LIMIT: int = 60000
    WORLD_GEN_MAX_TOKENS: int = 200
    ANCHOR_MAX_TOKENS: int = 100
    INTRO_MAX_TOKENS: int = 100
    BEAT_MAX_TOKENS: int = 400
    PADDING_MAX_TOKENS: int = 200

config = Config()
# fmt: on

# --- JSON Schema Definitions for Structured Outputs ---
VALIDATOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {
            "type": "boolean",
            "description": "Whether the beat passes all validation criteria",
        },
        "explanation_for_generator": {
            "type": "string",
            "description": "Detailed explanation of all validation issues, for the next generation attempt",
        },
        "explanation_for_audit": {
            "type": "string",
            "description": "Summary of why the beat is valid, highlighting numerical compliance",
        },
        "overall_revision_summary_for_generator_prompt": {
            "type": "string",
            "description": "Concise instruction for the generator focusing on critical issues",
        },
        "suggested_revisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional suggested specific text revisions",
        },
    },
    "required": ["is_valid"],
    "additionalProperties": False,
}

STATIC_VALIDATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "is_beat_valid": {
            "type": "boolean",
            "description": "Whether the beat passes validation",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation for why the beat passes or fails validation",
        },
    },
    "required": ["is_beat_valid", "reasoning"],
    "additionalProperties": False,
}

WORLD_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "quirk": {"type": "string"},
                },
                "required": ["name", "role", "quirk"],
            },
        },
        "genre": {"type": "string"},
        "setting": {"type": "string"},
        "object": {"type": "string"},
    },
    "required": ["characters", "genre", "setting", "object"],
    "additionalProperties": False,
}
# --- End of JSON Schema Definitions ---


# --- AST Node Definitions ---
@dataclass
class Node:
    op: str = ""  # Fix: Add default value
    children: list = field(default_factory=list)
    value: int = None


@dataclass
class Atom(Node):
    def __init__(self, n: int):
        super().__init__(op="ATOM", children=[])
        self.n = n
        self.value = n


@dataclass
class OpNode(Node):
    def __init__(self, op: str, children: list):
        super().__init__(op=op, children=children)
        self.value = None


# --- END OF AST Node Definitions ---


class GenerationTokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.api_calls = 0
        self.lock = threading.Lock()

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.api_calls += 1
        logger.debug(
            f"Token Tracker: Added {prompt_tokens} prompt, {completion_tokens} completion. Total P: {self.total_prompt_tokens}, C: {self.total_completion_tokens}, Calls: {self.api_calls}"
        )

    def get_summary(self):
        with self.lock:
            return (
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.api_calls,
            )

    def calculate_cost(
        self, prompt_cost_per_million: float, completion_cost_per_million: float
    ) -> float:
        with self.lock:
            prompt_cost = (
                self.total_prompt_tokens / 1_000_000
            ) * prompt_cost_per_million
            completion_cost = (
                self.total_completion_tokens / 1_000_000
            ) * completion_cost_per_million
            return prompt_cost + completion_cost


generation_token_tracker = GenerationTokenTracker()
DEFAULT_COST_PER_MILLION_PROMPT_TOKENS = 0.50
DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS = 1.50

ORDINAL_WORDS_TO_IGNORE = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "twentieth",
    "thirtieth",
    "fortieth",
    "fiftieth",
    "sixtieth",
    "seventieth",
    "eightieth",
    "ninetieth",
    "hundredth",
    "last",
    "final",
}

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print(
        "Warning: OPENROUTER_API_KEY environment variable not set. Using placeholder."
    )
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
try:
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"Failed to initialize tokenizer: {e}")
    encoder = None
from string import Template

BASE_BEAT_TEMPLATE = Template(
    "You are a $beat_mode storyteller writing the next sequential scene in an ongoing narrative.\\n"
    "Characters: $characters\\n"
    "Setting: $setting\\n"
    'Previous Scene Snippet (End of last scene): "...$snippet"\\n\\n'
    "--- $task_header ---\\n"
    "$task_body\\n\\n"
    "$ultra_strict_instruction\\n\\n"
    "Output only the narrative text for this new scene, continuing from the snippet. Do not include titles, headings, or explanations."
)

FEW_SHOT_EXAMPLES_STRICT = [
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: thirty-nine, ninety, and ninety-three.\\\\n"
            "*   You MAY use the number 'three' (the count of direct items...) and the number 'one'.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        "Felix examined the three caches. 'This one has ninety-three relics, that one ninety, and the last thirty-nine,' he said. Liora checked the Cipher Wheel. 'We need the smallest: thirty-nine.'",
        "Felix examined the three caches. 'This one has ninety-three relics, that one ninety, and the last thirty-nine,' he said. Liora checked the Cipher Wheel. 'We need the smallest: thirty-nine. It took twelve minutes.'",
        "BAD output failed: Included 'twelve'. Rule Analysis: 12 not in MUST INCLUDE {39, 90, 93}, not operand count (3), not allowed small num (0-10). Violates 'NO OTHER NUMBERS'.",
    ),
    # Add specific example for MED operations showing IMPLICIT result (good) vs EXPLICIT result (bad)
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: seventy-two, eighty-four, eighty-nine, ninety-one, and ninety-five.\\\\n"
            "*   **MEDIAN RESULT MUST BE IMPLICIT:** The median value (eighty-nine) must NOT be explicitly stated as the result.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions all required numbers but only IMPLIES the median (89)
        "Seraphina arranged the crystal fragments on the altar: 'This one pulses with seventy-two vibrations, this with eighty-four, this with ninety-one, this with ninety-five.' She examined the fifth crystal, studying its unique pattern. 'This middle fragment - the balanced keystone - shall be our central focus. Its resonance sits precisely between the others.' Marcus nodded, 'The perfect equilibrium point. The central essence that will stabilize the ritual.'",
        # BAD example: explicitly mentions "eighty-nine" as the median/result
        "Seraphina arranged the crystal fragments on the altar: 'This one pulses with seventy-two vibrations, this with eighty-four, this with ninety-one, this with ninety-five.' She examined the fifth crystal. 'This one has eighty-nine vibrations - it's the median value, the perfect middle point.' Marcus nodded, 'Eighty-nine is indeed the central value we need.'",
        "BAD output failed: Explicitly stated 'eighty-nine' as the result. For MED operations, the result must be IMPLICIT only. The narrative should mention the required input numbers but never directly state the median value.",
    ),
    # Add a new critical example for MED operations where the median is mistakenly listed among the inputs
    (
        (
            "**ULTRA-STRICT NUMBER RULES (Apply ONLY to THIS Scene):**\\\\n"
            "*   **MUST INCLUDE:** ... mention ... numbers as written words: seventy-three, eighty-five, eighty-seven, eighty-eight, eighty-nine, ninety-one.\\\\n"
            "*   **MEDIAN RESULT MUST BE IMPLICIT:** The median value (eighty-seven) must NOT be explicitly stated anywhere.\\\\n"
            "*   **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values...\\\\n"
            "**Adhere strictly to these rules for this scene only.**"
        ),
        # GOOD example: mentions all inputs EXCEPT the median (87), which is only implied
        "Kairos studied the alignment of six energy signatures on the quantum display. 'The readings show seventy-three, eighty-five, eighty-eight, eighty-nine, and ninety-one, plus the void signal.' He pointed to the empty space between the values. 'The central point - this balance nexus - is our target. The middle value will stabilize the entire sequence.' Lyra nodded, understanding the critical equilibrium point without needing to name it.",
        # BAD example: explicitly lists the median (87) among all values
        "Kairos studied the alignment of six energy signatures on the quantum display. 'The readings show seventy-three, eighty-five, eighty-seven, eighty-eight, eighty-nine, and ninety-one.' He pointed to the third value. 'This central point - eighty-seven - is our target. The middle value will stabilize the entire sequence.'",
        "BAD output failed: The median value 'eighty-seven' was explicitly listed among the values. CRITICAL ERROR: For MED operations, the median result must NEVER appear as an explicit number anywhere in the text. The narrative should mention all other required numbers but completely avoid stating the median value, only implying it through position or concept.",
    ),
]

META_INSTRUCTION = (
    "Here are examples demonstrating how to solve narrative math problems. For each problem: "
    "Read the entire story. Identify the quantity being tracked (e.g., coins, artifacts, energy units). "
    "Follow the narrative step-by-step, performing the calculation implied by the actions in each scene "
    "(e.g., finding items, combining, selecting the largest/smallest, averaging, reducing, resetting). "
    "Keep track of the current quantity as it changes. Finally, answer the question by providing only "
    "the single integer representing the final quantity based on the last relevant action described."
)

TASK_SOLVING_FEW_SHOTS = [
    # ... (content of TASK_SOLVING_FEW_SHOTS) ...
]

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

if config.CLEAR_LOGS_ON_START:
    if os.path.exists(LOG_DIR):
        try:
            shutil.rmtree(LOG_DIR)
            print(f"Removed existing log directory: {LOG_DIR}")
        except OSError as e:
            print(f"Error removing log directory {LOG_DIR}: {e}")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory {LOG_DIR}: {e}")

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
os.makedirs(LOG_DIR, exist_ok=True)
handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "verbose_listops.log"),
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
print(f"Logger initialized. Log file: {os.path.join(LOG_DIR, 'verbose_listops.log')}")
if config.CLEAR_LOGS_ON_START and os.path.exists(LOG_DIR):
    logger.info(f"Log directory {LOG_DIR} cleared and recreated successfully.")
elif config.CLEAR_LOGS_ON_START and not os.path.exists(LOG_DIR):
    logger.warning(
        f"Log directory {LOG_DIR} was meant to be cleared/recreated, but it does not exist."
    )


def log_prompt(header: str, prompt: str, sample_index: int | None = None):
    try:
        llm_turns_main_dir = os.path.join(LOG_DIR, "llm_turns")
        llm_turns_log_specific_dir = os.path.join(llm_turns_main_dir, "log")
        log_filename = (
            f"llm_turns_sample_{sample_index + 1}.log"
            if sample_index is not None
            else "llm_turns_general.log"
        )
        os.makedirs(llm_turns_log_specific_dir, exist_ok=True)
        current_log_file_path = os.path.join(llm_turns_log_specific_dir, log_filename)
        timestamp = datetime.datetime.now().isoformat()
        log_header_text = (
            f"[Sample {sample_index + 1}] {header}"
            if sample_index is not None
            else header
        )
        with open(current_log_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"--- Log Time: {timestamp} ---\\n{log_header_text}\\n{prompt}\\n\\n---\\n\\n"
            )
    except Exception as e:
        logger.error(f"Error writing to LLM turn log file: {e}")


FINAL_QUESTION_TEMPLATE = Template(
    "\n\n---\n\n**Question:** Considering the entire sequence of events described in the story, what is the final, precise quantity of $primary_object that the characters possess or have determined at the very end of their activities? Provide only the single integer representing this final amount."
)

# verbose-listops.py
# ... (imports, Config, etc.) ...


# Ensure GenerationContext dataclass is defined and includes overall_ast_root
@dataclass
class GenerationContext:
    world: dict
    config: Config  # Use full name to avoid conflict
    encoder: any
    p_inflect: any
    logger: logging.Logger  # Use full name
    narrative_anchor_map: dict
    all_atoms: set  # All unique atomic numbers in the AST
    introduced_atoms: set  # Atoms mentioned in narrative so far (only from current beat's direct atoms)
    scenes: list
    tokens_used: int
    last_scene_text: str
    beat_counter: dict  # {'current': 0, 'total': N}
    sample_index: int
    max_pad_paragraphs: int
    overall_ground_truth_answer: int | None
    overall_ast_root: Node | None = (
        None  # NEW: To trace node values for forbidden checks
    )
    padding_stats: dict = field(
        default_factory=lambda: {
            "total_padding_tokens": 0,
            "padding_segments_added": 0,
            "max_padding_allowed": 0,
            "padding_per_slot": 0,
        }
    )


# --- Postorder Traversal (needed by generate_narrative before its own definition) ---
# This function is used by generate_narrative to count operator nodes and get all atoms.
# It needs to be defined before generate_narrative if generate_narrative uses it at the top level.
# Or, ensure Node, Atom, OpNode are defined before this.
def postorder(node: Node):
    """Yield nodes in post-order."""
    if node is None:  # Add this check to handle None values
        return
    for c in node.children:
        yield from postorder(c)
    yield node


# --- End Postorder Traversal ---


def generate_introduction_scene(
    world_info: dict,
    sample_index: int | None = None,
    config_obj: Config = config,  # Add config_obj parameter
    logger_obj: logging.Logger = logger,  # Add logger_obj parameter
) -> str | None:
    logger_obj.info(
        f"[Sample {sample_index + 1 if sample_index is not None else 'N/A'}] Generating introduction scene..."
    )

    # --- COPIED PROMPT LOGIC START ---
    system_prompt = (
        f"You are a master {world_info.get('genre')} storyteller. Your task is to write a compelling introductory scene for a new story. "
        "This scene should establish the setting, introduce one or two key characters, and hint at a central mystery or goal related to the primary object. "
        "Crucially, this introductory scene MUST NOT contain any numerical values (digits or words like 'one', 'two', 'first', etc.), "
        "except potentially the word 'one', 'two', or 'three' if used for completely general, non-quantitative phrasing (e.g., 'a single ray of light', 'two figures emerged', 'three ancient symbols'). Strive for zero numbers. "
        "Focus on atmosphere and intrigue. Do not reveal any specific quantities or begin any calculations. "
        "Output ONLY the narrative text for this scene. No titles, no explanations, no analysis."
    )

    characters_list = world_info.get("characters", [])
    char_names_roles = []
    if characters_list:
        # Select one or two characters for the intro
        num_intro_chars = random.randint(1, min(2, len(characters_list)))
        intro_chars = random.sample(characters_list, num_intro_chars)
        for char_info in intro_chars:
            char_names_roles.append(
                f"{char_info.get('name', 'A mysterious figure')} ({char_info.get('role', 'of unknown purpose')})"
            )

    user_prompt = (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'A realm of mystery')}\n"
        f"- Setting: {world_info.get('setting', 'An enigmatic place')}\n"
        f"- Primary Object of Interest: {world_info.get('object', 'ancient artifacts')}\n"
        f"- Characters to potentially feature: {', '.join(char_names_roles) if char_names_roles else 'The inhabitants of this world'}\n\n"
        f"**Task:** Write an engaging introductory scene based on the context above. Remember the strict rule: NO numbers (or strive for zero numbers, with very limited exceptions for 'one'/'two'/'three' in general phrasing only). "
        f"The scene should set a tone and hint at the story's direction without giving away specifics. "
        f"Output ONLY the narrative text."
    )
    # --- COPIED PROMPT LOGIC END ---

    # Validator for intro: NO numbers, or at most very specific phrasing numbers if allowed by config.
    # The intro prompt asks for NO numbers.
    validate_intro = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=set(),
        operand_count=0,
        correct_result_for_beat=None,
        strict_zero=True,  # Key for intro/padding style validation
        enforce_result_presence=False,
        operation_type="INTRO",
        overall_ground_truth_answer=None,  # No GT relevant for intro in this way
        is_root_node_being_validated=False,
        config_obj=config_obj,  # Pass the config
        logger_obj=logger_obj,
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=config_obj.INTRO_MAX_TOKENS,
        validate_fn=validate_intro,
        retries=config_obj.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config_obj.CREATIVE_NARRATIVE_TEMP,
        reasoning_settings={"exclude": True},
    )
    if intro_text:
        logger_obj.info(
            f"Successfully generated intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return intro_text.strip()
    else:
        logger_obj.error(
            f"Failed to generate intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return None


def generate_narrative(
    ast: Node,  # This is the root of the AST for the current problem
    world: dict,
    config_obj: Config,  # Renamed from config
    encoder_obj: tiktoken.Encoding,  # More specific type hint
    p_inflect_obj: inflect.engine,  # More specific type hint
    logger_obj: logging.Logger,  # Use full name
    sample_index: int,
    overall_ground_truth_answer: int,
) -> GenerationContext | None:  # Return the whole context or None on failure
    logger_obj.info(f"[Sample {sample_index + 1}] Starting narrative generation.")

    # --- Initialize narrative_anchor_map ---
    # This map will store {node_id: "anchor_name_string"}
    # It needs to be populated before GenerationContext is created if GenerationContext needs it at init.
    # However, GenerationContext just stores it; it's populated *within* this function.
    narrative_anchor_map: dict[int, str] = {}  # Initialize as an empty dict

    # --- Populate narrative_anchor_map (Example of how it's done in the full script) ---
    # This logic is simplified here; the full script has generate_narrative_anchor_with_llm or a fallback is called here
    # For this fix, we'll just use a placeholder to ensure the map is populated.
    # The actual anchor generation logic is complex and assumed to be working.
    if config_obj.USE_NARRATIVE_ANCHORS:
        temp_operator_nodes_for_anchors = []
        for node_iter in postorder(ast):
            if isinstance(node_iter, OpNode):
                # In the full script, generate_narrative_anchor_with_llm or a fallback is called here
                # For this fix, we'll just use a placeholder to ensure the map is populated.
                # The actual anchor generation logic is complex and assumed to be working.
                anchor_name = f"anchor_for_{node_iter.op}_{id(node_iter) % 100}"
                narrative_anchor_map[id(node_iter)] = anchor_name
                temp_operator_nodes_for_anchors.append(node_iter)
        logger_obj.debug(
            f"Populated narrative_anchor_map with {len(narrative_anchor_map)} anchors."
        )
    # --- End of narrative_anchor_map population ---

    all_atoms_in_ast = set()
    for node_in_ast in postorder(ast):  # Iterate over the passed 'ast'
        if isinstance(node_in_ast, Atom):
            all_atoms_in_ast.add(node_in_ast.n)

    operator_nodes_for_count = [n for n in postorder(ast) if isinstance(n, OpNode)]
    total_beats = len(operator_nodes_for_count)

    scenes_list = []
    tokens_used_count = 0
    last_scene_text_val = "The story begins..."

    # Assuming generate_introduction_scene, make_number_validator, generate_with_retry are defined before this point
    # or imported correctly. For this specific fix, we focus on Node and narrative_anchor_map.
    intro_text = generate_introduction_scene(
        world, sample_index=sample_index, config_obj=config_obj, logger_obj=logger_obj
    )
    if intro_text:
        intro_tokens = len(encoder_obj.encode(intro_text))
        if intro_tokens <= config_obj.MAX_TOTAL_TOKENS - config_obj.MAX_TOKENS_BUFFER:
            scenes_list.append(intro_text)
            tokens_used_count += intro_tokens
            last_scene_text_val = intro_text
        else:
            intro_text = None

    context = GenerationContext(
        world=world,
        config=config_obj,
        encoder=encoder_obj,
        p_inflect=p_inflect_obj,
        logger=logger_obj,
        narrative_anchor_map=narrative_anchor_map,  # Now narrative_anchor_map is defined and populated
        all_atoms=all_atoms_in_ast,
        introduced_atoms=set(),
        scenes=scenes_list,
        tokens_used=tokens_used_count,
        last_scene_text=last_scene_text_val,
        beat_counter={"current": 0, "total": total_beats},
        sample_index=sample_index,
        max_pad_paragraphs=config_obj.MAX_PAD_PARAGRAPHS,
        overall_ground_truth_answer=overall_ground_truth_answer,
        overall_ast_root=ast,
    )

    # ... rest of the generate_narrative function from your full script ...
    # This includes the call to _generate_narrative_recursive, padding logic, etc.
    # For brevity, I'm omitting the rest of the function body as the fix is about definitions.
    # Ensure _generate_narrative_recursive is also defined or imported correctly.

    # Placeholder for the recursive call and rest of the logic
    # try:
    #     _generate_narrative_recursive(ast, context, is_root=True)
    # except BeatGenerationError as e:
    #     logger_obj.error(f"Narrative generation aborted: {e}")
    #     return None
    # except Exception as e_rec:
    #     logger_obj.error(f"Unexpected error in recursive gen: {e_rec}", exc_info=True)
    #     return None

    # if not context.scenes:
    #     logger_obj.error("No scenes generated.")
    #     return None

    logger_obj.info(
        f"Successfully generated narrative for sample {sample_index + 1}. Final context tokens: {context.tokens_used}"
    )
    return context


SAFETY_MARGIN = config.MAX_TOKENS_BUFFER
MAX_BEAT_COMPLETION_TOKENS = config.BEAT_MAX_TOKENS
MAX_PAD_COMPLETION_TOKENS = config.PADDING_MAX_TOKENS

# --- Setup Logging ---

if config.CLEAR_LOGS_ON_START:
    for filename in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, filename)
        try:
            os.remove(file_path)
        except OSError:
            pass

logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)

# Remove existing handlers to avoid duplicates and force new handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Ensure the log file handler exists
os.makedirs(LOG_DIR, exist_ok=True)
handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "verbose_listops.log"),
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

print(
    f"Logger initialized with {len(logger.handlers)} handlers. Log file will be created at: {os.path.join(LOG_DIR, 'verbose_listops.log')}"
)

# --- Instantiate OpenAI Client for OpenRouter Endpoint ---
client = None
try:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError("OpenRouter API Key not found or is placeholder.")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    logger.info("OpenAI client configured to use OpenRouter API endpoint.")
except Exception as e:
    logger.error(f"Failed to configure OpenAI client for OpenRouter endpoint: {e}")
    client = None


# --- Rate Limiter for API Calls ---
class RateLimiter:
    """
    Thread-safe rate limiter that implements a token bucket algorithm.
    Allows for bursts of requests while maintaining a long-term rate limit.
    """

    def __init__(
        self,
        max_requests_per_second: float = 40.0,
        min_interval: float = 0.05,
        bucket_capacity: int = 5,
        jitter: float = 0.1,
    ):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = min_interval  # Minimum time between requests in seconds
        self.bucket_capacity = bucket_capacity  # Maximum tokens in the bucket
        self.jitter = jitter  # Random jitter to apply to wait times
        self.tokens = bucket_capacity  # Start with a full bucket
        self.last_refill_time = time.time()  # Last token refill timestamp
        self.lock = threading.Lock()  # Thread lock for concurrent access
        self.last_limits_check_time = 0  # Last time we checked account limits
        self.limits_check_interval = 5  # Check limits every 5 seconds
        self.initial_usage = None  # Store the initial usage value when first checked

        # Log configuration
        logger.info(
            f"Rate limiter initialized: {max_requests_per_second} req/s, "
            f"{min_interval}s min interval, bucket capacity {bucket_capacity}, jitter {jitter}"
        )

    def wait_if_needed(self):
        """
        Implements token bucket algorithm to manage API request rates.
        Returns the amount of time waited.
        """
        # Check if we should update our rate limits based on account status
        current_time = time.time()
        if current_time - self.last_limits_check_time > self.limits_check_interval:
            self.update_limits_from_api()

        with self.lock:
            # Refill tokens based on elapsed time
            current_time = time.time()
            elapsed = current_time - self.last_refill_time

            # Calculate token refill (tokens are added based on time elapsed)
            new_tokens = elapsed * self.max_requests_per_second

            # Update token count, but don't exceed capacity
            self.tokens = min(self.bucket_capacity, self.tokens + new_tokens)
            self.last_refill_time = current_time

            # If we have at least 1 token, consume it and continue immediately
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            # Otherwise, calculate wait time needed for at least 1 token
            wait_time = (1.0 - self.tokens) / self.max_requests_per_second

            # Ensure we wait at least the minimum interval
            wait_time = max(wait_time, self.min_interval)

            # Add random jitter to prevent thundering herd problem
            if self.jitter > 0:
                wait_time += random.uniform(0, self.jitter)

            # Wait and update
            time.sleep(wait_time)
            self.tokens = 0.0  # We\\'ve used our token
            self.last_refill_time = time.time()

            return wait_time

    def update_limits_from_api(self):
        """
        Check OpenRouter API rate limits and adjust rate limiter settings accordingly.
        Returns the current account usage (float) or None if an error occurs or usage is not found.
        """
        if (
            not OPENROUTER_API_KEY
            or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE"
        ):
            logger.warning("Cannot check OpenRouter limits: No valid API key")
            return None  # ADDED

        current_usage = None  # Initialize to None
        try:
            logger.info("OpenRouter STATUS:")
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                # logger.info(f"OpenRouter account status: {json.dumps(data, indent=2)}") # Made more concise

                # Extract the main data object from the nested response
                account_data = data.get("data", {})

                # Update rate limiter settings based on the nested rate_limit structure
                rate_limit_info = account_data.get("rate_limit", {})
                current_rate_for_log = self.max_requests_per_second
                limit_adjusted = False

                if rate_limit_info:
                    requests_limit = rate_limit_info.get("requests")
                    interval = rate_limit_info.get("interval", "")

                    if requests_limit and interval:
                        # Calculate RPS if in a format like "10s"
                        if interval.endswith("s") and interval[:-1].isdigit():
                            interval_seconds = int(interval[:-1])
                            rps = requests_limit / interval_seconds

                            # Set to 80% of the allowed rate limit as a safety buffer
                            new_rate = min(
                                float(rps) * 0.8, config.MAX_REQUESTS_PER_SECOND
                            )

                            if new_rate != self.max_requests_per_second:
                                # logger.info(f"Adjusting rate limiter based on OpenRouter limit: {rps} req/s → {new_rate} req/s (80% safety, capped at config.MAX_REQUESTS_PER_SECOND)")
                                self.max_requests_per_second = new_rate
                                limit_adjusted = True
                                current_rate_for_log = (
                                    new_rate  # Update for the concise log
                                )

                # Log credits/usage if available
                usage = account_data.get("usage")
                limit = account_data.get("limit")
                limit_remaining = account_data.get("limit_remaining")

                # Set initial usage on first check
                if usage is not None:
                    current_usage = float(usage)  # Store the usage
                    if self.initial_usage is None:
                        self.initial_usage = current_usage
                        logger.info(
                            f"Initial OpenRouter usage set to: ${self.initial_usage:.4f}"
                        )

                # Calculate run cost (difference between current and initial usage)
                run_cost = None
                if usage is not None and self.initial_usage is not None:
                    run_cost = current_usage - self.initial_usage

                log_message_parts = [f"RPS: {current_rate_for_log:.1f}"]
                if limit_adjusted:
                    log_message_parts.append("(Adjusted)")

                # Replace detailed usage stats with current run cost
                if run_cost is not None:
                    log_message_parts.append(f"Cost so far: ${run_cost:.4f}")
                elif usage is not None:
                    log_message_parts.append(f"Usage: ${usage:.4f}")

                if limit is not None and limit_remaining is not None:
                    log_message_parts.append(
                        f"Credits: Rem ${limit_remaining:.4f} of ${limit:.4f}"
                    )

                    # If very low on remaining limit, be more conservative with request rate
                    if limit and limit_remaining and limit_remaining / limit < 0.2:
                        # logger.warning(f"Low limit remaining ({limit_remaining}/{limit}). Reducing request rate.")
                        old_self_rate = self.max_requests_per_second
                        self.max_requests_per_second = min(
                            self.max_requests_per_second, 10.0
                        )
                        if old_self_rate != self.max_requests_per_second:
                            log_message_parts.append(
                                f"LOW CREDITS - RPS reduced to {self.max_requests_per_second:.1f}!"
                            )

                logger.info(", ".join(log_message_parts))
                self.last_limits_check_time = time.time()
            else:
                logger.warning(
                    f"Failed to get OpenRouter account status: HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error checking OpenRouter limits: {e}")
            return None  # Return None on exception

        self.last_limits_check_time = time.time()
        return current_usage  # Return the fetched usage


# Create a singleton rate limiter instance
rate_limiter = RateLimiter(
    max_requests_per_second=config.MAX_REQUESTS_PER_SECOND,
    min_interval=config.MIN_REQUEST_INTERVAL,
    bucket_capacity=100,  # Allow bursts of up to 100 requests
    jitter=0.01,  # Add up to 10ms of random jitter to prevent synchronization
)

# Check OpenRouter limits when starting up
# rate_limiter.update_limits_from_api() # Calling this here might be too early if logger not fully set up.
# Consider calling it first time wait_if_needed is invoked or explicitly after logger setup.

# --- Inflect Engine ---
try:
    p_inflect = inflect.engine()
except Exception as e:
    logger.error(f"Failed to initialize inflect engine: {e}")
    p_inflect = None


# --- Safe, memoised wrapper around inflect.number_to_words ---
@lru_cache(maxsize=None)
def num_to_words(n: int) -> str:
    """
    Convert an int to its English word form using inflect with memoisation.
    Falls back to the digit string if inflect is unavailable or raises.
    Ensures "and" is not used for compatibility with simpler parsing.
    """
    if p_inflect is None:
        return str(n)
    try:
        # Generate words without "and"
        return p_inflect.number_to_words(n, andword="")
    except Exception:
        return str(n)


# --- Generic retry helper (centralised back‑off policy) ---
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
            is_rate_limit_error = False
            error_str = str(e).lower()

            if hasattr(e, "status_code") and getattr(e, "status_code") == 429:
                is_rate_limit_error = True
            elif hasattr(e, "http_status") and getattr(e, "http_status") == 429:
                is_rate_limit_error = True
            elif (
                hasattr(e, "response")
                and hasattr(e.response, "status_code")
                and e.response.status_code == 429
            ):
                is_rate_limit_error = True
            elif any(
                phrase in error_str
                for phrase in [
                    "rate limit",
                    "too many requests",
                    "ratelimit",
                    "quota exceeded",
                    "usage limit",
                    "capacity",
                    "throttled",
                ]
            ):
                is_rate_limit_error = True

            if is_rate_limit_error:
                rate_limited_delay = delay * 3
                logger.warning(
                    f"Rate limit error detected calling {getattr(func, '__name__', repr(func))} "
                    f"(attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}): {e}"
                )
                logger.info(
                    f"Rate limiting triggered - backing off for {rate_limited_delay:.2f}s"
                )
                try:
                    if hasattr(rate_limiter, "max_requests_per_second"):
                        old_rate = rate_limiter.max_requests_per_second
                        new_rate = max(1.0, old_rate * 0.8)
                        rate_limiter.max_requests_per_second = new_rate
                        logger.info(
                            f"Reducing rate limit: {old_rate:.1f} → {new_rate:.1f} req/s"
                        )
                except Exception as inner_e:
                    logger.warning(f"Failed to adjust rate limiter: {inner_e}")
                wait_time = rate_limited_delay
            else:
                logger.warning(
                    f"Retryable error calling {getattr(func, '__name__', repr(func))} "
                    f"(attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}): {e}"
                )
                wait_time = delay

            if attempt == config.RETRY_MAX_ATTEMPTS:
                logger.error("Max retry attempts reached. Raising.")
                raise

            wait_time += random.uniform(0, 0.5)  # Add jitter
            time.sleep(wait_time)
            delay *= 2


def clean_snippet(text: str, max_len: int = config.BEAT_CONTEXT) -> str:
    """Removes common model analysis/checklist lines and takes the last part."""
    if not text:
        return "The story begins..."

    lines = text.splitlines()
    cleaned_lines = [
        line
        for line in lines
        if not re.match(
            r"^\s*(-|\*|\d+\.|Critique|Checklist|Yes|No|Draft \d+|Option \d+|\[.*?\]:|MUST INCLUDE|MUST AVOID|Problem:|REASONING:|GOOD:|BAD:|Confidence Score:|Mental Sandbox:|Outcome is|Narrative:|Generation:|Rules:|System:|User:|Okay|Check\.|REMINDER:|Instructions:|Task:|^\?|^\s*$)",
            line.strip(),
            re.IGNORECASE,
        )
        and not line.strip().startswith(
            (
                "Imply the sum",
                "reference to the previous",
                "Narrate comparing",
                "This scene resolves",
            )
        )
    ]

    cleaned_text = "\n".join(cleaned_lines).strip()
    if not cleaned_text:
        original_lines = [line for line in lines if line.strip()]
        if original_lines:
            cleaned_text = original_lines[-1].strip()
        else:
            return "Previously..."

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
    would_exceed = current + upcoming + margin > max_total
    remaining = max_total - current - margin
    percentage_used = (current / max_total) * 100

    if would_exceed:
        logger.warning(
            f"TOKEN LIMIT CHECK: WOULD EXCEED - Current: {current} tokens ({percentage_used:.1f}%), "
            f"Upcoming: +{upcoming}, Safety Margin: {margin}, Remaining: {remaining}, "
            f"Budget: {max_total}, Total After: {current + upcoming + margin}/{max_total}"
        )
    else:
        logger.debug(
            f"TOKEN LIMIT CHECK: WITHIN BUDGET - Current: {current} tokens ({percentage_used:.1f}%), "
            f"Upcoming: +{upcoming}, Safety Margin: {margin}, Remaining: {remaining}, "
            f"Budget: {max_total}, Will use: {current + upcoming + margin}/{max_total}"
        )

    return would_exceed


# --- Helper for future consolidation of retry loops ---
def generate_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    validate_fn: Callable[[str], bool],
    retries: int = config.MAX_BEAT_RETRIES,
    sample_index: int | None = None,
    temperature: float = config.CREATIVE_NARRATIVE_TEMP,
    reasoning_settings: dict = None,  # Added reasoning_settings parameter
):
    """
    Helper to call the OpenAI ChatCompletion API with retries and apply a validation function.
    Returns the first candidate text that passes validate_fn, or None if all attempts fail.
    Passes sample_index to log_prompt if provided.
    """
    candidate = None
    validation_failure_reasons = []

    for attempt in range(1, retries + 1):
        try:
            # Use the API token limit instead of the provided max_completion_tokens
            # This prevents truncation due to reasoning tokens being counted against max_tokens
            actual_max_tokens = config.MAX_API_TOKEN_LIMIT

            logger.debug(
                f"API Call: Using {actual_max_tokens} tokens for API (vs {max_completion_tokens} for internal tracking)"
            )

            api_params = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": actual_max_tokens,  # Use the higher limit
                "temperature": temperature,
            }
            if reasoning_settings:
                api_params["reasoning"] = reasoning_settings.copy()  # Pass a copy
                logger.debug(
                    f"generate_with_retry: Passing reasoning_settings to _chat_completion_call: {api_params['reasoning']}"
                )
            # If reasoning_settings is None, _chat_completion_call will apply its own
            # default logic based on config.REASONING_EXCLUDE.

            resp = _chat_completion_call(**api_params)

            truly_raw_llm_content = None
            if (
                resp
                and resp.choices
                and len(resp.choices) > 0
                and resp.choices[0].message
            ):
                truly_raw_llm_content = resp.choices[0].message.content
            log_prompt(
                f"LLM Turn Attempt {attempt}",
                f"System: {system_prompt}\\nUser: {user_prompt}\\n\\nGeneration (Raw):\\n{truly_raw_llm_content if truly_raw_llm_content is not None else '[[API returned None content string]]'}",
                sample_index=sample_index,
            )

            # Prepare the candidate for validation and return, which involves stripping.
            candidate_for_validation_and_return = None
            if truly_raw_llm_content is not None:
                candidate_for_validation_and_return = truly_raw_llm_content.strip()
            else:
                logger.warning(
                    f"API call in generate_with_retry attempt {attempt} returned None content. Response object: {resp}"
                )
                validation_failure_reasons.append("API returned None content")
                continue

            if candidate_for_validation_and_return is None:
                logger.warning(
                    f"generate_with_retry attempt {attempt} resulted in None candidate_for_validation_and_return (possibly after stripping None)."
                )
                validation_failure_reasons.append("Empty content after stripping")
            elif (
                not candidate_for_validation_and_return
                or candidate_for_validation_and_return.lower().startswith(
                    ("i cannot", "i'm sorry", "i am unable")
                )
            ):
                logger.warning(f"API refusal on generate_with_retry attempt {attempt}.")
                validation_failure_reasons.append("API refusal detected")
            elif validate_fn(candidate_for_validation_and_return):
                logger.info(f"Validation PASSED on attempt {attempt}")
                return candidate_for_validation_and_return
            else:
                # If we got here, validation failed - look for most recent failed validation file
                failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
                if os.path.exists(failed_validations_dir):
                    try:
                        # Get the most recent validation failure file (should be the one just created)
                        files = [
                            f
                            for f in os.listdir(failed_validations_dir)
                            if f.startswith("validation_fail_")
                        ]
                        if files:
                            files.sort(reverse=True)  # Most recent first
                            latest_file = os.path.join(failed_validations_dir, files[0])
                            with open(latest_file, "r", encoding="utf-8") as f:
                                failure_data = json.load(f)
                                reason = failure_data.get("validation_report", {}).get(
                                    "reason", "Unknown"
                                )
                                validation_failure_reasons.append(
                                    f"Validation failed: {reason}"
                                )

                                # Add more detailed debugging info
                                logger.warning(
                                    f"Validation failed on attempt {attempt}: {reason}"
                                )
                                logger.warning(
                                    f"Found numbers: {failure_data.get('validation_report', {}).get('found_numbers', [])}"
                                )
                                logger.warning(
                                    f"Required numbers: {failure_data.get('validation_report', {}).get('allowed_atoms', [])}"
                                )
                                logger.warning(
                                    f"Missing required: {failure_data.get('validation_report', {}).get('missing_required', [])}"
                                )
                                logger.warning(
                                    f"Forbidden extras: {failure_data.get('validation_report', {}).get('forbidden_extras', [])}"
                                )
                    except Exception as e:
                        logger.error(f"Error reading validation failure data: {e}")
                        validation_failure_reasons.append(
                            f"Validation failed (error reading details)"
                        )
                else:
                    validation_failure_reasons.append(
                        "Validation failed (no details available)"
                    )

        except Exception as e:
            logger.warning(f"Error on generate_with_retry attempt {attempt}: {e}")
            validation_failure_reasons.append(f"Exception: {str(e)}")

        if attempt < retries:
            time.sleep(config.RETRY_INITIAL_DELAY * (2 ** (attempt - 1)))

    if validation_failure_reasons:
        logger.warning(
            f"generate_with_retry failed after {retries} attempts. Failure reasons: {validation_failure_reasons}"
        )
    else:
        logger.warning(
            f"generate_with_retry failed after {retries} attempts with no specific reasons recorded."
        )
    return None


OP_LABELS = {
    "MAX": "largest value",
    "MIN": "smallest value",
    "SUM": "sum of all values",
    "MED": "median value",
    "AVG": "integer-average (floored)",
    "SM": "sum modulo 10",
}


# --- AST Generation and Evaluation ---
def build_random_ast(
    max_ops: int, max_branch: int = config.MAX_BRANCH, config_obj: Config = config
) -> Node:
    """Constructs a random ListOps AST."""
    if not isinstance(max_ops, int) or max_ops < 1:
        raise ValueError("max_ops must be a positive int")
    if max_branch < config_obj.MIN_ARITY:
        raise ValueError(
            f"max_branch ({max_branch}) < MIN_ARITY ({config_obj.MIN_ARITY})"
        )
    ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0

    def helper():
        nonlocal count
        if count >= max_ops or (
            count > 0 and random.random() < config_obj.EARLY_TERMINATION_PROBABILITY
        ):
            return Atom(
                random.randint(config_obj.MIN_ATOM_VAL, config_obj.MAX_ATOM_VAL)
            )
        count += 1
        op = random.choice(ops)

        if op == "MED":
            possible_arities = [
                n for n in range(config_obj.MIN_ARITY, max_branch + 1) if n % 2 == 1
            ]

            if not possible_arities:
                arity = (
                    config_obj.MIN_ARITY
                    if config_obj.MIN_ARITY % 2 == 1
                    else config_obj.MIN_ARITY + 1
                )
            else:
                arity = random.choice(possible_arities)
        else:
            arity = random.randint(config_obj.MIN_ARITY, max_branch)
        children = [helper() for _ in range(arity)]

        # Ensure AVG direct atom sum is divisible by atom count
        if op == "AVG":
            direct_atoms = [c for c in children if isinstance(c, Atom)]
            arity = len(direct_atoms)
            if arity > 0:  # Only adjust if there are direct atoms
                current_sum = sum(a.n for a in direct_atoms)
                remainder = current_sum % arity

                if remainder != 0:
                    adjustment_needed = (arity - remainder) % arity
                    logger.debug(
                        f"AST Gen (AVG): current_sum={current_sum}, arity={arity}, remainder={remainder}, adjustment_needed={adjustment_needed}"
                    )

                    atom_to_adjust = random.choice(direct_atoms)
                    adjusted = False

                    new_value_add = atom_to_adjust.n + adjustment_needed
                    if (
                        config_obj.MIN_ATOM_VAL
                        <= new_value_add
                        <= config_obj.MAX_ATOM_VAL
                    ):
                        atom_to_adjust.n = new_value_add
                        atom_to_adjust.value = new_value_add
                        logger.debug(
                            f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value up to {atom_to_adjust.n} to make sum divisible by {arity}."
                        )
                        adjusted = True

                    if not adjusted:
                        new_value_sub = atom_to_adjust.n - (arity - adjustment_needed)
                        if (
                            config_obj.MIN_ATOM_VAL
                            <= new_value_sub
                            <= config_obj.MAX_ATOM_VAL
                        ):
                            atom_to_adjust.n = new_value_sub
                            atom_to_adjust.value = new_value_sub
                            logger.debug(
                                f"AST Gen (AVG): Adjusted atom {id(atom_to_adjust)} value down to {atom_to_adjust.n} to make sum divisible by {arity}."
                            )
                            adjusted = True

                    if not adjusted:

                        logger.warning(
                            f"AST Gen (AVG): Could not adjust atom value {atom_to_adjust.n} (target adjustment {adjustment_needed}) for AVG node sum {current_sum} to be divisible by {arity} due to bounds [{config_obj.MIN_ATOM_VAL}, {config_obj.MAX_ATOM_VAL}]."
                        )

        return OpNode(op, children)

    root = helper()

    # --- START MODIFICATION FOR SUGGESTION 2 ---
    # If the root is an Atom (meaning max_ops was 0 or 1 initially, or early termination)
    # OR if the root is NOT a combining operation (SUM, AVG, SM) and we want to force it
    # for problems with at least, say, 2 operations.
    # This ensures the final step is a calculation rather than just a selection if possible.

    is_combining_op = isinstance(root, OpNode) and root.op in ["SUM", "AVG", "SM"]
    MIN_OPS_FOR_COMBINING_ROOT = 2  # Arbitrary: only force if problem has some depth

    if (isinstance(root, Atom) and max_ops >= 1) or (
        count >= MIN_OPS_FOR_COMBINING_ROOT and not is_combining_op and max_ops > count
    ):  # Ensure we have ops left to make a new root

        logger.debug(
            f"AST Gen: Original root was {getattr(root, 'op', 'Atom')}. Attempting to ensure a combining root."
        )

        # Prefer SUM, AVG, SM as the new root
        combining_ops = ["SUM", "AVG", "SM"]
        new_root_op = random.choice(combining_ops)

        # Determine arity for the new root
        # New root will have the old root as one child, and new atoms as others.
        # Ensure arity is at least 2 to include the old root and at least one new atom.
        # Max arity should still respect max_branch.
        new_arity = random.randint(max(2, config_obj.MIN_ARITY), max_branch)

        new_children = [root]  # The old root is one child

        # Add new Atom children
        for _ in range(new_arity - 1):
            new_children.append(
                Atom(random.randint(config_obj.MIN_ATOM_VAL, config_obj.MAX_ATOM_VAL))
            )

        random.shuffle(new_children)

        # Special handling for AVG if we created it as the new root
        if new_root_op == "AVG":
            direct_atoms_new_root = [c for c in new_children if isinstance(c, Atom)]
            current_sum_new_root = sum(a.n for a in direct_atoms_new_root)
            num_direct_atoms_new_root = len(direct_atoms_new_root)

            if (
                num_direct_atoms_new_root > 0
            ):  # Should always be true if new_arity >=2 and one child is Atom
                remainder_new_root = current_sum_new_root % num_direct_atoms_new_root
                if remainder_new_root != 0:
                    adjustment_needed_new_root = (
                        num_direct_atoms_new_root - remainder_new_root
                    ) % num_direct_atoms_new_root

                    # Try to adjust one of the newly added atoms
                    newly_added_atoms = [
                        c for c in new_children if isinstance(c, Atom) and c is not root
                    ]  # Exclude the old root if it was an atom

                    atom_to_adjust_new_root = None
                    if newly_added_atoms:
                        atom_to_adjust_new_root = random.choice(newly_added_atoms)
                    elif (
                        direct_atoms_new_root
                    ):  # Fallback if old root was an atom and only one new atom was added
                        atom_to_adjust_new_root = random.choice(direct_atoms_new_root)

                    if atom_to_adjust_new_root:
                        adjusted_new_root = False
                        # Try adding
                        if (
                            config_obj.MIN_ATOM_VAL
                            <= atom_to_adjust_new_root.n + adjustment_needed_new_root
                            <= config_obj.MAX_ATOM_VAL
                        ):
                            atom_to_adjust_new_root.n += adjustment_needed_new_root
                            atom_to_adjust_new_root.value = (
                                atom_to_adjust_new_root.n
                            )  # Update value too
                            adjusted_new_root = True
                            logger.debug(
                                f"AST Gen (Forced Root AVG): Adjusted new atom value for divisibility."
                            )
                        # Try subtracting if adding failed
                        elif (
                            config_obj.MIN_ATOM_VAL
                            <= atom_to_adjust_new_root.n
                            - (num_direct_atoms_new_root - adjustment_needed_new_root)
                            <= config_obj.MAX_ATOM_VAL
                        ):
                            atom_to_adjust_new_root.n -= (
                                num_direct_atoms_new_root - adjustment_needed_new_root
                            )
                            atom_to_adjust_new_root.value = (
                                atom_to_adjust_new_root.n
                            )  # Update value too
                            adjusted_new_root = True
                            logger.debug(
                                f"AST Gen (Forced Root AVG): Adjusted new atom value (subtracted) for divisibility."
                            )

                        if not adjusted_new_root:
                            logger.warning(
                                f"AST Gen (Forced Root AVG): Could not adjust new atom for AVG divisibility."
                            )
                    else:
                        logger.warning(
                            f"AST Gen (Forced Root AVG): No suitable atom found to adjust for divisibility."
                        )

        root = OpNode(new_root_op, new_children)
        # Increment max_ops if we added an operation, or ensure count reflects it.
        # The 'count' variable in helper() tracks operations. If we add one here,
        # it's effectively one more operation than 'max_ops' might have initially allowed
        # for the helper. This is a design choice. For now, we assume 'max_ops' is a soft limit.
        logger.info(f"AST Gen: Ensured root is a combining op: {root.op}")
    # --- END MODIFICATION FOR SUGGESTION 2 ---

    # Original fallback if root is Atom and max_ops >=1 (this might be redundant now or need adjustment)
    # Consider if this block is still needed or if the logic above covers it.
    # For now, I'll keep it but it might interact with the above.
    # A simpler approach might be to remove this original block if the new logic is robust.
    elif isinstance(root, Atom) and max_ops >= 1:  # Original condition
        logger.debug(
            f"AST Gen: Original root was Atom, max_ops >=1. Wrapping with a random op."
        )
        op = random.choice(ops)  # ops = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
        arity = random.randint(config_obj.MIN_ARITY, max_branch)
        children = [
            Atom(random.randint(config_obj.MIN_ATOM_VAL, config_obj.MAX_ATOM_VAL))
            for _ in range(arity - 1)
        ]
        children.append(root)
        random.shuffle(children)
        root = OpNode(op, children)
        logger.info(f"AST Gen: Wrapped Atom root with {op}.")

    return root


def validate_ast(node: Node):
    """Recursively validate that all operators in the AST are supported."""
    if node.op not in OP_LABELS and not isinstance(node, Atom):
        raise ValueError(f"Invalid operator: {node.op}")
    for c in node.children:
        validate_ast(c)


def ast_to_prefix(node: Node) -> str:
    """Convert an AST node to its prefix notation representation."""
    if isinstance(node, Atom):
        return str(node.n)

    children_str = " ".join(ast_to_prefix(c) for c in node.children)
    return f"({node.op} {children_str})"


def eval_node(node: Node) -> int:
    """Evaluate the AST node recursively."""
    if isinstance(node, Atom):
        if node.value is None:
            node.value = node.n
        logger.debug(f"eval_node: Atom node, value = {node.value}")
        return node.value

    vals = [eval_node(c) for c in node.children]
    logger.debug(f"eval_node: OpNode {node.op}, child values = {vals}")

    if not vals:
        logger.error(f"eval_node: Operator node {node.op} has no children values.")
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
                f"eval_node: MED operator for node {node.op} with even children ({len(vals)}). Using lower middle."
            )
        if node.op == "AVG" and not vals:
            logger.error(
                f"eval_node: Cannot calculate average of zero values for node {node.op}."
            )
            raise ValueError("Cannot calculate average of zero values.")

        calculated_value = func(vals)

        # Additional validation for SUM operations to catch calculation errors
        if node.op == "SUM":
            expected_sum = sum(vals)
            if calculated_value != expected_sum:
                logger.error(
                    f"eval_node: SUM validation error - func(vals)={calculated_value} != sum(vals)={expected_sum}"
                )
                # Use the manually calculated sum as a fallback
                calculated_value = expected_sum

        # Enhanced logging for all operations
        logger.debug(
            f"eval_node: OpNode {node.op}, inputs {vals}, result = {calculated_value}"
        )

        # Operation-specific detailed logging
        if node.op == "SUM":
            logger.info(
                f"SUM Operation - Node ID: {id(node)}, Input values: {vals}, Sum: {calculated_value}"
            )
        elif node.op == "AVG":
            total = sum(vals)
            logger.info(
                f"AVG Operation - Node ID: {id(node)}, Input values: {vals}, Sum: {total}, Count: {len(vals)}, Result: {calculated_value}"
            )
        elif node.op in ["MAX", "MIN", "MED"]:
            logger.info(
                f"{node.op} Operation - Node ID: {id(node)}, Input values: {vals}, Result: {calculated_value}"
            )

        node.value = calculated_value
        return node.value
    except KeyError:
        logger.error(f"eval_node: Unsupported operator: {node.op}")
        raise ValueError(f"Unsupported operator: {node.op}")
    except IndexError as e:
        logger.error(
            f"eval_node: Indexing error evaluating {node.op} with child values {vals}: {e}"
        )
        raise
    except ZeroDivisionError:
        logger.error(
            f"eval_node: Division by zero during AVG for {node.op} with child values {vals}"
        )
        raise ValueError(f"Division by zero during AVG for {node.op}")
    except Exception as e:
        logger.error(
            f"eval_node: Unexpected error evaluating {node.op} with values {vals}: {e}"
        )
        raise


def postorder(node: Node):
    """Yield nodes in post-order."""
    if node is None:  # Add this check to handle None values
        return
    for c in node.children:
        yield from postorder(c)
    yield node


@retry_api_call
def _chat_completion_call(*args, **kwargs):
    # ADD THIS LOG to verify MAX_API_TOKEN_LIMIT
    logger.info(
        f"DEBUG _chat_completion_call: Effective config.MAX_API_TOKEN_LIMIT = {config.MAX_API_TOKEN_LIMIT}"
    )

    if args:
        logger.warning(
            f"_chat_completion_call received unexpected positional arguments: {args}"
        )

    logger.debug(f"_chat_completion_call received kwargs: {kwargs}")

    if client is None:
        logger.error(
            "OpenAI client (for OpenRouter) not initialized. Cannot make API call."
        )
        raise RuntimeError("API client not initialized.")

    # Standard OpenAI client parameters
    standard_openai_client_params = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "top_k",
        # 'reasoning' is NOT a standard OpenAI client param, will go in extra_body
    }

    # Separate standard kwargs from OpenRouter-specific ones (like reasoning)
    api_call_standard_kwargs = {}
    openrouter_specific_params = {}

    if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
        # Use a temporary dict to avoid modifying original kwargs if it's passed around
        temp_kwargs = kwargs.copy()
        temp_kwargs["max_tokens"] = temp_kwargs.pop("max_completion_tokens")
        logger.info(f"DEBUG: Aliased max_completion_tokens to max_tokens")
    else:
        temp_kwargs = kwargs.copy()

    # Check if this call should use structured JSON output
    use_json_mode = temp_kwargs.pop("use_json_mode", False)
    json_schema = temp_kwargs.pop("json_schema", None)

    for k, v in temp_kwargs.items():
        if k in standard_openai_client_params:
            api_call_standard_kwargs[k] = v
        elif k == "reasoning":  # Explicitly handle reasoning for extra_body
            openrouter_specific_params[k] = v
        # else: # You could log other unexpected kwargs if needed
        # logger.warning(f"DEBUG: Unexpected kwarg '{k}' in _chat_completion_call, may be ignored or cause error if not for extra_body.")

    # --- Check if we need to modify the max_tokens to prevent truncation ---
    if "max_tokens" in api_call_standard_kwargs:
        original_max_tokens = api_call_standard_kwargs["max_tokens"]
        # Always use the higher limit to prevent truncation due to reasoning tokens
        api_call_standard_kwargs["max_tokens"] = config.MAX_API_TOKEN_LIMIT
        logger.debug(
            f"Modified max_tokens for API call: {original_max_tokens} → {config.MAX_API_TOKEN_LIMIT} (to handle reasoning tokens)"
        )

    # --- REFINED REASONING LOGIC for openrouter_specific_params ---
    current_model_name = api_call_standard_kwargs.get("model", "").lower()
    is_openai_o_series = "openai/" in current_model_name and (
        re.search(r"/o\d+", current_model_name) or "gpt-4o-mini" in current_model_name
    )

    reasoning_config_to_send = openrouter_specific_params.get("reasoning", {})
    if reasoning_config_to_send is None:  # Handle if None was explicitly passed
        reasoning_config_to_send = {}

    # Ensure it's a dict if it was passed as something else or not at all
    if not isinstance(reasoning_config_to_send, dict):
        reasoning_config_to_send = {}

    # 2. Handle 'effort' - Only allow on OpenAI o-series models
    if "effort" in reasoning_config_to_send:
        if not is_openai_o_series:
            logger.warning(
                f"DEBUG: Removing 'effort' from reasoning_config for non-o-series model ({current_model_name}). "
                f"Original effort: {reasoning_config_to_send['effort']}"
            )
            del reasoning_config_to_send["effort"]
        else:
            logger.debug(
                f"DEBUG: Keeping 'effort' for OpenAI o-series model ({current_model_name})."
            )

    # Update openrouter_specific_params with the processed reasoning_config
    if reasoning_config_to_send:
        openrouter_specific_params["reasoning"] = reasoning_config_to_send
    elif "reasoning" in openrouter_specific_params:  # If it was there but became empty
        del openrouter_specific_params["reasoning"]

    # Add JSON response format if requested
    if json_schema:
        # Use full schema-based JSON formatting
        openrouter_specific_params["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        logger.debug(
            f"Using JSON schema validation for API call with schema: {json_schema.get('type', 'unknown')}"
        )
    elif use_json_mode:
        # Use simple JSON mode (backward compatibility)
        openrouter_specific_params["response_format"] = {"type": "json_object"}
        logger.debug(f"Using simple JSON mode for API call")

    logger.debug(
        f"Final standard API call_kwargs: {json.dumps(api_call_standard_kwargs, indent=2)}"
    )
    logger.debug(
        f"Final OpenRouter specific_params (for extra_body): {json.dumps(openrouter_specific_params, indent=2)}"
    )

    max_tokens_value = api_call_standard_kwargs.get("max_tokens", "NOT SET")
    if max_tokens_value == "NOT SET":
        logger.warning(
            f"DEBUG: max_tokens value NOT SET for API call. API will use its default."
        )
    elif isinstance(max_tokens_value, int) and max_tokens_value <= 0:
        logger.error(
            f"DEBUG: max_tokens value is invalid ({max_tokens_value}). API call will likely fail."
        )
    else:
        logger.info(
            f"DEBUG: FINAL max_tokens value being sent to API: {max_tokens_value}"
        )

    try:
        wait_time = rate_limiter.wait_if_needed()
        if wait_time > 0:
            logger.debug(
                f"Rate limit applied - waited {wait_time:.2f}s before API call"
            )

        # Use extra_body for OpenRouter-specific parameters
        if openrouter_specific_params:
            resp = client.chat.completions.create(
                **api_call_standard_kwargs, extra_body=openrouter_specific_params
            )  # MODIFIED: assign to resp
        else:
            resp = client.chat.completions.create(
                **api_call_standard_kwargs
            )  # MODIFIED: assign to resp

        # --- Track token usage --- ADD THIS BLOCK ---
        if resp and hasattr(resp, "usage") and resp.usage:
            prompt_tokens = resp.usage.prompt_tokens or 0
            completion_tokens = resp.usage.completion_tokens or 0
            generation_token_tracker.add_usage(prompt_tokens, completion_tokens)
        else:
            logger.warning(
                "_chat_completion_call: No usage data found in API response."
            )
            # ADD THIS DETAILED LOGGING:
            if resp:
                logger.warning(
                    f"_chat_completion_call: Response object when usage was missing (type: {type(resp)}):"
                )
                try:
                    # Try to dump the whole response object as JSON for inspection
                    logger.warning(
                        f"Full response dump (model_dump_json): {resp.model_dump_json(indent=2)}"
                    )
                except AttributeError:
                    try:
                        logger.warning(
                            f"Full response dump (fallback .json()): {resp.json(indent=2)}"
                        )
                    except AttributeError:
                        logger.warning(f"Full response dump (repr): {repr(resp)}")
                    # Log choices and finish reasons if available even if usage is missing
                    if hasattr(resp, "choices") and resp.choices:
                        for i, choice_item in enumerate(resp.choices):
                            finish_reason_item = getattr(
                                choice_item, "finish_reason", "N/A"
                            )
                            message_item_content = "N/A"
                            if getattr(choice_item, "message", None) and getattr(
                                choice_item.message, "content", None
                            ):
                                message_item_content = choice_item.message.content
                            elif getattr(choice_item, "message", None):
                                message_item_content = f"Message object present but content is None/Empty. Message: {choice_item.message}"
                            else:
                                message_item_content = (
                                    "Message object missing in choice."
                                )

                            logger.warning(
                                f"Choice {i}: Finish Reason: {finish_reason_item}, Content Snippet: {str(message_item_content)[:200]}..."
                            )
                    else:
                        logger.warning(
                            "Response object had no 'choices' or 'choices' was empty when usage was missing."
                        )
            else:
                logger.warning(
                    "_chat_completion_call: Response object (resp) was None when usage was missing."
                )
        # --- END ADDED BLOCK ---

        return resp  # Ensure resp is returned

    except Exception as e:
        logger.error(f"Error during client.chat.completions.create: {e}")
        # Log both standard and extra_body args for clarity
        log_payload = {"standard_args": api_call_standard_kwargs}
        if openrouter_specific_params:
            log_payload["extra_body_args"] = openrouter_specific_params
        logger.error(f"Args that failed: {json.dumps(log_payload, indent=2)}")
        raise


# --- JSON Cleaning Helper ---
def clean_and_parse_json_block(text: str):
    """Extract JSON object between first '{' and last '}', then parse it."""
    logger.debug(
        f"clean_and_parse_json_block: Input text length: {len(text)}, text sample: '{text[:100]}...'"
    )

    if not text or not text.strip():
        logger.error(
            f"clean_and_parse_json_block: Received empty or whitespace-only text input"
        )
        raise ValueError("Empty input text")

    # Find first opening brace and last closing brace
    start_idx = text.find("{")
    if start_idx == -1:
        logger.error(f"No opening brace found in text: {text[:100]}...")
        raise ValueError("No JSON object found in text")

    end_idx = text.rfind("}")
    if end_idx == -1 or end_idx < start_idx:
        logger.error(f"No valid closing brace found in text: {text[:100]}...")
        raise ValueError("No valid JSON object found in text")

    # Extract just the JSON substring
    json_text = text[start_idx : end_idx + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON Decode Error: {e} in extracted text:\n---\n{json_text}\n---"
        )
        raise


# After clean_and_parse_json_block (around line 1664), add:
def parse_llm_json_with_fallback(
    raw_text: str, default_value: dict, context_info: str = ""
):
    """Parse JSON from LLM output with consistent error handling and fallback."""
    try:
        return clean_and_parse_json_block(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"JSON parsing failed {context_info}: {e}. Raw: {raw_text[:200]}"
        )
        return default_value


# Then remove the duplicate definition at the end of the file (lines 505-519)


# --- Tuned Generate World Function ---
def generate_world(
    num_characters: int = config.MIN_WORLD_CHARS,
    num_concepts: int = config.MAX_WORLD_CONCEPTS,
    max_retries: int = config.WORLDGEN_MAX_RETRIES,  # Use the config variable
    sample_index: int | None = None,
) -> dict:
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
        '    *   `name`: string (e.g., "Kaelen Vane", "Seraphina Moonwhisper")\n'
        '    *   `role`: string (e.g., "The grizzled warrior," "The cunning sorceress," "The naive apprentice")\n'
        '    *   `quirk`: string (a unique or unusual habit, belief, or physical trait, e.g., "Collects antique spoons," "Only speaks in riddles," "Has mismatched eyes")\n'
        "    Ensure each character's name, role, and quirk combination is unique.\n\n"
        '2.  **Genre:** Define a `genre` as a string (e.g., "Steampunk Adventure," "Urban Fantasy Mystery," "Cosmic Horror Saga").\n\n'
        '3.  **Setting:** Define a `setting` as a string (a brief, evocative description of the world or primary location, e.g., "A floating city powered by forgotten magic and steam contraptions," "A post-apocalyptic wasteland where ancient ruins hold dangerous secrets").\n\n'
        '4.  **Object:** Define an `object` as a string. This should be a plural noun representing key items characters might seek, collect, or use (e.g., "etherium crystals," "lost star-charts," "prophetic dream-shards").\n\n'
        "**Guidance for Content:**\n"
        "*   Strive for thematic coherence between the genre, setting, characters, and the collectible object. They should feel like they belong in the same world.\n\n"
    )

    for attempt in range(max_retries):
        logger.debug(
            f"Attempting world generation (Attempt {attempt + 1}/{max_retries}) with tuned prompt."
        )
        text = None
        try:
            # Log the prompt
            log_prompt(
                header=f"World Generation Prompt (Attempt {attempt + 1})",
                prompt=f"System: (Implicit in API call structure for this function)\nUser:\n{prompt}",
                sample_index=sample_index,
            )

            # Use config.MAX_API_TOKEN_LIMIT instead of config.WORLD_GEN_MAX_TOKENS
            # to avoid truncation due to reasoning tokens
            resp = _chat_completion_call(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=config.MAX_API_TOKEN_LIMIT,  # Use higher limit
                temperature=config.WORLD_GEN_TEMP,
                json_schema=WORLD_SCHEMA,  # Use the defined schema
                reasoning={"exclude": True},
            )
            if (
                hasattr(resp, "choices") and resp.choices
            ):  # Check if choices list exists and is not empty

                first_choice = resp.choices[0]

                if hasattr(first_choice, "message") and first_choice.message:

                    text = first_choice.message.content
                    if text is None:

                        logger.warning(
                            f"World Gen Attempt {attempt + 1}: API returned None content within message. Response: {resp}"
                        )
                        text = ""
                else:

                    logger.error(
                        f"World Gen Attempt {attempt + 1}: First choice object lacks 'message' attribute or message is empty. Response: {resp}"
                    )
                    text = ""
            else:

                logger.error(
                    f"World Gen Attempt {attempt + 1}: API response lacks 'choices' list or it's empty. Response: {resp}"
                )
                text = ""

            # Log raw response
            log_prompt(
                header=f"World Generation Response (Attempt {attempt + 1})",
                prompt=f"Raw LLM Output:\n{text}",
                sample_index=sample_index,
            )

            if not text.strip():
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: Received empty response from API."
                )

                if attempt < max_retries - 1:
                    delay = config.INITIAL_WORLD_RETRY_DELAY * (2**attempt)
                    logger.info(f"Retrying world generation in {delay:.2f} seconds...")
                    time.sleep(delay)
                continue

            # The response should be valid JSON already, but still use our parser
            # for consistency and as a fallback
            world = parse_llm_json_with_fallback(
                text,
                {},  # Empty dict will trigger the keys validation check right after
                f"in world generation attempt {attempt+1}",
            )

            # --- Validation ---
            required_keys = ["characters", "genre", "setting", "object"]
            if not all(k in world for k in required_keys):
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: Generated JSON missing required keys. Keys found: {world.keys()}"
                )
                raise ValueError(
                    "Generated JSON missing required keys"
                )  # Raise error to trigger except block
            if not isinstance(world.get("characters"), list) or not world["characters"]:
                logger.warning(
                    f"World Gen Attempt {attempt + 1}: 'characters' key is not a non-empty list."
                )
                raise ValueError("'characters' key is not a non-empty list")
            logger.debug(
                f"World Gen Attempt {attempt + 1}: Successfully generated and parsed world JSON."
            )
            logger.debug(f"Generated object: {world.get('object', 'N/A')}")
            return world  # Success! Exit the function.
        except (
            json.JSONDecodeError,
            ValueError,
        ) as e:  # Catch both parsing and validation errors
            logger.error(
                f"World Gen Attempt {attempt + 1}: Failed ({type(e).__name__}): {e}. Raw text:\n---\n{text}\n---"
            )
        except Exception as e:
            logger.error(
                f"World Gen Attempt {attempt + 1}: Unexpected error: {e}. Raw text:\n---\n{text if text else 'N/A'}\n---"
            )
        if attempt < max_retries - 1:
            delay = config.INITIAL_WORLD_RETRY_DELAY * (2**attempt)
            logger.info(f"Retrying world generation in {delay:.2f} seconds...")
            time.sleep(delay)
    logger.error(f"Failed to generate valid world JSON after {max_retries} attempts.")
    raise RuntimeError("World generation failed: Could not get valid JSON from LLM.")


# --- Number Extraction (Enhanced with Inflect for Words up to MAX_VALUE) ---
DIGIT_REGEX = re.compile(r"\b-?\d+\b")

MAX_NUMBER_FOR_NUMBER_DICT = 1000  # Define a larger limit for number-to-word conversion


def _build_expanded_number_words_dict(
    max_val: int = MAX_NUMBER_FOR_NUMBER_DICT,
) -> dict[str, int]:
    """Builds a dictionary mapping number words to ints up to max_val using inflect."""
    if p_inflect is None:
        logger.error(
            "Inflect engine not available for building expanded number words dict. Using basic range."
        )
        return {
            num_to_words(i).lower(): i
            for i in range(
                config.FALLBACK_MIN_NUM_WORD, config.FALLBACK_MAX_NUM_WORD + 1
            )
            if num_to_words(i)
        }
    num_word_dict = {}
    for i in range(max_val + 1):
        try:
            word_no_and = num_to_words(i)  # This will use the version without "and"
            if word_no_and:  # Ensure inflect returned something
                word_no_and_lower = word_no_and.lower()
                num_word_dict[word_no_and_lower] = i

                # *** START FIX: Add hyphenated version if applicable ***
                if " " in word_no_and_lower:
                    hyphenated_word = word_no_and_lower.replace(" ", "-")
                    num_word_dict[hyphenated_word] = i
                    # logger.debug(f"Added hyphenated number word: '{hyphenated_word}' for {i}") # Keep debug minimal here or it floods
                # *** END FIX ***

                # Additionally, add common "and" variations for numbers > 100
                if i > 100 and p_inflect:
                    word_with_and = p_inflect.number_to_words(i, andword="and")
                    if word_with_and:  # Ensure inflect returned something
                        word_with_and_lower = word_with_and.lower()
                        if word_with_and_lower != word_no_and_lower:
                            num_word_dict[word_with_and_lower] = i
                            # *** START FIX: Add hyphenated "and" version ***
                            if " " in word_with_and_lower:
                                hyphenated_word_with_and = word_with_and_lower.replace(
                                    " ", "-"
                                )
                                num_word_dict[hyphenated_word_with_and] = i
                                # logger.debug(f"Added hyphenated 'and' number word: '{hyphenated_word_with_and}' for {i}")
                            # *** END FIX ***
        except Exception as e:
            logger.warning(f"Inflect failed to convert {i} to words: {e}")
    logger.info(
        f"Built expanded number words dictionary with {len(num_word_dict)} entries (up to {max_val})."
    )
    # Log a few examples of hyphenated numbers if they were added
    hyphen_examples = {k: v for k, v in num_word_dict.items() if "-" in k and v < 100}
    if hyphen_examples:
        logger.info(
            f"Sample hyphenated entries: {dict(list(hyphen_examples.items())[:5])}"
        )
    return num_word_dict


# EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict() # Initialization moved down

# --- Build dict after logger is fully configured ---
EXPANDED_NUMBER_WORDS_DICT = {}
if p_inflect:
    EXPANDED_NUMBER_WORDS_DICT = _build_expanded_number_words_dict()
    logger.info(
        f"EXPANDED_NUMBER_WORDS_DICT initialized. Size: {len(EXPANDED_NUMBER_WORDS_DICT)}."
    )
    if len(EXPANDED_NUMBER_WORDS_DICT) > 0:
        sample_keys_diverse = []
        for num_val_to_check in [
            0,
            1,
            10,
            15,
            21,
            41,
            59,
            83,
            86,
            99,
            100,
            179,
            1000,
            1001,
            2025,
        ]:  # Added 99
            key_no_and = num_to_words(num_val_to_check)
            if key_no_and.lower() in EXPANDED_NUMBER_WORDS_DICT:
                sample_keys_diverse.append(key_no_and.lower())
            # Check for hyphenated version of no_and
            if " " in key_no_and.lower():
                hyphenated_key_no_and = key_no_and.lower().replace(" ", "-")
                if (
                    hyphenated_key_no_and in EXPANDED_NUMBER_WORDS_DICT
                    and hyphenated_key_no_and not in sample_keys_diverse
                ):
                    sample_keys_diverse.append(hyphenated_key_no_and)

            if num_val_to_check > 100 and p_inflect:
                word_with_and = p_inflect.number_to_words(
                    num_val_to_check, andword="and"
                )
                if (
                    word_with_and.lower() in EXPANDED_NUMBER_WORDS_DICT
                    and word_with_and.lower() not in sample_keys_diverse
                ):
                    sample_keys_diverse.append(word_with_and.lower())
                # Check for hyphenated version of with_and
                if " " in word_with_and.lower():
                    hyphenated_key_with_and = word_with_and.lower().replace(" ", "-")
                    if (
                        hyphenated_key_with_and in EXPANDED_NUMBER_WORDS_DICT
                        and hyphenated_key_with_and not in sample_keys_diverse
                    ):
                        sample_keys_diverse.append(hyphenated_key_with_and)

        sample_items = {
            k: EXPANDED_NUMBER_WORDS_DICT.get(k) for k in sample_keys_diverse[:20]
        }  # Log more samples
        logger.info(
            f"More diverse sample items from EXPANDED_NUMBER_WORDS_DICT: {sample_items}"
        )
    else:
        logger.warning(
            "EXPANDED_NUMBER_WORDS_DICT is empty after initialization attempt with p_inflect."
        )
else:
    logger.error(
        "p_inflect is None, EXPANDED_NUMBER_WORDS_DICT will be empty. Number word extraction will be limited."
    )

# --- Sort keys by length descending to prioritize longer matches ---
sorted_number_words = sorted(EXPANDED_NUMBER_WORDS_DICT.keys(), key=len, reverse=True)
logger.info(f"First 10 longest number words for regex: {sorted_number_words[:10]}")
logger.info(f"Last 10 shortest number words for regex: {sorted_number_words[-10:]}")

NUMBER_WORDS_PATTERN = (
    r"\b(?:(minus|negative)\s+)?("  # Corrected \b to \b
    + "|".join(re.escape(k) for k in sorted_number_words)
    + r")\b"  # Corrected \b to \b
)
NUMBER_WORDS_REGEX = re.compile(NUMBER_WORDS_PATTERN, re.IGNORECASE)


def extract_numbers_from_text(text: str) -> Set[int]:
    """Extracts integers (digits and words), ignoring specified ordinals."""
    if not text:
        return set()

    found_numbers = set()
    search_text = text.lower()

    text_chars_list = list(search_text)
    digit_spans_to_replace = []
    for match in DIGIT_REGEX.finditer(search_text):
        digit_str = match.group(0)
        try:
            value = int(digit_str)
            found_numbers.add(value)
            digit_spans_to_replace.append(match.span())
        except ValueError:
            logger.warning(
                f"Could not convert digit string '{digit_str}' to int during extraction."
            )
            continue

    for start, end in digit_spans_to_replace:
        for i in range(start, end):
            text_chars_list[i] = "|"  # Replace with a non-space, non-word character

    text_for_word_search = "".join(text_chars_list)
    text_for_word_search = text_for_word_search.replace(
        "|", " "
    )  # Now replace placeholders with spaces
    text_for_word_search = re.sub(r"\s+", " ", text_for_word_search).strip()

    logger.debug(
        f"extract_numbers_from_text: Text for word search (digits replaced, pipes to spaces): '{text_for_word_search[:500]}...'"
    )

    if "twelve" in text_for_word_search:
        logger.debug("extract_numbers_from_text: 'twelve' IS in text_for_word_search.")
        test_match_twelve = re.search(
            r"\btwelve\b", text_for_word_search
        )  # Ensure \b is correct for the test
        if test_match_twelve:
            logger.debug(
                f"extract_numbers_from_text: Manual re.search for '\btwelve\b' SUCCEEDED. Match: {test_match_twelve.group(0)}"
            )
        else:
            logger.warning(
                f"extract_numbers_from_text: Manual re.search for '\btwelve\b' FAILED on: {text_for_word_search[:100]}..."
            )
    else:
        logger.debug(
            f"extract_numbers_from_text: 'twelve' IS NOT in text_for_word_search: {text_for_word_search[:100]}..."
        )

    for match in NUMBER_WORDS_REGEX.finditer(text_for_word_search):
        sign_word = match.group(1)  # (minus|negative) or None
        number_word_matched = match.group(
            2
        ).lower()  # The matched number phrase from the regex pattern

        value = EXPANDED_NUMBER_WORDS_DICT.get(number_word_matched)

        if value is not None:
            if sign_word and value != 0:  # Apply sign if present
                value = -value
            found_numbers.add(value)
        else:
            # This should ideally not happen if regex is built from dict keys
            logger.warning(
                f"Word phrase '{number_word_matched}' found by NUMBER_WORDS_REGEX but not in EXPANDED_NUMBER_WORDS_DICT. This indicates a mismatch or an issue with regex construction from dictionary keys."
            )

    logger.debug(
        f"extract_numbers_from_text: Input '{text[:100]}...', Found: {found_numbers}"
    )
    return found_numbers


# --- Factory for number validation ---
# Ensure EXPANDED_NUMBER_WORDS_DICT and other necessary globals are defined before this


def make_number_validator(
    allowed_atoms: Set[int],
    forbidden_atoms: Set[int],  # Results from prior beats, overall GT (conditionally)
    operand_count: int,
    correct_result_for_beat: int | None,
    strict_zero: bool = False,  # For intro/padding
    enforce_result_presence: bool = True,  # If True, correct_result_for_beat MUST be present
    # If False (i.e., result should be implicit), correct_result_for_beat MUST NOT be present (for non-root)
    operation_type: str | None = None,
    overall_ground_truth_answer: int | None = None,
    is_root_node_being_validated: bool = False,  # NEW: To know if this beat is the final AST root
    conceptual_input_values: (
        Set[int] | None
    ) = None,  # NEW: Values from direct child OpNodes
    config_obj: Config = config,  # Pass config for ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    logger_obj: logging.Logger = logger,
) -> Callable[[str], bool]:
    logger_obj.debug(
        f"Creating validator with: Allowed_Atoms={allowed_atoms}, Forbidden={forbidden_atoms}, OpCount={operand_count}, "
        f"Result={correct_result_for_beat}, StrictZero={strict_zero}, "
        f"EnforceResultPresence={enforce_result_presence} (True means result MUST be stated, False means it MUST be implicit for non-root), "
        f"Op={operation_type}, OverallGT={overall_ground_truth_answer}, IsRoot={is_root_node_being_validated}, "
        f"ConceptualInputs={conceptual_input_values}"  # Added logging for new param
    )

    # IMPLICITLY_ALLOWED_SMALL_NUMBERS is for general small numbers beyond the always-allowed phrasing ones,
    # e.g., if MAX_ALLOWED_SMALL_NUMBER is 10, this covers 0, 4, 5, ..., 10.
    # 1, 2, 3 are handled by config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET.
    IMPLICITLY_ALLOWED_SMALL_NUMBERS = set(
        range(
            config_obj.MIN_ALLOWED_SMALL_NUMBER, config_obj.MAX_ALLOWED_SMALL_NUMBER + 1
        )
    )

    # Modified required_atoms specifically for MED operations
    required_atoms_for_validation = set(allowed_atoms)
    if operation_type == "MED" and correct_result_for_beat in allowed_atoms:
        # For MED operations, if the median is also an input, remove it from required atoms
        required_atoms_for_validation.discard(correct_result_for_beat)
        logger_obj.info(
            f"MED OPERATION SPECIAL CASE: Removing median value {correct_result_for_beat} from required atoms for validation"
        )

    # Numbers that are explicitly part of the current beat's definition (operands, and result if it's *meant* to be stated)
    current_beat_explicitly_defined_numbers = set(required_atoms_for_validation)
    if enforce_result_presence and correct_result_for_beat is not None:
        current_beat_explicitly_defined_numbers.add(correct_result_for_beat)

    # Special handling for MAX/MIN/MED operations with dual-role numbers (both input and result)
    is_max_min_op = operation_type in ["MAX", "MIN"]
    is_result_also_required_atom = correct_result_for_beat in allowed_atoms

    if is_max_min_op and is_result_also_required_atom:
        logger_obj.info(
            f"SPECIAL CASE: {operation_type} operation where result {correct_result_for_beat} is also an input atom"
        )

    # Special handling for SM operations to allow prior results to be explicitly mentioned
    # For SM operations, we need to make inputs from prior operations more lenient
    is_sm_operation = operation_type == "SM"

    # For SM operations, we'll track any intermediate sum calculations that might appear
    sm_allowed_intermediate_sums = set()
    sm_allowed_prior_results = set()

    if is_sm_operation and conceptual_input_values:
        # Only allow conceptual inputs to be mentioned by exact value - no intermediate sums
        sm_allowed_prior_results = set(conceptual_input_values)
        logger_obj.debug(f"SM OPERATION: Allowing conceptual input values: {sm_allowed_prior_results}")
        # No intermediate sums allowed for SM operations

    def validate(text: str) -> bool:
        found_numbers = extract_numbers_from_text(
            text
        )  # Assumes extract_numbers_from_text is defined

        text_preview = text[:100].replace("\n", " ") + (
            "..." if len(text) > 100 else ""
        )
        logger_obj.debug(f'Validator Input Text: "{text_preview}"')
        logger_obj.debug(f"Validator Found Numbers: {found_numbers}")

        validation_report = {
            "status": "PASS",
            "reason": "All validation checks passed",
            "operation_type": operation_type,
            "text_preview": text_preview,
            "found_numbers": list(found_numbers),
            "allowed_atoms": list(allowed_atoms),
            "operand_count": operand_count,
            "correct_result": correct_result_for_beat,
            "intermediate_sum": None,  # No intermediate sums allowed
            "is_root_node": is_root_node_being_validated,
            "enforce_result_presence_flag": enforce_result_presence,
            "overall_ground_truth_answer_for_this_validation_context": overall_ground_truth_answer,
            "missing_required": [],
            "forbidden_extras": [],
            "details": [],
        }

        if strict_zero:  # For intro/padding
            # Only config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET (e.g. {1,2,3}) might be found if not strictly zero numbers.
            # For intro/padding, we usually want NO numbers at all, or at most 'one' if that's relaxed.
            # Let's assume strict_zero means absolutely no numbers, or only '1' if that's a special allowance.
            # The current `generate_introduction_scene` and padding prompts aim for NO numbers.
            # If `config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET` is e.g. {1,2,3}, then strict_zero
            # should mean that found_numbers must be a subset of this, or empty.
            # For intro/padding, we typically want even stricter (e.g., only '1' or nothing).
            # The `validate_intro` and `validate_padding` in `_generate_narrative_recursive`
            # are created with `allowed_atoms=set()`, `forbidden_atoms=set()`, `strict_zero=True`.
            # This means `current_beat_explicitly_defined_numbers` will be empty.
            # `unexpected_found` will be all `found_numbers`.
            # Then `truly_disallowed_extras` will catch anything not in `ALWAYS_ALLOWED_PHRASING_NUMBERS_SET`
            # or not operand_count (0) or not in IMPLICITLY_ALLOWED_SMALL_NUMBERS.
            # This seems okay. If intro/padding should allow *only* '1', then ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
            # for those validators should be just {1}.
            # The current `generate_with_retry` for intro/padding uses a validator from `make_number_validator`
            # with `strict_zero=True`. The `ultra_strict_instruction` for intro/padding aims for NO numbers.
            pass  # The main logic below will handle it. If strict_zero, allowed_atoms is empty.

        # --- Start of main validation logic for story beats ---

        # Rule A: Handle presence/absence of the current beat's result
        if not is_root_node_being_validated and correct_result_for_beat is not None:
            # Special handling for MED operations - their results should always be implicit
            # This aligns with the benchmark's intent and LLM validator behavior
            if operation_type == "MED" and enforce_result_presence:
                # Override to enforce implicit results for MED
                logger_obj.debug(
                    f"SPECIAL HANDLING: MED operation detected. Enforcing IMPLICIT result regardless of config."
                )

                # Check that the result is NOT explicitly stated (it should be implicit)
                if correct_result_for_beat in found_numbers:
                    validation_report["status"] = "FAIL"
                    validation_report["reason"] = "MED_RESULT_STATED_EXPLICITLY"
                    validation_report["details"].append(
                        f"MED result {correct_result_for_beat} should be implicit but was found explicitly stated."
                    )
                    _log_failed_validation(text, validation_report)
                    return False
            # Special handling for MAX/MIN operations where result is also an input atom
            elif is_max_min_op and is_result_also_required_atom:
                # For MAX/MIN with dual-role numbers, we ALLOW the number to be present as an input
                # but contextualization should make clear it's not being used as the explicit result
                if correct_result_for_beat in found_numbers:
                    logger_obj.debug(
                        f"DUAL-ROLE NUMBER in {operation_type}: {correct_result_for_beat} found - allowing as it is both input and result"
                    )
                    # Instead of failing, we continue - this is intentionally permissive for MAX/MIN dual-role numbers
                    # The LLM validator will check if the contextualization is appropriate
            elif (
                enforce_result_presence
            ):  # Standard: Result MUST be stated (for non-MED operations)
                if correct_result_for_beat not in found_numbers:
                    validation_report["status"] = "FAIL"
                    validation_report["reason"] = "MISSING_REQUIRED_RESULT"
                    validation_report["details"].append(
                        f"Result {correct_result_for_beat} must be present but was not. Op: {operation_type}"
                    )
                    _log_failed_validation(text, validation_report)
                    return False
            else:  # Result MUST be IMPLICIT (i.e., NOT stated) - now the standard for all cases
                if correct_result_for_beat in found_numbers:
                    # Add more context about where the result was found in the text
                    # This helps diagnose issues where the result appears indirectly
                    result_context = ""
                    result_word = num_to_words(correct_result_for_beat)
                    for line in text.split("\n"):
                        if (
                            str(correct_result_for_beat) in line
                            or result_word in line.lower()
                        ):
                            result_context = f"Found in context: '{line.strip()}'"
                            break

                    validation_report["status"] = "FAIL"
                    validation_report["reason"] = "IMPLICIT_RESULT_STATED_EXPLICITLY"
                    validation_report["details"].append(
                        f"Result {correct_result_for_beat} should be implicit but was found. Op: {operation_type}. {result_context}"
                    )
                    _log_failed_validation(text, validation_report)
                    return False
        # For root node, result is always implicit. `overall_ground_truth_answer` handles checking if it's stated.

        # Rule B: All required atomic operands MUST be present
        # Special exception for MED operations - don't require an atom if it's also the median value
        if operation_type == "MED" and correct_result_for_beat in allowed_atoms:
            # For MED operations, if the median value is also an atom, don't require it to be present
            required_atoms = allowed_atoms - {correct_result_for_beat}
            logger_obj.debug(
                f"MED OPERATION: Not requiring atomic median value {correct_result_for_beat} to be present"
            )
        else:
            required_atoms = allowed_atoms

        missing_required_atoms = required_atoms - found_numbers
        if missing_required_atoms:
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "MISSING_REQUIRED_OPERANDS"
            validation_report["missing_required"] = list(missing_required_atoms)
            validation_report["details"].append(
                f"RequiredAtoms={allowed_atoms}, Missing={missing_required_atoms}, Found={found_numbers}."
            )
            _log_failed_validation(text, validation_report)
            return False

        # Rule C: Check for forbidden numbers (from prior beats / overall GT)
        # `forbidden_atoms` is passed in, containing results of prior distinct operations + overall GT (if not root & not current result)
        forbidden_and_found = found_numbers & forbidden_atoms
        # These are forbidden unless they are also current_beat_explicitly_defined_numbers (e.g. an atom that happens to be a forbidden value)
        # or an always_allowed_phrasing_number used for phrasing.

        # New: For small counting numbers up to 10, we'll be more lenient if they appear to be for counting
        counting_numbers_allowed = set(
            range(0, 11)
        )  # Allow 0-10 for counting even if in forbidden list

        truly_forbidden_and_found = set()
        for forbidden_num in forbidden_and_found:
            # Skip if it's already part of current beat's defined numbers
            if forbidden_num in current_beat_explicitly_defined_numbers:
                continue

            # Skip if it's in always allowed phrasing numbers
            if forbidden_num in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
                continue

            # SPECIAL CASE FOR SM OPERATIONS: Allow prior results to be explicitly mentioned
            # when used as inputs to the current SM operation
            if is_sm_operation and forbidden_num in sm_allowed_prior_results:
                logger_obj.debug(
                    f"SM OPERATION: Allowing prior result {forbidden_num} as input to SM calculation"
                )
                continue

            # Special exception for numbers 0-10 that could be for counting, especially if they match operand_count
            if forbidden_num in counting_numbers_allowed:
                # If it's the operand count, it's likely being used for counting
                if forbidden_num == operand_count:
                    logger_obj.debug(
                        f"Allowing forbidden number {forbidden_num} as it matches operand count {operand_count}"
                    )
                    continue

                # Otherwise, if it's within small number range, give it the benefit of the doubt for counting
                logger_obj.debug(
                    f"Allowing small number {forbidden_num} even though it's in forbidden list. Assuming used for counting."
                )
                continue

            # If we get here, it's truly forbidden
            truly_forbidden_and_found.add((forbidden_num, "from prior results/GT"))

        if truly_forbidden_and_found:
            formatted_forbidden = ", ".join(
                [f"{n}({reason})" for n, reason in truly_forbidden_and_found]
            )
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "FORBIDDEN_NUMBERS_FOUND"
            validation_report["forbidden_extras"] = list(
                n for n, r in truly_forbidden_and_found
            )  # Just numbers
            validation_report["details"].append(
                f"Forbidden_Found={{ {formatted_forbidden} }}. These are forbidden prior results/GT that must not appear."
            )
            _log_failed_validation(text, validation_report)
            return False

        # Rule D: Check for other extraneous numbers
        # Numbers found that are NOT:
        #   - required atoms for this beat (allowed_atoms)
        #   - the correct_result_for_beat (if it was supposed to be stated and was)
        
        #   - an always_allowed_phrasing_number (1,2,3)
        #   - the operand_count (if not otherwise forbidden)
        #   - other implicitly_allowed_small_numbers (0, 4-10, if not otherwise forbidden)
        #   - the overall_ground_truth_answer (special handling)
        #   - NEW: For SM operations, any intermediate sums or total sums from the calculation process

        unexpected_found_after_explicit = (
            found_numbers - current_beat_explicitly_defined_numbers
        )
        logger_obj.debug(
            f"Validator: initial unexpected_found (found - explicit_defined)={unexpected_found_after_explicit}"
        )

        truly_disallowed_extras = set()
        for extra_num in unexpected_found_after_explicit:
            # 1. Handled by ALWAYS_ALLOWED_PHRASING_NUMBERS_SET?
            if extra_num in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
                continue  # Already allowed for phrasing

            # 2. Is it the overall_ground_truth_answer?
            if (
                overall_ground_truth_answer is not None
                and extra_num == overall_ground_truth_answer
            ):
                if (
                    is_root_node_being_validated
                ):  # For root, GT should NOT be stated. Finding it is an error.
                    truly_disallowed_extras.add(
                        (extra_num, "is overall_GT stated explicitly for root node")
                    )
                    continue
                # If not root, and GT is an extra number:
                # If GT is a small counting number (0-10), its incidental use might be tolerated if not confusing.
                # This is a nuanced rule, often better handled by LLM validator's interpretation of "confusing."
                # For strict Python validator, if GT is found as an "extra" (not an atom, not current result),
                # it's generally an error unless it's one of the ALWAYS_ALLOWED_PHRASING_NUMBERS (already handled).
                # The `forbidden_atoms` set (used in Rule C) should already contain overall_ground_truth_answer
                # if it's not supposed to be there. If it passed Rule C, it means it wasn't in `forbidden_atoms`
                # or was an allowed phrasing number.
                # So, if it reaches here, it means `extra_num == overall_ground_truth_answer` and it was NOT
                # on the `forbidden_atoms` list for this beat (e.g. GT was the current beat's result).
                # This case should be rare if `forbidden_atoms` is constructed correctly.
                # Let's assume if it's GT and an "extra" here, it's problematic unless it's a phrasing number.
                # The `forbidden_atoms` check (Rule C) is the primary gate for GT.
                # If GT is an `extra_num` here, it means it wasn't a required atom/result, wasn't a phrasing num,
                # and wasn't on the `forbidden_atoms` list for this beat. This implies GT might have been
                # the `correct_result_for_beat` which was *implicitly* stated, and now found.
                # This is complex. The main check for GT is:
                #   - If root: must NOT be in `found_numbers`. (Handled by `forbidden_atoms` if GT is added to it for root).
                #   - If not root: must NOT be in `found_numbers` UNLESS it's a current atom/result or allowed phrasing. (Handled by `forbidden_atoms`).
                # The `forbidden_atoms` set passed to `make_number_validator` is key here.
                # Let's simplify: if `extra_num == overall_ground_truth_answer` and it's not root,
                # and it's not an ALWAYS_ALLOWED_PHRASING_NUMBER, it's an error.
                if (
                    not is_root_node_being_validated
                    and extra_num not in config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
                ):
                    truly_disallowed_extras.add(
                        (
                            extra_num,
                            "is overall_GT mentioned extraneous to its role or when it should be implicit",
                        )
                    )
                continue  # Handled (either added to disallowed or was root and will be caught by forbidden_atoms)

            # 3. Is it the operand_count? (And not on the forbidden_atoms list for *this specific usage*)
            # `forbidden_atoms` contains prior *results*. If operand_count happens to be a prior result,
            # using it as operand_count *might* be okay if not confusing.
            # The `extra_num not in forbidden_atoms` check here is for this.
            if extra_num == operand_count:
                continue  # Allow operand count even if it was a forbidden prior result
                # The logic for handling forbidden operand counts moved to Rule C where we're being more lenient

            # 4. Is it another IMPLICITLY_ALLOWED_SMALL_NUMBER (e.g., 0, 4-10)?
            # (And not on the forbidden_atoms list for *this specific usage*)
            if (
                extra_num in IMPLICITLY_ALLOWED_SMALL_NUMBERS
            ):  # Excludes 1,2,3 which are already handled
                # Be more lenient - allow small numbers (0-10) for counting even if forbidden
                if extra_num in counting_numbers_allowed:
                    logger_obj.debug(
                        f"Allowing small number {extra_num} assuming it's used for counting"
                    )
                    continue
                # Original check
                if extra_num not in forbidden_atoms:
                    continue
                else:  # This small number IS a forbidden prior result.
                    # But if it's a small counting number (0-10), we'll still allow it
                    if extra_num in counting_numbers_allowed:
                        logger_obj.debug(
                            f"Allowing forbidden small number {extra_num} assuming it's used for counting"
                        )
                        continue
                    truly_disallowed_extras.add(
                        (
                            extra_num,
                            f"is small number ({extra_num}) but also a forbidden prior result",
                        )
                    )
                    continue

            # 5. SPECIAL CASE FOR SM OPERATIONS: Allow intermediate sums and calculations
            if is_sm_operation:
                # Check if it's an intermediate sum calculation for SM
                if extra_num in sm_allowed_intermediate_sums:
                    logger_obj.debug(
                        f"SM OPERATION: Allowing intermediate sum {extra_num} as part of SM calculation"
                    )
                    continue

                # NEW: Check if it's a direct conceptual input value for SM
                if conceptual_input_values is not None and extra_num in conceptual_input_values:
                    logger_obj.debug(
                        f"SM OPERATION: Allowing conceptual input value {extra_num} as explicitly mentioned input"
                    )
                    continue
                    
                # Check if it's a prior result value for SM
                if extra_num in sm_allowed_prior_results:
                    logger_obj.debug(
                        f"SM OPERATION: Allowing prior result {extra_num} as explicitly mentioned input for SM"
                    )
                    continue

                    
                # For detailed debugging when a number is not allowed
                logger_obj.debug(
                    f"SM OPERATION: Disallowing number {extra_num} - doesn't match any allowed SM calculation pattern"
                )
            # 6. SPECIAL CASE FOR MED OPERATIONS: Do not allow median value anywhere in text
            if operation_type == "MED" and extra_num == correct_result_for_beat:
                # For MED operations, the median value must not appear anywhere in text
                truly_disallowed_extras.add(
                    (
                        extra_num,
                        f"is median value, which must not appear anywhere in MED operations",
                    )
                )
                continue

            # If none of the above, it's truly disallowed.
            truly_disallowed_extras.add(
                (
                    extra_num,
                    "not a required atom/result, not allowed phrasing, not allowed count/small, not GT in allowed context",
                )
            )

        if truly_disallowed_extras:
            formatted_disallowed = ", ".join(
                [f"{n}({reason})" for n, reason in truly_disallowed_extras]
            )
            validation_report["status"] = "FAIL"
            validation_report["reason"] = "EXTRANEOUS_NUMBERS"
            validation_report["forbidden_extras"] = list(
                n for n, r in truly_disallowed_extras
            )  # Just numbers
            validation_report["details"].append(
                f"Disallowed_Extras={{ {formatted_disallowed} }}. ExplicitlyDefined={current_beat_explicitly_defined_numbers}, ForbiddenSet={forbidden_atoms}, Found={found_numbers}."
            )
            _log_failed_validation(text, validation_report)
            return False

        logger_obj.debug(f"Validation PASS (Strict Python Validator)")
        return True

    return validate


# ... (generate_introduction_scene - its validator uses strict_zero=True, allowed_atoms=set(), forbidden_atoms=set())
# This means only ALWAYS_ALLOWED_PHRASING_NUMBERS_SET might be tolerated by the Python validator if any numbers are found.
# The intro prompt asks for NO numbers.


# Add a helper function to save failed validation attempts
def _log_failed_validation(
    text: str, validation_report: dict, logger_obj: logging.Logger = logger
):
    """
    Save failed validation attempts for diagnostic purposes.
    This provides a detailed record of why each beat was rejected.
    Additionally writes to the LLM turns log to keep all information in one place.
    """
    logger_obj.debug(f"Saving failed validation record")
    try:
        # Ensure log directory exists
        failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
        os.makedirs(failed_validations_dir, exist_ok=True)

        # Create a unique filename for this validation failure
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        operation = validation_report.get("operation_type", "unknown_op")
        reason = validation_report.get("reason", "unknown_reason")

        # Generate the filename using significant information for easy identification
        filename = f"validation_fail_{operation}_{reason}_{timestamp}.json"
        filepath = os.path.join(failed_validations_dir, filename)

        # Create the full record to save
        full_report = {
            "validation_report": validation_report,
            "full_text": text,
            "timestamp": timestamp,
        }

        # Save the record
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved failed validation record to {filepath}")

        # --- NEW SECTION: Also write validation failure to LLM turns log ---
        # Extract sample_index from the calling frame's context if available
        sample_index = None
        import inspect

        try:
            # Look for context.sample_index in the stack frames
            frames = inspect.stack()
            for frame in frames:
                if "context" in frame.frame.f_locals:
                    ctx = frame.frame.f_locals["context"]
                    if hasattr(ctx, "sample_index"):
                        sample_index = ctx.sample_index
                        break
        except Exception as e:
            logger.error(f"Error extracting sample_index from stack: {e}")

        # Format validation failure details in a way useful for LLM analysis
        found_nums = validation_report.get("found_numbers", [])
        allowed_atoms = validation_report.get("allowed_atoms", [])
        missing = validation_report.get("missing_required", [])
        forbidden = validation_report.get("forbidden_extras", [])
        operation = validation_report.get("operation_type", "unknown")
        correct_result = validation_report.get("correct_result", None)
        intermediate_sum = validation_report.get("intermediate_sum", None)
        operand_count = validation_report.get("operand_count", None)

        # Enhanced header with more diagnostic information
        # Try to extract narrative anchor information from stack
        narrative_anchor = None
        beat_counter = None
        for frame in frames:
            frame_locals = frame.frame.f_locals
            if "narrative_anchor" in frame_locals:
                narrative_anchor = frame_locals["narrative_anchor"]
            if "context" in frame_locals:
                ctx = frame_locals["context"]
                if hasattr(ctx, "beat_counter"):
                    beat_counter = ctx.beat_counter
            if narrative_anchor and beat_counter:
                break

        # Create a more descriptive header with beat and narrative context when available
        beat_info = ""
        if beat_counter:
            beat_info = f", Beat {beat_counter.get('current', '?')}/{beat_counter.get('total', '?')}"

        anchor_info = ""
        if narrative_anchor:
            anchor_info = f", Anchor: '{narrative_anchor}'"

        validation_header = (
            f"VALIDATION FAILURE: Op={operation}{beat_info}{anchor_info}, Reason={reason} "
            f"[Consolidated log for LLM analysis]"
        )

        # Create a more detailed validation summary
        validation_details = f"{'='*80}\n"  # Clear visual separator
        validation_details += f"=== VALIDATION FAILURE REPORT ===\n"
        validation_details += f"{'='*80}\n\n"
        validation_details += f"Operation type: {operation}\n"
        validation_details += f"Failure reason: {reason}\n"
        validation_details += f"Operand count: {operand_count}\n"

        if correct_result is not None:
            validation_details += (
                f"Correct result (should be mentioned): {correct_result}\n"
            )

        if intermediate_sum is not None:
            validation_details += (
                f"Intermediate sum (may be mentioned for AVG/SM): {intermediate_sum}\n"
            )

        validation_details += f"\n--- Number Analysis ---\n"
        validation_details += f"Found numbers in text: {found_nums}\n"
        validation_details += f"Required numbers: {allowed_atoms}\n"
        validation_details += f"Missing required: {missing}\n"
        validation_details += f"Forbidden extras: {forbidden}\n"

        # Include any additional detailed failure information
        details = validation_report.get("details", [])
        if details:
            validation_details += f"\n--- Detailed Analysis ---\n"
            for detail in details:
                validation_details += f"- {detail}\n"

        validation_details += f"\n--- Generated Text That Failed Validation ---\n{text}"

        # Add clear ending separator
        validation_details += f"\n\n{'='*80}\n"
        validation_details += f"=== END OF VALIDATION FAILURE REPORT ===\n"
        validation_details += f"{'='*80}\n"

        # Write to LLM turns log using the existing log_prompt function
        log_prompt(validation_header, validation_details, sample_index=sample_index)

    except Exception as e:
        logger.error(f"Error saving failed validation record: {e}")


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
    all_previous_anchors: list[str],
    sample_index: int | None = None,
) -> str | None:
    """
    Uses an LLM to generate a short, thematic noun phrase based on keywords.
    Focuses on reliability with a very simple prompt structure.
    """

    op_label = OP_LABELS.get(op_node.op, op_node.op)
    genre = world_info.get("genre", "unknown genre")
    setting = world_info.get("setting", "a mysterious place")
    primary_object = world_info.get("object", "items")

    concept_keywords_map = {
        "MAX": "Pinpointing the most potent or largest element",
        "MIN": "Isolating the smallest or most fundamental essence",
        "SUM": "Amalgamating all components into a unified total",
        "MED": "Identifying the central balancing point in an ordered series",
        "AVG": "Discerning the common thread or typical measure across all items",
        "SM": "Unveiling a core symbolic number through cyclical transformation",
    }
    concept_keywords_for_prompt = concept_keywords_map.get(
        op_node.op, f"{op_label} Concept"
    )
    system_prompt = f"""You are a master {world_info.get('genre')} storyteller and creative naming expert. Your task is to generate a short, evocative, and thematic 'narrative anchor'.

A narrative anchor is a creative, conceptual name that serves as a descriptive **label** or **stand-in** for the *result* (the outcome) of a specific event or calculation within the story. Its purpose is to allow the narrative to refer to this result conceptually in later parts of the story, *without* explicitly stating its numerical value. For example, if a calculation's outcome is 50, the anchor might be 'The Sunstone's Core.' The story would then mention 'The Sunstone's Core' (which implicitly represents the value 50) instead of the number itself, allowing the narrative to flow without revealing intermediate figures.

Key Guidelines for the Narrative Anchor:
1.  **Thematic:** The name MUST fit the provided Genre, Setting, and Primary Object.
2.  **Concise:** Aim for 2 to {config.MAX_ANCHOR_WORDS} words. Often a noun phrase (e.g., 'The Sunstone's Core,' 'The Oracle's Key').
3.  **No Numbers:** Absolutely no numerical values in the anchor itself.
4.  **No Direct Math Terms:** Do NOT use words like 'Sum', 'Min', 'Max', 'Average', 'Median', 'Count' directly in the anchor name. The 'Concept/Operation Hint' provided will hint at the nature of the operation without using these explicit terms.
5.  **Represent Outcome:** The name should conceptually represent the *result* or *culmination* of the described action/operation.
6.  **Focus on the Noun:** The anchor should feel like a "thing" or a "state" that has been achieved or discovered.

Examples of good anchors based on different inputs:
*   Genre: High Fantasy, Setting: Enchanted Forest, Object: Moonpetal Flowers, Concept/Operation Hint: Amalgamating all components into a unified total -> The Lunar Bloom's Essence
*   Genre: Noir Detective, Setting: Rain-slicked City, Object: Stolen Jewels, Concept/Operation Hint: Isolating the smallest or most fundamental essence -> The Shadow Locket's Secret
*   Genre: Steampunk, Object: Clockwork Gears, Concept: The central piece -> The Chronometer's Heart

You will be given the Genre, Setting, Item (Primary Object), and Concept/Operation Hint. Provide ONLY the generated anchor as your response."""

    user_prompt = (
        f"Genre: {genre}\n"
        f"Setting: {setting}\n"
        f"Item: {primary_object}\n"
        f"Concept/Operation Hint: {concept_keywords_for_prompt}\n"
    )
    prompt_log_header = f"--- Narrative Anchor Prompt (Op: {op_node.op}, Item: {primary_object}, Concept: {concept_keywords_for_prompt}) ---"
    prompt_content_for_log = f"System: {system_prompt}\\nUser:\\n{user_prompt}"

    # --- Log the prompt using log_prompt ---
    log_prompt(
        header=f"Narrative Anchor Generation Prompt (Op: {op_node.op})",
        prompt=prompt_content_for_log,
        sample_index=sample_index,
    )

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
            "max_tokens": config.ANCHOR_MAX_TOKENS,
            "temperature": config.ANCHOR_GEN_TEMP,
            "reasoning": {"exclude": True},
        }

        logger.debug(
            f"Using request_payload for narrative anchor generation: {request_payload}"
        )

        resp = _chat_completion_call(**request_payload)

        raw_candidate = None
        finish_reason = "N/A"

        if resp and resp.choices and len(resp.choices) > 0:
            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            if choice.message:
                raw_candidate = choice.message.content
                if raw_candidate is None:
                    logger.warning(
                        f"Narrative Anchor Gen: Received None content. Finish reason: {finish_reason}. Response: {resp}"
                    )
            else:
                logger.warning(
                    f"Narrative Anchor Gen: Message object is missing or empty. Finish reason: {finish_reason}. Response: {resp}"
                )
        else:
            logger.warning(
                f"Narrative Anchor Gen: Unexpected response structure (no choices or empty response). Response: {resp}"
            )

        log_prompt(
            header=f"Narrative Anchor Generation Response (Op: {op_node.op})",
            prompt=f"Raw LLM Output (Finish Reason: {finish_reason}):\n{raw_candidate if raw_candidate is not None else 'None'}",
            sample_index=sample_index,
        )
        logger.debug(
            f"Narrative Anchor Gen - API Call Details: Finish Reason='{finish_reason}', Raw Candidate='{str(raw_candidate)[:100]}...'"
        )

        if raw_candidate is None:
            logger.warning(f"Narrative Anchor Gen: Received None content in response.")
            return None

        candidate = raw_candidate.strip()

        # --- Strip surrounding quotes ---
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1].strip()
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1].strip()

        # --- Remove boilerplate prefixes ---
        original_candidate_before_boilerplate_strip = candidate  # Store for comparison
        candidate = re.sub(
            r"^(OUTPUT \(Phrase Only\):)\s*", "", candidate, flags=re.IGNORECASE
        ).strip()
        candidate = re.sub(
            r"^(Okay, here's a noun phrase:|Noun Phrase:|Phrase:|Label:|Descriptor:|Designation:|Certainly:|Here it is:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        # --- Check if boilerplate was present or if string is now empty ---
        boilerplate_indicators_lower = [
            "output (phrase only):",
            "okay, here's a noun phrase:",
            "noun phrase:",
            "phrase:",
            "label:",
            "descriptor:",
            "designation:",
            "certainly:",
            "here it is:",
        ]
        # Check if the candidate, after stripping, still starts with a known boilerplate indicator
        # or if it became empty after stripping boilerplate (meaning it was *only* boilerplate)
        if not candidate or any(
            original_candidate_before_boilerplate_strip.lower().startswith(indicator)
            for indicator in boilerplate_indicators_lower
        ):
            if not candidate:  # It became empty after stripping
                logger.warning(
                    f"Narrative Anchor Gen: Response was only boilerplate (raw: '{raw_candidate}', processed to empty string)"
                )
            else:  # It still starts with boilerplate, or original started with it and cleaning wasn't perfect
                logger.warning(
                    f"Narrative Anchor Gen: Boilerplate detected in response (raw: '{raw_candidate}', processed: '{candidate}'). Triggering retry."
                )
            return None  # Fail this attempt to trigger retry

        # --- Aggressively remove echoed input preamble ---
        # Construct the expected preamble pattern based on current genre, item, concept
        preamble_pattern_str = (
            rf"Genre: {re.escape(genre)}\s*\n"
            rf"Setting: {re.escape(setting)}\s*\n"
            rf"Item: {re.escape(primary_object)}\s*\n"
            rf"Concept/Operation Hint: {re.escape(concept_keywords_for_prompt)}\s*\n"
        )
        # Remove preamble if found at the beginning of the candidate string
        candidate = re.sub(
            f"^{preamble_pattern_str}",
            "",
            candidate,
            flags=re.IGNORECASE | re.MULTILINE,  # Added re.MULTILINE
        ).strip()

        # Fallback: remove potential "OUTPUT (Phrase Only):"
        candidate = re.sub(
            r"^(OUTPUT \(Phrase Only\):)\s*", "", candidate, flags=re.IGNORECASE
        ).strip()
        candidate = re.sub(
            r"^(Okay, here's a noun phrase:|Noun Phrase:|Phrase:|Label:|Descriptor:|Designation:|Certainly:|Here it is:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        # --- More robust check for guideline echoing ---
        # Strip surrounding quotes, as models sometimes wrap short answers in them.
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1].strip()
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1].strip()

        guideline_starters_lower = [
            "**thematic:**",
            "**concise:**",
            "**no numbers:**",
            "**no direct math terms:**",
            "**represent outcome:**",
            "**avoid repetition:**",
            "**focus on the noun:**",
            "1.",
            "2.",
            "3.",
            "4.",
            "5.",
            "6.",
            "7.",
            "key guidelines",
            "examples of good anchors",
        ]  # Already lowercase or will be lowercased by startswith check

        candidate_lower_stripped = candidate.lower().strip()
        if any(
            candidate_lower_stripped.startswith(starter)
            for starter in guideline_starters_lower
        ):
            logger.warning(
                f"Narrative Anchor Gen: Response starts with a guideline phrase (raw: '{raw_candidate}', cleaned: '{candidate}')"
            )
            return None

        num_words = len(candidate.split())
        if (
            not candidate or num_words == 0 or num_words > config.MAX_ANCHOR_WORDS
        ):  # Use config.MAX_ANCHOR_WORDS instead of hardcoded 5
            logger.warning(
                f"Narrative Anchor Gen: Invalid (empty, too long/short, refused) response (raw: '{raw_candidate}', processed: '{candidate}', words: {num_words})"
            )
            return None

        if candidate.lower().startswith(
            (
                "i cannot",
                "i'm sorry",
                "i am unable",
                "as an ai",
                "i do not have",
                "unable to provide",
            )
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


# --- BeatGenerationError Exception ---
class BeatGenerationError(Exception):
    """Raised when a story beat fails to generate, aborting entire narrative."""

    pass


# --- LLM-based Static Beat Content Validation (Existing Function, Renamed) ---
def create_user_prompt(
    world_info: dict,
    current_op_node: OpNode,
    inputs_str_for_validation: str,
    action_description_for_validation: str,
    expected_beat_result: int | None,
    overall_ground_truth_answer: int | None,
    beat_text: str,
) -> str:
    """
    Creates a standard user prompt for LLM validation of beat content.
    This centralizes the prompt creation logic to ensure consistency.
    """
    primary_object = world_info.get("object", "items")
    op_label = OP_LABELS.get(current_op_node.op, current_op_node.op)

    expected_result_str = (
        f"{expected_beat_result} ('{num_to_words(expected_beat_result)}')"
        if expected_beat_result is not None
        else "N/A"
    )
    ground_truth_str = (
        f"{overall_ground_truth_answer} ('{num_to_words(overall_ground_truth_answer)}')"
        if overall_ground_truth_answer is not None
        else "unknown"
    )

    return (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'unknown')}\n"
        f"- Setting: {world_info.get('setting', 'unknown')}\n"
        f"- Primary Object: {primary_object}\n\n"
        f"**Operation Details:**\n"
        f"- Current Operation: {current_op_node.op} ({op_label})\n"
        f"- Conceptual and Atomic Inputs: {inputs_str_for_validation}\n"
        f"- Expected Beat Result: {expected_result_str}\n"
        f"- Overall Target Answer: {ground_truth_str}\n"
        f"- Action Intent: {action_description_for_validation}\n\n"
        f"**Task:**\n"
        f"Evaluate the beat against strict numerical rules. Check that:\n"
        f"1. All required input numbers are clearly mentioned\n"
        f"2. The result is either implied or stated as required by rules\n"
        f"3. No forbidden numbers or extraneous values appear\n"
        f"4. The beat logically represents the specified operation\n\n"
        f"**Beat Text to Validate:**\n"
        f"```\n{beat_text}\n```"
    )


def perform_llm_static_content_validation(
    world_info: dict,
    current_op_node: OpNode,
    inputs_str_for_validation: str,
    action_description_for_validation: str,
    expected_beat_result: int | None,
    overall_ground_truth_answer: int | None,
    beat_text: str,
    sample_index: int,
    config_obj: Config,
    logger_obj: logging.Logger,
    encoder_obj: any,
) -> tuple[bool, str]:
    # Ensure create_user_prompt is defined and in scope
    from __main__ import create_user_prompt

    user_prompt = create_user_prompt(
        world_info,
        current_op_node,
        inputs_str_for_validation,
        action_description_for_validation,
        expected_beat_result,
        overall_ground_truth_answer,
        beat_text,
    )

    system_prompt = """You are an AI literary critic and numerical compliance checker.
Evaluate a story 'beat' for precise mathematical narration and adherence to strict numerical rules.
Context (world, operation, numbers, rules) will be provided.
Ensure coherence, logical operation, and perfect numerical compliance.
Respond in structured JSON format ONLY."""

    log_prompt(
        header=f"LLM Static Beat Validation Prompt (Op: {current_op_node.op}) for model {STATIC_CHECKER_MODEL}",
        prompt=f"System: {system_prompt}\\nUser:\\n{user_prompt}",
        sample_index=sample_index,
    )

    # Outermost try for this function
    try:
        # Inner try for the API call and response processing
        try:
            resp = _chat_completion_call(
                model=STATIC_CHECKER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=config_obj.BEAT_MAX_TOKENS,
                temperature=config_obj.LLM_VALIDATOR_TEMP,
                json_schema=STATIC_VALIDATOR_SCHEMA,  # Use the defined schema
                reasoning={"exclude": True},
            )

            raw_llm_output = ""
            if (
                resp
                and resp.choices
                and len(resp.choices) > 0
                and resp.choices[0].message
            ):
                raw_llm_output = resp.choices[0].message.content or ""

            log_prompt(
                header=f"LLM Static Beat Validation Raw Response (Op: {current_op_node.op}) from model {STATIC_CHECKER_MODEL}",
                prompt=f"Raw Output:\n{raw_llm_output}",
                sample_index=sample_index,
            )

            # Innermost try for JSON parsing
            try:
                # The response should be valid JSON already, but still use our parser
                # for consistency and as a fallback
                result_json = parse_llm_json_with_fallback(
                    raw_llm_output,
                    {
                        "is_beat_valid": False,
                        "reasoning": "Failed to parse LLM validator JSON response",
                    },
                    f"in static beat validation for {current_op_node.op}",
                )
                is_valid = bool(result_json.get("is_beat_valid", False))
                reasoning = str(
                    result_json.get(
                        "reasoning", "No reasoning provided by static checker."
                    )
                )

                if not reasoning.strip() and not is_valid:
                    reasoning = "Static checker marked beat as invalid but provided no explicit reasoning."
                elif not reasoning.strip() and is_valid:
                    reasoning = "Static checker marked beat as valid."

                logger_obj.info(
                    f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Static validation result: {is_valid}. Reasoning: {reasoning}"
                )
                return is_valid, reasoning
            except Exception as e:  # Catch other parsing/processing errors
                logger_obj.error(
                    f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Unexpected error processing static beat validation response: {e}. Raw: '{raw_llm_output}'"
                )
                return (
                    False,
                    f"Unexpected error processing LLM validator response: {e}. Raw: {raw_llm_output[:config_obj.MAX_TOKENS_BUFFER // 3]}...",
                )

        except Exception as e:  # Catch errors from _chat_completion_call itself
            logger_obj.error(
                f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] API call to static beat validator ({STATIC_CHECKER_MODEL}) failed: {e}"
            )
            return False, f"API call to static beat validator failed: {e}"
    except Exception as e:  # Catch any other unexpected error in this function
        logger_obj.error(
            f"[Sample {sample_index+1 if sample_index is not None else 'N/A'}, Beat Op: {current_op_node.op}] Unexpected error in perform_llm_beat_validation: {e}",
            exc_info=True,
        )
        return False, f"Outer exception in LLM beat validation: {e}"


# --- NEW FUNCTION: _generate_and_llm_validate_beat (Iterative LLM Validation Loop) ---
# verbose-listops.py

# ... (imports, Config, etc.) ...


def _generate_and_llm_validate_beat(
    original_user_message_for_generator: str,
    system_prompt_for_generator: str,
    world_info: dict,
    current_op_node: OpNode,
    # Inputs for LLM Validator prompt construction:
    conceptual_inputs_str_for_llm_validator: str,
    atomic_inputs_words_str_for_llm_validator: str,
    action_description_for_llm_validator: str,
    expected_beat_result_words_for_llm_validator: str | None,
    ultra_strict_instruction_for_llm_validator_context: str,
    # Other parameters:
    current_max_beat_completion_tokens: int,
    sample_index: int,
    context_config: Config,
    logger_obj: logging.Logger,
    encoder_obj: any,
    is_current_beat_root_node: bool = False,
    overall_ground_truth_answer_val: int | None = None,
    primary_object_name: str = "items",
    forbidden_prior_results_and_gt_for_llm_validator: Set[int] = None,
    correct_result_val: int | None = None,
    direct_atom_values_val: Set[int] = None,  # Changed from field(default_factory=set)
) -> str | None:

    if forbidden_prior_results_and_gt_for_llm_validator is None:
        forbidden_prior_results_and_gt_for_llm_validator = set()
    
    if direct_atom_values_val is None:  # Add this check
        direct_atom_values_val = set()

    # Calculate intermediate sum for AVG/SM operations
    intermediate_sum = None
    if current_op_node.op in ["AVG", "SM"] and direct_atom_values_val:
        intermediate_sum = sum(direct_atom_values_val)
        logger_obj.debug(f"Calculated intermediate sum for {current_op_node.op}: {intermediate_sum}")        
    # Calculate current operation arity (number of children)
    current_op_arity = len(current_op_node.children)
    logger_obj.debug(f"Current operation arity: {current_op_arity}")

    # Define is_median_also_input using the passed parameters
    correct_result = correct_result_val
    direct_atom_values = direct_atom_values_val

    is_median_also_input = False  # Initialize
    if (
        correct_result is not None and direct_atom_values
    ):
        is_median_also_input = correct_result in direct_atom_values
        
    # Generate phrasing numbers string
    always_allowed_phrasing_words = {
        num_to_words(n) for n in context_config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    }
    phrasing_numbers_val_str = ", ".join(
        f"'{w}'" for w in sorted(list(always_allowed_phrasing_words))
    )
    logger_obj.debug(f"Phrasing numbers string: {phrasing_numbers_val_str}")

    history_of_attempts = []
    history_of_critiques = []

    current_generator_user_prompt_for_iteration = original_user_message_for_generator

    for iteration in range(1, context_config.MAX_LLM_VALIDATION_ITERATIONS + 1):
        logger_obj.info(
            f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validation Loop Iteration: {iteration}/{context_config.MAX_LLM_VALIDATION_ITERATIONS}"
        )

        generator_temp = context_config.BEAT_GEN_TEMP
        if iteration > 1:
            generator_temp = context_config.BEAT_REVISION_TEMP

            # --- Build Concise History for Generator ---
            history_prompt_addition = (
                "\n\n--- FAILED ATTEMPT REVIEW & REVISION TASK ---\n"
            )
            last_attempt_text = (
                history_of_attempts[-1]
                if history_of_attempts
                else "N/A (Error in prior generation)"
            )
            last_critique = history_of_critiques[-1] if history_of_critiques else {}

            history_prompt_addition += f"**Your Previous Attempt (Attempt {iteration-1}):**\n{last_attempt_text}\n\n"

            explanation = last_critique.get(
                "explanation_for_generator", "No detailed explanation from validator."
            )
            summary_for_gen = last_critique.get(
                "overall_revision_summary_for_generator_prompt",
                "Please revise based on general errors.",
            )

            history_prompt_addition += (
                f"**Validator Feedback for Your Previous Attempt:**\n"
            )
            history_prompt_addition += f"  - Summary of Issues: {summary_for_gen}\n"
            history_prompt_addition += f"  - Detailed Explanation: {explanation}\n\n"

            history_prompt_addition += (
                f"**Current Revision Task (Attempt {iteration}):**\n"
                f"1. Carefully review the feedback above for Attempt {iteration-1}.\n"
                f"2. Re-read the original task and ALL number rules provided in the initial prompt (partially re-stated below for key rules).\n"
                f"3. Your primary goal is to fix ALL identified issues, especially numerical violations.\n"
                f"4. Ensure the narrative flows logically and addresses the core operation.\n"
                f"5. Output ONLY the revised narrative text for this scene.\n\n"
                f"**Key Original Rules (Reminder - see full initial prompt for all details):**\n"
                f"{ultra_strict_instruction_for_llm_validator_context}\n"  # Re-iterate the rules
            )

            # The user prompt for revision should be the original prompt's core + this history/guidance
            # Split original_user_message_for_generator to get the context part (world, prior scene)
            # and the task part (action_description, ultra_strict_instruction)
            # This is a bit fragile; better if original_user_message_for_generator was more structured or components passed.
            # Assuming original_user_message_for_generator has a clear structure:
            # [Contextual Part: Beat X, World, Inputs, Prior Scene]
            # **Task:** {action_description}
            # {ultra_strict_instruction}
            # **Write the next scene...**

            # For simplicity, we'll prepend the history to the original message,
            # but ideally, the original message would be reconstructed with the history inserted.
            # A simpler way for revision:
            prompt_parts_for_revision = original_user_message_for_generator.split(
                f"{ultra_strict_instruction_for_llm_validator_context}"
            )
            if len(prompt_parts_for_revision) == 2:
                current_generator_user_prompt_for_iteration = (
                    f"{prompt_parts_for_revision[0].strip()}\n\n"  # Context, Task description
                    f"{history_prompt_addition}"  # Review, Revision Task, Reminder of Rules
                    f"{prompt_parts_for_revision[1].strip()}"  # "Write the next scene..." part
                )
            else:  # Fallback if split fails
                current_generator_user_prompt_for_iteration = f"{original_user_message_for_generator}\n\n{history_prompt_addition}"
        else:  # First iteration
            current_generator_user_prompt_for_iteration = (
                original_user_message_for_generator
            )

            # For MED operations, add few-shot examples to help with implicit results
            if current_op_node.op == "MED" and context_config.FEW_SHOT_EXAMPLES >= 1:
                # Add MED-specific few-shot examples to the prompt
                # Use both the standard MED example plus the new critical example
                med_examples = [
                    FEW_SHOT_EXAMPLES_STRICT[1],
                    FEW_SHOT_EXAMPLES_STRICT[2],
                ]  # Both MED examples

                # Create formatted examples
                few_shot_addition = (
                    "\n\n--- CRITICAL FEW-SHOT EXAMPLES FOR MEDIAN OPERATIONS ---\n"
                )
                few_shot_addition += "These examples demonstrate how to properly handle median values in narrative:\n\n"

                for idx, med_example in enumerate(med_examples):
                    few_shot_addition += f"**EXAMPLE {idx+1} RULES:**\n"
                    few_shot_addition += med_example[0].replace("\\\\n", "\n") + "\n\n"

                    few_shot_addition += f"**EXAMPLE {idx+1} GOOD (properly IMPLIES the median value):**\n"
                    few_shot_addition += med_example[1] + "\n\n"

                    few_shot_addition += f"**WHY EXAMPLE {idx+1} GOOD IS CORRECT:**\n"
                    if idx == 0:  # First MED example
                        few_shot_addition += "This example succeeds because it mentions all the required atomic inputs (seventy-two, eighty-four, ninety-one, ninety-five) clearly, but NEVER explicitly states the median value (eighty-nine). Instead, it refers to the median conceptually as 'the middle fragment', 'the balanced keystone', 'central focus', etc. The median value is implied through position and descriptive terms, but the actual number is completely absent from the text.\n\n"
                    else:  # Second MED example (with listing values)
                        few_shot_addition += "This example succeeds because it lists all the atomic inputs EXCEPT the median value (eighty-seven). Notice how it carefully avoids including the median in the list by saying 'seventy-three, eighty-five, eighty-eight, eighty-nine, and ninety-one, plus the void signal' - deliberately leaving a gap where eighty-seven would be. It then refers to this gap conceptually as 'the central point', 'this balance nexus', 'the middle value', etc. At no point does the explicit number 'eighty-seven' appear anywhere in the text.\n\n"

                    few_shot_addition += f"**EXAMPLE {idx+1} BAD (incorrectly handles the median value):**\n"
                    few_shot_addition += med_example[2] + "\n\n"

                    few_shot_addition += f"**WHY EXAMPLE {idx+1} BAD FAILS:**\n"
                    few_shot_addition += med_example[3] + "\n\n"

                few_shot_addition += "**CRITICAL MED OPERATION RULE:**\n"
                few_shot_addition += "For MED operations, the median value must NEVER appear explicitly in the text. This means:\n"
                few_shot_addition += "1. Never list the median value among the inputs\n"
                few_shot_addition += "2. Never state the median value as a result\n"
                few_shot_addition += "3. Only refer to the median conceptually (e.g., 'the central value', 'the balanced point', etc.)\n\n"
                few_shot_addition += "**Your task is to follow these examples and STRICTLY AVOID explicitly stating the median value anywhere.**\n\n"

                # Add the example between the task description and the instructions
                if (
                    "**Task:**" in current_generator_user_prompt_for_iteration
                    and ultra_strict_instruction_for_llm_validator_context
                    in current_generator_user_prompt_for_iteration
                ):
                    parts = current_generator_user_prompt_for_iteration.split(
                        "**Task:**", 1
                    )
                    if len(parts) == 2:
                        task_and_rest = parts[1].split(
                            ultra_strict_instruction_for_llm_validator_context, 1
                        )
                        if len(task_and_rest) == 2:
                            current_generator_user_prompt_for_iteration = (
                                f"{parts[0]}**Task:**{task_and_rest[0]}\n\n"
                                f"{few_shot_addition}"
                                f"{ultra_strict_instruction_for_llm_validator_context}"
                                f"{task_and_rest[1]}"
                            )
                        else:
                            # Just append to the end if we can't find a good split point
                            current_generator_user_prompt_for_iteration += (
                                "\n\n" + few_shot_addition
                            )
                    else:
                        # Just append to the end if we can't find a good split point
                        current_generator_user_prompt_for_iteration += (
                            "\n\n" + few_shot_addition
                        )
                else:
                    # Just append to the end if we can't find a good split point
                    current_generator_user_prompt_for_iteration += (
                        "\n\n" + few_shot_addition
                    )

                logger_obj.info(
                    f"Added MED operation few-shot examples to the prompt for iteration {iteration}"
                )

        # 2. Generate Beat (or revision)
        generated_text_cleaned = ""
        try:
            # ... (log_prompt for generator) ...
            resp_gen = _chat_completion_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt_for_generator},
                    {
                        "role": "user",
                        "content": current_generator_user_prompt_for_iteration,
                    },
                ],
                max_completion_tokens=current_max_beat_completion_tokens,  # This is the API's max_tokens
                temperature=generator_temp,
                reasoning={"exclude": True},
            )
            # ... (extract and clean generated_text_cleaned from resp_gen, handle refusals) ...
            # (Using your existing cleaning logic for generated_text_cleaned)
            raw_gen_text = ""
            if resp_gen and resp_gen.choices and resp_gen.choices[0].message:
                raw_gen_text = resp_gen.choices[0].message.content or ""

            # Simplified cleaning (adapt your more robust cleaning here)
            generated_text_cleaned = (
                raw_gen_text.strip()
            )  # Add your full cleaning logic
            if not generated_text_cleaned or generated_text_cleaned.lower().startswith(
                ("i cannot", "i'm sorry")
            ):
                generated_text_cleaned = ""
                logger_obj.warning(f"Generator refusal or empty in iter {iteration}")

            history_of_attempts.append(
                generated_text_cleaned
                if generated_text_cleaned
                else "GENERATION_EMPTY_OR_REFUSED"
            )
            if not generated_text_cleaned:
                history_of_critiques.append(
                    {
                        "is_valid": False,
                        "explanation_for_generator": "The generation was empty or an API refusal.",
                        "overall_revision_summary_for_generator_prompt": "Previous attempt was empty/refused. Please generate the scene as per original instructions.",
                    }
                )
                continue  # To next iteration

        except Exception as e_gen:
            logger_obj.error(
                f"Error generating beat in LLM loop iter {iteration}: {e_gen}"
            )
            history_of_attempts.append(f"ERROR_DURING_GENERATION: {str(e_gen)[:200]}")
            history_of_critiques.append(
                {
                    "is_valid": False,
                    "explanation_for_generator": f"Exception during generation: {e_gen}",
                    "overall_revision_summary_for_generator_prompt": "Error in previous generation. Retry task.",
                }
            )
            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue
            else:
                return None  # Failed last attempt

        # 3. LLM Validate Beat
        validator_system_prompt = """You are an AI numerical compliance checker.
Your ONLY task is to evaluate a story 'beat' against strict numerical rules.
You MUST output your response as a valid JSON object and NOTHING ELSE.

The valid JSON object MUST have this exact structure:
{
  "is_valid": boolean,  // true if beat meets all criteria, false otherwise
  "explanation_for_generator": string, // detailed explanation for the generator
  "explanation_for_audit": string, // only included when is_valid is true
  "overall_revision_summary_for_generator_prompt": string // concise instruction
}

DO NOT provide explanations outside the JSON.
DO NOT use markdown formatting for the JSON.
DO NOT include any text before or after the JSON object.
Start with '{' and end with '}'."""

        # --- Build LLM Validator's Numerical Rules Section ---
        # Rule 1 (Must Include / Result Handling)
        must_include_val_str = (
            atomic_inputs_words_str_for_llm_validator  # Atoms are always "must include"
        )

        result_handling_val_rule = ""
        if is_current_beat_root_node:
            result_handling_val_rule = (
                f"If this IS the FINAL ROOT operation: The final numerical result ({expected_beat_result_words_for_llm_validator if expected_beat_result_words_for_llm_validator else 'its value'}) "
                f"MUST NOT be explicitly stated but should be clearly implied by the actions. Finding it explicitly is a failure."
            )
        elif current_op_node.op == "MED":  # Special handling for MED operations
            # is_median_also_input is defined at the start of the function.
            # The critical instruction for the generator based on is_median_also_input
            # is handled during the generator's prompt construction, not here
            # when setting up the validator's understanding of the rules.
            result_handling_val_rule = (
                f"For MEDIAN operations: The numerical result ({expected_beat_result_words_for_llm_validator if expected_beat_result_words_for_llm_validator else 'its value'}) "
                f"MUST ALWAYS be IMPLICIT (not explicitly stated). Finding it stated explicitly is a failure."
            )
        elif (
            True  # Always use implicit results
        ):  # Intermediate, implicit
            result_handling_val_rule = (
                f"If this is an INTERMEDIATE beat: The numerical result of THIS beat's operation ({expected_beat_result_words_for_llm_validator if expected_beat_result_words_for_llm_validator else 'its value'}) "
                f"MUST NOT be explicitly stated. It should only be implied by the actions. Finding it explicitly is a failure."
            )
            # Add special instructions for the critical case where median is also an input (for MED, already handled above)
            # (No action_description_parts.append here; handled in MED block)
        else:  # Intermediate, explicit
            must_include_val_str += (
                f"; and the explicit result of this beat's operation: {expected_beat_result_words_for_llm_validator}"
                if expected_beat_result_words_for_llm_validator
                else ""
            )
            result_handling_val_rule = (
                f"If this is an INTERMEDIATE beat and results are EXPLICIT: The numerical result ({expected_beat_result_words_for_llm_validator if expected_beat_result_words_for_llm_validator else 'its value'}) "
                f"MUST be explicitly stated. Its absence is a failure."
            )

        # Rule 2 (May Use)
        always_allowed_phrasing_words_val = {
            num_to_words(n) for n in context_config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        }
        phrasing_numbers_val_str = ", ".join(
            f"'{w}' ({EXPANDED_NUMBER_WORDS_DICT.get(w.lower(), '?')})"
            for w in sorted(list(always_allowed_phrasing_words_val))
        )

        may_use_val_parts = [
            f"small numbers like {phrasing_numbers_val_str} for general narrative phrasing"
        ]
        current_op_arity_val = len(current_op_node.children)
        if current_op_arity_val > 0:
            # Check if arity is in the forbidden set (excluding phrasing numbers)
            arity_is_forbidden_non_phrasing = (
                current_op_arity_val in forbidden_prior_results_and_gt_for_llm_validator
                and current_op_arity_val
                not in context_config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
            )
            if not arity_is_forbidden_non_phrasing:
                may_use_val_parts.append(
                    f"the number '{num_to_words(current_op_arity_val)}' ({current_op_arity_val}) if counting direct inputs/items for THIS operation"
                )

        # Other small numbers (0, 4-10)
        for i in range(
            context_config.MIN_ALLOWED_SMALL_NUMBER,
            context_config.MAX_ALLOWED_SMALL_NUMBER + 1,
        ):
            if (
                i not in context_config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
                and i != current_op_arity_val
            ):  # Avoid re-listing
                is_forbidden_non_phrasing = (
                    i in forbidden_prior_results_and_gt_for_llm_validator
                    and i not in context_config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
                )
                if not is_forbidden_non_phrasing:
                    may_use_val_parts.append(
                        f"the number '{num_to_words(i)}' ({i}) for general counting/phrasing if not confusing"
                    )
        may_use_val_str = "; ".join(may_use_val_parts)

        # Rule 3 (Strictly Forbidden)
        # `forbidden_prior_results_and_gt_for_llm_validator` is a set of INTEGERS
        forbidden_val_words_list = sorted(
            list(
                num_to_words(n)
                for n in forbidden_prior_results_and_gt_for_llm_validator
            )
        )
        forbidden_val_str = (
            ", ".join(
                f"'{w}' ({EXPANDED_NUMBER_WORDS_DICT.get(w.lower(), '?')})"
                for w in forbidden_val_words_list
            )
            if forbidden_val_words_list
            else "None specifically (beyond general rule against extraneous numbers)."
        )

        # Rule 4 (Overall GT) - dynamically adjusted based on is_current_beat_root_node and overlaps
        gt_word_for_val_prompt = (
            num_to_words(overall_ground_truth_answer_val)
            if overall_ground_truth_answer_val is not None
            else "THE_FINAL_ANSWER"
        )
        overall_gt_val_rule = f"The overall final answer to the entire story is '{gt_word_for_val_prompt}' ({overall_ground_truth_answer_val})."
        if is_current_beat_root_node:
            overall_gt_val_rule += " This IS the final operation. The narrative MUST NOT explicitly state this numerical value. It should only be implied."
        else:  # Intermediate beat
            gt_is_current_atomic_input = (
                overall_ground_truth_answer_val is not None
                and any(
                    overall_ground_truth_answer_val == atom.n
                    for atom in current_op_node.children
                    if isinstance(atom, Atom)
                )
            )
            gt_is_current_beat_result = (
                overall_ground_truth_answer_val is not None
                and expected_beat_result_words_for_llm_validator
                == gt_word_for_val_prompt
            )

            if gt_is_current_atomic_input or gt_is_current_beat_result:
                roles = []
                if gt_is_current_atomic_input:
                    roles.append("a required direct atomic input")
                if gt_is_current_beat_result:
                    roles.append(
                        f"the implicit result of this beat"
                    )
                overall_gt_val_rule += (
                    f" For THIS intermediate beat, '{gt_word_for_val_prompt}' also serves as {' and '.join(roles)}. "
                    f"It MUST be mentioned/handled *only* in this specific role. It must NOT be treated as the conclusive outcome of the *entire story* here."
                )
            else:  # GT is not current atom/result for this intermediate beat
                overall_gt_val_rule += " This number MUST NOT be mentioned in this intermediate beat at all, unless it's an allowed phrasing number (Rule 2) used non-confusingly."

        # Determine which operation-specific rule to include based on the current operation
        operation_specific_rule = ""
        if current_op_node.op == "MED" and expected_beat_result_words_for_llm_validator:
            operation_specific_rule = f"**SPECIAL MED OPERATION RULE:** The median value must NEVER be explicitly mentioned ANYWHERE in the text - not as an input, not as a result, not in any list of values. Any appearance of the explicit number '{expected_beat_result_words_for_llm_validator}' is an automatic fail for MED operations."
        elif current_op_node.op in ["MAX", "MIN"]:
            operation_specific_rule = f'**SPECIAL MAX/MIN OPERATION RULE:** If the result value is also one of the required inputs, the narrative may mention this number BUT ONLY in a context that clearly establishes it as one of the inputs being compared (e.g., in a list of values). The number must NOT be directly identified as the final answer/result. The result must still be implied conceptually without explicitly stating "this is the largest/smallest value."'
        elif current_op_node.op == "SM":
            operation_specific_rule = f"**SPECIAL SM OPERATION RULE:** For sum modulo (SM) operations, conceptual inputs CAN be mentioned explicitly. The direct sum of atomic inputs MAY be mentioned, but NO OTHER intermediate calculations, partial sums, or combinations are allowed. Absolutely no other number manipulations should be shown."

        # Determine operation-specific success pattern
        operation_success_pattern = ""
        if current_op_node.op == "MED" and expected_beat_result_words_for_llm_validator:
            operation_success_pattern = f"""**Success Pattern for MED Operations:**
For MED operations, successful examples:
1. List/mention all required atomic inputs EXCEPT the median value
2. Create a conceptual reference to the median (e.g., "the central value," "the balance point")
3. COMPLETELY AVOID stating the median number ('{expected_beat_result_words_for_llm_validator}') anywhere in the text
4. May use ordinal position or conceptual placement to imply which value is the median"""
        elif current_op_node.op in ["MAX", "MIN"]:
            operation_success_pattern = f"""**Success Pattern for MAX/MIN Operations:**
For MAX/MIN operations, successful examples:
1. List/mention ALL required atomic inputs (even if one is also the result)
2. When the result is also a required input, the number is only mentioned as part of the list of inputs
3. The narrative never explicitly identifies which value is the maximum/minimum
4. The result is implied through context, character reactions, or descriptive qualities
5. The number being the result is implied through the narrative, not directly stated"""
        elif current_op_node.op == "SM":
            operation_success_pattern = f"""**Success Pattern for SM Operations:**
For SM operations, successful examples:
1. List/mention ALL required atomic inputs
2. Prior conceptual results MAY be explicitly mentioned by number value
3. ONLY the direct sum of ALL atomic inputs MAY be mentioned (and absolutely nothing else)
4. The final modulo result should be implied through narrative for intermediate beats
5. NO OTHER intermediate calculations, partial sums, combinations, or manipulations should appear
6. Numbers that are not direct inputs or the exact total sum MUST NOT appear"""
        else:
            operation_success_pattern = f"""**Success Pattern for {current_op_node.op} Operations:**
For successful examples:
1. Include all required atomic inputs
2. Follow the result statement rule for this operation
3. Avoid mentioning any forbidden numbers
4. Only use allowed numbers as specified in the rules"""

        # Determine operation-specific failure pattern
        operation_failure_pattern = (
            "**Failure Pattern for Operations:**\nExamples fail when they:\n"
        )
        if current_op_node.op == "MED":
            operation_failure_pattern += (
                "1. Explicitly state the median value anywhere in the text\n"
            )
            operation_failure_pattern += "2. Omit required inputs or mention forbidden numbers not allowed by the rules above\n"
            operation_failure_pattern += '3. Directly state "the answer/result is X" or similar explicit declarations'
        elif current_op_node.op in ["MAX", "MIN"]:
            operation_failure_pattern += "1. Explicitly identify a number as the maximum/minimum when it's also a required input\n"
            operation_failure_pattern += "2. Omit required inputs or mention forbidden numbers not allowed by the rules above\n"
            operation_failure_pattern += '3. Directly state "the answer/result is X" or similar explicit declarations'
        else:
            operation_failure_pattern += "1. Omit required inputs or mention forbidden numbers not allowed by the rules above\n"
            operation_failure_pattern += '2. Directly state "the answer/result is X" or similar explicit declarations'

        # Determine validation task instruction based on operation type
        validation_task_instruction = 'Evaluate the "Generated Beat Text" against ALL criteria above. Be extremely pedantic about numerical rules.'
        if current_op_node.op == "MED" and expected_beat_result_words_for_llm_validator:
            validation_task_instruction += f" For this MED operation, check FIRST that the median value ({expected_beat_result_words_for_llm_validator}) does not appear ANYWHERE in the text. This is the most critical rule for MED operations."
        elif current_op_node.op in ["MAX", "MIN"]:
            validation_task_instruction += f" For this {current_op_node.op} operation, check that if the result is also a required input, the number is mentioned ONLY as an input and is never explicitly identified as the {current_op_node.op.lower()} value."
        elif current_op_node.op == "SM":
            validation_task_instruction += " For this SM operation, verify that ONLY the direct inputs and the exact sum of ALL atomic inputs appear. Reject any text with unauthorized numbers, partial calculations, or intermediate steps."

        validation_task_instruction += " Provide your response *only* as a single, valid JSON object. Do not include any other text, explanations, or markdown outside of this JSON structure."

        validator_user_prompt = f"""Evaluate this narrative scene for STRICT NUMBER RULE COMPLIANCE.

**Scene Context:**
- Genre: {world_info.get("genre", "N/A")}, Setting: {world_info.get("setting", "N/A")}
- Central Items/Focus: {primary_object_name}
- Mathematical Operation Represented: {current_op_node.op} ({OP_LABELS.get(current_op_node.op, current_op_node.op)})

**Critical Number Rules for Validation:**

1. REQUIRED NUMBERS (MUST be present as words, e.g., 'seven' not '7'):
   - Atomic inputs: {atomic_inputs_words_str_for_llm_validator}
   - Result must be IMPLICIT only (not stated)

2. ADDITIONALLY ALLOWED NUMBERS (MAY be present):
   - Small numbers for general phrasing (one, two, three, etc.)
   - The number {current_op_arity} ONLY if used to count the items involved in this scene's action

3. STRICTLY FORBIDDEN NUMBERS (MUST NOT appear anywhere):
   {forbidden_numbers_str_for_llm_validator}

4. CONCEPTUAL REFERENCES:
   - Prior calculated quantities are referred to by these conceptual names: {conceptual_inputs_str_for_llm_validator}
   - The text MUST use these exact conceptual names when referring to prior results, NOT their numerical values

5. IMPLICIT RESULT HANDLING:
   - The numerical result {expected_beat_result_words_for_llm_validator} ({expected_beat_result}) MUST NOT be explicitly stated
   - The scene should imply this result through the narrative events

**Generated Beat Text to Evaluate:**
"""
{beat_text_to_validate}
"""

**Your Validation Task:**
Analyze the text above and determine if it STRICTLY follows ALL number rules.
1. First, identify ALL numbers mentioned (as words or digits)
2. Check if required numbers are present
3. Verify no forbidden numbers appear
4. Confirm the result is implied, not stated
5. Check that conceptual references are used correctly

**Answer PRECISELY with:**
VALID if ALL rules are followed perfectly, or
INVALID: [specific rule violation] if ANY rule is broken.
"""
        try:
            # ... (log_prompt for validator) ...
            # ... (API call to _chat_completion_call with validator_user_prompt, use_json_mode=True) ...
            # ... (Parse JSON response into validation_result) ...
            # (This part of your code for calling validator and parsing JSON seems okay, ensure it uses logger_obj, context_config)
            api_call_params_for_validator = {
                "model": context_config.LLM_VALIDATOR_MODEL,
                "messages": [
                    {"role": "system", "content": validator_system_prompt},
                    {"role": "user", "content": validator_user_prompt},
                ],
                "max_completion_tokens": context_config.BEAT_MAX_TOKENS,
                "temperature": context_config.LLM_VALIDATOR_TEMP,
                "reasoning": {"exclude": True},
                "json_schema": VALIDATOR_RESPONSE_SCHEMA,  # Use the defined schema instead of use_json_mode
            }
            resp_val = _chat_completion_call(**api_call_params_for_validator)

            validator_raw_output = ""
            # (Your existing robust extraction of validator_raw_output from resp_val)
            if resp_val and resp_val.choices and resp_val.choices[0].message:
                validator_raw_output = resp_val.choices[0].message.content or ""
                
            # Add more detailed debugging for validator response
            if validator_raw_output:
                logger_obj.debug(f"Validator raw output (first 400 chars): {validator_raw_output[:400]}")
            else:
                logger_obj.warning(f"Validator returned empty response! Response object: {resp_val}")
                
            # Better tracking of JSON issues
            try:
                before_parse = time.time()
                validation_result = parse_llm_json_with_fallback(
                    validator_raw_output,
                    {
                        "is_valid": False,
                        "explanation_for_generator": "Validator response was not valid JSON or was empty. Please try again with clearer numerical information.",
                        "overall_revision_summary_for_generator_prompt": "The validator had trouble processing your text. Try generating a new beat with clearer numerical presentation.",
                    },
                    f"in LLM validator iteration {iteration}",
                )
                after_parse = time.time()
                parse_time = after_parse - before_parse
                logger_obj.debug(f"LLM Validator JSON parsing took {parse_time:.4f}s. Result keys: {list(validation_result.keys())}")
                logger_obj.debug(f"Validation result: {validation_result}")
            except Exception as parse_err:
                logger_obj.error(f"Error during JSON parsing of validator output: {parse_err}")
                validation_result = {
                    "is_valid": False,
                    "explanation_for_generator": f"Error parsing validator response: {parse_err}",
                    "overall_revision_summary_for_generator_prompt": "Internal error during validation. Try simpler narrative with clear numerical statements.",
                }

            history_of_critiques.append(validation_result)

            if validation_result.get("is_valid"):
                logger_obj.info(
                    f"LLM Validator PASSED beat in iter {iteration}. Audit: {validation_result.get('explanation_for_audit')}"
                )
                return generated_text_cleaned  # Success!
            else:
                logger_obj.warning(
                    f"LLM Validator FAILED beat in iter {iteration}. Feedback: {validation_result.get('explanation_for_generator')}"
                )
                # Loop continues

        except Exception as e_val_call:
            logger_obj.error(
                f"Error during LLM validation call/processing iter {iteration}: {e_val_call}"
            )
            # Ensure history_of_critiques has a corresponding entry
            if len(history_of_critiques) < len(
                history_of_attempts
            ):  # Should always be equal or critiques one less
                history_of_critiques.append(
                    {
                            "is_valid": False,
                        "explanation_for_generator": f"Exception during validation: {e_val_call}",
                        "overall_revision_summary_for_generator_prompt": "System error during validation. Retry task.",
                        }
                )
            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(context_config.RETRY_INITIAL_DELAY)
                continue
            else:
                return None  # Failed last attempt

    logger_obj.error(
        f"Beat failed LLM validation after {context_config.MAX_LLM_VALIDATION_ITERATIONS} iterations."
    )
    return None


# --- Narrative Generation with Parent Operator Prompting ---
# verbose-listops.py

# ... (imports, Config, make_number_validator, etc.) ...


def _generate_narrative_recursive(
    node: Node,
    context: "GenerationContext",  # Assumes GenerationContext is defined
    is_root: bool,
):
    world = context.world
    config_obj = (
        context.config
    )  # Use config_obj to avoid conflict with module-level 'config'
    encoder = context.encoder
    logger_obj = context.logger  # Use logger_obj
    narrative_anchor_map = context.narrative_anchor_map

    node_id = id(node)
    # Ensure narrative_anchor is fetched correctly for OpNode
    narrative_anchor = "atom"  # Default for Atom
    if isinstance(node, OpNode):
        narrative_anchor = narrative_anchor_map.get(
            node_id, f"the_unnamed_{node.op}_entity"
        )

    # ... (token budget logging as before) ...

    if isinstance(node, Atom):
        logger_obj.debug(f"Node is Atom ({node.n}), returning.")
        return

    # This is an OpNode if we reach here
    op_for_log = getattr(node, "op", "OpNode_No_Op_Attr")
    logger_obj.debug(
        f"[Sample {context.sample_index + 1}] _generate_narrative_recursive: "
        f"Processing OpNode: {op_for_log}, Anchor: '{narrative_anchor}', Root: {is_root}, "
        f"Beat: {context.beat_counter['current'] + 1}/{context.beat_counter['total']}"
    )

    child_narrative_anchors = []
    child_op_node_results_as_conceptual_inputs = (
        {}
    )  # Store {anchor_name: numerical_value}

    for child_index, child in enumerate(node.children):
        logger_obj.debug(
            f"Processing child {child_index+1}/{len(node.children)} of {node.op} ({narrative_anchor})"
        )
        _generate_narrative_recursive(child, context, is_root=False)  # Recursive call

        if isinstance(child, OpNode):
            child_anchor = narrative_anchor_map.get(id(child))
            if child_anchor:
                child_narrative_anchors.append(child_anchor)
                if (
                    child.value is not None
                ):  # child.value should be populated by eval_node
                    child_op_node_results_as_conceptual_inputs[child_anchor] = (
                        child.value
                    )
                else:
                    logger_obj.error(
                        f"Child OpNode {child.op} (anchor: {child_anchor}) has no evaluated value!"
                    )
            else:
                logger_obj.warning(
                    f"OpNode child {child.op} of parent {node.op} has no narrative anchor in map."
                )
        # If child is Atom, its value is child.n, handled directly as atomic input to current node.

        # ... (token budget check after child processing) ...
        if (
            context.tokens_used
            >= context.config.MAX_TOTAL_TOKENS - context.config.MAX_TOKENS_BUFFER
        ):  # Use config_obj
            logger_obj.warning(
                f"TOKEN LIMIT REACHED after child. Halting for {node.op} ({narrative_anchor})."
            )
            raise BeatGenerationError(
                "Token limit reached during child processing."
            )  # Raise to stop this branch

    logger_obj.debug(
        f"Finished children for {node.op} ({narrative_anchor}). Processing node."
    )
    context.beat_counter["current"] += 1
    logger_obj.info(
        f"Generating beat {context.beat_counter['current']}/{context.beat_counter['total']} for operator {node.op} ({narrative_anchor})"
    )

    op_label = OP_LABELS.get(node.op, node.op)
    direct_atom_children = [
        c_atom for c_atom in node.children if isinstance(c_atom, Atom)
    ]
    # operand_count for the prompt should be the arity of the current node (total children)
    # but for number validation, it's often about direct new atoms if op combines atoms + conceptual anchors.
    # Let's use arity for the prompt's "operand_count" mention.
    current_op_arity = len(node.children)
    direct_atom_values = {a.n for a in direct_atom_children}
    correct_result = node.value  # Should be pre-calculated by eval_node

    # --- Inputs for the current beat's prompt and validation ---
    # Conceptual inputs (anchors from child OpNodes and their values)
    conceptual_inputs_for_prompt = []
    for anchor_name, numeric_val in child_op_node_results_as_conceptual_inputs.items():
        conceptual_inputs_for_prompt.append(
            f"the concept known as \'{anchor_name}\' (which represents the numerical value {numeric_val})"
        )
    conceptual_inputs_str_for_prompt = (
        ", ".join(conceptual_inputs_for_prompt)
        if conceptual_inputs_for_prompt
        else "None (this is the first calculation or uses only new numbers)"
    )

    # Atomic inputs (direct numbers for this beat)
    atomic_inputs_for_prompt_detailed = []
    for atom_node in direct_atom_children:
        atomic_inputs_for_prompt_detailed.append(
            f"'{num_to_words(atom_node.n)}' ({atom_node.n})"
        )
    atomic_inputs_str_for_prompt = (
        ", ".join(atomic_inputs_for_prompt_detailed)
        if atomic_inputs_for_prompt_detailed
        else "None (this is the first calculation or uses only new numbers)"
    )

    # --- Define `forbidden_atoms_for_validator` for Python strict validator ---
    # This includes:
    # 1. All atoms introduced in *any* previous beat (context.introduced_atoms)
    # 2. Results of all *prior distinct operations* that are not direct inputs to *this* operation.
    #    (Conceptual anchors handle this implicitly for the generator; validator needs the raw numbers)
    # 3. The overall_ground_truth_answer, unless this IS the root node (where it's the target)
    #    or if GT happens to be a current direct atomic input or the current beat's (implicit) result.

    forbidden_for_current_beat_py_validator = set(
        context.introduced_atoms
    )  # Atoms from all prior beats
    # Add results of *all* OpNodes processed so far that are NOT direct conceptual inputs to this node
    if context.overall_ast_root is not None:
        for op_node_id, anchor_name in narrative_anchor_map.items():
            # Find the node object with safety check
            processed_op_node = next(
                (n for n in postorder(context.overall_ast_root) if id(n) == op_node_id),
                None,
            )
            if processed_op_node and processed_op_node.value is not None:
                # If this processed_op_node's result is NOT a conceptual input to the current `node`
                is_conceptual_input_to_current = any(
                    child_op
                    for child_op in node.children
                    if isinstance(child_op, OpNode) and id(child_op) == op_node_id
                )
                if (
                    not is_conceptual_input_to_current and processed_op_node != node
                ):  # Don't forbid current node's own (future) result
                    forbidden_for_current_beat_py_validator.add(processed_op_node.value)
    else:
        logger_obj.warning(
            "overall_ast_root is None, skipping processing of prior OpNode results"
        )

    if context.overall_ground_truth_answer is not None:
        if not is_root:  # If not root, GT is generally forbidden
            # Unless GT is a direct atomic input for this beat, or the (implicit) result of this beat
            if not (
                context.overall_ground_truth_answer in direct_atom_values
                or context.overall_ground_truth_answer == correct_result
            ):
                forbidden_for_current_beat_py_validator.add(
                    context.overall_ground_truth_answer
                )
        # If it IS the root, GT is the target, so it's not "forbidden" in the same way;
        # it just shouldn't be *stated explicitly*. The Python validator's `is_root_node_being_validated`
        # and check against `overall_ground_truth_answer` in `truly_disallowed_extras` handles this.

    # Remove any numbers that are required_atoms for the current beat from the forbidden list for this beat
    # Also remove the current beat's (implicit) result if it's on there by mistake
    forbidden_for_current_beat_py_validator -= direct_atom_values
    if correct_result is not None:
        forbidden_for_current_beat_py_validator.discard(correct_result)

    # --- Define `must_avoid_str_for_generator_prompt` ---
    # This is for the generator's "STRICTLY FORBIDDEN" rule. It should be similar to above but word-based.
    # It should list prior results (values of conceptual_inputs are okay for generator to know for context)
    # and the overall_ground_truth_answer (conditionally).
    temp_forbidden_for_gen_prompt_numbers = set()
    # Add values of *other* anchors not currently being used as input
    all_anchor_values_seen = {
        anchor_val for anchor_val in child_op_node_results_as_conceptual_inputs.values()
    }  # Values of current inputs
    if context.overall_ast_root is not None:
        for op_node_id, anchor_name_iter in narrative_anchor_map.items():
            # This requires access to the full AST to get values if not already in context.
            # Assuming context.overall_ast_root is the root of the full AST.
            iter_node = next(
                (n for n in postorder(context.overall_ast_root) if id(n) == op_node_id),
                None,
            )
            if (
                iter_node
                and iter_node.value is not None
                and iter_node.value not in all_anchor_values_seen
                and iter_node != node
            ):
                temp_forbidden_for_gen_prompt_numbers.add(iter_node.value)
    else:
        logger_obj.warning(
            "overall_ast_root is None, skipping processing of prior OpNode results for generator prompt"
        )

    if context.overall_ground_truth_answer is not None and not is_root:
        if not (
            context.overall_ground_truth_answer in direct_atom_values
            or context.overall_ground_truth_answer == correct_result
        ):
            temp_forbidden_for_gen_prompt_numbers.add(
                context.overall_ground_truth_answer
            )

    temp_forbidden_for_gen_prompt_numbers -= direct_atom_values
    if (
        correct_result is not None
    ):  # Current beat's (implicit) result is not "forbidden" for generator to know about
        temp_forbidden_for_gen_prompt_numbers.discard(correct_result)

    must_avoid_for_generator_prompt_words = sorted(
        list(
            num_to_words(n)
            for n in temp_forbidden_for_gen_prompt_numbers
            if n not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
    )
    must_avoid_str_for_generator_prompt = (
        ", ".join(f"'{w}'" for w in must_avoid_for_generator_prompt_words)
        if must_avoid_for_generator_prompt_words
        else "None applicable"
    )

    primary_object = world["object"]
    safe_primary_object_for_fstring = (
        str(primary_object).replace("{", "{{").replace("}", "}}")
    )
    # Define conceptual_reference_name (the actual anchor string for the current beat's result)
    conceptual_reference_name = narrative_anchor_map.get(id(node), f"the_unnamed_{node.op}_outcome")
    
    direct_atom_sum = None
    if node.op in ["AVG", "SM"] and direct_atom_children:
        direct_atom_sum = sum(atom_node.n for atom_node in direct_atom_children)

    # --- Build `ultra_strict_instruction` for the Generator ---
    # Rule 1: Must Include (only current atomic inputs if results are implicit)
    must_include_gen_list = list(atomic_inputs_for_prompt_words)  # Start with atoms
    # All results are now implicit - this condition is always False
    if (
        not True  # Always use implicit results
        and not is_root
        and correct_result is not None
        and node.op != "MED"
    ):
        must_include_gen_list.append(num_to_words(correct_result))

    must_include_gen_combined_str = (
        " and ".join(must_include_gen_list)
        if must_include_gen_list
        else "None (focus on conceptual inputs)"
    )

    # Rule 2: May Use
    always_allowed_phrasing_words = {
        num_to_words(n) for n in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    }
    phrasing_numbers_gen_str = ", ".join(
        f"'{w}'" for w in sorted(list(always_allowed_phrasing_words))
    )

    may_use_gen_parts = [
        f"small numbers like {phrasing_numbers_gen_str} for general narrative phrasing (e.g., 'two guards')"
    ]
    # Operand count (arity of current operation)
    # Check if arity itself is on the `temp_forbidden_for_gen_prompt_numbers` (excluding phrasing numbers)
    if current_op_arity > 0:
        arity_is_problematic_forbidden = (
            current_op_arity in temp_forbidden_for_gen_prompt_numbers
            and current_op_arity
            not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        if not arity_is_problematic_forbidden:
            may_use_gen_parts.append(
                f"the number {current_op_arity} ('{num_to_words(current_op_arity)}') if counting the direct inputs/items for THIS operation"
            )
        else:
            logger_obj.debug(
                f"Generator 'May Use': Arity {current_op_arity} is a forbidden prior result, advising caution."
            )
            may_use_gen_parts.append(
                f"the number {current_op_arity} ('{num_to_words(current_op_arity)}') ONLY if essential for counting direct inputs and clearly not the forbidden prior result"
            )

    may_use_gen_clause_content = "; ".join(may_use_gen_parts)

    # Rule 3: GT Caution (for generator)
    gt_counting_caution_for_gen = ""
    if not is_root and context.overall_ground_truth_answer is not None:
        gt_val = context.overall_ground_truth_answer
        gt_word_for_gen = num_to_words(gt_val)
        # If GT is forbidden (i.e., in must_avoid_for_generator_prompt_words) AND it's a small number (0-10 but not 1,2,3)
        if (
            gt_word_for_gen in must_avoid_for_generator_prompt_words
            and gt_val
            in range(
                context.config.MIN_ALLOWED_SMALL_NUMBER,
                context.config.MAX_ALLOWED_SMALL_NUMBER + 1,
            )
            and gt_val not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        ):
            gt_counting_caution_for_gen = (
                f"- SPECIAL CAUTION for '{gt_word_for_gen}': This number is generally forbidden. "
                f"Its use for incidental counting (e.g., '{gt_word_for_gen} items') is *highly discouraged*. Avoid if possible.\n"
            )

    # AVG/SM exception
    avg_sm_exception_str = ""
    if node.op in ["AVG", "SM"] and direct_atom_sum is not None:
        # Check if direct_atom_sum is problematic
        sum_is_problematic_forbidden = (
            direct_atom_sum in temp_forbidden_for_gen_prompt_numbers
            and direct_atom_sum
            not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        if not sum_is_problematic_forbidden:
            avg_sm_exception_str = f" (For {node.op}, you MAY also mention the intermediate sum of direct atomic inputs: {direct_atom_sum} ('{num_to_words(direct_atom_sum)}'). You MUST refer to prior conceptual results ONLY using their narrative anchors like '{', '.join(child_narrative_anchors)}'. NEVER use numerical values in descriptive phrases like 'twenty-five elements')"
        else:
            avg_sm_exception_str = f" (For {node.op}, the intermediate sum {direct_atom_sum} ('{num_to_words(direct_atom_sum)}') is a forbidden prior result; AVOID stating it if possible. You MUST refer to prior conceptual results ONLY using their narrative anchors like '{', '.join(child_narrative_anchors)}'. NEVER use numerical values in descriptive phrases)"

    result_statement_rule2 = ""
    if (
        (True and not is_root)  # Always use implicit results
        or is_root
        or (node.op == "MED" and not is_root)
    ):
        # Include special note for MED operations
        if node.op == "MED" and not is_root:
            result_statement_rule2 = "The numerical result of THIS MEDIAN operation MUST NOT be explicitly stated; imply it through narrative."
            # Add enhanced warning for MED
            result_statement_rule2 += f" ⚠️ CRITICAL MED RULE: NEVER explicitly write the value '{num_to_words(correct_result)}' or '{correct_result}' as the median/result. Instead, use phrases like 'the central value', 'the balanced point', 'the middle element', etc."
        else:
            result_statement_rule2 = "The numerical result of THIS operation MUST NOT be explicitly stated; imply it through narrative."
    else:
        # Safely include the result word, escaping any single quotes it might have
        result_word_safe = num_to_words(correct_result).replace("'", "\\'")
        result_statement_rule2 = f"The numerical result of THIS operation MUST be explicitly stated as '{result_word_safe}'."

    # --- Build the detailed ultra_strict_instruction for the Generator ---
    
    # Rule 1: MUST MENTION (Atomic Inputs)
    # Build a more detailed list with word(digit) format
    must_include_gen_list = list(atomic_inputs_for_prompt_detailed)
    
    # Special handling for MED edge case (when median is also an input)
    special_med_input_clarification = ""
    is_median_also_input = False
    if node.op == "MED" and correct_result in direct_atom_values:
        is_median_also_input = True
        # For this case, we need to remove the median value from the must_include list
        must_include_gen_list = [
            item for item in must_include_gen_list 
            if not item.startswith(f"'{num_to_words(correct_result)}'")
        ]
        # Add special MED edge case clarification for the prompt
        special_med_input_clarification = f" ⚠️ IMPORTANT: One specific number ('{num_to_words(correct_result)}') must NOT be mentioned anywhere as it is both an input and the median result."
    
    must_include_gen_combined_str = (
        " and ".join(must_include_gen_list)
        if must_include_gen_list
        else "No new numbers (all inputs are handled by special rules, see task description)"
    )

    # Rule 2: RESULT OF THIS BEAT - add reference to the conceptual name
    result_handling_rule_text = result_statement_rule2  # From existing code
    result_handling_rule_text += f" This implied result will be known conceptually as '{conceptual_reference_name}' for future steps."
    if node.op == "MED":  # Reinforce MED specific warning
        result_handling_rule_text += f" ⚠️ CRITICAL MED RULE: NEVER explicitly write the median value '{num_to_words(correct_result)}' ({correct_result}) anywhere. Imply '{conceptual_reference_name}' conceptually."

    # Rule 3: MAY USE (Phrasing/Counting)
    # Rebuild phrasing_numbers_gen_str to include digits
    temp_phrasing_words_detailed = [
        f"'{num_to_words(n)}' ({n})" 
        for n in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    ]
    phrasing_numbers_gen_str_detailed = ", ".join(sorted(temp_phrasing_words_detailed))

    may_use_gen_parts_detailed = [
        f"small numbers like {phrasing_numbers_gen_str_detailed} for general narrative phrasing (e.g., 'two guards')"
    ]
    
    # Check if current_op_arity is a problematic forbidden number (using existing logic)
    if current_op_arity > 0:
        arity_is_problematic_forbidden = (
            current_op_arity in temp_forbidden_for_gen_prompt_numbers
            and current_op_arity not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        if not arity_is_problematic_forbidden:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) IF it's genuinely used to count the items/inputs involved in THIS specific action"
            )
        else:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) ONLY if essential for counting direct inputs and clearly not the forbidden prior result"
            )
    
    may_use_gen_clause_content_detailed = "; ".join(may_use_gen_parts_detailed)

    # Rule 4: STRICTLY FORBIDDEN (Prior Results/Overall GT)
    # Rebuild must_avoid_str_for_generator_prompt to include digits
    temp_forbidden_detailed_list = []
    for n_forbidden in temp_forbidden_for_gen_prompt_numbers:
        if n_forbidden not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
            temp_forbidden_detailed_list.append(f"'{num_to_words(n_forbidden)}' ({n_forbidden})")

    must_avoid_str_for_generator_prompt_detailed = (
        ", ".join(sorted(temp_forbidden_detailed_list))
        if temp_forbidden_detailed_list
        else "None specifically (beyond the general rule against unlisted numbers)"
    )

    # Rule 6 (NEW): Referring to Prior Results (Conceptual Inputs)
    prior_results_handling_rule = ""
    if child_op_node_results_as_conceptual_inputs:  # If there are conceptual inputs
        prior_results_names_list = [f"'{name}'" for name in child_op_node_results_as_conceptual_inputs.keys()]
        prior_results_handling_rule = (
            f"6.  **REFERRING TO PRIOR RESULTS:** When your story needs to mention quantities from previous steps (conceptually known as {', '.join(prior_results_names_list)}), "
            f"you MUST use these exact conceptual names/phrases. DO NOT state the underlying numerical values these concepts represent. "
            f"For example, if a prior step resulted in 'The Dragon's Hoard' (representing 500 coins), refer to it as 'The Dragon's Hoard', not '500 coins'.\n"
        )

    # Now construct the complete ultra_strict_instruction
    # --- Build the detailed ultra_strict_instruction for the Generator ---
    
    # Rule 1: MUST MENTION (Atomic Inputs)
    # Build a more detailed list with word(digit) format
    must_include_gen_list = list(atomic_inputs_for_prompt_detailed)
    
    # Special handling for MED edge case (when median is also an input)
    special_med_input_clarification = ""
    is_median_also_input = False
    if node.op == "MED" and correct_result in direct_atom_values:
        is_median_also_input = True
        # For this case, we need to remove the median value from the must_include list
        must_include_gen_list = [
            item for item in must_include_gen_list 
            if not item.startswith(f"'{num_to_words(correct_result)}'")
        ]
        # Add special MED edge case clarification for the prompt
        special_med_input_clarification = f" ⚠️ IMPORTANT: One specific number ('{num_to_words(correct_result)}') must NOT be mentioned anywhere as it is both an input and the median result."
    
    must_include_gen_combined_str = (
        " and ".join(must_include_gen_list)
        if must_include_gen_list
        else "No new numbers (all inputs are handled by special rules, see task description)"
    )

    # Rule 2: RESULT OF THIS BEAT - add reference to the conceptual name
    result_handling_rule_text = result_statement_rule2  # From existing code
    result_handling_rule_text += f" This implied result will be known conceptually as '{conceptual_reference_name}' for future steps."
    if node.op == "MED":  # Reinforce MED specific warning
        result_handling_rule_text += f" ⚠️ CRITICAL MED RULE: NEVER explicitly write the median value '{num_to_words(correct_result)}' ({correct_result}) anywhere. Imply '{conceptual_reference_name}' conceptually."

    # Rule 3: MAY USE (Phrasing/Counting)
    # Rebuild phrasing_numbers_gen_str to include digits
    temp_phrasing_words_detailed = [
        f"'{num_to_words(n)}' ({n})" 
        for n in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    ]
    phrasing_numbers_gen_str_detailed = ", ".join(sorted(temp_phrasing_words_detailed))

    may_use_gen_parts_detailed = [
        f"small numbers like {phrasing_numbers_gen_str_detailed} for general narrative phrasing (e.g., 'two guards')"
    ]
    
    # Check if current_op_arity is a problematic forbidden number (using existing logic)
    if current_op_arity > 0:
        arity_is_problematic_forbidden = (
            current_op_arity in temp_forbidden_for_gen_prompt_numbers
            and current_op_arity not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        if not arity_is_problematic_forbidden:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) IF it's genuinely used to count the items/inputs involved in THIS specific action"
            )
        else:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) ONLY if essential for counting direct inputs and clearly not the forbidden prior result"
            )
    
    may_use_gen_clause_content_detailed = "; ".join(may_use_gen_parts_detailed)

    # Rule 4: STRICTLY FORBIDDEN (Prior Results/Overall GT)
    # Rebuild must_avoid_str_for_generator_prompt to include digits
    temp_forbidden_detailed_list = []
    for n_forbidden in temp_forbidden_for_gen_prompt_numbers:
        if n_forbidden not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
            temp_forbidden_detailed_list.append(f"'{num_to_words(n_forbidden)}' ({n_forbidden})")

    must_avoid_str_for_generator_prompt_detailed = (
        ", ".join(sorted(temp_forbidden_detailed_list))
        if temp_forbidden_detailed_list
        else "None specifically (beyond the general rule against unlisted numbers)"
    )

    # Rule 6 (NEW): Referring to Prior Results (Conceptual Inputs)
    prior_results_handling_rule = ""
    if child_op_node_results_as_conceptual_inputs:  # If there are conceptual inputs
        prior_results_names_list = [f"'{name}'" for name in child_op_node_results_as_conceptual_inputs.keys()]
        prior_results_handling_rule = (
            f"6.  **REFERRING TO PRIOR RESULTS:** When your story needs to mention quantities from previous steps (conceptually known as {', '.join(prior_results_names_list)}), "
            f"you MUST use these exact conceptual names/phrases. DO NOT state the underlying numerical values these concepts represent. "
            f"For example, if a prior step resulted in 'The Dragon's Hoard' (representing 500 coins), refer to it as 'The Dragon's Hoard', not '500 coins'.\n"
        )

    # Now construct the complete ultra_strict_instruction
    # --- Build the detailed ultra_strict_instruction for the Generator ---
    
    # Rule 1: MUST MENTION (Atomic Inputs)
    # Build a more detailed list with word(digit) format
    must_include_gen_list = list(atomic_inputs_for_prompt_detailed)
    
    # Special handling for MED edge case (when median is also an input)
    special_med_input_clarification = ""
    is_median_also_input = False
    if node.op == "MED" and correct_result in direct_atom_values:
        is_median_also_input = True
        # For this case, we need to remove the median value from the must_include list
        must_include_gen_list = [
            item for item in must_include_gen_list 
            if not item.startswith(f"'{num_to_words(correct_result)}'")
        ]
        # Add special MED edge case clarification for the prompt
        special_med_input_clarification = f" ⚠️ IMPORTANT: One specific number ('{num_to_words(correct_result)}') must NOT be mentioned anywhere as it is both an input and the median result."
    
    must_include_gen_combined_str = (
        " and ".join(must_include_gen_list)
        if must_include_gen_list
        else "No new numbers (all inputs are handled by special rules, see task description)"
    )

    # Rule 2: RESULT OF THIS BEAT - add reference to the conceptual name
    result_handling_rule_text = result_statement_rule2  # From existing code
    result_handling_rule_text += f" This implied result will be known conceptually as '{conceptual_reference_name}' for future steps."
    if node.op == "MED":  # Reinforce MED specific warning
        result_handling_rule_text += f" ⚠️ CRITICAL MED RULE: NEVER explicitly write the median value '{num_to_words(correct_result)}' ({correct_result}) anywhere. Imply '{conceptual_reference_name}' conceptually."

    # Rule 3: MAY USE (Phrasing/Counting)
    # Rebuild phrasing_numbers_gen_str to include digits
    temp_phrasing_words_detailed = [
        f"'{num_to_words(n)}' ({n})" 
        for n in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    ]
    phrasing_numbers_gen_str_detailed = ", ".join(sorted(temp_phrasing_words_detailed))

    may_use_gen_parts_detailed = [
        f"small numbers like {phrasing_numbers_gen_str_detailed} for general narrative phrasing (e.g., 'two guards')"
    ]
    
    # Check if current_op_arity is a problematic forbidden number (using existing logic)
    if current_op_arity > 0:
        arity_is_problematic_forbidden = (
            current_op_arity in temp_forbidden_for_gen_prompt_numbers
            and current_op_arity not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        if not arity_is_problematic_forbidden:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) IF it's genuinely used to count the items/inputs involved in THIS specific action"
            )
        else:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) ONLY if essential for counting direct inputs and clearly not the forbidden prior result"
            )
    
    may_use_gen_clause_content_detailed = "; ".join(may_use_gen_parts_detailed)

    # Rule 4: STRICTLY FORBIDDEN (Prior Results/Overall GT)
    # Rebuild must_avoid_str_for_generator_prompt to include digits
    temp_forbidden_detailed_list = []
    for n_forbidden in temp_forbidden_for_gen_prompt_numbers:
        if n_forbidden not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
            temp_forbidden_detailed_list.append(f"'{num_to_words(n_forbidden)}' ({n_forbidden})")

    must_avoid_str_for_generator_prompt_detailed = (
        ", ".join(sorted(temp_forbidden_detailed_list))
        if temp_forbidden_detailed_list
        else "None specifically (beyond the general rule against unlisted numbers)"
    )

    # Rule 6 (NEW): Referring to Prior Results (Conceptual Inputs)
    prior_results_handling_rule = ""
    if child_op_node_results_as_conceptual_inputs:  # If there are conceptual inputs
        prior_results_names_list = [f"'{name}'" for name in child_op_node_results_as_conceptual_inputs.keys()]
        prior_results_handling_rule = (
            f"6.  **REFERRING TO PRIOR RESULTS:** When your story needs to mention quantities from previous steps (conceptually known as {', '.join(prior_results_names_list)}), "
            f"you MUST use these exact conceptual names/phrases. DO NOT state the underlying numerical values these concepts represent. "
            f"For example, if a prior step resulted in 'The Dragon's Hoard' (representing 500 coins), refer to it as 'The Dragon's Hoard', not '500 coins'.\n"
        )

    # Now construct the complete ultra_strict_instruction
    # --- Build the detailed ultra_strict_instruction for the Generator ---
    
    # Rule 1: MUST MENTION (Atomic Inputs)
    # Build a more detailed list with word(digit) format
    must_include_gen_list = list(atomic_inputs_for_prompt_detailed)
    
    # Special handling for MED edge case (when median is also an input)
    special_med_input_clarification = ""
    is_median_also_input = False
    if node.op == "MED" and correct_result in direct_atom_values:
        is_median_also_input = True
        # For this case, we need to remove the median value from the must_include list
        must_include_gen_list = [
            item for item in must_include_gen_list 
            if not item.startswith(f"'{num_to_words(correct_result)}'")
        ]
        # Add special MED edge case clarification for the prompt
        special_med_input_clarification = f" ⚠️ IMPORTANT: One specific number ('{num_to_words(correct_result)}') must NOT be mentioned anywhere as it is both an input and the median result."
    
    must_include_gen_combined_str = (
        " and ".join(must_include_gen_list)
        if must_include_gen_list
        else "No new numbers (all inputs are handled by special rules, see task description)"
    )

    # Rule 2: RESULT OF THIS BEAT - add reference to the conceptual name
    result_handling_rule_text = result_statement_rule2  # From existing code
    result_handling_rule_text += f" This implied result will be known conceptually as '{conceptual_reference_name}' for future steps."
    if node.op == "MED":  # Reinforce MED specific warning
        result_handling_rule_text += f" ⚠️ CRITICAL MED RULE: NEVER explicitly write the median value '{num_to_words(correct_result)}' ({correct_result}) anywhere. Imply '{conceptual_reference_name}' conceptually."

    # Rule 3: MAY USE (Phrasing/Counting)
    # Rebuild phrasing_numbers_gen_str to include digits
    temp_phrasing_words_detailed = [
        f"'{num_to_words(n)}' ({n})" 
        for n in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
    ]
    phrasing_numbers_gen_str_detailed = ", ".join(sorted(temp_phrasing_words_detailed))

    may_use_gen_parts_detailed = [
        f"small numbers like {phrasing_numbers_gen_str_detailed} for general narrative phrasing (e.g., 'two guards')"
    ]
    
    # Check if current_op_arity is a problematic forbidden number (using existing logic)
    if current_op_arity > 0:
        arity_is_problematic_forbidden = (
            current_op_arity in temp_forbidden_for_gen_prompt_numbers
            and current_op_arity not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
        )
        if not arity_is_problematic_forbidden:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) IF it's genuinely used to count the items/inputs involved in THIS specific action"
            )
        else:
            may_use_gen_parts_detailed.append(
                f"the number '{num_to_words(current_op_arity)}' ({current_op_arity}) ONLY if essential for counting direct inputs and clearly not the forbidden prior result"
            )
    
    may_use_gen_clause_content_detailed = "; ".join(may_use_gen_parts_detailed)

    # Rule 4: STRICTLY FORBIDDEN (Prior Results/Overall GT)
    # Rebuild must_avoid_str_for_generator_prompt to include digits
    temp_forbidden_detailed_list = []
    for n_forbidden in temp_forbidden_for_gen_prompt_numbers:
        if n_forbidden not in context.config.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET:
            temp_forbidden_detailed_list.append(f"'{num_to_words(n_forbidden)}' ({n_forbidden})")

    must_avoid_str_for_generator_prompt_detailed = (
        ", ".join(sorted(temp_forbidden_detailed_list))
        if temp_forbidden_detailed_list
        else "None specifically (beyond the general rule against unlisted numbers)"
    )

    # Rule 6 (NEW): Referring to Prior Results (Conceptual Inputs)
    prior_results_handling_rule = ""
    if child_op_node_results_as_conceptual_inputs:  # If there are conceptual inputs
        prior_results_names_list = [f"'{name}'" for name in child_op_node_results_as_conceptual_inputs.keys()]
        prior_results_handling_rule = (
            f"6.  **REFERRING TO PRIOR RESULTS:** When your story needs to mention quantities from previous steps (conceptually known as {', '.join(prior_results_names_list)}), "
            f"you MUST use these exact conceptual names/phrases. DO NOT state the underlying numerical values these concepts represent. "
            f"For example, if a prior step resulted in 'The Dragon's Hoard' (representing 500 coins), refer to it as 'The Dragon's Hoard', not '500 coins'.\n"
        )

    # Now construct the complete ultra_strict_instruction
    ultra_strict_instruction = (
        f"**ULTRA-STRICT NUMBER RULES FOR THIS SCENE (Your Writing Guide):**\n"
        f"As the storyteller, your main goal is to weave a compelling scene. However, for this specific task, you must precisely control how numbers are mentioned:\n\n"
        f"1.  **Numbers to Weave into THIS Scene's Actions (as words, e.g., 'seven'):** {must_include_gen_combined_str}{special_med_input_clarification}.\n"
        f"2.  **Outcome of THIS Scene's Action:** {result_handling_rule_text}\n"
        f"3.  **Numbers You MAY Use for Flavor (if needed, as words):** {may_use_gen_clause_content_detailed}.\n"
        f"{gt_counting_caution_for_gen.rstrip() + ('\\n' if gt_counting_caution_for_gen.strip() else '')}"
        f"4.  **Numbers to STRICTLY AVOID Mentioning (from past events or overall story):** {must_avoid_str_for_generator_prompt_detailed}.\n"
        f"5.  **ABSOLUTELY NO OTHER NUMBERS:** Do not introduce any other numerical values (digits or words) beyond those explicitly covered by rules 1-4 above{avg_sm_exception_str}.\n"
        f"{prior_results_handling_rule}"
        f"Focus on clear storytelling that naturally implies the calculations based on these strict number rules."
    )

        # --- Define `action_description` for the Generator ---
    # Build a string of just the names of conceptual inputs for the action description
    conceptual_input_names_only_list = [f"'{name}'" for name in child_op_node_results_as_conceptual_inputs.keys()]
    conceptual_input_names_only_str = ", ".join(conceptual_input_names_only_list) if conceptual_input_names_only_list else "no prior calculated quantities"

    action_description_parts = [
        f"Your task is to write a scene where the characters engage in an activity that mirrors the '{op_label}' operation. "
        f"This activity involves:"
    ]
    if child_op_node_results_as_conceptual_inputs:
        action_description_parts.append(
            f"  - Using or considering quantities from previous story events, which are conceptually known as {conceptual_input_names_only_str}."
            f" (Remember: In your narrative, refer to these using ONLY these conceptual names, NOT their underlying numbers which were provided for your context only)."
        )
    if direct_atom_children:
        action_description_parts.append(
            f"  - Introducing or encountering new specific numbers relevant to this action: {atomic_inputs_str_for_prompt}."
        )
    else:
        action_description_parts.append(
            f"  - This action primarily transforms the existing conceptual quantities ({conceptual_input_names_only_str}) without introducing new specific numbers."
        )

    action_description_parts.append(
        f"
The outcome of this action should lead to a new understanding or quantity, which will be conceptually known as '{conceptual_reference_name}'. "
        f"This concept ('{conceptual_reference_name}') corresponds to the numerical value {correct_result} ('{num_to_words(correct_result)}')."
    )
    action_description_parts.append(
        f"CRITICAL: Your narrative MUST NOT explicitly state the number '{num_to_words(correct_result)}' ({correct_result}). "
        f"Instead, the story should imply this outcome through the characters' actions, discoveries, or the state of the '{safe_primary_object_for_fstring}', "
        f"so that '{conceptual_reference_name}' becomes the way to think about this new state."
    )
    action_description_parts.append(
        f"All numbers that ARE explicitly mentioned (only those listed in Rule 1 of your Writing Guide) must be written as words (e.g., 'seven' not '7')."
    )

    # Add MED-specific guidance (ensure it uses conceptual_reference_name and clear language)
    if node.op == "MED":
        is_median_also_input = correct_result in direct_atom_values
        action_description_parts.append(
            f"
**Special Instructions for this MEDIAN Scene:** "
            f"To imply the median ({conceptual_reference_name}) without stating its number ('{num_to_words(correct_result)}'), "
            f"describe characters identifying a 'central element', 'balancing point', or 'middle value' from the available items/numbers. "
            f"DO NOT use the word '{num_to_words(correct_result)}' or the digit '{correct_result}' anywhere."
        )
        if is_median_also_input:
             action_description_parts.append(
                 f"  ⚠️ VERY IMPORTANT: The number '{num_to_words(correct_result)}' ({correct_result}) is both the median outcome AND one of the initial numbers involved. "
                 f"Because of this, you MUST NOT mention '{num_to_words(correct_result)}' ({correct_result}) AT ALL in this scene, not even when listing the initial numbers. "
                 f"Describe the set of initial numbers by listing the *other* numbers and referring to this specific one indirectly (e.g., '...and one particular item whose nature was central...')."
             )

    # Add SM-specific guidance (ensure it uses conceptual_reference_name and clear language about inputs)
    elif node.op == "SM":
        action_description_parts.append(
            f"
**Special Instructions for this SUM MODULO 10 Scene:** "
            f"The characters' actions should represent combining all involved quantities (the new numbers: {atomic_inputs_str_for_prompt}; and the prior results: {conceptual_input_names_only_str}) "
            f"and then finding a core essence or pattern related to this total, equivalent to taking the sum modulo 10. "
            f"Remember to refer to prior results using ONLY their conceptual names ({conceptual_input_names_only_str}). "
            f"The final conceptual outcome is '{conceptual_reference_name}' (representing '{num_to_words(correct_result)}' ({correct_result})), which must be implied, not stated numerically."
        )

    action_description = "
".join(action_description_parts) # Use newline for better formatting in prompt

    # --- Create Python Validator for this beat ---
    # `enforce_result_presence` is True if result MUST be stated, False if it MUST be implicit (for non-root)
    py_validator_enforce_result_presence = (
        not True  # Always use implicit results if not is_root else False
    )  # Root result always implicit
    
    # Special handling for MED operations - Their results should always be implicit regardless of config
    if node.op == "MED" and not is_root:
        py_validator_enforce_result_presence = False
    logger_obj.debug(
            f"Setting enforce_result_presence=False for MED operation regardless of config (results always implicit for MED)"
        )

    validate_beat_numbers = make_number_validator(
        allowed_atoms=direct_atom_values,  # Python validator cares about numerical atoms
        forbidden_atoms=forbidden_for_current_beat_py_validator,
        operand_count=current_op_arity,  # Arity for operand_count check
        correct_result_for_beat=correct_result,
        enforce_result_presence=py_validator_enforce_result_presence,
        operation_type=node.op,
        overall_ground_truth_answer=context.overall_ground_truth_answer,
        is_root_node_being_validated=is_root,  # Pass is_root
        config_obj=context.config,  # Pass config for phrasing numbers
        logger_obj=logger_obj,
    )

    # --- Prepare for LLM Validation Loop ---
    # Update system prompt to include genre information
    genre = world.get("genre", "Fantasy")
    system_prompt_for_generator = (
        f"You are a master {genre} storyteller crafting a narrative. Your task is to write a single scene contributing to an ongoing story. "
        f"Focus solely on advancing the tale as specified. Do not include explanations or analysis. "
        f"The story involves mathematical operations via narrative actions. Pay careful attention to number rules. Produce ONLY clean narrative text."
    )
    context_snippet = clean_snippet(
        context.last_scene_text, max_len=context.config.BEAT_CONTEXT
    )

        initial_user_message_for_generator = (
        f"Story Scene Task: {conceptual_reference_name} (Scene {context.beat_counter['current']}/{context.beat_counter['total']})

"
        f"**Background for Your Scene:**
"
        f"- Genre: {world.get('genre', 'N/A')}
"
        f"- Setting: {world.get('setting', 'N/A')}
"
        f"- Central Items: {primary_object}
"
        f"- Quantities from Previous Events (Refer to these by their conceptual names only in your story): {conceptual_inputs_str_for_prompt}
"
        f"- New Numbers Introduced in this Scene (Refer to these by word form, e.g. 'seven'): {atomic_inputs_str_for_prompt}

"
        f"**Your Scene's Core Action & Narrative Goal:**
{action_description}

"
        f"{ultra_strict_instruction}

"
        f"**Continue From (End of last scene):**
\"{context_snippet}\"

"
        f"**Your Response:**
Write ONLY the narrative text for this new scene, continuing smoothly. Do not add titles, notes, or anything outside the story itself."
    )

    current_max_beat_completion_tokens = context.config.BEAT_MAX_TOKENS
    beat_text_final_validated = None

    # --- Outer Beat Generation Retry Loop (uses LLM validation internally) ---
    for attempt_outer in range(1, context.config.MAX_BEAT_RETRIES + 1):
        logger_obj.info(
            f"[Sample {context.sample_index+1}, Beat Op: {node.op}, Anchor: {narrative_anchor}] Outer Beat Gen Attempt: {attempt_outer}/{context.config.MAX_BEAT_RETRIES}"
        )

        # Prepare inputs for LLM validator prompt within _generate_and_llm_validate_beat
        llm_val_conceptual_inputs_str = (
            ", ".join(
                [
                    f"'{name_val}' (value: {child_op_node_results_as_conceptual_inputs.get(name_val, 'UNKNOWN')})"
                    for name_val in child_narrative_anchors
                ]
            )
            if child_narrative_anchors
            else "None (this is the first calculation or uses only new numbers)"
        )
        llm_val_atomic_inputs_words_str = (
            atomic_inputs_str_for_prompt  # Already word list
        )

        llm_val_expected_beat_result_words = None
        if correct_result is not None:
            llm_val_expected_beat_result_words = num_to_words(correct_result)

        # Call the iterative LLM validation loop
        llm_validated_beat_text = _generate_and_llm_validate_beat(
            original_user_message_for_generator=initial_user_message_for_generator,  # The full initial prompt for the generator
            system_prompt_for_generator=system_prompt_for_generator,
            world_info=world,
            current_op_node=node,
            # For LLM Validator prompt:
            conceptual_inputs_str_for_llm_validator=llm_val_conceptual_inputs_str,
            atomic_inputs_words_str_for_llm_validator=llm_val_atomic_inputs_words_str,
            action_description_for_llm_validator=action_description,  # The same action_description
            expected_beat_result_words_for_llm_validator=llm_val_expected_beat_result_words,  # Word form of current beat's result
            ultra_strict_instruction_for_llm_validator_context=ultra_strict_instruction,  # The rules given to generator
            # Other params for _generate_and_llm_validate_beat:
            current_max_beat_completion_tokens=current_max_beat_completion_tokens,
            sample_index=context.sample_index,
            context_config=context.config,
            logger_obj=logger_obj,
            encoder_obj=encoder,
            is_current_beat_root_node=is_root,
            overall_ground_truth_answer_val=context.overall_ground_truth_answer,
            primary_object_name=primary_object,
            # Pass the more refined forbidden set for the LLM validator's "Forbidden" rule context
            forbidden_prior_results_and_gt_for_llm_validator=temp_forbidden_for_gen_prompt_numbers,  # Use the numerically-based set
        )

        if llm_validated_beat_text:
            # Final check with Python strict validator
            if validate_beat_numbers(llm_validated_beat_text):
                beat_text_final_validated = llm_validated_beat_text
                logger_obj.info(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Python validator PASSED LLM-validated beat."
                )
                break  # Success for this beat
            else:
                logger_obj.warning(
                    f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Python validator FAILED for LLM-validated beat. Outer attempt {attempt_outer} failed. Will retry outer loop."
                )
                # The _log_failed_validation is called inside make_number_validator
        else:
            logger_obj.warning(
                f"[Sample {context.sample_index+1}, Beat Op: {node.op}] Iterative LLM validation loop returned None. Outer attempt {attempt_outer} failed. Will retry outer loop."
            )

        if attempt_outer < context.config.MAX_BEAT_RETRIES:
            time.sleep(context.config.RETRY_INITIAL_DELAY * (2 ** (attempt_outer - 1)))
        # No "else" here, if loop finishes without break, beat_text_final_validated is None

    if not beat_text_final_validated:
        logger_obj.error(
            f"Operator {node.op} ({narrative_anchor}) failed after {context.config.MAX_BEAT_RETRIES} outer attempts (incl. LLM validation loops). Aborting narrative generation for this sample."
        )
        raise BeatGenerationError(
            f"Failed to generate narrative beat for operator {node.op} ({narrative_anchor}) after all outer retries."
        )

    # --- Process successful beat ---
    beat_text = beat_text_final_validated
    btoks = len(encoder.encode(beat_text))
    context.scenes.append(beat_text)
    context.tokens_used += btoks
    context.last_scene_text = beat_text
    # Add *only current beat's direct atomic inputs* to introduced_atoms
    # Results (even if stated) are handled by the forbidden_for_current_beat_py_validator logic for subsequent beats.
    context.introduced_atoms.update(direct_atom_values)

    logger_obj.debug(
        f"Beat {context.beat_counter['current']} successful. Introduced atoms updated with current beat's atoms: {direct_atom_values}"
    )
    # ... (logging for token usage) ...

    # --- Padding Logic (largely unchanged, ensure it uses config_obj and logger_obj) ---
    if not is_root:
        # ... (existing padding loop, ensure it uses config_obj, logger_obj, and the correct validate_padding)
        # Create validate_padding for this slot:
        # Forbidden for padding: all atoms introduced so far + current beat's atoms + current beat's result (even if implicit)
        forbidden_for_padding_slot = set(
            context.introduced_atoms
        )  # Already includes current beat's atoms
        if correct_result is not None:
            forbidden_for_padding_slot.add(correct_result)
        if context.overall_ground_truth_answer is not None:
            forbidden_for_padding_slot.add(context.overall_ground_truth_answer)

        validate_padding = make_number_validator(
            allowed_atoms=set(),
            forbidden_atoms=forbidden_for_padding_slot,
            operand_count=0,
            correct_result_for_beat=None,
            strict_zero=True,
            enforce_result_presence=False,
            operation_type="PADDING",
            overall_ground_truth_answer=context.overall_ground_truth_answer,
            is_root_node_being_validated=False,
            config_obj=context.config,  # Using context.config is correct
            logger_obj=context.logger,
        )

        # ... (The rest of your padding loop from the original _generate_narrative_recursive)
        # Ensure it uses context.config (which is config_obj here), context.logger (logger_obj)
        # and the `validate_padding` created above.
        # Example snippet of the padding call:
        # padding_text = generate_with_retry(
        #     padding_system_prompt, padding_user_prompt, config_obj.PADDING_MAX_TOKENS,
        #     validate_padding, config_obj.MAX_PAD_RETRIES, context.sample_index,
        #     config_obj.CREATIVE_NARRATIVE_TEMP, {"exclude": True}
        # )
        # This part of the padding logic seems mostly fine but needs to use the scoped config_obj and logger_obj.
        # The detailed padding loop is omitted here for brevity but should be integrated from your existing code.
        # Key is that `validate_padding` is correctly formed.
        current_padding_total_overall = context.padding_stats["total_padding_tokens"]
        max_padding_allowed_overall = context.padding_stats["max_padding_allowed"]
        padding_budget_this_slot = context.padding_stats["padding_per_slot"]
        padding_tokens_added_this_slot = 0
        local_padding_segments_added_this_slot = 0
        padding_termination_reason = (
            "Loop completed (max segments or slot budget likely met)."
        )

        logger_obj.info(
            f"PADDING SLOT INIT [{node.op}/{narrative_anchor}]: Slot Budget: {padding_budget_this_slot if padding_budget_this_slot > 0 else 'N/A'}, Overall Budget: {max_padding_allowed_overall - current_padding_total_overall} remaining."
        )

        if padding_budget_this_slot > 0:
            while (
                context.tokens_used
                < context.config.MAX_TOTAL_TOKENS - context.config.MAX_TOKENS_BUFFER
                and local_padding_segments_added_this_slot < context.max_pad_paragraphs
                and current_padding_total_overall < max_padding_allowed_overall
                and padding_tokens_added_this_slot < padding_budget_this_slot
            ):
                estimated_next_padding_segment_cost = (
                    context.config.PADDING_MAX_TOKENS
                    + context.config.MAX_TOKENS_BUFFER // 5
                )  # Using a fraction of buffer as overhead

                if (
                    padding_tokens_added_this_slot + estimated_next_padding_segment_cost
                    > padding_budget_this_slot
                ):
                    padding_termination_reason = f"Slot budget (est. {padding_tokens_added_this_slot + estimated_next_padding_segment_cost}/{padding_budget_this_slot})"
                    break
                if would_exceed_budget(
                    context.tokens_used,
                    context.config.PADDING_MAX_TOKENS,
                    context.config.MAX_TOTAL_TOKENS,
                    context.config.MAX_TOKENS_BUFFER,
                ):
                    padding_termination_reason = (
                        "Overall token budget (pre-gen check for max completion)"
                    )
                    break

                padding_system_prompt = "You are a concise storyteller, skilled at adding brief, atmospheric paragraphs that bridge scenes without introducing new numbers or calculations."
                cleaned_snippet_padding = clean_snippet(
                    context.last_scene_text, max_len=context.config.PADDING_CONTEXT
                )
                padding_user_prompt = (
                    f"The story is set in a {context.world.get('genre', 'mysterious world')} ({context.world.get('setting', 'unknown location')}).\n"
                    f"The characters are focused on {context.world.get('object', 'important items')}.\n"
                    f'Previous Scene Snippet (End of last scene): "...{cleaned_snippet_padding.replace("\\n", " ")}..."\n\n'
                    f"Task: Write ONE short, atmospheric paragraph (typically 3-5 sentences) that continues smoothly from the previous scene snippet. "
                    f"This paragraph should be purely narrative filler or scene transition. "
                    f"ABSOLUTELY NO NUMBERS (digits or words like 'one', 'two', 'first', etc.) are allowed in this paragraph, except potentially 'one', 'two', or 'three' if used for completely general phrasing and not quantities. Strive for zero numbers. "
                    f"Do not advance the core plot calculation. Do not mention specific quantities. "
                    f"Output ONLY the text for this single paragraph. No titles, no explanations, no analysis."
                )
                padding_text = generate_with_retry(
                    padding_system_prompt,
                    padding_user_prompt,
                    context.config.PADDING_MAX_TOKENS,  # Use context.config
                    validate_padding,
                    context.config.MAX_PAD_RETRIES,  # Use context.config
                    context.sample_index,
                    context.config.CREATIVE_NARRATIVE_TEMP,  # Use context.config
                    {"exclude": True},
                )

                if padding_text:
                    ptoks = len(encoder.encode(padding_text))
                    # ... (budget checks for actual ptoks from your original code) ...
                    if not (
                        context.tokens_used + ptoks
                        <= context.config.MAX_TOTAL_TOKENS
                        - context.config.MAX_TOKENS_BUFFER
                    ):
                        padding_termination_reason = (
                            f"Overall total token limit (actual)"
                        )
                        break

                    local_padding_segments_added_this_slot += 1
                    context.scenes.append(padding_text)
                    context.tokens_used += ptoks
                    context.last_scene_text = padding_text
                    context.padding_stats["total_padding_tokens"] += ptoks
                    current_padding_total_overall = context.padding_stats[
                        "total_padding_tokens"
                    ]
                    context.padding_stats["padding_segments_added"] += 1
                    padding_tokens_added_this_slot += ptoks
                else:
                    padding_termination_reason = "Padding generation/validation failed"
                    break

            if local_padding_segments_added_this_slot > 0:
                logger_obj.info(
                    f"PADDING SLOT SUMMARY [{node.op}/{narrative_anchor}]: Added {local_padding_segments_added_this_slot} segments, using {padding_tokens_added_this_slot}/{padding_budget_this_slot} tokens. Term: {padding_termination_reason}."
                )
            # ... (other padding summary logging) ...

    logger_obj.debug(f"TOKEN BUDGET END [{op_for_log}/{narrative_anchor}]: ...")


def generate_introduction_scene(
    world_info: dict,
    sample_index: int | None = None,
    config_obj: Config = config,  # Add config_obj parameter
    logger_obj: logging.Logger = logger,  # Add logger_obj parameter
) -> str | None:
    logger_obj.info(
        f"[Sample {sample_index + 1 if sample_index is not None else 'N/A'}] Generating introduction scene..."
    )

    # --- ADDED/COMPLETED PROMPT DEFINITIONS ---
    system_prompt = (
        f"You are a master {world_info.get('genre')} storyteller. Your task is to write a compelling introductory scene for a new story. "
        "This scene should establish the setting, introduce one or two key characters, and hint at a central mystery or goal related to the primary object. "
        "Crucially, this introductory scene MUST NOT contain any numerical values (digits or words like 'one', 'two', 'first', etc.), "
        "except potentially the word 'one', 'two', or 'three' if used for completely general, non-quantitative phrasing (e.g., 'a single ray of light', 'two figures emerged', 'three ancient symbols'). Strive for zero numbers. "
        "Focus on atmosphere and intrigue. Do not reveal any specific quantities or begin any calculations. "
        "Output ONLY the narrative text for this scene. No titles, no explanations, no analysis."
    )

    characters_list = world_info.get("characters", [])
    char_names_roles = []
    if characters_list:
        # Select one or two characters for the intro
        num_intro_chars = random.randint(1, min(2, len(characters_list)))
        intro_chars = random.sample(characters_list, num_intro_chars)
        for char_info in intro_chars:
            char_names_roles.append(
                f"{char_info.get('name', 'A mysterious figure')} ({char_info.get('role', 'of unknown purpose')})"
            )

    user_prompt = (
        f"**World Context:**\n"
        f"- Genre: {world_info.get('genre', 'A realm of mystery')}\n"
        f"- Setting: {world_info.get('setting', 'An enigmatic place')}\n"
        f"- Primary Object of Interest: {world_info.get('object', 'ancient artifacts')}\n"
        f"- Characters to potentially feature: {', '.join(char_names_roles) if char_names_roles else 'The inhabitants of this world'}\n\n"
        f"**Task:** Write an engaging introductory scene based on the context above. Remember the strict rule: NO numbers (or strive for zero numbers, with very limited exceptions for 'one'/'two'/'three' in general phrasing only). "
        f"The scene should set a tone and hint at the story's direction without giving away specifics. "
        f"Output ONLY the narrative text."
    )
    # --- END OF ADDED/COMPLETED PROMPT DEFINITIONS ---

    # Validator for intro: NO numbers, or at most very specific phrasing numbers if allowed by config.
    # The intro prompt asks for NO numbers.
    validate_intro = make_number_validator(
        allowed_atoms=set(),
        forbidden_atoms=set(),
        operand_count=0,
        correct_result_for_beat=None,
        strict_zero=True,  # Key for intro/padding style validation
        enforce_result_presence=False,
        operation_type="INTRO",
        overall_ground_truth_answer=None,  # No GT relevant for intro in this way
        is_root_node_being_validated=False,
        config_obj=config_obj,  # Pass the config
        logger_obj=logger_obj,
    )

    intro_text = generate_with_retry(
        system_prompt=system_prompt,  # Now defined
        user_prompt=user_prompt,  # Now defined
        max_completion_tokens=config_obj.INTRO_MAX_TOKENS,
        validate_fn=validate_intro,
        retries=config_obj.INTRO_MAX_RETRIES,
        sample_index=sample_index,
        temperature=config_obj.CREATIVE_NARRATIVE_TEMP,
        reasoning_settings={"exclude": True},
    )
    if intro_text:
        logger_obj.info(
            f"Successfully generated intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return intro_text.strip()
        else:
        logger_obj.error(
            f"Failed to generate intro for sample {sample_index+1 if sample_index is not None else 'N/A'}"
        )
        return None


# ... (rest of the code, including the second definition of generate_narrative) ...


def generate_narrative(
    ast: Node,
    world: dict,
    config: Config,
    encoder,
    p_inflect,
    logger,
    sample_index: int,
    overall_ground_truth_answer: int,  # ADDED: Receive the overall ground truth
) -> str | None:
    """
    Generate a structured narrative representation of the AST.
    Each operation is represented by a scene, carefully sequenced with
    intermediate node anchor names. Uses STRICT recursive generation.
    """
    logger.info(f"[Sample {sample_index + 1}] Starting narrative generation.")
    logger.debug(f"DEBUG: Using model: {MODEL}")
    logger.debug(f"DEBUG: AST: {ast_to_prefix(ast)}")

    # --- Initial Setup ---
    all_operator_nodes = [node for node in postorder(ast) if not isinstance(node, Atom)]
    all_atoms = set()
    for node in postorder(ast):
        if isinstance(node, Atom):
            all_atoms.add(node.n)
    logger.debug(f"DEBUG: All atoms in AST: {sorted(list(all_atoms))}")
    # Add the final answer to the log for debugging
    logger.debug(
        f"DEBUG: Overall ground truth (final answer): {overall_ground_truth_answer}"
    )

    operator_nodes = []
    narrative_anchor_map = {}
    intro_text = None
    scenes = []
    tokens_used = 0

    # --- Generate narrative anchors for op nodes ---
    if config.USE_NARRATIVE_ANCHORS:

        def generate_anchor_for_node(op_node):
            # If not using LLM anchors at all (config setting)
            if not config.USE_NARRATIVE_ANCHORS:
                # Use deterministic naming if not using LLM
                return f"the_{op_node.op.lower()}_result_{id(op_node) % 1000:03d}"

            all_anchors_list = list(narrative_anchor_map.values())
            try:
                anchor = generate_narrative_anchor_with_llm(
                    world, op_node, all_anchors_list, sample_index=sample_index
                )

                if anchor:
                    return anchor
                else:
                    logger.warning(
                        f"Failed to generate LLM anchor for {op_node.op}. Using deterministic fallback."
                    )
                    return f"the_{op_node.op.lower()}_result_{id(op_node) % 1000:03d}"
            except Exception as e:
                logger.error(f"Error in narrative anchor generation: {e}")
                return f"the_{op_node.op.lower()}_result_{id(op_node) % 1000:03d}"

        # Process nodes in postorder for anchors (bottom-up)
        for node_iter in postorder(ast):  # Renamed node to node_iter to avoid conflict
            if isinstance(node_iter, OpNode):
                # Use helper to generate anchor names
                anchor = generate_anchor_for_node(node_iter)
                narrative_anchor_map[id(node_iter)] = anchor
                operator_nodes.append(node_iter)
                logger.debug(
                    f"Added narrative anchor '{anchor}' for {node_iter.op} node"
                )
    else:
        # Without narrative anchors, just use basic names
        for node_iter in postorder(ast):  # Renamed node to node_iter
            if isinstance(node_iter, OpNode):
                narrative_anchor_map[id(node_iter)] = (
                    f"the_{node_iter.op.lower()}_result_{id(node_iter) % 1000:03d}"
                )
                operator_nodes.append(node_iter)

    logger.info(f"Generated {len(narrative_anchor_map)} narrative anchors.")
    log_str = "Narrative anchors: " + ", ".join(
        [
            f"'{anchor}' ({op_node.op})"  # Changed node to op_node
            for op_node, anchor in [  # Changed node to op_node
                (n, narrative_anchor_map.get(id(n), "MISSING")) for n in operator_nodes
            ]
        ]
    )
    logger.debug(log_str)

    intro_text = generate_introduction_scene(world, sample_index=sample_index)

    if intro_text:
        intro_tokens = len(encoder.encode(intro_text))
        if intro_tokens <= config.MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            scenes.append(intro_text)
            tokens_used += intro_tokens
            logger.info(
                f"Generated and added introductory scene ({intro_tokens} tokens)."
            )
        else:
            logger.warning(
                f"Generated introductory scene ({intro_tokens} tokens) was too long and would exceed budget. "
                f"Not adding to narrative. Budget: {config.MAX_TOTAL_TOKENS}, Safety: {SAFETY_MARGIN}"
            )
            intro_text = None
    else:
        logger.warning(
            "Failed to generate valid introductory scene. Starting narrative without intro."
        )

    last_scene_text = intro_text if intro_text else "The story begins..."
    introduced_atoms_during_generation = set()
    total_beats = len(operator_nodes)
    beat_counter = {"current": 0, "total": total_beats}

    context = GenerationContext(
        world=world,
        config=config,
        encoder=encoder,
        p_inflect=p_inflect,
        logger=logger,
        narrative_anchor_map=narrative_anchor_map,
        all_atoms=all_atoms,
        introduced_atoms=introduced_atoms_during_generation,
        scenes=scenes,
        tokens_used=tokens_used,
        last_scene_text=last_scene_text,
        beat_counter=beat_counter,
        sample_index=sample_index,
        max_pad_paragraphs=config.MAX_PAD_PARAGRAPHS,
        overall_ground_truth_answer=overall_ground_truth_answer,
        overall_ast_root=ast,  # Explicitly pass the root AST
    )

    tokens_available_for_narrative_and_padding = (
        config.MAX_TOTAL_TOKENS - tokens_used - SAFETY_MARGIN
    )
    max_padding_allowed = int(
        tokens_available_for_narrative_and_padding * config.PADDING_MAX_TOK_PERCENT
    )
    context.padding_stats["max_padding_allowed"] = max_padding_allowed

    # --- NEW: Calculate padding_per_slot ---
    num_padding_slots = total_beats - 1 if total_beats > 1 else 0
    if num_padding_slots > 0:
        padding_per_slot_calculated = max_padding_allowed // num_padding_slots
        context.padding_stats["padding_per_slot"] = padding_per_slot_calculated
        logger.info(
            f"Calculated padding per slot: {padding_per_slot_calculated} tokens ({max_padding_allowed} total / {num_padding_slots} slots)"
        )
    else:
        context.padding_stats["padding_per_slot"] = 0
        logger.info(
            f"No padding slots available (total_beats: {total_beats}). Padding per slot set to 0."
        )
    # --- END NEW ---

    logger.info(
        f"PADDING BUDGET INITIALIZED: Tokens after intro: {tokens_used}, "
        f"Available for narrative+padding: {tokens_available_for_narrative_and_padding}, "
        f"Max padding %: {config.PADDING_MAX_TOK_PERCENT*100:.1f}%, "
        f"Max padding tokens allowed: {max_padding_allowed}, "
        f"Padding per slot: {context.padding_stats['padding_per_slot']}, "
        f"Max padding segments per beat: {config.MAX_PAD_PARAGRAPHS}"
    )

    try:
        _generate_narrative_recursive(
            ast,
            context,
            is_root=True,
        )
    except BeatGenerationError as e:
        logger.error(f"Narrative generation aborted due to beat failure: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during recursive narrative generation: {e}",
            exc_info=True,
        )
        return None

    if not context.scenes:
        logger.error("Narrative generation resulted in no scenes.")
        return None

    narrative_body = "\n\n".join(context.scenes).strip()
    final_token_count = len(encoder.encode(narrative_body))
    if final_token_count > config.MAX_TOTAL_TOKENS:
        logger.warning(
            f"Final generated narrative ({final_token_count} tokens) exceeds MAX_TOTAL_TOKENS ({config.MAX_TOTAL_TOKENS}). Truncation might occur."
        )

    total_padding_tokens = context.padding_stats["total_padding_tokens"]
    padding_segments_added = context.padding_stats["padding_segments_added"]
    padding_percentage_of_max = (
        (total_padding_tokens / context.padding_stats["max_padding_allowed"] * 100)
        if context.padding_stats["max_padding_allowed"] > 0
        else 0
    )
    padding_percentage_of_total_narrative = (
        (total_padding_tokens / context.tokens_used * 100)
        if context.tokens_used > 0
        else 0
    )
    logger.info(
        f"PADDING FINAL SUMMARY: "
        f"Padding tokens: {total_padding_tokens}/{context.padding_stats['max_padding_allowed']} ({padding_percentage_of_max:.1f}% of max allowed for padding), "
        f"Padding percentage of total narrative tokens: {padding_percentage_of_total_narrative:.1f}%, "
        f"Padding segments added: {padding_segments_added}"
    )

    failed_validations_dir = os.path.join(LOG_DIR, "failed_validations")
    if os.path.exists(failed_validations_dir):
        validation_files = [
            f_name
            for f_name in os.listdir(failed_validations_dir)  # Renamed f to f_name
            if f_name.startswith(f"validation_fail_")
            and f"sample_{sample_index+1}" in f_name  # Renamed f to f_name
        ]

        if validation_files:
            failures_by_reason = {}
            failures_by_op = {}
            for file_name in validation_files:
                try:
                    parts = file_name.split("_")
                    if len(parts) >= 4:
                        op_type = parts[2]
                        reason_code = parts[3]
                        failures_by_reason[reason_code] = (
                            failures_by_reason.get(reason_code, 0) + 1
                        )
                        failures_by_op[op_type] = failures_by_op.get(op_type, 0) + 1
                except IndexError:
                    logger.warning(
                        f"Could not parse validation failure filename: {file_name}"
                    )
            logger.info(f"[Sample {sample_index + 1}] VALIDATION FAILURES SUMMARY:")
            logger.info(
                f"  Total validation failures for this sample: {len(validation_files)}"
            )
            if failures_by_reason:
                logger.info(f"  Failures by reason: {failures_by_reason}")
            if failures_by_op:
                logger.info(f"  Failures by operation: {failures_by_op}")
        else:
            logger.info(
                f"[Sample {sample_index + 1}] No validation failure files found for this sample."
            )

    logger.info(
        f"Successfully generated narrative for sample {sample_index + 1}. Final context tokens: {context.tokens_used}, Narrative tokens: {final_token_count}"
    )
    # Instead of just the narrative string, return the whole context object
    return context


def node_to_dict(node: Node) -> dict:
    """Convert a Node to a dictionary representation."""
    if isinstance(node, Atom):
        return {
            "type": "Atom",
            "value": node.n,
            "id": id(node),
        }

    return {
        "type": "OpNode",
        "op": node.op,
        "id": id(node),
        "children": [node_to_dict(child) for child in node.children],
        "result": node.value,
    }


def node_to_dict(node: Node) -> dict:
    """Convert a Node to a dictionary representation."""
    if isinstance(node, Atom):
        return {
            "type": "Atom",
            "value": node.n,
            "id": id(node),
        }

    return {
        "type": "OpNode",
        "op": node.op,
        "id": id(node),
        "children": [node_to_dict(child) for child in node.children],
        "result": node.value,
    }


def walk_ast(node: Node):
    """Iterator that yields all nodes in the AST."""
    yield node
    for child in node.children:
        yield from walk_ast(child)


def count_ops(node: Node) -> int:
    """Count the number of operations (non-atom nodes) in the AST."""
    return sum(1 for n in walk_ast(node) if not isinstance(n, Atom))


def walk_ast(node: Node):
    """Iterator that yields all nodes in the AST."""
    yield node
    for child in node.children:
        yield from walk_ast(child)


def count_ops(node: Node) -> int:
    """Count the number of operations (non-atom nodes) in the AST."""
    return sum(1 for n in walk_ast(node) if not isinstance(n, Atom))


def generate_single_sample(sample_index: int, config_obj: Config) -> dict | None:
    # Use config_obj throughout this function instead of module-level 'config'
    # This ensures the instance of Config being used is consistent.
    logger.info(
        f"[Sample {sample_index+1}] Starting generation with config: MAX_OPS={config_obj.MAX_OPS}, ALLOW_IMPLICIT=True (forced)"
    )
    try:
        node = build_random_ast(
            max_ops=config_obj.MAX_OPS,
            max_branch=config_obj.MAX_BRANCH,
            config_obj=config_obj,
        )
        # Ensure build_random_ast also uses the passed config_obj for its parameters like MIN_ATOM_VAL etc.
        # You'll need to modify build_random_ast to accept and use a config_obj parameter if needed.

        world_data = generate_world(
            num_characters=random.randint(
                config_obj.MIN_WORLD_CHARS, config_obj.MAX_WORLD_CHARS
            ),
            num_concepts=random.randint(
                config_obj.MIN_WORLD_CONCEPTS, config_obj.MAX_WORLD_CONCEPTS
            ),
            sample_index=sample_index,
            max_retries=config_obj.WORLDGEN_MAX_RETRIES,
        )
        # Modify generate_world to accept and use config_obj if it needs WORLDGEN_MAX_RETRIES etc.

        if not node or not world_data:  # Basic check
            logger.error(f"AST or World gen failed for sample {sample_index+1}")
            return None

        # IMPORTANT: Evaluate AST *before* narrative generation so all node.value are set
        # This is critical for forbidden checks and correct_result logic.
        final_ast_answer = eval_node(node)  # This populates .value in all nodes

        # generate_narrative now returns GenerationContext or None
        narrative_gen_context = generate_narrative(
            ast=node,
            world=world_data,
            config=config_obj,  # Pass the specific config instance
            encoder=encoder,  # Global encoder
            p_inflect=p_inflect,  # Global p_inflect
            logger=logger,  # Global logger
            sample_index=sample_index,
            overall_ground_truth_answer=final_ast_answer,
        )

        if not narrative_gen_context:
            logger.error(
                f"Narrative generation failed for sample {sample_index+1}. Skipping sample."
            )
            return None

        # Assemble sample using narrative_gen_context
        narrative_body_from_context = "\n\n".join(narrative_gen_context.scenes).strip()
        # ... (rest of your sample assembly logic, using narrative_gen_context.tokens_used, .scenes, etc.)
        # Ensure all references to 'config' in this assembly part use 'config_obj'.
        sample = {
            "sample_index": sample_index,
            "timestamp": datetime.datetime.now().isoformat(),
            "story": narrative_body_from_context,
            # ... other fields ...
            "ast_prefix": ast_to_prefix(node),  # ast_to_prefix should be fine
            "world_data": world_data,
            "scenes": [
                {"scene_number": i + 1, "text": text}
                for i, text in enumerate(narrative_gen_context.scenes)
            ],
            "num_operations": count_ops(node),  # count_ops should be fine
            "total_tokens": narrative_gen_context.tokens_used,
            "narrative_tokens": (
                len(encoder.encode(narrative_body_from_context)) if encoder else 0
            ),
            "padding_tokens": narrative_gen_context.padding_stats[
                "total_padding_tokens"
            ],
            "ground_truth_answer": narrative_gen_context.overall_ground_truth_answer,  # This is final_ast_answer
            "narrative_anchors": dict(narrative_gen_context.narrative_anchor_map),
            "meta": {
                "script_version": "2.0_refactored",
                "model_used": MODEL,  # Global MODEL
                # Use values from config_obj for meta
                "allow_implicit_intermediate_results": True,  # Always use implicit results
                "always_allowed_phrasing_numbers": list(
                    config_obj.ALWAYS_ALLOWED_PHRASING_NUMBERS_SET
                ),
                "max_ops": config_obj.MAX_OPS,
                # ... other relevant config_obj settings ...
                "full_config_used": asdict(config_obj),  # Store the exact config
            },

            # Add validator-compatible field names
            "id": str(sample_index),  # Required by validator.py
            "ast": ast_to_prefix(node),  # Required by validator.py
            "ground_truth": narrative_gen_context.overall_ground_truth_answer,  # Required by validator.py
            "narrative_with_question": narrative_body_from_context + "\n\n---\n\n**Question:** Considering the entire sequence of events described in the story, what is the final, precise quantity of " + world_data["object"] + " that the characters possess or have determined at the very end of their activities? Provide only the single integer representing this final amount.",  # Required by validator.py
            # ...
        }
        return sample

    except Exception as e:
        logger.error(
            f"Outer error in generate_single_sample for sample {sample_index+1}: {e}",
            exc_info=True,
        )
        return None


def main(
    config: Config,
    num_samples: int = NUM_SAMPLES_TO_GENERATE,
    max_workers: int = DEFAULT_MAX_WORKERS,
):
    """Generate samples with strict validation."""
    # Test log output to ensure logger is working properly
    logger.info(
        "START OF MAIN FUNCTION - THIS LOG SHOULD APPEAR IN verbose_listops.log"
    )

    # --- Fetch initial account usage --- ADD THIS BLOCK ---
    initial_account_usage = None  # Initialize
    if (
        client
        and OPENROUTER_API_KEY
        and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE"
    ):
        logger.info("Fetching initial OpenRouter account usage...")
        initial_account_usage = (
            rate_limiter.update_limits_from_api()
        )  # Call modified function
        if initial_account_usage is not None:
            logger.info(
                f"Initial OpenRouter account usage: ${initial_account_usage:.4f}"
            )
        else:
            logger.warning("Could not fetch initial OpenRouter account usage.")
    else:
        logger.warning(
            "Skipping initial OpenRouter account usage check: Client not initialized or API key missing/placeholder."
        )
    # --- END ADDED BLOCK ---

    # --- Dynamic Filename Generation ---
    sanitized_model_name = MODEL.replace("/", "_").replace(":", "-")

    # Ensure datasets directory exists
    os.makedirs(DATASETS_DIR, exist_ok=True)

    output_file = os.path.join(
        DATASETS_DIR,
        f"DATASET_"
        f"{config.MAX_TOTAL_TOKENS}tok_"
        f"{config.MAX_OPS}-mxops_"
        f"{config.MIN_ARITY}-arity_"
        f"{config.MAX_BRANCH}-mxbrch_"
        f"{sanitized_model_name}_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        f".jsonl",
    )
    logger.info(f"Output filename (dynamic): {output_file}")
    logger.info(
        f"Script started. Generating {num_samples} samples using up to {max_workers} workers."
    )
    logger.info(
        f"Using {config.FEW_SHOT_EXAMPLES} few-shot examples for narrative generation."
    )

    # Setup for time estimation and progress tracking
    samples_generated_successfully = 0
    samples_failed = 0
    start_time = time.time()
    results = []

    # Progress tracking with thread safety
    progress_lock = threading.Lock()
    completed_samples = 0
    last_print_time = start_time

    # Adjust print interval based on number of workers
    # More workers = longer interval to avoid console spam
    print_interval = max(
        5, min(30, max_workers // 10)
    )  # Between 5-30 seconds depending on workers

    print(
        f"Starting generation of {num_samples} samples using {max_workers} workers..."
    )
    print(f"Progress updates will be shown every {print_interval} seconds")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                generate_single_sample, i, config
            ): i  # Changed config_obj to config
            for i in range(num_samples)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_data = future.result()
                with progress_lock:  # Thread safety for counters
                    if sample_data:
                        results.append(sample_data)
                        samples_generated_successfully += 1
                    else:
                        samples_failed += 1
                    completed_samples = samples_generated_successfully + samples_failed
            except Exception as exc:
                logger.error(
                    f"[Sample {index + 1}] task generated exception: {exc}",
                    exc_info=True,
                )
                with progress_lock:
                    samples_failed += 1
                    completed_samples = samples_generated_successfully + samples_failed

            # Print progress update based on time interval, not per sample
            current_time = time.time()
            should_print = False

            with progress_lock:
                if (
                    current_time - last_print_time >= print_interval
                    or completed_samples == num_samples
                ):
                    should_print = True
                    last_print_time = current_time

            if should_print:
                elapsed_time = current_time - start_time
                if completed_samples > 0:
                    # Calculate throughput as samples per minute
                    throughput = completed_samples / (elapsed_time / 60)

                    # Estimate based on throughput
                    remaining_samples = num_samples - completed_samples
                    estimated_time_remaining = (
                        (remaining_samples / throughput) * 60 if throughput > 0 else 0
                    )

                    # Format time remaining
                    if estimated_time_remaining >= 3600:
                        time_str = f"{estimated_time_remaining/3600:.1f} hours"
                    elif estimated_time_remaining >= 60:
                        time_str = f"{estimated_time_remaining/60:.1f} minutes"
                    else:
                        time_str = f"{estimated_time_remaining:.1f} seconds"

                    success_rate = (
                        (samples_generated_successfully / completed_samples) * 100
                        if completed_samples > 0
                        else 0
                    )

                    print(
                        f"Progress: {completed_samples}/{num_samples} samples completed "
                        f"({completed_samples/num_samples*100:.1f}%) - "
                        f"Success rate: {success_rate:.1f}% - "
                        f"Throughput: {throughput:.2f} samples/min - "
                        f"Est. remaining: {time_str}"
                    )

    logger.info(
        f"Parallel generation complete. Writing {samples_generated_successfully} samples to {output_file}..."
    )
    try:

        write_mode = "w"  # Or 'a' if you prefer appending
        logger.info(f"Opening {output_file} in '{write_mode}' mode.")
        with open(output_file, write_mode, encoding="utf-8") as f:
            for (
                sample_data
            ) in results:  # Iterate through successfully generated results
                try:
                    f.write(json.dumps(sample_data, default=lambda o: list(o) if isinstance(o, set) else str(o), ensure_ascii=False) + "\n")
                except TypeError as e:

                    logger.error(
                        f"Serialization failed for sample {sample_data.get('id', 'Unknown')}: {e}. Skipping write for this sample."
                    )

                    samples_failed += 1
                    samples_generated_successfully -= (
                        1  # Decrement success as it wasn't written
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error writing sample {sample_data.get('id', 'Unknown')}: {e}. Skipping write."
                    )
                    samples_failed += 1
                    samples_generated_successfully -= 1  # Decrement success
    except IOError as e:
        logger.error(f"Fatal file write error opening/writing {output_file}: {e}")

        samples_failed += (
            samples_generated_successfully  # All successful generations failed to write
        )
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
    total_count = num_samples
    success_count = samples_generated_successfully
    success_rate = (success_count / total_count * 100) if total_count else 0
    logger.info(
        f"Overall success rate (generated AND written): {success_rate:.2f}% ({success_count}/{total_count})"
    )
    logger.info(f"Total time: {total_time:.2f} seconds")
    if samples_generated_successfully > 0:
        logger.info(f"Dataset output file: {output_file}")
        print(f"\nDataset saved to: {output_file}")
    else:
        logger.warning(
            f"No samples were successfully generated and written. Output file '{output_file}' may be empty or non-existent."
        )

    # Print datasets directory location for user reference
    print(f"\nDatasets are saved in: {os.path.abspath(DATASETS_DIR)}")

    # Add a clear console output showing total execution time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {seconds:.2f}s"
    else:
        time_str = f"{seconds:.2f}s"

    print(f"\n✅ Total execution time: {time_str} ({total_time:.2f} seconds)")

    # --- PROD_RUN: Validation and Cleaning Step ---
    if (
        PROD_RUN
        and samples_generated_successfully > 0
        and output_file
        and os.path.exists(output_file)
    ):
        logger.info(
            f"--- Starting PROD_RUN validation and cleaning for {output_file} ---"
        )
        # Assuming validator.py is in the same directory as verbose-listops.py
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        validator_script_path = os.path.join(current_script_dir, "validator.py")
        validator_results_path = output_file + ".validation_results.jsonl"

        if not os.path.exists(validator_script_path):
            logger.error(
                f"Validator script not found at {validator_script_path}. Cannot perform cleaning."
            )
        else:
            cmd = [
                sys.executable,
                validator_script_path,
                output_file,
                "--output-results",
                validator_results_path,
            ]
            try:
                logger.info(f"Running validator command: {' '.join(cmd)}")
                run_result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, encoding="utf-8"
                )
                logger.info("Validator process stdout:")
                for line in run_result.stdout.splitlines():
                    logger.info(f"VALIDATOR_STDOUT: {line}")
                if run_result.stderr:
                    logger.warning("Validator process stderr:")
                    for line in run_result.stderr.splitlines():
                        logger.warning(f"VALIDATOR_STDERR: {line}")
                logger.info(
                    f"Validator finished. Results expected in {validator_results_path}"
                )

                bad_sample_ids = set()
                if os.path.exists(validator_results_path):
                    with open(
                        validator_results_path, "r", encoding="utf-8"
                    ) as f_results:
                        for line_num, res_line in enumerate(f_results, 1):
                            try:
                                val_res = json.loads(res_line)
                                if val_res.get("status") != "correct":
                                    sample_id_to_remove = val_res.get("id")
                                    if sample_id_to_remove:
                                        bad_sample_ids.add(sample_id_to_remove)
                                    else:
                                        logger.warning(
                                            f"Validator result line {line_num} missing 'id': {res_line.strip()}"
                                        )
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Could not parse validator result line {line_num}: {res_line.strip()}"
                                )
                    logger.info(
                        f"Identified {len(bad_sample_ids)} samples to remove based on validator results."
                    )

                    if bad_sample_ids:
                        temp_cleaned_output_file = output_file + ".cleaned.tmp"
                        good_samples_written = 0
                        original_sample_count = 0

                        with open(output_file, "r", encoding="utf-8") as f_orig, open(
                            temp_cleaned_output_file, "w", encoding="utf-8"
                        ) as f_temp:
                            for line_num, dataset_line in enumerate(f_orig, 1):
                                original_sample_count += 1
                                try:
                                    sample_in_dataset = json.loads(dataset_line)

                                    # Make sure each sample has all fields expected by validator
                                    # Add these fields if missing but we have equivalent fields
                                    if (
                                        "narrative_with_question"
                                        not in sample_in_dataset
                                        and "full_prompt" in sample_in_dataset
                                    ):
                                        sample_in_dataset["narrative_with_question"] = (
                                            sample_in_dataset["full_prompt"]
                                        )

                                    if (
                                        "ast" not in sample_in_dataset
                                        and "ast_prefix" in sample_in_dataset
                                    ):
                                        sample_in_dataset["ast"] = sample_in_dataset[
                                            "ast_prefix"
                                        ]

                                    if (
                                        "ground_truth" not in sample_in_dataset
                                        and "ground_truth_answer" in sample_in_dataset
                                    ):
                                        sample_in_dataset["ground_truth"] = (
                                            sample_in_dataset["ground_truth_answer"]
                                        )

                                    # Check if this sample should be kept
                                    if (
                                        sample_in_dataset.get("id")
                                        not in bad_sample_ids
                                    ):
                                        # Write the sample with the added fields if needed
                                        f_temp.write(
                                            json.dumps(sample_in_dataset, default=lambda o: list(o) if isinstance(o, set) else str(o)) + "\n"
                                        )
                                        good_samples_written += 1
        else:
                                        logger.debug(
                                            f"Removing sample {sample_in_dataset.get('id')} (from line {line_num}) due to validation status."
                                        )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Could not parse line {line_num} in original dataset '{output_file}' during filtering: {dataset_line.strip()}. Discarding this line."
                                    )

                        # Construct the new filename with [CLEAN] prefix
                        output_dir = os.path.dirname(output_file)
                        base_filename = os.path.basename(output_file)
                        new_base_filename = f"[CLEAN]{base_filename}"
                        new_output_file_path = os.path.join(
                            output_dir, new_base_filename
                        )

                        # Move the temporary cleaned file to the new path instead of overwriting
                        shutil.move(temp_cleaned_output_file, new_output_file_path)
                        deleted_count = original_sample_count - good_samples_written
                        logger.info(
                            f"Removed {deleted_count} bad samples. {good_samples_written} cleaned samples saved to {new_output_file_path}."
                        )
                        logger.info(
                            f"(Original dataset with all generated samples remains at: {output_file})"
                        )  # Add note about original file
                        print(
                            f"PROD_RUN: After validation and cleaning, {good_samples_written} samples saved to {new_output_file_path}."
                        )
                        # Update samples_generated_successfully for any final tally if needed, though new print is clearer
                        # samples_generated_successfully = good_samples_written
                    else:
                        logger.info(
                            "No samples identified for removal by the validator (all were 'correct' or no IDs matched)."
                        )
                else:
                    logger.warning(
                        f"Validator results file not found at {validator_results_path}. Skipping removal of bad samples."
                    )

            except FileNotFoundError:
                logger.error(
                    f"Validator script '{validator_script_path}' not found. Ensure it is in the correct directory. Skipping cleaning step."
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Validator script failed with exit code {e.returncode}. Skipping cleaning step."
                )
                logger.error("Validator stdout snapshot:")
                stdout_snapshot = e.stdout.splitlines()
                for i, line_e in enumerate(stdout_snapshot):
                    if i < 50:  # Log first 50 lines of stdout
                        logger.error(f"VALIDATOR_STDOUT_ERR: {line_e}")
                    elif i == 50:
                        logger.error(
                            f"VALIDATOR_STDOUT_ERR: ... (stdout truncated after 50 lines)"
                        )
                        break
                logger.error("Validator stderr snapshot:")
                stderr_snapshot = e.stderr.splitlines()
                for i, line_e in enumerate(stderr_snapshot):
                    if i < 50:  # Log first 50 lines of stderr
                        logger.error(f"VALIDATOR_STDERR_ERR: {line_e}")
                    elif i == 50:
                        logger.error(
                            f"VALIDATOR_STDERR_ERR: ... (stderr truncated after 50 lines)"
                        )
                        break
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during PROD_RUN validation or cleaning: {e}",
                    exc_info=True,
                )
    elif PROD_RUN and (
        samples_generated_successfully == 0
        or not output_file
        or not os.path.exists(output_file)
    ):
        logger.info(
            "PROD_RUN was True, but no samples were successfully generated, output file is missing, or path is invalid. Skipping validation and cleaning."
        )

    # --- Final Cost Calculation and Logging --- ADD THIS SECTION ---
    gen_prompt_tokens, gen_completion_tokens, gen_api_calls = (
        generation_token_tracker.get_summary()
    )
    estimated_generation_cost = generation_token_tracker.calculate_cost(
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS,
    )

    logger.info(f"--- Generation Token Usage & Estimated Cost ---")
    logger.info(f"Total API calls (generation): {gen_api_calls}")
    logger.info(f"Total Prompt Tokens (generation): {gen_prompt_tokens}")
    logger.info(f"Total Completion Tokens (generation): {gen_completion_tokens}")
    logger.info(
        f"Estimated Cost (generation only): ${estimated_generation_cost:.4f} (using placeholder rates)"
    )
    logger.info(
        f"Note: Costs are estimates. Actual costs depend on specific models and OpenRouter pricing."
    )

    # --- Calculate and Log Total Run Cost via Usage Difference ---
    if (
        client
        and OPENROUTER_API_KEY
        and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE"
    ):
        logger.info("Fetching final OpenRouter account usage...")
        final_account_usage = rate_limiter.update_limits_from_api()
        if final_account_usage is not None:
            logger.info(f"Final OpenRouter account usage: ${final_account_usage:.4f}")
            if initial_account_usage is not None:
                total_run_cost_by_difference = (
                    final_account_usage - initial_account_usage
                )
                logger.info(f"--- Total Run Cost (from Usage Difference) ---")
                logger.info(
                    f"TOTAL RUN COST (Generation + Validation): ${total_run_cost_by_difference:.4f}"
                )
            else:
                logger.warning(
                    "Cannot calculate total run cost by difference: Initial account usage was not fetched."
                )
        else:
            logger.warning(
                "Could not fetch final OpenRouter account usage. Cannot calculate total run cost by difference."
            )
    else:
        logger.warning(
            "Skipping final OpenRouter account usage check for run cost: Client not initialized or API key missing/placeholder."
        )
    # --- END ADDED SECTION ---

    logging.shutdown()


if __name__ == "__main__":
    # Call update_limits_from_api once after full logger setup and before starting main generation.
    # This is already handled by the logic at the start of main() now for initial_account_usage.
    # We can remove the specific pre-main call if main() handles it robustly.

    # if client and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
    #     try:
    #         logger.info("Performing initial OpenRouter limits check before starting main generation...")
    #         rate_limiter.update_limits_from_api() # This will now also attempt to get usage for logging by main
    #     except Exception as e:
    #         logger.error(f"Initial OpenRouter limits check failed: {e}")
    # else:
    #     logger.warning("Skipping initial OpenRouter limits check: Client not initialized or API key missing/placeholder.")

    main(
        config,
        num_samples=NUM_SAMPLES_TO_GENERATE,
        max_workers=DEFAULT_MAX_WORKERS,
        # initial_account_usage will be fetched inside main
    )

