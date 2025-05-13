import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import threading
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

# --- Initial configuration logging ---
# Replace lines 9-31 with more concise setup info
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Log API key status (safely, without exposing the key)
if OPENROUTER_API_KEY:
    key_preview = (
        OPENROUTER_API_KEY[:4] + "..." + OPENROUTER_API_KEY[-4:]
        if len(OPENROUTER_API_KEY) > 8
        else "***"
    )
    print(f"API key: {key_preview} ✓")
else:
    print("⚠️ WARNING: OpenRouter API key not found!")

# fmt: off

# Recommended to use a fast and capable model for validation.
MODEL_FOR_VALIDATION = os.environ.get("VALIDATION_MODEL", "google/gemini-2.5-pro-preview")
MAX_WORKERS = int(os.environ.get("VALIDATION_MAX_WORKERS", 100))
LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_LEVEL = logging.INFO  # Use INFO or WARNING for console to reduce verbosity
# Default dataset path, can be overridden by command-line argument
DEFAULT_DATASET_FILE_PATH = "datasets/DATASET_10000tok_8-mxops_3-arity_6-mxbrch_google_gemini-2.5-pro-preview-03-25_20250508-1414.jsonl"
# This default is now handled by argparse

# --- Rate Limiter Configuration (for validator.py) ---
VALIDATION_MAX_REQUESTS_PER_SECOND = 900.0
VALIDATION_MIN_REQUEST_INTERVAL = 0.001
VALIDATION_BUCKET_CAPACITY = 100
VALIDATION_LIMITS_CHECK_INTERVAL = 5  # seconds
VALIDATION_JITTER = 0.01

# --- Token Costing (Placeholders - User should configure based on models used) ---
DEFAULT_COST_PER_MILLION_PROMPT_TOKENS_VALIDATOR = 0.50  # e.g., $0.50 / 1M prompt tokens
DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS_VALIDATOR = 1.50 # e.g., $1.50 / 1M completion tokens

# fmt: on

# --- Logging Setup ---
# Configure root logger
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)

# Create and configure file handler for detailed logs
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/validator_detailed.log")
file_handler.setLevel(LOG_LEVEL)
file_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s"
)
file_handler.setFormatter(file_format)

# Create and configure console handler for concise logs
console_handler = logging.StreamHandler()
console_handler.setLevel(CONSOLE_LOG_LEVEL)
console_format = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_format)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# --- Token Cost Tracker (same as in verbose-listops.py) ---
class TokenCostTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_api_calls += 1
        logger.debug(
            f"ValidatorTokenTracker: Added {prompt_tokens} prompt, {completion_tokens} completion. Call #{self.total_api_calls}. Totals: P={self.total_prompt_tokens}, C={self.total_completion_tokens}"
        )

    def get_summary(self) -> tuple[int, int, int]:
        return (
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_api_calls,
        )

    def calculate_cost(
        self, cost_per_million_prompt: float, cost_per_million_completion: float
    ) -> float:
        prompt_cost = (self.total_prompt_tokens / 1_000_000) * cost_per_million_prompt
        completion_cost = (
            self.total_completion_tokens / 1_000_000
        ) * cost_per_million_completion
        return prompt_cost + completion_cost


# Global instance for tracking token usage during validation
validation_token_tracker = TokenCostTracker()

# --- OpenAI Client for OpenRouter ---
client = None
try:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        logger.error(
            "OpenRouter API key missing or invalid. Set OPENROUTER_API_KEY in .env file."
        )
    else:
        logger.debug(f"Initializing client for model: {MODEL_FOR_VALIDATION}...")
        client = OpenAI(
            api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
        )
        logger.info(f"Client initialized for {MODEL_FOR_VALIDATION}")
except Exception as e:
    logger.error(f"Failed to initialize client: {e}")


# --- Rate Limiter Class (copied from verbose-listops.py and adapted) ---
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
        self.min_interval = min_interval
        self.bucket_capacity = bucket_capacity
        self.jitter = jitter
        self.tokens = float(bucket_capacity)  # Ensure tokens is float
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        self.last_limits_check_time = 0.0  # Ensure float
        self.limits_check_interval = VALIDATION_LIMITS_CHECK_INTERVAL  # Use constant

        logger.debug(
            f"Rate limiter initialized for validator: {self.max_requests_per_second} req/s, "
            f"{self.min_interval}s min interval, bucket capacity {self.bucket_capacity}, jitter {self.jitter}"
        )

    def wait_if_needed(self):
        current_time = time.time()
        if current_time - self.last_limits_check_time > self.limits_check_interval:
            self.update_limits_from_api()

        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_refill_time
            new_tokens = elapsed * self.max_requests_per_second
            self.tokens = min(float(self.bucket_capacity), self.tokens + new_tokens)
            self.last_refill_time = current_time

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            wait_time = (1.0 - self.tokens) / self.max_requests_per_second
            wait_time = max(wait_time, self.min_interval)
            if self.jitter > 0:
                wait_time += random.uniform(0, self.jitter)

            time.sleep(wait_time)
            self.tokens = 0.0
            self.last_refill_time = time.time()
            return wait_time

    def update_limits_from_api(self):
        if (
            not OPENROUTER_API_KEY
            or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE"
        ):
            logger.warning(
                "[ValidatorRateLimiter] Cannot check OpenRouter limits: No valid API key"
            )
            return

        try:
            logger.debug(
                "[ValidatorRateLimiter] Checking OpenRouter rate limits and remaining credits..."
            )
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                account_data = data.get("data", {})
                rate_limit_info = account_data.get("rate_limit", {})
                current_rate_for_log = self.max_requests_per_second
                limit_adjusted = False

                if rate_limit_info:
                    requests_limit = rate_limit_info.get("requests")
                    interval_str = rate_limit_info.get(
                        "interval", ""
                    )  # Renamed to avoid conflict

                    if requests_limit and interval_str:
                        if interval_str.endswith("s") and interval_str[:-1].isdigit():
                            interval_seconds = int(interval_str[:-1])
                            if interval_seconds > 0:
                                rps = requests_limit / interval_seconds
                                # Use VALIDATION_MAX_REQUESTS_PER_SECOND for capping
                                new_rate = min(
                                    float(rps) * 0.8, VALIDATION_MAX_REQUESTS_PER_SECOND
                                )
                                if new_rate != self.max_requests_per_second:
                                    self.max_requests_per_second = new_rate
                                    limit_adjusted = True
                                    current_rate_for_log = new_rate

                usage = account_data.get("usage")
                limit_val = account_data.get("limit")  # Renamed to avoid conflict
                limit_remaining = account_data.get("limit_remaining")

                log_message_parts = [
                    f"OR Limits (Validator): Current RPS: {current_rate_for_log:.1f}"
                ]
                if limit_adjusted:
                    log_message_parts.append("(Adjusted)")
                if usage is not None:
                    log_message_parts.append(f"Usage: ${usage:.4f}")
                if limit_val is not None and limit_remaining is not None:
                    log_message_parts.append(
                        f"Credits: Rem ${limit_remaining:.4f} of ${limit_val:.4f}"
                    )
                    if (
                        limit_val
                        and limit_remaining
                        and (limit_remaining / limit_val < 0.2)
                    ):
                        old_self_rate = self.max_requests_per_second
                        self.max_requests_per_second = min(
                            self.max_requests_per_second, 10.0
                        )
                        if old_self_rate != self.max_requests_per_second:
                            log_message_parts.append(
                                f"LOW CREDITS - RPS reduced to {self.max_requests_per_second:.1f}!"
                            )

                # Log detailed info to debug, but only critical info to console
                logger.debug(", ".join(log_message_parts))

                # If there's a critical limit issue, log to console as warning
                if limit_adjusted or (
                    limit_val
                    and limit_remaining
                    and (limit_remaining / limit_val < 0.2)
                ):
                    logger.warning(
                        f"OpenRouter limits adjusted - RPS: {current_rate_for_log:.1f}"
                    )
            else:
                logger.warning(
                    f"[ValidatorRateLimiter] Failed to get OpenRouter account status: HTTP {response.status_code}"
                )
        except Exception as e:
            logger.error(
                f"[ValidatorRateLimiter] Error checking OpenRouter limits: {e}"
            )

        self.last_limits_check_time = time.time()


# Instantiate the RateLimiter for the validator
rate_limiter = RateLimiter(
    max_requests_per_second=VALIDATION_MAX_REQUESTS_PER_SECOND,
    min_interval=VALIDATION_MIN_REQUEST_INTERVAL,
    bucket_capacity=VALIDATION_BUCKET_CAPACITY,
    jitter=VALIDATION_JITTER,
)

# --- System prompt for validation ---
# Template-based JSON approach to force proper formatting
# In validator.py

VALIDATION_SYSTEM_PROMPT = """
You are an expert AI system meticulously validating samples from the Verbose ListOps benchmark. This benchmark tests narrative reasoning in LLMs by embedding hierarchical ListOps computations (MAX, MIN, MED, SUM, AVG, SM) within lengthy, coherent narratives.

The benchmark's key characteristics:
1. It embeds a nested ListOps computation task within a distracting but semantically coherent narrative.
2. The narrative unfolds in post-order (inside-out) evaluation of the ListOps AST.
3. Models must extract computational signals while filtering out relevant but task-irrelevant narrative content.
4. **Crucially, intermediate numerical results of operations are NOT explicitly stated in the narrative.** Instead, they are referenced conceptually by thematic names or phrases (e.g., "The Sunstone's Core," "the outcome of the first analysis"). The LLM being evaluated must perform the computation and track these values internally.
5. The narrative MUST explicitly state the direct atomic numerical inputs for the current operation step (as words), **with a special exception for MEDIAN operations where an input is also the result (see below).**

CRITICAL TASK: For EACH `ast_evaluation_steps` entry, you must verify:
1.  **Input Fidelity (`ast_inputs_verified` field):** Did the narrative segment for this operation correctly use ALL inputs specified by the AST for this node? This means:
    *   All direct atomic numbers for this step MUST be mentioned in the narrative for this step.
        *   **SPECIAL EXCEPTION FOR MEDIAN OPERATIONS:** If the `operation_type` is "MED", AND an atomic number from `inputs_from_ast` is IDENTICAL to the `result_from_ast` (the median value), then this specific number SHOULD NOT be mentioned in the narrative. In this specific case, for `ast_inputs_verified` to be true regarding this number, you verify its *absence* from the narrative itemization. All *other* atomic inputs for that MEDIAN step must still be present.
    *   For inputs that are results from child operations: The narrative MUST reference these conceptually (e.g., using the thematic name introduced when that child operation was described). You should assume this conceptual reference correctly carries forward the `result_from_ast` of that child operation when evaluating the current step's inputs.
    *   If the narrative describes an operation using a different set of inputs (e.g., ignores a conceptual reference, invents new numbers, or fails to mention required atomic inputs, considering the MEDIAN exception above), `ast_inputs_verified` MUST be `false`.
2.  **Result Handling (`intermediate_result_implicit` field):** For this operation step (unless it's the very final operation of the entire AST), did the narrative correctly AVOID stating the numerical `result_from_ast` and instead imply it or associate it with a conceptual name? This should be `true` for all intermediate steps. For the final AST operation, this field can be "N/A" if the result is not stated, or `false` if it is stated (leading to VALID_BUT_TRIVIAL).
3.  **Narrative Consistency (`narrative_consistent` field):** Based on the above, is the narrative segment for this step an accurate and clear representation of the AST operation, its *correct* inputs (atomic and conceptual), and its *implied* result (for intermediate steps)? If `ast_inputs_verified` is `false` or `intermediate_result_implicit` is `false` (for intermediate steps where it should be true), then `narrative_consistent` for this step MUST also be `false`.

CRITICAL: Your ENTIRE response must be ONLY a valid JSON object with NO additional text.
Use this exact template, replacing values in [SQUARE_BRACKETS] with your analysis:

{
  "id": "[SAMPLE_ID]",
  "overall_status": "[VALID or INVALID_NARRATIVE or VALID_BUT_TRIVIAL]",
  "final_ast_value": [INTEGER_RESULT],
  "matches_ground_truth": [true or false],
  "narrative_consistent": [true or false],
  "ast_evaluation_steps": [
    {
      "step": 1,
      "operation_node_description": "[Brief description of the AST node, e.g., (SUM 10 20 (MIN 5 8))]",
      "operation_type": "[MAX, MIN, SUM, AVG, MED, SM]",
      "inputs_from_ast": [LIST_OF_EXPECTED_NUMERICAL_INPUTS_FROM_AST_FOR_THIS_NODE],
      "result_from_ast": [INTEGER_RESULT_OF_THIS_OPERATION_ON_AST_INPUTS],
      "ast_inputs_verified": [true or false],
      "intermediate_result_implicit": [true or false or "N/A"],
      "narrative_consistent": [true or false],
      "itemization_complete": [true or false], // Still useful to know if all *other* atoms were itemized
      "explanation": "[BRIEF_EXPLANATION, detailing any input discrepancies or narrative issues for this step, explicitly note if MEDIAN exception applied]"
    }
    // Additional steps...
  ],
  "narrative_analysis": {
    "strengths": "[[STRENGTH1], [STRENGTH2], ...]",
    "weaknesses": "[[WEAKNESS1], [WEAKNESS2], ...]",
    "inconsistencies": "[[INCONSISTENCY1], [INCONSISTENCY2], ...]"
  },
  "detailed_reason": "[Detailed explanation of any issues found, especially if overall_status is not VALID]",
  "summary": "[One-sentence summary of the validation outcome]"
}

ListOps operators:
- MAX: Maximum value of inputs
- MIN: Minimum value of inputs
- MED: Median value (if even count, the lower of the two middle values after sorting)
- SUM: Sum of inputs
- AVG: Integer floor of the sum of inputs divided by the count of inputs
- SM: Sum of inputs modulo 10 (result is sum % 10)

Guidelines for `narrative_consistent` field in `ast_evaluation_steps`:
- Mark `true` if the narrative segment for this step accurately and clearly conveys the operation, its AST-defined inputs (atoms mentioned according to rules including MEDIAN exception, conceptual references used for prior results), and correctly implies the result without stating it numerically (for intermediate steps).
- Mark `false` if `ast_inputs_verified` is `false`, or if `intermediate_result_implicit` is `false` for an intermediate step where it should be true, or for significant ambiguity or misleading information.
- **FOR THE FINAL EVALUATION STEP ONLY:** The narrative is designed to lead up to the final answer but *NOT* explicitly state it. `intermediate_result_implicit` can be marked "N/A" or `true` for this final step if the result is not stated. If the narrative segment for this final step *does* explicitly state the numerical result of this final operation, then `narrative_consistent` for this final step MUST be marked `false`, and your `explanation` for this step must clearly state: "Final answer revealed in narrative." In this specific scenario (math correct, prior steps consistent, but final answer revealed), set `overall_status` to `VALID_BUT_TRIVIAL`.

DO NOT write any text outside of this JSON format.
"""

# --- JSON Schema for validation output ---
# In validator.py (typically near VALIDATION_SYSTEM_PROMPT)

VALIDATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "overall_status": {
            "type": "string",
            "enum": [
                "VALID",
                "INVALID_NARRATIVE",  # This will now encompass math errors if they lead to narrative inconsistency
                "VALID_BUT_TRIVIAL",
            ],
        },
        "final_ast_value": {
            "type": "integer",
            "description": "The ground truth result from evaluating the AST.",
        },
        "matches_ground_truth": {
            "type": "boolean",
            "description": "Does the 'final_ast_value' field match the 'ground_truth' provided in the input sample?",
        },
        "narrative_consistent": {
            "type": "boolean",
            "description": "Overall narrative consistency based on all steps.",
        },
        "ast_evaluation_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "integer",
                        "description": "Sequential step number in the evaluation",
                    },
                    "operation_node_description": {
                        "type": "string",
                        "description": "Brief description of the AST node being evaluated, e.g., (SUM 10 20 (MIN 5 8))",
                    },
                    "operation_type": {
                        "type": "string",
                        "enum": ["MAX", "MIN", "MED", "SUM", "AVG", "SM"],
                        "description": "Type of ListOps operation performed",
                    },
                    "inputs_from_ast": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Numerical input values for this step as defined by the AST (including resolved results from children)",
                    },
                    "result_from_ast": {
                        "type": "integer",
                        "description": "Numerical result of this operation step based on AST inputs",
                    },
                    "ast_inputs_verified": {
                        "type": "boolean",
                        "description": "Whether the narrative segment correctly used all inputs_from_ast (direct atoms mentioned, conceptual references for children ops used)",
                    },
                    "intermediate_result_implicit": {  # NEW FIELD
                        "type": [
                            "boolean",
                            "string",
                        ],  # boolean true/false, or string "N/A"
                        "description": "True if an intermediate result was NOT stated numerically. 'N/A' for the final AST operation step if its result is not stated.",
                    },
                    "narrative_consistent": {
                        "type": "boolean",
                        "description": "Whether the narrative segment for this step accurately and clearly conveys the operation, its AST-defined inputs, and its (implied for intermediate) result.",
                    },
                    "itemization_complete": {
                        "type": "boolean",
                        "description": "Whether the narrative fully itemizes all direct atomic numerical inputs for this step",
                    },
                    # "correct_aggregate_provided" is REMOVED
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of this step, highlighting any input fidelity issues, result handling, or narrative discrepancies",
                    },
                },
                "required": [
                    "step",
                    "operation_node_description",
                    "operation_type",
                    "inputs_from_ast",
                    "result_from_ast",
                    "ast_inputs_verified",
                    "intermediate_result_implicit",  # ADDED
                    "narrative_consistent",
                    "itemization_complete",
                    "explanation",
                ],
            },
        },
        "narrative_analysis": {
            "type": "object",
            "properties": {
                "strengths": {"type": "array", "items": {"type": "string"}},
                "weaknesses": {"type": "array", "items": {"type": "string"}},
                "inconsistencies": {"type": "array", "items": {"type": "string"}},
            },
        },
        "detailed_reason": {
            "type": "string",
            "description": "Overall detailed explanation if not VALID.",
        },
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the validation outcome.",
        },
    },
    "required": [
        "id",
        "overall_status",
        "final_ast_value",
        "matches_ground_truth",
        "narrative_consistent",
        "ast_evaluation_steps",
        "summary",
    ],
}


# --- Construct User Prompt ---
def construct_user_prompt(sample: dict) -> str:
    """
    Formats the sample data into a concise, structured user prompt for the LLM.
    Reads canonical field names as output by verbose-listops.py's DRY generate_single_sample.
    """
    sample_id = sample.get("id", "Unknown_ID_from_validator")
    ast_representation = sample.get(
        "ast_str", "AST_STR_MISSING_IN_SAMPLE"
    )  # EXPECTS 'ast_str'
    ground_truth_val = sample.get(
        "ground_truth_value", "GT_VALUE_MISSING_IN_SAMPLE"
    )  # EXPECTS 'ground_truth_value'

    # The validator LLM needs the full narrative context including the question.
    # verbose-listops.py now outputs this directly as 'full_text_for_eval'.
    narrative_with_question_for_validator = sample.get("full_text_for_eval", "")

    if not ast_representation or ast_representation == "AST_STR_MISSING_IN_SAMPLE":
        logger.warning(f"[{sample_id}] 'ast_str' field missing or empty in sample.")
    if (
        ground_truth_val is None or ground_truth_val == "GT_VALUE_MISSING_IN_SAMPLE"
    ):  # Check for None specifically
        logger.warning(
            f"[{sample_id}] 'ground_truth_value' field missing or empty in sample."
        )
    if not narrative_with_question_for_validator:
        logger.warning(
            f"[{sample_id}] 'full_text_for_eval' field missing or empty in sample. Validator LLM will get empty narrative."
        )

    user_prompt = f"""
Validate this ListOps dataset sample. Your response must be ONLY valid JSON:

ID: {sample_id}
AST: {ast_representation}
Expected Answer: {ground_truth_val if ground_truth_val not in [None, "GT_VALUE_MISSING_IN_SAMPLE"] else "N/A"}

Narrative (full version used for validation, including the question):
{narrative_with_question_for_validator}

Perform a detailed evaluation based on the VLOps methodology (implicit intermediate results, conceptual references):
1. Evaluate each step of the AST, showing your work. For each step, identify the operation type, its inputs (atomic and resolved conceptual references), and its AST-based result.
2. For each step, verify Input Fidelity: Did the narrative correctly use ALL AST-defined inputs (atomic numbers mentioned, conceptual references for prior results used correctly)?
3. For each step, verify Result Handling: For intermediate steps, was the numerical result correctly NOT stated and instead implied or associated with a conceptual name? For the final step, was the result not stated (or if stated, is it trivial)?
4. For each step, assess Narrative Consistency: Is the narrative segment an accurate and clear representation of the AST operation, its inputs, and its (implied) result?
5. Analyze the overall narrative for strengths, weaknesses, and inconsistencies.
6. Provide detailed reasoning for any issues found.

RESPOND ONLY WITH THE JSON OBJECT MATCHING THE TEMPLATE - NO OTHER TEXT.
"""
    # Note: The detailed template is in VALIDATION_SYSTEM_PROMPT. This user prompt just provides the data.
    return user_prompt


# --- API Call Function ---
def get_llm_response(sample: dict, sample_id: str) -> str | None:
    if not client:
        logger.error(f"[{sample_id}] Client not initialized.")
        return None

    user_prompt = construct_user_prompt(sample)

    # Basic retry mechanism
    for attempt in range(3):  # Retry up to 3 times
        try:
            # More concise logging - remove redundant info
            if attempt > 0:
                # Use logger for retries for consistency, print might be lost in worker threads
                logger.info(
                    f"Sample {sample_id}: Retry {attempt+1}/3 for LLM validation call."
                )

            rate_limiter.wait_if_needed()  # Added rate limiter call

            response = None  # Initialize response to None

            # Temporarily increase client logging level to reduce console output
            httpx_logger = logging.getLogger("httpx")
            original_level = httpx_logger.level
            httpx_logger.setLevel(logging.WARNING)  # Only show warnings and errors

            try:
                response = client.chat.completions.create(
                    model=MODEL_FOR_VALIDATION,
                    messages=[
                        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=65000,  # Consider making this configurable or smaller if full schema is too large
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "validation_result",
                            "strict": True,
                            "schema": VALIDATION_OUTPUT_SCHEMA,
                        },
                    },
                )
            except Exception as schema_error:
                logger.debug(
                    f"[{sample_id}] json_schema format not supported or failed: {schema_error}. Trying json_object."
                )
                try:
                    # Try simple json_object if json_schema is not supported
                    rate_limiter.wait_if_needed()
                    response = client.chat.completions.create(
                        model=MODEL_FOR_VALIDATION,
                        messages=[
                            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=65000,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                except Exception as object_error:
                    logger.debug(
                        f"[{sample_id}] json_object format not supported or failed: {object_error}. Falling back to standard call."
                    )
                    # Fall back to standard format with no response_format parameter
                    rate_limiter.wait_if_needed()
                    response = client.chat.completions.create(
                        model=MODEL_FOR_VALIDATION,
                        messages=[
                            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=65000,
                        temperature=0.0,
                    )
            finally:
                # Restore original logging level
                httpx_logger.setLevel(original_level)

            # Robust check for response and content
            if (
                response
                and response.choices
                and len(response.choices) > 0
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                llm_output = response.choices[0].message.content
                logger.debug(
                    f"[{sample_id}] Raw LLM response: '{llm_output[:200]}...'"
                )  # Log just the start

                # --- Track token usage for validator ---
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                    validation_token_tracker.add_usage(prompt_tokens, completion_tokens)
                else:
                    logger.warning(
                        f"[{sample_id}] No usage data found in validation API response."
                    )
                return llm_output
            else:
                logger.warning(
                    f"[{sample_id}] API call attempt {attempt + 1} returned None or malformed response. Response object: {response}"
                )
                # Continue to next retry attempt if response is not as expected

        except Exception as e:
            logger.warning(
                f"[{sample_id}] API call attempt {attempt + 1} failed with exception: {e}"
            )
            # Fallthrough to retry logic

        if attempt < 2:  # If not the last attempt
            sleep_time = 2**attempt  # Exponential backoff: 1, 2 seconds
            logger.info(
                f"[{sample_id}] Waiting {sleep_time}s before next validation API attempt."
            )
            time.sleep(sleep_time)
        else:
            logger.error(
                f"[{sample_id}] API call for validation failed after 3 attempts."
            )
            return None  # Explicitly return None after all retries fail

    return None  # Should be unreachable if loop logic is correct, but as a fallback.


# --- Parse LLM Response ---
def parse_llm_validation_output(llm_response_text: str, sample_id: str) -> dict | None:
    """
    Parse the LLM's JSON validation response into a Python dictionary.
    Uses multiple strategies to try to extract valid JSON from potentially malformed responses.
    """
    if not llm_response_text:
        logger.warning(f"[{sample_id}] LLM response was empty or None.")
        return None

    # Log what kind of response format we seem to be dealing with
    if llm_response_text.strip().startswith("{") and llm_response_text.strip().endswith(
        "}"
    ):
        logger.debug(f"[{sample_id}] Response appears to be in proper JSON format.")
    elif "```json" in llm_response_text:
        logger.debug(
            f"[{sample_id}] Response contains markdown-style JSON code blocks."
        )
    else:
        logger.debug(
            f"[{sample_id}] Response does not appear to be in JSON format. First 50 chars: {llm_response_text[:50]}"
        )

    # Try multiple strategies to extract valid JSON

    # Strategy 1: Look for JSON in markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(.+?)\s*```", llm_response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            parsed_validation = json.loads(json_str)
            logger.debug(
                f"[{sample_id}] Successfully parsed LLM validation response from markdown code block."
            )
            return parsed_validation
        except json.JSONDecodeError as e:
            logger.debug(
                f"[{sample_id}] Found markdown code block but failed to parse as JSON: {e}. Content: {json_str[:100]}... Trying other methods."
            )

    # Strategy 2: Look for JSON between curly braces (might be missing outer formatting)
    # Find the first { and the last } that might contain a complete JSON object
    brace_match = re.search(r"(\{.+\})", llm_response_text, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(1)
        try:
            parsed_validation = json.loads(json_str)
            logger.debug(
                f"[{sample_id}] Successfully parsed LLM validation response from braces extraction."
            )
            return parsed_validation
        except json.JSONDecodeError as e:
            logger.debug(
                f"[{sample_id}] Found content in braces but failed to parse as JSON: {e}. Content: {json_str[:100]}... Trying other methods."
            )

    # Strategy 3: Try to clean the text and parse as JSON
    # Strip potential markdown, extra spaces, etc.
    json_str = llm_response_text.strip()

    # Remove any triple backticks
    json_str = re.sub(r"```(?:json)?|```", "", json_str)

    # Remove potential leading/trailing text not part of JSON
    # Look for first { and last } to find potential JSON boundaries
    start_idx = json_str.find("{")
    end_idx = json_str.rfind("}")

    if start_idx >= 0 and end_idx > start_idx:
        json_str = json_str[start_idx : end_idx + 1]

        try:
            parsed_validation = json.loads(json_str)
            logger.debug(
                f"[{sample_id}] Successfully parsed LLM validation response after cleaning."
            )
            return parsed_validation
        except json.JSONDecodeError as e:
            logger.debug(
                f"[{sample_id}] Failed to parse cleaned JSON string: {e}. Content: {json_str[:100]}... Trying one more approach."
            )

    # Strategy 4: Try to fix common JSON formatting errors
    try:
        # Replace single quotes with double quotes (a common error)
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        # Fix potential trailing commas in arrays and objects
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        parsed_validation = json.loads(json_str)
        logger.debug(
            f"[{sample_id}] Successfully parsed LLM validation response after fixing common JSON errors."
        )
        return parsed_validation
    except json.JSONDecodeError as e:
        logger.error(
            f"[{sample_id}] Failed to parse LLM validation response as JSON: {e}"
        )
        logger.error(
            f"[{sample_id}] Raw response preview: {llm_response_text[:300]}..."
        )
        return None


# --- Process a Single Sample ---
def validate_sample(sample: dict) -> dict:
    sample_id = sample.get("id", "Unknown_ID")

    # Check for new canonical required fields that validator.py needs to operate
    required_fields = ["id", "ast_str", "ground_truth_value", "full_text_for_eval"]
    missing_fields = [field for field in required_fields if field not in sample]

    if missing_fields:
        logger.error(
            f"[{sample_id}] Sample is missing required fields for validation: {', '.join(missing_fields)}. Skipping."
        )
        return {
            "id": sample_id,
            "status": "error",
            "reason": f"Missing required fields for validation: {', '.join(missing_fields)}",
            "llm_response": None,
            "parsed_validation": None,
            "ground_truth_answer": sample.get(
                "ground_truth_value"
            ),  # Use the canonical name
        }

    # Ensure ground_truth_value is an integer
    try:
        if isinstance(sample["ground_truth_value"], (str, float)):
            sample["ground_truth_value"] = int(sample["ground_truth_value"])
    except (ValueError, TypeError):
        logger.error(
            f"[{sample_id}] Ground truth value '{sample['ground_truth_value']}' is not a valid integer. Skipping."
        )
        return {
            "id": sample_id,
            "status": "error",
            "reason": "Invalid ground_truth_value format in sample.",
            "llm_response": None,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_value"),
        }

    logger.debug(f"[{sample_id}] Validating sample...")
    llm_response_text = get_llm_response(sample, sample_id)

    if llm_response_text is None:
        logger.error(f"[{sample_id}] Failed to get LLM response.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "LLM call failed or returned no response.",
            "llm_response": None,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_value"),
        }

    parsed_validation = parse_llm_validation_output(llm_response_text, sample_id)

    if parsed_validation is None:
        logger.warning(
            f"[{sample_id}] Could not parse validation results from LLM response."
        )
        return {
            "id": sample_id,
            "status": "error",
            "reason": "Could not parse validation results from LLM response.",
            "llm_response": llm_response_text,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_value"),
        }

    overall_status_from_llm = parsed_validation.get(
        "overall_status", "UNDEFINED"
    ).upper()
    final_script_status = "error"

    if overall_status_from_llm == "VALID":
        final_script_status = "correct"
    elif overall_status_from_llm == "VALID_BUT_TRIVIAL":
        final_script_status = "trivial"
    elif overall_status_from_llm == "INVALID_NARRATIVE":
        final_script_status = "incorrect"
    else:
        final_script_status = "error"
        logger.warning(
            f"[{sample_id}] LLM returned overall_status: {overall_status_from_llm}, mapped to script error."
        )

    llm_reported_matches_ground_truth = parsed_validation.get(
        "matches_ground_truth", False
    )
    if final_script_status == "correct" and not llm_reported_matches_ground_truth:
        logger.warning(
            f"[{sample_id}] LLM reported VALID, but 'matches_ground_truth' was false. "
            f"LLM's final_ast_value: {parsed_validation.get('final_ast_value')} vs Sample GT: {sample.get('ground_truth_value')}."
        )

    reason_for_log = parsed_validation.get("summary", "No summary provided by LLM.")
    if final_script_status != "correct":
        reason_for_log = parsed_validation.get("detailed_reason", reason_for_log)
        save_detailed_validation_log(sample_id, sample, parsed_validation)
    else:
        reason_for_log = "Sample validated as correct by LLM."

    logger.debug(
        f"[{sample_id}] Final Script Status: {final_script_status.upper()}. LLM Overall Status: {overall_status_from_llm}."
    )

    if parsed_validation:
        for step_eval in parsed_validation.get("ast_evaluation_steps", []):
            if not step_eval.get("narrative_consistent", True):
                logger.debug(
                    f"[{sample_id}] Step {step_eval.get('step')} (Op: {step_eval.get('operation_type')}) "
                    f"flagged by LLM as narrative inconsistent: {step_eval.get('explanation', 'No details')}"
                )

    return {
        "id": sample_id,
        "status": final_script_status,
        "reason": reason_for_log,
        "llm_response": llm_response_text,
        "parsed_validation": parsed_validation,
        "ground_truth_answer": sample.get(
            "ground_truth_value"
        ),  # Use the canonical name
    }


# --- Save Detailed Validation Log ---
def save_detailed_validation_log(sample_id, sample, validation_result):
    """
    Save a detailed validation log for failed samples to assist with debugging.
    Uses canonical field names from the sample as output by verbose-listops.py.
    """
    try:
        log_dir = os.path.join("logs", "failed_validations")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"fail_validation_{sample_id}.json")

        # Use the canonical field names from the sample
        full_text_content = sample.get("full_text_for_eval", "")

        detailed_log = {
            "sample_input_to_validator": {
                "id": sample.get("id"),
                "ast_str": sample.get("ast_str"),
                "ground_truth_value": sample.get("ground_truth_value"),
                "full_text_for_eval_preview": (
                    full_text_content[:500] + "..."
                    if len(full_text_content) > 500
                    else full_text_content
                ),
            },
            "llm_validation_result": validation_result,
        }

        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(detailed_log, f, indent=2)
        logger.info(f"[{sample_id}] Saved detailed validation log to {log_file_path}")
    except Exception as e:
        logger.error(f"[{sample_id}] Failed to save detailed validation log: {e}")


def load_dataset(file_path: str) -> list:
    dataset = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Skipping invalid JSON line {line_num} in {file_path}: {e}. Line: '{line.strip()}'"
                    )
        logger.info(f"Loaded {len(dataset)} samples from {file_path}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        # Potentially raise the error or return empty list to be handled by caller
        # For now, it returns empty list and caller checks.
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
    return dataset


# --- Main Processing Logic ---
def run_validation_process(dataset_file_path: str, output_results_path: str | None):
    if not client:
        logger.error("⚠️ Client not initialized. Check API key.")
        return

    dataset = load_dataset(dataset_file_path)
    if not dataset:
        logger.warning(f"No samples loaded from dataset. Exiting.")
        return

    results = []
    total_samples = len(dataset)
    print(
        f"🔍 Validating {total_samples} samples using {MODEL_FOR_VALIDATION.split('/')[-1]}"
    )
    start_time = time.time()
    processed_count = 0

    # For time estimation
    sample_times = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="Validator"
    ) as executor:
        future_to_sample_id = {
            executor.submit(validate_sample, sample): sample.get("id", f"sample_{i}")
            for i, sample in enumerate(dataset)
        }

        for future in concurrent.futures.as_completed(future_to_sample_id):
            sample_id_for_log = future_to_sample_id[future]
            sample_start_time = time.time()

            try:
                result = future.result()
                results.append(result)

                # Track sample processing time for better estimates
                sample_times.append(time.time() - sample_start_time)

                # Only keep the last 10 samples for average calculation
                if len(sample_times) > 10:
                    sample_times.pop(0)
            except Exception as exc:
                logger.error(f"[{sample_id_for_log}] Exception: {exc}", exc_info=True)
                results.append(
                    {
                        "id": sample_id_for_log,
                        "status": "error",
                        "reason": f"Task exception: {exc}",
                        "llm_response": None,
                        "parsed_validation": None,
                        "ground_truth_answer": "N/A due to exception",
                    }
                )

            # Update progress
            processed_count += 1
            progress_pct = (processed_count / total_samples) * 100

            # Calculate time remaining
            elapsed = time.time() - start_time
            if sample_times:
                avg_time_per_sample = sum(sample_times) / len(sample_times)
                remaining_samples = total_samples - processed_count
                est_remaining_time = (
                    remaining_samples * avg_time_per_sample / MAX_WORKERS
                )

                # Format time remaining
                if est_remaining_time > 60:
                    time_str = f"{est_remaining_time/60:.1f} min"
                else:
                    time_str = f"{est_remaining_time:.0f} sec"

                # Progress bar (50 chars wide)
                bar_width = 40
                filled_width = int(progress_pct / 100 * bar_width)
                bar = "█" * filled_width + "░" * (bar_width - filled_width)

                # Inline progress update (overwrite previous line)
                print(
                    f"\r[{bar}] {progress_pct:.1f}% ({processed_count}/{total_samples}) ETA: {time_str}",
                    end="",
                )
            else:
                print(
                    f"\r({processed_count}/{total_samples}) {progress_pct:.1f}%", end=""
                )

    # Print newline after progress bar
    print("\n" + "=" * 40)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate time statistics
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_display = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
    elif minutes > 0:
        time_display = f"{int(minutes)}m {seconds:.1f}s"
    else:
        time_display = f"{seconds:.1f}s"

    print(f"✅ Validation completed in {time_display}\n")

    # Count results by category
    correct_count = sum(1 for r in results if r["status"] == "correct")
    incorrect_count = sum(1 for r in results if r["status"] == "incorrect")
    error_count = sum(1 for r in results if r["status"] == "error")
    trivial_count = sum(1 for r in results if r["status"] == "trivial")

    # Count by validation categories
    validation_categories = {
        "VALID": 0,
        "INVALID_MATH": 0,
        "INVALID_NARRATIVE": 0,
        "INVALID_MATH_AND_NARRATIVE": 0,
        "VALID_BUT_TRIVIAL": 0,
        "UNDEFINED": 0,  # For cases where we couldn't determine a category
    }

    for r in results:
        if r["parsed_validation"] and "overall_status" in r["parsed_validation"]:
            status = r["parsed_validation"]["overall_status"].upper()
            if status in validation_categories:
                validation_categories[status] += 1
            else:
                validation_categories["UNDEFINED"] += 1
        else:
            validation_categories["UNDEFINED"] += 1

    total_processed = len(results)

    if total_processed > 0:
        accuracy = (correct_count / total_processed) * 100 if total_processed > 0 else 0

        # Cleaner summary format with visual indicators
        print("\n📊 VALIDATION RESULTS")
        print("=" * 40)
        print(f"Dataset: {os.path.basename(dataset_file_path)}")
        print(f"Model:   {MODEL_FOR_VALIDATION.split('/')[-1]}")
        print(f"✓ Correct:   {correct_count}/{total_processed} ({accuracy:.1f}%)")
        print(
            f"✗ Incorrect: {incorrect_count}/{total_processed} ({(incorrect_count/total_processed)*100:.1f}%)"
        )
        print(
            f"◆ Trivial:   {trivial_count}/{total_processed} ({(trivial_count/total_processed)*100:.1f}%)"
        )
        print(
            f"⚠ Errors:    {error_count}/{total_processed} ({(error_count/total_processed)*100:.1f}%)"
        )

        # Show breakdown by category if relevant
        if (
            incorrect_count > 0 or error_count > 0 or trivial_count > 0
        ):  # Broaden condition to include trivial
            print("\n🔍 VALIDATION CATEGORY BREAKDOWN (from LLM validator)")
            for category, count in validation_categories.items():
                if count > 0:  # Only show non-zero categories
                    category_display = (
                        category.replace("INVALID_", "").title().replace("_", " ")
                    )
                    symbol = (
                        "✓"
                        if category == "VALID"
                        else (
                            "◆"
                            if category == "VALID_BUT_TRIVIAL"
                            else ("✗" if "INVALID" in category else "⚠")
                        )
                    )  # Symbol for trivial
                    print(
                        f"{symbol} {category_display}: {count} ({(count/total_processed)*100:.1f}%)"
                    )
    else:
        print("❌ No samples were processed successfully.")

    # If there were errors, provide a note about log files
    if (
        error_count > 0 or incorrect_count > 0 or trivial_count > 0
    ):  # Broaden condition for log note
        print("\n⚠️ Check logs/failed_validations/ for details on specific failures.")

    # --- Log Token Usage Summary for Validator ---
    val_prompt_tokens, val_completion_tokens, val_api_calls = (
        validation_token_tracker.get_summary()
    )
    estimated_validation_cost = validation_token_tracker.calculate_cost(
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS_VALIDATOR,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS_VALIDATOR,
    )
    logger.info(f"--- Validation Token Usage & Estimated Cost ---")
    logger.info(f"Total API calls (validation): {val_api_calls}")
    logger.info(f"Total Prompt Tokens (validation): {val_prompt_tokens}")
    logger.info(f"Total Completion Tokens (validation): {val_completion_tokens}")
    logger.info(
        f"Estimated Cost (validation only): ${estimated_validation_cost:.4f} (using placeholder rates)"
    )

    # Print a summary line for verbose-listops.py to parse
    print(
        f"VALIDATOR_TOKEN_USAGE_SUMMARY:prompt_tokens={val_prompt_tokens},completion_tokens={val_completion_tokens},api_calls={val_api_calls}"
    )

    # Write detailed results to output file if path is provided
    if output_results_path:
        try:
            with open(output_results_path, "w", encoding="utf-8") as f_out:
                for res_item in results:
                    f_out.write(json.dumps(res_item) + "\n")
            logger.info(f"Full validation results saved to {output_results_path}")
            print(f"Results saved to {os.path.basename(output_results_path)}")
        except Exception as e:
            logger.error(
                f"Failed to write validation results to {output_results_path}: {e}"
            )
            print(
                f"ERROR: Failed to write validation results to {output_results_path}: {e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Verbose ListOps dataset samples."
    )
    parser.add_argument(
        "dataset_file_path",
        nargs="?",
        default=DEFAULT_DATASET_FILE_PATH,
        help="Path to the .jsonl dataset file to validate. Defaults to a predefined path if not provided.",
    )
    parser.add_argument(
        "--output-results",
        help="Path to save the detailed validation results for all samples (JSONL).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_file_path):
        logger.error(f"Dataset file not found: '{args.dataset_file_path}'")
        print(f"Dataset file not found: '{args.dataset_file_path}'")
        print(
            "Usage: python validator.py <path_to_dataset.jsonl> [--output-results <path_for_results.jsonl>]"
        )
    elif not client:
        logger.error("Client not initialized. Check API key.")
    else:
        # Perform initial limits check if client is available
        if (
            rate_limiter
            and OPENROUTER_API_KEY
            and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE"
        ):
            try:
                logger.debug(
                    "Performing initial OpenRouter limits check before starting validation..."
                )
                rate_limiter.update_limits_from_api()
            except Exception as e_limits:  # Renamed to avoid conflict
                logger.error(f"Initial OpenRouter limits check failed: {e_limits}")
        else:
            logger.debug(
                "Skipping initial OpenRouter limits check: API key missing or invalid."
            )
        run_validation_process(args.dataset_file_path, args.output_results)
