# evaluator.py
import os
import json
import random
import datetime
import logging
import logging.handlers
import time
import re
import concurrent.futures
import threading
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
import tiktoken # For token counting if needed for prompts, though OpenRouter provides usage
from dotenv import load_dotenv
from tabulate import tabulate
import requests # For RateLimiter's OpenRouter status check

load_dotenv()

# --- Configuration ---
# Define the LLMs to test via OpenRouter
# Reasoning settings:
#   None: Default behavior for the model.
#   {"exclude": True}: Explicitly disable reasoning if the model supports it.
#   {"effort": "low" | "medium" | "high"}: For OpenAI 'o' series models or others supporting it.
EVAL_CONFIG = [
    {
        "model_id": "openai/gpt-4o",
        "display_name": "GPT-4o",
        "reasoning_settings": {"effort": "high"}, # Example for reasoning
        "max_completion_tokens": 50,
        "temperature": 0.0,
    },
    {
        "model_id": "anthropic/claude-3.5-sonnet",
        "display_name": "Claude 3.5 Sonnet",
        "reasoning_settings": None, # Example for default
        "max_completion_tokens": 50,
        "temperature": 0.0,
    },
    {
        "model_id": "google/gemini-2.5-pro-latest",
        "display_name": "Gemini 2.5 Pro",
        "reasoning_settings": None,
        "max_completion_tokens": 50,
        "temperature": 0.0,
    },
    {
        "model_id": "google/gemini-2.5-flash-preview:thinking", # Example of a specific variant
        "display_name": "Gemini 2.5 Flash (Thinking)",
        "reasoning_settings": {"effort": "medium"}, # Assuming it might support this
        "max_completion_tokens": 50,
        "temperature": 0.0,
    },
    # Add more models as needed
]

# Path to the generated dataset (JSONL format)
# Assumes the dataset file is in a 'datasets' subdirectory relative to this script
# And that it contains 'full_text_for_eval' and 'ground_truth_value'
DEFAULT_DATASET_FILENAME = "[2_EVAL_READY]_DATASET_10000tok_8mxops_4minarity_8mxbrch_google_gemini-2.5-flash-preview-thinking_20240729-120000.jsonl"
DEFAULT_DATASET_SUBDIR = "datasets/10000tok_8mxops_4minarity_8mxbrch_google_gemini-2.5-flash-preview-thinking_20240729-120000" # Example sub-dir

OUTPUT_DIR = "evaluation_results"
MAX_WORKERS = 10  # Number of concurrent API calls
RETRY_ATTEMPTS = 5
RETRY_INITIAL_DELAY = 1.0  # seconds
API_TIMEOUT = 120 # seconds for API calls

# System prompt to guide the LLM's answer format
# The {question} placeholder will be filled with the actual question from the dataset if needed,
# but 'full_text_for_eval' already contains the question.
SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert at reading comprehension and following instructions. "
    "Please answer the question based *only* on the provided text. "
    "Provide only the single integer as your answer. Do not include any other words, explanations, or punctuation."
)

# Cost estimation (example, adjust to OpenRouter's actual pricing for the models)
# These are placeholders; OpenRouter pricing varies per model.
# We will primarily rely on the usage reported by the API.
DEFAULT_COST_PER_MILLION_PROMPT_TOKENS = 0.50  # USD
DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS = 1.50  # USD

# --- Global Variables & Setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set. API calls will fail.")
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE" # Placeholder

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"Failed to initialize tokenizer: {e}. Token counts in prompts might be estimated differently.")
    tokenizer = None

client = None
if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
    try:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=API_TIMEOUT,
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client for OpenRouter: {e}")
else:
    print("OpenRouter client not initialized due to missing or placeholder API key.")

# --- Logging Setup (similar to verbose-listops.py) ---
LOG_DIR_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_logs")
os.makedirs(LOG_DIR_EVAL, exist_ok=True)

logger = logging.getLogger("llm_evaluator")
logger.setLevel(logging.DEBUG)
# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler for detailed logs
eval_log_file = os.path.join(LOG_DIR_EVAL, f"evaluation_run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
file_handler = logging.handlers.RotatingFileHandler(
    filename=eval_log_file,
    maxBytes=5 * 1024 * 1024, # 5 MB
    backupCount=3,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler for INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info(f"Logging initialized. Detailed logs at: {eval_log_file}")


# --- Token Tracker (adapted from verbose-listops.py) ---
class EvaluationTokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.api_calls = 0
        self.lock = threading.Lock()
        self.costs_per_model = {} # Tracks {'model_id': {'prompt_tokens': X, ... 'cost': Y}}

    def add_usage(self, model_id: str, prompt_tokens: int, completion_tokens: int, cost: float = 0.0):
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.api_calls += 1

            if model_id not in self.costs_per_model:
                self.costs_per_model[model_id] = {'prompt_tokens': 0, 'completion_tokens': 0, 'api_calls': 0, 'estimated_cost': 0.0}
            
            self.costs_per_model[model_id]['prompt_tokens'] += prompt_tokens
            self.costs_per_model[model_id]['completion_tokens'] += completion_tokens
            self.costs_per_model[model_id]['api_calls'] += 1
            # Note: OpenRouter API response for usage doesn't directly give cost.
            # We'd need to fetch model pricing. For now, this is a rough estimate.
            # A more accurate way is to check account balance before/after.
            model_specific_prompt_cost = float(os.getenv(f"{model_id.upper().replace('/', '_').replace('-', '_')}_PROMPT_COST_PM", DEFAULT_COST_PER_MILLION_PROMPT_TOKENS))
            model_specific_completion_cost = float(os.getenv(f"{model_id.upper().replace('/', '_').replace('-', '_')}_COMPLETION_COST_PM", DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS))
            
            current_call_cost = (prompt_tokens / 1_000_000 * model_specific_prompt_cost) + \
                                (completion_tokens / 1_000_000 * model_specific_completion_cost)
            self.costs_per_model[model_id]['estimated_cost'] += current_call_cost

        logger.debug(
            f"Token Tracker ({model_id}): Added P:{prompt_tokens}, C:{completion_tokens}. Cost: ${current_call_cost:.4f}. "
            f"Model Totals - P:{self.costs_per_model[model_id]['prompt_tokens']}, C:{self.costs_per_model[model_id]['completion_tokens']}, Calls:{self.costs_per_model[model_id]['api_calls']}, Est.Cost:${self.costs_per_model[model_id]['estimated_cost']:.4f}"
        )

    def get_summary(self):
        with self.lock:
            total_estimated_cost = sum(data['estimated_cost'] for data in self.costs_per_model.values())
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_api_calls": self.api_calls,
                "total_estimated_cost": total_estimated_cost,
                "details_per_model": self.costs_per_model.copy()
            }

evaluation_token_tracker = EvaluationTokenTracker()

# --- Rate Limiter (adapted from verbose-listops.py) ---
class RateLimiter:
    def __init__(self, max_requests_per_second: float = 10.0, min_interval: float = 0.1, bucket_capacity: int = 10, jitter: float = 0.05):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = min_interval
        self.bucket_capacity = bucket_capacity
        self.jitter = jitter
        self.tokens = bucket_capacity
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        self.last_limits_check_time = 0
        self.limits_check_interval = 60 # Check OpenRouter limits less frequently for eval
        self.initial_usage = None

        logger.info(
            f"Rate limiter initialized: {max_requests_per_second} req/s, "
            f"{min_interval}s min interval, bucket capacity {bucket_capacity}, jitter {jitter}"
        )

    def wait_if_needed(self):
        current_time = time.time()
        # if current_time - self.last_limits_check_time > self.limits_check_interval: # Disabling for now, can be noisy
        #     self.update_limits_from_api()

        with self.lock:
            current_time = time.time() # Re-fetch time inside lock
            elapsed = current_time - self.last_refill_time
            new_tokens = elapsed * self.max_requests_per_second
            self.tokens = min(self.bucket_capacity, self.tokens + new_tokens)
            self.last_refill_time = current_time

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            wait_time = (1.0 - self.tokens) / self.max_requests_per_second
            wait_time = max(wait_time, self.min_interval)
            if self.jitter > 0:
                wait_time += random.uniform(0, self.jitter)
            
            logger.debug(f"Rate limiting: waiting for {wait_time:.3f}s")
            time.sleep(wait_time)
            self.tokens = 0.0 
            self.last_refill_time = time.time()
            return wait_time

    def update_limits_from_api(self): # Simplified from generator
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
            logger.warning("Cannot check OpenRouter limits: No valid API key")
            return None

        try:
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                usage = data.get("usage")
                if usage is not None:
                    current_usage = float(usage)
                    if self.initial_usage is None:
                        self.initial_usage = current_usage
                        logger.info(f"Initial OpenRouter usage set to: ${self.initial_usage:.4f}")
                    
                    run_cost = current_usage - (self.initial_usage if self.initial_usage is not None else current_usage)
                    logger.info(f"OpenRouter Status: Current Usage: ${current_usage:.4f}, Cost this run (approx): ${run_cost:.4f}")
                    return current_usage
                else:
                    logger.info(f"OpenRouter Status: No usage data in response. Data: {data}")
            else:
                logger.warning(f"Failed to get OpenRouter account status: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking OpenRouter limits: {e}")
        self.last_limits_check_time = time.time()
        return None

rate_limiter_global = RateLimiter()

# --- Core Evaluation Functions ---

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Loads dataset from a JSONL file."""
    dataset = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.error(f"Skipping malformed JSON line in {filepath}: {line.strip()}")
        logger.info(f"Loaded {len(dataset)} samples from {filepath}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {filepath}")
    except Exception as e:
        logger.error(f"Error loading dataset from {filepath}: {e}")
    return dataset

def extract_answer_from_llm_response(response_text: str) -> Optional[int]:
    """
    Extracts a single integer answer from the LLM's response text.
    The prompt asks for "only the single integer".
    """
    if not response_text:
        return None
    
    # Remove potential markdown code blocks if LLM wraps output
    response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()
    response_text = re.sub(r"```text\s*|\s*```", "", response_text).strip()
    response_text = re.sub(r"```\s*|\s*```", "", response_text).strip()

    # Try to find a single integer, possibly negative.
    # This regex looks for an optional minus sign followed by digits,
    # ensuring it's a standalone number (bounded by non-digits or start/end of string).
    matches = re.findall(r'(?<!\d)-?\d+(?!\d)', response_text)
    
    # If the LLM just gives the number, it might be the only content.
    # Or if it says "The answer is -5", matches will be ['-5'].
    # If it says "The answer is -5.", matches will be ['-5'].
    # If it says "-5 is the answer", matches will be ['-5'].

    if len(matches) == 1:
        try:
            return int(matches[0])
        except ValueError:
            logger.warning(f"Found single number-like string '{matches[0]}' but failed to convert to int.")
            return None
    elif len(matches) > 1:
        # The prompt asked for a *single* integer. Multiple integers means the LLM failed to follow instructions.
        # However, sometimes LLMs might restate the question or numbers from the prompt.
        # A simple heuristic: if the *entire string* is just a number, that's a strong candidate.
        try:
            cleaned_response = response_text.strip().rstrip('.') # Remove trailing period
            return int(cleaned_response)
        except ValueError:
            # If the whole string isn't an int, and we have multiple matches, it's ambiguous.
            logger.warning(f"Multiple numbers found in response, and response is not a single number: '{response_text}'. Matches: {matches}. Ambiguous.")
            return None
    else: # No numbers found
        logger.debug(f"No integer found in response: '{response_text}'")
        return None

def _api_call_with_retry(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_settings: Optional[Dict[str, Any]],
    sample_id: str
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """
    Makes an API call with retries.
    Returns (response_text, prompt_tokens, completion_tokens, error_message).
    """
    if not client:
        return None, None, None, "OpenRouter client not initialized."

    for attempt in range(RETRY_ATTEMPTS):
        try:
            rate_limiter_global.wait_if_needed()
            
            messages = [{"role": "user", "content": user_prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            api_params = {
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            extra_body = {}
            if reasoning_settings:
                # Check if model is OpenAI 'o' series for 'effort'
                is_openai_o_series = "openai/" in model_id and (re.search(r"/gpt-4o", model_id))
                
                current_reasoning_config = reasoning_settings.copy()
                if "effort" in current_reasoning_config and not is_openai_o_series:
                    logger.debug(f"Model {model_id} is not an OpenAI 'o' series. Removing 'effort' from reasoning.")
                    del current_reasoning_config["effort"]
                
                if current_reasoning_config: # If anything left in reasoning
                     extra_body["reasoning"] = current_reasoning_config
            
            logger.debug(f"API Call (Sample {sample_id}, Model {model_id}, Attempt {attempt+1}): Params: {api_params}, ExtraBody: {extra_body}")

            if extra_body:
                response = client.chat.completions.create(**api_params, extra_body=extra_body)
            else:
                response = client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            evaluation_token_tracker.add_usage(model_id, prompt_tokens, completion_tokens)
            
            logger.debug(f"API Success (Sample {sample_id}, Model {model_id}): Response: '{response_text[:100]}...', P_tok: {prompt_tokens}, C_tok: {completion_tokens}")
            return response_text, prompt_tokens, completion_tokens, None

        except Exception as e:
            error_msg = f"API call failed (Sample {sample_id}, Model {model_id}, Attempt {attempt+1}/{RETRY_ATTEMPTS}): {type(e).__name__} - {e}"
            logger.warning(error_msg)
            if attempt == RETRY_ATTEMPTS - 1:
                return None, None, None, error_msg
            time.sleep(RETRY_INITIAL_DELAY * (2 ** attempt) + random.uniform(0, 0.5))
    return None, None, None, "Max retry attempts reached for API call." # Should not be reached if loop completes

def evaluate_model_on_sample(
    model_config: Dict[str, Any],
    sample: Dict[str, Any],
    sample_idx: int, # For progress tracking
    total_samples: int # For progress tracking
) -> Dict[str, Any]:
    """Evaluates a single model on a single sample."""
    model_id = model_config["model_id"]
    display_name = model_config["display_name"]
    sample_id = sample.get("id", f"sample_{sample_idx}")
    
    # The 'full_text_for_eval' from your dataset already contains the narrative and the question.
    prompt_text = sample["full_text_for_eval"]
    ground_truth = sample["ground_truth_value"]

    # Construct system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE # Use the global template

    logger.info(f"Evaluating Sample {sample_id} ({sample_idx+1}/{total_samples}) with Model {display_name} ({model_id})...")

    response_text, p_tokens, c_tokens, error = _api_call_with_retry(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=prompt_text,
        max_tokens=model_config["max_completion_tokens"],
        temperature=model_config["temperature"],
        reasoning_settings=model_config["reasoning_settings"],
        sample_id=sample_id
    )

    extracted_answer = None
    is_correct = False

    if error:
        logger.error(f"Evaluation failed for Sample {sample_id}, Model {model_id}: {error}")
    elif response_text is not None:
        extracted_answer = extract_answer_from_llm_response(response_text)
        if extracted_answer is not None:
            is_correct = (extracted_answer == ground_truth)
            logger.info(
                f"Sample {sample_id}, Model {model_id}: Extracted={extracted_answer}, GT={ground_truth}, Correct={is_correct}"
            )
        else:
            logger.warning(
                f"Sample {sample_id}, Model {model_id}: Could not extract answer from response: '{response_text[:100]}...'"
            )
    else: # Should be caught by error but as a fallback
        error = "API call returned no response text and no error."
        logger.error(f"Evaluation failed for Sample {sample_id}, Model {model_id}: {error}")


    return {
        "sample_id": sample_id,
        "model_id": model_id,
        "model_display_name": display_name,
        "prompt_text_hash": hash(prompt_text), # To save space if prompt is huge
        "system_prompt": system_prompt,
        "raw_response": response_text,
        "extracted_answer": extracted_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "error": error,
        "timestamp": datetime.datetime.now().isoformat(),
    }

def generate_summary_table(results: List[Dict[str, Any]], dataset_name: str) -> Tuple[str, str]:
    """Generates a summary table from evaluation results."""
    summary_data = {}
    for res in results:
        model_name = res["model_display_name"]
        if model_name not in summary_data:
            summary_data[model_name] = {
                "total": 0, "correct": 0, "incorrect": 0, "errors": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "api_calls_model": 0
            }
        
        summary_data[model_name]["total"] += 1
        summary_data[model_name]["api_calls_model"] +=1 # Each result is one call attempt
        if res["error"]:
            summary_data[model_name]["errors"] += 1
        elif res["is_correct"]:
            summary_data[model_name]["correct"] += 1
        else: # Not an error, but incorrect (includes extraction failure)
            summary_data[model_name]["incorrect"] += 1
        
        if res["prompt_tokens"] is not None:
            summary_data[model_name]["prompt_tokens"] += res["prompt_tokens"]
        if res["completion_tokens"] is not None:
            summary_data[model_name]["completion_tokens"] += res["completion_tokens"]

    table_data = []
    headers = [
        "Model", "Total", "Correct", "Incorrect", "Errors", "Accuracy (%)",
        "Avg P-Tok", "Avg C-Tok", "Total Calls", "Est. Cost ($)"
    ]
    
    token_summary = evaluation_token_tracker.get_summary()

    for model_name, data in summary_data.items():
        accuracy = (data["correct"] / (data["total"] - data["errors"]) * 100) if (data["total"] - data["errors"]) > 0 else 0
        avg_p_tok = data["prompt_tokens"] / data["api_calls_model"] if data["api_calls_model"] > 0 else 0
        avg_c_tok = data["completion_tokens"] / data["api_calls_model"] if data["api_calls_model"] > 0 else 0
        
        # Find original model_id to get cost from token_tracker
        original_model_id = next((m["model_id"] for m in EVAL_CONFIG if m["display_name"] == model_name), None)
        est_cost = 0.0
        if original_model_id and original_model_id in token_summary["details_per_model"]:
            est_cost = token_summary["details_per_model"][original_model_id]["estimated_cost"]

        table_data.append([
            model_name, data["total"], data["correct"], data["incorrect"], data["errors"],
            f"{accuracy:.2f}", f"{avg_p_tok:.1f}", f"{avg_c_tok:.1f}", data["api_calls_model"], f"{est_cost:.4f}"
        ])

    # Sort by accuracy descending
    table_data.sort(key=lambda row: float(row[5]), reverse=True)

    title = f"Evaluation Summary: {dataset_name} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
    
    # Markdown table
    markdown_table_str = f"## {title}\n\n"
    markdown_table_str += tabulate(table_data, headers=headers, tablefmt="pipe")
    
    # CSV table
    csv_table_lines = [",".join(headers)]
    for row in table_data:
        csv_table_lines.append(",".join(map(str, row)))
    csv_table_str = "\n".join(csv_table_lines)
    
    return markdown_table_str, csv_table_str

# --- Main Execution ---
def main(args):
    logger.info(f"Starting evaluation run with {MAX_WORKERS} workers.")
    logger.info(f"Dataset path: {args.dataset_path}")
    if args.num_samples > 0:
        logger.info(f"Will evaluate on the first {args.num_samples} samples.")

    dataset = load_dataset(args.dataset_path)
    if not dataset:
        logger.error("No data loaded. Exiting.")
        return

    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        logger.info(f"Trimmed dataset to {len(dataset)} samples for this run.")

    all_results = []
    
    # Create a unique run ID for output files
    dataset_basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{dataset_basename}_eval_{run_timestamp}"
    
    current_run_output_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(current_run_output_dir, exist_ok=True)
    logger.info(f"Output for this run will be in: {current_run_output_dir}")

    detailed_results_file = os.path.join(current_run_output_dir, "detailed_results.jsonl")
    summary_table_md_file = os.path.join(current_run_output_dir, "summary_table.md")
    summary_table_csv_file = os.path.join(current_run_output_dir, "summary_table.csv")

    # Fetch initial OpenRouter usage if possible
    rate_limiter_global.update_limits_from_api()


    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="EvalWorker") as executor:
        for model_config in EVAL_CONFIG:
            for i, sample in enumerate(dataset):
                # Basic validation of sample structure
                if "full_text_for_eval" not in sample or "ground_truth_value" not in sample:
                    logger.warning(f"Skipping sample {sample.get('id', i)} due to missing required fields.")
                    continue
                tasks.append(executor.submit(evaluate_model_on_sample, model_config, sample, i, len(dataset)))

        for i, future in enumerate(concurrent.futures.as_completed(tasks)):
            try:
                result = future.result()
                all_results.append(result)
                # Log progress intermittently
                if (i + 1) % (len(tasks) // 10 if len(tasks) >=10 else 1) == 0 or (i+1) == len(tasks):
                     logger.info(f"Completed {i+1}/{len(tasks)} evaluation tasks...")
            except Exception as e:
                logger.error(f"A task generated an exception: {e}", exc_info=True)

    # Save detailed results
    try:
        with open(detailed_results_file, 'w', encoding='utf-8') as f_detailed:
            for res in all_results:
                f_detailed.write(json.dumps(res) + "\n")
        logger.info(f"Detailed results saved to {detailed_results_file}")
    except IOError as e:
        logger.error(f"Failed to write detailed results: {e}")

    # Generate and save summary table
    if all_results:
        markdown_summary, csv_summary = generate_summary_table(all_results, dataset_basename)
        
        print("\n" + "="*30 + " EVALUATION SUMMARY " + "="*30)
        print(markdown_summary)
        print("="*80)

        try:
            with open(summary_table_md_file, 'w', encoding='utf-8') as f_md:
                f_md.write(markdown_summary)
            logger.info(f"Markdown summary table saved to {summary_table_md_file}")
        except IOError as e:
            logger.error(f"Failed to write markdown summary: {e}")

        try:
            with open(summary_table_csv_file, 'w', encoding='utf-8') as f_csv:
                f_csv.write(csv_summary)
            logger.info(f"CSV summary table saved to {summary_table_csv_file}")
        except IOError as e:
            logger.error(f"Failed to write CSV summary: {e}")
    else:
        logger.info("No results to summarize.")

    # Log token usage and cost
    final_token_summary = evaluation_token_tracker.get_summary()
    logger.info(f"Total Evaluation Token Usage: {final_token_summary}")
    print(f"\nTotal Estimated Evaluation Cost: ${final_token_summary['total_estimated_cost']:.4f}")
    for model_id, data in final_token_summary.get("details_per_model", {}).items():
        print(f"  Model {model_id}: P-Toks: {data['prompt_tokens']}, C-Toks: {data['completion_tokens']}, Calls: {data['api_calls']}, Est.Cost: ${data['estimated_cost']:.4f}")

    # Fetch final OpenRouter usage
    rate_limiter_global.update_limits_from_api()

    logger.info("Evaluation run finished.")
    logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluations on a Verbose ListOps dataset.")
    parser.add_argument(
        "dataset_path",
        nargs="?", # Makes it optional
        help="Path to the JSONL dataset file generated by verbose-listops.py",
        default=None # Default to None, will be handled below
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0, # 0 means all samples
        help="Number of samples to evaluate from the dataset (0 for all). Default: 0",
    )
    
    args = parser.parse_args()

    if args.dataset_path is None:
        # Construct default path if not provided
        # This assumes your script is in the same directory as the 'datasets' folder
        # and the verbose-listops.py script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_full_path = os.path.join(script_dir, DEFAULT_DATASET_SUBDIR, DEFAULT_DATASET_FILENAME)
        
        if os.path.exists(default_full_path):
            args.dataset_path = default_full_path
            logger.info(f"No dataset_path provided, using default: {args.dataset_path}")
        else:
            logger.error(f"Default dataset path not found: {default_full_path}")
            logger.error("Please provide a valid dataset_path argument or ensure the default dataset exists.")
            sys.exit(1)
    
    if not os.path.exists(args.dataset_path):
        logger.error(f"Specified dataset_path does not exist: {args.dataset_path}")
        sys.exit(1)

    main(args)