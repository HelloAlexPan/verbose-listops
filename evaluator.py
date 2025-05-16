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

import openai # Import for specific exception types
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
from tabulate import tabulate
import requests
import pandas as pd

load_dotenv()

# --- Configuration ---

# JSON Schema for structured output (only for models that support it for NARRATIVE tasks)
ANSWER_SCHEMA = {
    "name": "single_integer_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "integer",
                "description": "The single integer result of the evaluation.",
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}

EVAL_CONFIG = [ # NO CHANGES TO THIS AS PER REQUEST
    {
        "model_id": "google/gemini-2.5-pro-preview",
        "display_name": "Gemini 2.5 Pro",
        "reasoning_settings": {"exclude": True},
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "google/gemini-2.5-flash-preview:thinking",
        "display_name": "Gemini 2.5 Flash Thinking",
        "reasoning_settings": {"exclude": True},
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "openai/o4-mini-high",
        "display_name": "o4 Mini High",
        "reasoning_settings": {"effort": "high", "exclude": True},
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "openai/gpt-4.1",
        "display_name": "GPT-4.1",
        "reasoning_settings": None,
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": True,
    },
    {
        "model_id": "x-ai/grok-3-mini-beta",
        "display_name": "Grok 3 Mini High",
        "reasoning_settings": {"effort": "high", "exclude": True},
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "qwen/qwen3-235b-a22b",
        "display_name": "Qwen3 235B A22B",
        "reasoning_settings": {"exclude": True},
        "max_completion_tokens": 50000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "deepseek/deepseek-r1",
        "display_name": "DeepSeek R1",
        "reasoning_settings": {"exclude": True},
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": True,
    },
    {
        "model_id": "deepseek/deepseek-chat-v3-0324",
        "display_name": "DeepSeek V3 0324",
        "reasoning_settings": {"exclude": True},
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "anthropic/claude-3.7-sonnet:thinking",
        "display_name": "Claude 3.7 Sonnet",
        "reasoning_settings": {"effort": "high", "exclude": True},
        "max_completion_tokens": 60000,
        "temperature": 0.0,
        "supports_structured_output": False,
    },
    {
        "model_id": "meta-llama/llama-4-maverick",
        "display_name": "Llama 4 Maverick",
        "reasoning_settings": None,
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": True,
    },
]


DEFAULT_DATASET_FILENAME = "[MAIN]_DATASET_10000tok_8mxops_4minarity_8mxbrch_google_gemini-2.5-flash-preview-thinking_20250515-212340.jsonl"
DEFAULT_DATASET_SUBDIR = "datasets"
OUTPUT_DIR = "evaluation_results"
MAX_WORKERS = 1000
API_RETRY_ATTEMPTS = 5 
INFINITE_API_RETRY_INITIAL_DELAY = 10.0 
INFINITE_API_RETRY_MAX_DELAY = 300.0    
REGULAR_API_RETRY_INITIAL_DELAY = 1.0 
API_TIMEOUT = 120
JSON_FIX_ATTEMPTS = 3

INITIAL_RPS_TARGET = 10.0 
MAX_OVERALL_RPS_CONFIG = 1000.0 
MIN_RPS_TARGET = 1.0 
RATE_LIMITER_BUCKET_CAPACITY = 1000 
RATE_LIMITER_MIN_INTERVAL = 0.05 
RATE_LIMITER_JITTER = 0.02

NARRATIVE_SYSTEM_PROMPT_TEMPLATE = ( 
    "IMPORTANT:Respond ONLY with a single integer. DO NOT include any other words, explanations, or punctuation."
)

# --- Global Variables & Setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set. API calls will fail.")
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"

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

LOG_DIR_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_logs")
os.makedirs(LOG_DIR_EVAL, exist_ok=True)

logger = logging.getLogger("llm_evaluator")
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

eval_log_file = os.path.join(LOG_DIR_EVAL, f"evaluation_run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
file_handler = logging.handlers.RotatingFileHandler(
    filename=eval_log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info(f"Logging initialized. Detailed logs at: {eval_log_file}")

class EvaluationTokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.api_calls = 0
        self.lock = threading.Lock()
        self.tokens_per_model = {} # Simplified: key is just model_id

    def add_usage(self, model_id: str, prompt_tokens: int, completion_tokens: int): # Removed eval_type
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.api_calls += 1
            
            if model_id not in self.tokens_per_model:
                self.tokens_per_model[model_id] = {'prompt_tokens': 0, 'completion_tokens': 0, 'api_calls': 0}
            
            self.tokens_per_model[model_id]['prompt_tokens'] += prompt_tokens
            self.tokens_per_model[model_id]['completion_tokens'] += completion_tokens
            self.tokens_per_model[model_id]['api_calls'] += 1
        logger.debug(
            f"Token Tracker ({model_id}): Added P:{prompt_tokens}, C:{completion_tokens}. "
            f"Model Totals - P:{self.tokens_per_model[model_id]['prompt_tokens']}, "
            f"C:{self.tokens_per_model[model_id]['completion_tokens']}, "
            f"Calls:{self.tokens_per_model[model_id]['api_calls']}"
        )

    def get_summary(self):
        with self.lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_api_calls": self.api_calls,
                "details_per_model": self.tokens_per_model.copy() # Simplified
            }
evaluation_token_tracker = EvaluationTokenTracker()

class RateLimiter:
    def __init__(self, 
                 max_requests_per_second: float = INITIAL_RPS_TARGET, 
                 min_interval: float = RATE_LIMITER_MIN_INTERVAL, 
                 bucket_capacity: int = RATE_LIMITER_BUCKET_CAPACITY, 
                 jitter: float = RATE_LIMITER_JITTER):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = min_interval
        self.bucket_capacity = bucket_capacity
        self.jitter = jitter
        self.tokens = float(bucket_capacity) 
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        self.last_limits_check_time = 0
        self.limits_check_interval = 30 
        self.initial_account_usage = None
        self.current_account_usage = None
        logger.info(
            f"Rate limiter initialized: Target RPS {self.max_requests_per_second:.2f}, "
            f"Min Interval {self.min_interval:.3f}s, Bucket Capacity {self.bucket_capacity}, Jitter {self.jitter:.3f}"
        )

    def wait_if_needed(self):
        current_time = time.time()
        if current_time - self.last_limits_check_time > self.limits_check_interval:
            self.update_limits_and_usage() 

        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_refill_time
            new_tokens = elapsed * self.max_requests_per_second
            self.tokens = min(float(self.bucket_capacity), self.tokens + new_tokens)
            self.last_refill_time = current_time

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            wait_time_for_token = (1.0 - self.tokens) / self.max_requests_per_second
            effective_wait_time = max(wait_time_for_token, self.min_interval)
            
            if self.jitter > 0:
                effective_wait_time += random.uniform(0, self.jitter * effective_wait_time) 

            logger.debug(f"Rate limiting: Target RPS {self.max_requests_per_second:.2f}. Waiting for {effective_wait_time:.3f}s (tokens: {self.tokens:.2f})")
            time.sleep(effective_wait_time)
            
            self.tokens = max(0.0, self.tokens + (effective_wait_time * self.max_requests_per_second) - 1.0) 
            self.last_refill_time = time.time() 
            return effective_wait_time

    def update_limits_and_usage(self) -> Optional[float]: 
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
            logger.warning("Cannot check OpenRouter limits/usage: No valid API key")
            return None
        
        self.last_limits_check_time = time.time()
        logger.debug("Attempting to update OpenRouter limits and usage...")

        try:
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10, 
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                
                usage_str = data.get("usage")
                if usage_str is not None:
                    self.current_account_usage = float(usage_str)
                    if self.initial_account_usage is None:
                        self.initial_account_usage = self.current_account_usage
                        logger.info(f"Initial OpenRouter account usage recorded: ${self.initial_account_usage:.4f}")
                    else:
                        logger.info(f"Updated OpenRouter account usage: ${self.current_account_usage:.4f}")
                else:
                    logger.debug("OpenRouter Status: No 'usage' field in API response.")

                rate_limit_info = data.get("rate_limit", {})
                if rate_limit_info:
                    requests_limit = rate_limit_info.get("requests") 
                    interval_str = rate_limit_info.get("interval", "") 

                    if requests_limit and interval_str and interval_str.endswith('s'):
                        try:
                            interval_seconds = int(interval_str[:-1])
                            if interval_seconds > 0:
                                reported_rps = float(requests_limit) / interval_seconds
                                new_target_rps = max(MIN_RPS_TARGET, min(reported_rps * 0.8, MAX_OVERALL_RPS_CONFIG))
                                
                                if new_target_rps != self.max_requests_per_second:
                                    logger.info(
                                        f"Dynamically adjusting RateLimiter RPS. "
                                        f"OpenRouter reported: {requests_limit} req / {interval_seconds}s (~{reported_rps:.2f} RPS). "
                                        f"Old target: {self.max_requests_per_second:.2f} RPS. "
                                        f"New target (80% of reported, capped): {new_target_rps:.2f} RPS."
                                    )
                                    with self.lock: 
                                        self.max_requests_per_second = new_target_rps
                                else:
                                    logger.debug(f"RateLimiter RPS ({self.max_requests_per_second:.2f}) already aligned with 80% of reported OpenRouter limit or caps.")
                            else:
                                logger.warning(f"OpenRouter reported invalid interval_seconds: {interval_seconds}")
                        except ValueError:
                            logger.warning(f"Could not parse OpenRouter rate limit interval: {interval_str}")
                    else:
                        logger.debug("OpenRouter rate_limit info not found or in unexpected format in API response.")
                else:
                    logger.debug("No 'rate_limit' field in OpenRouter API response.")
                return self.current_account_usage
            else:
                logger.warning(f"Failed to get OpenRouter account status: HTTP {response.status_code}, Response: {response.text}")
        except requests.exceptions.RequestException as e: 
            logger.error(f"Network error checking OpenRouter limits/usage: {e}")
        except Exception as e:
            logger.error(f"Error checking OpenRouter limits/usage: {e}", exc_info=True)
        return None

    def get_initial_usage(self) -> Optional[float]: return self.initial_account_usage
    def get_current_usage(self) -> Optional[float]: return self.current_account_usage

rate_limiter_global = RateLimiter(
    max_requests_per_second=INITIAL_RPS_TARGET,
    min_interval=RATE_LIMITER_MIN_INTERVAL,
    bucket_capacity=RATE_LIMITER_BUCKET_CAPACITY,
    jitter=RATE_LIMITER_JITTER
)


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
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

def fix_and_parse_json(text: str, schema: Dict = ANSWER_SCHEMA) -> Optional[Dict]:
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data and isinstance(data["answer"], int):
            logger.debug("Direct JSON parsing successful.")
            return data
    except json.JSONDecodeError:
        logger.debug("Direct JSON parsing failed, attempting extraction.")

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_text_candidate = match.group(0)
        try:
            data = json.loads(json_text_candidate)
            if isinstance(data, dict) and "answer" in data and isinstance(data["answer"], int):
                logger.debug(f"Extracted JSON block and parsed successfully: {json_text_candidate[:100]}...")
                return data
            else:
                logger.warning(f"Extracted JSON block, but schema mismatch: {data}. Original text: {text[:100]}...")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse extracted JSON block: {json_text_candidate[:100]}... Original text: {text[:100]}...")
    else:
        logger.debug(f"No JSON-like block found with regex in: {text[:100]}...")
    return None

def extract_answer_from_llm_response(response_text: str, model_used_structured_output: bool) -> Optional[int]:
    # Removed eval_type parameter as it's no longer needed for specific "ANSWER: x" logic
    if not response_text:
        return None

    if model_used_structured_output: 
        parsed_json = None
        for attempt in range(JSON_FIX_ATTEMPTS): 
            parsed_json = fix_and_parse_json(response_text) 
            if parsed_json:
                if "answer" in parsed_json and isinstance(parsed_json["answer"], int):
                    return parsed_json["answer"] # Return directly if schema matches
                else: 
                    break 
            elif attempt < JSON_FIX_ATTEMPTS - 1:
                break 
        if not parsed_json:
             logger.warning(f"Failed to parse response as valid/schema-compliant JSON after {JSON_FIX_ATTEMPTS} attempts (SO was attempted): '{response_text[:100]}...'. Falling back to regex.")

    cleaned_text = re.sub(r"```(?:json|text)?\s*|\s*```", "", response_text).strip()
    try:
        candidate_from_full_string = cleaned_text.strip().rstrip('.')
        if candidate_from_full_string:
            if re.fullmatch(r"-?\d+", candidate_from_full_string):
                val = int(candidate_from_full_string)
                logger.debug(f"Extracted answer {val} by parsing the full cleaned string (regex fallback).")
                return val
    except ValueError:
        pass

    matches = re.findall(r'(?<!\d)-?\d+(?!\d)', cleaned_text)
    if len(matches) == 1:
        try:
            val = int(matches[0])
            logger.debug(f"Extracted single number {val} via regex findall (regex fallback).")
            return val
        except ValueError:
            logger.warning(f"Found single number-like string '{matches[0]}' but failed to convert to int (regex fallback).")
            return None
    elif len(matches) > 1:
        try:
            last_match_val = int(matches[-1])
            logger.info(f"Multiple numbers found by regex ({matches}). Using the last one: {last_match_val} from text (regex fallback): '{cleaned_text[:100]}...'")
            return last_match_val
        except ValueError:
            logger.warning(f"Multiple numbers found by regex, last one '{matches[-1]}' failed to convert to int (regex fallback).")
            return None
    else:
        logger.debug(f"No standalone integer found in response via regex findall (regex fallback): '{cleaned_text[:100]}...'")
        return None

def _api_call_with_retry(
    model_config: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    sample_id: str
    # Removed eval_type from here
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    if not client:
        return None, None, None, "OpenRouter client not initialized."

    model_id = model_config["model_id"]
    reasoning_settings = model_config["reasoning_settings"]
    # Structured output is now only based on model_config, not eval_type
    use_structured_output_for_this_call = model_config.get("supports_structured_output", False)

    api_attempt_count = 0 
    current_regular_api_retry_delay = REGULAR_API_RETRY_INITIAL_DELAY
    current_infinite_api_retry_delay = INFINITE_API_RETRY_INITIAL_DELAY

    while True: 
        api_attempt_count += 1
        is_indefinitely_retryable_api_error = True
        api_error_for_this_attempt = "" 

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
            final_reasoning_config = None

            if reasoning_settings:
                current_reasoning_config = reasoning_settings.copy()
                supports_effort = False
                if model_id.startswith("openai/"):
                    if model_id.startswith("openai/o") or "gpt-4o" in model_id:
                        supports_effort = True
                elif model_id == "x-ai/grok-3-mini-beta":
                    supports_effort = True
                
                if "effort" in current_reasoning_config and not supports_effort:
                    del current_reasoning_config["effort"]
                
                if current_reasoning_config:
                     extra_body["reasoning"] = current_reasoning_config
                     final_reasoning_config = current_reasoning_config
            
            if use_structured_output_for_this_call:
                extra_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": ANSWER_SCHEMA
                }
            
            log_msg_parts = [
                f"API Call (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count})", # Removed EvalType
                f"Params: model={api_params['model']}, temp={api_params['temperature']}, max_tok={api_params['max_tokens']}",
            ]
            if final_reasoning_config: log_msg_parts.append(f"Reasoning: {final_reasoning_config}")
            if use_structured_output_for_this_call: log_msg_parts.append("Struct.Output: True")
            logger.debug(" | ".join(log_msg_parts))

            response_obj = client.chat.completions.create(**api_params, extra_body=extra_body if extra_body else None)

            if hasattr(response_obj, 'error') and response_obj.error and hasattr(response_obj.error, 'message'):
                error_code = getattr(response_obj.error, 'code', None)
                error_message_from_body = response_obj.error.message
                error_detail = f"Error in response body: Code {error_code} - {error_message_from_body}"
                if hasattr(response_obj.error, 'metadata') and response_obj.error.metadata:
                    error_detail += f" | Metadata: {response_obj.error.metadata}"
                
                api_error_for_this_attempt = f"API Success (HTTP 200) but error in response content. {error_detail}."
                logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id})") # Removed EvalType

                if error_code == 429 and ("rate-limited upstream" in error_message_from_body.lower() or "quota" in error_message_from_body.lower()):
                    is_indefinitely_retryable_api_error = True
                elif str(error_code).startswith('5') or error_code == 500 or (hasattr(response_obj.error, 'type') and response_obj.error.type == 'server_error'): 
                    is_indefinitely_retryable_api_error = True
                
            elif not response_obj.choices or not response_obj.choices[0].message:
                api_error_for_this_attempt = "API Success but no choices/message in response."
                logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}). Response: {response_obj.model_dump_json(indent=2)}") # Removed EvalType
            
            else: 
                response_text = response_obj.choices[0].message.content
                prompt_tokens = response_obj.usage.prompt_tokens if response_obj.usage else 0
                completion_tokens = response_obj.usage.completion_tokens if response_obj.usage else 0
                
                if "google/gemini" in model_id:
                    logger.info(
                        f"RAW RESPONSE (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count}):\n" # Removed EvalType
                        f"---BEGIN RAW RESPONSE---\n{response_text}\n---END RAW RESPONSE---"
                    )

                evaluation_token_tracker.add_usage(model_id, prompt_tokens, completion_tokens) # Removed eval_type
                logger.debug(f"API Success (Sample {sample_id}, Model {model_id}): Response: '{str(response_text)[:100] if response_text else ""}...', P_tok: {prompt_tokens}, C_tok: {completion_tokens}") # Removed EvalType
                return response_text, prompt_tokens, completion_tokens, None 

        except openai.APIStatusError as e:
            api_error_for_this_attempt = f"APIStatusError: Status {e.status_code} - {e.message}"
            logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count})") # Removed EvalType
            if e.status_code in [429, 500, 502, 503, 504]: 
                is_indefinitely_retryable_api_error = True
        except openai.APIConnectionError as e:
            api_error_for_this_attempt = f"APIConnectionError: {e}"
            logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count})") # Removed EvalType
            is_indefinitely_retryable_api_error = True
        except openai.BadRequestError as e:
            api_error_for_this_attempt = f"BadRequestError: {e}"
            logger.error(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count}) - Not retryable.") # Removed EvalType
            return None, None, None, api_error_for_this_attempt 
        except openai.APIError as e: 
            api_error_for_this_attempt = f"APIError ({type(e).__name__}): {e}"
            logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count})") # Removed EvalType
            if hasattr(e, 'status_code') and e.status_code in [429, 500, 502, 503, 504]:
                 is_indefinitely_retryable_api_error = True
        except requests.exceptions.RequestException as e: 
            api_error_for_this_attempt = f"Network error (requests): {type(e).__name__} - {e}"
            logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count})") # Removed EvalType
            is_indefinitely_retryable_api_error = True
        except Exception as e:
            api_error_for_this_attempt = f"General Exception: {type(e).__name__} - {e}"
            logger.warning(f"{api_error_for_this_attempt} (Sample {sample_id}, Model {model_id}, API_Attempt {api_attempt_count})", exc_info=True) # Removed EvalType

        if is_indefinitely_retryable_api_error:
            delay = min(current_infinite_api_retry_delay, INFINITE_API_RETRY_MAX_DELAY)
            logger.info(f"Indefinite API retry for {model_id} (Sample {sample_id}): Waiting {delay:.2f}s. API_Attempt {api_attempt_count}. Error: {api_error_for_this_attempt.splitlines()[0]}") # Removed EvalType
            time.sleep(delay + random.uniform(0, 0.1 * delay)) 
            current_infinite_api_retry_delay = min(current_infinite_api_retry_delay * 1.5, INFINITE_API_RETRY_MAX_DELAY)
            current_regular_api_retry_delay = REGULAR_API_RETRY_INITIAL_DELAY 
        elif api_attempt_count < API_RETRY_ATTEMPTS:
            delay = current_regular_api_retry_delay
            logger.warning(f"Regular API retry for {model_id} (Sample {sample_id}): Waiting {delay:.2f}s. API_Attempt {api_attempt_count}/{API_RETRY_ATTEMPTS}. Error: {api_error_for_this_attempt.splitlines()[0]}") # Removed EvalType
            time.sleep(delay + random.uniform(0, 0.1 * delay))
            current_regular_api_retry_delay *= 2
            current_infinite_api_retry_delay = INFINITE_API_RETRY_INITIAL_DELAY 
        else: 
            final_api_error_message = api_error_for_this_attempt if api_error_for_this_attempt else f"Max regular API retry attempts ({API_RETRY_ATTEMPTS}) reached."
            logger.error(f"Giving up API call for {model_id} (Sample {sample_id}) after {API_RETRY_ATTEMPTS} non-indefinitely-retryable API attempts. Last error: {final_api_error_message.splitlines()[0]}") # Removed EvalType
            return None, None, None, final_api_error_message
        
    return None, None, None, "Exited API retry loop unexpectedly after all attempts."


def evaluate_model_on_sample(
    model_config: Dict[str, Any],
    sample: Dict[str, Any],
    sample_idx: int,
    total_samples: int
    # Removed evaluation_type
) -> Dict[str, Any]:
    model_id = model_config["model_id"]
    display_name = model_config["display_name"]
    sample_id = sample.get("id", f"sample_{sample_idx}")
    ground_truth = sample["ground_truth_value"]
    
    # Always use narrative evaluation settings
    prompt_text = sample["full_text_for_eval"]
    system_prompt = NARRATIVE_SYSTEM_PROMPT_TEMPLATE
    model_supports_structured_output_for_this_call = model_config.get("supports_structured_output", False)

    logger.info(f"Evaluating Sample {sample_id} ({sample_idx+1}/{total_samples}) with Model {display_name} ({model_id}). SO Attempt: {model_supports_structured_output_for_this_call}")

    # No more content/format retry loop here, _api_call_with_retry handles API retries
    response_text, p_tokens, c_tokens, api_error = _api_call_with_retry(
        model_config=model_config,
        system_prompt=system_prompt,
        user_prompt=prompt_text,
        max_tokens=model_config["max_completion_tokens"],
        temperature=model_config["temperature"],
        sample_id=sample_id
        # Removed eval_type
    )

    extracted_answer = None
    is_correct = False
    json_extraction_issue = False 
    final_error_for_result = api_error # Start with API error if any

    if not api_error and response_text is not None:
        extracted_answer = extract_answer_from_llm_response(
            response_text, 
            model_supports_structured_output_for_this_call
            # Removed eval_type
        )
        if extracted_answer is not None:
            is_correct = (extracted_answer == ground_truth)
        else: # No answer extracted
            if model_supports_structured_output_for_this_call:
                json_extraction_issue = True
                final_error_for_result = "Could not extract answer via JSON or regex fallback after SO attempt."
            else:
                final_error_for_result = "Could not extract answer from response (non-SO)."
            logger.warning(f"Sample {sample_id}, Model {model_id}: {final_error_for_result} Raw: '{str(response_text)[:200] if response_text else ""}...'")
    
    elif not api_error and response_text is None: # Should be rare if _api_call_with_retry works
        final_error_for_result = "API call returned no response text but no specific API error."
        logger.error(f"Sample {sample_id}, Model {model_id}: {final_error_for_result}")
    
    if extracted_answer is not None: # Log final extraction outcome
         logger.info(
                f"FINAL RESULT Sample {sample_id}, Model {model_id}: Extracted={extracted_answer}, GT={ground_truth}, Correct={is_correct}"
            )
    elif final_error_for_result:
        logger.error(f"FINAL ERROR Sample {sample_id}, Model {model_id}: {final_error_for_result}")


    return {
        "sample_id": sample_id,
        "model_id": model_id,
        "model_display_name": display_name,
        # "evaluation_type": "narrative", # Can add this if you want to keep the field
        "prompt_text_hash": hash(prompt_text),
        "system_prompt": system_prompt,
        "raw_response": response_text, 
        "extracted_answer": extracted_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "prompt_tokens": p_tokens, 
        "completion_tokens": c_tokens,
        "error": final_error_for_result, 
        "timestamp": datetime.datetime.now().isoformat(),
        "used_structured_output_attempt": model_supports_structured_output_for_this_call,
        "json_extraction_issue": json_extraction_issue
    }

def generate_summary_table(results: List[Dict[str, Any]], dataset_name: str) -> Tuple[str, str]:
    summary_data = {} 
    for res in results:
        model_name = res["model_display_name"]

        if model_name not in summary_data:
            summary_data[model_name] = {
                "total": 0, "correct": 0, "api_errors": 0,
                "content_errors": 0, # Combined no_extract and wrong_val
                "prompt_tokens": 0, "completion_tokens": 0, 
                "so_attempts": 0, "so_parsed_schema_ok": 0,
            }
        
        summary_data[model_name]["total"] += 1
        summary_data[model_name]["prompt_tokens"] += res.get("prompt_tokens", 0) or 0
        summary_data[model_name]["completion_tokens"] += res.get("completion_tokens", 0) or 0
        
        if res.get("used_structured_output_attempt"):
            summary_data[model_name]["so_attempts"] += 1
        
        is_api_error = False
        if res["error"]:
            api_err_indicators = ["APIStatusError", "APIConnectionError", "APIError", "Network error", "error in response content", "no choices/message"]
            if any(indicator in res["error"] for indicator in api_err_indicators):
                summary_data[model_name]["api_errors"] += 1
                is_api_error = True

        if not is_api_error: 
            if res["extracted_answer"] is not None:
                if res.get("used_structured_output_attempt"):
                     summary_data[model_name]["so_parsed_schema_ok"] +=1
                if res["is_correct"]:
                    summary_data[model_name]["correct"] += 1
                else: 
                    summary_data[model_name]["content_errors"] += 1
            else: 
                summary_data[model_name]["content_errors"] += 1
    
    table_data = []
    headers = [
        "Model", 
        "Accuracy (%)", "Correct", "Content Err", "API Err", "Total",
        "Avg P-Tok", "Avg C-Tok", "SO OK (%)"
    ]

    for model_name, data in summary_data.items():
        valid_for_acc = data["total"] - data["api_errors"]
        accuracy = (data["correct"] / valid_for_acc * 100) if valid_for_acc > 0 else 0.0
        
        avg_p_tok = data["prompt_tokens"] / data["total"] if data["total"] > 0 else 0
        avg_c_tok = data["completion_tokens"] / data["total"] if data["total"] > 0 else 0
        
        so_ok_rate = (data["so_parsed_schema_ok"] / data["so_attempts"] * 100) if data["so_attempts"] > 0 else float('nan')
        so_str = f"{so_ok_rate:.1f}%" if not pd.isna(so_ok_rate) else "N/A"

        table_data.append([
            model_name, 
            f"{accuracy:.1f}", data["correct"], data["content_errors"], data["api_errors"], data["total"],
            f"{avg_p_tok:.0f}", f"{avg_c_tok:.0f}", 
            so_str
        ])

    table_data.sort(key=lambda row: (float(row[1].replace('%','').replace('N/A','-1'))), reverse=True) 
    
    title = f"Narrative Evaluation Summary: {dataset_name} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
    markdown_table_str = f"## {title}\n\n" + tabulate(table_data, headers=headers, tablefmt="pipe")
    
    csv_headers_simple = [h.replace(' (%)','_Pct').replace(' ','_') for h in headers]
    csv_table_lines = [",".join(csv_headers_simple)]
    for row in table_data:
        csv_row = [str(item).replace('%','') for item in row]
        csv_table_lines.append(",".join(csv_row))
        
    csv_table_str = "\n".join(csv_table_lines)
    return markdown_table_str, csv_table_str

def main(args):
    logger.info(f"Starting evaluation run with {MAX_WORKERS} workers.")
    logger.info(f"Dataset path: {args.dataset_path}")
    if args.num_samples > 0:
        logger.info(f"Will evaluate on the first {args.num_samples} samples.")

    logger.info("Fetching initial OpenRouter account usage...")
    initial_account_usage = rate_limiter_global.update_limits_and_usage() 
    if initial_account_usage is None:
        logger.warning("Could not fetch initial OpenRouter account usage. Cost calculation by difference will not be available.")

    dataset = load_dataset(args.dataset_path)
    if not dataset:
        logger.error("No data loaded. Exiting.")
        return

    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        logger.info(f"Trimmed dataset to {len(dataset)} samples for this run.")

    all_results = []
    dataset_basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{dataset_basename}_eval_{run_timestamp}"
    current_run_output_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(current_run_output_dir, exist_ok=True)
    logger.info(f"Output for this run will be in: {current_run_output_dir}")

    detailed_results_file = os.path.join(current_run_output_dir, "detailed_results.jsonl")
    summary_table_md_file = os.path.join(current_run_output_dir, "summary_table.md")
    summary_table_csv_file = os.path.join(current_run_output_dir, "summary_table.csv")

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="EvalWorker") as executor:
        for model_config in EVAL_CONFIG:
            for i, sample in enumerate(dataset):
                if "full_text_for_eval" not in sample or \
                   "ground_truth_value" not in sample : # Removed ast_str check
                    logger.warning(f"Skipping sample {sample.get('id', i)} due to missing required fields (full_text_for_eval or ground_truth_value).")
                    continue
                
                tasks.append(executor.submit(evaluate_model_on_sample, model_config, sample, i, len(dataset))) # Removed eval_type

        processed_tasks = 0
        total_tasks_to_process = len(tasks) 
        for future in concurrent.futures.as_completed(tasks):
            processed_tasks +=1
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                logger.error(f"A task future generated an unhandled exception: {e}", exc_info=True)
            
            log_interval = max(1, total_tasks_to_process // 20 if total_tasks_to_process > 0 else 1)
            if processed_tasks % log_interval == 0 or processed_tasks == total_tasks_to_process:
                    logger.info(f"Completed {processed_tasks}/{total_tasks_to_process} evaluation tasks...")

    try:
        with open(detailed_results_file, 'w', encoding='utf-8') as f_detailed:
            for res in all_results:
                f_detailed.write(json.dumps(res) + "\n")
        logger.info(f"Detailed results saved to {detailed_results_file}")
    except IOError as e:
        logger.error(f"Failed to write detailed results: {e}")

    if all_results:
        markdown_summary, csv_summary = generate_summary_table(all_results, dataset_basename)
        print("\n" + "="*30 + " EVALUATION SUMMARY " + "="*30)
        print(markdown_summary)
        print("="*80)
        try:
            with open(summary_table_md_file, 'w', encoding='utf-8') as f_md: f_md.write(markdown_summary)
            logger.info(f"Markdown summary table saved to {summary_table_md_file}")
        except IOError as e: logger.error(f"Failed to write markdown summary: {e}")
        try:
            with open(summary_table_csv_file, 'w', encoding='utf-8') as f_csv: f_csv.write(csv_summary)
            logger.info(f"CSV summary table saved to {summary_table_csv_file}")
        except IOError as e: logger.error(f"Failed to write CSV summary: {e}")
    else:
        logger.info("No results to summarize.")

    final_token_summary = evaluation_token_tracker.get_summary()
    logger.info(f"Total Evaluation Token Usage Summary: {final_token_summary}")
    
    print("\n--- Token Usage Per Model ---") # Simplified
    for model_id_key, data in final_token_summary.get("details_per_model", {}).items(): # Simplified
        model_display_name = next((mcfg["display_name"] for mcfg in EVAL_CONFIG if mcfg["model_id"] == model_id_key), model_id_key)
        print(f"  Model: {model_display_name:<25} | P-Toks: {data['prompt_tokens']:<6}, C-Toks: {data['completion_tokens']:<6}, Calls: {data['api_calls']}")


    logger.info("Fetching final OpenRouter account usage...")
    final_account_usage = rate_limiter_global.update_limits_and_usage() 

    if initial_account_usage is not None and final_account_usage is not None:
        total_run_cost = final_account_usage - initial_account_usage
        logger.info(f"Total Run Cost (based on OpenRouter balance difference): ${total_run_cost:.4f}")
        print(f"\nTotal Run Cost (OpenRouter): ${total_run_cost:.4f} (Final: ${final_account_usage:.4f} - Initial: ${initial_account_usage:.4f})")
    elif final_account_usage is not None:
         logger.info(f"Final OpenRouter account usage: ${final_account_usage:.4f}. Initial usage was not available for cost difference calculation.")
         print(f"\nFinal OpenRouter account usage: ${final_account_usage:.4f} (Initial usage was not available for cost difference calculation).")
    else:
        logger.warning("Could not determine total run cost from OpenRouter balance.")
        print("\nCould not determine total run cost from OpenRouter balance.")

    logger.info("Evaluation run finished.")
    logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluations on Verbose ListOps narratives.") # Updated description
    parser.add_argument(
        "dataset_path", nargs="?", default=None,
        help="Path to the JSONL dataset file. If not provided, tries to use a default path."
    )
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help="Number of samples to evaluate (0 for all). Default: 0",
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_full_path = os.path.join(script_dir, DEFAULT_DATASET_SUBDIR, DEFAULT_DATASET_FILENAME)

        if os.path.exists(default_full_path):
            args.dataset_path = default_full_path
            logger.info(f"No dataset_path provided, using default: {args.dataset_path}")
        else:
            logger.error(f"Default dataset path not found: {default_full_path}")
            logger.error("Please provide a valid dataset_path argument or ensure the default dataset (DEFAULT_DATASET_SUBDIR, DEFAULT_DATASET_FILENAME) exists relative to the script.")
            sys.exit(1)

    if not os.path.exists(args.dataset_path):
        logger.error(f"Specified dataset_path does not exist: {args.dataset_path}")
        sys.exit(1)

    main(args)