import os
import json
import re
import time
import datetime
import asyncio
import aiohttp
import requests
import threading
import random
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
except ImportError:
    # If dotenv is not installed, define a simple function to parse .env file
    def load_dotenv(dotenv_path=None):
        if dotenv_path and os.path.isfile(dotenv_path):
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
        return True

# --- Configuration ---
EVAL_CONFIG = [
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
        "model_id": "anthropic/claude-3.7-sonnet",
        "display_name": "Claude 3.7 Sonnet",
        "reasoning_settings": None,
        "max_completion_tokens": 30000,
        "temperature": 0.0,
        "supports_structured_output": True,
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

# Dataset configuration (as provided by the user)
DEFAULT_DATASET_DIRNAME = "10000tok_8mxops_4minarity_8mxbrch_google_gemini-2.5-flash-preview-thinking_20250515-212340"
DEFAULT_DATASET_SUBDIR = "eval_logs/datasets"

# Script behavior configuration
NUM_SAMPLES_TO_RUN = 1000  # Increased number of samples to process
MAX_CONCURRENT_REQUESTS = 800  # Maximum concurrent API requests - set to your rps capacity
TIMEOUT_SECONDS = 60  # Timeout for API calls
MAX_REQUESTS_PER_SECOND = 800.0  # Default max requests per second
MIN_REQUEST_INTERVAL = 0.001  # Minimum interval between requests

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
        print(
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
            self.tokens = 0.0  # We've used our token
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
            print("Cannot check OpenRouter limits: No valid API key")
            return None

        current_usage = None  # Initialize to None
        try:
            print("Checking OpenRouter rate limits...")
            response = requests.get(
                url="https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()

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
                                float(rps) * 0.8, MAX_REQUESTS_PER_SECOND
                            )

                            if new_rate != self.max_requests_per_second:
                                self.max_requests_per_second = new_rate
                                limit_adjusted = True
                                current_rate_for_log = new_rate

                # Log credits/usage if available
                usage = account_data.get("usage")
                limit = account_data.get("limit")
                limit_remaining = account_data.get("limit_remaining")

                # Set initial usage on first check
                if usage is not None:
                    current_usage = float(usage)  # Store the usage
                    if self.initial_usage is None:
                        self.initial_usage = current_usage
                        print(
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
                        old_self_rate = self.max_requests_per_second
                        self.max_requests_per_second = min(
                            self.max_requests_per_second, 10.0
                        )
                        if old_self_rate != self.max_requests_per_second:
                            log_message_parts.append(
                                f"LOW CREDITS - RPS reduced to {self.max_requests_per_second:.1f}!"
                            )

                print(", ".join(log_message_parts))
                self.last_limits_check_time = time.time()
            else:
                print(
                    f"Failed to get OpenRouter account status: HTTP {response.status_code}"
                )

        except Exception as e:
            print(f"Error checking OpenRouter limits: {e}")
            return None  # Return None on exception

        self.last_limits_check_time = time.time()
        return current_usage  # Return the fetched usage

# Create a singleton rate limiter instance
rate_limiter = RateLimiter(
    max_requests_per_second=MAX_REQUESTS_PER_SECOND,
    min_interval=MIN_REQUEST_INTERVAL,
    bucket_capacity=200,  # Allow bursts of up to 200 requests
    jitter=0.005,  # Add up to 5ms of random jitter to prevent synchronization
)

# Try to load environment variables from .env file in venv directory
venv_env_path = os.path.join("venv", ".env")
if os.path.isfile(venv_env_path):
    load_dotenv(venv_env_path)
    print(f"Loaded environment variables from {venv_env_path}")

# Get API key from environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# --- System Prompt for ListOps ---
LISTOPS_SYSTEM_PROMPT = """You will be given a ListOps equation to evaluate. ListOps is a language for performing operations on lists of numbers.

!! CRITICAL OPERATOR RULES !!

SM (Sum Modulo 10):
- Calculate the sum of all numbers, THEN take modulo 10 (remainder when divided by 10)
- Example: (SM 6 7 8) = (6+7+8) % 10 = 21 % 10 = 1
- Example: (SM 95 98 97) = (95+98+97) % 10 = 290 % 10 = 0
- COMMON ERROR: SM is NOT the same as SUM

AVG (Average with Floor Division):
- Calculate the sum of numbers DIVIDED BY the count of numbers
- ALWAYS ROUND DOWN (floor division) to the nearest integer
- Example: (AVG 1 2 8) = (1+2+8)/3 = 11/3 = 3.67 → ROUND DOWN to 3
- COMMON ERROR: Never round up or to nearest integer

MED (Median with Floor Division for Even Lists):
- For odd-length lists: middle value after sorting
- For even-length lists: average of two middle values ROUNDED DOWN
- Example: (MED 1 3 5 7) = Sorted [1,3,5,7] → middle values (3+5)/2 = 4
- COMMON ERROR: Forgetting to round down for even-length lists

MAX (Maximum):
- Returns the LARGEST value from the list
- Example: (MAX 3 9 1 7) = 9
- COMMON ERROR: Confusing with SUM (never add the values)

MIN (Minimum):
- Returns the SMALLEST value from the list
- Example: (MIN 3 9 1 7) = 1
- COMMON ERROR: Confusing with other operations

SUM (Regular Sum):
- Adds ALL numbers together
- Example: (SUM 1 2 3) = 1+2+3 = 6
- COMMON ERROR: Confusing with SM (never take modulo 10)

The operators you will encounter include:
*   `SUM`: Calculates the regular sum of the numbers in the list. Example: `(SUM 1 2 3)` results in `6`.
*   `SM`: Calculates the sum modulo 10 (the remainder after dividing the sum by 10). 
     - Example 1: `(SM 1 2 3)` results in `(1+2+3) % 10 = 6`.
     - Example 2: `(SM 18 9 7)` results in `(18+9+7) % 10 = 34 % 10 = 4`.
     - Example 3: `(SM 95 98 97)` results in `(95+98+97) % 10 = 290 % 10 = 0`.
*   `AVG`: Calculates the arithmetic mean of the numbers in the list, rounded DOWN to the nearest integer (floor division). 
     - Example 1: `(AVG 1 2 6)` results in `(1+2+6)/3 = 9/3 = 3`.
     - Example 2: `(AVG 1 2 8)` = `(1+2+8)/3 = 11/3 = 3.67`, which rounds DOWN to `3`.
     - Example 3: `(AVG 7 8)` = `(7+8)/2 = 15/2 = 7.5`, which rounds DOWN to `7`.
*   `MIN`: Finds the minimum value among the numbers in the list. Example: `(MIN 1 2 3)` results in `1`.
*   `MAX`: Finds the maximum value among the numbers in the list. Example: `(MAX 1 2 3)` results in `3`.
*   `MED`: Finds the median value in the list. 
     - For lists with odd length, it's the middle value after sorting. Example: `(MED 1 3 5)` results in `3`.
     - For lists with even length, it's the average of the two middle values, rounded DOWN. Example: `(MED 1 3 5 7)` results in `(3+5)/2 = 4`.
     - Example with even length and rounding: `(MED 2 4 6 9)` = `(4+6)/2 = 5`.

IMPORTANT RULES:
1. `SM` is NOT the same as `SUM`:
   - SM takes the sum modulo 10 (remainder after dividing by 10)
   - If you calculate (SM 8 7 9 6), you must first sum all numbers: 8+7+9+6 = 30
   - Then take modulo 10: 30 % 10 = 0
   - So (SM 8 7 9 6) = 0, NOT 30
2. For ALL calculations involving division (AVG and sometimes MED), always round DOWN to the nearest integer.
   - Example: 11/3 = 3.67 → round DOWN to 3 (not up to 4)
   - Example: 7.5 → round DOWN to 7 (not up to 8)
3. Always evaluate the innermost expressions first, and then work your way outward.
4. The final answer must be a single integer with no explanations.

Direct Comparison between operators:
- (SUM 3 4 5) = 3+4+5 = 12
- (SM 3 4 5) = (3+4+5) % 10 = 12 % 10 = 2
- (MAX 3 4 5) = 5
- (AVG 3 4 5) = (3+4+5)/3 = 12/3 = 4

For deeply nested expressions, the key is to methodically work inside-out:

Example 1: `(SUM (MIN 1 2) (MAX 3 4))`
1. First evaluate `(MIN 1 2)` = 1
2. Then evaluate `(MAX 3 4)` = 4
3. Finally, evaluate `(SUM 1 4)` = 5

Example 2: `(AVG (MAX (MIN 1 3 5) (MIN 2 4 6)) (SUM 7 8))`
1. First evaluate `(MIN 1 3 5)` = 1
2. Then evaluate `(MIN 2 4 6)` = 2
3. Then evaluate `(MAX 1 2)` = 2
4. Then evaluate `(SUM 7 8)` = 15
5. Finally, evaluate `(AVG 2 15)` = 17/2 = 8.5 rounded DOWN to `8`

Example 3 (More Complex): `(MED (SM (MAX (MIN 8 3 6) (AVG 7 9 11)) (SUM 5 4)) (MIN 6 2) (MAX 9 7))`
1. Evaluate `(MIN 8 3 6)` = 3
2. Evaluate `(AVG 7 9 11)` = 27/3 = 9
3. Evaluate `(MAX 3 9)` = 9
4. Evaluate `(SUM 5 4)` = 9
5. Evaluate `(SM 9 9)` = (9 + 9) % 10 = 18 % 10 = 8  <-- Note: This is modulo 10, NOT simply 18
6. Evaluate `(MIN 6 2)` = 2
7. Evaluate `(MAX 9 7)` = 9
8. Finally, evaluate `(MED 8 2 9)` = [2, 8, 9] (sorted) → middle value = 8

Example 4 (Focus on SM): `(SM (MAX 15 8 12) (SM 5 6 7) (MIN 10 2 9))`
1. Evaluate `(MAX 15 8 12)` = 15
2. Evaluate `(SM 5 6 7)` = (5+6+7) % 10 = 18 % 10 = 8
3. Evaluate `(MIN 10 2 9)` = 2
4. Finally, evaluate `(SM 15 8 2)` = (15+8+2) % 10 = 25 % 10 = 5

Example 5 (Focus on AVG): `(AVG 7 8 15 18 9)`
1. Sum the numbers: 7+8+15+18+9 = 57
2. Count the numbers: 5
3. Divide: 57/5 = 11.4
4. Round DOWN: floor(11.4) = 11

Example 6 (Focus on MED with even list): `(MED 5 9 1 8)`
1. Sort the numbers: [1, 5, 8, 9]
2. Even length list (4 elements), so take the two middle values: 5 and 8
3. Average them: (5+8)/2 = 6.5
4. Round DOWN: floor(6.5) = 6
5. Result: 6

For evaluation of very complex/deep expressions with 6+ levels of nesting, break it down systematically:
1. First, identify all innermost parentheses (those without nested parentheses inside them)
2. Evaluate each of those and replace with their results
3. Repeat the process, working outward until you reach the final result

Remember, ListOps requires you to track the exact operation at each step:
- Use ONLY the appropriate operator at each step (SUM, SM, MAX, MIN, AVG, MED)
- Never mix up operators (e.g., applying MAX when you should be applying SUM)
- For operations with multiple steps like SM, always complete all steps (sum first, then modulo 10)
- For AVG and MED with even lists, always round DOWN (floor division)

Your task is to evaluate the given ListOps equation and return ONLY the final integer result.
Do not include any explanations, reasoning, or any other text besides the numerical answer.
For example, if the equation evaluates to 42, your response should be exactly '42'.
"""

# Create detailed logs directory
LOGS_DIR = "detailed_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def get_timestamp():
    """Return a formatted timestamp for log files"""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def log_interaction(model_name, sample_id, equation, system_prompt, raw_response, parsed_answer, ground_truth, is_correct):
    """Log the full details of an LLM interaction to a file"""
    log_filename = f"{LOGS_DIR}/llm_interactions_{get_timestamp()}.jsonl"
    
    log_entry = {
        "timestamp": get_timestamp(),
        "model": model_name,
        "sample_id": sample_id,
        "equation": equation,
        "system_prompt": system_prompt,
        "raw_response": raw_response,
        "parsed_answer": parsed_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct
    }
    
    with open(log_filename, "a", encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    return log_filename

# --- Helper Functions ---

def load_samples(dataset_path, num_samples):
    """Loads a specified number of samples from a JSONL dataset file."""
    samples = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line {i+1}: {e}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return []
    return samples

def get_listops_equation(sample):
    """Extracts the ListOps equation (ast_str) from a sample."""
    return sample.get("ast_str")

def parse_llm_response(response_text):
    """Parses the LLM response to extract a single integer."""
    if not response_text:
        return None, "Error: Empty response"
    
    if "Error: Invalid Equation" in response_text: # Check for explicit error from LLM
        return None, "Error: Invalid Equation (reported by LLM)"
    
    # First look for answers that are just a clean number (possibly with whitespace)
    clean_text = response_text.strip()
    if clean_text.isdigit() or (clean_text.startswith('-') and clean_text[1:].isdigit()):
        return int(clean_text), None
    
    # Check for final answer statements - these should have highest priority
    final_answer_patterns = [
        r"(?:final answer|final result|answer is|result is)[^\d-]*(-?\d+)",  # "final answer: 42"
        r"(?:the final answer is|the answer is)[^\d-]*(-?\d+)",  # More specific patterns
        r"(?:final integer result|final integer answer)[^\d-]*(-?\d+)",
        r"(?<=\bfinal\s+answer\s*[:=]?\s*)(-?\d+)",  # Lookahead/behind for exact matches
        r"\bfinal\s*[:=]\s*(-?\d+)",  # "final: 42"
    ]
    
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1]), None
            except ValueError:
                continue
    
    # Check for boxed answers in LaTeX/Markdown format (common in math explanations)
    boxed_patterns = [
        r"\$\\boxed\{(-?\d+)\}\$",  # LaTeX boxed: $\boxed{42}$
        r"\$?\\boxed\{(-?\d+)\}\$?",  # Variations with optional outer $
        r"\$\{(-?\d+)\}\$",  # ${42}$
        r"\\boxed\{(-?\d+)\}",  # \boxed{42}
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            try:
                return int(matches[-1]), None
            except ValueError:
                continue
    
    # Check for the last line containing only a number (common in LLM responses)
    lines = response_text.strip().split('\n')
    for line in reversed(lines):  # Process lines from bottom to top
        line = line.strip()
        if line and (line.isdigit() or (line.startswith('-') and line[1:].isdigit())):
            return int(line), None
    
    # Look for other answer patterns
    answer_patterns = [
        r"answer:?\s*(-?\d+)",  # "answer: 42" or "Answer: 42"
        r"(?:equals|=)[^\d-]*(-?\d+)",  # "equals 42" or "= 42"
        r"(?<=\s)(-?\d+)(?=\s*$)",  # Number at the end of text with space before
        r"(?:\s|^)(-?\d+)(?:\s*$)",  # Number as the last token with space before
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1]), None
            except ValueError:
                continue
    
    # Code block or inline code patterns
    code_patterns = [
        r"`(-?\d+)`",  # `42`
        r"```(?:.*\n)?(-?\d+)(?:\n.*)?```",  # ```42``` or ```python\n42\n```
        r"'(-?\d+)'",  # '42'
        r'"(-?\d+)"',  # "42"
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if matches:
            try:
                return int(matches[-1]), None
            except ValueError:
                continue
    
    # Check for obvious numeric patterns with decimal points and convert to integers
    float_patterns = [
        r"(-?\d+\.\d+)",  # Standard decimal
        r"(-?\d+),(\d+)",  # European format with comma
    ]
    
    for pattern in float_patterns:
        if pattern == r"(-?\d+),(\d+)":
            # Handle European format
            comma_matches = re.findall(pattern, response_text)
            if comma_matches:
                try:
                    # Combine whole and decimal part
                    whole, decimal = comma_matches[-1]
                    float_val = float(f"{whole}.{decimal}")
                    # Always round down for ListOps
                    return int(float_val), None
                except ValueError:
                    continue
        else:
            # Handle standard decimals
            float_matches = re.findall(pattern, response_text)
            if float_matches:
                try:
                    # Take the last match and convert to integer by flooring (rounding down)
                    return int(float(float_matches[-1])), None
                except ValueError:
                    continue
    
    # Last resort: look for any integer in the text and take the last one
    # This is more permissive but can help with unusual formats
    integer_matches = re.findall(r"(?<!\S)(-?\d+)(?!\S|\.\d)", response_text)
    if integer_matches:
        try:
            # Use the last integer found, which is most likely to be the final answer
            return int(integer_matches[-1]), None
        except ValueError:
            return None, f"Error: Regex matched non-integer '{integer_matches[-1]}' in '{response_text}'"
    
    return None, f"Error: No integer found in response"

def eval_with_tolerance(predicted, ground_truth, tolerance=2):
    """Evaluates if the predicted answer matches the ground truth with adaptive tolerance based on value magnitude."""
    if predicted is None or ground_truth is None:
        return False
    
    try:
        pred_val = int(predicted)
        gt_val = int(ground_truth)
        
        # Exact match
        if pred_val == gt_val:
            return True
        
        # Small values (0-20): Be very strict
        if abs(gt_val) <= 20:
            return abs(pred_val - gt_val) <= 1  # Allow off by at most 1
            
        # Medium values (21-100): Allow small absolute difference
        elif abs(gt_val) <= 100:
            return abs(pred_val - gt_val) <= tolerance
            
        # Large values (101-1000): Allow small percentage difference
        elif abs(gt_val) <= 1000:
            percentage_diff = abs((pred_val - gt_val) / gt_val)
            return percentage_diff <= 0.05  # Within 5%
            
        # Very large values (>1000): Allow slightly larger percentage difference
        else:
            percentage_diff = abs((pred_val - gt_val) / gt_val)
            return percentage_diff <= 0.08  # Within 8%
            
    except (ValueError, TypeError, ZeroDivisionError):
        return False

def verify_ground_truth(sample_id, equation, ground_truth):
    """
    Attempts to verify if the provided ground truth value is correct by computing the equation.
    
    Args:
        sample_id: Identifier for the sample
        equation: The ListOps equation string
        ground_truth: The claimed ground truth value
        
    Returns:
        A tuple of (is_verified, computed_value, explanation)
    """
    try:
        # Clean up the equation if needed
        clean_equation = equation.strip()
        
        # Use our evaluation function to compute the correct answer
        computed_value = eval_listops_expression(clean_equation)
        
        if computed_value == int(ground_truth):
            return True, computed_value, "Ground truth verified"
        else:
            return False, computed_value, f"Ground truth {ground_truth} doesn't match computed value {computed_value}"
    except Exception as e:
        return False, None, f"Error verifying ground truth: {str(e)}"

async def call_openrouter_llm_async(client, model_config, equation_str, system_prompt):
    """Async version to call a specified LLM via OpenRouter with the ListOps equation."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Evaluate this ListOps equation: {equation_str}\nPlease give only the final numeric answer."}
    ]

    # Set common parameters that encourage precise, direct answers
    request_params = {
        "model": model_config["model_id"],
        "messages": messages,
        "temperature": model_config["temperature"],
        "max_tokens": model_config["max_completion_tokens"],
        "timeout": TIMEOUT_SECONDS,
        "top_p": 0.95,  # Focus on higher probability tokens
        "frequency_penalty": 0.0,  # No penalty for token repetition
        "presence_penalty": 0.0,   # No penalty for new tokens
    }

    # Handle reasoning_settings parameter - OpenRouter may not support this directly
    # so we'll move it to extra_body if it exists
    if model_config.get("reasoning_settings") is not None:
        # Create or update extra_body with reasoning_settings
        extra_body = request_params.get("extra_body", {}) or {}
        extra_body["reasoning_settings"] = model_config["reasoning_settings"]
        request_params["extra_body"] = extra_body
    
    start_time = time.time()
    
    try:
        # Wait based on rate limiter before making API call
        wait_time = rate_limiter.wait_if_needed()
        if wait_time > 0:
            print(f"Rate limiter: waited {wait_time:.2f}s before API call to {model_config['display_name']}")
        
        completion = await client.chat.completions.create(**request_params)
        end_time = time.time()
        
        response_content = completion.choices[0].message.content.strip() if completion.choices[0].message.content else ""
        parsed_answer, error_msg = parse_llm_response(response_content)
        
        usage = completion.usage if hasattr(completion, 'usage') else None
        tokens_used = usage.total_tokens if usage else None
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None

        return {
            "model_display_name": model_config["display_name"],
            "raw_response": response_content,
            "parsed_answer": parsed_answer,
            "error": error_msg,
            "time_taken": end_time - start_time,
            "tokens_used": tokens_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    except Exception as e:
        end_time = time.time()
        error_msg = str(e)
        print(f"API Error for model {model_config['display_name']}: {error_msg}")
        
        # If we hit a rate limit error, adjust rate limiter and wait longer
        if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
            # Reduce the rate limit by 20%
            with rate_limiter.lock:
                old_rate = rate_limiter.max_requests_per_second
                new_rate = max(1.0, old_rate * 0.8)  # Don't go below 1 req/s
                rate_limiter.max_requests_per_second = new_rate
                print(f"Rate limit hit. Reducing rate: {old_rate:.1f} → {new_rate:.1f} req/s")
            
            wait_time = 5 + random.uniform(0, 5)  # Random backoff between 5-10 seconds
            print(f"Waiting {wait_time:.1f} seconds before retry...")
            await asyncio.sleep(wait_time)
            
        return {
            "model_display_name": model_config["display_name"],
            "raw_response": None,
            "parsed_answer": None,
            "error": f"API Error: {error_msg}",
            "time_taken": end_time - start_time,
            "tokens_used": None,
            "prompt_tokens": None,
            "completion_tokens": None
        }

async def process_task(sem, client, task, model_metrics, system_prompt):
    """Process a single task with rate limiting and retries"""
    async with sem:
        model_conf = task["model_conf"]
        sample_id = task["sample_id"]
        equation = task["equation"]
        ground_truth = task["ground_truth"]
        
        print(f"Processing: Sample {sample_id} with model {model_conf['display_name']}")
        
        # Try up to 3 times with backoff
        max_retries = 3
        for retry_attempt in range(max_retries):
            try:
                result = await call_openrouter_llm_async(client, model_conf, equation, system_prompt)
                
                # If successful, break out of retry loop
                if not (result.get("error") and "API Error" in result.get("error", "")):
                    break
                    
                # If we hit a rate limit error, adjust rate limiter and wait longer
                if result.get("error") and ("rate limit" in result.get("error").lower() or 
                                          "too many requests" in result.get("error").lower()):
                    wait_time = (2 ** retry_attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    print(f"Rate limit error. Retry {retry_attempt+1}/{max_retries}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # For other API errors, wait a shorter time
                    wait_time = (2 ** retry_attempt) * 2
                    print(f"API error. Retry {retry_attempt+1}/{max_retries}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                # For unexpected exceptions, log and wait
                error_msg = str(e)
                print(f"Exception in process_task for {model_conf['display_name']}: {error_msg}")
                
                if retry_attempt < max_retries - 1:
                    wait_time = (2 ** retry_attempt) * 3
                    print(f"Unexpected error. Retry {retry_attempt+1}/{max_retries}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # On last attempt, create error result
                    result = {
                        "model_display_name": model_conf["display_name"],
                        "raw_response": None,
                        "parsed_answer": None,
                        "error": f"Unrecoverable error after {max_retries} attempts: {error_msg}",
                        "time_taken": 0,
                        "tokens_used": None,
                        "prompt_tokens": None,
                        "completion_tokens": None
                    }
        
        # Track performance metrics
        is_correct = False
        if ground_truth is not None and result.get("parsed_answer") is not None:
            try:
                gt_int = int(ground_truth)
                model_metrics[model_conf["display_name"]]["total"] += 1
                
                # Use tolerance-based evaluation
                is_correct = eval_with_tolerance(result["parsed_answer"], ground_truth)
                if is_correct:
                    model_metrics[model_conf["display_name"]]["correct"] += 1
                elif result.get("error"):
                    model_metrics[model_conf["display_name"]]["errors"] += 1
                    
                if result.get("time_taken", 0) > 0:
                    model_metrics[model_conf["display_name"]]["avg_time"].append(result["time_taken"])
            except (ValueError, TypeError):
                print(f"Warning: Could not compare ground truth '{ground_truth}' for sample {sample_id}")
                
        # Log the full interaction details
        log_file = log_interaction(
            model_conf["display_name"], 
            sample_id, 
            equation, 
            system_prompt, 
            result.get("raw_response"), 
            result.get("parsed_answer"), 
            ground_truth, 
            is_correct
        )
        
        # Add sample information to result
        result.update({
            "sample_id": sample_id,
            "equation": equation,
            "ground_truth": ground_truth,
            "model": model_conf["display_name"],
            "detailed_log": log_file,
            "is_correct": is_correct
        })
        
        # Print info
        print(f"  Model: {model_conf['display_name']}")
        if result.get("error") and result.get("parsed_answer") is None:
            print(f"    Error: {result.get('error')}")
            if result.get('raw_response'):
                 print(f"    Raw Response: '{result.get('raw_response', '')[:50]}...'")
        else:
            print(f"    Parsed Answer: {result.get('parsed_answer')}")
            if result.get("error"):
                 print(f"    Parsing Note: {result.get('error')}")
            print(f"    Raw Response: '{result.get('raw_response', '')[:50]}...'")
        
        time_taken = result.get("time_taken", 0)
        tokens_used = result.get("tokens_used", "N/A")
        print(f"    Time: {time_taken:.2f}s, Tokens: {tokens_used}")
        print(f"    Correct: {is_correct} (Ground Truth: {ground_truth})")
        
        return result

async def run_evaluations(client, samples, models):
    """Run evaluations concurrently with rate limiting"""
    tasks = []
    model_metrics = {model_conf["display_name"]: {"correct": 0, "total": 0, "errors": 0, "avg_time": []} 
                     for model_conf in models}
    
    # Create all tasks
    sample_tasks = {}  # Group tasks by sample_id for later ground truth verification
    
    for sample in samples:
        sample_id = sample.get("id", f"Unknown")
        equation = get_listops_equation(sample)
        ground_truth = sample.get("ground_truth_value")
        
        if not equation:
            print(f"Sample {sample_id}: Skipping, no 'ast_str' (equation) found.")
            continue
        
        # Create a task for each model
        for model_conf in models:
            task = {
                "model_conf": model_conf,
                "sample_id": sample_id,
                "equation": equation,
                "ground_truth": ground_truth
            }
            tasks.append(task)
            
            # Also store by sample_id for ground truth verification
            if sample_id not in sample_tasks:
                sample_tasks[sample_id] = {
                    "equation": equation,
                    "ground_truth": ground_truth,
                    "model_results": []
                }
    
    # Check if we have any tasks
    if not tasks:
        print("No valid tasks to process.")
        return []
    
    print(f"Processing {len(tasks)} evaluation tasks across {len(samples)} samples with {len(models)} models.")
    
    # Update concurrent requests based on rate limiter settings
    dynamic_concurrent = min(MAX_CONCURRENT_REQUESTS, int(rate_limiter.max_requests_per_second * 1.5))
    print(f"Using dynamic concurrency level: {dynamic_concurrent} concurrent requests")
    
    # Create semaphore for limiting concurrent requests
    sem = asyncio.Semaphore(dynamic_concurrent)
    
    # Process tasks with semaphore to limit concurrency
    print("Starting concurrent processing...")
    start_time = time.time()
    results = []
    
    # Create a list of coroutines for the tasks
    coroutines = [process_task(sem, client, task, model_metrics, LISTOPS_SYSTEM_PROMPT) for task in tasks]
    
    # Split into manageable chunks to avoid memory issues with very large datasets
    chunk_size = 500
    chunks = [coroutines[i:i + chunk_size] for i in range(0, len(coroutines), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} tasks)")
        chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
        
        # Handle exceptions and store results
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"Task error: {result}")
                continue
            if result:  # Skip None results
                results.append(result)
                
                # Also store in sample_tasks for ground truth verification
                sample_id = result.get("sample_id")
                if sample_id in sample_tasks:
                    model_result = {
                        "model": result.get("model"),
                        "parsed_answer": result.get("parsed_answer")
                    }
                    sample_tasks[sample_id]["model_results"].append(model_result)
    
    # Calculate elapsed time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Update metrics with sample verification
    ground_truth_verifications = {}
    suspicious_gt_samples = 0
    
    for sample_id, sample_data in sample_tasks.items():
        # Verify ground truth if we have at least 3 models with answers
        model_results = sample_data["model_results"]
        if len(model_results) >= 3:
            is_verified, computed_value, explanation = verify_ground_truth(
                sample_id, sample_data["equation"], sample_data["ground_truth"]
            )
            
            ground_truth_verifications[sample_id] = {
                "is_verified": is_verified,
                "computed_value": computed_value,
                "explanation": explanation
            }
            
            if not is_verified:
                suspicious_gt_samples += 1
    
    # Print summary metrics by model
    total_correct = 0
    total_samples = 0
    total_errors = 0
    
    print("\n----- Evaluation Results -----")
    print(f"Processed {len(results)} tasks in {total_time:.2f} seconds")
    print(f"Average time per task: {total_time/len(results):.4f} seconds")
    
    model_table = []
    for model_name, metrics in model_metrics.items():
        avg_time = sum(metrics["avg_time"]) / len(metrics["avg_time"]) if metrics["avg_time"] else 0
        accuracy = (metrics["correct"] / metrics["total"]) * 100 if metrics["total"] > 0 else 0
        
        model_table.append({
            "Model": model_name,
            "Accuracy": f"{accuracy:.2f}%",
            "Correct": metrics["correct"],
            "Total": metrics["total"],
            "Errors": metrics["errors"],
            "Avg Time": f"{avg_time:.4f}s"
        })
        
        total_correct += metrics["correct"]
        total_samples += metrics["total"]
        total_errors += metrics["errors"]
    
    # Sort by accuracy
    model_table.sort(key=lambda x: float(x["Accuracy"].replace("%", "")), reverse=True)
    
    # Print model table
    print("\nModel Performance:")
    header = f"{'Model':<25} {'Accuracy':<10} {'Correct':<10} {'Total':<10} {'Errors':<10} {'Avg Time':<10}"
    print(header)
    print("-" * len(header))
    
    for row in model_table:
        print(f"{row['Model']:<25} {row['Accuracy']:<10} {row['Correct']:<10} {row['Total']:<10} {row['Errors']:<10} {row['Avg Time']:<10}")
    
    # Print overall metrics
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print("\nOverall Metrics:")
    print(f"Total Correct: {total_correct}/{total_samples} ({overall_accuracy:.2f}%)")
    print(f"Total Errors: {total_errors}")
    
    # Ground truth verification summary
    print(f"\nGround Truth Verification:")
    print(f"Suspicious Ground Truth Values: {suspicious_gt_samples} samples")
    
    # Save ground truth verification data
    if ground_truth_verifications:
        verification_file = f"ground_truth_verification_{get_timestamp()}.json"
        with open(verification_file, "w") as f:
            json.dump(ground_truth_verifications, f, indent=2)
        print(f"Ground truth verification data saved to {verification_file}")
    
    return results

# --- New function to create simplified log file ---
def create_simple_answers_log(results, timestamp=None):
    """Creates a simple log file with just raw responses, equations, and ground truth."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"simple_answers_{timestamp}.log"
    
    with open(log_filename, "w") as f:
        f.write("# Simple Answers Log\n")
        f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Format: Sample ID, Equation, Ground Truth, LLM Raw Response\n\n")
        f.write("# Note: ListOps Operators:\n")
        f.write("#   SUM: Regular sum\n")
        f.write("#   SM: Sum modulo 10 (sum % 10)\n")
        f.write("#   AVG: Average with floor division\n")
        f.write("#   MIN: Minimum value\n")
        f.write("#   MAX: Maximum value\n")
        f.write("#   MED: Median value\n\n")
        
        for sample_id, data in sorted(results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            f.write(f"Sample {sample_id}:\n")
            f.write(f"Equation: {data['equation']}\n")
            
            # Verify ground truth using our parser
            gt_value = data['ground_truth']
            try:
                computed_value = eval_listops_expression(data['equation'])
                if int(gt_value) == computed_value:
                    f.write(f"Ground Truth: {gt_value} (✓ Verified)\n")
                else:
                    f.write(f"Ground Truth: {gt_value} (⚠️ Incorrect, should be {computed_value})\n")
            except:
                f.write(f"Ground Truth: {gt_value} (? Unable to verify)\n")
            
            # Add model responses
            for i, response in enumerate(data['responses']):
                model_name = response['model']
                raw_response = response['raw_response']
                parsed = response['parsed_answer']
                is_correct = response['is_correct']
                
                # Add is_correct_with_computed marker
                is_correct_with_computed = False
                try:
                    is_correct_with_computed = (parsed is not None and parsed == computed_value)
                except:
                    pass
                
                # Format correctness indicators
                gt_indicator = "✓" if is_correct else "✗"
                computed_indicator = "✓" if is_correct_with_computed else "✗"
                
                f.write(f"\nModel {i+1} ({model_name}):\n")
                f.write(f"Raw Response: {raw_response}\n")
                f.write(f"Parsed Answer: {parsed} ({gt_indicator} vs Ground Truth, {computed_indicator} vs Computed)\n")
            
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"Created simple answers log: {log_filename}")
    return log_filename

# --- Main Execution ---
async def main_async():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set it before running the script: export OPENROUTER_API_KEY='your_key_here'")
        return

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "ListOps Eval Script"}
    )

    # Initialize rate limiter and check initial API rate limits
    print("Initializing rate limiter and checking API limits...")
    initial_usage = rate_limiter.update_limits_from_api()
    if initial_usage is not None:
        print(f"Initial API usage: ${initial_usage:.4f}")

    # Look for dataset files in the subdirectory
    dataset_dir_path = os.path.join(DEFAULT_DATASET_SUBDIR, DEFAULT_DATASET_DIRNAME)
    
    # Find the first .jsonl file in the directory
    dataset_full_path = None
    if os.path.isdir(dataset_dir_path):
        for file in os.listdir(dataset_dir_path):
            if file.endswith(".jsonl"):
                dataset_full_path = os.path.join(dataset_dir_path, file)
                break
    
    if not dataset_full_path:
        print(f"Error: No .jsonl files found in {dataset_dir_path}")
        return
    
    print(f"Using dataset from: {dataset_full_path}")
    
    # Load samples
    samples = load_samples(dataset_full_path, NUM_SAMPLES_TO_RUN)
    if not samples:
        print("Error: No samples loaded from dataset.")
        return
    
    print(f"Successfully loaded {len(samples)} samples from the dataset.")
    
    # Run evaluations
    print("\nRunning evaluations...")
    print(f"Using rate limiter with max {rate_limiter.max_requests_per_second} req/s")
    
    # Create directory for detailed results if it doesn't exist
    timestamp = get_timestamp()
    results_dir = f"equation_llm_evaluation_results/[MAIN]_DATASET_{DEFAULT_DATASET_DIRNAME}_equation_llm_eval_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    results_summary = await run_evaluations(client, samples, EVAL_CONFIG)
    end_time = time.time()
    
    # Calculate total elapsed time
    total_elapsed = end_time - start_time
    
    # Create the simplified answers log file
    timestamp = get_timestamp()
    # Organize results by sample ID for the simple log
    by_sample = {}
    for result in results_summary:
        sample_id = result.get("sample_id")
        if sample_id not in by_sample:
            by_sample[sample_id] = {
                "equation": result.get("equation"),
                "ground_truth": result.get("ground_truth"),
                "responses": []
            }
        
        by_sample[sample_id]["responses"].append({
            "model": result.get("model"),
            "raw_response": result.get("raw_response"),
            "parsed_answer": result.get("parsed_answer"),
            "is_correct": result.get("is_correct")
        })

    simple_log_filename = create_simple_answers_log(by_sample, timestamp)
    
    # Check final API usage if we got the initial usage
    if initial_usage is not None:
        final_usage = rate_limiter.update_limits_from_api()
        if final_usage is not None:
            run_cost = final_usage - initial_usage
            print(f"\nAPI Usage:")
            print(f"Initial: ${initial_usage:.4f}")
            print(f"Final: ${final_usage:.4f}")
            print(f"Total Cost: ${run_cost:.4f}")
            print(f"Cost per sample: ${run_cost/len(samples):.6f}")
    
    # Print completion message
    print(f"\nTotal elapsed time: {total_elapsed:.2f} seconds")
    print(f"Simple log file created at: {simple_log_filename}")
    
    # Final report
    print("\nEvaluation complete! Full detailed logs are available in:")
    print(f"- {LOGS_DIR} (raw API interactions)")
    print(f"- {simple_log_filename} (simplified answers log)")

    # Return summary for any external callers
    return results_summary

def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())

def eval_listops_expression(expr_str):
    """
    Evaluates a ListOps expression according to the proper rules.
    
    Args:
        expr_str: A string containing a ListOps expression like "(MAX (MIN 1 2) 3)"
        
    Returns:
        The evaluated result as an integer
    """
    # Import re if not already imported
    import re
    
    # Tokenize the expression
    tokens = re.findall(r'\(|\)|SUM|SM|MIN|MAX|MED|AVG|\d+', expr_str)
    
    def parse_expr(tokens, idx=0):
        if tokens[idx] != '(':
            # Must be a number
            return int(tokens[idx]), idx + 1
        
        # Get operator
        operator = tokens[idx + 1]
        idx += 2  # Skip '(' and operator
        
        # Parse operands
        operands = []
        while idx < len(tokens) and tokens[idx] != ')':
            if tokens[idx] == '(':
                val, next_idx = parse_expr(tokens, idx)
                operands.append(val)
                idx = next_idx
            elif tokens[idx].isdigit():
                operands.append(int(tokens[idx]))
                idx += 1
            else:
                idx += 1
        
        # Evaluate based on operator
        result = None
        if operator == "SUM":
            result = sum(operands)
        elif operator == "SM":
            result = sum(operands) % 10  # SM is sum modulo 10
        elif operator == "AVG":
            result = sum(operands) // len(operands)  # Integer division (floor)
        elif operator == "MIN":
            result = min(operands)
        elif operator == "MAX":
            result = max(operands)
        elif operator == "MED":
            sorted_ops = sorted(operands)
            if len(sorted_ops) % 2 == 1:
                # Odd length, take middle element
                result = sorted_ops[len(sorted_ops) // 2]
            else:
                # Even length, take average of two middle elements and floor it
                mid1 = sorted_ops[len(sorted_ops) // 2 - 1]
                mid2 = sorted_ops[len(sorted_ops) // 2]
                result = (mid1 + mid2) // 2
        
        return result, idx + 1  # Skip ')'
    
    result, _ = parse_expr(tokens)
    return result

def verify_ground_truths(equations_with_gt):
    """
    Verifies ground truth values for a set of equations.
    
    Args:
        equations_with_gt: A dictionary mapping sample IDs to dictionaries with 'equation' and 'ground_truth' keys
        
    Returns:
        A dictionary with verification results
    """
    results = {}
    for sample_id, data in equations_with_gt.items():
        equation = data.get('equation')
        ground_truth = data.get('ground_truth')
        
        if not equation or ground_truth is None:
            continue
            
        try:
            # Manually evaluate the equation
            computed_value = eval_listops_expression(equation)
            
            # Compare with ground truth
            is_correct = int(ground_truth) == computed_value
            
            results[sample_id] = {
                'equation': equation,
                'ground_truth': int(ground_truth),
                'computed_value': computed_value,
                'is_correct': is_correct
            }
            
        except Exception as e:
            results[sample_id] = {
                'equation': equation,
                'ground_truth': int(ground_truth) if ground_truth else None,
                'error': str(e)
            }
    
    return results

if __name__ == "__main__":
    main()