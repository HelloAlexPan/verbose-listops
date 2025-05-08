import os
import json
import concurrent.futures
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
import re

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Recommended to use a fast and capable model for validation.
MODEL_FOR_VALIDATION = os.environ.get("VALIDATION_MODEL", "google/gemini-flash-1.5") 
MAX_WORKERS = int(os.environ.get("VALIDATION_MAX_WORKERS", 10))
LOG_LEVEL = logging.INFO
# Default dataset path, can be overridden by command-line argument
DEFAULT_DATASET_FILE_PATH = "datasets/DATASET_10000tok_8-mxops_3-arity_6-mxbrch_google_gemini-2.5-pro-preview-03-25_20250508-1200.jsonl"

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI Client for OpenRouter ---
client = None
if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
    logger.error("OpenRouter API Key not found or is a placeholder. Please set the OPENROUTER_API_KEY environment variable.")
else:
    try:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        logger.info(f"OpenAI client configured for OpenRouter with model {MODEL_FOR_VALIDATION}.")
    except Exception as e:
        logger.error(f"Failed to configure OpenAI client for OpenRouter: {e}")

# --- API Call Function ---
def get_llm_response(prompt_text: str, sample_id: str) -> str | None:
    if not client:
        logger.error(f"[{sample_id}] OpenAI client not initialized. Skipping API call.")
        return None

    # Basic retry mechanism
    for attempt in range(3): # Retry up to 3 times
        try:
            logger.debug(f"[{sample_id}] Attempting API call (Attempt {attempt + 1}/3)")
            system_prompt = (
                "You are an expert AI assistant. Your task is to read the provided text, "
                "which contains a story and a question at the end. The question asks for a "
                "single integer result based on the story. "
                "Your response MUST be ONLY that single integer. Do not include any other text, "
                "explanations, or formatting. For example, if the answer is 42, your response should be '42'."
            )

            response = client.chat.completions.create(
                model=MODEL_FOR_VALIDATION,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=30,  # Answer should be short (integer + potential minor surrounding text)
                temperature=0.0, # We want deterministic extraction
            )
            llm_output = response.choices[0].message.content
            logger.debug(f"[{sample_id}] Raw LLM response: '{llm_output}'")
            return llm_output
        except Exception as e:
            logger.warning(f"[{sample_id}] API call attempt {attempt + 1} failed: {e}")
            if attempt < 2: # If not the last attempt
                sleep_time = 2 ** attempt # Exponential backoff: 1, 2 seconds
                logger.info(f"[{sample_id}] Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"[{sample_id}] API call failed after 3 attempts.")
                return None
    return None # Should be unreachable if loop completes, but as a fallback.

# --- Answer Parsing ---
def parse_llm_answer(llm_response_text: str, sample_id: str) -> int | None:
    if not llm_response_text:
        logger.warning(f"[{sample_id}] LLM response was empty or None.")
        return None
    
    cleaned_response = llm_response_text.strip()
    
    # Attempt direct conversion if the stripped response is purely an integer
    if re.fullmatch(r"-?\d+", cleaned_response):
        try:
            return int(cleaned_response)
        except ValueError:
            # This should not happen if re.fullmatch passed, but as a safeguard
            logger.warning(f"[{sample_id}] Matched pure integer but failed to convert: '{cleaned_response}'")
            pass # Fall through to regex search

    # If not a pure integer, search for numbers in the text.
    # This regex finds integers, possibly with leading/trailing non-numeric characters.
    matches = re.findall(r"-?\d+", cleaned_response)
    
    if matches:
        if len(matches) > 1:
            logger.warning(f"[{sample_id}] Multiple numbers found in response: {matches}. Using the first one: {matches[0]}. Full response: '{llm_response_text}'")
        try:
            return int(matches[0])
        except ValueError:
            logger.warning(f"[{sample_id}] Found number via regex but failed to convert: '{matches[0]}' from '{llm_response_text}'")
            return None
    else:
        logger.warning(f"[{sample_id}] No integer found in LLM response: '{llm_response_text}'")
        return None

# --- Process a Single Sample ---
def validate_sample(sample: dict) -> dict:
    sample_id = sample.get("id", "Unknown_ID")
    narrative_prompt = sample.get("narrative_prompt")
    ground_truth_answer_str = sample.get("ground_truth_answer") # Ground truth is int in data

    if narrative_prompt is None or ground_truth_answer_str is None:
        logger.error(f"[{sample_id}] Sample is missing 'narrative_prompt' or 'ground_truth_answer'. Skipping.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "Missing critical data in sample.",
            "llm_response": None,
            "parsed_answer": None,
            "expected_answer": ground_truth_answer_str
        }
    
    try:
        ground_truth_answer = int(ground_truth_answer_str)
    except ValueError:
        logger.error(f"[{sample_id}] Ground truth answer '{ground_truth_answer_str}' is not a valid integer. Skipping.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "Invalid ground_truth_answer format in sample.",
            "llm_response": None,
            "parsed_answer": None,
            "expected_answer": ground_truth_answer_str
        }

    logger.info(f"[{sample_id}] Validating sample...")
    llm_response_text = get_llm_response(narrative_prompt, sample_id)
    
    if llm_response_text is None:
        logger.error(f"[{sample_id}] Failed to get LLM response.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "LLM call failed or returned no response.",
            "llm_response": None,
            "parsed_answer": None,
            "expected_answer": ground_truth_answer
        }
        
    parsed_llm_answer = parse_llm_answer(llm_response_text, sample_id)

    if parsed_llm_answer is None:
        logger.warning(f"[{sample_id}] Could not parse an integer answer from LLM. Expected: {ground_truth_answer}, Got raw: '{llm_response_text}'")
        return {
            "id": sample_id,
            "status": "incorrect", # Treat parsing failure as incorrect for validation purposes
            "reason": "Could not parse integer from LLM response.",
            "llm_response": llm_response_text,
            "parsed_answer": None,
            "expected_answer": ground_truth_answer
        }

    is_correct = (parsed_llm_answer == ground_truth_answer)
    status = "correct" if is_correct else "incorrect"
    logger.info(f"[{sample_id}] Result: {status.upper()}. Expected: {ground_truth_answer}, Got: {parsed_llm_answer}.")
    
    return {
        "id": sample_id,
        "status": status,
        "reason": "Comparison complete.",
        "llm_response": llm_response_text,
        "parsed_answer": parsed_llm_answer,
        "expected_answer": ground_truth_answer
    }

# --- Load Dataset ---
def load_dataset(file_path: str) -> list:
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping invalid JSON line {line_num} in {file_path}: {e}. Line: '{line.strip()}'")
        logger.info(f"Loaded {len(dataset)} samples from {file_path}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
    return dataset

# --- Main ---
def main(dataset_file_path: str):
    if not client:
        logger.error("OpenAI client for OpenRouter could not be initialized. Aborting validation.")
        return

    dataset = load_dataset(dataset_file_path)
    if not dataset:
        logger.warning(f"No samples loaded from {dataset_file_path}. Exiting.")
        return

    results = []
    logger.info(f"Starting validation of {len(dataset)} samples with {MAX_WORKERS} workers using model {MODEL_FOR_VALIDATION}...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='ValidatorWorker') as executor:
        future_to_sample_id = {executor.submit(validate_sample, sample): sample.get("id", f"sample_{i}") for i, sample in enumerate(dataset)}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_sample_id)):
            sample_id_for_log = future_to_sample_id[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logger.error(f"[{sample_id_for_log}] Generated an exception during validation task: {exc}", exc_info=True)
                results.append({
                    "id": sample_id_for_log, # type: ignore
                    "status": "error",
                    "reason": f"Task execution exception: {exc}",
                    "llm_response": None,
                    "parsed_answer": None,
                    "expected_answer": "N/A due to exception"
                })
            finally:
                if (i + 1) % (len(dataset) // 10 if len(dataset) >=10 else 1) == 0 or (i + 1) == len(dataset): # Log progress roughly every 10% or at the end
                     logger.info(f"Processed {i + 1}/{len(dataset)} samples...")


    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Validation finished in {total_time:.2f} seconds.")

    correct_count = sum(1 for r in results if r["status"] == "correct")
    incorrect_count = sum(1 for r in results if r["status"] == "incorrect")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    total_processed = len(results)
    
    if total_processed > 0:
        accuracy = (correct_count / total_processed) * 100 if total_processed > 0 else 0
        valid_comparisons = correct_count + incorrect_count # Samples where an LLM answer was successfully parsed
        accuracy_of_comparisons = (correct_count / valid_comparisons) * 100 if valid_comparisons > 0 else 0

        logger.info("--- Validation Summary ---")
        logger.info(f"Dataset: {dataset_file_path}")
        logger.info(f"Validation Model: {MODEL_FOR_VALIDATION}")
        logger.info(f"Total Samples Attempted: {len(dataset)}")
        logger.info(f"Total Samples Processed (results collected): {total_processed}")
        logger.info(f"Correct: {correct_count}")
        logger.info(f"Incorrect (LLM answer parsed but differs): {incorrect_count}")
        logger.info(f"Errors (API call, parsing, or other issues): {error_count}")
        logger.info(f"Overall Accuracy (Correct / Total Processed): {accuracy:.2f}%")
        if error_count > 0 or incorrect_count > 0 : # Show this only if there were issues
            logger.info(f"Accuracy of Parsed Responses (Correct / (Correct + Incorrect)): {accuracy_of_comparisons:.2f}%")

        # Log details of incorrect and error samples for easier debugging
        for r in results:
            if r["status"] == "incorrect":
                logger.warning(f"DETAILS INCORRECT: ID: {r['id']}, Expected: {r['expected_answer']}, Got: {r['parsed_answer']}, Raw LLM: '{r['llm_response'][:150].replace('\n', ' ')}...'")
            elif r["status"] == "error" and r['reason'] != "Missing critical data in sample." and r['reason'] != "Invalid ground_truth_answer format in sample.": # Don't log details for data format errors again
                logger.error(f"DETAILS ERROR: ID: {r['id']}, Reason: {r['reason']}, Expected: {r['expected_answer']}")
    else:
        logger.info("No samples were processed or no results collected.")

if __name__ == "__main__":
    dataset_path_to_use = DEFAULT_DATASET_FILE_PATH
    # Allow overriding dataset path via command line argument for flexibility
    import sys
    if len(sys.argv) > 1:
        dataset_path_to_use = sys.argv[1]
        logger.info(f"Using dataset path from command line: {dataset_path_to_use}")
    else:
        logger.info(f"Using default dataset path: {dataset_path_to_use}")
        
    if not os.path.exists(dataset_path_to_use):
        logger.error(f"CRITICAL: Dataset file '{dataset_path_to_use}' does not exist. Please check the path.")
        logger.error("You can specify a dataset path as a command-line argument: python validator.py path/to/your/dataset.jsonl")
    elif not client:
        logger.error("CRITICAL: OpenRouter client not initialized. Check OPENROUTER_API_KEY. Exiting.")
    else:
        main(dataset_path_to_use)
