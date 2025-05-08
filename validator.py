import os
import json
import concurrent.futures
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
import re
import argparse

# --- Initial configuration logging ---
# Replace lines 9-31 with more concise setup info
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Log API key status (safely, without exposing the key)
if OPENROUTER_API_KEY:
    key_preview = OPENROUTER_API_KEY[:4] + "..." + OPENROUTER_API_KEY[-4:] if len(OPENROUTER_API_KEY) > 8 else "***"
    print(f"API key: {key_preview} ✓")
else:
    print("⚠️ WARNING: OpenRouter API key not found!")

# Recommended to use a fast and capable model for validation.
MODEL_FOR_VALIDATION = os.environ.get("VALIDATION_MODEL", "google/gemini-2.5-pro-preview-03-25") 
MAX_WORKERS = int(os.environ.get("VALIDATION_MAX_WORKERS", 100))  # Increased to better use 1k RPS capability
LOG_LEVEL = logging.DEBUG
# Default dataset path, can be overridden by command-line argument
DEFAULT_DATASET_FILE_PATH = "datasets/DATASET_10000tok_8-mxops_3-arity_6-mxbrch_google_gemini-2.5-pro-preview-03-25_20250508-1414.jsonl"
# This default is now handled by argparse

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI Client for OpenRouter ---
client = None
if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
    logger.error("⚠️ No valid API key found. Please set OPENROUTER_API_KEY in .env file.")
    print("⚠️ ERROR: API key missing or invalid.")
else:
    try:
        print(f"Initializing client for model: {MODEL_FOR_VALIDATION}...")
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        print(f"Client initialized ✓")
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        print(f"⚠️ ERROR initializing client: {e}")

# --- System prompt for validation ---
# Template-based JSON approach to force proper formatting
VALIDATION_SYSTEM_PROMPT = """
You are an expert AI system meticulously validating samples from the Verbose ListOps benchmark. This benchmark tests narrative reasoning in LLMs by embedding hierarchical ListOps computations (MAX, MIN, MED, SUM, AVG, SM) within lengthy, coherent narratives.

The benchmark's key characteristics:
1. It embeds a nested ListOps computation task within a distracting but semantically coherent narrative
2. The narrative unfolds in post-order (inside-out) evaluation of the ListOps AST
3. Models must extract computational signals while filtering out relevant but task-irrelevant narrative content
4. It independently controls for both context length and reasoning complexity
5. "Incomplete Itemization with Correct Aggregation": A core design feature.
   - For a given operation, the narrative might explicitly list only *some* of the numerical inputs.
   - However, it *must* then provide an explicit aggregate total (e.g., "yielding a total of X", "resulting in X items") that *is the correct result of the current ListOps operation on ALL its inputs (including those not itemized and those from previous sub-operations)*.
   - This stated aggregate IS the value to be carried forward. If this aggregate is correct for the operation, the step is mathematically valid even if itemization was incomplete.

CRITICAL: Your ENTIRE response must be ONLY a valid JSON object with NO additional text.
Use this exact template, replacing values in [SQUARE_BRACKETS] with your analysis:

{
  "id": "[SAMPLE_ID]",
  "overall_status": "[VALID or INVALID_MATH or INVALID_NARRATIVE or INVALID_MATH_AND_NARRATIVE]",
  "final_ast_value": [INTEGER_RESULT],
  "matches_ground_truth": [true or false],
  "narrative_consistent": [true or false],
  "ast_evaluation_steps": [
    {
      "step": 1,
      "operation_node_description": "[Brief description of the AST node being evaluated, e.g., (SUM 10 20 (MIN 5 8))]",
      "operation_type": "[MAX, MIN, SUM, AVG, MED, SM]",
      "inputs": [LIST_OF_NUMBERS],
      "result": [INTEGER_RESULT],
      "narrative_consistent": [true or false],
      "itemization_complete": [true or false],
      "correct_aggregate_provided": [true or false],
      "explanation": "[BRIEF_EXPLANATION]"
    },
    // Additional steps...
  ],
  "narrative_analysis": {
    "strengths": ["[STRENGTH1]", "[STRENGTH2]", ...],
    "weaknesses": ["[WEAKNESS1]", "[WEAKNESS2]", ...],
    "inconsistencies": ["[INCONSISTENCY1]", "[INCONSISTENCY2]", ...]
  },
  "detailed_reason": "[Detailed explanation of any issues found, especially if overall_status is not VALID]",
  "summary": "[One-sentence summary of the validation outcome and primary reason if not VALID]"
}

ListOps operators:
- MAX: Maximum value of inputs
- MIN: Minimum value of inputs
- MED: Median value (if even count, the lower of the two middle values after sorting)
- SUM: Sum of inputs
- AVG: Integer floor of the sum of inputs divided by the count of inputs
- SM: Sum of inputs modulo 10 (result is sum % 10)

Guidelines for `narrative_consistent` field in `ast_evaluation_steps`:
- Mark `true` if the narrative segment for this step accurately and clearly conveys the operation, its inputs (or a correct aggregate that IS the operation's result), and its result. Minor stylistic awkwardness is acceptable if the mathematical progression is clear.
- Mark `false` for significant ambiguity, misleading information, if the stated aggregate does not match the operation's true result, or if the math is unclear.

DO NOT write any text outside of this JSON format, not even explanations or clarifications.
"""

# --- JSON Schema for validation output ---
VALIDATION_OUTPUT_SCHEMA = {
  "type": "object",
  "properties": {
    "id": {"type": "string"},
    "overall_status": {
      "type": "string", 
      "enum": ["VALID", "INVALID_MATH", "INVALID_NARRATIVE", "INVALID_MATH_AND_NARRATIVE"]
    },
    "final_ast_value": {"type": "integer"},
    "matches_ground_truth": {"type": "boolean"},
    "narrative_consistent": {"type": "boolean"},
    "ast_evaluation_steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "step": {"type": "integer"},
          "operation_node_description": {"type": "string"},
          "operation_type": {"type": "string"},
          "inputs": {"type": "array", "items": {"type": "integer"}},
          "result": {"type": "integer"},
          "narrative_consistent": {"type": "boolean"},
          "itemization_complete": {"type": "boolean"},
          "correct_aggregate_provided": {"type": "boolean"},
          "explanation": {"type": "string"}
        }
      }
    },
    "narrative_analysis": {
      "type": "object",
      "properties": {
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "inconsistencies": {"type": "array", "items": {"type": "string"}}
      }
    },
    "detailed_reason": {"type": "string"},
    "summary": {"type": "string"}
  },
  "required": ["id", "overall_status", "final_ast_value", "matches_ground_truth", "narrative_consistent", "ast_evaluation_steps", "summary"]
}

# --- Construct User Prompt ---
def construct_user_prompt(sample: dict) -> str:
    """
    Formats the sample data into a concise, structured user prompt for the LLM.
    """
    # Don't include a preview - we need the full narrative for proper validation
    prompt_data = {
        "id": sample.get("id", ""),
        "ast_prefix": sample.get("ast_prefix", ""),
        "ground_truth_answer": sample.get("ground_truth_answer", "")
    }
    
    # Simple prompt that highlights the structure needed
    user_prompt = f"""
Validate this ListOps dataset sample. Your response must be ONLY valid JSON:

ID: {prompt_data['id']}
AST: {prompt_data['ast_prefix']}
Expected Answer: {prompt_data['ground_truth_answer']}

Narrative (full version used for validation):
{sample.get("narrative_prompt", "")}

Perform a detailed evaluation:
1. Evaluate each step of the AST, showing your work. For each step, identify the operation type, its inputs, and its result.
2. For each step, check if the narrative correctly represents the operation and numbers with particular attention to:
   - `itemization_complete`: Does the narrative explicitly list all numerical inputs for the current operation?
   - `correct_aggregate_provided`: Does the narrative state an aggregate value (e.g., "totaling X", "resulting in X") that IS THE CORRECT RESULT of the current ListOps operation on all its inputs (including results from sub-operations)? This is key. If true, the step is mathematically sound for progression, even if itemization was incomplete.
3. Analyze the narrative for strengths, weaknesses, and inconsistencies.
4. Provide detailed reasoning for any issues found.

Key Validation Principle:
- "Incomplete Itemization with Correct Aggregation": If the narrative provides an explicit aggregate that IS the correct result of the current operation, the step is valid for calculation purposes, even if not all individual inputs were listed.
- `narrative_consistent`: Judge if the narrative for the step is clear and mathematically sound. Minor stylistic issues are okay. Major issues include ambiguity that obscures the math or misleading statements.

RESPOND ONLY WITH THE JSON OBJECT MATCHING THE TEMPLATE - NO OTHER TEXT.
"""
    return user_prompt

# --- API Call Function ---
def get_llm_response(sample: dict, sample_id: str) -> str | None:
    if not client:
        logger.error(f"[{sample_id}] Client not initialized.")
        return None

    user_prompt = construct_user_prompt(sample)
    
    # Define a proper JSON schema for validation results
    validation_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The sample ID from the input"
            },
            "overall_status": {
                "type": "string",
                "enum": ["VALID", "INVALID_MATH", "INVALID_NARRATIVE", "INVALID_MATH_AND_NARRATIVE"],
                "description": "Whether the sample is valid based on math and narrative consistency"
            },
            "final_ast_value": {
                "type": "integer",
                "description": "The final calculated value from evaluating the AST"
            },
            "matches_ground_truth": {
                "type": "boolean",
                "description": "Whether the calculated value matches the expected ground truth"
            },
            "narrative_consistent": {
                "type": "boolean",
                "description": "Whether the narrative correctly represents all operations and numbers"
            },
            "ast_evaluation_steps": {
                "type": "array",
                "description": "Step-by-step evaluation of the AST",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "description": "Sequential step number in the evaluation"},
                        "operation_node_description": {"type": "string", "description": "Brief description of the AST node being evaluated, e.g., (SUM 10 20 (MIN 5 8))"},
                        "operation_type": {"type": "string", "enum": ["MAX", "MIN", "MED", "SUM", "AVG", "SM"], "description": "Type of ListOps operation performed (MAX, MIN, etc.)"},
                        "inputs": {"type": "array", "items": {"type": "integer"}, "description": "Input values for this step"},
                        "result": {"type": "integer", "description": "Result of this operation step"},
                        "narrative_consistent": {"type": "boolean", "description": "Whether the narrative segment for this step accurately and clearly conveys the operation, its inputs (or a correct aggregate that IS the operation's result), and its result. Minor stylistic issues are acceptable if math is clear. Mark false for significant ambiguity or misleading info."},
                        "itemization_complete": {"type": "boolean", "description": "Whether the narrative fully itemizes all individual numerical inputs for this operation"},
                        "correct_aggregate_provided": {"type": "boolean", "description": "Whether the narrative explicitly states an aggregate value that is the correct result of the current ListOps operation"},
                        "explanation": {"type": "string", "description": "Brief explanation of this step and any issues"}
                    },
                    "required": ["step", "operation_node_description", "operation_type", "inputs", "result", "narrative_consistent", "itemization_complete", "correct_aggregate_provided", "explanation"]
                }
            },
            "narrative_analysis": {
                "type": "object",
                "description": "Analysis of the narrative's strengths and weaknesses",
                "properties": {
                    "strengths": {"type": "array", "items": {"type": "string"}, "description": "Strong points of the narrative"},
                    "weaknesses": {"type": "array", "items": {"type": "string"}, "description": "Weak points or issues in the narrative"},
                    "inconsistencies": {"type": "array", "items": {"type": "string"}, "description": "Specific inconsistencies between narrative and AST"}
                }
            },
            "detailed_reason": {
                "type": "string",
                "description": "Detailed explanation of any issues found"
            },
            "summary": {
                "type": "string",
                "description": "One-sentence summary of the validation outcome and primary reason if not VALID"
            }
        },
        "required": ["id", "overall_status", "final_ast_value", "matches_ground_truth", "narrative_consistent", "ast_evaluation_steps", "summary"],
        "additionalProperties": False
    }
    
    # Basic retry mechanism
    for attempt in range(3): # Retry up to 3 times
        try:
            # More concise logging - remove redundant info
            if attempt > 0:
                print(f"Sample {sample_id}: Retry {attempt+1}/3")
            
            # Try to use json_schema response format if available
            try:
                response = client.chat.completions.create(
                    model=MODEL_FOR_VALIDATION,
                    messages=[
                        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=65000,  # Use maximum output token limit
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "validation_result",
                            "strict": True,
                            "schema": validation_schema
                        }
                    }
                )
            except Exception as schema_error:
                logger.debug(f"[{sample_id}] json_schema format not supported: {schema_error}")
                try:
                    # Try simple json_object if json_schema is not supported
                    response = client.chat.completions.create(
                        model=MODEL_FOR_VALIDATION,
                        messages=[
                            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=65000,  # Use maximum output token limit
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                except Exception as object_error:
                    logger.debug(f"[{sample_id}] json_object format not supported: {object_error}")
                    # Fall back to standard format with no response_format parameter
                    response = client.chat.completions.create(
                        model=MODEL_FOR_VALIDATION,
                        messages=[
                            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=65000,  # Use maximum output token limit
                        temperature=0.0
                    )
                
            llm_output = response.choices[0].message.content
            logger.debug(f"[{sample_id}] Raw LLM response: '{llm_output[:200]}...'") # Log just the start
            return llm_output
        except Exception as e:
            logger.warning(f"[{sample_id}] API call attempt {attempt + 1} failed: {e}")
            if attempt < 2: # If not the last attempt
                sleep_time = 2 ** attempt # Exponential backoff: 1, 2 seconds
                time.sleep(sleep_time)
            else:
                logger.error(f"[{sample_id}] API call failed after 3 attempts.")
                return None
    return None

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
    if llm_response_text.strip().startswith('{') and llm_response_text.strip().endswith('}'):
        logger.debug(f"[{sample_id}] Response appears to be in proper JSON format.")
    elif '```json' in llm_response_text:
        logger.debug(f"[{sample_id}] Response contains markdown-style JSON code blocks.")
    else:
        logger.debug(f"[{sample_id}] Response does not appear to be in JSON format. First 50 chars: {llm_response_text[:50]}")
    
    # Try multiple strategies to extract valid JSON
    
    # Strategy 1: Look for JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(.+?)\s*```', llm_response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            parsed_validation = json.loads(json_str)
            logger.debug(f"[{sample_id}] Successfully parsed LLM validation response from markdown code block.")
            return parsed_validation
        except json.JSONDecodeError as e:
            logger.debug(f"[{sample_id}] Found markdown code block but failed to parse as JSON: {e}. Content: {json_str[:100]}... Trying other methods.")
    
    # Strategy 2: Look for JSON between curly braces (might be missing outer formatting)
    # Find the first { and the last } that might contain a complete JSON object
    brace_match = re.search(r'(\{.+\})', llm_response_text, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(1)
        try:
            parsed_validation = json.loads(json_str)
            logger.debug(f"[{sample_id}] Successfully parsed LLM validation response from braces extraction.")
            return parsed_validation
        except json.JSONDecodeError as e:
            logger.debug(f"[{sample_id}] Found content in braces but failed to parse as JSON: {e}. Content: {json_str[:100]}... Trying other methods.")
    
    # Strategy 3: Try to clean the text and parse as JSON
    # Strip potential markdown, extra spaces, etc.
    json_str = llm_response_text.strip()
    
    # Remove any triple backticks
    json_str = re.sub(r'```(?:json)?|```', '', json_str)
    
    # Remove potential leading/trailing text not part of JSON
    # Look for first { and last } to find potential JSON boundaries
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')
    
    if start_idx >= 0 and end_idx > start_idx:
        json_str = json_str[start_idx:end_idx+1]
        
        try:
            parsed_validation = json.loads(json_str)
            logger.debug(f"[{sample_id}] Successfully parsed LLM validation response after cleaning.")
            return parsed_validation
        except json.JSONDecodeError as e:
            logger.debug(f"[{sample_id}] Failed to parse cleaned JSON string: {e}. Content: {json_str[:100]}... Trying one more approach.")
    
    # Strategy 4: Try to fix common JSON formatting errors
    try:
        # Replace single quotes with double quotes (a common error)
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        # Fix potential trailing commas in arrays and objects
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        parsed_validation = json.loads(json_str)
        logger.debug(f"[{sample_id}] Successfully parsed LLM validation response after fixing common JSON errors.")
        return parsed_validation
    except json.JSONDecodeError as e:
        logger.error(f"[{sample_id}] Failed to parse LLM validation response as JSON: {e}")
        logger.error(f"[{sample_id}] Raw response preview: {llm_response_text[:300]}...")
        return None

# --- Process a Single Sample ---
def validate_sample(sample: dict) -> dict:
    sample_id = sample.get("id", "Unknown_ID")
    
    # Check for required fields
    required_fields = ["id", "ast_prefix", "ground_truth_answer", "narrative_prompt"]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        logger.error(f"[{sample_id}] Sample is missing required fields: {', '.join(missing_fields)}. Skipping.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": f"Missing required fields: {', '.join(missing_fields)}",
            "llm_response": None,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_answer")
        }
    
    # Ensure ground_truth_answer is an integer
    try:
        if isinstance(sample["ground_truth_answer"], str):
            sample["ground_truth_answer"] = int(sample["ground_truth_answer"])
    except (ValueError, TypeError):
        logger.error(f"[{sample_id}] Ground truth answer '{sample['ground_truth_answer']}' is not a valid integer. Skipping.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "Invalid ground_truth_answer format in sample.",
            "llm_response": None,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_answer")
        }

    logger.info(f"[{sample_id}] Validating sample...")
    llm_response_text = get_llm_response(sample, sample_id)
    
    if llm_response_text is None:
        logger.error(f"[{sample_id}] Failed to get LLM response.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "LLM call failed or returned no response.",
            "llm_response": None,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_answer")
        }
        
    parsed_validation = parse_llm_validation_output(llm_response_text, sample_id)

    if parsed_validation is None:
        logger.warning(f"[{sample_id}] Could not parse validation results from LLM response.")
        return {
            "id": sample_id,
            "status": "error",
            "reason": "Could not parse validation results from LLM response.",
            "llm_response": llm_response_text,
            "parsed_validation": None,
            "ground_truth_answer": sample.get("ground_truth_answer")
        }
    
    # Map LLM's overall_status to our status categories
    overall_status = parsed_validation.get("overall_status", "").upper()
    
    if overall_status == "VALID":
        status = "correct"
    elif overall_status in ["INVALID_MATH", "INVALID_NARRATIVE", "INVALID_MATH_AND_NARRATIVE"]:
        status = "incorrect"
    else:
        status = "error"
        logger.warning(f"[{sample_id}] Unrecognized overall_status: {overall_status}")
    
    # Check if the ground truth matches what the LLM calculated
    matches_ground_truth = parsed_validation.get("matches_ground_truth", False)
    
    # Prepare detailed reason based on validation results
    if status != "correct":
        # Use the detailed_reason if available, otherwise fall back to summary
        reason = parsed_validation.get("detailed_reason", 
                 parsed_validation.get("summary", "Validation failed according to LLM analysis."))
        
        # Save detailed log for failed validations
        save_detailed_validation_log(sample_id, sample, parsed_validation)
    else:
        reason = "Sample validated successfully."
    
    logger.info(f"[{sample_id}] Result: {status.upper()}. LLM Overall status: {overall_status}. Matches ground truth: {matches_ground_truth}.")

    # Add more detailed logging based on narrative consistency
    if parsed_validation:
        for step_eval in parsed_validation.get("ast_evaluation_steps", []):
            if not step_eval.get("narrative_consistent"):
                 logger.warning(f"[{sample_id}] Step {step_eval.get('step')} (Op: {step_eval.get('operation_type')}) flagged as narrative inconsistent: {step_eval.get('explanation', 'No details')}")
    
    return {
        "id": sample_id,
        "status": status,
        "reason": reason,
        "llm_response": llm_response_text,
        "parsed_validation": parsed_validation,
        "ground_truth_answer": sample.get("ground_truth_answer")
    }

# --- Save Detailed Validation Log ---
def save_detailed_validation_log(sample_id, sample, validation_result):
    """
    Save a detailed validation log for failed samples to assist with debugging.
    """
    try:
        # Create directories if they don't exist
        log_dir = os.path.join("logs", "failed_validations")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a log file with detailed information
        log_file_path = os.path.join(log_dir, f"fail_validation_{sample_id}.json")
        
        # Prepare the detailed log data
        detailed_log = {
            "sample": {
                "id": sample.get("id"),
                "ast_prefix": sample.get("ast_prefix"),
                "ground_truth_answer": sample.get("ground_truth_answer"),
                # Truncate narrative for log file size
                "narrative_prompt_preview": sample.get("narrative_prompt", "")[:500] + "..." 
                if len(sample.get("narrative_prompt", "")) > 500 else sample.get("narrative_prompt", "")
            },
            "validation_result": validation_result
        }
        
        # Write the log file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_log, f, indent=2)
            
        logger.info(f"[{sample_id}] Saved detailed validation log to {log_file_path}")
    except Exception as e:
        logger.error(f"[{sample_id}] Failed to save detailed validation log: {e}")

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
    print(f"\n🔍 Validating {total_samples} samples with {MAX_WORKERS} workers using {MODEL_FOR_VALIDATION}")
    print("=" * 60)
    start_time = time.time()
    processed_count = 0
    
    # For time estimation
    sample_times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='Validator') as executor:
        future_to_sample_id = {executor.submit(validate_sample, sample): sample.get("id", f"sample_{i}") for i, sample in enumerate(dataset)}
        
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
                results.append({
                    "id": sample_id_for_log,
                    "status": "error",
                    "reason": f"Task exception: {exc}",
                    "llm_response": None,
                    "parsed_validation": None,
                    "ground_truth_answer": "N/A due to exception"
                })
            
            # Update progress
            processed_count += 1
            progress_pct = (processed_count / total_samples) * 100
            
            # Calculate time remaining
            elapsed = time.time() - start_time
            if sample_times:
                avg_time_per_sample = sum(sample_times) / len(sample_times)
                remaining_samples = total_samples - processed_count
                est_remaining_time = remaining_samples * avg_time_per_sample / MAX_WORKERS
                
                # Format time remaining
                if est_remaining_time > 60:
                    time_str = f"{est_remaining_time/60:.1f} min"
                else:
                    time_str = f"{est_remaining_time:.0f} sec"
                    
                # Progress bar (50 chars wide)
                bar_width = 40
                filled_width = int(progress_pct / 100 * bar_width)
                bar = '█' * filled_width + '░' * (bar_width - filled_width)
                
                # Inline progress update (overwrite previous line)
                print(f"\r[{bar}] {progress_pct:.1f}% ({processed_count}/{total_samples}) ETA: {time_str}", end='')
            else:
                print(f"\r({processed_count}/{total_samples}) {progress_pct:.1f}%", end='')

    # Print newline after progress bar
    print("\n" + "=" * 60)
    
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
    
    # Count by validation categories
    validation_categories = {
        "VALID": 0,
        "INVALID_MATH": 0,
        "INVALID_NARRATIVE": 0, 
        "INVALID_MATH_AND_NARRATIVE": 0,
        "UNDEFINED": 0  # For cases where we couldn't determine a category
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
        # valid_comparisons = correct_count + incorrect_count # Not used
        # accuracy_of_valid = (correct_count / valid_comparisons) * 100 if valid_comparisons > 0 else 0 # Not used

        # Cleaner summary format with visual indicators
        print("\n📊 VALIDATION RESULTS")
        print("=" * 60)
        print(f"Dataset: {os.path.basename(dataset_file_path)}")
        print(f"Model:   {MODEL_FOR_VALIDATION.split('/')[-1]}")
        print(f"✓ Correct:   {correct_count}/{total_processed} ({accuracy:.1f}%)")
        print(f"✗ Incorrect: {incorrect_count}/{total_processed} ({(incorrect_count/total_processed)*100:.1f}%)")
        print(f"⚠ Errors:    {error_count}/{total_processed} ({(error_count/total_processed)*100:.1f}%)")
        
        # Show breakdown by category if relevant
        if incorrect_count > 0 or error_count > 0 : # Show if any non-correct items
            print("\n🔍 VALIDATION CATEGORY BREAKDOWN (from LLM validator)")
            for category, count in validation_categories.items():
                if count > 0:  # Only show non-zero categories
                    category_display = category.replace("INVALID_", "").title().replace("_", " ")
                    symbol = "✓" if category == "VALID" else ("✗" if "INVALID" in category else "⚠")
                    print(f"{symbol} {category_display}: {count} ({(count/total_processed)*100:.1f}%)")
    else:
        print("❌ No samples were processed successfully.")

    # If there were errors, provide a note about log files
    if error_count > 0 or incorrect_count > 0 : # Broaden condition
        print("\n⚠️ Some samples were not 'VALID'. Check logs/failed_validations/ for details on specific failures.")

    # Write detailed results to output file if path is provided
    if output_results_path:
        try:
            with open(output_results_path, 'w', encoding='utf-8') as f_out:
                for res_item in results:
                    f_out.write(json.dumps(res_item) + "\n")
            logger.info(f"Full validation results summary saved to {output_results_path}")
            print(f"Full validation results summary saved to {output_results_path}")
        except Exception as e:
            logger.error(f"Failed to write validation results to {output_results_path}: {e}")
            print(f"ERROR: Failed to write validation results to {output_results_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Verbose ListOps dataset samples.")
    parser.add_argument("dataset_file_path", nargs='?', default=DEFAULT_DATASET_FILE_PATH, help="Path to the .jsonl dataset file to validate. Defaults to a predefined path if not provided.")
    parser.add_argument("--output-results", help="Path to save the detailed validation results for all samples (JSONL).")
    
    args = parser.parse_args()
        
    if not os.path.exists(args.dataset_file_path):
        print(f"❌ ERROR: Dataset file not found: '{args.dataset_file_path}'")
        print("Usage: python validator.py <path_to_dataset.jsonl> [--output-results <path_for_results.jsonl>]")
    elif not client:
        print("❌ ERROR: Client not initialized. Check API key.")
    else:
        run_validation_process(args.dataset_file_path, args.output_results)
