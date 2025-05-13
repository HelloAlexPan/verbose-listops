#!/usr/bin/env python3
"""
dual_dataset_generator.py

Generates two distinct datasets for the verbose-listops benchmark:
1. A validation dataset - smaller, focused on quality checks for ML researchers to validate sample integrity
2. A benchmark dataset - larger, for evaluating LLM performance on the reasoning task

For questions, contact: [your contact info]
"""

import os
import json
import concurrent.futures
import random
import datetime
import logging
import time
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the Python path if not already there
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from verbose_listops module (file is verbose-listops.py but module name uses underscore)
try:
    # First try with underscore
    from verbose_listops import (
        Config,
        NUM_SAMPLES_TO_GENERATE,
        DEFAULT_MAX_WORKERS,
        DATASETS_DIR,
        MODEL,
        logger,
        generate_single_sample,
        generation_token_tracker,
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS,
    )
except ImportError:
    # If that fails, try with dash by manually importing the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("verbose_listops", os.path.join(current_dir, "verbose-listops.py"))
    verbose_listops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verbose_listops)
    
    # Now access from the module
    Config = verbose_listops.Config
    NUM_SAMPLES_TO_GENERATE = verbose_listops.NUM_SAMPLES_TO_GENERATE
    DEFAULT_MAX_WORKERS = verbose_listops.DEFAULT_MAX_WORKERS
    DATASETS_DIR = verbose_listops.DATASETS_DIR
    MODEL = verbose_listops.MODEL
    logger = verbose_listops.logger
    generate_single_sample = verbose_listops.generate_single_sample
    generation_token_tracker = verbose_listops.generation_token_tracker
    DEFAULT_COST_PER_MILLION_PROMPT_TOKENS = verbose_listops.DEFAULT_COST_PER_MILLION_PROMPT_TOKENS
    DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS = verbose_listops.DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS

# Setup logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
script_logger = logging.getLogger("dual_generator")


def generate_validation_dataset(num_samples=100, max_workers=10):
    """Generate a smaller dataset specifically for validation purposes"""
    
    # Define validation dataset config
    validation_config = Config()
    validation_config.MAX_OPS = 8
    validation_config.MAX_TOTAL_TOKENS = 10000
    
    # Create output filename - much simpler naming convention
    sanitized_model_name = MODEL.replace("/", "_").replace(":", "-")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    
    validation_output = os.path.join(
        DATASETS_DIR,
        f"verbose-listops-validation-dataset_{timestamp}.jsonl",
    )
    
    script_logger.info(f"Generating {num_samples} validation samples...")
    print(f"Generating {num_samples} validation samples...")
    
    validation_samples = []
    start_time = time.time()
    
    # Sequential generation for validation dataset for better control
    for i in range(num_samples):
        try:
            sample = generate_single_sample(i, validation_config)
            if sample:
                # For validation dataset, we need to include markup about beats vs padding
                enhanced_sample = {
                    "id": sample.get("id", str(i)),
                    "ast_prefix": sample.get("ast_prefix", sample.get("ast", "")),
                    "ground_truth": sample.get("ground_truth_answer", sample.get("ground_truth", 0)),
                    "generator_model": MODEL,
                    "generation_timestamp": datetime.datetime.now().isoformat(),
                    "dataset_type": "validation",
                }
                
                # Add markup to the narrative showing beats vs padding
                scenes = sample.get("scenes", [])
                marked_narrative = []
                
                # First scene is usually intro
                if scenes and len(scenes) > 0:
                    marked_narrative.append(f"<!-- INTRODUCTION -->\n{scenes[0]}")
                
                # Identify beats and padding sections
                beat_counter = sample.get("beat_counter", {}).get("total", 0)
                for idx, scene in enumerate(scenes[1:], 1):
                    # The first N scenes after intro are beats (where N is beat_counter)
                    # Everything else is padding
                    if idx <= beat_counter:
                        beat_op = "UNKNOWN_OPERATION"
                        if sample.get("meta", {}).get("operations", None):
                            # If operations metadata is available, use it
                            operations = sample.get("meta", {}).get("operations", [])
                            if idx-1 < len(operations):
                                beat_op = operations[idx-1]
                        marked_narrative.append(f"<!-- BEAT {idx}: {beat_op} -->\n{scene}")
                    else:
                        marked_narrative.append(f"<!-- PADDING -->\n{scene}")
                
                # Join with double newlines for readability
                enhanced_sample["narrative_with_markup"] = "\n\n".join(marked_narrative)
                validation_samples.append(enhanced_sample)
                
                # Show progress
                print(f"\rGenerated validation sample {len(validation_samples)}/{num_samples}", end="")
        except Exception as e:
            script_logger.error(f"Error generating validation sample {i}: {e}", exc_info=True)
    
    # Write samples to file
    if validation_samples:
        with open(validation_output, 'w', encoding='utf-8') as f_val:
            for sample in validation_samples:
                f_val.write(json.dumps(sample, default=lambda o: list(o) if isinstance(o, set) else str(o), ensure_ascii=False) + '\n')
        
        time_taken = time.time() - start_time
        script_logger.info(f"Validation dataset generation completed in {time_taken:.2f} seconds. {len(validation_samples)}/{num_samples} samples saved.")
        print(f"\nValidation dataset saved to: {validation_output}")
        return validation_output
    else:
        script_logger.error("Failed to generate any valid validation samples.")
        print("\nFailed to generate any valid validation samples.")
        return None


def generate_benchmark_dataset(num_samples=1000, max_workers=DEFAULT_MAX_WORKERS):
    """Generate a larger dataset for benchmarking LLMs"""
    
    # Define benchmark dataset config
    benchmark_config = Config()
    benchmark_config.MAX_OPS = 8
    benchmark_config.MAX_TOTAL_TOKENS = 10000
    
    # Create output filename - much simpler naming convention
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    benchmark_output = os.path.join(
        DATASETS_DIR,
        f"verbose-listops-benchmark-dataset_{timestamp}.jsonl",
    )
    
    script_logger.info(f"Generating {num_samples} benchmark samples using {max_workers} workers...")
    print(f"Generating {num_samples} benchmark samples using {max_workers} workers...")
    
    benchmark_samples = []
    start_time = time.time()
    
    # Parallel generation for benchmark dataset
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(generate_single_sample, i, benchmark_config): i
            for i in range(num_samples)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample = future.result()
                if sample:
                    # Keep only essential fields for the benchmark dataset
                    minimal_sample = {
                        "id": sample.get("id", str(index)),
                        "ast_prefix": sample.get("ast_prefix", sample.get("ast", "")),
                        "ground_truth": sample.get("ground_truth_answer", sample.get("ground_truth", 0)),
                        "narrative_with_question": sample.get("narrative_with_question", ""),
                        "generator_model": MODEL,
                        "generation_timestamp": datetime.datetime.now().isoformat(),
                        "dataset_type": "benchmark"
                    }
                    benchmark_samples.append(minimal_sample)
                completed += 1
                
                # Show progress
                progress_pct = (completed / num_samples) * 100
                print(f"\rGenerated benchmark sample {completed}/{num_samples} ({progress_pct:.1f}%)", end="")
            except Exception as e:
                script_logger.error(f"Error generating benchmark sample {index}: {e}")
                completed += 1
    
    # Write samples to file
    if benchmark_samples:
        with open(benchmark_output, 'w', encoding='utf-8') as f_bench:
            for sample in benchmark_samples:
                f_bench.write(json.dumps(sample, default=lambda o: list(o) if isinstance(o, set) else str(o), ensure_ascii=False) + '\n')
        
        time_taken = time.time() - start_time
        script_logger.info(f"Benchmark dataset generation completed in {time_taken:.2f} seconds. {len(benchmark_samples)}/{num_samples} samples saved.")
        print(f"\nBenchmark dataset saved to: {benchmark_output}")
        return benchmark_output
    else:
        script_logger.error("Failed to generate any valid benchmark samples.")
        print("\nFailed to generate any valid benchmark samples.")
        return None


def main():
    """Main function to generate both datasets"""
    print("\n" + "=" * 80)
    print("VERBOSE LISTOPS DUAL DATASET GENERATOR")
    print("=" * 80)
    
    # Parse command line arguments
    validation_samples = 100  # Default
    benchmark_samples = NUM_SAMPLES_TO_GENERATE  # Default from verbose-listops
    max_workers = DEFAULT_MAX_WORKERS
    
    # Check if the user provided command line arguments
    if len(sys.argv) > 1:
        try:
            validation_samples = int(sys.argv[1])
            if len(sys.argv) > 2:
                benchmark_samples = int(sys.argv[2])
            if len(sys.argv) > 3:
                max_workers = int(sys.argv[3])
        except ValueError:
            print("Error: Arguments must be integers.")
            print("Usage: python dual_dataset_generator.py [validation_samples] [benchmark_samples] [max_workers]")
            return
    
    # Reset token tracker for clean tracking
    generation_token_tracker.total_prompt_tokens = 0
    generation_token_tracker.total_completion_tokens = 0
    generation_token_tracker.api_calls = 0
    
    # First generate validation dataset
    print("\n" + "=" * 80)
    print(f"STEP 1: GENERATING VALIDATION DATASET ({validation_samples} samples)")
    print("=" * 80)
    
    validation_output = generate_validation_dataset(validation_samples, max_workers=min(10, max_workers))
    
    # Then generate benchmark dataset
    print("\n" + "=" * 80)
    print(f"STEP 2: GENERATING BENCHMARK DATASET ({benchmark_samples} samples)")
    print("=" * 80)
    
    benchmark_output = generate_benchmark_dataset(benchmark_samples, max_workers)
    
    # Print summary
    print("\n" + "=" * 80)
    print("DATASET GENERATION SUMMARY")
    print("=" * 80)
    if validation_output:
        print(f"Validation dataset: {validation_output}")
    else:
        print("Validation dataset: GENERATION FAILED")
    
    if benchmark_output:
        print(f"Benchmark dataset: {benchmark_output}")
    else:
        print("Benchmark dataset: GENERATION FAILED")
    
    print("\nDatasets are saved in: " + os.path.abspath(DATASETS_DIR))
    
    # Token usage and cost summary
    gen_prompt_tokens, gen_completion_tokens, gen_api_calls = generation_token_tracker.get_summary()
    estimated_generation_cost = generation_token_tracker.calculate_cost(
        DEFAULT_COST_PER_MILLION_PROMPT_TOKENS,
        DEFAULT_COST_PER_MILLION_COMPLETION_TOKENS,
    )
    print(f"\nToken usage summary:")
    print(f"- Total API calls: {gen_api_calls}")
    print(f"- Total prompt tokens: {gen_prompt_tokens}")
    print(f"- Total completion tokens: {gen_completion_tokens}")
    print(f"- Estimated cost: ${estimated_generation_cost:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        script_logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nError: {e}") 