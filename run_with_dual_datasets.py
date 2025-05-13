#!/usr/bin/env python3
"""
run_with_dual_datasets.py

This script runs verbose-listops.py to generate the main dataset, and then 
automatically runs dual_dataset_generator.py to create the validation and benchmark datasets.

Usage:
  python3 run_with_dual_datasets.py [num_samples] [max_workers]
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and handle its output"""
    print(f"\n=== Running {description} ===")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n{description} completed successfully with exit code {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n{description} failed with error: {e}")
        return False

def main():
    # Get command line arguments
    args = sys.argv[1:]
    num_samples = args[0] if len(args) > 0 else "100"  # Default to 100 samples
    max_workers = args[1] if len(args) > 1 else "10"   # Default to 10 workers
    
    # Current directory for the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to verbose-listops.py
    verbose_listops_path = os.path.join(script_dir, "verbose-listops.py")
    
    # Path to dual_dataset_generator.py
    dual_dataset_generator_path = os.path.join(script_dir, "dual_dataset_generator.py")
    
    # Check if the scripts exist
    if not os.path.exists(verbose_listops_path):
        print(f"Error: verbose-listops.py not found at {verbose_listops_path}")
        return 1
        
    if not os.path.exists(dual_dataset_generator_path):
        print(f"Error: dual_dataset_generator.py not found at {dual_dataset_generator_path}")
        return 1
    
    # Run verbose-listops.py
    verbose_listops_cmd = [
        sys.executable,
        verbose_listops_path,
        # Add additional arguments if needed
    ]
    
    # For verbose-listops.py, we actually don't pass num_samples and max_workers on command line
    # as they're defined in the constants in the file. We'd need to modify it to accept args.
    
    # Set environment variable to indicate this is a production run
    os.environ["PROD_RUN"] = "true"
    
    if not run_command(verbose_listops_cmd, "verbose-listops.py"):
        print("Verbose-listops.py failed. Not running dual dataset generator.")
        return 1
    
    # Calculate validation and benchmark sample counts based on main dataset
    validation_samples = min(100, max(20, int(num_samples) // 10))  # 10% of main, between 20-100
    benchmark_samples = min(1000, max(100, int(num_samples)))  # Similar to main size, capped at 1000
    dual_max_workers = min(int(max_workers), 50)  # Limit workers for dual generation
    
    # Run dual_dataset_generator.py
    dual_dataset_cmd = [
        sys.executable,
        dual_dataset_generator_path,
        str(validation_samples), 
        str(benchmark_samples),
        str(dual_max_workers)
    ]
    
    run_command(dual_dataset_cmd, "dual_dataset_generator.py")
    
    print("\n=== All processing completed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 