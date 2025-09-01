#!/usr/bin/env python3
"""
Script to generate missing score files for specific models.
This script evaluates result files and generates corresponding score files.
"""

import os
import sys
import subprocess
from pathlib import Path

# Define the missing models that need score generation
MISSING_MODELS = [
    "deepseek-ai_DeepSeek-V3.1-thinking-off",
    "deepseek-ai_DeepSeek-V3.1-thinking-on"
]

def generate_scores_for_model(model_name, result_dir="result", score_dir="score"):
    """Generate score files for a specific model."""
    print(f"Generating scores for model: {model_name}")
    
    # Check if result directory exists for this model
    model_result_dir = os.path.join(result_dir, model_name)
    if not os.path.exists(model_result_dir):
        print(f"Error: Result directory {model_result_dir} does not exist")
        return False
    
    # Run the evaluation command
    try:
        cmd = [
            sys.executable, "-m", "bfcl_eval", "evaluate",
            "--model", model_name,
            "--result-dir", result_dir,
            "--score-dir", score_dir
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"Successfully generated scores for {model_name}")
            print(f"Output: {result.stdout}")
            return True
        else:
            print(f"Error generating scores for {model_name}")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Exception occurred while generating scores for {model_name}: {e}")
        return False

def main():
    """Main function to generate missing score files."""
    print("Starting score generation for missing models...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Create score directory if it doesn't exist
    os.makedirs("score", exist_ok=True)
    
    success_count = 0
    total_models = len(MISSING_MODELS)
    
    for model_name in MISSING_MODELS:
        if generate_scores_for_model(model_name):
            success_count += 1
        print("-" * 80)
    
    print(f"\nScore generation completed!")
    print(f"Success: {success_count}/{total_models} models")
    
    if success_count < total_models:
        print("Some models failed to generate scores. Please check the error messages above.")
        sys.exit(1)
    else:
        print("All missing scores generated successfully!")

if __name__ == "__main__":
    main()