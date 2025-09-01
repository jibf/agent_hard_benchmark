#!/usr/bin/env python3
"""
Conda-safe evaluation script that avoids problematic dependencies.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def fix_cohere_import():
    """Temporarily fix the cohere import issue."""
    cohere_file = Path("bfcl_eval/model_handler/api_inference/cohere.py")
    
    if cohere_file.exists():
        # Read the original file
        with open(cohere_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a backup
        backup_file = cohere_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Fix the problematic line
        fixed_content = content.replace(
            ') -> tuple[cohere.types.ChatResponse, float]:',
            ') -> tuple[any, float]:'
        )
        
        # Write the fixed content
        with open(cohere_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed cohere import in {cohere_file}")
        return backup_file
    
    return None

def restore_cohere_import(backup_file):
    """Restore the original cohere file."""
    if backup_file and backup_file.exists():
        cohere_file = backup_file.with_suffix('')
        with open(backup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(cohere_file, 'w', encoding='utf-8') as f:
            f.write(content)
        backup_file.unlink()
        print("Restored original cohere file")

def run_evaluation_safe():
    """Run evaluation with safety measures."""
    backup_file = None
    
    try:
        # Fix the cohere import issue
        backup_file = fix_cohere_import()
        
        models = ["deepseek-ai_DeepSeek-V3.1-thinking-off", "deepseek-ai_DeepSeek-V3.1-thinking-on"]
        
        for model in models:
            print(f"Evaluating {model}...")
            
            cmd = [
                sys.executable, "-m", "bfcl_eval", "evaluate",
                "--model", model,
                "--result-dir", "result",
                "--score-dir", "score"
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    print(f"Successfully evaluated {model}")
                    print(f"Output: {result.stdout}")
                else:
                    print(f"Error evaluating {model}")
                    print(f"Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"Evaluation of {model} timed out")
            except Exception as e:
                print(f"Exception during evaluation of {model}: {e}")
    
    finally:
        # Always restore the original file
        if backup_file:
            restore_cohere_import(backup_file)

if __name__ == "__main__":
    print("Starting conda-safe evaluation...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    run_evaluation_safe()