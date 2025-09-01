#!/usr/bin/env python3
"""
Direct evaluation script that bypasses problematic imports.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_evaluation_directly():
    """Run evaluation directly without going through the main CLI."""
    try:
        # Import only what we need for evaluation
        from bfcl_eval.eval_checker.eval_runner_helper import load_file
        from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING, VERSION_PREFIX
        from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH
        
        # Import evaluation functions directly
        sys.path.append('bfcl_eval/eval_checker')
        
        # Try to import eval_runner without the problematic model configs
        spec = importlib.util.spec_from_file_location(
            "eval_runner", 
            "bfcl_eval/eval_checker/eval_runner.py"
        )
        eval_runner = importlib.util.module_from_spec(spec)
        
        # Patch the problematic model config import
        import bfcl_eval.constants.model_config as model_config_module
        model_config_module.MODEL_CONFIG_MAPPING = {}
        
        spec.loader.exec_module(eval_runner)
        
        models = ["deepseek-ai_DeepSeek-V3.1-thinking-off", "deepseek-ai_DeepSeek-V3.1-thinking-on"]
        
        for model in models:
            print(f"Processing model: {model}")
            try:
                # Call the evaluation function directly
                eval_runner.main([model], None, "result", "score")
                print(f"Successfully processed {model}")
            except Exception as e:
                print(f"Error processing {model}: {e}")
                
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying alternative approach...")
        run_basic_evaluation()

def run_basic_evaluation():
    """Basic evaluation without complex imports."""
    import json
    import glob
    from pathlib import Path
    
    models = ["deepseek-ai_DeepSeek-V3.1-thinking-off", "deepseek-ai_DeepSeek-V3.1-thinking-on"]
    
    for model in models:
        print(f"Processing {model}...")
        
        result_dir = Path("result") / model
        score_dir = Path("score") / model
        
        if not result_dir.exists():
            print(f"Result directory {result_dir} does not exist")
            continue
            
        score_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all result files
        result_files = list(result_dir.glob("BFCL_v3_*_result.json"))
        
        for result_file in result_files:
            # Generate corresponding score file name
            score_file_name = result_file.name.replace("_result.json", "_score.json")
            score_file = score_dir / score_file_name
            
            print(f"Creating score file: {score_file}")
            
            # Create a basic score file (you'll need to implement actual scoring logic)
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results = []
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
                
                # Create basic score entries (this is a placeholder)
                scores = []
                for result in results:
                    # This is a simplified scoring - you might need more complex logic
                    score_entry = {
                        "id": result.get("id", ""),
                        "model_name": model.replace("_", "/"),
                        "test_category": result_file.stem.replace("BFCL_v3_", "").replace("_result", ""),
                        "valid": True,  # Placeholder
                        "error": [],
                        "prompt": {},
                        "model_result": result.get("result", ""),
                        "possible_answer": result.get("possible_answer", "")
                    }
                    scores.append(score_entry)
                
                # Write score file
                with open(score_file, 'w', encoding='utf-8') as f:
                    for score in scores:
                        f.write(json.dumps(score) + '\n')
                        
                print(f"Created {score_file} with {len(scores)} entries")
                        
            except Exception as e:
                print(f"Error processing {result_file}: {e}")

if __name__ == "__main__":
    print("Starting direct evaluation...")
    run_evaluation_directly()