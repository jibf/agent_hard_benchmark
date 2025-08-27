import json
import os
import argparse
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from prompts import flawed_gt_filtering
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool
from datetime import datetime

load_dotenv()


class DatasetAssessor:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        self.model = "openai/gpt-4.1"
    
    def extract_conversation(self, line_data: Dict[str, Any]) -> tuple:
        conversations = line_data.get('conversations', [])
        available_function_list = line_data.get('functions', [])
        user_prompt = conversations[0].get('content', '')   # first message in the conversation is the user prompt
        
        return user_prompt, conversations, available_function_list

    def assess_sample(self, user_prompt: str, conversations: str, available_function_list: list) -> Dict[str, Any]:
        """Assess a single sample using LLM."""

        evaluation_prompt = flawed_gt_filtering.prompt.format(
            user_prompt=user_prompt,
            conversations=json.dumps(conversations),
            available_function_list=json.dumps(available_function_list)
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            try:    # Reorder the keys for a better readability
                result = {
                    "is_flawed": result["is_flawed"], 
                    "reasoning_summary": result["reasoning_summary"],
                    **{k: v for k, v in result.items() if k not in ["is_flawed", "reasoning_summary"]}
                }
            except KeyError:
                print("KeyError: 'is_flawed' or 'reasoning_summary' not found in result")
            return result
            
        except Exception as e:
            print(f"Error assessing sample: {e}")
            return {"error": str(e)}
    
    def assess_dataset(self, jsonl_file_path: str, output_file_path: str = None, proc_num: int = 1):
        """Assess the entire dataset and save results."""
        
        # Load all data first
        test_data = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as input_file:
            for line_num, line in enumerate(input_file, 1):
                line_data = json.loads(line.strip())
                sample_id = line_data.get('id', f'sample_{line_num}')
                test_data.append((line_data, sample_id, line_num))
        
        if proc_num > 1:
            # Use multiprocessing
            with Pool(processes=proc_num) as pool:
                with tqdm(total=len(test_data), desc="Processing samples", unit="sample") as pbar:
                    results = []
                    for data in test_data:
                        result = pool.apply_async(process_sample, (data,))
                        results.append(result)
                    
                    # Write results as they complete
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        for result in results:
                            result_data = result.get()
                            if result_data:
                                output_file.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                                output_file.flush()
                            pbar.update(1)
        else:
            # Sequential processing
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for line_data, sample_id, line_num in tqdm(test_data, desc="Processing samples"):
                    print(f"Processing {sample_id} (line {line_num})...")
                    
                    # Extract prompt and function call
                    user_prompt, conversations, available_function_list = self.extract_conversation(line_data)
                    
                    if not user_prompt or not conversations:
                        print(f"Warning: Missing prompt or function call in {sample_id}")
                        continue
                    
                    # Assess the sample
                    assessment = self.assess_sample(user_prompt, conversations, available_function_list)
                    print(assessment)
                    
                    # Prepare result
                    result = {
                        "id": sample_id,
                        "original_data": line_data,
                        "assessment": assessment
                    }
                    
                    # Write to output file
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_file.flush()
                    
                    print(f"Completed {sample_id}: Score {assessment.get('is_flawed', 'N/A')}")
                

def process_sample(data_tuple):
    """Process a single sample for multiprocessing."""
    line_data, sample_id, line_num = data_tuple
    
    # Create assessor instance in worker process to avoid pickle issues
    assessor = DatasetAssessor()
    
    print(f"Processing {sample_id} (line {line_num})...")
    
    # Extract prompt and function call
    user_prompt, function_call, function_list = assessor.extract_conversation(line_data)
    
    if not user_prompt or not function_call:
        print(f"Warning: Missing prompt or function call in {sample_id}")
        return None
    
    # Assess the sample
    assessment = assessor.assess_sample(user_prompt, function_call, function_list)
    
    # Prepare result
    result = {
        "id": sample_id,
        "assessment": assessment
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Assess dataset using LLM")
    parser.add_argument('--input', '-i', required=False, help='Input JSONL file path', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/ComplexFuncBench.jsonl'))
    parser.add_argument('--output', '-o', help='Output JSONL file path (default: input_assessed.jsonl)')
    parser.add_argument('--proc_num', '-p', type=int, default=1, help='Number of processes for multiprocessing (default: 1)')
    
    args = parser.parse_args()
    
    assessor = DatasetAssessor()
    
    if args.output is None:
        timestamp = datetime.now().strftime("%m%d-%H%M")
        args.output = f"results/ComplexFuncBench_assessed-{timestamp}.jsonl"

    print(f"Starting dataset assessment with LLM...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Processes: {args.proc_num}")
    
    assessor.assess_dataset(args.input, args.output, args.proc_num)
    print(f"Assessment complete! Results saved to {args.output}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()