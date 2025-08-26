import json
import os
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv
from prompt import prompt
from tqdm import tqdm

load_dotenv()

class DatasetAssessor:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        self.model = "openai/gpt-4.1" # TODO: subject to change
    
    def extract_prompt_and_function_call(self, line_data: Dict[str, Any]) -> tuple:
        """Extract prompt and ground-truth function call from a data line."""
        conversations = line_data.get('conversations', [])
        
        # Extract user prompt
        user_prompt = ""
        function_call = ""
        
        for conv in conversations:
            if conv.get('role') == 'user':
                user_prompt = conv.get('content', '')
            elif conv.get('role') == 'assistant' and 'function_call' in conv:
                function_call = json.dumps(conv['function_call'], indent=2)
        
        return user_prompt, function_call
    
    def assess_sample(self, user_prompt: str, function_call: str) -> Dict[str, Any]:
        """Assess a single sample using GPT-4.1."""
        
        evaluation_prompt = prompt.format(
            prompt=user_prompt,
            function_call=json.dumps(function_call)
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
            return result
            
        except Exception as e:
            print(f"Error assessing sample: {e}")
            return {"error": str(e)}
    
    def assess_dataset(self, jsonl_file_path: str, output_file_path: str = None):
        """Assess the entire dataset and save results."""
        
        if output_file_path is None:
            output_file_path = jsonl_file_path.replace('.jsonl', '_assessed.jsonl')
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            
            for line_num, line in tqdm(enumerate(input_file, 1)):
                line_data = json.loads(line.strip())
                sample_id = line_data.get('id', f'sample_{line_num}')
                
                print(f"Processing {sample_id} (line {line_num})...")
                
                # Extract prompt and function call
                user_prompt, function_call = self.extract_prompt_and_function_call(line_data)
                
                if not user_prompt or not function_call:
                    print(f"Warning: Missing prompt or function call in {sample_id}")
                    continue
                
                # Assess the sample
                assessment = self.assess_sample(user_prompt, function_call)
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
                
                print(f"Completed {sample_id}: Score {assessment.get('score', 'N/A')}")
                

def main():
    assessor = DatasetAssessor()
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    dataset_path = os.path.join(parent_dir, "data", "ComplexFuncBench.jsonl")
    output_path = os.path.join(parent_dir, "data", "ComplexFuncBench_assessed.jsonl")
    
    print("Starting dataset assessment with GPT-4.1...")
    assessor.assess_dataset(dataset_path, output_path)
    print(f"Assessment complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()