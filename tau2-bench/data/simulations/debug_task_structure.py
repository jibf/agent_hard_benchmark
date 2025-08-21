import json
import glob

def debug_task_structure():
    """Debug the task structure in JSON files"""
    
    # Get a telecom file
    telecom_files = glob.glob("*telecom*.json")
    if telecom_files:
        filepath = telecom_files[0]
        print(f"Examining file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nFile structure keys: {list(data.keys())}")
        
        if 'tasks' in data:
            print(f"\nNumber of tasks: {len(data['tasks'])}")
            if data['tasks']:
                print(f"First task keys: {list(data['tasks'][0].keys())}")
                print(f"First task: {json.dumps(data['tasks'][0], indent=2)}")
        
        if 'simulations' in data:
            print(f"\nNumber of simulations: {len(data['simulations'])}")
            if data['simulations']:
                print(f"First simulation keys: {list(data['simulations'][0].keys())}")
                print(f"First simulation: {json.dumps(data['simulations'][0], indent=2)}")
        
        # Look for task_id patterns in simulations
        print(f"\nTask IDs in simulations:")
        task_ids = set()
        for sim in data['simulations']:
            task_id = sim.get('task_id')
            if task_id:
                task_ids.add(task_id)
        
        print(f"Unique task IDs found: {len(task_ids)}")
        for task_id in sorted(list(task_ids))[:10]:  # Show first 10
            print(f"  {task_id}")
        
        # Check if there are any complex task IDs like the ones we found
        complex_task_ids = [tid for tid in task_ids if '[' in tid and ']' in tid]
        print(f"\nComplex task IDs (containing brackets): {len(complex_task_ids)}")
        for task_id in complex_task_ids[:5]:  # Show first 5
            print(f"  {task_id}")

if __name__ == "__main__":
    debug_task_structure()
