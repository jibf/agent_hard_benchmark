
import argparse, json, os
from tqdm import tqdm
from model_inference.inference_map import inference_map
from model_inference.apimodel_inference import APIModelInference
from model_inference.dynamic_apimodel import create_dynamic_apimodel_config
from category import ACE_DATA_CATEGORY
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_args():

    parser = argparse.ArgumentParser()
    # Model name
    parser.add_argument("--model", type=str, default="gpt-4o", nargs="+", help="Name of the model(s) to use")

    # For local models, specify the model path
    parser.add_argument("--model-path", type=str, help="Path to the model for local models")

    # Category of the model you want to test, default is "all"
    parser.add_argument("--category", type=str, default="test_all", nargs="+", help="Category of the model you want to test")

    # Temperature parameter to control randomness of model output, default is 0.7
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter to control randomness of model output")
    # Top-p parameter to control diversity of model output, default is 1
    parser.add_argument("--top-p", type=float, default=1, help="Top-p parameter to control diversity of model output")
    # Maximum number of tokens to generate, default is 1200
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum number of tokens to generate")
    # Number of GPUs to use, default is 1
    parser.add_argument("--num-gpus", default=1, type=int, help="Number of GPUs to use")
    # GPU memory utilization rate, default is 0.9
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float, help="GPU memory utilization rate")
    # Language for model output, choose 'en' for English or 'zh' for Chinese
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"], help="Language for model output, choose 'en' for English or 'zh' for Chinese")
    # Number of threads to use for concurrency, default is 1
    parser.add_argument("--num-threads", type=int, default=1, help="Number of threads to use for concurrency")
    # Maximum number of dialog turns allowed for agent interactions
    parser.add_argument("--max-dialog-turns", type=int, default=40, help="Maximum number of dialog turns allowed for agent interactions")
    # Model used by the user role in the agent, it is recommended to use an advanced large model
    parser.add_argument("--user-model", type=str, default="openai/gpt-4o-20240806", help="Model used by the user role in the agent")
    # Output directory for results
    parser.add_argument("--output-dir", type=str, default="./", help="Directory to save results (result_all and score_all will be created here)")
    args = parser.parse_args()
    return args



def load_test_cases(base_path, filenames):
    cases = []
    
    for filename in filenames:
        file_path = os.path.join(base_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                cases.extend(json.loads(line) for line in file)
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON in file - {file_path}")
    return cases


def sort_json(file):
    data = []
    with open(file,'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Extract filename from full path to avoid issues with absolute paths containing "agent"
    filename = os.path.basename(file)

    if "multi_turn" in filename and "agent" not in filename:
        data = sorted(data, key=lambda x: tuple(map(int, x["id"].split("_")[-2:])))
    else:
        data = sorted(data, key=lambda x: int(x["id"].split("_")[-1]))
    with open(file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')  

def generate_singal(args, model_name, test_case):
    model_path = args.model_path
    result_path = args.result_path

    # Get the inference handler, with dynamic model support
    if model_name in inference_map:
        inference_handler = inference_map[model_name]
    else:
        # Check if we have API_KEY and BASE_URL for dynamic model creation
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        if api_key and base_url:
            print(f"Model {model_name} not found in inference_map, creating dynamic model with API_KEY and BASE_URL")
            inference_handler = create_dynamic_apimodel_config(model_name, api_key, base_url)
        else:
            raise KeyError(f"Model {model_name} not found in inference_map and no API_KEY/BASE_URL provided for dynamic model creation")
    
    # Check if it's an API model (APIModelInference) or local model (CommonInference)
    # Handle both predefined classes and dynamic classes
    if inference_handler == APIModelInference or (hasattr(inference_handler, '__name__') and 'APIModelInference' in str(inference_handler.__bases__ if hasattr(inference_handler, '__bases__') else [])):
        # API models (both predefined and dynamic) don't need model_path or tensor_parallel_size parameter
        model_inference = inference_handler(model_name, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, max_dialog_turns=args.max_dialog_turns, user_model=args.user_model, language=args.language)
    else:
        # Local models need tensor_parallel_size parameter
        model_inference = inference_handler(model_name, model_path, args.temperature, args.top_p, args.max_tokens, args.max_dialog_turns, args.user_model, args.language, args.num_gpus)

    if "agent" in test_case["id"]:
        id, question, functions = (
                    test_case["id"],
                    test_case["question"],
                    test_case["function"],
                )
        if isinstance(functions, (dict, str)):
            functions = [functions]
        time = ""
        profile = ""
        result, process_list = model_inference.inference(question, functions, time, profile,test_case, id)
        result_to_write = {
            "id": id,
            "result": result,
            "process": process_list
        }
        model_inference.write_result(result_to_write, model_name, result_path)

    elif "preference" in test_case["id"]:
        id, question, functions, profile = (
                    test_case["id"],
                    test_case["question"],
                    test_case["function"],
                    test_case["profile"],
                )
        time = ""
        if isinstance(functions, (dict, str)):
            functions = [functions]

        result = model_inference.inference(question, functions, time, profile, test_case, id)

        result_to_write = {
            "id": id,
            "result": result,
        }
        model_inference.write_result(result_to_write, model_name, result_path)
    
    else:
        id, question, functions, time = (
                    test_case["id"],
                    test_case["question"],
                    test_case["function"],
                    test_case["time"],
                )
        profile = ""
        if isinstance(functions, (dict, str)):
            functions = [functions]

        result = model_inference.inference(question, functions, time, profile, test_case, id)

        result_to_write = {
            "id": id,
            "result": result,
        }
        model_inference.write_result(result_to_write, model_name, result_path)

def generate_results(args, model_name, test_case, completed_id_set):
    with ThreadPoolExecutor(max_workers = args.num_threads) as executor:
        futures = []
        for test_case in test_cases_total:
            if test_case["id"] not in completed_id_set:
                future = executor.submit(generate_singal, args, model_name, test_case)
                futures.append(future)

        with tqdm(total=len(futures), desc="Processing Tasks", leave=True) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()  # Catch exceptions in tasks
                    pbar.update(1)
                except Exception as e:
                    print(f"Task raised an exception: {e}")
                    # You can choose whether to continue executing tasks after catching an exception, or to terminate the program
                    raise
        print("All tasks have been completed.")



 
if __name__ == "__main__":


    args = get_args()

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.category) is not list:
        args.category = [args.category]

    
    # Use output_dir parameter for result paths
    result_base = os.path.join(args.output_dir, "result_all")

    paths = {
        "zh": {"data_path": "./data_all/data_zh/", "result_path": f"{result_base}/result_zh/"},
        "en": {"data_path": "./data_all/data_en/", "result_path": f"{result_base}/result_en/"},
    }

    data_path = paths[args.language]["data_path"]
    result_path = paths[args.language]["result_path"]
    args.result_path = result_path

    # Get the filenames of the test cases
    test_names = {test_name for category in args.category for test_name in ACE_DATA_CATEGORY[category]}
    test_files = [f"data_{test_name}.json" for test_name in test_names]

    for model_name in args.model:
        folder_path = os.path.join(result_path, model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        completed_id_set = set()
        # Count the cases that have already been generated to avoid duplication
        for file in test_names:
            file_name = f"data_{file}_result.json"
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_data = json.loads(line)
                        completed_id_set.add(line_data["id"])
        # Read data
        test_cases_total = load_test_cases(data_path, test_files)

        if len(test_cases_total) > 0:
            generate_results(args, model_name, test_cases_total, completed_id_set)
        
        # Multithreading may disrupt the order of execution, so the result ids need to be reordered
        for file in test_names:
            file_name = f"data_{file}_result.json"
            file_path = os.path.join(folder_path, file_name)
            sort_json(file_path)