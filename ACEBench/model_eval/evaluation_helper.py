
import glob
import json
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

from model_eval.utils import *
from model_eval.checker import *
from model_inference.prompt_en import *
from model_inference.prompt_zh import *

REST_API_GROUND_TRUTH_FILE_PATH = "api_status_check_ground_truth_REST.json"
EXECTUABLE_API_GROUND_TRUTH_FILE_PATH = "api_status_check_ground_truth_executable.json"

COLUMNS = [
    "Model",
    "bool",
    "enum",
    "number",
    "list",
    "object_short",
    "object_deep",
    "atom",
    "single_turn_singal_function",
    "single_turn_parallel_function",
    "single_turn",
    "multiple_turn_switch",
    "multiple_turn_adjust",
    "multiple_turn",
    "similar_api",
    "profile",
    "normal_summary",
    "incomplete",
    "error",
    "irrelevant",
    "special_summary",
    "agent_multi_turn",
    "agent_multi_turn_process",
    "agent_multi_step",
    "agent_multi_step_process",
    "agent_summary",
    "Summary",
]

closed_model_list = ["o1-mini", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4-turbo-2024-04-09", "qwen-max", "doubao-pro-32k", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "deepseek-chat"]

V100_x8_PRICE_PER_HOUR = 22.032

def extract_after_test(input_string):
    parts = input_string.split("data_")[1].split("_result")[0].split(".json")[0]
    return parts

def find_file_with_suffix(folder_path, suffix):
    json_files_pattern = os.path.join(folder_path, "*.json")
    for json_file in glob.glob(json_files_pattern):
        if suffix == "multi_turn":
            json_file = folder_path + "data_multi_turn.json"
            return json_file
        if suffix in json_file.split("/")[-1]:
            return json_file

def load_file(file_path):
    result = []
    with open(file_path, encoding='utf-8') as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result

def is_empty_output(decoded_output):
    if not is_function_call_format_valid(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True

def multiplt_turn_accuracy(score_list):
    end_score_list = []
    process_score_list = []
    for score in score_list:
        if False in score["valid"]:
            end_score = 0
        else:
            end_score = 1
        process_score = score["valid"].count(True) / len(score["valid"])
        process_score = round(process_score, 3)

        end_score_list.append(end_score)
        process_score_list.append(process_score)
    end_score_total = round(sum(end_score_list) / len(end_score_list), 3)
    process_score_total = round(sum(process_score_list) / len(process_score_list), 3)
    return end_score_total, process_score_total

def calculate_weighted_accuracy(accuracy_dict_list):
    total_count = 0
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        total_count += accuracy_dict["total_count"]
        total_accuracy += accuracy_dict["accuracy"] * accuracy_dict["total_count"]

    if total_count == 0:
        return {"accuracy": 0, "total_count": 0}

    return {"accuracy": round(total_accuracy / total_count, 3), "total_count": total_count}

def calculate_unweighted_accuracy(accuracy_dict_list):
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        total_accuracy += accuracy_dict["accuracy"]

    if len(accuracy_dict_list) == 0:
        return {"accuracy": 0, "total_count": 0}

    return {"accuracy": round(total_accuracy / len(accuracy_dict_list), 3), "total_count": 0}

def update_result_table_with_score_file(leaderboard_table, score_path):
    entries = os.scandir(score_path)

    # Filter out the subdirectories
    subdirs = [entry.path for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in subdirs:
        # Pattern to match JSON files in this subdirectory
        json_files_pattern = os.path.join(subdir, "*.json")
        model_name = subdir.split(score_path)[1]
        # Find and process all JSON files in the subdirectory
        for model_score_json in glob.glob(json_files_pattern):
            if "process" not in model_score_json:
                if "agent" not in model_score_json:
                    metadata = load_file(model_score_json)[0]
                    accuracy, total_count = metadata["accuracy"], metadata["total_count"]
                    test_category = model_score_json.split("_score.json")[0].split("/")[-1]
                    test_category = test_category.split("\\")[-1]
                    if model_name not in leaderboard_table:
                        leaderboard_table[model_name] = {}
                    if test_category not in leaderboard_table[model_name]:
                        leaderboard_table[model_name][test_category] = {
                            "accuracy": accuracy,
                            "total_count": total_count,
                        }
                else:
                    metadata = load_file(model_score_json)[0]
                    accuracy, process_accuracy, total_count = metadata["end_to_end_accuracy"], metadata["process_accuracy"], metadata["total_count"]
                    test_category = model_score_json.split("_score.json")[0].split("/")[-1]
                    test_category = test_category.split("\\")[-1]
                    if model_name not in leaderboard_table:
                        leaderboard_table[model_name] = {}
                    if test_category not in leaderboard_table[model_name]:
                        leaderboard_table[model_name][test_category] = {
                            "accuracy": accuracy,
                            "process_accuracy": process_accuracy,
                            "total_count": total_count,
                        }

# Calculate weighted accuracy
def generate_result_csv(leaderboard_table, output_path):
    data_close = []
    data_open = []
    for model_name, value in leaderboard_table.items():
        unusal_lose = value.get("data_special_incomplete", {"accuracy": 0, "total_count": 0})
        unusal_error = value.get("data_special_error_param", {"accuracy": 0, "total_count": 0})
        unusal_exceeding = value.get("data_special_irrelevant", {"accuracy": 0, "total_count": 0})

        atom_bool = value.get("data_normal_atom_bool", {"accuracy": 0, "total_count": 0})
        atom_enum = value.get("data_normal_atom_enum", {"accuracy": 0, "total_count": 0})
        atom_number = value.get("data_normal_atom_number", {"accuracy": 0, "total_count": 0})  # updated
        atom_list = value.get("data_normal_atom_list", {"accuracy": 0, "total_count": 0})      # updated
        atom_object_deep = value.get("data_normal_atom_object_deep", {"accuracy": 0, "total_count": 0})  # updated
        atom_object_short = value.get("data_normal_atom_object_short", {"accuracy": 0, "total_count": 0})  # updated

        normal_ss = value.get("data_normal_single_turn_single_function", {"accuracy": 0, "total_count": 0})  # updated
        normal_sp = value.get("data_normal_single_turn_parallel_function", {"accuracy": 0, "total_count": 0})  # updated
        normal_ms = value.get("data_normal_multi_turn_user_switch", {"accuracy": 0, "total_count": 0})  # updated
        normal_ma = value.get("data_normal_multi_turn_user_adjust", {"accuracy": 0, "total_count": 0})  # updated
        normal_similar = value.get("data_normal_similar_api", {"accuracy": 0, "total_count": 0})  # updated
        normal_profile = value.get("data_normal_preference", {"accuracy": 0, "total_count": 0})  # updated

        agent_turn = value.get("data_agent_multi_turn", {"accuracy": 0, "process_accuracy": 0, "total_count": 0})
        agent_step = value.get("data_agent_multi_step", {"accuracy": 0, "process_accuracy": 0, "total_count": 0})
        
        special_total = calculate_unweighted_accuracy(
            [unusal_lose, unusal_error, unusal_exceeding]
        )
        
        normal_total = calculate_unweighted_accuracy(
            [normal_ss, normal_sp, normal_ms, normal_ma, normal_similar, normal_profile, atom_bool, atom_enum, atom_number, atom_list, atom_object_deep, atom_object_short]
        )

        agent_total = calculate_unweighted_accuracy(
            [agent_turn, agent_step]
        )

        atom_total = calculate_unweighted_accuracy(
            [atom_bool, atom_enum, atom_number, atom_list, atom_object_deep, atom_object_short]
        )

        singal_turn_total = calculate_unweighted_accuracy(
            [normal_ss, normal_sp]
        )
        
        multi_turn_total = calculate_unweighted_accuracy(
            [normal_ms, normal_ma]
        )

        summary = special_total["accuracy"] * 0.2676 + normal_total["accuracy"] * 0.578 + agent_total["accuracy"] * 0.1545
        summary = round(summary, 3)

        if model_name in closed_model_list:
            data_close.append(
                [
                    model_name,
                    atom_bool["accuracy"],
                    atom_enum["accuracy"],
                    atom_number["accuracy"],
                    atom_list["accuracy"],
                    atom_object_deep["accuracy"],
                    atom_object_short["accuracy"],
                    atom_total["accuracy"],
                    normal_ss["accuracy"],
                    normal_sp["accuracy"],
                    singal_turn_total["accuracy"],
                    normal_ms["accuracy"],
                    normal_ma["accuracy"],
                    multi_turn_total["accuracy"],
                    normal_similar["accuracy"],
                    normal_profile["accuracy"],
                    normal_total["accuracy"],
                    unusal_lose["accuracy"],
                    unusal_error["accuracy"],
                    unusal_exceeding["accuracy"],
                    special_total["accuracy"],
                    agent_turn["accuracy"],
                    agent_turn["process_accuracy"],
                    agent_step["accuracy"],
                    agent_step["process_accuracy"],
                    agent_total["accuracy"],
                    summary,
                ]
            )
        else:
            data_open.append(
                [
                    model_name,
                    atom_bool["accuracy"],
                    atom_enum["accuracy"],
                    atom_number["accuracy"],
                    atom_list["accuracy"],
                    atom_object_deep["accuracy"],
                    atom_object_short["accuracy"],
                    atom_total["accuracy"],
                    normal_ss["accuracy"],
                    normal_sp["accuracy"],
                    singal_turn_total["accuracy"],
                    normal_ms["accuracy"],
                    normal_ma["accuracy"],
                    multi_turn_total["accuracy"],
                    normal_similar["accuracy"],
                    normal_profile["accuracy"],
                    normal_total["accuracy"],
                    unusal_lose["accuracy"],
                    unusal_error["accuracy"],
                    unusal_exceeding["accuracy"],
                    special_total["accuracy"],
                    agent_turn["accuracy"],
                    agent_turn["process_accuracy"],
                    agent_step["accuracy"],
                    agent_step["process_accuracy"],
                    agent_total["accuracy"],
                    summary,
                ]
            )

    # Sort data_close by summary in descending order
    sorted_data_close = sorted(data_close, key=lambda x: x[-1], reverse=True)

    # Sort data_open by summary in descending order
    sorted_data_open = sorted(data_open, key=lambda x: x[-1], reverse=True)

    # Merge sorted data_close and data_open
    data = sorted_data_close + sorted_data_open

    wb = Workbook()
    ws = wb.active

    data.insert(0, COLUMNS)  

    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = ws.cell(row=i + 1, column=j + 1, value=value)
            cell.alignment = Alignment(horizontal="center", vertical="center")

    filepath = os.path.join(output_path, "result.xlsx")
    wb.save(filepath)

def collapse_json_objects(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    objects = []
    depth = 0
    obj_start = 0
    for i, char in enumerate(content):
        if char == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                obj = content[obj_start: i + 1]
                objects.append(obj)

    with open(file_path, "w") as out_file:
        for obj in objects:
            json_obj = json.loads(obj)
            compact_json = json.dumps(json_obj, separators=(",", ":"))
            out_file.write(compact_json + "\n")

def convert_answer(answer):
    if answer == "":
        return answer
    result = [f"{key}({', '.join([f'{k}={v}' if isinstance(v, str) else f'{k}={v}' for k, v in value.items()])})" for key, value in answer.items()]
    return result

def convert_result_to_excel(model_name, category, paths):
    INPUT_PATH = paths["INPUT_PATH"]
    PROMPT_PATH = paths["PROMPT_PATH"]
    POSSIBLE_ANSWER_PATH = paths["POSSIBLE_ANSWER_PATH"]
    SCORE_PATH = paths["OUTPUT_PATH"]
    if "zh" in INPUT_PATH:
        language = "zh"
    else:
        language = "en"
    
    prompt_file = build_data_path(PROMPT_PATH, category)
    answer_file = build_data_path(POSSIBLE_ANSWER_PATH, category)
    result_file = build_result_path(INPUT_PATH, model_name, category, "_result.json")
    score_file = build_result_path(SCORE_PATH, model_name, category, "_score.json")
   
    prompt_list = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            id = data["id"]
            if "time" not in list(data.keys()):
                time = ""
            else:
                time = data["time"]
            if "profile" not in list(data.keys()):
                profile = ""
            else:
                profile = data["profile"]
            functions = data["function"]
            question = data["question"]
            function_prompt = ''
            for function in functions:
                function_prompt = function_prompt + str(function) + "\n"
            if language == "zh":
                if "special" in category:
                    system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time=time, function=functions)
                elif "preference" in category:
                    system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(profile=profile, function=functions)
                else:
                    system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time=time, function=functions)
                user_prompt = USER_PROMPT_ZH.format(question=question)
            elif language == "en":
                if "special" in category:
                    system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time, function=functions)
                elif "preference" in category:
                    system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=profile, function=functions)
                else:
                    system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=time, function=functions)
                user_prompt = USER_PROMPT_EN.format(question=question)

            prompt = system_prompt + "\n" + user_prompt
            prompt_list.append({"id": id, "prompt": prompt, "question": question})
    
    with open(answer_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            data = json.loads(line)
            if "special" in category:
                answer = data["ground_truth"]
            else:
                answer = data["ground_truth"]
                if isinstance(answer, list):
                    answer_list = []
                    for answer_item in answer:
                        answer_list.append(convert_answer(answer_item))
                    prompt_list[index]["excepted_answer"] = answer_list
                else:
                    answer = convert_answer(answer)
                    prompt_list[index]["excepted_answer"] = answer

    with open(result_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            data = json.loads(line)
            result = data["result"]
            prompt_list[index]["model_answer"] = result
            prompt_list[index]["flag"] = 'true'
            prompt_list[index]["error_reason"] = ""

    with open(score_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= 1:
                data = json.loads(line)
                prompt_list[index - 1]["flag"] = 'false'
                prompt_list[index - 1]["error_reason"] = data["error"]

    df = pd.DataFrame(prompt_list)

    if language == 'zh':
        folder_path = "../result_excel/zh/" + model_name
    elif language == 'en':
        folder_path = "../result_excel/en/" + model_name
    save_path = folder_path + "/data_" + category + ".xlsx"
    if not os.path.exists(folder_path):
        # Create folder if it doesn't exist
        os.makedirs(folder_path)
    df.to_excel(save_path, index=False)

def merge_result(folder_path):
    # Get all Excel files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # List to store data from all files
    all_data = []

    # Read each Excel file and add its data to all_data
    for file in excel_files:
        if "special" not in file and "similar" not in file:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)  # Read Excel file
            all_data.append(df)
            
    for file in excel_files:
        if "special" in file or "similar" in file:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)  # Read Excel file
            all_data.append(df)

    # Merge all DataFrames
    merged_data = pd.concat(all_data, ignore_index=True)
    model_name = folder_path.split("/")[-1]
    save_name = model_name + "_output.xlsx"

    # Write the merged data to a new Excel file
    merged_data.to_excel(os.path.join(folder_path, save_name), index=False)
