
import re 
import os
import json
from collections import Counter


def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Prevent error when the list is empty
    return sum(numbers) / len(numbers)


def get_lose_param(text):

    # Use regular expressions to extract parameters and API names
    params_match = re.search(r'\((.*?)\)', text)
    api_match = re.findall(r'\(.*?\)', text)

    if params_match and api_match:
        # Extract parameters
        params = params_match.group(1).split(', ')
        # Extract API name
        api_name = api_match[1][1:-1]  # Remove parentheses

        return api_name, params


def is_function_call_format_valid(decoded_output):
    # Ensure the output is a list of dictionaries
    if type(decoded_output) == list:
        for item in decoded_output:
            if type(item) != dict:
                return False
        return True
    return False



def save_score_as_json(filename, data, subdir=None):

    # If a subdirectory is specified, ensure the subdirectory exists and construct the full file path
    if subdir:
        os.makedirs(subdir, exist_ok=True)
        filename = os.path.join(subdir, filename)

    def _find_and_warn_sets(d):

        for key, value in d.items():
            if isinstance(value, set):
                print(f"Warning: Found a set in key '{key}', value: {value}")
            elif isinstance(value, dict):
                _find_and_warn_sets(value)  # Recursively check nested dictionaries
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _find_and_warn_sets(item)  # Check dictionaries in the list

    # Write the list of dictionaries to the file
    with open(filename, "w", encoding="utf-8") as file:
        for i, entry in enumerate(data):
            _find_and_warn_sets(entry)  # Check and warn about sets
            json_str = json.dumps(entry, ensure_ascii=False)
            file.write(json_str)
            if i < len(data) - 1:  # If not the last entry, add a newline
                file.write("\n")


def sum_key_list(data):
    key_counter = Counter()
    for dictionary in data:
        key_counter.update(dictionary.keys())
    key_count_dict = dict(key_counter)
    return key_count_dict


def flatten_dates(d):
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in d.items()}



def standardize_string(input_string: str):

    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')



def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Prevent error when the list is empty
    return sum(numbers) / len(numbers)


        
def find_description(func_descriptions, name):

    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] in name:
                return func_description
        return None
    else:
        return func_descriptions
    
def find_function(model_output_item, possible_answers):
    fun_name = list(model_output_item.keys())[0]
    for possible_answer in possible_answers:
        if fun_name in possible_answer:
            return possible_answer
    return False


def get_possible_answer_type(possible_answer):
        
    if possible_answer != "":  # Optional parameter
        return type(possible_answer)
    return None

def build_result_path(base_path, model_name, category, suffix=".json"):
    file_name = f"data_{category}{suffix}"
    return os.path.join(base_path, model_name, file_name)


def build_data_path(base_path, category, suffix=".json"):
    file_name = f"data_{category}{suffix}"
    return os.path.join(base_path, file_name)

