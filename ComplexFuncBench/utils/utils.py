import json
import re
import time
import traceback
import logging


def load_json(dir_path):
    if dir_path.endswith('.json'):
        return json.load(open(dir_path, 'r'))
    elif dir_path.endswith('.jsonl'):
        return [json.loads(line) for line in open(dir_path, 'r')]


def save_json(data, dir_path):
    if dir_path.endswith('.json'):
        with open(dir_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif dir_path.endswith(".jsonl"):
        with open(dir_path, 'w') as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

def decode_json(json_str):
    if json_str is None:
        return None
    json_str = json_str.strip('```JSON\n').strip('```json\n').strip('\n```')
    json_str = json_str.replace('\n', '').replace('False', 'false').replace('True', 'true')
    try:
        return json.loads(json_str)
    except:
        return None


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in {func.__name__}: {e}")
            tb = traceback.format_exc()
            print(f"Traceback:\n{tb}")
            return None  
    return wrapper


def apply_decorator_to_all_methods(decorator):
    def class_decorator(cls):
        for attr in dir(cls):
            if callable(getattr(cls, attr)) and not attr.startswith("__"):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return class_decorator


from functools import wraps
def retry(max_attempts=5, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                response = func(*args, **kwargs)
                if response is not None:
                    return response
                attempt += 1
                print(f"Attempt {attempt}/{max_attempts} failed.")
                time.sleep(delay)
            return response
        return wrapper
    return decorator