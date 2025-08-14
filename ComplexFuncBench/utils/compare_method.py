import json
import copy
import gc
import random
import torch
import numpy as np
import warnings
from FlagEmbedding import FlagModel
from scipy.optimize import linear_sum_assignment

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", message=".*fast tokenizer.*")

from utils.utils import *
from utils.rapidapi import RapidAPICall
from models.gpt import GPTModel
from prompts.compare import system_prompt, user_prompt
from utils.logger import Logger

class CompareFCBase:
    def __init__(self, args, logger) -> None:
        self.embedding = FlagModel('BAAI/bge-large-en-v1.5', 
                        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                        use_fp16=True, devices='cpu')

        with open("utils/tool_info.json", 'r') as f:
            tool_info = json.load(f)
        tool_info = tool_info['booking-com15']
        self.api_call = RapidAPICall(tool="booking-com15", tool_info=tool_info)
        self.model = GPTModel("gpt-4o-2024-08-06")
        self.logger = logger
        self.error_message = []
        self.exact_match_dict = load_json("utils/exact_match_values.json")
        self.free_function_list = ["Location_to_Lat_Long", "Search_Hotel_Destination", "Search_Attraction_Location", 
                                   "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location"]

    def format_check(self, func_call, functions):
        name_to_func = {func['name']: func for func in functions}

        if func_call['name'] not in name_to_func:
            return {"error": f"Function {func_call['name']} is not defined in the function list."}
        
        used_func = name_to_func[func_call['name']]
        required_params = used_func['parameters']['required']
        if not set(required_params).issubset(set(func_call['arguments'].keys())):
            # 找到在required_params中，但不在func_call['arguments'].keys()中的参数
            missing_param = set(required_params) - set(func_call['arguments'].keys())
            return {"error": f"Function {used_func['name']} requires parameters {required_params}, but {list(func_call['arguments'].keys())} do not provide {missing_param}"}
        
        if not set(func_call['arguments'].keys()).issubset(set(used_func['parameters']['properties'].keys())):
            missing_param = set(func_call['arguments'].keys()) - set(used_func['parameters']['properties'].keys())
            return {"error":f"Function {used_func['name']} does not have parameters {missing_param}" }
        
        # 参数类型验证
        for param_name, param_value in func_call['arguments'].items():
            if used_func['parameters']['properties'][param_name]['type'] == "string":
                if not isinstance(param_value, str):
                    return {"error": f"Parameter {param_name} of function {used_func['name']} should be a string, but {type(param_value)} is provided."}
                
            elif used_func['parameters']['properties'][param_name]['type'] == "number":
                if not isinstance(param_value, int) and not isinstance(param_value, float):
                    return {"error": f"Parameter {param_name} of function {used_func['name']} should be a number, but {type(param_value)} is provided."}
                
            elif used_func['parameters']['properties'][param_name]['type'] == "boolean":
                if not isinstance(param_value, bool):
                    return {"error": f"Parameter {param_name} of function {used_func['name']} should be a boolean, but {type(param_value)} is provided."}
                
            elif used_func['parameters']['properties'][param_name]['type'] == "array":
                if not isinstance(param_value, list):
                    return {"error": f"Parameter {param_name} of function {used_func['name']} should be an array, but {type(param_value)} is provided."}
        
        return True
    
    def add_free_function(self, convs):
        # free function is optional
        self.free_functions = {}
        for i, turn in enumerate(convs):
            if "function_call" not in turn:
                continue
            for j, func_call in enumerate(turn['function_call']):
                if func_call['name'] in self.free_function_list:
                    if json.dumps(func_call) not in self.free_functions:
                        self.free_functions[json.dumps(func_call)] = {
                            "called": False, 
                            "obs": convs[i+1]['content'][j]
                        }

    def rule_based(self, predict, golden):
        """
        Rule-based Match.
        """
        if predict['name'] != golden['name']:
            return False
        if sorted(predict['arguments'].keys()) != sorted(golden['arguments'].keys()):
            return False
        for k, v in predict['arguments'].items():
            if k == "categories_filter":
                pred_filter = [s.strip() for s in v.split(",")]
                golden_filter = [s.strip() for s in golden['arguments'][k].split(",")]
                if set(golden_filter) == set(pred_filter):
                    continue
            if v != golden['arguments'][k]:
                return False
        
        return True

    def response_based(self, predict, golden):
        try:
            resp_1 = self.api_call._call(predict)
            if resp_1 == {}:
                self.error_message = f"API call failed for {predict}."
                return False
            if isinstance(resp_1, dict):
                if "status" in resp_1 and resp_1["status"] == False:
                    self.error_response = resp_1
            else:
                self.error_response = resp_1
            resp_2 = self.api_call._call(golden)
        except:
            return False
        if resp_1 is None or resp_2 is None:
            return False
        
        return resp_1 == resp_2

    def similarity_based(self, predict, golden):
        embedding_1 = self.embedding.encode([json.dumps(predict, ensure_ascii=False)])
        embedding_2 = self.embedding.encode([json.dumps(golden, ensure_ascii=False)])
        similarity = embedding_1 @ embedding_2.T
        del embedding_1, embedding_2
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.debug(f"Similarity-based comparison output: {similarity[0][0]}")
        return similarity[0][0] > 0.98

    def llm_based(self, functions, history, predict, golden):
        kwargs = {
            "functions": json.dumps(functions, ensure_ascii=False),
            "history": json.dumps(history, ensure_ascii=False),
            "function_call_1": json.dumps(predict, ensure_ascii=False),
            "function_call_2": json.dumps(golden, ensure_ascii=False),
        }

        output = self.model(system_prompt, user_prompt, **kwargs)

        decode_output = decode_json(output)

        if decode_output:
            self.logger.debug(f"LLM-based comparison output: {decode_output}")
            return decode_output['is_equal']
        else:
            return None


class CompareFC(CompareFCBase):
    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)

    def value_checker(self, pred_call, golden_call):
        if pred_call['name'] != golden_call['name']:
            return False, {"error_type": "func_error", "content": "Do not call the correct function."}
        param_list = self.exact_match_dict[pred_call['name']]
        for k, v in golden_call['arguments'].items():
            if k in param_list:
                if k not in pred_call['arguments']:
                    return False, {"error_type": "param_missing", "content": f"Missing parameter {k} in prediction."}
                elif k != "categories_filter" and pred_call['arguments'][k] != v:
                    return False, {"error_type": "value_error", "content": f"Parameter {k} value is not correct in prediction."}
                elif k == "categories_filter":
                    golden_filter = [s.strip() for s in v.split(",")]
                    pred_filter = [s.strip() for s in pred_call['arguments'][k].split(",")]
                    if not set(golden_filter) == set(pred_filter):
                        return False, {"error_type": "value_error", "content": f"Parameter {k} value is not correct in prediction."}
        return True, ""
    
    def remove_called_fc(self, golden, golden_obs):
        pop_index = []
        for singel_golden in golden:
            if json.dumps(singel_golden) in self.free_functions and self.free_functions[json.dumps(singel_golden)]['called'] == True:
                pop_index.append(golden.index(singel_golden))
        
        for index in sorted(pop_index, reverse=True):
            golden.pop(index)
            golden_obs.pop(index)
        return golden, golden_obs

    def get_error_message(self, pred_call, golden_call):
        # value error
        for k, v in golden_call['arguments'].items():
            if k not in pred_call['arguments']:
                return {"error_type": "param_missing", "content": f"Missing parameter {k} in prediction."}
            if v != pred_call['arguments'][k]:
                return {"error_type": "value_error", "content": f"Parameter {k} value do not equal to golden."}
        
        # hallucination
        for k, v in pred_call['arguments'].items():
            if k not in golden_call['arguments']:
                return {'error_type': "param_hallucination", "content": f"Parameter {k} is hallucinated."}
            
    def mapping_call(self, predict, golden, golden_obs):
        def sort_arguments(call_list):
            for value in call_list:
                sorted_arguments = {k: value['arguments'][k] for k in sorted(value['arguments'])}
                value['arguments'] = sorted_arguments
        sort_arguments(predict)
        sort_arguments(golden)    

        # exact match
        exact_matches = []
        remaining_predict = []
        remaining_predict_index = {}
        remaining_golden = []
        remaining_golden_index = {}
        matched_indices = set()

        for p_index, p_value in enumerate(predict):
            match_found = False
            for g_index, g_value in enumerate(golden):
                if g_index in matched_indices:
                    continue
                if p_value == g_value:
                    exact_matches.append({
                        "idx": p_index,
                        "pred_call": p_value,
                        "golden_call": g_value,
                        "golden_obs": golden_obs[g_index]
                    })
                    matched_indices.add(g_index)
                    match_found = True
                    break
                elif json.dumps(p_value) in self.free_functions:
                    exact_matches.append({
                        "idx": p_index,
                        "pred_call": p_value,
                        "golden_call": p_value,
                        "golden_obs": self.free_functions[json.dumps(p_value)]['obs']
                    })
                    match_found = True
                    self.free_functions[json.dumps(p_value)]['called'] = True
                    break

            if not match_found:
                remaining_predict.append(p_value)
                remaining_predict_index[len(remaining_predict) - 1] = p_index

        for g_index, g_value in enumerate(golden):
            if g_index not in matched_indices:
                remaining_golden.append(g_value)  
                remaining_golden_index[len(remaining_golden) - 1] = g_index
        
        if remaining_predict == [] or remaining_golden == []:
            return exact_matches
        
        # embedding match
        pred_embed = self.embedding.encode([json.dumps(value, ensure_ascii=False) for value in remaining_predict])
        gold_embed = self.embedding.encode([json.dumps(value, ensure_ascii=False) for value in remaining_golden])
        matrix = pred_embed @ gold_embed.T
        
        del pred_embed, gold_embed
        torch.cuda.empty_cache()
        gc.collect()

        row_ind, col_ind = linear_sum_assignment(-matrix)  

        embedding_matches = []
        for i, j in zip(row_ind, col_ind):
            embedding_matches.append({
                "idx": remaining_predict_index[i],
                "pred_call": remaining_predict[i],
                "golden_call": remaining_golden[j],
                "golden_obs": golden_obs[remaining_golden_index[j]]
            })
        matching = exact_matches + embedding_matches

        return matching

    def compare_single_call(self, functions, history, pred_call, golden_call):
        self.logger.info(f"Start compare_single_call: \n{pred_call}\n{golden_call}")
        # rule-based
        if self.rule_based(pred_call, golden_call):
            self.logger.info(f"Rule-based compare success.")
            return True, None
        
        is_valid, error_message = self.value_checker(pred_call, golden_call)
        if not is_valid:
            self.logger.info(f"{error_message}")
            return False, error_message
        
        # Response-based
        if self.response_based(pred_call, golden_call):
            self.logger.info(f"Response-based compare success.")
            return True, None
        
        # LLM-based
        if self.llm_based(functions, history, pred_call, golden_call):
            self.logger.info(f"LLM-based compare success.")
            return True, None
        
        self.logger.info(f"All compare method failed.")
        return False, None

    def compare_turn_prediction(self, functions, history, predict, golden, golden_obs):
        self.error_message = []
        golden, golden_obs = self.remove_called_fc(golden, golden_obs)
        
        if len(golden) == 0:
            raise NotImplementedError()

        match_list = self.mapping_call(predict, golden, golden_obs)

        format_error = {}
        success_map, success_matched = {}, []
        for match_item in match_list:
            # format error check
            message = self.format_check(match_item['pred_call'], functions)
            if message == True:
                is_match, single_message = self.compare_single_call(functions, history, match_item['pred_call'], match_item['golden_call'])
                if is_match:
                    success_map[match_item['idx']] = match_item['golden_obs']
                    success_matched.append(match_item['golden_call'])
                else:
                    if single_message:
                        self.error_message.append(single_message)
                    else:
                        self.error_message.append(self.get_error_message(match_item['pred_call'], match_item['golden_call']))

            else:
                format_error[match_item['idx']] = message
        
        self.logger.info(f"Success matched: {success_matched}")
        
        return self.error_message, success_map, success_matched, format_error