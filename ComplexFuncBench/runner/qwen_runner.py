import re
import copy
import json
from models.qwen import QwenModel
from runner.base_runner import ModelRunner


class QwenRunner(ModelRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.model_name = args.model_name
        self.model = QwenModel(self.model_name)
    
    def get_standard_functions(self, functions):
        return [{"type": "function", "function": copy.deepcopy(func)} for func in functions]

    def get_standard_fc(self, tool_call):
        try:
            return {"name": tool_call['function']['name'], "arguments": json.loads(tool_call['function']['arguments'])}
        except:
            return None
    
    def run(self, data):
        convs, functions = data['conversations'], data['functions']
        self.CompareClass.add_free_function(convs)
        standard_functions = self.get_standard_functions(functions)

        messages = []
        query = convs[0]['content']
        messages.append({"role": "user", "content": query})

        self.init_golden(convs)

        while True:
            llm_response = self.model(messages, tools=standard_functions)

            if llm_response is None:
                raise NotImplementedError("LLM response is None")

            if llm_response['tool_calls'] != None:
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{llm_response}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "`self.golden_fcs == []`. Expected to stop. But Model continue to output function call."})
                if llm_response['content'] is None:
                    llm_response['content'] = ""
                self.model.messages.append(llm_response)
                tool_calls = llm_response['tool_calls']

                function_calls = []
                for tool_call in tool_calls:
                    function_call = self.get_standard_fc(tool_call)
                    if function_call is None:
                        return self.return_result(messages, {"error_type": "decode_error", "content": f"{tool_call} is not Valid."})
                    function_calls.append(function_call)
                self.logger.info(f"Function Calls: \n{json.dumps(function_calls, ensure_ascii=False, indent=4)}\n")
                self.logger.info(f"Golden Function Call: \n{json.dumps(self.golden_fcs, ensure_ascii=False, indent=4)}\n")
                messages.append({"role": "assistant", "function_call": function_calls})
                
                self.error_message, success_map, success_matched, format_error = self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1], 
                    copy.deepcopy(function_calls), self.golden_fcs, 
                    self.golden_obs
                )
                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)
                self.correct_count += len(success_map)

                real_time_obs = []
                for t, function_call in enumerate(function_calls):
                    if t in success_map:
                        temp_obs = success_map[t]
                    elif t in format_error:
                        temp_obs = format_error[t]
                    else:
                        temp_obs = self.unexpect_call_resp
                        
                    real_time_obs.append(temp_obs)
                    self.model.messages.append(
                        {
                            "name": function_call['name'],
                            "role": "tool",
                            "content": json.dumps(temp_obs, ensure_ascii=False)
                        }
                    )
                self.process_matches(success_matched)
                    
                self.logger.info(f"Observations:\n{json.dumps(real_time_obs, ensure_ascii=False, indent=4)}\n")
                messages.append({"role": "observation", "content": real_time_obs})

            elif llm_response['tool_calls'] == None:
                final_response = llm_response['content']
                self.logger.info(f"Final Response: {final_response}\n")
                messages.append({"role": "assistant", "content": final_response})

                return self.return_result(messages)

            else:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})