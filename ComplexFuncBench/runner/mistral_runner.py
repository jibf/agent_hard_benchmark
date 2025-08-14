import re
import copy
import json
from models.mistral import MistralModel
from runner.base_runner import ModelRunner


class MistralRunner(ModelRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.model_name = args.model_name
        self.model = MistralModel(self.model_name)

    def replace_invalid_chars(self, s):
        valid_pattern = re.compile(r'[a-zA-Z0-9_-]')
        result = ''.join([char if valid_pattern.match(char) else '-' for char in s])
        return result[:64]
    
    def get_standard_functions(self, functions):
        self.name_dict = {api['name']: self.replace_invalid_chars(api['name']) for api in functions}
        gpt_functions = [{"type": "function", "function": copy.deepcopy(func)} for func in functions]
        for func in gpt_functions:
            func['function']['name'] = self.name_dict[func['function']['name']]
        return gpt_functions

    def get_standard_fc(self, call):
        tool_call = copy.deepcopy(call)

        function_call = {}
        function_call['name'] = next((k for k, v in self.name_dict.items() if v == tool_call.function.name), None)
        if function_call['name'] is None:
            return None
        function_call['arguments'] = json.loads(tool_call.function.arguments)
        if function_call['arguments'] is None:
            function_call['arguments'] = {}

        return function_call
    
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
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})

            if llm_response.tool_calls:
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{llm_response.tool_calls}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "`self.golden_fcs == []`. Expected to stop. But Model continue to output function call."})
                self.model.messages.append(llm_response)
                tool_calls = llm_response.tool_calls

                function_calls = []
                for tool_call in tool_calls:
                    function_call = self.get_standard_fc(tool_call)
                    if function_call is None:
                        return self.return_result(messages, {"error_type": "name_error", "content": f"{tool_call.function} is not Valid."})
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
                            "role": "tool",
                            "name": self.name_dict[function_call['name']],
                            "content": json.dumps(temp_obs, ensure_ascii=False),
                            "tool_call_id": tool_calls[t].id,
                        }
                    )

                self.process_matches(success_matched)
                    
                self.logger.info(f"Observations:\n{json.dumps(real_time_obs, ensure_ascii=False, indent=4)}\n")
                messages.append({"role": "observation", "content": real_time_obs})

            elif llm_response.content is not None:
                final_response = llm_response.content
                self.logger.info(f"Final Response: {final_response}\n")
                messages.append({"role": "assistant", "content": final_response})

                return self.return_result(messages, self.error_message)

            else:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})