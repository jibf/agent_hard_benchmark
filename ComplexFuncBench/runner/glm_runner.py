import re
import copy
import json
from models.glm import GLMAPIModel, GLMVllmModel
from runner.base_runner import ModelRunner


class GLMRunner(ModelRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.model_name = args.model_name
        self.model = GLMVllmModel(args.vllm_url, self.model_name)

    def run(self, data):
        convs, functions = data['conversations'], data['functions']
        self.CompareClass.add_free_function(convs)

        messages = []
        query = convs[0]['content']
        messages.append({"role": "user", "content": query})

        self.init_golden(convs)

        while True:
            llm_response = self.model(messages, tools=functions)
           
            if "function_call" in json.dumps(llm_response, ensure_ascii=False):
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{llm_response}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "`self.golden_fcs == []`. Expected to stop. But Model continue to output function call."})
                if len(llm_response) == 2:
                    messages.append(llm_response[0])
                    self.logger.info(f"Thought: {llm_response[0]['content']}")

                function_calls = llm_response[-1]['function_call']
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
                    if not isinstance(temp_obs, str):
                        temp_obs = json.dumps(temp_obs, ensure_ascii=False)
                    messages.append({"role": "observation", "content": temp_obs})

                self.process_matches(success_matched)
                    
                self.logger.info(f"Observations:\n{json.dumps(real_time_obs, ensure_ascii=False, indent=4)}\n")
                # messages.append({"role": "observation", "content": real_time_obs})
                
            elif llm_response is not None:
                final_response = llm_response[0]['content']
                self.logger.info(f"Final Response: {final_response}\n")
                messages.append({"role": "assistant", "content": final_response})

                return self.return_result(messages)
            
            elif llm_response.finish_reason == "length":
                self.logger.info(f"{llm_response}")
                return self.return_result(messages, {"error_type": "exceed_max_length", "content": "The response is too long."})

            else:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})
            

class GLMAPIRunner(GLMRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.model_name = args.model_name
        self.model = GLMAPIModel(self.model_name)
    
    def replace_invalid_chars(self, s):
        # 使用正则表达式匹配有效字符
        valid_pattern = re.compile(r'[a-zA-Z0-9_-]')
        
        # 使用list comprehension替换不符合要求的字符
        result = ''.join([char if valid_pattern.match(char) else '-' for char in s])
        
        # 如果字符串长度超过64个字符，截断为前64个字符
        return result[:64]
    
    def get_standard_functions(self, functions):
        # self.name_dict = {api['name']: self.replace_invalid_chars(api['name']) for api in functions}
        gpt_functions = [{"type": "function", "function": copy.deepcopy(func)} for func in functions]
        # for func in gpt_functions:
        #     func['function']['name'] = self.name_dict[func['function']['name']]
        return gpt_functions

    def get_standard_fc(self, tool_call):
        try:
            return {"name": tool_call['function']['name'], "arguments": json.loads(tool_call['function']['arguments'])}
        except:
            return None

    def run(self, data):
        convs, functions = data['conversations'], data['functions']
        self.CompareClass.add_free_function(convs)
        gpt_functions = self.get_standard_functions(functions)

        messages = []
        query = convs[0]['content']
        messages.append({"role": "user", "content": query})

        self.init_golden(convs)

        while True:
            llm_response = self.model(messages, tools=gpt_functions)
            if llm_response is None:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})

            if llm_response.finish_reason == "tool_calls":
                if self.golden_fcs == []:
                    self.logger.error(f"Output FC:\n{llm_response.tool_calls}")
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "`self.golden_fcs == []`. Expected to stop. But Model continue to output function call."})
                if llm_response.message is not None:
                    self.model.messages.append({"role": "assistant", "content": llm_response.message.content})
                    self.logger.info(f"Thought: {llm_response.message.content}")
                self.model.messages.append({"role": "assistant", "tool_calls": llm_response.tool_calls})
                # self.model.message.append(llm_response.message.model_dump())
                tool_calls = llm_response.tool_calls

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
                            "role": "tool",
                            "content": json.dumps(temp_obs, ensure_ascii=False),
                            "tool_call_id": tool_calls[t].id,
                        }
                    )

                self.process_matches(success_matched)
                    
                self.logger.info(f"Observations:\n{json.dumps(real_time_obs, ensure_ascii=False, indent=4)}\n")
                messages.append({"role": "observation", "content": real_time_obs})

            elif llm_response.finish_reason == "stop":
                final_response = llm_response.message.content
                self.logger.info(f"Final Response: {final_response}\n")
                messages.append({"role": "assistant", "content": final_response})

                return self.return_result(messages)

            else:
                self.logger.info(f"{llm_response}")
                return self.return_result(messages, {"error_type": "unknown_error", "content": "llm_response is None"})