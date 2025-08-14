import json
import requests
import copy
import os
import random
from utils.utils import *


class RapidAPICall():
    def __init__(self, tool, tool_info):
        self.remote = True
        self.name_to_url = tool_info['name_to_url']
        self.headers = {
            "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
            "X-RapidAPI-Host": tool_info['host']
        }
        self.path_params = tool_info['path_params']
        self.tool = tool
        
    @retry(max_attempts=3)
    def _call(self, func_call):
        self.url = self.name_to_url[func_call["name"]]
        params_copy = copy.deepcopy(func_call['arguments'])
        
        param_dict = {}
        for path_param in self.path_params:
            if f"{{{path_param}}}" in self.url and path_param in params_copy:
                param_dict[path_param] = params_copy.pop(path_param)
        self.url = self.url.format(**param_dict)

        for k, value in params_copy.items():
            if k == "legs":
                params_copy[k] = json.dumps(value, ensure_ascii=False)
        try:
            response = requests.get(self.url, headers=self.headers, params=params_copy)
        except:
            return None

        if response.status_code == 200:
            # print("Request success.")
            response = response.json()
            if response['status'] == True:
                if "timestamp" in response:
                    response.pop("timestamp")
                if "data" in response:
                    response = response['data']
            return response
        else:
            return None


    def observation_shorten(self, response):
        if isinstance(response, dict):
            keys_to_delete = [key for key, value in response.items() if (value in ["", None, {}, []])]
            for key in keys_to_delete:
                response.pop(key)

            for key, value in response.items():
                response[key] = self.observation_shorten(value)
                
        elif isinstance(response, list):
            if len(response) > 10 and isinstance(response[0], dict):
                response = response[:10]
            response = [self.observation_shorten(item) for item in response]

        return response


if __name__ == "__main__":
    with open("utils/tool_info.json", 'r') as f:
        tool_info = json.load(f)
    tool_info = tool_info['booking-com15']
    api_call = RapidAPICall(tool="booking-com15", tool_info=tool_info)
    func_call = {
                "name": "Search_Flights_Multi_Stops",
                "arguments": {
                    "legs": [
                        {
                            "fromId": "ORD.AIRPORT",
                            "toId": "HND.AIRPORT",
                            "date": "2024-09-01"
                        },
                        {
                            "fromId": "HND.AIRPORT",
                            "toId": "PVG.AIRPORT",
                            "date": "2024-09-05"
                        },
                        {
                            "fromId": "PVG.AIRPORT",
                            "toId": "ORD.AIRPORT",
                            "date": "2024-09-10"
                        }
                    ]
                }
            }
    response = api_call._call(func_call)
    print(response)
    