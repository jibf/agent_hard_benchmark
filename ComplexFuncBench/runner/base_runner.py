import json
import copy
from utils.compare_method import CompareFC

class ModelRunner:
    def __init__(self, args, logger):
        self.logger = logger
        
        self.error_message = None
        self.unexpect_call_resp = {"api_status": True, "content": "There is a problem with your api call, please double-check for possible problems."}
    
        self.CompareClass = CompareFC(args, logger)
        self.free_function_list = self.CompareClass.free_function_list

    def only_free_function(self, temp_fcs):
        for call in temp_fcs:
            if call['name'] == "Search_Hotels" and call["arguments"]["search_type"] == "hotel":
                return True

        temp_fcs = set([fc["name"] for fc in temp_fcs])

        return temp_fcs.issubset(set(self.free_function_list))
 
    def get_success_turn(self, remain_fcs, total_fcs):
        remain_ids = []
        for idx, fc_list in enumerate(total_fcs):
            for remain_fc in remain_fcs:
                if remain_fc in fc_list:
                    remain_ids.append(idx)
        if remain_ids == []:
            return len(total_fcs)
        
        return max(min(remain_ids), 0)

    def init_golden(self, convs):
        self.fc_chain = []
        self.obs_chain = []
        for turn in convs:
            if "function_call" in turn:
                self.fc_chain.append(turn['function_call'])
            elif turn['role'] == "observation":
                self.obs_chain.append(turn['content'])
        
        assert len(self.fc_chain) == len(self.obs_chain), "function call and observation length mismatch."

        self.turn_id, self.correct_count = 0, 0
        self.golden_fcs, self.golden_obs = copy.deepcopy(self.fc_chain[self.turn_id]), copy.deepcopy(self.obs_chain[self.turn_id])

        if self.only_free_function(self.golden_fcs):
            self.update_current_golden()

    def update_current_golden(self):
        self.turn_id += 1
        if self.turn_id < len(self.fc_chain):
            self.golden_fcs.extend(copy.deepcopy(self.fc_chain[self.turn_id]))
            self.golden_obs.extend(copy.deepcopy(self.obs_chain[self.turn_id])) 
    
    def return_result(self, messages, error_info=None):
        if error_info:
            success_turn = self.get_success_turn(self.golden_fcs, self.fc_chain)
            return messages, error_info, success_turn, self.correct_count

        # free function post process
        if len(self.golden_fcs) != 0:
            for call in self.golden_fcs:
                if call['name'] == "Search_Hotels" and call["arguments"]["search_type"] == "hotel":
                    self.golden_fcs.remove(call)
                if call['name'] in ["Search_Hotel_Destination", "Search_Attraction_Location", "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location"]:
                    self.golden_fcs.remove(call)
            
        if self.turn_id < len(self.fc_chain) or len(self.golden_fcs) > 0:
            self.logger.info(f"turn id  = {self.turn_id}; len(golden_answer) = {len(self.fc_chain)}")
            self.logger.info(f"golden_function_calls = {self.golden_fcs}")
            return self.return_result(messages, {"error_type": "stop_early", "content": "Stop early."})
        elif len(self.golden_fcs) == 0:
            return messages, "Success.", len(self.fc_chain), self.correct_count
        else:
            raise NotImplementedError("Unexpected error.")
    
    def process_matches(self, success_matched):
        for matched in success_matched:
            if matched in self.golden_fcs:
                self.golden_obs.pop(self.golden_fcs.index(matched))
                self.golden_fcs.remove(matched)
        
        if len(success_matched) > 0:  
            self.update_current_golden()

        for k, v in self.CompareClass.free_functions.items():
            if v['called'] == True and json.loads(k) in self.golden_fcs:
                self.golden_obs.pop(self.golden_fcs.index(json.loads(k)))
                self.golden_fcs.remove(json.loads(k))
        
        if self.only_free_function(self.golden_fcs):
            self.update_current_golden()