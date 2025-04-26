class Score_builder:
    def ground_fill(self, ground_details):
        self.total_score = 0
        (
            self.total_arguments,
            self.total_variables_transfer,
            self.total_function_calls,
            self.total_single_tool_calls,
            self.total_multi_tool_calls,
            self.total_plan_execution,
        ) = ground_details
        return self

    def result(self, point: float, code_details, plan_details):
        self.total_score = point
        self.arguments_count = code_details[0]
        self.variables_transfer_count = code_details[1]
        self.function_calls_count = code_details[2]
        self.single_tool_calls_count = code_details[3]
        self.multi_tool_calls_count = code_details[4]
        self.intersected_plan_count = code_details[5]
        self.all_plan = plan_details[5]
        return self

    def fail(self):
        self.total_score = 0
        self.arguments_count = 0
        self.variables_transfer_count = 0
        self.function_calls_count = 0
        self.single_tool_calls_count = 0
        self.multi_tool_calls_count = 0
        self.intersected_plan_count = 0
        self.all_plan = self.total_plan_execution
        return self
