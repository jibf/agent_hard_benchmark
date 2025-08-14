from typing import Any, Text
from dataclasses import dataclass

@dataclass
class SimpleTemplatePrompt:
    template: str
    args_order: list

    def __call__(self, **kwargs: Any) -> Text:
        self.cur_template = self.template
        args = [kwargs[arg] for arg in self.args_order]
        for i, arg in enumerate(args):
            if isinstance(arg, int): arg = str(arg)
            self.cur_template = self.cur_template.replace(f"[args{str(i+1)}]", arg)
        return self.cur_template
