

def convert_input_argument(**kwargs):
    if "gemini" in kwargs["model"].lower() or "claude" in kwargs["model"].lower():
        kwargs.pop("store", None)
    
    if "gpt-5" in kwargs["model"].lower():
        kwargs["temperature"] = 1.0

    messages = kwargs.get("messages", None)
    if messages is not None:
        for message in messages:
            model_extra = getattr(message, "model_extra", None)
            if model_extra is None:
                continue
            # remove reasoning content
            if model_extra.get("reasoning_content", None) is not None:
                model_extra["reasoning_content"] = None
            if model_extra.get("reasoning", None) is not None:
                model_extra["reasoning"] = None

    kwargs["max_tokens"] = 16384

    return kwargs