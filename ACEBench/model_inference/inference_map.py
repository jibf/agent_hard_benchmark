from model_inference.common_inference import CommonInference
from model_inference.apimodel_inference import APIModelInference


inference_map_groups = {
    # All API models now use dynamic creation with unified API_KEY/BASE_URL pattern
    # This ensures consistent configuration across all benchmarks
    APIModelInference: [],
    CommonInference: [
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "qwen2.5-7b-instruct-local",
        "Meta-Llama-3.1-8B-Instruct-local",
        "watt-tool-8B-local",
        "Hammer2.1-7b-local",
        "ToolACE-8B-local",
        "functionary-small-v3.1-local",
        "xLAM-7b-r-local",
        "Llama-3.1-8B-Instruct-local",
        "Qwen2.5-7B-Instruct-local",
        "MiniCPM3-4B-local",
        "Phi-3-mini-128k-instruct-local",
        "Hammer2.1-3b-local",
        "Qwen2.5-3B-Instruct-local",
        "Llama-3.2-3B-Instruct-local",
        "Qwen3-8B-local",
        "Qwen3-32B-local",
    ],
}


inference_map = {model: handler for handler, models in inference_map_groups.items() for model in models}