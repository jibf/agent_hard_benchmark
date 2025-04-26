How to calculate on DrafterBench

1. Install the dependencies
    pip install -r requirements.txt

2. Prepare your model
2.1 For API calling, set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...
    MISTRAL_API_KEY=...
    DEEPINFRA_API_KEY=...
    HUGGINGFACE_TOKEN=...

2.2 For customized model, provide your vllm url when running evaluation.py
    --vllm_url http://xx.xx.xx.xx:8000/v1

3. Run DrafterBench
    python evaluation.py --model gpt-4o --model-provider openai --temperature 0.0 --exp_name your_experiment_name

3.1 To run tasks in a specific language style, use --task_group flag. You chose each language style in ["structured", "unstructured"]. For example:
    python evaluation.py --model gpt-4o --model-provider openai --task_group structured
    The command will run only the tasks in structured language

3.2 To have a clear view of the result, you can use --push_result_to flag to upload the results to your huggingface dataset.
    python evaluation.py --model gpt-4o --model-provider openai --task_group structured --push_result_to xxx/xxxx(Your dataset)