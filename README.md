# DrafterBench
The initial version of DrafterBench

## How to evaluate on DrafterBench

### Preparation
First, download the repository.

```shell
git clone https://github.com/Eason-Li-AIS/DrafterBench.git
cd Drafter_Bench
```

Then, install the dependencies.

```shell
pip install -r requirements.txt
```

### Serve MOdel
- For API calling, set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

    ```shell
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...
    MISTRAL_API_KEY=...
    DEEPINFRA_API_KEY=...
    HUGGINGFACE_TOKEN=...
    ```
- For customized model, provide your vllm url when running evaluation.py

    ```shell
    --vllm_url http://xx.xx.xx.xx:8000/v1
    ```

### Run evaluation

```shell
python evaluation.py --model gpt-4o --model-provider openai --temperature 0.0 --exp_name your_experiment_name
```

- To run tasks in a specific language style, use --task_group flag. You chose each language style in ["structured", "unstructured"]. For example:

  ```shell
  python evaluation.py --model gpt-4o --model-provider openai --task_group structured
  ```
  The command will run only the tasks in structured language.

- To have a clear view of the result, you can use --push_result_to flag to upload the results to your huggingface dataset.

  ```shell
  python evaluation.py --model gpt-4o --model-provider openai --task_group structured --push_result_to xxx/xxxx(Your dataset)
  ```
