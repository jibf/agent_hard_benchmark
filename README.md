# :wrench: DrafterBench
This repository is the official implementation of DrafterBench. We provide evaluation data, codes, and a brief introduction.

![Static Badge](https://img.shields.io/badge/Code_License-MIT_License-blue) ![Static Badge](https://img.shields.io/badge/Linux_%2F_OSX-passing-green) ![Static Badge](https://img.shields.io/badge/Window-failing-red) ![Static Badge](https://img.shields.io/badge/python-3.11-purple)


---

## :star: Introducing DrafterBench

The DrafterBench is designed to evaluate large language models (LLMs) as an agent to automate monotonous, low-tech, and high-labor-intensity tasks in industry. Our initiative is drawing revision, which is a representation task in civil engineering that urgently needs to be automated. We took the following workflow to simulate the working scenario and evaluate the strengths and limitations of LLMs as automation agents.

![Automation Workflow](/figures/Workflow.png "Automation Workflow")

After the stage of preprocessing, the drawing revision tasks (summarized from the real world, totalling 1920 across 12 types) are converted into natural language processing (NLP) tasks to evaluate complex function calls instructed by intricate and lengthy content commands. We designed over 40 drawing revision tools and provided them to LLMs, which play different functions. Some of them aim to make visible changes to drawings, while the others serve necessary preparations for them (e.g., opening the file or transferring critical arguments). It's difficult to determine whether the tools called are effective and functioning properly from the revised drawings, especially when checking if there are redundant or duplicated invisible tools. Therefore, to accurately evaluate the models‘ performance, we score their responses based on the operation chains rather than the revised drawing results.

To record the operation chains, we prepared dual functions for the tools provided to the LLMs. Each dual function has the same name, input, and output type as the original tools, and its function is to capture the operations and valuable data in a well-structured JSON format (e.g., argument value, data type, etc.). During the working of the benchmark, the original tools called by the models will be replaced with dual functions to record the operation chains and help the final assessment.

There are four essential capabilities evaluated by DrafterBench:
- **Structured data understanding**
- **Function execution**
- **Instruction following**
- **Critical reasoning**

![Capabilities Illustration](/figures/Capabilities.png "Capabilities Illustration")

## :ski: Table of Contents

- [Dataset Summary](#dataset-summary)
- [Quick Start](#quick-start)
- [LeaderBoard](#leaderboard)

---

## :clipboard: <span id="dataset-summary">Dataset Summary</span>

The DrafterBench is constructed on tasks over three object elements, four operations, and six complexity controllers.

| Elements         | Operations              | Complexity Controllers                       | Capacities Investigated by Various Complexity         |
|------------------|-------------------------|----------------------------------------------|-------------------------------------------------------|
| Text             | Add new content         |Language style (Structured/Unstructured)      |Structured data understanding                          |
| Table            | Revise content          |Task categories                               |Function execution                                     |
| Vector entity    | Change position         |Objects per instruction (Single/Multiple)     |Instruction following                                  |
|                  | Update format           |Operations per object (Single/Multiple)       |Instruction following                                  |
|                  |                         |Instruction completeness (Complete/Incomplete)|Critical reasoning                                     |
|                  |                         |Detail ambiguity (Precise/Vague)              |Critical reasoning                                     |

The dataset is [available here](https://huggingface.co/datasets/Eason666/DrafterBenchmark) on Huggingface.

## :fire: <span id="quick-start">Quick Start</span>

### Preparation
First, download the repository.

```shell
git clone https://github.com/Eason-Li-AIS/DrafterBench.git
cd DrafterBench
```

Then, install the dependencies.

```shell
pip install -e .
```

### Serve Model
- For API calling, set up your OpenAI / Anthropic / Google / Mistral / Deepinfra / AnyScale or other API keys as environment variables.

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
Specify the --model and --model-provider flags to run DrafterBench. The supported models and providers are [available here](https://docs.litellm.ai/docs/providers). You can name your experiment with the --exp_name flag, or it will be set as "model+time+task_group" by default.
```shell
python evaluation.py --model gpt-4o-2024-08-06 --model-provider openai --temperature 0.0
```

- To run tasks of a specific set, use the --task_group flag. You can choose each set in ["Structured", "Unstructured", "Precise", "Vague", "Complete", "Error", "Single_Object", "Multiple_Objects", "Single_Operation", "Multiple_Operations"]. For example:

  ```shell
  python evaluation.py --model gpt-4o-2024-08-06 --model-provider openai --task_group Structured
  ```
  This command will run only the tasks in a structured language. The default task group is "All" tasks.

- To have a clear view of the result, you can set up your huggingface token, 
  ```shell
   HUGGINGFACE_TOKEN=...
  ```
  then use the --huggingface_user_name flag to provide your Huggingface user name. Our benchmark will create a new dataset repository with the --exp_name and push the results to it. This repository is private by default, you can create a public repository by setting the --huggingface_private flag to False.
  ```shell
  python evaluation.py --model gpt-4o-2024-08-06 --model-provider openai --task_group Structured --huggingface_user_name XXXXX(Replace "XXXXX" with your Huggingface username)
  ```
- The default prompts for 12 tasks can be found in ./prompts. You are encouraged to develop your own prompts to achieve a higher score. To do so, simply replace the default prompts in .txt file with your new prompts.

## :mortar_board: <span id="leaderboard">LeaderBoard</span>

| Metric  |  OpenAI o1 | gpt-4o-2024-08-06 | Claude3.5-sonnet | Deepseek-v3-685B | Qwen2.5-72B-Instruct | Llama3-70B-Instruct |
|---------|-----------|------------|------------------|-------------|----------|-----------------|
| Structured language    | **81.58**     | 75.14      | 74.34            | <ins>75.66</ins>       | 74.57    | 68.96           |
| Unstructured language  | **82.26**     | 73.84      | <ins>78.20</ins>            | 75.04       | 72.16    | 67.92           |
| Precise detail      | **89.82**     | 79.41      | <ins>81.15</ins>            | 78.79       | 75.12    | 71.36           |
| Vague detail      | **74.02**     | 69.57      | 71.39            | <ins>71.91</ins>       | 71.55    | 65.37           |
| Complete instruction     | 80.98     | 81.70      | 83.72            | <ins>85.59</ins>       | **87.58**    | 83.10           |
| Incomplete instruction (Error)     | **82.85**     | 67.27      | <ins>68.83</ins>            | 65.10       | 59.16    | 53.78           |
| Single object  | **83.23**     | 73.75      | <ins>74.97</ins>            | 74.94       | 74.18    | 67.22           |
| Multiple objects | **80.60**     | 75.23      | <ins>77.57</ins>            | 75.76       | 72.56    | 69.66           |
| Single operation  | **82.45**     | 75.84      | 76.62            | <ins>77.17</ins>       | 75.88    | 71.02           |
| Multiple operations | **80.84**     | 71.79      | <ins>75.56</ins>            | 71.70       | 68.36    | 63.27           |
| Average of 12 tasks      | **81.92**     | 74.49      | <ins>76.27</ins>            | 76.05       | 73.37    | 68.44           |
|  Comprehensive score       |  **79.90** :trophy:    | 71.76      | <ins>73.79</ins> :gem:           | 73.09       | 70.52    | 64.95           |

Note: We have recently upgraded DrafterBench to be more challenging. Although the trend of models' ability is very consistent with the above leaderboard, some models may score lower than the records.

## Citation

If you use DrafterBench in your research, please consider citing our paper:

```bibtex
@article{drafterbench,
  title={DrafterBenchmark: Benchmarking Large Language Models for Tasks Automation in Civil Engineering},
  author={Yinsheng Li, Zhen Dong, Yi Shao.},
  year={2025},
  url={},
}
