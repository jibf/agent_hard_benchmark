# :wrench: DrafterBench
This repository is the official implementation of DrafterBench. We provide evaluation data, codes, and a brief introduction.

![Static Badge](https://img.shields.io/badge/Code_License-MIT_License-blue) ![Static Badge](https://img.shields.io/badge/Linux_%2F_OSX-passing-green) ![Static Badge](https://img.shields.io/badge/Window-failing-red) ![Static Badge](https://img.shields.io/badge/python-3.10%2B-purple)


---

## :star: Introducing DrafterBench

The DrafterBench is designed to evaluate large language models (LLMs) as an agent to automate monotonous, low-tech, and high-labor-intensity tasks in industry. Our starting point is the drawing revision task complained about by drafters and engineers in **civil engineering**. We took a deep dive into the expected workflow of automation agents on these tasks, simulated the work situation, and evaluated the strengths and limitations of LLMs as automation agents.

In this work, after preprocessing, the drafter tasks (summarized from the real world, in total of 1920 over 12 types) are converted to NLP tasks that evaluate complex function callings instructed by long content commands in vision tasks. Over 40 drawing revision tools are tailored and provided to LLMs. Meanwhile, their dual functions/tools are designed with the same tool name, input, and the same type of output, but they record factual operations rather than make changes to drawings. Dual functions were introduced to cope with the fact that two different chains of operations may accidentally produce the same output drawing since some operations are not visible. Therefore, a more accurate evaluation can be conducted based on their records than output drawings.

![Automation Workflow](/figures/Workflow.png "Automation Workflow")

DrafterBench evaluates models focusing on four essential capabilities:
- **Structured textual data understanding**
- **Complex function calling**
- **Long content instruction following**
- **Critical thinking**

![Capabilities Illustration](/figures/Capabilities.png "Capabilities Illustration")

## :ski: Table of Contents

- [Dataset Summary](#dataset-summary)
- [Quick Start](#quick-start)
- [LeaderBoard](#leaderboard)

---

## :clipboard: <span id="dataset-summary">Dataset Summary</span>

The DrafterBench is constructed on tasks over three object elements, four operations, and six complexity controllers.

| Elements       | Operations | Complexity Controllers |
|--------------|--------------|--------------|
| Text         | Add new content                  |Language style (Structured/Unstructured)                  |
| Table         | Revise existing content                  |Details ambiguity (Precise/Vague)                  |
| Vector entities         | Change position                 |Instruction completeness (Completed/Error)                  |
|          | Update format                  |Objects per instructions (Single/Multiple)                  |
|          |                   |Maximum operation length per object (Single/Multiple)                 |
|          |                   |Task type                    |

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
pip install -r requirements.txt
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
python evaluation.py --model gpt-4o-2024-08-06 --model-provider openai --temperature 0.0 --exp_name model+time+task_group
```

- To run tasks of a specific set, use --task_group flag. You can choose each set in ["Structured", "Unstructured", "Precise", "Vague", "Completed", "Error", "Single_Object", "Multiple_Objects", "Single_Operation", "Multiple_Operations"]. For example:

  ```shell
  python evaluation.py --model gpt-4o-2024-08-06 --model-provider openai --task_group Structured
  ```
  This command will run only the tasks in a structured language.

- To have a clear view of the result, you can set up your huggingface token, 
  ```shell
   HUGGINGFACE_TOKEN=...
  ```
  then use --huggingface_user_name flag to provide your Huggingface user name. Our benchmark will create a new dataset repository with --exp_name and push the results to it. This repository is private by default, you can create a public repository by setting the --huggingface_private flag to False.
  ```shell
  python evaluation.py --model gpt-4o-2024-08-06 --model-provider openai --task_group Structured --huggingface_user_name XXXXX(Replace "XXXXX" with your Huggingface username)
  ```
- The default prompts for 12 tasks can be found in ./prompts. You are encouraged to develop your own prompts to achieve a higher score. To do so, just replace the default prompts in .txt file with your new prompts.

## :mortar_board: <span id="leaderboard">LeaderBoard</span>

| Metric  |  OpenAI o1 | gpt-4o-2024-08-06 | Claude3.5-sonnet | Deepseek-v3-685B | Qwen2.5-72B-Instruct | Llama3-70B-Instruct |
|---------|-----------|------------|------------------|-------------|----------|-----------------|
| Structured language    | **81.58**     | 75.14      | 74.34            | <ins>75.66</ins>       | 74.57    | 68.96           |
| Unstructured language  | **82.26**     | 73.84      | <ins>78.20</ins>            | 75.04       | 72.16    | 67.92           |
| Precise detail      | **89.82**     | 79.41      | <ins>81.15</ins>            | 78.79       | 75.12    | 71.36           |
| Vague detail      | **74.02**     | 69.57      | 71.39            | <ins>71.91</ins>       | 71.55    | 65.37           |
| Completed instruction     | 80.98     | 81.70      | 83.72            | <ins>85.59</ins>       | **87.58**    | 83.10           |
| Error (incompleted) instruction     | **82.85**     | 67.27      | <ins>68.83</ins>            | 65.10       | 59.16    | 53.78           |
| Single object  | **83.23**     | 73.75      | <ins>74.97</ins>            | 74.94       | 74.18    | 67.22           |
| Multiple objects | **80.60**     | 75.23      | <ins>77.57</ins>            | 75.76       | 72.56    | 69.66           |
| Single operation  | **82.45**     | 75.84      | 76.62            | <ins>77.17</ins>       | 75.88    | 71.02           |
| Multiple operations | **80.84**     | 71.79      | <ins>75.56</ins>            | 71.70       | 68.36    | 63.27           |
| Average of 12 tasks      | **81.92**     | 74.49      | <ins>76.27</ins>            | 76.05       | 73.37    | 68.44           |
|  Comprehensive score       |  **79.90** :trophy:    | 71.76      | <ins>73.79</ins> :gem:           | 73.09       | 70.52    | 64.95           |

Note: Due to additional internal updates, models are more difficult to score highly in this benchmark, and some models may score lower than the records in this table.

## Citation

If you use DrafterBench in your research, please consider citing our paper:

```bibtex
@article{drafterbench,
  title={DrafterBenchmark: Benchmarking Large Language Models for Tasks Automation in Civil Engineering},
  author={Yinsheng Li, Zhen Dong, Yi Shao.},
  year={2025},
  url={},
}
