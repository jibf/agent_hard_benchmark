# :wrench: DrafterBench
This repository is the official implementation of DrafterBench. We provide evaluation data, codes, and a brief introduction.

![Static Badge](https://img.shields.io/badge/Code_License-MIT_License-blue) ![Static Badge](https://img.shields.io/badge/Linux_%2F_OSX-passing-green) ![Static Badge](https://img.shields.io/badge/Window-failing-red) ![Static Badge](https://img.shields.io/badge/python-3.10%2B-purple)


---

## :star: Introducing DrafterBench

The DrafterBench is designed to evaluate large language models (LLMs) as an agent to automate monotonous, low-tech, and high-labor-intensity tasks in industry. Our starting point is the drawing revision task complained about by drafters and engineers in **civil engineering**. We took a deep dive into the expected workflow of automation agents on these tasks, simulated the work situation, and evaluated the strengths and limitations of LLMs as automation agents.

![Automation Workflow](/figure/workflow.png "Automation Workflow")

DrafterBench focuses on four essential capabilities:
- **Adapting dynamic language styles**
- **Complex function callings**
- **Batch processing**
- **Critical thinking**

DrafterBench provides a comprehensive evaluation of LLMs with a total of 1920 user instructions over 12 types of drawing revision tasks.

## :ski: Table of Contents

- [Dataset Summary](#dataset-summary)
- [Quick Start](#quick-start)
- [LeaderBoard](#leaderboard)

---

## :clipboard: Dataset Summary

The DrafterBench is constructed on tasks over three object elements, four operations, and six complexity controllers.

| Elements       | Operations | Complexity Controllers |
|--------------|--------------|--------------|
| Text         | Add new content                  |Language style (Structured/Unstructured)                  |
| Table         | Revisit existing content                  |Details ambiguity (Precise/Vague)                  |
| Vector entities         | Map                  |Instruction completeness (Complete/Incomplete)                  |
|          | Update format                  |Objects per instructions (Single/Multiple)                  |
|          |                   |Maximum operation length per object                  |
|          |                   |Task type                    |

The dataset is [available here](https://huggingface.co/datasets/Eason666/DrafterBenchmark) on Huggingface.

## :fire: Quick Start

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

- To run tasks of a specific set, use --task_group flag. You can choose each set in ["Structured", "Unstructured", "Precise", "", "Completed", "Error", "Single_Object", "Multiple_Objects", "Single_Operation", "Multiple_Operations"]. For example:

  ```shell
  python evaluation.py --model gpt-4o --model-provider openai --task_group structured
  ```
  This command will run only the tasks in a structured language.

- To have a clear view of the result, you can set up your huggingface token, 
  ```shell
   HUGGINGFACE_TOKEN=...
  ```
  then use --huggingface_user_name flag to provide your Huggingface user name. Our benchmark will create a new dataset repository with --exp_name and push the results to it.
  ```shell
  python evaluation.py --model gpt-4o --model-provider openai --task_group structured --push_result_to xxx/xxxx(Your dataset)
  ```

## :mortar_board: LeaderBoard

| Metric  |  OpenAI o1 | GPT-4o | Claude3.5-sonnet | Deepseek v3 | Qwen 2.5 | Llama3-Instruct |
|---------|-----------|------------|------------------|-------------|----------|-----------------|
| Structured language    | **81.58**     | 75.14      | 74.34            | <ins>75.66</ins>       | 74.57    | 68.96           |
| Unstructured language  | **82.26**     | 73.84      | <ins>78.20</ins>            | 75.04       | 72.16    | 67.92           |
| Precise detail      | **89.82**     | 79.41      | <ins>81.15</ins>            | 78.79       | 75.12    | 71.36           |
| Vague detail      | **74.02**     | 69.57      | 71.39            | <ins>71.91</ins>       | 71.55    | 65.37           |
| Complete instruction     | 80.98     | 81.70      | 83.72            | <ins>85.59</ins>       | **87.58**    | 83.10           |
| Error (incomplete) instruction     | **82.85**     | 67.27      | <ins>68.83</ins>            | 65.10       | 59.16    | 53.78           |
| Single object  | **83.23**     | 73.75      | <ins>74.97</ins>            | 74.94       | 74.18    | 67.22           |
| Multiple objects | **80.60**     | 75.23      | <ins>77.57</ins>            | 75.76       | 72.56    | 69.66           |
| Single operation  | **82.45**     | 75.84      | 76.62            | <ins>77.17</ins>       | 75.88    | 71.02           |
| Multiple operations | **80.84**     | 71.79      | <ins>75.56</ins>            | 71.70       | 68.36    | 63.27           |
| Tasks      | **81.92**     | 74.49      | <ins>76.27</ins>            | 76.05       | 73.37    | 68.44           |
|  Comprehensive score       |  **79.90** :trophy:    | 71.76      | <ins>73.79</ins> :gem:           | 73.09       | 70.52    | 64.95           |

## Citation

If you use DrafterBench in your research, please consider citing our paper:

```bibtex
@article{drafterbench,
  title={DrafterBenchmark: Benchmarking Large Language Models for Tasks Automation in Civil Engineering},
  author={Yinsheng Li, Zhen Dong, Yi Shao.},
  year={2025},
  url={},
}
