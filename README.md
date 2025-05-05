# DrafterBench
This repository is the offical implementation of DrafterBench. We provides evaluation data, evaluation code and a brief introduction about DrafterBench.

---

## Introducing DrafterBench

The DrafterBench is designed to evaluate language language models (LLMs) as a agent to automating monotonous, low-tech, and high-labor-intensity tasks in industry. The drawing revision task, in Civil Engineering, complained by drafters and engineers is our start point. We took a deep dive into the expected workflow of automation agents on this tasks, simulated the works situation, and evaluate the strenghts and limitations of LLMs as automation agents.

![Automation Workflow](.figure/workflow.png "Automation Workflow")

DrafterBench focues on four essential capabilities:
- **Adpating dynamic language styles**
- **Complex function callings**
- **Batch processing**
- **Critical thinking**

DrafterBench provide a comprehensive evaulation on the LLMs with a total of 1920 user instructions over 12 types if drawing revision tasks.

## Table of Content

- [Dataset Summary](#dataset-summary)
- [Quick Start](#quick-start)
- [LeaderBoard](#leaderboard)

---

## Dataset Summary

The DrafterBench is constructed on tasks over three object elements, four operations and six complexity controllers.

| Elements       | Operations | Complexity Controllers |
|--------------|--------------|--------------|
| Text         | Adding new content                  |Language style (Structured/Unstructured)                  |
| Table         | Revsising exsist content                  |Details ambiguity (Precise/Vague)                  |
| Vector entities         | Mapping                  |Instruction completeness (Complete/Incomplete)                  |
|          | Updating format                  |Objects per instructions (Single/Multiple)                  |
|          |                   |Maxmium operation length per object                  |
|          |                   |Task type                    |

The dataset is [available](https://huggingface.co/datasets/Eason666/DrafterBenchmark) on Huggingface Hub

## Quick Start

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

- To run tasks in a specific language style, use --task_group flag. You chose each language style in ["structured", "unstructured"]. For example:

  ```shell
  python evaluation.py --model gpt-4o --model-provider openai --task_group structured
  ```
  The command will run only the tasks in structured language.

- To have a clear view of the result, you can use --push_result_to flag to upload the results to your huggingface dataset.

  ```shell
  python evaluation.py --model gpt-4o --model-provider openai --task_group structured --push_result_to xxx/xxxx(Your dataset)
  ```

## LeaderBoard

| Metric  |  OpenAI o1 | ChatGPT-4o | Claude3.5-sonnet | Deepseek v3 | Qwen 2.5 | Llama3-Instruct |
|---------|-----------|------------|------------------|-------------|----------|-----------------|
| *STRU*    | **81.58**     | 75.14      | 74.34            | <ins>75.66</ins>       | 74.57    | 68.96           |
| *U-STRU*  | **82.26**     | 73.84      | <ins>78.20</ins>            | 75.04       | 72.16    | 67.92           |
| *P*       | **89.82**     | 79.41      | <ins>81.15</ins>            | 78.79       | 75.12    | 71.36           |
| *VA*      | **74.02**     | 69.57      | 71.39            | <ins>71.91</ins>       | 71.55    | 65.37           |
| *COM*     | 80.98     | 81.70      | 83.72            | <ins>85.59</ins>       | **87.58**    | 83.10           |
| *ERR*     | **82.85**     | 67.27      | <ins>68.83</ins>            | 65.10       | 59.16    | 53.78           |
| *SIN-OB*  | **83.23**     | 73.75      | <ins>74.97</ins>            | 74.94       | 74.18    | 67.22           |
| *MULT-OB* | **80.60**     | 75.23      | <ins>77.57</ins>            | 75.76       | 72.56    | 69.66           |
| *SIN-OP*  | **82.45**     | 75.84      | 76.62            | <ins>77.17</ins>       | 75.88    | 71.02           |
| *MULT-OP* | **80.84**     | 71.79      | <ins>75.56</ins>            | 71.70       | 68.36    | 63.27           |
| *TA*      | **81.92**     | 74.49      | <ins>76.27</ins>            | 76.05       | 73.37    | 68.44           |
|  Comprehensive score       |  **79.90** :trophy:    | 71.76      | <ins>73.79</ins> :gem:           | 73.09       | 70.52    | 64.95           |

*STU*:Structured language, *U-STU*:Unstructured language, *P*:Precise detail, *VA*:Vague detail, *COM*:Complete instruction, *ERR*:Error(incomplete) instruction, *SIN-OB*:Single object, *MULT-OB*:Multiple objects, *SIN-OP*:Single operation, *MULT-OP*:Multiple operations, *TA*:Tasks

## Citation

If you use DrafterBench in your research, please cite our paper:

```bibtex
@article{drafterbench,
  title={DrafterBenchmark: Benchmarking Large Language Models for Tasks Automation in Civil Engineering},
  author={Yinsheng Li, Zhen Dong, Yi Shao.},
  year={2025},
  url={},
}
