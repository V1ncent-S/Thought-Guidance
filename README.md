# Thought Guidance (TG)

This repository contains the official implementation for the paper: **"[Thought Guidance: Steering Large Reasoning Models Towards Expert-Level and Precise Thinking via Reasoning-Time Guidance]"**.

Thought Guidance (TG) is a novel framework designed to enhance the reasoning capabilities of Large Language Models (LLMs) by dynamically steering their thought processes. Instead of relying on a single, static prompt, TG uses an **Expert Process Graph (EPG)** to provide iterative, state-aware guidance, ensuring the model's reasoning aligns with structured, expert-defined workflows. This approach significantly improves performance on complex tasks requiring deep reasoning, particularly in specialized domains.

## Getting Started

Follow these instructions to set up the environment and run the evaluation scripts.

### Prerequisites

**Install the required packages:**
It is highly recommended to use a virtual environment (e.g., `venv` or `conda`).
```sh
pip install -r requirements.txt
```

## Usage

The main script `thought_guidance.py` is used to run evaluations on different tasks using the Thought Guidance framework.

### Command-Line Arguments

The script accepts the following command-line arguments:

| Argument          | Description                                                                                             | Required/Optional | Default |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ----------------- | ------- |
| `model_path`      | The local path to your Hugging Face model checkpoint directory.                                         | **Required**      | `None`  |
| `task_type`       | The evaluation task to perform. Choices: `medical`, `astronomy`, `law`, `math`, `qa`.                     | **Required**      | `None`  |
| `--num_samples`   | The number of data samples to process from the test set.                                                | Optional          | `100`   |

### Running an Evaluation

Navigate to the project directory and run the script using the following format:

```sh
python thought_guidance.py [model_path] [task_type] --num_samples [number]
```

### Example
To run an evaluation on the medical task using a model located at /path/to/my_model and evaluating on the first 100 samples:

```sh
python thought_guidance.py /path/to/my_model medical --num_samples 100
```