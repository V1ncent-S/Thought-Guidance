# Thought Guidance- Retrieval Augmented Generation (TG-RAG)

This repository contains the official implementation for the paper: "TG-RAG: A Reasoning-Time Retrieval Framework for Steering Expert-Level Reasoning".

TG-RAG is a novel reasoning-time retrieval framework designed to mitigate "Cognitive Drift" in Large Reasoning Models (LRMs)  by proactively steering their thought processes. Unlike traditional RAG methods that provide passive context, TG-RAG employs a dynamic "Interrupt-Retrieve-Generate" (IRG) cycle. It retrieves procedural directives from a structured Expert Process Graph (EPG) —modeled as a "Chain-of-Trees" —and injects them directly into the model's active reasoning chain. This approach establishes a new paradigm of Thought Guidance (TG) , ensuring faithful adherence to expert-defined workflows in complex, specialized domains.

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
| `task_type`       | The evaluation task to perform. Choices: `medical`, `astronomy`, `math`, `qa`.                     | **Required**      | `None`  |
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