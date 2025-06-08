# Thought Guidance

This repository contains the **anonymous implementation** of our paper  
**"Thought Guidance: Steering Reasoning Models Toward More Expert and Precise Thinking"**,  
submitted to EMNLP 2025.

## 🧠 Overview

We propose a novel prompting technique for large reasoning models, called **Thought Guidance**, which dynamically inserts guidance tokens during the model's thinking process. By steering the reasoning trajectory, our method enables the model to produce more **expert-like** and **precise** thinking paths, ultimately improving its performance on complex reasoning tasks.

## 🧪 Running Inference

To evaluate our method on the medical diagnosis reasoning tasks:

```bash 
python main.py --input_path data/dev.json --model llama2-13b --insert_method guided
```

Here we only show the examples of medical diagnosis scenarios in the paper. We will upload the fully organized code later.
