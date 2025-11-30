
main_hierarchy = {
    "child_type": "sequential",
    "child": [
        {
            "cot_guide": {
                "words": "Okay, let's answer this question now. First, I need to outline a step-by-step plan.",
                "max_words": 2000
            }
        },
        {
            "cot_guide": {
                "words": "Next, let's solve the problem according to the plan.",
                "max_words": 2000
            }
        },
        {
            "cot_guide": {
                "words": "Okay, I have given the answer. Now, let’s reflect on whether there are any calculation errors and pay attention to the output format requirements.",
                "max_words": 1000
            }
        }
    ]
}


conclusion_hierarchy = {
    "cot_guide": {
        "words": "Okay, I have given the answer. Now, let’s reflect on whether there are any calculation errors and pay attention to the output format requirements.",
        "max_words": 200
    }    
}

gsm8k_cot_tree = {
    "child_type": "sequential",
    "child": [
        main_hierarchy,
        # conclusion_hierarchy
    ]
}


system = """You are an intelligent assistant. Please answer the math questions given to you.
#Output format
Please output your answer after ####. Do not output other content.
#### [Your answer]
## Output example:
#### 123
"""

guide_system_prompt = """You are an intelligent assistant. Please answer the math questions given to you.
# Thinking format
When you are thinking, please separate each step of your thinking process and wrap them with <analysis> and </analysis>, for example: 
<think>
<analysis>
First thinking step
</analysis>
...
</think>
The length of each step should be controlled within the {max_words} word count.
"""

GEN_PROMPT = """# Question
{data_prompt}
"""

GEN_PROMPT_COT = """# Question
{data_prompt}

Please complete the next step of reasoning according to the guidance below.
Guidance: {cot_guide_}
"""
