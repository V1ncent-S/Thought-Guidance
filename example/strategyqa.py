
main_hierarchy = {
    "child_type": "sequential",
    "child": [
        {
            "cot_guide": {
                "words": "Okay, let's answer this question now. First, I need to confirm what implicit sub-questions this question has. For example:",
                "max_words": 100
            }
        },
        {
            "cot_guide": {
                "words": "Now let's analyze the sub-questions one by one.",
                "max_words": 200
            }
        },
    ]
}


conclusion_hierarchy = {
    "cot_guide": {
        "words": "Okay, let's synthesize the findings from the previous analysis and answer the question.",
        "max_words": 200
    },
    "child_type": "sequential",
    "child": [
          {
              "cot_guide": {
                  "words": "Finally, I have given the answer. Now let’s reflect on whether there are any factual errors in the previous analysis and pay attention to the output format requirements.",
                  "max_words": 200
              }
          }
    ]
    
}

strategyqa_cot_tree = {
    "child_type": "sequential",
    "child": [
        main_hierarchy,
        conclusion_hierarchy
    ]
}


system = """You are an intelligent assistant. Please answer the logic questions given to you.
#Output format
Please output your answer in the following format. Do not output other content.
Answer: Your answer (True or False)
## Output example:
Answer: True
"""

guide_system_prompt = """You are an intelligent assistant. Please answer the logic questions given to you.
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
