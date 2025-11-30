from vllm import LLM, SamplingParams
import re

conclusion_hierarchy = {
    "cot_guide": {
        "knowledge": "When deciding on the final astronomical object, first check the accuracy of each astronomical observation result. Then, summarize which astronomical phenomena were ruled out by each observation result, and then use the process of elimination to make the final judgment on the astronomical object. When determining an astronomical object, you can use the process of elimination to determine the correct astronomical object. If the astronomical observation results rule out an astronomical object, there is no need to consider the possibility of the object. If all astronomical objects are ruled out, then all observation results must be comprehensively considered, including those that cannot be ruled out by the observation results. The probability of these objects can be reconsidered, and finally the most likely astronomical object is selected.",
        "words": "Now, let me summarize the above analysis and select a astronomical object from the list of possible astronomical objects. ",            
        "max_words": 2048
    },
    "child_type": "sequential",
    "child": [
        {
            "cot_guide": {
                "knowledge": "When reflecting, first check whether the judgment of each astronomical observation result is correct, and then, when making the final decision by elimination, check whether the possible astronomical objects are correctly excluded based on the observation results. If a certain observation result excludes a certain astronomical object, there is no need to consider the possibility of the astronomical object. If all astronomical objects are ruled out, then all observation results must be comprehensively considered, including those that cannot be ruled out by the observation results. The probability of these objects can be reconsidered, and finally the most likely astronomical object is selected.",
                "words": "At last, i have made a judgement, let's reflect on whether the evidence is reliable. ",
                "max_words": 1024
            }
        },
    ]
    
}

system_prompt = """You are an expert in extracting content from guidebooks. I will give you a guidebook that introduces you to some incredible astronomical objects and the methods used to observe and understand them. Please use the following format to extract the guidelines on how to rule out related astronomical objects based on the results of different astronomical observations. Please only output a list, do not output anything else:
[
    (name of the first astronomical observation, corresponding astronomical object, corresponding guideline in the guidebook), 
    (name of the second astronomical observation, corresponding astronomical object, corresponding guideline in the guidebook), 
    ...
]
For example:
[
    ("Parallax Measurement", "Quasars", "Observing significant parallax means Quasars are ruled out."),
    ("Radial Velocity Measurement", "Quasars", "Outcome: Radial Velocity between -1000 and 1000: A radial velocity within this range rules out Quasars."),
    ("Radial Velocity Measurement", "Brown Dwarfs and Rogue Planets", "Outcome: Radial Velocity between 1000 and 30000: Observations in this range would rule out Brown Dwarfs and Rogue Planets.")
]
# Notes
Only one guideline is extracted for each astronomical observation and related astronomical objects. It contains all the guidelines related to the observation and objects.
The extracted guideline should be clear and not misleading.
"""

def extract_cot_tree(guide_book, llm, tokenizer):
    child = []

    inpt = f"Here is the guidebook you need to extract:\n{guide_book}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": inpt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=0.65,
        max_tokens=8192,
        n=2
        # stop=['\n']
    )
    responses = llm.generate([text], sampling_params)
    generated_texts = [out.text.strip() for out in responses[0].outputs]
    
    generated_texts = [t.split("</think>")[1].strip() if '</think>' in t else t.strip() for t in generated_texts]
    extracted_content = ''
    for text in generated_texts:
        try:
            text = text.split('[')[-1].split(']')[0]
            list_str = '[' + text + ']'
            extracted_content = eval(list_str)
            for test_name, disease, knowledge in extracted_content:
                i = 1
            break
        except:
            print(text)
            continue
        
    print('Extracted_content:\n', extracted_content)
    for observation_name, object, knowledge in extracted_content:
        child_tree = {
            "cot_guide": {
                "knowledge": knowledge,
                "words": f"Now, let me check the {observation_name} result, the result shows ",
                "max_words": 50
            },
            "child_type": "choices",
            "child": [
                {
                    "choice_words": f"observation result can rule out {object}",
                    "cot_guide": {
                        "words": f"Accoding to the {observation_name} result and astronomical knowledge, i can rule out {object}. So it can't be {object}",
                        "max_words": 0,
                    }
                },
                {
                    "choice_words": f"observation result can not rule out {object}",
                    "cot_guide": {
                        "words": f"The {observation_name} result can not rule out {object}. If other observation results still cannot rule out the {object}, then it may be the {object}.",
                        "max_words": 0,
                    }
                }
            ]
        }
        child.append(child_tree)

    main_hierarchy = {
        "child_type": "parallel",
        "child": child
    }
    knowledge = "Astronomical observation knowledge for reference: \n{knowledge}\n".format(knowledge=extracted_content)
    conclusion_hierarchy['cot_guide']['knowledge'] = knowledge + conclusion_hierarchy['cot_guide']['knowledge']
    conclusion_hierarchy['child'][0]['cot_guide']['knowledge'] = knowledge + conclusion_hierarchy['child'][0]['cot_guide']['knowledge']
    medical_analysis_cot_tree = {
        "child_type": "sequential",
        "child": [
            main_hierarchy,
            conclusion_hierarchy
        ]
    }
    return medical_analysis_cot_tree




system = """
You are a astronomical consultant. Below is a list of possible astronomical objects and the results of some astronomical observations. In addition, we will provide you with relevant knowledge about these possible astronomical objects. Please decide the most likely astronomical object among the possible astronomical objects based on the following information.
Please analyze the observation results based on the astronomical knowledge provided, and determine whether the observation results can rule out the corresponding astronomical object.
# Output format
Please list the key evidence for the decision in the following JSON format, give a conclusion in one sentence, and make a judgement for the astronomical object:
{
    "Base 1": description of key evidence 1,
    "Base 2": description of key evidence 2,
    ...,
    "Summary": analysis summary,
    "judgement": correct astronomical object (Must be one and only one of the possible astronomical objects)
}
"""

guide_system_prompt = """
You are a astronomical consultant. Below is a list of possible astronomical objects and the results of some astronomical observations. Please judge the most likely astronomical object among the possible astronomical objects based on the following information.
# Thinking format
When you are thinking, please separate each step of your thinking process and wrap them with <analysis> and </analysis>, for example: 
<think>
<analysis>
First thinking step
</analysis>
...
</think>
Please analyze the observation results based on the astronomical knowledge provided, and determine whether the observation results can rule out the corresponding astronomical object.
The length of each step should be controlled within the {max_words} word count.
"""

GEN_PROMPT = """# Input Data
{data_prompt}
"""

GEN_PROMPT_COT = """# Input Data
{data_prompt}
Please complete the next step of reasoning according to the guidance below.
guidance: {cot_guide_}
"""

guide_system_prompt_know = """
You are a astronomical consultant. Below is a list of possible astronomical objects and the results of some astronomical observations. In addition, we will provide you with relevant knowledge about these possible astronomical objects. Please judge the most likely astronomical object among the possible astronomical objects based on the following information.
# Thinking format
When you are thinking, please separate each step of your thinking process and wrap them with <analysis> and </analysis>, for example: 
<think>
<analysis>
First thinking step
</analysis>
...
</think>
Please analyze the observation results based on the astronomical knowledge provided, and determine whether the observation results can rule out the corresponding astronomical object.
The length of each step should be controlled within the {max_words} word count.
Please strictly follow the astronomical knowledge provided during your thinking process.
"""

GEN_PROMPT_KNOW = """# Input Data
{data_prompt}
# Astronomical knowledge
{knowledge}
Please strictly follow the astronomical knowledge provided during your thinking process.
"""

GEN_PROMPT_KNOW_COT = """# Input Data
{data_prompt}
# Astronomical knowledge
{knowledge}
Please strictly follow the astronomical knowledge provided during your thinking process.
Please complete the next step of reasoning according to the guidance below.
guidance: {cot_guide_}
"""

