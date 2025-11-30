from vllm import LLM, SamplingParams

conclusion_hierarchy = {
    "cot_guide": {
        "knowledge": "When making the final diagnosis, first check whether the diagnosis of each test result of the patient is correct, then summarize what diseases are excluded by each test result, and then use the method of elimination to make the final diagnosis of the patient. \nWhen diagnosing a user's disease, the exclusion method can be used to diagnose the patient's disease. If a patient's test result excludes a disease, there is no need to consider the possibility of the disease (sometimes different diagnostic results of a patient may conflict with the judgment of a disease, but if one of the test results excludes the disease, the disease can be directly excluded). If all diseases are ruled out, then comprehensively consider the results of all the tests and must select a likely disease.",
        "words": "Now, let me summarize the above analysis and select a disease from the list of possible diseases for the patient, ",            
        "max_words": 2048
    },
    "child_type": "sequential",
    "child": [
        {
            "cot_guide": {
                "knowledge": "When reflecting, first check whether the diagnosis of each test result of the patient is correct, and then check whether the possible diseases are correctly excluded according to the test results when making the final diagnosis by the method of elimination. If a patient's test result excludes a disease, there is no need to consider the possibility of the disease (sometimes different diagnostic results of a patient may conflict with the judgment of a disease, but if one of the test results excludes the disease, the disease can be directly excluded). If all diseases are ruled out, then comprehensively consider the results of all the tests and must select a likely disease.",
                "words": "At last, i have made a diagnosis, let's reflect on whether the evidence is reliable. ",
                "max_words": 1024
            }
        },
    ]
    
}

system_prompt = """You are an expert in extracting content from guidebooks. I will give you a guidebook that provides a straightforward understanding of certain medical conditions and the diagnostic tests used to evaluate them. Please use the following format to extract the guidelines on how to rule out related diseases based on the results of different diagnosis test. Please only output a list, do not output anything else:
[
    (name of the first diagnostic test, corresponding disease, corresponding guideline in the guidebook), 
    (name of the second diagnostic test, corresponding disease, corresponding guideline in the guidebook), 
    ...
    (name of the last diagnostic test, corresponding disease, corresponding guideline in the guidebook)
]
For example:
[
    ("Blood Glucose Test", "Pre-Diabetes", "If blood glucose is between 0-99 or 126-500 mg/dL: Rule out Pre-Diabetes."),
    ("Peripheral Blood Smear", "Leukemia", "- **Normal**: Leukemia can be ruled out.\n- **Microcytic Hypochromic**: Leukemia can be ruled out.\n- **Macrocytic**: Leukemia can be ruled out.\n- **Sickle Cells**: Leukemia can be ruled out.\n- **Blast Cells**: No specific diseases can be ruled out.")
]
# Notes
Only one guideline is extracted for each diagnostic test. It contains all the guidelines related to the diagnosis test.
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
    for test_name, disease, knowledge in extracted_content:
        child_tree = {
            "cot_guide": {
                "knowledge": knowledge,
                "words": f"Now, let me check the patient's {test_name} result, the result shows",
                "max_words": 75
            },
            "child_type": "choices",
            "child": [
                {
                    "choice_words": f"test result can rule out {disease}",
                    "cot_guide": {
                        "words": f"Accoding to the {test_name} result and medical knowledge, i can rule out {disease}. So it is impossible for the patient to acquire the {disease}.",
                        "max_words": 0,
                    }
                },
                {
                    "choice_words": f"test result can not rule out {disease}",
                    "cot_guide": {
                        "words": f"The {test_name} result can not rule out {disease}. If the follow-up test results still cannot rule out the {disease}, then the patient may have the {disease}.",
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
    knowledge = "Medical diagnostic knowledge for reference: \n{knowledge}\n".format(knowledge=extracted_content)
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
You are a medical consultant. Below is a list of possible diseases that a patient may have and the results of some of the tests he has done. In addition, we will provide you with relevant knowledge about these possible diseases. Please diagnose the most likely disease among the patient's possible diseases based on the following information.
Please analyze the patient's examination results based on the medical knowledge provided, and determine whether the examination results can rule out the corresponding disease.
# Output format
Please list the key evidence for the diagnosis in the following JSON format, give a conclusion in one sentence, and make a diagnosis for the patient (the patient has only one of the possible diseases):
{
    "Base 1": description of key evidence 1,
    "Base 2": description of key evidence 2,
    ...,
    "Summary": analysis summary,
    "diagnosis": patient's disease (Must be one and only one of the patient's possible diseases)
}
"""

guide_system_prompt = """
You are a medical consultant. Below is a list of possible diseases that a patient may have and the results of some of the tests he has done. Please diagnose the most likely disease among the patient's possible diseases based on the following information.
# Thinking format
When you are thinking, please separate each step of your thinking process and wrap them with <analysis> and </analysis>, for example: 
<think>
<analysis>
First thinking step
</analysis>
...
</think>
Please analyze the patient's examination results based on the medical knowledge provided, and determine whether the examination results can rule out the corresponding disease.
"""

GEN_PROMPT = """# User data
{data_prompt}
"""

GEN_PROMPT_COT = """# User data
{data_prompt}
# Please complete the next step of reasoning according to the guidance below.
guidance: {cot_guide_}
"""

guide_system_prompt_know = """
You are a medical consultant. Below is a list of possible diseases that a patient may have and the results of some of the tests he has done. In addition, we will provide you with relevant knowledge about these possible diseases. Please diagnose the most likely disease among the patient's possible diseases based on the following information.
# Thinking format
When you are thinking, please separate each step of your thinking process and wrap them with <analysis> and </analysis>, for example: 
<think>
<analysis>
First thinking step
</analysis>
...
</think>
Please analyze the patient's examination results based on the medical knowledge provided, and determine whether the examination results can rule out the corresponding disease.
The length of each step should be controlled within the {max_words} word count.
Please strictly follow the medical diagnostic guidelines provided during your thinking process.
"""


GEN_PROMPT_KNOW = """# User data
{data_prompt}
# Medical diagnosis knowledge
{knowledge}
Please strictly follow the medical diagnostic guidelines provided during your thinking process.
"""

GEN_PROMPT_KNOW_COT = """# User data
{data_prompt}
# Medical diagnosis knowledge
{knowledge}
Please strictly follow the medical diagnostic guidelines provided during your thinking process.
# Please complete the next step of reasoning according to the guidance below.
guidance: {cot_guide_}
"""
