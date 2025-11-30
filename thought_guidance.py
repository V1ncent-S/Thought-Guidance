from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from collections import Counter
import re
from json_repair import repair_json
import numpy as np

import argparse  
import pandas as pd

def setup_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", 
        type=str, 
        default=None,
        help="Please enter the path to your model checkpoint"
    )
    parser.add_argument(
        "task_type", 
        type=str, 
        choices=['medical', 'astronomy', 'law', 'math', 'qa'],
        help="Please select the task to be evaluated, optional: ['medical', 'astronomy', 'law', 'math', 'qa']"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="The number of data samples to process, default is 100"
    )
    return parser.parse_args()



def query_qwen_model(msgs, n):
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        n=n
        # stop=['\n']
    )
    responses = llm.generate([text], sampling_params)
    generated_texts = [out.text.strip() for out in responses[0].outputs]
    generated_texts = [t.split("</think>")[1].strip() if '</think>' in t else t.strip() for t in generated_texts]
    return generated_texts




def generate_with_cot_words(data_prompt, cot_prefix, cot_guide):

    cot_guide_words = cot_guide["words"]
    max_words = cot_guide["max_words"]
    knowledge = cot_guide.get('knowledge')

    if max_words == 0:
        return '<analysis>\n' + cot_guide_words + '\n</analysis>\n'
    else:
        if knowledge is None:
            prompt = GEN_PROMPT.format(
                data_prompt=data_prompt,
            )
            messages = [
                {"role": "system", "content": guide_system_prompt.format(max_words=max_words)},
                {"role": "user", "content": prompt}
            ]
        else:
            prompt = GEN_PROMPT_KNOW.format(
                data_prompt=data_prompt,
                knowledge=knowledge
            )
            messages = [
                {"role": "system", "content": guide_system_prompt_know.format(max_words=max_words)},
                {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max_words+50,
            stop=['</think>', '</analysis>']
        )
        
        text += cot_prefix + '<analysis>\n' + cot_guide_words

        responses = llm.generate([text], sampling_params)
        generated_text = responses[0].outputs[0].text

        return '<analysis>\n' + cot_guide_words + generated_text.strip() + '\n</analysis>\n'


selection_sys_prompt = """Suppose the following is an analysis text by an analysis expert. Based on the analysis results, please select the option that matches the analysis description from the given options, output the serial number directly, and do not output other content."""
selection_prompt_temp = """## analysis text
{prefix}
## Options
{actions}"""

selection_sys_prompt_ch = """假设下面是一段分析专家的分析文本，请根据分析结果，从给出的选项里选出符合分析描述的选项，直接输出序号，不要输出其他内容。"""
selection_prompt_temp_ch = """## 分析文本
{prefix}
## 选项
{actions}"""

def select_cot_words(data_prompt, cot_prefix, cot_choices):
    cot_choices += ["Neither is suitable."]
    # cot_choices += ['都不合适。']  # for law scenario
    
    actions = [f"{idx+1}.{cot_words}" for idx,cot_words in enumerate(cot_choices)]
    action_str = "\n".join(actions)

    prompt = selection_prompt_temp.format(prefix=cot_prefix, actions=action_str)
    # prompt = selection_prompt_temp_ch.format(prefix=cot_prefix, actions=action_str)  # for law scenario

    # prompt += ' /no_think'   # for Qwen3 series model to accelerate the selection process

    messages = [
        {"role": "system", "content": selection_sys_prompt},
        # {"role": "system", "content": selection_sys_prompt_ch},   # for law scenario
        {"role": "user", "content": prompt}
    ]
    generated_responses = query_qwen_model(messages, 2)
    valid_choices = []
    for resp in generated_responses:
        if resp[:1].isdigit():
            choice = int(resp[:1])
            if choice <= len(cot_choices):
                valid_choices.append(choice)
    if len(valid_choices) == 0:
        return -1
    counts = Counter(valid_choices)
    choice_idx = counts.most_common(1)[0][0]
    if choice_idx == len(cot_choices):
        return -1
    else:
        return choice_idx-1


open_guidance_system_prompt = """You are an expert analyst and a master of logical reasoning. Your task is to determine the single most logical next step in a complex reasoning process, based on the problem description and the reasoning so far. 
You must NOT perform the step yourself. Your entire output should be a short, concise phrase (3-7 words) describing this next analytical step. Do not add any preamble, explanation, or quotation marks. Just provide the name of the step."""

open_guidance_user_prompt = """
### Original Problem ###
{query}

### Reasoning So Far ###
{history}
---
Given the analysis so far, what is the single most logical next step to continue the investigation?
"""

def generate_open_guidance(query, history):

    user_prompt = open_guidance_user_prompt.format(query=query, history=history)
    

    messages = [
        {"role": "system", "content": open_guidance_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    generated_responses = query_qwen_model(messages, 1)
    
    return generated_responses[0]


def run_with_cot_tree(data_prompt, cot_prefix, cot_tree):
    cot = ""
    if "cot_guide" in cot_tree:
        generated_words = generate_with_cot_words(
            data_prompt,
            cot_prefix,
            cot_tree['cot_guide']
        )
        cot += generated_words 
        cot_prefix += generated_words

    if "child" in cot_tree:
        if cot_tree['child_type'] == "sequential":
            for child_tree in cot_tree['child']:
                generated_words = run_with_cot_tree(data_prompt, cot_prefix, child_tree)
                cot_prefix += generated_words + "\n\n"
                cot += generated_words + "\n\n"
            cot += "\n\n"
        elif cot_tree['child_type'] == "parallel":
            parallel_cot_words = []
            for child_tree in cot_tree['child']:
                generated_words = run_with_cot_tree(data_prompt, cot_prefix, child_tree)
                parallel_cot_words.append(generated_words)
            cot += "\n\n".join(parallel_cot_words)
        elif cot_tree['child_type'] == "choices":
            cot_choices = [tree['choice_words'] for tree in cot_tree['child']]
            choice_idx = select_cot_words(data_prompt, cot_prefix, cot_choices)
            if choice_idx >= 0:
                child_tree = cot_tree['child'][choice_idx]
            else:
                open_guidance = generate_open_guidance(data_prompt, cot_prefix)
                child_tree = {
                    "cot_guide": {
                        "words": open_guidance,
                        "max_words": 512
                    }
                }
            generated_words = run_with_cot_tree(data_prompt, cot_prefix, child_tree)
            cot += generated_words + "\n\n" 
        else:
            pass


    return cot.strip()


def cot_guide_generate(system, data_prompt, cot_tree):
    cot = run_with_cot_tree(data_prompt, "", cot_tree)
    
    cot = cot.replace('<analysis>\n', '')
    cot = cot.replace('<analysis>', '')
    cot = cot.replace('</analysis>\n', '')
    cot = cot.replace('</analysis>', '')

    messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": data_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    prompt = text + cot + '\n</think>\n'

    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=2048,
    )
    responses = llm.generate([prompt], sampling_params)
    generated_text = responses[0].outputs[0].text
    return prompt + generated_text


if __name__ == "__main__":
    args = setup_parser()

    llm = LLM(
        model=args.model_path,
        max_model_len=16384,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True
    )
    tokenizer = llm.get_tokenizer()

    if args.task_type == 'medical':
        from example.medical import guide_system_prompt, GEN_PROMPT, guide_system_prompt_know, GEN_PROMPT_KNOW
        from example.medical import extract_cot_tree, system

        data_list = []
        with open("/data/Medical_analysis/truth_num=4+action_num=6+valid_truth_num=1.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(json.loads(line))

        result_list = []
        for i, data in enumerate(data_list[:args.num_samples]):
            seed = data['seed']
                
            prompt = 'The patient can only suffer from the following diseases: ' + str(data["truths"]) + '. \nThe results of the patient\'s examination were:\n ' + str(data["observations"]) + '.\nPlease diagnose the most likely disease among the patient\'s possible diseases based on the given information.'
            with open(f'/data/Medical_analysis/knowledge_book/truth_num=4+action_num=6+valid_truth_num=1/seed={seed}.txt', 'r', encoding='utf-8') as f:
                guide_book = f.read()

            cot_tree = extract_cot_tree(guide_book, llm, tokenizer)

            answer = cot_guide_generate(system, prompt, cot_tree)
            try:
                ans = answer.split("</think>")[-1].split('{')[1].split('}')[0]
                ans = '{' + ans + '}'
                res = ans.replace("\n", "")
                res = json.loads(repair_json(res))
                res = res['diagnosis']
            except:
                res = 'no result'

            label = data['valid_truth']

            x = {}
            x['system'] = system
            x['input'] = prompt
            x['output'] = answer
            x['res'] = res
            x['label'] = label
            result_list.append(x)
        
        with open('Medical_result_TG.json', 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
    
    elif args.task_type == 'astronomy':

        from example.astronomy import extract_cot_tree, system
        from example.astronomy import guide_system_prompt, GEN_PROMPT, guide_system_prompt_know, GEN_PROMPT_KNOW    

        data_list = []
        with open("/data/AstronomyEnv/truth_num=4+action_num=6+valid_truth_num=1.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(json.loads(line))

        result_list = []
        for i, data in enumerate(data_list[:args.num_samples]):
            seed = data['seed']
            
            prompt = 'This astronomical phenomenon can only be one of the following astronomical objects:\n' + str(data["truths"]) + '\nThe astronomical observation results of this phenomenon are:\n' + str(data["observations"]) + '\nPlease judge the most likely astonomical object among the possible objects based on the given information.'        
            with open(f'/data/AstronomyEnv/knowledge_book/truth_num=4+action_num=6+valid_truth_num=1/seed={seed}.txt', 'r') as f:
                guide_book = f.read()
                    
            cot_tree = extract_cot_tree(guide_book, llm, tokenizer)

            answer = cot_guide_generate(system, prompt, cot_tree)
            try:
                ans = answer.split("</think>")[-1].split('{')[1].split('}')[0]
                ans = '{' + ans + '}'
                res = ans.replace("\n", "")
                res = json.loads(repair_json(res))
                res = res['judgement']
            except:
                res = 'no result'

            label = data['valid_truth']

            x = {}
            x['system'] = system
            x['input'] = prompt
            x['output'] = answer
            x['res'] = res
            x['label'] = label
            result_list.append(x)
        
        with open('Astronomy_result_TG.json', 'w', encoding='utf-8') as f:
                json.dump(result_list, f, ensure_ascii=False, indent=4)

    elif args.task_type == 'law':

        from example.law_calculation import guide_system_prompt, GEN_PROMPT, guide_system_prompt_know, GEN_PROMPT_KNOW
        from example.law_calculation import law_calculation_cot_tree, system

        with open('/data/Law/3-7.json', 'r', encoding='utf-8') as file:
            data_list = json.load(file)

        result_list = []
        for i, row in enumerate(data_list[:args.num_samples]):
            prompt = '# 案情文本\n' + row['question']
            answer = cot_guide_generate(system, prompt, law_calculation_cot_tree)
            label = float(row['answer'].split("犯罪金额:")[1].split("元")[0])
            pattern = r"{.*?}"
            try:
                res = answer.split('</think>')[-1].replace("\n", "")
                json_str = re.findall(pattern, res)[0]
                res = json.loads(repair_json(json_str))
                score = float(res['答案'])
            except:
                score = -1
            x = {}
            x['system'] = system
            x['input'] = row['question']
            x['output'] = answer
            x['score'] = score
            x['label'] = label
            result_list.append(x)
            
        with open('Law_result_TG.json', 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)

    elif args.task_type == 'math':
        from example.gsm8k import guide_system_prompt, GEN_PROMPT
        from example.gsm8k import gsm8k_cot_tree, system


        data_df = pd.read_parquet('/data/gsm8k/test-00000-of-00001.parquet')

        result_list = []
        for i, row in data_df.head(args.num_samples).iterrows():
            prompt = '# Question\n' + row['question']
            answer = cot_guide_generate(system, prompt, gsm8k_cot_tree)

            label = row['answer'].strip()

            try:
                res = answer.split('</think>')[-1].split('####')[-1].strip()
                score = res
            except:
                score = None

            x = {}
            x['system'] = system
            x['input'] = row['question']
            x['output'] = answer
            x['score'] = score
            x['label'] = label
            result_list.append(x)
            
        with open('GSM8K_result_TG.json', 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)

    elif args.task_type == 'qa':
        from example.strategyqa import guide_system_prompt, GEN_PROMPT
        from example.strategyqa import strategyqa_cot_tree, system

        with open('/data/StrategyQA/strategyqa.json', 'r', encoding='utf-8') as file:
            data_list = json.load(file)

        result_list = []
        for i, row in enumerate(data_list[:args.num_samples]):
            facts = "\n".join([f"{i}. {fact}" for i, fact in enumerate(row['facts'], start=1)])
            prompt = '# Question\n' + row['question'] + "\n# Supporting Facts:\n" + facts

            answer = cot_guide_generate(system, prompt, strategyqa_cot_tree)
            label = row['answer']
            
            try:
                res = answer.split('</think>')[-1].split('Answer:')[-1].strip()
                score = bool(res)
            except:
                score = None

            x = {}
            x['system'] = system
            x['input'] = row['question']
            x['output'] = answer
            x['score'] = score
            x['label'] = label
            result_list.append(x)
            
        with open('SQA_result_TG.json', 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
