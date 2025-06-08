from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from collections import Counter
import re
from json_repair import repair_json
import numpy as np
from Medical_analysis import guide_system_prompt, GEN_PROMPT, guide_system_prompt_know, GEN_PROMPT_KNOW


def query_qwen_model(msgs):
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        n=2
    )
    responses = llm.generate([text], sampling_params)
    generated_texts = [out.text.strip() for out in responses[0].outputs]
    generated_texts = [t.split("</think>")[1].strip() if '</think>' in t else t.strip() for t in generated_texts]
    return generated_texts




def generate_with_cot_words(data_prompt, cot_prefix, cot_guide):
    """ Generate max_words words based on the given cot guide words."""

    cot_guide_words = cot_guide["words"]
    max_words = cot_guide["max_words"]
    knowledge = cot_guide.get('knowledge')

    if max_words == 0:
        return '<analysis>' + cot_guide_words + '</analysis>'
    else:
        if knowledge is None:
            prompt = GEN_PROMPT.format(
                data_prompt=data_prompt,
            )
            messages = [
                {"role": "system", "content": guide_system_prompt.format(min_words=max_words)},
                {"role": "user", "content": prompt}
            ]
        else:
            prompt = GEN_PROMPT_KNOW.format(
                data_prompt=data_prompt,
                knowledge=knowledge
            )
            messages = [
                {"role": "system", "content": guide_system_prompt_know.format(min_words=max_words)},
                {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            temperature=0.5,
            max_tokens=max_words+50,
            stop=['</think>', '</analysis>']
        )
        text += cot_prefix + '<analysis>\n' + cot_guide_words
        responses = llm.generate([text], sampling_params)
        generated_text = responses[0].outputs[0].text

        return '<analysis>' + cot_guide_words + generated_text + '</analysis>'


selection_sys_prompt = """Suppose the following is an analysis text by an analysis expert. Based on the analysis results, please select the option that matches the analysis description from the given options, output the serial number directly, and do not output other content."""
selection_prompt_temp = """## analysis text
{prefix}
## Options
{actions}"""


def select_cot_words(data_prompt, cot_prefix, cot_choices):
    cot_choices += ["Neither is suitable."]
    
    actions = [f"{idx+1}.{cot_words}" for idx,cot_words in enumerate(cot_choices)]
    action_str = "\n".join(actions)
    prompt = selection_prompt_temp.format(prefix=cot_prefix, actions=action_str)
    messages = [
        {"role": "system", "content": selection_sys_prompt},
        {"role": "user", "content": prompt}
    ]
    generated_responses = query_qwen_model(messages)
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
                generated_words = run_with_cot_tree(data_prompt, cot_prefix, cot_tree['child'][choice_idx])
                cot += generated_words + "\n\n" 
        elif cot_tree['child_type'] == "tool_calling":
            for child_tree in cot_tree['child']:
                tool_result = run_with_cot_tree(data_prompt, generated_words, child_tree)
                cot += tool_result + "\n\n"
        else:
            pass

    return cot.strip()


def cot_guide_generate(system, data_prompt, cot_tree):
    cot = run_with_cot_tree(data_prompt, "", cot_tree)
    cot = cot.replace('<analysis>', '')
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
    
    prompt = text + cot + '</think>\n'

    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=2048,
    )
    responses = llm.generate([prompt], sampling_params)
    generated_text = responses[0].outputs[0].text
    return prompt + generated_text


if __name__ == "__main__":
    from Medical_analysis import extract_cot_tree, system, extract_cot_chain
    
    MODEL_PATH = "/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    llm = LLM(
        model=MODEL_PATH,
        max_model_len=16384,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True
    )
    tokenizer = llm.get_tokenizer()

    data_list = []
    with open("data/Medical_analysis/truth_num=4+action_num=6+valid_truth_num=1.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))
    result = 0
    result_list = []
    for i, data in enumerate(data_list):
        prompt = 'The patient can only suffer from the following diseases: ' + str(data["truths"]) + '. \nThe results of the patient\'s examination were:\n ' + str(data["observations"]) + '.\nPlease diagnose the most likely disease among the patient\'s possible diseases based on the given information.'

        seed = data['seed']
        with open(f'data/Medical_analysis/knowledge_book/truth_num=4+action_num=6+valid_truth_num=1/seed={seed}.txt', 'r') as f:
            guide_book = f.read()
        cot_tree = extract_cot_tree(guide_book, llm, tokenizer)

        answer = cot_guide_generate(system, prompt, cot_tree)
        print(answer)
        print(i)
        try:
            ans = answer.split("</think>")[-1].split('{')[1].split('}')[0]
            ans = '{' + ans + '}'
            res = ans.replace("\n", "")
            res = json.loads(repair_json(res))
            res = res['diagnosis']
        except:
            res = input()
        label = data['valid_truth']
        print(label)
        if label in res:
            result += 1
        x = {}
        x['system'] = system
        x['input'] = prompt
        x['output'] = answer
        x['label'] = label
        result_list.append(x)
        print("Current result:\n", result/len(result_list))

    print("Result:\n", result/len(data_list))
    with open('Medical_result_td.json', 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

