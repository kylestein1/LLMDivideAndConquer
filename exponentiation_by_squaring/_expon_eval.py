import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json
import os
import re
import math

def init_model(lora_dir):
    model = AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b', cache_dir='/scratch/kyle/LLMDivideAndConquer/cache', torch_dtype=torch.float16, device_map='auto')
    model = PeftModel.from_pretrained(model, lora_dir)
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate(model, tokenizer, prompt, max_length):
    input_tok = tokenizer([prompt], add_special_tokens=False, padding=True)
    input_ids = torch.LongTensor(input_tok['input_ids']).cuda()
    attention_mask = torch.LongTensor(input_tok['attention_mask']).cuda()
    tokenized_samples = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length, 
        do_sample=False, 
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.batch_decode(tokenized_samples, skip_special_tokens=True)[0]
    return generated_text

def recursive_generate(model, tokenizer, prompt, max_length, max_depth):
    if max_depth == 0:
        raise ValueError("Max depth reached")
    if prompt in cache:
        return cache[prompt]
    output = generate(model, tokenizer, prompt, max_length)
    exp_matches = re.findall(r'\d+\^\d+', output)
    for match in exp_matches[1:]:    
        exp_result = recursive_generate(model, tokenizer, f"{match} = ", max_length, max_depth-1)
        output = output.replace(match, parse_last_num(exp_result), 1)
    output = generate(model, tokenizer, f"{output} = ", max_length)
    cache[prompt] = output
    return output

def parse_last_num(s):
    matches = re.findall(r'\d+', s)
    return matches[-1] if matches else None

def check_correct(pred, gold):
    try: 
        if int(pred) == int(gold):
            return True
    except:
        if pred == gold:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=['train', 'test'])
    parser.add_argument("--style", type=str, choices=['recursive', 'scratchpad', 'baseline'])
    parser.add_argument("--lora-dir", type=str)
    parser.add_argument("--checkpoints", nargs='+', type=str)
    args = parser.parse_args()
    
    global cache

    for checkpoint in args.checkpoints:
        model, tokenizer = init_model(os.path.join(args.lora_dir, checkpoint))
        
        cahce = {}

        with open(f"exponentiation_by_squaring/exponentiation_{args.split}_{args.style}.json", 'r') as f:
            data = json.load(f)

        if args.style == "baseline" or args.style == "scratchpad":
            for exponent in data.keys():
                count = 0
                for sample in data[exponent]:
                    pred = generate(model, tokenizer, sample['input'], max_length=2048)
                    correct = check_correct(parse_last_num(pred), sample['output'])
                    count += 1 if correct else 0
                    sample['correct'] = correct
                    sample['pred'] = " ".join(pred.split())
                print(f"[EXPONENT {exponent}] Num Correct: {count}")
                
        elif args.style == "recursive":
            for exponent in data.keys():
                count = 0
                for sample in data[exponent]:
                    try:
                        pred = recursive_generate(model, tokenizer, f"{sample['input']}", 2048, math.ceil(math.log2(int(exponent))) + 1)
                    except:
                        sample['correct'] = False
                        sample['pred'] = "Max depth reached"
                        continue
                    correct = check_correct(parse_last_num(pred), sample['output'])
                    count += 1 if correct else 0
                    sample['correct'] = correct
                    sample['pred'] = " ".join(pred.split())
                print(f"[EXPONENT {exponent}] Num Correct: {count}")
                        

        with open(os.path.join(args.lora_dir, checkpoint, f"exponentiation_{args.split}_{args.style}_pred.json"), 'w') as f:
            json.dump(data, f, indent=4)