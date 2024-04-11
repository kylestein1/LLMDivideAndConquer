import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import ast
import re
import argparse
import json
import math
import os

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
    input_length = input_ids.shape[1]
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
    multiply_matches = re.findall(r'multiply\(\d+,\s*\d+\)', output)
    if len(multiply_matches) < 2:
        return output  # hopefully this never hits
    for match in multiply_matches[1:]:
        multiply_result = recursive_generate(model, tokenizer, f"{match} = ", max_length, max_depth - 1)
        output = output.replace(match, multiply_result.split()[-1])
    output = generate(model, tokenizer, f"{output} = ", max_length)
    cache[prompt] = output
    return output

def check_correct(pred, gold):
    return str(pred) == str(gold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--style", type=str)
    parser.add_argument("--lora-dir", type=str)
    parser.add_argument("--checkpoints", nargs='+', type=str)
    args = parser.parse_args()

    global cache

    for checkpoint in args.checkpoints:
        cache = {}

        model, tokenizer = init_model(os.path.join(args.lora_dir, checkpoint))

        with open(f"fast_mult/fastmult_{args.split}_{args.style}.json", 'r') as f:
            data = json.load(f)

        if args.style == "baseline" or args.style == "scratchpad":
            for i in data.keys():
                count = 0
                for j in range(len(data[i])):
                    pred = generate(model, tokenizer, f"{data[i][j]['input']}", 2048)
                    correct = check_correct(pred.split()[-1], data[i][j]['output'])
                    count += 1 if correct else 0
                    data[i][j]['correct'] = correct
                    data[i][j]['pred_logs'] = pred.strip()
                print(f"[DIGITS {i}] Num Correct: {count}")

        elif args.style == "recursive":
            for i in data.keys():
                count = 0
                for j in range(len(data[i])):
                    try:
                        pred = recursive_generate(model, tokenizer, f"{data[i][j]['input']}", 2048, math.ceil(math.log2(int(i))) + 1)
                    except:
                        data[i][j]['correct'] = False
                        data[i][j]['pred_logs'] = "Max depth reached"
                        continue

                    correct = check_correct(pred.split()[-1], data[i][j]['output'])
                    count += 1 if correct else 0
                    data[i][j]['correct'] = correct
                    data[i][j]['pred_logs'] = pred.strip()
                print(f"[DIGITS {i}] Num Correct: {count}")

        with open(os.path.join(args.lora_dir, checkpoint, f"fastmult_{args.split}_{args.style}_pred.json"), 'w') as f:
            json.dump(data, f, indent=4)