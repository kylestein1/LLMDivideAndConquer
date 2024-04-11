import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json
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

def check_correct(pred, gold):
    return str(pred) == str(gold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=['train', 'test'])
    parser.add_argument("--style", type=str, choices=['recursive', 'scratchpad', 'baseline'])
    parser.add_argument("--lora-dir", type=str)
    parser.add_argument("--checkpoints", nargs='+', type=str)
    args = parser.parse_args()

    for checkpoint in args.checkpoints:
        model, tokenizer = init_model(os.path.join(args.lora_dir, checkpoint))

        with open(f"exponentiation/exponentiation_{args.split}_{args.style}.json", 'r') as f:
            data = json.load(f)

        for exponent in data.keys():
            count = 0
            for sample in data[exponent]:
                pred = generate(model, tokenizer, sample['input'], max_length=2048)
                correct = check_correct(pred.split()[-1], sample['output'])
                count += 1 if correct else 0
                sample['correct'] = correct
                sample['pred'] = pred.strip()

            print(f"[EXPONENT {exponent}] Num Correct: {count}")

        with open(os.path.join(args.lora_dir, checkpoint, f"exponentiation_{args.split}_{args.style}_pred.json"), 'w') as f:
            json.dump(data, f, indent=4)