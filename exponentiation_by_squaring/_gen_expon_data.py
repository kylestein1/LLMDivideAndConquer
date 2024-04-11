"""
Dataset construction for exponentiation by squaring.
"""

import argparse
import random
import json

def exponentiation_by_squaring_recursive(base, exponent):
    if exponent == 0:
        return 1, [f"{base}^{exponent} = 1"]
    elif exponent % 2 == 0:
        sqrt_result, sqrt_logs = exponentiation_by_squaring_recursive(base, exponent // 2)
        result = sqrt_result * sqrt_result
        current_log = [f"{base}^{exponent} = ({base}^{exponent // 2})^2 = {sqrt_result}^2 = {result}"]
        return result, sqrt_logs + current_log
    else:
        sqrt_result, sqrt_logs = exponentiation_by_squaring_recursive(base, (exponent - 1) // 2)
        result = base * sqrt_result * sqrt_result
        current_log = [f"{base}^{exponent} = {base} * ({base}^{(exponent - 1) // 2})^2 = {base} * {sqrt_result}^2 = {result}"]
        return result, sqrt_logs + current_log

def exponentiation_by_squaring_scratchpad(base, exponent, level=0, side='start'):
    log_steps = []
    if exponent > 0:
        log_steps.append(f"{'  '*level}[{side}] Splitting: {base}^{exponent}")
        if exponent % 2 == 0:
            sqrt_result, log_sqrt = exponentiation_by_squaring_scratchpad(base, exponent // 2, level + 1, 'left')
            result = sqrt_result * sqrt_result
            log_steps += log_sqrt
            log_steps.append(f"{'  '*level}[{side}] Combining: {base}^{exponent} = ({base}^{exponent // 2})^2 = {sqrt_result}^2 = {result}")
        else:
            sqrt_result, log_sqrt = exponentiation_by_squaring_scratchpad(base, (exponent - 1) // 2, level + 1, 'left')
            result = base * sqrt_result * sqrt_result
            log_steps += log_sqrt
            log_steps.append(f"{'  '*level}[{side}] Combining: {base}^{exponent} = {base} * ({base}^{(exponent - 1) // 2})^2 = {base} * {sqrt_result}^2 = {result}")
    else:
        result = 1
        log_steps.append(f"{'  '*level}[{side}] Base case: {base}^{exponent} = 1")
    
    return result, log_steps

def generate_training_data(style):
    data = []
    for base in range(2, 11):
        for exponent in range(0, 16):
            for _ in range(100):
                result, recursive_logs = exponentiation_by_squaring_recursive(base, exponent)
                _, scratchpad_logs = exponentiation_by_squaring_scratchpad(base, exponent)
                if exponent == 0:
                    data.append({
                        "input": f"{base}^{exponent} = ",
                        "output": "1"
                    })
                elif style == "recursive":
                    data.append({
                        "input": f"{base}^{exponent} = ",
                        "output": f"{result}",
                        "logs": recursive_logs
                    })
                elif style == "scratchpad":
                    data.append({
                        "input": f"{base}^{exponent} = ",
                        "output": "\n".join(scratchpad_logs)
                    })
                elif style == "baseline":
                    data.append({
                        "input": f"{base}^{exponent} = ",
                        "output": f"{result}"
                    })
                else:
                    raise ValueError(f"Invalid style: {style}")

    random.shuffle(data)
    with open(f"exponentiation_by_squaring/exponentiation_train_{style}.json", "w") as f:
        json.dump(data, f, indent=4)

def generate_testing_data(split, style):
    data = {}
    for exponent in [0, 5, 10, 15, 20, 25, 30]:
        data[exponent] = []
        for _ in range(25 if split == 'test' else 5):
            base = random.randint(2, 10)
            result, recursive_logs = exponentiation_by_squaring_recursive(base, exponent)
            _, scratchpad_logs = exponentiation_by_squaring_scratchpad(base, exponent)
            if style == "recursive":
                data[exponent].append({
                    "input": f"{base}^{exponent} = ",
                    "output": f"{result}",
                    "logs": recursive_logs
                })
            elif style == "scratchpad":
                data[exponent].append({
                    "input": f"{base}^{exponent} = ",
                    "output": f"{result}",
                    "logs": scratchpad_logs
                })
            elif style == "baseline":
                data[exponent].append({
                    "input": f"{base}^{exponent} = ",
                    "output": f"{result}",
                    "logs": None
                })
            else:
                raise ValueError(f"Invalid style: {style}")

    with open(f"exponentiation_by_squaring/exponentiation_{split}_{style}.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--style", type=str)
    args = parser.parse_args()
    random.seed(42)
    if args.split == "train":
        data = generate_training_data(args.style)
    else:
        data = generate_testing_data(args.split, args.style)