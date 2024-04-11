"""
Dataset construction for Fast Multiplication
"""

import argparse
import random
import json

def fast_multiply_recursive(x, y):
    if x < 10 or y < 10:
        return x * y, [f"multiply({x}, {y}) = {x * y}"]

    n = max(len(str(x)), len(str(y)))
    mid = n // 2

    a, b = divmod(x, 10**mid)
    c, d = divmod(y, 10**mid)

    ac, ac_logs = fast_multiply_recursive(a, c)
    bd, bd_logs = fast_multiply_recursive(b, d)
    ad_bc, ad_bc_logs = fast_multiply_recursive(a + b, c + d)

    result = ac * 10**(2*mid) + (ad_bc - ac - bd) * 10**mid + bd
    current_log = [f"multiply({x}, {y}) = {ac} * 10^{2*mid} + ({ad_bc} - {ac} - {bd}) * 10^{mid} + {bd} = {result}"]

    return result, ac_logs + bd_logs + ad_bc_logs + current_log

def fast_multiply_scratchpad(x, y, level=0, side='start'):
    log_steps = []
    if x < 10 or y < 10:
        log_steps.append(f"{'  '*level}[{side}] Base case: multiply({x}, {y}) = {x * y}")
        return x * y, log_steps

    n = max(len(str(x)), len(str(y)))
    mid = n // 2

    a, b = divmod(x, 10**mid)
    c, d = divmod(y, 10**mid)

    log_steps.append(f"{'  '*level}[{side}] Splitting: multiply({x}, {y})")

    ac, ac_logs = fast_multiply_scratchpad(a, c, level + 1, 'left')
    bd, bd_logs = fast_multiply_scratchpad(b, d, level + 1, 'right')
    ad_bc, ad_bc_logs = fast_multiply_scratchpad(a + b, c + d, level + 1, 'combined')

    log_steps += ac_logs + bd_logs + ad_bc_logs

    result = ac * 10**(2*mid) + (ad_bc - ac - bd) * 10**mid + bd
    log_steps.append(f"{'  '*level}[{side}] Combining: multiply({x}, {y}) = {ac} * 10^{2*mid} + ({ad_bc} - {ac} - {bd}) * 10^{mid} + {bd} = {result}")

    return result, log_steps

def generate_random_number(min_val, max_val):
    return random.randint(min_val, max_val)

def generate_training_data(style):
    data = []
    for i in range(1, 16):
        for j in range(5000):
            x = generate_random_number(10**(i-1), 10**i - 1)
            y = generate_random_number(10**(i-1), 10**i - 1)
            result, recursive_logs = fast_multiply_recursive(x, y)
            _, scratchpad_logs = fast_multiply_scratchpad(x, y)
            if style == "recursive":
                data.append({
                    "input": f"multiply({x}, {y}) = ",
                    "output": f"{result}",
                    "logs": recursive_logs
                })
            elif style == "scratchpad":
                data.append({
                    "input": f"multiply({x}, {y}) = ",
                    "output": "\n".join(scratchpad_logs)
                })
            elif style == "baseline":
                data.append({
                    "input": f"multiply({x}, {y}) = ",
                    "output": f"{result}"
                })
            else:
                raise ValueError(f"Invalid style: {style}")

    random.shuffle(data)
    with open(f"fast_mult/fastmult_train_{style}.json", "w") as f:
        json.dump(data, f, indent=4)

def generate_testing_data(split, style):
    data = {}
    for i in [1, 5, 10, 15, 20, 25, 30]:
        data[i] = []
        for j in range(25 if split == 'test' else 5):
            x = generate_random_number(10**(i-1), 10**i - 1)
            y = generate_random_number(10**(i-1), 10**i - 1)
            result, recursive_logs = fast_multiply_recursive(x, y)
            _, scratchpad_logs = fast_multiply_scratchpad(x, y)
            if style == "recursive":
                data[i].append({
                    "input": f"multiply({x}, {y}) = ",
                    "output": f"{result}",
                    "logs": recursive_logs
                })
            elif style == "scratchpad":
                data[i].append({
                    "input": f"multiply({x}, {y}) = ",
                    "output": f"{result}",
                    "logs": scratchpad_logs
                })
            elif style == "baseline":
                data[i].append({
                    "input": f"multiply({x}, {y}) = ",
                    "output": f"{result}",
                    "logs": None
                })
            else:
                raise ValueError(f"Invalid style: {style}")

    with open(f"fast_mult/fastmult_{split}_{style}.json", "w") as f:
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