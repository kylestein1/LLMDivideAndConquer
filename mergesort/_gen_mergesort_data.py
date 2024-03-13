"""
Dataset construction for mergesort.
"""

import argparse
import random
import json

def merge_sort_recursive(arr):
    if len(arr) <= 1:
        return arr, [f"sort({arr}) = {arr}"]
    
    mid = len(arr) // 2
    left_half, left_logs = merge_sort_recursive(arr[:mid])
    right_half, right_logs = merge_sort_recursive(arr[mid:])
    
    merged_array = merge_recursive(left_half, right_half)
    current_log = [f"sort({arr}) = merge({left_half}, {right_half}) = {merged_array}"]

    return merged_array, left_logs + right_logs + current_log

def merge_recursive(left, right):
    sorted_array = []
    left_index, right_index = 0, 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            sorted_array.append(left[left_index])
            left_index += 1
        else:
            sorted_array.append(right[right_index])
            right_index += 1

    sorted_array.extend(left[left_index:])
    sorted_array.extend(right[right_index:])

    return sorted_array

def merge_sort_scratchpad(arr):
    if len(arr) <= 1:
        return arr, f"[{', '.join(map(str, arr))}]"
    
    mid = len(arr) // 2
    left_part, left_str = merge_sort_scratchpad(arr[:mid])
    right_part, right_str = merge_sort_scratchpad(arr[mid:])
    
    merged, merge_str = merge_scratchpad(left_part, right_part)
    
    output_str = f"merge(sort({left_str}), sort({right_str})) = {merge_str}"
    
    return merged, output_str

def merge_scratchpad(left, right):
    result = []
    left_index, right_index = 0, 0
    merge_str = f"merge([{', '.join(map(str, left))}], [{', '.join(map(str, right))}])"
    
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    
    result += left[left_index:]
    result += right[right_index:]
    
    return result, f"{merge_str} = [{', '.join(map(str, result))}]"


def generate_random_array(length, min_val, max_val):
    return [random.randint(min_val, max_val) for _ in range(length)]

def generate_training_data(style):
    data = []
    for i in range(1, 16):
        for j in range(5000):
            arr = generate_random_array(i, -50, 50)
            sorted_arr, recursive_logs = merge_sort_recursive(arr)
            _, scratchpad_logs = merge_sort_scratchpad(arr)
            if i == 1:
                data.append({
                    "input": f"sort({arr}) = ",
                    "output": f"{arr}"
                })
            elif style == "recursive":
                data.extend([
                    {
                    "input": f"sort({arr}) = ",
                    "output": f"merge(sort({arr[:len(arr)//2]}), sort({arr[len(arr)//2:]}))"
                    },
                    {
                    "input": f"sort({arr}) = merge({sorted(arr[:len(arr)//2])}, {sorted(arr[len(arr)//2:])}) = ",
                    "output": f"{sorted_arr}"
                    }
                ])
            elif style == "scratchpad":
                data.append({
                    "input": f"sort({arr}) = ",
                    "output": f"{scratchpad_logs}"
                
                })
            elif style == "baseline":
                data.append({
                    "input": f"sort({arr}) = ",
                    "output": f"{sorted_arr}"
                })
            else: 
                raise ValueError(f"Invalid style: {style}")

    random.shuffle(data)
    with open(f"mergesort/mergesort_train_{style}.json", "w") as f:
        json.dump(data, f, indent=4)

def generate_testing_data(split, style):
    data = {}
    for i in [1, 5, 10, 15, 20, 25, 30]:
        data[i] = []
        for j in range(25 if split == 'test' else 5):
            arr = generate_random_array(i, -50, 50)
            sorted_arr, recursive_logs = merge_sort_recursive(arr)
            _, scratchpad_logs = merge_sort_scratchpad(arr)
            if style == "recursive":
                data[i].append({
                    "input": f"sort({arr}) = ",
                    "output": f"{sorted_arr}",
                    "logs": f"{recursive_logs}"
                    })
            elif style == "scratchpad":
                data[i].append({
                    "input": f"sort({arr}) = ",
                    "output": f"{sorted_arr}",
                    "logs": f"{scratchpad_logs}"
                })
            elif style == "baseline":
                data[i].append({
                    "input": f"sort({arr}) = ",
                    "output": f"{sorted_arr}",
                    "logs": None,
                })
            else: 
                raise ValueError(f"Invalid style: {style}")
            
    with open(f"mergesort/mergesort_{split}_{style}.json", "w") as f:
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
    
