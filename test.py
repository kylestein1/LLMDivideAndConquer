def merge_sort(arr, level=0, side='start'):
    log_steps = []
    if len(arr) > 1:
        log_steps.append(f"{'  '*level}[{side}] Splitting: {arr}")
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        # Merge sort the two halves and collect their logs
        sorted_L, log_L = merge_sort(L, level + 1, 'left')
        sorted_R, log_R = merge_sort(R, level + 1, 'right')
        
        # Combine the logs from left and right halves
        log_steps += log_L + log_R

        # Merging the sorted halves
        i = j = k = 0
        while i < len(sorted_L) and j < len(sorted_R):
            if sorted_L[i] < sorted_R[j]:
                arr[k] = sorted_L[i]
                i += 1
            else:
                arr[k] = sorted_R[j]
                j += 1
            k += 1

        while i < len(sorted_L):
            arr[k] = sorted_L[i]
            i += 1
            k += 1

        while j < len(sorted_R):
            arr[k] = sorted_R[j]
            j += 1
            k += 1
        
        log_steps.append(f"{'  '*level}[{side}] Merging: {arr}")
    else:
        log_steps.append(f"{'  '*level}[{side}] Base case: {arr}")
    
    return arr, log_steps

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr, log_steps = merge_sort(arr)
print("Sorting steps:")
for step in log_steps:
    print(step)
print("Sorted array:", sorted_arr)