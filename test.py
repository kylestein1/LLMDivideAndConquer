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


def exp_by_squaring(base, exponent, level=0, side='start'):
    log_steps = []
    if exponent == 0:
        log_steps.append(f"{'  '*level}[{side}] Base case: {base}^{exponent} = 1")
        return 1, log_steps
    elif exponent % 2 == 0:
        log_steps.append(f"{'  '*level}[{side}] Even exponent: {base}^{exponent}")
        sqrt_result, log_sqrt = exp_by_squaring(base, exponent // 2, level + 1, 'left')
        result = sqrt_result * sqrt_result
        log_steps += log_sqrt
        log_steps.append(f"{'  '*level}[{side}] Combining: {base}^{exponent} = ({base}^{exponent // 2})^2 = {sqrt_result}^2 = {result}")
    else:
        log_steps.append(f"{'  '*level}[{side}] Odd exponent: {base}^{exponent}")
        sqrt_result, log_sqrt = exp_by_squaring(base, (exponent - 1) // 2, level + 1, 'left')
        result = base * sqrt_result * sqrt_result
        log_steps += log_sqrt
        log_steps.append(f"{'  '*level}[{side}] Combining: {base}^{exponent} = {base} * ({base}^{(exponent - 1) // 2})^2 = {base} * {sqrt_result}^2 = {result}")
    return result, log_steps

# Example usage
base = 2
exponent = 10
result, log_steps = exp_by_squaring(base, exponent)
print(f"Calculating {base}^{exponent}:")
for step in log_steps:
    print(step)
print(f"{base}^{exponent} = {result}")

def fast_multiply(x, y, level=0, side='start'):
    log_steps = []
    if x < 10 or y < 10:
        log_steps.append(f"{'  '*level}[{side}] Base case: multiply({x}, {y}) = {x * y}")
        return x * y, log_steps

    n = max(len(str(x)), len(str(y)))
    mid = n // 2

    a, b = divmod(x, 10**mid)
    c, d = divmod(y, 10**mid)

    log_steps.append(f"{'  '*level}[{side}] Splitting: multiply({x}, {y})")

    ac, ac_logs = fast_multiply(a, c, level + 1, 'left')
    bd, bd_logs = fast_multiply(b, d, level + 1, 'right')
    ad_bc, ad_bc_logs = fast_multiply(a + b, c + d, level + 1, 'combined')

    log_steps += ac_logs + bd_logs + ad_bc_logs

    result = ac * 10**(2*mid) + (ad_bc - ac - bd) * 10**mid + bd
    log_steps.append(f"{'  '*level}[{side}] Combining: multiply({x}, {y}) = {ac} * 10^{2*mid} + ({ad_bc} - {ac} - {bd}) * 10^{mid} + {bd} = {result}")

    return result, log_steps

# Example usage
x = 1234
y = 5678
result, log_steps = fast_multiply(x, y)
print(f"Calculating {x} * {y}:")
for step in log_steps:
    print(step)
print(f"{x} * {y} = {result}")