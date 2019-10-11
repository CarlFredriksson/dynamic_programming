def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

memory = { 0: 0, 1:1 }
def fib_memoization(n):
    if n not in memory:
        memory[n] = fib_memoization(n - 1) + fib_memoization(n - 2)
    return memory[n]

def fib_bottom_up(n):
    if n == 0:
        return 0

    previous_fib = 0
    current_fib = 1
    for _ in range(n - 1):
        new_fib = previous_fib + current_fib
        previous_fib = current_fib
        current_fib = new_fib
    return current_fib
