"""Print a Fibonacci series up to n."""
print("Enter the no. of terms for the fibonacci sequence")
n = int(input())
a, b = 0, 1
while a < n:
    print(a, end=' ')
    a, b = b, a + b
print()