import os
import sys

print("Current working directory:", os.getcwd())
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Python path:", sys.path)

print("Hello, debugging world!")

# Set a breakpoint here
x = 5
y = 10
z = x + y
print(f"The sum is: {z}") 