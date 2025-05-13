import sys
import time
print("Hello from test_script.py!")
print(f"Arguments received: {sys.argv[1:]}")
time.sleep(3) # Simulate some work
print("Script finished.")
sys.stderr.write("This is a test error message to stderr.\\n")
# sys.exit(1) # Optionally test non-zero exit code
