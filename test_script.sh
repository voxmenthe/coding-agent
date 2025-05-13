#!/bin/bash
# test_script.sh
echo "Hello from test_script.sh!"
echo "Arguments received: $@"
sleep 3 # Simulate some work
echo "Shell script finished." >&2 # Output to stderr
# exit 1 # Optionally test non-zero exit code
