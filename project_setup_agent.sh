#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


# Upgrade pip and install poetry
pip install --upgrade pip
pip install poetry

# Update the lock file if necessary
poetry lock

# Install dependencies and the project
poetry install

echo "Project setup complete!"