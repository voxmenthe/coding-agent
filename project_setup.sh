#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


# Upgrade pip and install poetry
pip install --upgrade pip
pip install poetry

# Install dependencies without the project itself
poetry install --no-root

# Update the lock file if necessary
poetry lock

# Install dependencies and the project
poetry install

# Create and install the IPython kernel for the project
# poetry run python -m python -m ipykernel install --user --name=crv3 --display-name "Clever Routing v3"
# poetry run python -m ipykernel install --sys-prefix --name=crv3 --display-name "Clever Routing v3"
python -m ipykernel install --user --name=coda --display-name "Coding Agent" # install globally outside of poetry

echo "Jupyter kernel 'coda' has been installed."


echo "Project setup complete!"