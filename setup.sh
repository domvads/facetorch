#!/usr/bin/env bash
set -e

# Update pip
python -m pip install --upgrade pip

# Install library with requirements from environment.yml
pip install -e .

# Install development requirements if present
if [ -f requirements.dev.txt ]; then
    pip install -r requirements.dev.txt
fi
