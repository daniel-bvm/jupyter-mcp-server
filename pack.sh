#!/bin/bash

# Exit immediately on error
set -e

# Check if Git working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Git working directory is not clean. Please commit or stash changes before proceeding."
    exit 1
fi

find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

rm -rf spark.zip
zip -r spark.zip jupyter_mcp_server config.json Dockerfile pyproject.toml system_prompt.txt