find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

rm -rf vibe_ds.zip
zip -r vibe_ds.zip jupyter_mcp_server config.json Dockerfile pyproject.toml system_prompt.txt