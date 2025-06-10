find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

rm -rf spark.zip
zip -r spark.zip jupyter_mcp_server config.json Dockerfile pyproject.toml system_prompt.txt