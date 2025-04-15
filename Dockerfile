from nikolasigmoid/py-mcp-proxy:latest

copy jupyter_mcp_server jupyter_mcp_server
copy pyproject.toml pyproject.toml
copy config.json config.json

run pip install .