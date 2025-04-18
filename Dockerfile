from nikolasigmoid/py-mcp-proxy:latest

copy jupyter_mcp_server jupyter_mcp_server
copy pyproject.toml pyproject.toml
copy config.json config.json

env NOTEBOOK_PATH="notebook.ipynb"
env NOTEBOOK_PORT="34587"
env HTTP_DISPLAY_URL="http://localhost:$NOTEBOOK_PORT/lab/tree/$NOTEBOOK_PATH"

run pip install .

run python -m pip install jupyterlab jupyter-collaboration ipykernel
run python -m pip uninstall -y pycrdt datalayer_pycrdt
run python -m pip install datalayer_pycrdt