from nikolasigmoid/py-mcp-proxy:latest

copy jupyter_mcp_server jupyter_mcp_server
copy pyproject.toml pyproject.toml
copy config.json config.json

copy system_prompt.txt system_prompt.txt

env NOTEBOOK_PATH="notebook.ipynb"
env NOTEBOOK_PORT="34587"
env HTTP_DISPLAY_URL="http://localhost:$NOTEBOOK_PORT/doc/tree/$NOTEBOOK_PATH"
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y libpq-dev gcc

run pip install .
run python -m pip install jupyterlab ipykernel \
    && python -m pip install "jupyter_collaboration==4.0.1"

RUN sed -i '/"owner": self._username,/a \                "name": self._username,' /usr/local/lib/python3.12/site-packages/jupyter_nbmodel_client/client.py \
    && echo "Patched jupyter_nbmodel_client/client.py to include user name in awareness." \
    || echo "WARNING: Failed to patch jupyter_nbmodel_client/client.py"

# run python -m pip uninstall -y pycrdt datalayer_pycrdt jupyter-ydoc \
# && python -m pip install datalayer_pycrdt jupyter-ydoc==v3.0.3

expose 34587
run pip install --force-reinstall --no-cache-dir pycrdt
