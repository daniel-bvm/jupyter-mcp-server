FROM docker.io/nikolasigmoid/jupyter-mcp-base:latest

COPY jupyter_mcp_server jupyter_mcp_server
COPY pyproject.toml pyproject.toml

ENV NOTEBOOK_PATH="notebook.ipynb"
ENV NOTEBOOK_PORT="34587"
ENV HTTP_DISPLAY_URL="http://localhost:$NOTEBOOK_PORT/notebooks/$NOTEBOOK_PATH"
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install .

RUN sed -i '/"owner": self._username,/a \                "name": self._username,' /usr/local/lib/python3.12/site-packages/jupyter_nbmodel_client/client.py \
    && echo "Patched jupyter_nbmodel_client/client.py to include user name in awareness." \
    || echo "WARNING: Failed to patch jupyter_nbmodel_client/client.py"

EXPOSE 34587
RUN pip install --force-reinstall --no-cache-dir pycrdt

COPY config.json config.json
COPY system_prompt.txt system_prompt.txt