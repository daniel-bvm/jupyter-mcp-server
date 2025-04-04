# Copyright (c) 2023-2024 Datalayer, Inc.
# Copyright (c) 2025 Alexander Isaev
# BSD 3-Clause License

FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY jupyter_mcp_server/* jupyter_mcp_server/

# Install main package, dependencies, AND Pillow
RUN pip install -e . Pillow --no-cache-dir

# Add "name" field to awareness state in jupyter_nbmodel_client to fix KeyError in jupyter_collaboration v2.0.1
RUN sed -i '/"owner": self._username,/a \                "name": self._username,' /usr/local/lib/python3.10/site-packages/jupyter_nbmodel_client/client.py \
    && echo "Patched jupyter_nbmodel_client/client.py to include user name in awareness." \
    || echo "WARNING: Failed to patch jupyter_nbmodel_client/client.py"

RUN pip uninstall -y pycrdt datalayer_pycrdt
RUN pip install datalayer_pycrdt

CMD ["python", "-m", "jupyter_mcp_server.server"]