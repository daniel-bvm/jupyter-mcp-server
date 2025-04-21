
# 🪐 ✨ Jupyter MCP Server

[![Github Actions Status](https://github.com/datalayer/jupyter-mcp-server/workflows/Build/badge.svg)](https://github.com/datalayer/jupyter-mcp-server/actions/workflows/build.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/jupyter-mcp-server)](https://pypi.org/project/jupyter-mcp-server)
[![smithery badge](https://smithery.ai/badge/@datalayer/jupyter-mcp-server)](https://smithery.ai/server/@datalayer/jupyter-mcp-server)

Jupyter MCP Server is a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server implementation that provides interaction with 📓 Jupyter notebooks running in any JupyterLab (works also with your 💻 local JupyterLab).

![Jupyter MCP Server](https://assets.datalayer.tech/jupyter-mcp/jupyter-mcp-server-claude-demo.gif)


## Start JupyterLab

Make sure you have the following installed. The collaboration package is needed as the modifications made on the notebook can be seen thanks to [Jupyter Real Time Collaboration](https://jupyterlab.readthedocs.io/en/stable/user/rtc.html).

```bash
python -m pip install jupyterlab jupyter-collaboration ipykernel
python -m pip uninstall -y pycrdt datalayer_pycrdt
python -m pip install datalayer_pycrdt
```

Then, start JupyterLab with the following command.

```bash
jupyter lab --port 8888 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.disable_check_xsrf=true --allow_remote_access=true
```

Now the jupyter server is already deploy in `http://localhost:8888` 