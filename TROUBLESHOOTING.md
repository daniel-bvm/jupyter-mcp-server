# Troubleshooting Guide - Extended Jupyter MCP Server

This guide lists common issues encountered during the setup and use of the Extended Jupyter MCP Server, along with their known causes and solutions based on debugging sessions (April, 2025).

## Common Issues & Solutions

* **Docker Image Won't Run:**
    * **Symptom:** Docker fails to start the container, possibly with architecture-related errors (e.g., `exec format error`).
    * **Cause:** Likely an architecture mismatch between the pre-built image (if used) and your machine (e.g., ARM vs AMD64).
    * **Fix:** Build the image locally using the project's `Dockerfile` which ensures compatibility: `docker build -t jupyter-mcp-server:latest .`

* **Connection Errors (`ReadTimeout`, `403 Forbidden`, `/api/kernels` errors):**
    * **Symptoms:** MCP client fails to connect, server logs show timeouts trying to reach Jupyter, or forbidden errors accessing Jupyter APIs.
    * **Causes & Fixes:**
        * **Firewall:** Ensure your OS/network firewall allows incoming connections to the JupyterLab port (e.g., 8888) from Docker's network.
        * **Windows QuickEdit Mode:** If running JupyterLab in Windows Terminal/Command Prompt, disable "QuickEdit Mode" (Right-click title bar -> Properties -> Options -> Untick QuickEdit Mode) as it can pause the process and cause timeouts.
        * **Token Mismatch:** Verify the `TOKEN` environment variable in your MCP client configuration (e.g., `claude_desktop_config.json`) **exactly** matches the token used with `--IdentityProvider.token` when launching `jupyter lab`.
        * **Incorrect `SERVER_URL`:** Ensure `SERVER_URL` in the client config points correctly to JupyterLab *without* the token. Use `http://host.docker.internal:8888` for Docker Desktop on Windows/Mac, or `http://localhost:8888` for Linux when using `--network=host` for the Docker container.

* **Environment/Extension Issues (`404 /api/collaboration`, `ModuleNotFoundError`, `_load_jupyter_server_extension function was not found`):**
    * **Symptoms:** Tools fail trying to use collaboration features, server logs show errors loading `jupyter_collaboration`.
    * **Causes & Fixes:**
        * **Environment Conflicts:** Strongly recommend using a dedicated `conda` environment (e.g., `jupyter_mcp_env` described in the main README setup). Avoid mixing Python installations. Ensure you `conda activate jupyter_mcp_env` before running `pip` or `jupyter lab`.
        * **Incorrect `jupyter_collaboration` Version:** Debugging showed incompatibility with v3.x/v4.x. **Fix:** Ensure you have installed exactly `jupyter_collaboration==2.0.1` using `python -m pip install "jupyter_collaboration==2.0.1"`.
        * **Extension Not Enabled:** Enable the extension within the correct environment: `jupyter server extension enable jupyter_collaboration --py --sys-prefix`.
        * **Incorrect Pip:** Always use `python -m pip install ...` within the activated conda environment to ensure packages install correctly for that environment.
        * **Permissions (Less common with Conda):** If installing packages globally requires admin rights and fails, run your terminal "As Administrator". Conda environments usually avoid this.

* **Live View Sync Issues (`KeyError: 'name'`, "Connection is already closed"):**
    * **Symptoms:** Claude interactions cause `KeyError: 'name'` in JupyterLab logs, connection seems to drop, UI sync breaks.
    * **Cause:** `jupyter_collaboration` v2.0.1 expected user awareness info (`"name"`) that the underlying `jupyter_nbmodel_client` wasn't sending.
    * **Fix:** The project's `Dockerfile` includes a patch (`RUN sed ...`) to modify the `jupyter_nbmodel_client/client.py` file *inside the Docker image* to add this field. Ensure you are using an image built with this `Dockerfile`.

* **MCP Server Hangs (Tool finishes in logs, no result sent back to client):**
    * **Symptoms:** Claude hangs waiting for a response after executing a tool that modifies the notebook.
    * **Cause:** Likely race conditions or event loop blocking when modifying notebook state via `NbModelClient` and immediately stopping the connection. Also, long-running kernel tasks could block the server if not handled carefully.
    * **Fixes:**
        * An `await asyncio.sleep(0.1)` delay was added in the server code before `notebook.stop()` in modifying tools.
        * `execute_cell` and `execute_all_cells` were changed to dispatch execution requests to a separate thread (`asyncio.to_thread`). This makes them "fire-and-forget" – they return quickly but don't wait for kernel completion. This prevents hangs but means status isn't tracked.
        * **Recommendation:** For long-running kernel tasks (e.g., model training), reduce verbosity in your notebook code (e.g., `model.fit(..., verbose=False)`) to minimize message traffic that could contribute to hangs.

* **Tool Errors (`AttributeError`, `TypeError`, Invalid Yjs Operations):**
    * **Symptoms:** Specific tools fail with errors indicating incorrect method calls or data types.
    * **Cause:** Initial versions of some tools used incorrect APIs for `pycrdt` objects (`YText`, `YArray`, `YMap`).
    * **Fix:** Tools were rewritten to use the correct `pycrdt` methods: `del text[start:end]`, `text.insert(...)`, `del array[index]`, `array.insert(...)`, etc.

* **Notebook File Corruption (`KeyError: 'cell_type'` during save):**
    * **Symptoms:** Jupyter server logs show errors when saving the notebook, indicating invalid cell structure in the Yjs document.
    * **Cause:** Issues with how cells were created or moved, sometimes not using explicit Yjs types for nested structures (`outputs`, `metadata`) or issues with Yjs object references during moves.
    * **Fix:** Cell creation (`add_cell`, `split_cell`) now explicitly instantiates `YText`, `YArray`, `YMap` for cell components. The `move_cell` operation was simplified back to `del`/`insert` which appears stable with robust cell creation.

* **Async/Event Loop Errors (`RuntimeError: no running event loop`, `Already running asyncio`):**
    * **Symptoms:** Server fails to start or crashes immediately.
    * **Cause:** Incorrectly mixing synchronous and asynchronous code managing the event loop.
    * **Fix:** The server entry point was restructured to use `async def main()` with `asyncio.run(main())` and `await mcp.run_stdio_async()`.

* **Server Crash on Exit (`AttributeError: 'shutdown'`, `stop_channels'`):**
    * **Symptoms:** Error occurs during server shutdown/cleanup.
    * **Cause:** Calling incorrect cleanup methods on the `KernelClient`.
    * **Fix:** The cleanup code now uses the correct `kernel.stop(shutdown_kernel=False)` method.

* **Blank JupyterLab UI:**
    * **Symptoms:** The MCP server seems connected and tools might partially work via the client, but the JupyterLab interface in your web browser is blank or unresponsive.
    * **Cause:** This is likely a frontend issue within JupyterLab itself, possibly an unrelated JavaScript error or a conflict with another JupyterLab extension.
    * **Fix:** Open your browser's Developer Console (usually F12) and check the "Console" tab for any red error messages when loading JupyterLab. These messages can help diagnose the frontend problem.