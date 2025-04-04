# Copyright (c) 2023-2024 Datalayer, Inc.
# Copyright (c) 2025 Alexander Isaev
# BSD 3-Clause License

import re
import logging
import os
import asyncio
from typing import Any, List, Dict, Optional, AsyncGenerator # Added AsyncGenerator
import nbformat
import requests
from urllib.parse import urljoin, quote
from functools import partial
import io
from PIL import Image # If Pillow is installed
import base64
from pycrdt import Array as YArray, Map as YMap, Text as YText
from contextlib import asynccontextmanager # Added for context manager

from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import (
    NbModelClient,
    get_jupyter_notebook_websocket_url,
)
from mcp.server.fastmcp import FastMCP

# --- MCP Instance ---
mcp = FastMCP("jupyter")

# --- Configuration ---
NOTEBOOK_PATH = os.getenv("NOTEBOOK_PATH", "notebook.ipynb")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8888")
TOKEN = os.getenv("TOKEN", "MY_TOKEN")
OUTPUT_WAIT_DELAY = float(os.getenv("OUTPUT_WAIT_DELAY", "0.5"))
# Delay after modification before closing connection (allows Yjs sync)
# Could be configurable, using 0.5s as established before
MODIFY_SYNC_DELAY = 0.5

# --- Logging Setup ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Use a more detailed format for debugging if needed
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s [%(funcName)s]: %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
# Reduce log spam from underlying libraries if needed
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("jupyter_server_ydoc").setLevel(logging.WARNING)

# --- Global Kernel Client ---
# Initialized synchronously at startup in the __main__ block
kernel: Optional[KernelClient] = None

# --- Helper Functions ---

def _try_set_awareness(notebook_client: NbModelClient, tool_name: str):
    """Helper to attempt setting awareness state with logging."""
    # (Implementation remains the same as before)
    try:
        awareness = getattr(notebook_client, 'awareness', getattr(notebook_client, '_awareness', None))
        if awareness:
            user_info = {"name": notebook_client.username, "color": "#FFA500"}
            awareness.set_local_state({"user": user_info})
            logger.debug(f"Awareness state set in {tool_name}.")
        else:
            logger.warning(f"Could not find awareness attribute on NbModelClient in {tool_name}.")
    except Exception as e:
        logger.warning(f"Could not set awareness state in {tool_name}: {e}", exc_info=False)


def _parse_index_from_message(message: str) -> Optional[int]:
    """Parses the cell index from messages like 'Code cell added at index 5.'"""
    # (Implementation remains the same as before)
    if isinstance(message, str):
        match = re.search(r"index (\d+)", message)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                logger.error(f"Could not parse integer from index match: {match.group(1)}")
                return None
    logger.error(f"Could not find index pattern in message: {message}")
    return None


def extract_output(output: dict) -> str:
    """Extracts readable output from a Jupyter cell output dictionary."""
    # (Implementation remains the same as before)
    output_type = output.get("output_type")
    if output_type == "stream":
        return output.get("text", "")
    elif output_type in ["display_data", "execute_result"]:
        data = output.get("data", {})
        if "text/plain" in data:
            return data["text/plain"]
        elif "text/html" in data:
            return "[HTML Output]"
        elif "image/png" in data:
            return "[Image Output (PNG)]"
        else:
            return f"[{output_type} Data: keys={list(data.keys())}]"
    elif output_type == "error":
        # Consider returning more from traceback if needed, kept simple for now
        return f"Error: {output.get('ename', 'Unknown')}: {output.get('evalue', '')}"
    else:
        return f"[Unknown output type: {output_type}]"

# --- Async Context Manager for Notebook Connection ---

@asynccontextmanager
async def notebook_connection(tool_name: str, modify: bool = False) -> AsyncGenerator[NbModelClient, None]:
    """Provides a managed connection to the NbModelClient."""
    notebook: Optional[NbModelClient] = None
    logger.debug(f"[{tool_name}] Establishing notebook connection...")
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # Setting awareness might be desired here or within the tool
        # _try_set_awareness(notebook, tool_name) # Optional: uncomment if always needed
        await notebook.start()
        await notebook.wait_until_synced()
        logger.debug(f"[{tool_name}] Notebook connection synced.")
        yield notebook # Provide the connected client to the 'with' block
    except Exception as e:
        logger.error(f"[{tool_name}] Error during notebook connection/sync: {e}", exc_info=True)
        raise ConnectionError(f"Failed to connect/sync notebook for {tool_name}: {e}") from e
    finally:
        if notebook:
            logger.debug(f"[{tool_name}] Closing notebook connection...")
            # Add delay only if modification happened (allows Yjs sync propagation)
            if modify:
                logger.debug(f"[{tool_name}] Applying sync delay ({MODIFY_SYNC_DELAY}s) after modification.")
                await asyncio.sleep(MODIFY_SYNC_DELAY)
            try:
                # Check if the client's run task exists and is running before stopping
                # Accessing private attribute __run, consider if library offers public check
                run_task = getattr(notebook, '_NbModelClient__run', None)
                if run_task and not run_task.done():
                     await notebook.stop()
                     logger.debug(f"[{tool_name}] Notebook connection stopped.")
                else:
                     logger.debug(f"[{tool_name}] Notebook connection already stopped or not running.")
            except Exception as final_e:
                logger.error(f"[{tool_name}] Error stopping notebook connection in finally: {final_e}")

# --- Helper for Jupyter Contents API Requests ---

async def _jupyter_api_request(method: str, api_path: str, **kwargs) -> requests.Response:
    """Makes an authenticated request to the Jupyter Server API asynchronously."""
    # (Implementation remains the same as before, using global config)
    global SERVER_URL, TOKEN, logger
    base_url = SERVER_URL if SERVER_URL.endswith('/') else SERVER_URL + '/'
    quoted_api_path = "/".join(quote(part) for part in api_path.lstrip('/').split('/')) # Ensure relative path
    full_url = urljoin(base_url, f"api/contents/{quoted_api_path}")
    headers = {"Authorization": f"token {TOKEN}"}
    logger.debug(f"Making Jupyter API {method} request to: {full_url}")

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(requests.request, method, full_url, headers=headers, timeout=15, **kwargs)
        )
        response.raise_for_status()
        logger.debug(f"Jupyter API request successful (Status: {response.status_code})")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Jupyter API request failed for {method} {full_url}: {e}", exc_info=False)
        status = e.response.status_code if e.response is not None else "N/A"
        raise ConnectionError(f"API request failed (Status: {status}): {e}") from e

# --- Helper for Running Temporary Code Cells ---

async def _run_temporary_code(code_content: str, wait_seconds: float, tool_name: str) -> str:
    """Adds, executes, gets output, and deletes a temporary code cell."""
    cell_index: Optional[int] = None
    logger.info(f"[{tool_name}] Running temporary code cell...")
    try:
        # Use add_code_cell_on_bottom for reliability as previously established
        add_result = await add_code_cell_on_bottom(code_content)
        cell_index = _parse_index_from_message(add_result)
        if cell_index is None:
            raise RuntimeError(f"Failed to add temporary cell. Add result: {add_result}")
        logger.info(f"[{tool_name}] Temporary cell added at index {cell_index}.")

        exec_result = await execute_cell(cell_index)
        if "[Error" in exec_result: # Check for error string prefix
            raise RuntimeError(f"Failed to start execution for temporary cell {cell_index}. Exec result: {exec_result}")
        logger.info(f"[{tool_name}] Execution requested for cell {cell_index}. Waiting {wait_seconds}s...")

        # Wait for execution to likely complete (or timeout)
        await asyncio.sleep(wait_seconds)
        logger.info(f"[{tool_name}] Finished waiting for cell {cell_index}.")

        # Get output *without* additional wait
        cell_output = await get_cell_output(cell_index, wait_seconds=0)
        logger.info(f"[{tool_name}] Output received for cell {cell_index}.")
        return cell_output # Success

    except Exception as e:
         logger.error(f"[{tool_name}] Error during temporary code execution: {e}", exc_info=True)
         # Return a formatted error specific to this process
         return f"[Error in {tool_name}: {e}]"
    finally:
        # Ensure cleanup attempt even if errors occurred after cell creation
        if cell_index is not None:
            try:
                logger.info(f"[{tool_name}] Attempting cleanup delete for temporary cell {cell_index}.")
                delete_result = await delete_cell(cell_index)
                logger.info(f"[{tool_name}] Deletion result for cell {cell_index} in finally: {delete_result}")
            except Exception as final_del_e:
                logger.error(f"[{tool_name}] Cleanup delete failed for cell {cell_index} in finally: {final_del_e}")


# --- MCP Tools ---

@mcp.tool()
async def list_notebook_directory() -> str:
    """
    Lists files and directories in the target notebook's location.
    Directories are indicated with a trailing '/'. Helps find notebook paths.
    """
    tool_name = "list_notebook_directory"
    logger.info(f"Executing {tool_name} tool.")
    global NOTEBOOK_PATH # Need to read the current target
    try:
        current_dir = os.path.dirname(NOTEBOOK_PATH)
        dir_display_name = current_dir if current_dir else "<Jupyter Root>"
        logger.info(f"[{tool_name}] Listing contents of directory: '{dir_display_name}'")

        response = await _jupyter_api_request("GET", current_dir)
        content_data = response.json()

        if content_data.get("type") != "directory":
            return f"[Error: Path '{dir_display_name}' is not a directory]"

        items = content_data.get("content", [])
        if not items:
            return f"Directory '{dir_display_name}' is empty."

        formatted_items = []
        # Sort dirs first, then alphabetically (case-insensitive)
        key_func = lambda x: (x.get('type') != 'directory', x.get('name','').lower())
        for item in sorted(items, key=key_func):
            name = item.get("name")
            if name:
                formatted_items.append(f"{name}/" if item.get("type") == "directory" else name)

        logger.info(f"[{tool_name}] Found {len(formatted_items)} items in '{dir_display_name}'.")
        return f"Contents of '{dir_display_name}':\n- " + "\n- ".join(formatted_items)

    except ConnectionError as e:
         return f"[Error listing directory: {e}]" # Error from helper
    except Exception as e:
        logger.error(f"[{tool_name}] Unexpected error: {e}", exc_info=True)
        return f"[Unexpected Error in {tool_name}: {e}]"


@mcp.tool()
async def get_file_content(file_path: str, max_image_dim: int = 1024) -> str:
    """
    Retrieves file content. Text is returned directly. Images are resized
    if large and returned as base64 Data URI. Other binary files described.

    Args:
        file_path: Relative path to file from Jupyter root. No '..'.
        max_image_dim: Max width/height for images before resizing. Default 1024.

    Returns:
        str: Text content, image Data URI string, binary description, or error.
    """
    tool_name = "get_file_content"
    logger.info(f"Executing {tool_name} tool for path: {file_path}")

    if ".." in file_path or os.path.isabs(file_path):
        return "[Error: Invalid file path. Must be relative and not contain '..']"

    try:
        response = await _jupyter_api_request("GET", file_path)
        file_data = response.json()

        file_type = file_data.get("type")
        if file_type != "file":
            return f"[Error: Path '{file_path}' is not a file (type: {file_type})]"

        content = file_data.get("content")
        content_format = file_data.get("format")
        mimetype = file_data.get("mimetype", "")
        filename = file_data.get("name", os.path.basename(file_path))

        if content is None: return f"[Error: No content found for '{file_path}']"

        if content_format == "text":
            return content
        elif content_format == "base64":
            logger.debug(f"[{tool_name}] Processing base64 content (MIME: {mimetype}).")
            if mimetype and mimetype.startswith("image/") and Image:
                try:
                    decoded_bytes = base64.b64decode(content)
                    img_buffer = io.BytesIO(decoded_bytes)
                    img = Image.open(img_buffer)
                    original_size = img.size
                    resized = False

                    if img.width > max_image_dim or img.height > max_image_dim:
                        logger.info(f"[{tool_name}] Resizing image '{filename}' from {original_size}")
                        img.thumbnail((max_image_dim, max_image_dim))
                        resized_buffer = io.BytesIO()
                        save_format = 'PNG' if img.format != 'JPEG' else 'JPEG'
                        img.save(resized_buffer, format=save_format)
                        content = base64.b64encode(resized_buffer.getvalue()).decode('ascii')
                        if save_format == 'PNG': mimetype = 'image/png'
                        resized = True

                    data_uri = f"data:{mimetype};base64,{content}"
                    logger.info(f"[{tool_name}] Returning {'resized' if resized else 'original'} image '{filename}' as Data URI.")
                    return f"![{filename}]({data_uri})" # Return as Markdown image
                except ImportError:
                     logger.warning("Pillow library not found. Cannot resize image.")
                     # Fall through
                except Exception as img_err:
                    logger.error(f"[{tool_name}] Error processing image {filename}: {img_err}", exc_info=True)
                    return f"[Error processing image file '{filename}': {img_err}]"
            # Fallback for non-image or if Pillow fails/missing
            return f"[Binary Content (MIME: {mimetype}, base64 encoded): {content[:80]}...]"
        else:
            return f"[Unsupported file content format '{content_format}']"

    except ConnectionError as e:
         # Check for 404 specifically
         if isinstance(e.__cause__, requests.exceptions.HTTPError) and e.__cause__.response.status_code == 404:
             return f"[Error: File not found at path: '{file_path}']"
         else:
             return f"[Error retrieving file: {e}]"
    except Exception as e:
        logger.error(f"[{tool_name}] Unexpected error for '{file_path}': {e}", exc_info=True)
        return f"[Unexpected Error in {tool_name} for '{file_path}': {e}]"


@mcp.tool()
def set_target_notebook(new_notebook_path: str) -> str:
    """
    Changes the target notebook path for subsequent tool calls (session only).
    Path must be relative to Jupyter root and contain no '..'.

    Args:
        new_notebook_path: The new relative path (e.g., "subdir/notebook.ipynb").
    """
    tool_name = "set_target_notebook"
    global NOTEBOOK_PATH # Declare intent to modify the global variable
    old_path = NOTEBOOK_PATH
    logger.info(f"Executing {tool_name}. Current: '{old_path}', New: '{new_notebook_path}'")

    if os.path.isabs(new_notebook_path) or ".." in new_notebook_path:
        logger.error(f"[{tool_name}] Invalid path: '{new_notebook_path}'.")
        return f"[Error: Invalid path '{new_notebook_path}'. Must be relative, no '..'.]"

    NOTEBOOK_PATH = new_notebook_path
    logger.info(f"[{tool_name}] Target notebook path changed to: '{NOTEBOOK_PATH}'")
    return f"Target notebook path set to '{NOTEBOOK_PATH}'."


@mcp.tool()
async def add_cell(content: str, cell_type: str, index: Optional[int] = None) -> str:
    """
    Adds a cell ('code' or 'markdown') with content at specified index (or appends).
    Uses robust Yjs type creation including YMap for metadata.

    Args:
        content: Source content for the new cell.
        cell_type: Must be 'code' or 'markdown'.
        index: 0-based index to insert at; appends if None or invalid.
    """
    tool_name = "add_cell"
    logger.info(f"Executing {tool_name}. Type: {cell_type}, Index: {index}")

    if cell_type not in ["code", "markdown"]:
        return f"[Error: Invalid cell_type '{cell_type}'. Must be 'code' or 'markdown'.]"

    try:
        async with notebook_connection(tool_name, modify=True) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            insert_index = num_cells if index is None or not (0 <= index <= num_cells) else index
            log_index_msg = f"Appending at index {insert_index}" if index is None or index >= num_cells else f"Inserting at index {insert_index}"
            logger.info(f"[{tool_name}] {log_index_msg}.")

            # Prepare cell dict with explicit Yjs types
            new_cell_pre_ymap: Dict[str, Any] = {
                "cell_type": cell_type,
                "source": YText(content)
            }

            if cell_type == "code":
                base = nbformat.v4.new_code_cell(source="")
                new_cell_pre_ymap["metadata"] = YMap(base.metadata)
                new_cell_pre_ymap["outputs"] = YArray()
                new_cell_pre_ymap["execution_count"] = None
            else: # markdown
                base = nbformat.v4.new_markdown_cell(source="")
                new_cell_pre_ymap["metadata"] = YMap(base.metadata)

            # Perform insertion in transaction
            with ydoc.ydoc.transaction():
                ycell_map = YMap(new_cell_pre_ymap)
                ycells.insert(insert_index, ycell_map)

            logger.info(f"[{tool_name}] Successfully inserted {cell_type} cell at index {insert_index}.")
            return f"{cell_type.capitalize()} cell added at index {insert_index}."

    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed: {e}", exc_info=True)
        return f"[Error adding {cell_type} cell: {e}]"


@mcp.tool()
async def add_code_cell_on_bottom(cell_content: str) -> str:
    """
    (DEPRECATED - Use add_cell) Adds a code cell at the end of the notebook.
    Kept for compatibility or if add_cell proves unstable.

    Args:
        cell_content: Code content for the new cell.
    """
    # This now uses the library directly, which was deemed more stable previously
    # If add_cell with manual YMap creation is stable, this could be removed.
    tool_name = "add_code_cell_on_bottom"
    logger.info(f"Executing {tool_name}.")
    try:
        # Use context manager even though we call library method
        async with notebook_connection(tool_name, modify=True) as notebook:
             _try_set_awareness(notebook, tool_name) # Set awareness if needed
             # Use the library's potentially more reliable method
             cell_index = notebook.add_code_cell(cell_content)
             logger.info(f"[{tool_name}] Added code cell at index {cell_index} using library method.")
             return f"Code cell added at index {cell_index}."
    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed: {e}", exc_info=True)
        return f"[Error adding code cell on bottom: {e}]"


@mcp.tool()
async def execute_cell(cell_index: int) -> str:
    """
    Sends request to execute a specific code cell by index (fire-and-forget).
    Uses asyncio.to_thread to avoid blocking. Does NOT wait for completion.

    Args:
        cell_index: The 0-based index of the code cell to execute.
    """
    tool_name = "execute_cell"
    global kernel # Needs the kernel client
    logger.info(f"Executing {tool_name} for cell index {cell_index}")

    # Ensure kernel is available and alive, attempt restart if not
    if not kernel or not kernel.is_alive():
        logger.warning(f"[{tool_name}] Kernel client not alive. Attempting restart...")
        try:
            # Reinitialize and start the global kernel object
            kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
            kernel.start() # This is synchronous in the original library
            await asyncio.sleep(0.5) # Give it a moment to establish connection maybe
            if not kernel.is_alive(): raise RuntimeError("Kernel restart failed")
            logger.info(f"[{tool_name}] Kernel client restarted.")
        except Exception as kernel_err:
            logger.error(f"[{tool_name}] Failed to restart kernel client: {kernel_err}", exc_info=True)
            return f"[Error: Kernel client connection failed for cell {cell_index}.]"

    try:
        # Need a brief notebook connection to validate index/type and dispatch execution
        # No modification needed for dispatch itself, but underlying state might change later
        async with notebook_connection(tool_name, modify=False) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            if not (0 <= cell_index < num_cells):
                return f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"
            if ycells[cell_index].get("cell_type") != "code":
                 return f"[Error: Cell index {cell_index} is not a code cell]"

            # --- Dispatch execution to thread ---
            try:
                logger.info(f"[{tool_name}] Dispatching execution request for cell {cell_index} to thread...")
                # Assuming notebook.execute_cell is synchronous or handles its own async internally
                await asyncio.to_thread(notebook.execute_cell, cell_index, kernel)
                logger.info(f"[{tool_name}] Execution request dispatched for cell {cell_index}.")
                return f"Execution request sent for cell at index {cell_index}."
            except Exception as exec_dispatch_err:
                logger.error(f"[{tool_name}] Error dispatching execution request: {exec_dispatch_err}", exc_info=True)
                return f"[Error dispatching execution for cell {cell_index}: {exec_dispatch_err}]"
            # --- End dispatch ---
        # Context manager handles notebook.stop() after dispatching
    except Exception as e:
        logger.error(f"[{tool_name}] Error during setup/validation: {e}", exc_info=True)
        return f"[Error preparing execution for cell {cell_index}: {e}]"


@mcp.tool()
async def execute_all_cells() -> str:
    """
    Sends execution requests sequentially for all code cells (fire-and-forget).
    Does NOT wait for completion.
    """
    tool_name = "execute_all_cells"
    global kernel # Needs kernel client
    logger.info(f"Executing {tool_name} tool.")

    # Ensure kernel is available and alive, attempt restart if not
    # (Duplicated logic from execute_cell - could be refactored)
    if not kernel or not kernel.is_alive():
        logger.warning(f"[{tool_name}] Kernel client not alive. Attempting restart...")
        try:
            kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
            kernel.start()
            await asyncio.sleep(0.5)
            if not kernel.is_alive(): raise RuntimeError("Kernel restart failed")
            logger.info(f"[{tool_name}] Kernel client restarted.")
        except Exception as kernel_err:
            logger.error(f"[{tool_name}] Failed to restart kernel client: {kernel_err}", exc_info=True)
            return f"[Error: Kernel client connection failed. Cannot execute cells.]"

    total_code_cells = 0
    cells_requested = 0
    request_error = None

    try:
        async with notebook_connection(tool_name, modify=False) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)
            logger.info(f"[{tool_name}] Found {num_cells} cells. Iterating...")

            for i, cell_data in enumerate(ycells):
                if cell_data.get("cell_type") == "code":
                    total_code_cells += 1
                    try:
                        logger.info(f"[{tool_name}] Sending execution request for code cell {i}/{num_cells-1}...")
                        # Directly call execute_cell (which now uses to_thread)
                        # We don't need to wait here, just dispatch sequentially
                        # NOTE: This might rapidly queue many thread tasks
                        await execute_cell(i) # Reuse the single cell execution logic
                        cells_requested += 1
                        # Add a tiny sleep to avoid overwhelming the thread pool instantly?
                        await asyncio.sleep(0.05)
                    except Exception as send_err:
                        logger.error(f"[{tool_name}] Error occurred while requesting execution for cell {i}: {send_err}", exc_info=True)
                        request_error = f"Error requesting execution for cell {i}: {send_err}"
                        break # Stop processing further cells on error

        # Format result string after loop/context exit
        if request_error:
            return f"Stopped sending requests due to error: {request_error}. Sent requests for {cells_requested}/{total_code_cells} code cells."
        else:
            return f"Successfully sent execution requests for all {total_code_cells} code cells found."

    except Exception as e:
        logger.error(f"[{tool_name}] Error during setup or iteration: {e}", exc_info=True)
        return f"[Error in {tool_name}: {e}]"


@mcp.tool()
async def get_cell_output(cell_index: int, wait_seconds: float = OUTPUT_WAIT_DELAY) -> str:
    """
    Retrieves output of a code cell by index. Waits briefly if requested.

    Args:
        cell_index: Index of the cell to get output from.
        wait_seconds: Time to wait before reading output. Default from config.
    """
    tool_name = "get_cell_output"
    logger.info(f"Executing {tool_name} for cell index {cell_index}, waiting {wait_seconds}s")

    # Perform wait outside the connection context if needed
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    try:
        # Read-only operation
        async with notebook_connection(tool_name, modify=False) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            if not (0 <= cell_index < num_cells):
                 return f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"

            cell_data = ycells[cell_index]
            outputs = cell_data.get("outputs", [])

            if outputs:
                # Convert Yjs outputs (usually maps) to Python dicts for processing
                output_texts = [extract_output(dict(output)) for output in outputs]
                cell_output_str = "\n".join(output_texts).strip()
                return cell_output_str if cell_output_str else "[Cell output is empty]"
            else:
                # Check if it's a code cell that might have run without output
                if cell_data.get("cell_type") == "code":
                    exec_count = cell_data.get("execution_count")
                    if exec_count is not None:
                        return "[Cell executed, but has no output]"
                    else:
                        return "[Code cell not executed or has no output]"
                else: # Not a code cell
                    return "[Cannot get output from non-code cell]"

    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed for index {cell_index}: {e}", exc_info=True)
        return f"[Error getting output for cell {cell_index}: {e}]"


@mcp.tool()
async def delete_cell(cell_index: int) -> str:
    """Deletes a specific cell by its index."""
    tool_name = "delete_cell"
    logger.info(f"Executing {tool_name} for cell index {cell_index}")
    try:
        async with notebook_connection(tool_name, modify=True) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            if not (0 <= cell_index < num_cells):
                 return f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"

            logger.info(f"[{tool_name}] Attempting to delete cell at index {cell_index}.")
            with ydoc.ydoc.transaction():
                del ycells[cell_index]

            logger.info(f"[{tool_name}] Successfully submitted deletion for cell {cell_index}.")
            return f"Cell at index {cell_index} deleted."

    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed for index {cell_index}: {e}", exc_info=True)
        return f"[Error deleting cell {cell_index}: {e}]"


@mcp.tool()
async def move_cell(from_index: int, to_index: int) -> str:
    """
    Moves a cell from from_index to to_index using simple delete/re-insert.
    """
    # Removed the duplicate definition. This is the simpler version.
    tool_name = "move_cell"
    logger.info(f"Executing {tool_name} from {from_index} to {to_index}")
    try:
        async with notebook_connection(tool_name, modify=True) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            # Validation
            if not (0 <= from_index < num_cells):
                return f"[Error: from_index {from_index} out of bounds (0-{num_cells-1})]"
            # Allow inserting at the very end (index num_cells)
            if not (0 <= to_index <= num_cells):
                return f"[Error: to_index {to_index} out of bounds (0-{num_cells})]"
            if from_index == to_index:
                return f"Cell is already at index {from_index}."

            # Perform move within transaction
            with ydoc.ydoc.transaction():
                item_to_move = ycells[from_index] # Get reference
                del ycells[from_index]            # Delete reference
                ycells.insert(to_index, item_to_move) # Insert reference

            logger.info(f"[{tool_name}] Cell moved from {from_index} to {to_index}.")
            return f"Cell moved from index {from_index} to {to_index}."

    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed: {e}", exc_info=True)
        return f"[Error moving cell from {from_index} to {to_index}: {e}]"


@mcp.tool()
async def search_notebook_cells(search_string: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Searches cell sources for a string. Returns list of matching cells.

    Args:
        search_string: String to search for.
        case_sensitive: Perform case-sensitive search (default: False).

    Returns:
        List of dicts [{'index', 'cell_type', 'source'}] or empty list on error/no match.
    """
    tool_name = "search_notebook_cells"
    logger.info(f"Executing {tool_name} for: '{search_string}' (case_sensitive={case_sensitive})")
    matches = []
    try:
        # Use get_all_cells helper tool
        all_cells_result = await get_all_cells()

        # Check if the helper returned an error structure
        if isinstance(all_cells_result, list) and all_cells_result and "error" in all_cells_result[0]:
             logger.error(f"[{tool_name}] Failed to retrieve cells: {all_cells_result[0]['error']}")
             return [] # Return empty on error

        if not isinstance(all_cells_result, list):
             logger.error(f"[{tool_name}] Unexpected response type from get_all_cells: {type(all_cells_result)}")
             return []

        # Perform search
        search_term = search_string if case_sensitive else search_string.lower()
        for cell in all_cells_result:
            source = cell.get("source", "") # Default to empty string if missing
            if isinstance(source, str): # Ensure source is string
                source_to_search = source if case_sensitive else source.lower()
                if search_term in source_to_search:
                    matches.append({
                        "index": cell.get("index"),
                        "cell_type": cell.get("cell_type"),
                        "source": source # Return original source
                    })
            else:
                 logger.warning(f"[{tool_name}] Cell index {cell.get('index')} has non-string source. Skipping.")


        logger.info(f"[{tool_name}] Search complete. Found {len(matches)} matches.")
        return matches

    except Exception as e:
        logger.error(f"[{tool_name}] Unexpected error: {e}", exc_info=True)
        return [] # Return empty list on unexpected errors


@mcp.tool()
async def split_cell(cell_index: int, line_number: int) -> str:
    """
    Splits a cell at a specific line number (1-based).

    Args:
        cell_index: Index of the cell to split.
        line_number: 1-based line number to split at.
    """
    tool_name = "split_cell"
    logger.info(f"Executing {tool_name} for index {cell_index} at line {line_number}")

    if line_number < 1: return "[Error: line_number must be 1 or greater]"

    try:
        async with notebook_connection(tool_name, modify=True) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            # Validation
            if not (0 <= cell_index < num_cells):
                return f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"

            cell_data_read = ycells[cell_index]
            original_source = str(cell_data_read.get("source", "")) # Read as string
            original_cell_type = cell_data_read.get("cell_type", "code")
            source_lines = original_source.splitlines(True) # Keep line endings

            # Allow splitting *after* last line (creates empty second cell)
            if not (1 <= line_number <= len(source_lines) + 1):
                return f"[Error: Line number {line_number} out of range (1-{len(source_lines) + 1})]"

            # Calculate source parts
            source_part1 = "".join(source_lines[:line_number-1])
            source_part2 = "".join(source_lines[line_number-1:])
            new_cell_index = cell_index + 1

            # Perform updates within transaction
            with ydoc.ydoc.transaction():
                # 1. Update original cell
                cell_data_write = ycells[cell_index]
                source_obj = cell_data_write.get("source")
                if isinstance(source_obj, YText):
                    # Replace content efficiently
                    del source_obj[0 : len(str(source_obj))]
                    source_obj.insert(0, source_part1)
                else:
                    cell_data_write["source"] = YText(source_part1) # Create if missing/wrong type

                # Clear outputs/count for original cell if it was code
                if original_cell_type == "code":
                    outputs_obj = cell_data_write.get("outputs")
                    if isinstance(outputs_obj, YArray): outputs_obj.clear()
                    else: cell_data_write["outputs"] = YArray()
                    cell_data_write["execution_count"] = None

                # 2. Create and insert the new cell
                new_cell_pre_ymap: Dict[str, Any] = {
                    "cell_type": original_cell_type,
                    "source": YText(source_part2)
                }
                if original_cell_type == "code":
                    base = nbformat.v4.new_code_cell(source="")
                    new_cell_pre_ymap["metadata"] = YMap(base.metadata)
                    new_cell_pre_ymap["outputs"] = YArray()
                    new_cell_pre_ymap["execution_count"] = None
                else: # markdown
                    base = nbformat.v4.new_markdown_cell(source="")
                    new_cell_pre_ymap["metadata"] = YMap(base.metadata)

                ycells.insert(new_cell_index, YMap(new_cell_pre_ymap))

            logger.info(f"[{tool_name}] Cell {cell_index} split. New cell created at index {new_cell_index}.")
            return f"Cell {cell_index} split at line {line_number}. New cell created at index {new_cell_index}."

    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed for index {cell_index}: {e}", exc_info=True)
        return f"[Error splitting cell {cell_index}: {e}]"


@mcp.tool()
async def get_all_cells() -> list[dict[str, Any]]:
    """Retrieves basic info for all cells (index, type, source, exec_count)."""
    tool_name = "get_all_cells"
    logger.info(f"Executing {tool_name} tool.")
    all_cells_data = []
    try:
        async with notebook_connection(tool_name, modify=False) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            logger.info(f"[{tool_name}] Processing {len(ycells)} cells.")
            for i, cell_data_y in enumerate(ycells):
                 cell_info = {
                    "index": i,
                    "cell_type": cell_data_y.get("cell_type"),
                    "source": str(cell_data_y.get("source", "")), # Ensure string conversion
                    "execution_count": None # Default
                 }
                 # Add execution_count only if it's a code cell
                 if cell_info["cell_type"] == "code":
                      cell_info["execution_count"] = cell_data_y.get("execution_count")
                 all_cells_data.append(cell_info)
        return all_cells_data
    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed: {e}", exc_info=True)
        # Return error indicator within the list structure if preferred, or just empty
        return [{"error": f"Error during cell retrieval: {e}"}]


@mcp.tool()
async def edit_cell_source(cell_index: int, new_content: str) -> str:
    """Edits the source content of a specific cell by its index."""
    tool_name = "edit_cell_source"
    logger.info(f"Executing {tool_name} for cell index {cell_index}")
    try:
        async with notebook_connection(tool_name, modify=True) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            num_cells = len(ycells)

            if not (0 <= cell_index < num_cells):
                return f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"

            cell_data = ycells[cell_index]
            with ydoc.ydoc.transaction():
                source_obj = cell_data.get("source")
                if isinstance(source_obj, YText):
                    # Efficiently replace content
                    del source_obj[0 : len(str(source_obj))]
                    source_obj.insert(0, new_content)
                else: # If source missing or not YText, create/replace it
                    cell_data["source"] = YText(new_content)

            logger.info(f"[{tool_name}] Updated source for cell index {cell_index}.")
            return f"Source updated for cell at index {cell_index}."

    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed for index {cell_index}: {e}", exc_info=True)
        return f"[Error editing cell {cell_index}: {e}]"


@mcp.tool()
async def get_kernel_variables(wait_seconds: int = 2) -> str:
    """
    Lists variables in the kernel namespace using %whos via a temporary cell.

    Args:
        wait_seconds: Seconds to wait for execution. Default 2.
    """
    tool_name = "get_kernel_variables"
    logger.info(f"Executing {tool_name}")
    code_content = "%whos"
    # Use the helper function for temporary code execution
    return await _run_temporary_code(code_content, wait_seconds, tool_name)


@mcp.tool()
async def get_all_outputs() -> dict[int, str]:
    """
    Retrieves output strings for all code cells.

    Returns:
        dict: Mapping cell index to its combined output string or status.
    """
    tool_name = "get_all_outputs"
    logger.info(f"Executing {tool_name} tool.")
    all_outputs_map = {}
    try:
        async with notebook_connection(tool_name, modify=False) as notebook:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            logger.info(f"[{tool_name}] Processing {len(ycells)} cells for outputs.")
            for i, cell_data in enumerate(ycells):
                if cell_data.get("cell_type") == "code":
                    outputs = cell_data.get("outputs", [])
                    if outputs:
                        output_texts = [extract_output(dict(output)) for output in outputs]
                        output_str = "\n".join(output_texts).strip()
                        all_outputs_map[i] = output_str if output_str else "[No output]"
                    else:
                        exec_count = cell_data.get("execution_count")
                        all_outputs_map[i] = "[No output]" if exec_count is not None else "[Not executed]"
            # Implicitly skips non-code cells
        logger.info(f"[{tool_name}] Found outputs/status for {len(all_outputs_map)} code cells.")
        return all_outputs_map
    except Exception as e:
        logger.error(f"[{tool_name}] Tool execution failed: {e}", exc_info=True)
        return {-1: f"[Error retrieving outputs: {e}]"} # Return error indicator


@mcp.tool()
async def install_package(package_name: str, timeout_seconds: int = 60) -> str:
    """
    Installs a package in the kernel using '!pip install' via a temporary cell.

    Args:
        package_name: Name of the package to install.
        timeout_seconds: Time to wait for pip command. Default 60.
    """
    tool_name = "install_package"
    logger.info(f"Executing {tool_name} for package: {package_name}")
    # Basic sanitization
    safe_package_name = "".join(c for c in package_name if c.isalnum() or c in '-_==.')
    if not safe_package_name or safe_package_name != package_name:
        return f"[Error: Invalid package name '{package_name}']"

    code_content = f"!python -m pip install {safe_package_name}"
    # Use the helper function for temporary code execution
    return await _run_temporary_code(code_content, timeout_seconds, tool_name)


@mcp.tool()
async def list_installed_packages(wait_seconds: int = 5) -> str:
    """
    Lists installed packages in the kernel using '!pip list' via a temporary cell.

    Args:
        wait_seconds: Time to wait for pip command. Default 5.
    """
    tool_name = "list_installed_packages"
    logger.info(f"Executing {tool_name}")
    code_content = "!python -m pip list"
    # Use the helper function for temporary code execution
    return await _run_temporary_code(code_content, wait_seconds, tool_name)


# --- Main Application Logic ---

async def main():
    """Main async function to run the MCP server and handle kernel cleanup."""
    global kernel, logger, mcp # Ensure access to globals

    logger.info(f"Starting Jupyter MCP Server main async function...")
    logger.info(f"Target notebook: {NOTEBOOK_PATH} on {SERVER_URL} [Log Level: {LOG_LEVEL}]")

    # Kernel should be initialized in the __main__ block before calling main()
    if not kernel or not kernel.is_alive():
         logger.error("Kernel not initialized or not alive at start of main(). Exiting.")
         return # Cannot run without a kernel

    try:
        logger.info("Starting MCP server async stdio run...")
        await mcp.run_stdio_async()
        logger.info("MCP server async stdio run finished.")
    except Exception as e: # General handler
        logger.error(f"Caught general exception in main run: {e}", exc_info=True)
    finally:
        logger.info("Starting cleanup...")
        if kernel and kernel.is_alive():
            logger.info("Stopping kernel client connection (leaving kernel process running)...")
            try:
                # Use the correct method to stop client channels without killing kernel
                kernel.stop(shutdown_kernel=False)
                logger.info("Kernel client connection stopped.")
            except AttributeError:
                logger.error("AttributeError during kernel stop. kernel.stop() method missing or changed?", exc_info=True)
            except Exception as stop_e:
                logger.error(f"Error stopping kernel client connection: {stop_e}", exc_info=True)
        else:
            logger.info("Kernel client already stopped or not initialized.")
        logger.info("Cleanup finished.")

# --- Entry point ---
if __name__ == "__main__":
    # Perform synchronous setup here, like kernel initialization

    logger.info(f"Initializing KernelClient synchronously for {SERVER_URL}...")
    try:
        kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
        # Start is synchronous in the library version used previously
        kernel.start()
        # Add a small delay/check to ensure it's alive? Library might block until ready.
        # Let's assume start() blocks or is fast enough.
        if kernel.is_alive():
             logger.info("KernelClient started successfully synchronously.")
        else:
             logger.error("KernelClient failed to start synchronously.")
             kernel = None # Ensure kernel is None if start fails
    except Exception as e:
        logger.error(f"Failed to initialize KernelClient at startup: {e}", exc_info=True)
        kernel = None

    # Only proceed if kernel started successfully
    if kernel:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Server stopped by user (KeyboardInterrupt).")
        except Exception as e:
            logger.error(f"Unhandled exception at top level: {e}", exc_info=True)
    else:
        logger.critical("Could not initialize Jupyter Kernel client. Server cannot start.")