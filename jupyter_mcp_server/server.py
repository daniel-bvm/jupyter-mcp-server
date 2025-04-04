# Copyright (c) 2023-2024 Datalayer, Inc.
# BSD 3-Clause License
import re
import traceback # For more detailed error logging if needed
import logging
import os
import asyncio # Make sure asyncio is imported
from typing import Any, List, Dict, Optional # Import necessary types
import nbformat
import requests
import json
from urllib.parse import urljoin, quote
from functools import partial
import io
from PIL import Image # If Pillow is installed
import base64
import uuid
import mimetypes
from pathlib import Path
from pycrdt import Text as YText, Map as YMap # Import YText and YMap

from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import (
    NbModelClient,
    get_jupyter_notebook_websocket_url,
)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("jupyter")

# --- Configuration ---
NOTEBOOK_PATH = os.getenv("NOTEBOOK_PATH", "notebook.ipynb")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8888")
TOKEN = os.getenv("TOKEN", "MY_TOKEN")
OUTPUT_WAIT_DELAY = float(os.getenv("OUTPUT_WAIT_DELAY", "0.5")) # Delay for get_cell_output

# --- Logging Setup ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
# Reduce log spam from underlying libraries if needed
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("jupyter_server_ydoc").setLevel(logging.WARNING)

# --- Global Kernel Client ---
# Initialize once at startup
try:
    logger.info(f"Initializing KernelClient for {SERVER_URL}...")
    kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
    kernel.start() # Ensure connection is attempted at start
    logger.info("KernelClient started.")
except Exception as e:
    logger.error(f"Failed to initialize KernelClient at startup: {e}", exc_info=True)
    # Depending on severity, you might want to exit or handle this
    kernel = None # Ensure kernel is None if start fails



# --- Helper Functions ---
def _try_set_awareness(notebook_client: NbModelClient, tool_name: str):
    """Helper to attempt setting awareness state with logging."""
    try:
        # Attempt to access awareness object - might need adjustment based on library version
        awareness = getattr(notebook_client, 'awareness', getattr(notebook_client, '_awareness', None))
        if awareness:
            # Use username property from NbModelClient (__init__ sets self._username)
            user_info = {"name": notebook_client.username, "color": "#FFA500"}
            awareness.set_local_state({"user": user_info})
            logger.debug(f"Awareness state set in {tool_name}.")
        else:
             logger.warning(f"Could not find awareness attribute on NbModelClient in {tool_name}.")
    except Exception as e:
        logger.warning(f"Could not set awareness state in {tool_name}: {e}", exc_info=False)

# helper function to parse cell index from messages
def _parse_index_from_message(message: str) -> int | None:
    """Parses the cell index from messages like 'Code cell added at index 5.'"""
    if isinstance(message, str): # Ensure message is a string
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
    output_type = output.get("output_type")
    if output_type == "stream":
        return output.get("text", "")
    elif output_type in ["display_data", "execute_result"]:
        data = output.get("data", {})
        if "text/plain" in data:
            return data["text/plain"]
        elif "text/html" in data:
            return "[HTML Output]" # Keep it simple
        elif "image/png" in data:
            return "[Image Output (PNG)]"
        else:
            # Return a simple string for unknown output types
            return f"[{output_type} Data: keys={list(data.keys())}]"
    elif output_type == "error":
        return f"Error: {output.get('ename', 'Unknown')}: {output.get('evalue', '')}"
    else:
        return f"[Unknown output type: {output_type}]"
    
# --- Helper for Jupyter API Requests ---
async def _jupyter_api_request(method: str, api_path: str, **kwargs) -> requests.Response:
    """Makes an authenticated request to the Jupyter Server API asynchronously."""
    global SERVER_URL, TOKEN, logger # Access globals
    # Ensure SERVER_URL ends with a slash for urljoin
    base_url = SERVER_URL if SERVER_URL.endswith('/') else SERVER_URL + '/'
    # Safely join and quote the API path component (leaving '/' separators)
    # Quote avoids issues with spaces or special chars in path parts
    quoted_api_path = "/".join(quote(part) for part in api_path.split('/'))
    full_url = urljoin(base_url, f"api/contents/{quoted_api_path}")
    headers = {"Authorization": f"token {TOKEN}"}
    logger.debug(f"Making Jupyter API {method} request to: {full_url}")

    try:
        loop = asyncio.get_event_loop()
        # Run the synchronous requests call in a thread pool executor
        response = await loop.run_in_executor(
            None, # Use default executor
            partial(requests.request, method, full_url, headers=headers, timeout=15, **kwargs) # Increased timeout slightly
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.debug(f"Jupyter API request successful (Status: {response.status_code})")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Jupyter API request failed for {method} {full_url}: {e}", exc_info=False) # Log less verbose error
        # Re-raise a more specific error maybe, or let the tool handle it
        raise ConnectionError(f"API request failed: {e}") from e

# --- MCP Tools ---

@mcp.tool()
async def list_notebook_directory() -> str:
    """
    Lists files and directories in the same location as the current target notebook,
    relative to the Jupyter Lab server's starting directory.
    Helps find correct notebook names/paths.
    Directories are indicated with a trailing '/'.
    """
    logger.info("Executing list_notebook_directory tool.")
    global NOTEBOOK_PATH # Need to read the current target
    try:
        # Get the directory part of the current notebook path. '' means root.
        current_dir = os.path.dirname(NOTEBOOK_PATH)
        dir_display_name = current_dir if current_dir else "<Jupyter Root>"
        logger.info(f"Listing contents of directory: '{dir_display_name}'")

        response = await _jupyter_api_request("GET", current_dir)
        content_data = response.json() # Parse JSON response

        if content_data.get("type") != "directory":
            logger.error(f"API response for '{current_dir}' was not type 'directory'.")
            return f"[Error: Path '{dir_display_name}' is not a directory on the server]"

        items = content_data.get("content", [])
        if not items:
            return f"Directory '{dir_display_name}' is empty."

        # Format the output list
        formatted_items = []
        for item in sorted(items, key=lambda x: (x.get('type') != 'directory', x.get('name','').lower())): # Sort dirs first, then alphabetically
            name = item.get("name")
            item_type = item.get("type")
            if name:
                if item_type == "directory":
                    formatted_items.append(f"{name}/") # Add trailing slash to dirs
                else:
                    formatted_items.append(name)

        logger.info(f"Found {len(formatted_items)} items in '{dir_display_name}'.")
        return f"Contents of '{dir_display_name}':\n- " + "\n- ".join(formatted_items)

    except ConnectionError as e: # Catch errors from the helper
         return f"[Error listing directory: {e}]"
    except Exception as e:
        logger.error(f"Unexpected error in list_notebook_directory: {e}", exc_info=True)
        return f"[Unexpected Error listing directory: {e}]"
    
@mcp.tool()
async def get_file_content(file_path: str, max_image_dim: int = 1024) -> str:
    """
    Retrieves file content. Text is returned directly. Images are resized
    if large (preserving aspect ratio, max dimension specified by max_image_dim)
    and returned as a base64 Data URI string. Other binary files are described.

    Args:
        file_path: Relative path to file from Jupyter server root. No '..'.
        max_image_dim: Max width/height for images before resizing. Default 1024.

    Returns:
        str: Text content, image Data URI string, binary description, or error message.
    """
    logger.info(f"Executing get_file_content tool for path: {file_path}")

    if ".." in file_path or os.path.isabs(file_path):
        logger.error(f"Invalid file path requested: '{file_path}'.")
        return "[Error: Invalid file path. Must be relative and not contain '..']"

    try:
        response = await _jupyter_api_request("GET", file_path) # Use helper
        file_data = response.json()

        file_type = file_data.get("type")
        if file_type != "file":
            logger.warning(f"Path '{file_path}' is not a file (type: {file_type}).")
            return f"[Error: Path '{file_path}' is not a file (type: {file_type})]"

        content = file_data.get("content")
        content_format = file_data.get("format")
        mimetype = file_data.get("mimetype", "")
        filename = file_data.get("name", os.path.basename(file_path))

        if content is None:
            return f"[Error: No content found for file '{file_path}']"

        if content_format == "text":
            logger.info(f"Returning text content for '{file_path}'.")
            return content
        elif content_format == "base64":
            logger.info(f"Processing base64 content for '{file_path}' (MIME: {mimetype}).")
            if mimetype and mimetype.startswith("image/") and Image: # Check if Pillow was imported
                try:
                    decoded_bytes = base64.b64decode(content)
                    img_buffer = io.BytesIO(decoded_bytes)
                    img = Image.open(img_buffer)
                    original_size = img.size

                    # Resize if image exceeds max dimension
                    if img.width > max_image_dim or img.height > max_image_dim:
                        logger.info(f"Resizing image '{filename}' from {original_size} (max dim: {max_image_dim})")
                        img.thumbnail((max_image_dim, max_image_dim))
                        resized_buffer = io.BytesIO()
                        # Save resized image back to buffer (use PNG for simplicity, could try original format)
                        save_format = 'PNG' if img.format != 'JPEG' else 'JPEG' # Keep JPEG if original
                        img.save(resized_buffer, format=save_format)
                        resized_bytes = resized_buffer.getvalue()
                        # Re-encode the potentially smaller image data
                        content = base64.b64encode(resized_bytes).decode('ascii')
                        logger.info(f"Resized to {img.size}. New base64 length: {len(content)}")
                        # Update mimetype if we forced PNG
                        if save_format == 'PNG': mimetype = 'image/png'

                    # Format as Data URI
                    data_uri = f"data:{mimetype};base64,{content}"
                    logger.info(f"Returning potentially resized image '{filename}' as Data URI.")
                    # Wrap in Markdown for potentially better display in some clients
                    return f"![{filename}]({data_uri})"

                except ImportError:
                     logger.warning("Pillow library not found. Cannot resize image. Returning original base64.")
                     # Fall through to return description or raw base64 if Pillow missing
                except Exception as img_err:
                    logger.error(f"Error processing image {filename}: {img_err}", exc_info=True)
                    return f"[Error processing image file '{filename}': {img_err}]"

            # Fallback for non-image binary or if Pillow failed/missing
            logger.info(f"Returning description for binary file '{filename}'.")
            return f"[Binary Content (MIME: {mimetype}, base64 encoded): {content[:80]}...]"
        else:
            logger.warning(f"Unknown content format '{content_format}' for file '{file_path}'.")
            return f"[Unsupported file content format '{content_format}']"

    except ConnectionError as e:
         if isinstance(e.__cause__, requests.exceptions.HTTPError) and e.__cause__.response.status_code == 404:
              logger.warning(f"File not found at path: '{file_path}'")
              return f"[Error: File not found at path: '{file_path}']"
         else:
              logger.error(f"Connection error retrieving file '{file_path}': {e}")
              return f"[Error retrieving file content: {e}]"
    except Exception as e:
        logger.error(f"Unexpected error in get_file_content for '{file_path}': {e}", exc_info=True)
        return f"[Unexpected Error retrieving file '{file_path}': {e}"
    
@mcp.tool()
def set_target_notebook(new_notebook_path: str) -> str:
    """
    Changes the target notebook file path for subsequent tool calls.

    NOTE: This change only lasts for the current server session.
    If the server restarts, it will revert to the path set in the configuration.
    Ensure the path is relative to the Jupyter Lab starting directory.

    Args:
        new_notebook_path: The new relative path to the target notebook (e.g., "subdir/another_notebook.ipynb").

    Returns:
        str: Confirmation message indicating the new target path.
    """
    global NOTEBOOK_PATH # Declare intent to modify the global variable
    old_path = NOTEBOOK_PATH
    logger.info(f"Executing set_target_notebook tool. Current path: '{old_path}', New path: '{new_notebook_path}'")

    # Basic validation/sanitization (optional but recommended)
    # Avoid absolute paths or directory traversal for security
    if os.path.isabs(new_notebook_path) or ".." in new_notebook_path:
         logger.error(f"Invalid notebook path provided: '{new_notebook_path}'. Must be relative and contain no '..'.")
         return f"[Error: Invalid path '{new_notebook_path}'. Path must be relative.]"

    NOTEBOOK_PATH = new_notebook_path
    logger.info(f"Target notebook path changed to: '{NOTEBOOK_PATH}'")
    return f"Target notebook path set to '{NOTEBOOK_PATH}'. Subsequent tools will use this path."

@mcp.tool()
async def add_cell(content: str, cell_type: str, index: Optional[int] = None) -> str:
    """Adds a new cell with the specified content and type at a given index.
       If index is None or invalid, appends the cell to the end.
    """
    logger.info(f"Executing add_cell tool. Type: {cell_type}, Index: {index}")
    notebook: NbModelClient | None = None
    result_str = "[Error: Unknown issue adding cell]"

    if cell_type not in ["code", "markdown"]:
        return f"[Error: Invalid cell_type '{cell_type}'. Must be 'code' or 'markdown'.]"

    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)

        insert_index: int
        if index is None or not (0 <= index <= num_cells):
            insert_index = num_cells
            logger.info(f"Provided index '{index}' invalid or None. Appending cell at index {insert_index}.")
        else:
            insert_index = index
            logger.info(f"Attempting to insert cell at specified index {insert_index}.")

        new_cell_dict: Dict[str, Any]
        if cell_type == "code":
            new_cell_dict = nbformat.v4.new_code_cell(source=content)
        else:
            new_cell_dict = nbformat.v4.new_markdown_cell(source=content)

        with ydoc.ydoc.transaction():
            # Create YMap ensuring 'source' becomes YText automatically by ypy
            # if nbformat dictionary is correctly structured.
            # The direct YMap conversion should handle nested types like source string -> YText.
            ycell_map = YMap(new_cell_dict)
            ycells.insert(insert_index, ycell_map)

        logger.info(f"Successfully inserted {cell_type} cell at index {insert_index}.")
        result_str = f"{cell_type.capitalize()} cell added at index {insert_index}."

        await asyncio.sleep(0.5)
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in add_cell (type: {cell_type}, index: {index}): {e}", exc_info=True)
        result_str = f"Error adding {cell_type} cell: {e}"
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (add_cell): {final_e}")




@mcp.tool()
async def add_code_cell(cell_content: str) -> str:
    """Adds a code cell to the Jupyter notebook without executing it.

    Args:
        cell_content: The code content for the new cell.

    Returns:
        str: Confirmation message including the index of the added cell.
             Example: "Code cell added at index 5."
    """
    logger.info("Executing add_code_cell tool.")
    notebook = None
    cell_index = -1
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        _try_set_awareness(notebook, "add_code_cell")
        await notebook.start()
        cell_index = notebook.add_code_cell(cell_content)
        logger.info(f"Added code cell at index {cell_index}.")
        await asyncio.sleep(0.5) # Crucial delay before stop
        await notebook.stop()
        notebook = None # Mark as stopped
        result_str = f"Code cell added at index {cell_index}."
        logger.info(f"add_code_cell tool completed. Preparing to return: {result_str}")
        return result_str
    except Exception as e:
        logger.error(f"Error in add_code_cell: {e}", exc_info=True)
        result_str = f"Error adding code cell: {e}"
        if notebook:
             try: await notebook.stop()
             except: pass
        logger.info(f"add_code_cell tool failed. Preparing to return error: {result_str}")
        return result_str
    finally:
         if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try:
                 logger.warning("Stopping notebook client in finally block (add_code_cell)")
                 await notebook.stop()
             except Exception as final_e:
                 logger.error(f"Error stopping notebook in finally (add_code_cell): {final_e}")


@mcp.tool()
async def execute_cell(cell_index: int) -> str:
    """Starts the execution of a specific code cell by its index.
       Does not wait for completion or return output.

    Args:
        cell_index: The index of the cell to execute (0-based).

    Returns:
        str: Confirmation message that execution was initiated.
    """
    logger.info(f"Executing execute_cell tool for cell index {cell_index}")
    global kernel
    if not kernel or not kernel.is_alive():
        # ... (kernel restart logic as before) ...
        logger.warning("Kernel client not alive... Attempting restart.")
        try:
            kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
            kernel.start()
            logger.info("Kernel client restarted.")
        except Exception as kernel_err:
            logger.error(f"Failed to restart kernel client: {kernel_err}", exc_info=True)
            return f"Error: Kernel client connection failed. Cannot execute cell {cell_index}."

    # We need NbModelClient only to get access to the execute_cell method
    # which likely just acts as a proxy to the KernelClient anyway.
    # Let's see if we can trigger execution without keeping the client open.
    notebook = None
    try:
        # Create client, but we might not need to fully start/stop its WebSocket connection
        notebook = NbModelClient(
             get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # Awareness patch might not even be needed if we don't connect WS, but keep for now
        _try_set_awareness(notebook, "execute_cell")

        # We need to ensure the notebook model is loaded to get the cell ID if execute_cell needs it
        # Let's still start/sync briefly.
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        if not (0 <= cell_index < len(ycells)):
             await notebook.stop() # Stop if index invalid
             return f"[Error: Cell index {cell_index} is out of bounds]"

        logger.info(f"Initiating execution for cell index {cell_index}.")
        # --- MODIFICATION: Don't create background task, just call directly? ---
        # The NbModelClient.execute_cell might just send a message via the kernel client
        # and return quickly without needing to be a background task itself.
        # Let's try calling it directly first. If *this* blocks, then backgrounding is needed.
        logger.info(f"Initiating execution for cell index {cell_index}.")

        # --- Add diagnostic logging ---
        try:
            cell_to_exec = ycells[cell_index]
            logger.info(f"DEBUG: Type of cell object at index {cell_index}: {type(cell_to_exec)}")
            if hasattr(cell_to_exec, 'get'): # Check if it's dict-like (or YMap-like)
                source_obj = cell_to_exec.get("source")
                logger.info(f"DEBUG: Type of source object at index {cell_index}: {type(source_obj)}")
                logger.info(f"DEBUG: Content of source object: {str(source_obj)[:100]}...") # Log first 100 chars
            else:
                 logger.warning(f"DEBUG: Cell object at index {cell_index} is not dict-like.")
        except Exception as log_err:
            logger.error(f"DEBUG: Error during pre-execution logging: {log_err}")
        # --- End diagnostic logging ---
        
        notebook.execute_cell(cell_index, kernel)
        logger.info(f"Execution request sent for cell {cell_index}.")

        # --- MODIFICATION: REMOVE stop() ---
        await asyncio.sleep(0.1) # Delay likely not needed if not stopping
        # --- END MODIFICATION ---
        notebook = None # Dereference

        result_str = f"Execution request sent for cell at index {cell_index}."
        logger.info(f"execute_cell tool completed. Preparing to return: {result_str}")
        return result_str # Return immediately after *requesting* execution

    except Exception as e:
        logger.error(f"Error in execute_cell: {e}", exc_info=True)
        result_str = f"Error executing cell {cell_index}: {e}"
        # Ensure we attempt to stop if an error occurred after start
        if notebook:
            try: await notebook.stop()
            except: pass
        return result_str
    # No finally block needed if we don't stop in the main path
    
@mcp.tool()
async def execute_all_cells(cell_timeout: int = 60) -> str:
    """
    Executes all code cells in the current target notebook sequentially from top to bottom.
    Waits for each cell to complete or timeout before proceeding to the next.

    Args:
        cell_timeout: Max time (seconds) to wait for each individual cell execution. Default 60.

    Returns:
        str: Confirmation message indicating completion, or the first error/timeout encountered.
    """
    logger.info(f"Executing execute_all_cells tool (cell timeout: {cell_timeout}s).")
    global kernel # Use global kernel, ensure it's ready
    notebook: NbModelClient | None = None
    executed_count = 0
    total_code_cells = 0
    first_error_msg = None

    # Check/Restart Kernel if needed
    if not kernel or not kernel.is_alive():
        logger.warning("Kernel client not alive. Attempting restart...")
        try:
            kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
            kernel.start()
            logger.info("Kernel client restarted.")
        except Exception as kernel_err:
            logger.error(f"Failed to restart kernel client: {kernel_err}", exc_info=True)
            return f"[Error: Kernel client connection failed. Cannot execute cells.]"

    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # No awareness patch needed here as we aren't primarily modifying structure via this client
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)
        logger.info(f"Found {num_cells} cells. Iterating to execute code cells.")

        for i, cell_data in enumerate(ycells):
            if cell_data.get("cell_type") == "code":
                total_code_cells += 1
                # Skip empty code cells? Optional, for now execute them.
                # source = str(cell_data.get("source", ""))
                # if not source.strip():
                #     logger.info(f"Skipping empty code cell {i}/{num_cells-1}.")
                #     executed_count += 1 # Count as "executed" successfully
                #     continue

                logger.info(f"Requesting execution for cell {i}/{num_cells-1}...")
                try:
                    # Await the execute_cell call with the specified timeout
                    result = await asyncio.wait_for(
                        notebook.execute_cell(i, kernel),
                        timeout=float(cell_timeout)
                    )

                    # Check the kernel's reply for errors
                    status = result.get('status') if result else 'unknown'
                    if status == 'error':
                         error_content = result.get('error', {}) if result else {}
                         err_msg = f"Error in cell {i}: {error_content.get('ename', 'Unknown')}: {error_content.get('evalue', '')}"
                         logger.error(err_msg)
                         first_error_msg = err_msg
                         break # Stop processing on kernel error
                    elif status != 'ok':
                         logger.warning(f"Cell {i} finished with unexpected status: {status}")
                         # Continue to next cell despite non-ok status? Yes.

                    executed_count += 1
                    logger.info(f"Cell {i} execution completed (status: {status}).")

                except asyncio.TimeoutError:
                    logger.error(f"Execution timed out for cell {i} after {cell_timeout}s.")
                    first_error_msg = f"Execution timed out for cell {i} after {cell_timeout}s."
                    break # Stop processing on timeout
                except Exception as exec_err:
                    logger.error(f"Unexpected error during notebook.execute_cell for index {i}: {exec_err}", exc_info=True)
                    first_error_msg = f"Error occurred while executing cell {i}: {exec_err}"
                    break # Stop processing on other errors

        # --- Loop finished or broken ---
        result_str: str
        if first_error_msg:
            result_str = f"Execution stopped: {first_error_msg}. Processed {executed_count}/{total_code_cells} code cells."
        else:
            result_str = f"Successfully executed all {total_code_cells} code cells."

        # --- Crucial Delay before stopping client ---
        await asyncio.sleep(0.5)
        await notebook.stop()
        notebook = None
        logger.info(f"execute_all_cells tool completed. Preparing to return: {result_str}")
        return result_str

    except Exception as e:
        logger.error(f"Error in execute_all_cells tool: {e}", exc_info=True)
        result_str = f"Error during setup or iteration for execute_all_cells: {e}"
        return result_str
    finally:
        # Ensure stop is attempted if client was created and might still be running
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try:
                 logger.warning("Stopping notebook client in finally block (execute_all_cells)")
                 await notebook.stop()
             except Exception as final_e:
                 logger.error(f"Error stopping notebook in finally (execute_all_cells): {final_e}")

@mcp.tool()
async def get_cell_output(cell_index: int, wait_seconds: float = OUTPUT_WAIT_DELAY) -> str:
    """Retrieves the output of a specific code cell by its index.
       Waits briefly for output to appear after execution starts.

    Args:
        cell_index: The index of the cell to get output from (0-based).
        wait_seconds: Time in seconds to wait before reading output (allows kernel time). Default 0.5.

    Returns:
        str: The combined text output(s) of the cell, or a message indicating no output/error.
    """
    logger.info(f"Executing get_cell_output tool for cell index {cell_index}, waiting {wait_seconds}s")
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    notebook = None
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        _try_set_awareness(notebook, "get_cell_output")
        await notebook.start()
        await notebook.wait_until_synced()

        cell_output_str = "[No output found or cell index invalid]"
        try:
            ydoc = notebook._doc
            ycells = ydoc._ycells
            if 0 <= cell_index < len(ycells):
                cell_data = ycells[cell_index]
                outputs = cell_data.get("outputs", [])
                if outputs:
                    output_texts = [extract_output(dict(output)) for output in outputs]
                    cell_output_str = "\n".join(output_texts).strip()
                    if not cell_output_str: # Handle cases where output exists but is empty text
                        cell_output_str = "[Cell output is empty]"
                else:
                    exec_count = cell_data.get("execution_count")
                    if exec_count is not None:
                         cell_output_str = "[Cell executed, but has no output]"
                    else:
                         # Could also check cell_type here, markdown cells have no count
                         if cell_data.get("cell_type") == "markdown":
                             cell_output_str = "[Cannot get output from markdown cell]"
                         else:
                             cell_output_str = "[Cell output not available or cell not executed]"
            else:
                cell_output_str = f"[Error: Cell index {cell_index} is out of bounds]"

        except Exception as read_err:
            logger.error(f"Error reading output for cell {cell_index}: {read_err}", exc_info=True)
            cell_output_str = f"[Error reading output for cell {cell_index}: {read_err}]"

        # No need for delay here as we are only reading state
        await notebook.stop()
        notebook = None # Mark as stopped
        logger.info(f"get_cell_output tool completed for index {cell_index}. Preparing to return.")
        return cell_output_str

    except Exception as e:
        logger.error(f"Error in get_cell_output tool: {e}", exc_info=True)
        result_str = f"Error getting cell output for index {cell_index}: {e}"
        if notebook:
             try: await notebook.stop()
             except: pass
        logger.info(f"get_cell_output tool failed. Preparing to return error: {result_str}")
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try:
                 logger.warning("Stopping notebook client in finally block (get_cell_output)")
                 await notebook.stop()
             except Exception as final_e:
                 logger.error(f"Error stopping notebook in finally (get_cell_output): {final_e}")

@mcp.tool()
async def delete_cell(cell_index: int) -> str:
    """Deletes a specific cell by its index."""
    logger.info(f"Executing delete_cell tool for cell index {cell_index}")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue in delete_cell for index {cell_index}]"
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # No awareness patch needed here
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)

        if not (0 <= cell_index < num_cells):
            result_str = f"[Error: Cell index {cell_index} is out of bounds (0-{num_cells-1})]"
            logger.warning(result_str)
            if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                 await notebook.stop()
                 notebook = None
            return result_str

        logger.info(f"Attempting to delete ycells[{cell_index}] via 'del' operator.")
        with ydoc.ydoc.transaction():
            del ycells[cell_index] # Use standard Python del on the YArray proxy

        logger.info(f"Successfully submitted deletion for cell at index {cell_index} via Yjs.")
        result_str = f"Cell at index {cell_index} deleted." # Note: Deletion might not be reflected immediately by other tools due to sync timing.

        await asyncio.sleep(0.5) # Delay after modification
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in delete_cell for index {cell_index}: {e}", exc_info=True)
        result_str = f"Error deleting cell {cell_index}: {e}"
        return result_str
    finally:
         if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (delete_cell): {final_e}")

                
@mcp.tool()
async def move_cell(from_index: int, to_index: int) -> str:
    """Moves a cell from one index to another using Yjs array manipulation."""
    logger.info(f"Executing move_cell tool from {from_index} to {to_index}")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue in move_cell from {from_index} to {to_index}]"
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # No awareness patch needed here
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)

        # Validate indices before transaction
        if not (0 <= from_index < num_cells):
            result_str = f"[Error: from_index {from_index} is out of bounds (0-{num_cells-1})]"
        elif not (0 <= to_index <= num_cells): # Target can be end
             result_str = f"[Error: to_index {to_index} is out of bounds (0-{num_cells})]"
        else:
            # --- MODIFICATION: Use Yjs array move ---
            logger.info(f"Attempting to move cell from {from_index} to {to_index} using ycells.move_to()")
            with ydoc.ydoc.transaction():
                ycells.move_to(to_index, from_index) # ypy uses move_to(target_index, current_index)
            # --- END MODIFICATION ---
            logger.info(f"Moved cell from {from_index} to {to_index}.")
            result_str = f"Cell moved from index {from_index} to {to_index}."

        if "Error" in result_str:
             logger.warning(result_str)
             if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                  await notebook.stop()
                  notebook = None
             return result_str

        await asyncio.sleep(0.5) # Delay after modification
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in move_cell from {from_index} to {to_index}: {e}", exc_info=True)
        result_str = f"Error moving cell from {from_index} to {to_index}: {e}"
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (move_cell): {final_e}")
@mcp.tool()
async def merge_cells(indices: list[int]) -> str:
    """Merges adjacent cells into the first cell, deleting the others."""
    logger.info(f"Executing merge_cells tool for indices {indices}")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue merging cells {indices}]"
    if not indices or len(indices) < 2:
        return "[Error: Need at least two consecutive indices to merge.]"
    indices.sort()
    if any(indices[i+1] != indices[i] + 1 for i in range(len(indices) - 1)):
        return f"[Error: Indices {indices} are not consecutive.]"

    target_index = indices[0]
    indices_to_delete = indices[1:]

    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)

        if not (0 <= target_index < num_cells and all(0 <= i < num_cells for i in indices_to_delete)):
             result_str = f"[Error: Indices {indices} out of bounds (0-{num_cells-1})]"
             logger.warning(result_str)
             if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                  await notebook.stop()
                  notebook = None
             return result_str

        # Perform merge and delete within a single transaction
        with ydoc.ydoc.transaction():
            sources_to_merge = []
            final_cell_type = "markdown"
            for i in indices:
                 cell_data = ycells[i]
                 # Convert YText to str for joining
                 sources_to_merge.append(str(cell_data.get("source", "")))
                 if cell_data.get("cell_type") == "code":
                     final_cell_type = "code"

            merged_source = "\n".join(sources_to_merge)

            # Modify first cell using YText methods
            target_cell = ycells[target_index]
            source_obj = target_cell.get("source")
            if isinstance(source_obj, YText):
                # Correct way for pycrdt.Text: delete existing, insert new
                existing_content = str(source_obj)
                if existing_content:
                    source_obj.delete(0, len(existing_content))
                source_obj.insert(0, merged_source)
            else:
                # If it's not YText (or doesn't exist), replace/create it
                target_cell["source"] = YText(merged_source)

            target_cell["cell_type"] = final_cell_type
            if "outputs" in target_cell: target_cell["outputs"].clear()
            if "execution_count" in target_cell: target_cell["execution_count"] = None
            logger.info(f"Merging into cell {target_index}. New type: {final_cell_type}. Deleting {indices_to_delete}")

            # Delete subsequent cells in reverse order
            for i in sorted(indices_to_delete, reverse=True):
                 del ycells[i]

        result_str = f"Cells {indices} merged into index {target_index}."

        await asyncio.sleep(0.5)
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in merge_cells for indices {indices}: {e}", exc_info=True)
        result_str = f"Error merging cells {indices}: {e}"
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (merge_cells): {final_e}")



@mcp.tool()
async def split_cell(cell_index: int, line_number: int) -> str:
    """Splits a cell at a specific line number (1-based)."""
    logger.info(f"Executing split_cell tool for index {cell_index} at line {line_number}")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue splitting cell {cell_index}]"
    if line_number < 1:
        return "[Error: line_number must be 1 or greater]"

    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)

        if not (0 <= cell_index < num_cells):
             result_str = f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"
             logger.warning(result_str)
             if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                  await notebook.stop()
                  notebook = None
             return result_str

        cell_data_read = ycells[cell_index]
        original_source = str(cell_data_read.get("source", "")) # Read as string
        original_cell_type = cell_data_read.get("cell_type", "markdown")
        source_lines = original_source.splitlines(True)

        if line_number > len(source_lines) + 1:
             result_str = f"[Error: Line number {line_number} beyond end ({len(source_lines)} lines)]"
             logger.warning(result_str)
             if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                  await notebook.stop()
                  notebook = None
             return result_str

        new_cell_index = cell_index + 1
        with ydoc.ydoc.transaction():
             source_part1 = "".join(source_lines[:line_number-1])
             source_part2 = "".join(source_lines[line_number-1:])
            
             cell_data_write = ycells[cell_index]
             source_obj = cell_data_write.get("source")
             if isinstance(source_obj, YText):
                 # Correct way for pycrdt.Text: delete existing, insert new
                 existing_content = str(source_obj)
                 if existing_content:
                     source_obj.delete(0, len(existing_content))
                 source_obj.insert(0, source_part1) # Insert the first part
             else:
                 # If it's not YText (or doesn't exist), replace/create it
                 cell_data_write["source"] = YText(source_part1)

             if original_cell_type == "code":
                 if "outputs" in cell_data_write: cell_data_write["outputs"].clear()
                 if "execution_count" in cell_data_write: cell_data_write["execution_count"] = None

            # Create new cell dict using nbformat
             if original_cell_type == "code":
                 new_cell_dict = nbformat.v4.new_code_cell(source=source_part2)
             else:
                 new_cell_dict = nbformat.v4.new_markdown_cell(source=source_part2)

             # Convert to YMap and insert into YArray
             logger.info(f"Splitting cell {cell_index}. Inserting new {original_cell_type} cell at index {new_cell_index}")
             ycells.insert(new_cell_index, YMap(new_cell_dict))

        result_str = f"Cell {cell_index} split at line {line_number}. New cell created at index {new_cell_index}."

        await asyncio.sleep(0.5)
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in split_cell for index {cell_index} at line {line_number}: {e}", exc_info=True)
        result_str = f"Error splitting cell {cell_index}: {e}"
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (split_cell): {final_e}")



@mcp.tool()
async def get_all_cells() -> list[dict[str, Any]]:
    """Retrieves basic info (index, type, source) for all cells."""
    logger.info("Executing get_all_cells tool.")
    all_cells = []
    notebook: NbModelClient | None = None
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # No awareness patch needed here
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        logger.info(f"Processing {len(ycells)} cells for content.")
        for i, cell_data_y in enumerate(ycells):
             # --- MODIFICATION: Convert YText source to str ---
             cell_info = {
                 "index": i,
                 "cell_type": cell_data_y.get("cell_type"),
                 "source": str(cell_data_y.get("source", "")), # Convert YText to str
                 # Add execution_count for code cells if desired
                 "execution_count": cell_data_y.get("execution_count") if cell_data_y.get("cell_type") == "code" else None
             }
             # --- END MODIFICATION ---
             all_cells.append(cell_info)

        await notebook.stop() # No delay needed for read-only
        notebook = None
        return all_cells
    except Exception as e:
        logger.error(f"Error in get_all_cells: {e}", exc_info=True)
        if notebook:
             try: await notebook.stop()
             except: pass
        return [{"error": f"Error during retrieval: {e}"}]
    finally:
         if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try: await notebook.stop()
             except Exception as final_e: logger.error(f"Error stopping notebook in finally (get_all_cells): {final_e}")

@mcp.tool()
async def edit_cell_source(cell_index: int, new_content: str) -> str:
    """Edits the source content of a specific cell by its index."""
    logger.info(f"Executing edit_cell_source tool for cell index {cell_index}")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue in edit_cell_source for index {cell_index}]"
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        if 0 <= cell_index < len(ycells):
            cell_data = ycells[cell_index] # This should be a YMap
            with ydoc.ydoc.transaction():
                 source_obj = cell_data.get("source") # Get the source object
                if isinstance(source_obj, YText):
                    # Correct way for pycrdt.Text: delete existing, insert new
                    existing_content = str(source_obj)
                    if existing_content: # Check if there's actually text to delete
                        source_obj.delete(0, len(existing_content))
                    source_obj.insert(0, new_content) # Insert the new content at the beginning
                 else:
                     # If it's not YText (or doesn't exist), replace/create it
                     cell_data["source"] = YText(new_content)
            logger.info(f"Updated source for cell index {cell_index}.")
            result_str = f"Source updated for cell at index {cell_index}."
        else:
            result_str = f"[Error: Cell index {cell_index} is out of bounds (0-{len(ycells)-1})]"
            logger.warning(result_str)
            if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                 await notebook.stop()
                 notebook = None
            return result_str

        await asyncio.sleep(0.5)
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in edit_cell_source for index {cell_index}: {e}", exc_info=True)
        result_str = f"Error editing cell {cell_index}: {e}"
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (edit_cell_source): {final_e}")

@mcp.tool()
async def get_all_outputs() -> dict[int, str]:
    """Retrieves the combined output string for all code cells in the notebook.

    Returns:
        dict[int, str]: A dictionary mapping cell index to its combined output string.
                        Returns "[No output]" if a code cell has no output.
                        Non-code cells are skipped.
    """
    logger.info("Executing get_all_outputs tool.")
    all_outputs = {}
    notebook = None
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        _try_set_awareness(notebook, "get_all_outputs") # Use helper
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        logger.info(f"Processing {len(ycells)} cells for outputs.")
        for i, cell_data in enumerate(ycells):
            if cell_data.get("cell_type") == "code":
                outputs = cell_data.get("outputs", [])
                if outputs:
                    output_texts = [extract_output(dict(output)) for output in outputs]
                    output_str = "\n".join(output_texts).strip()
                    all_outputs[i] = output_str if output_str else "[No output]"
                else:
                    # Check execution count to see if it ran
                    exec_count = cell_data.get("execution_count")
                    if exec_count is not None:
                         all_outputs[i] = "[No output]"
                    else:
                         all_outputs[i] = "[Not executed]" # Indicate not run yet

        await notebook.stop() # No delay needed for read-only
        notebook = None
        logger.info(f"get_all_outputs tool completed. Found outputs for {len(all_outputs)} code cells.")
        return all_outputs

    except Exception as e:
        logger.error(f"Error in get_all_outputs: {e}", exc_info=True)
        if notebook:
             try: await notebook.stop()
             except: pass
        # Return the dictionary accumulated so far, or an error indicator if preferred
        all_outputs[-1] = f"Error during retrieval: {e}" # Use index -1 for error
        return all_outputs
    finally:
         if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try:
                 logger.warning("Stopping notebook client in finally block (get_all_outputs)")
                 await notebook.stop()
             except Exception as final_e:
                 logger.error(f"Error stopping notebook in finally (get_all_outputs): {final_e}")



@mcp.tool()
async def install_package(package_name: str, timeout_seconds: int = 60) -> str:
    """
    Attempts to install a Python package into the kernel's environment
    by adding and executing a '!pip install' command in a temporary cell,
    waiting a fixed time for completion, retrieving output, and deleting the cell.

    NOTE: This interacts with the kernel environment, not the MCP server's.
    Success depends on network access from the kernel and sufficient wait time.
    The actual install might take longer than the wait time.

    Args:
        package_name: The name of the package to install (e.g., 'pandas', 'requests').
        timeout_seconds: Max seconds to wait after starting install before getting output. Default 60.

    Returns:
        str: The output from the pip install command attempt, or an error message.
    """
    logger.info(f"Executing install_package tool for package: {package_name}")
    # Basic sanitization (avoid complex shell injection)
    safe_package_name = "".join(c for c in package_name if c.isalnum() or c in '-_==.')
    if not safe_package_name or safe_package_name != package_name:
        logger.error(f"Invalid characters in package name: {package_name}")
        return f"[Error: Invalid package name '{package_name}']"

    code_content = f"!python -m pip install {safe_package_name}"
    logger.info(f"Preparing to execute: {code_content}")
    cell_output = f"[Error: Failed to retrieve output for install command]" # Default error
    cell_index: int | None = None

    try:
        # 1. Add cell
        add_result = await add_code_cell(code_content)
        cell_index = _parse_index_from_message(add_result)
        if cell_index is None:
            logger.error(f"Failed to add cell for package install. Result: {add_result}")
            return f"[Error adding cell for install: {add_result}]"
        logger.info(f"Install cell added at index {cell_index}.")

        # 2. Execute cell
        exec_result = await execute_cell(cell_index)
        if "Error" in exec_result:
            logger.error(f"Failed to start execution for install cell {cell_index}. Result: {exec_result}")
            # Try to delete the cell anyway
            try: await delete_cell(cell_index)
            except: pass
            return f"[Error starting execution for install: {exec_result}]"
        logger.info(f"Execution started for install cell {cell_index}. Waiting {timeout_seconds}s...")

        # 3. Wait (Fixed duration)
        await asyncio.sleep(timeout_seconds)
        logger.info(f"Finished waiting for install cell {cell_index}.")

        # 4. Get Output (use wait_seconds=0 as we already waited)
        cell_output = await get_cell_output(cell_index, wait_seconds=0)
        logger.info(f"Output received for install cell {cell_index}.")

        # 5. Delete Cell (cleanup - wrap in try/except)
        logger.info(f"Attempting to delete install cell {cell_index}.")
        try:
            delete_result = await delete_cell(cell_index)
            logger.info(f"Deletion result for cell {cell_index}: {delete_result}")
        except Exception as del_e:
            logger.error(f"Failed to delete cell {cell_index} after install: {del_e}", exc_info=True)
            # Continue to return the output we got

        # 6. Return pip output
        return cell_output

    except Exception as e:
        logger.error(f"Error during install_package orchestration for {package_name}: {e}", exc_info=True)
        # Try cleanup if index known
        if cell_index is not None:
            try:
                logger.warning(f"Attempting cleanup delete for cell {cell_index} after error.")
                await delete_cell(cell_index)
            except Exception as final_del_e:
                 logger.error(f"Cleanup delete failed for cell {cell_index}: {final_del_e}")
        return f"[Error installing package {package_name}: {e}]"


@mcp.tool()
async def list_installed_packages(wait_seconds: int = 5) -> str:
    """
    Lists Python packages installed in the kernel's environment
    by adding and executing a '!pip list' command in a temporary cell,
    waiting briefly, retrieving output, and deleting the cell.

    NOTE: This interacts with the kernel environment, not the MCP server's.

    Args:
        wait_seconds: Seconds to wait after starting list before getting output. Default 5.

    Returns:
        str: The output from the pip list command attempt, or an error message.
    """
    logger.info("Executing list_installed_packages tool")
    code_content = "!python -m pip list"
    cell_output = f"[Error: Failed to retrieve output for list command]" # Default error
    cell_index: int | None = None

    try:
        # 1. Add cell
        add_result = await add_code_cell(code_content)
        cell_index = _parse_index_from_message(add_result)
        if cell_index is None:
            logger.error(f"Failed to add cell for package list. Result: {add_result}")
            return f"[Error adding cell for list: {add_result}]"
        logger.info(f"List cell added at index {cell_index}.")

        # 2. Execute cell
        exec_result = await execute_cell(cell_index)
        if "Error" in exec_result:
            logger.error(f"Failed to start execution for list cell {cell_index}. Result: {exec_result}")
            try: await delete_cell(cell_index)
            except: pass
            return f"[Error starting execution for list: {exec_result}]"
        logger.info(f"Execution started for list cell {cell_index}. Waiting {wait_seconds}s...")

        # 3. Wait
        await asyncio.sleep(wait_seconds)
        logger.info(f"Finished waiting for list cell {cell_index}.")

        # 4. Get Output (use wait_seconds=0 as we already waited)
        cell_output = await get_cell_output(cell_index, wait_seconds=0)
        logger.info(f"Output received for list cell {cell_index}.")

        # 5. Delete Cell
        logger.info(f"Attempting to delete list cell {cell_index}.")
        try:
            delete_result = await delete_cell(cell_index)
            logger.info(f"Deletion result for cell {cell_index}: {delete_result}")
        except Exception as del_e:
            logger.error(f"Failed to delete cell {cell_index} after list: {del_e}", exc_info=True)

        # 6. Return pip output
        return cell_output

    except Exception as e:
        logger.error(f"Error during list_installed_packages orchestration: {e}", exc_info=True)
        if cell_index is not None:
            try:
                logger.warning(f"Attempting cleanup delete for cell {cell_index} after error.")
                await delete_cell(cell_index)
            except Exception as final_del_e:
                 logger.error(f"Cleanup delete failed for cell {cell_index}: {final_del_e}")
        return f"[Error listing packages: {e}]"
                   
                 
if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    # Set lower levels for specific noisy libraries if desired
    # logging.getLogger("websockets").setLevel(logging.WARNING)
    # logging.getLogger("jupyter_server_ydoc").setLevel(logging.INFO) # Maybe useful
    logger.info(f"Starting Jupyter MCP Server for notebook: {NOTEBOOK_PATH} on {SERVER_URL} [Log Level: {log_level}]")
    mcp.run(transport="stdio")