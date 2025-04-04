# Copyright (c) 2023-2024 Datalayer, Inc.
# Copyright (c) 2025 Alexander Isaev
# BSD 3-Clause License

import re
import logging
import os
import asyncio # Make sure asyncio is imported
from typing import Any, List, Dict, Optional # Import necessary types
import nbformat
import requests
from urllib.parse import urljoin, quote
from functools import partial
import io
from PIL import Image # If Pillow is installed
import base64
from pycrdt import Array as YArray, Map as YMap, Text as YText

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
    """Adds a new cell with specified content and type to the notebook.

    Ensures correct Yjs types are used internally for synchronization.
    If the provided index is None or out of bounds, the cell will be appended
    to the end of the notebook.

    Args:
        content (str): The source content (code or markdown) for the new cell.
        cell_type (str): The type of the cell. Must be either 'code' or 'markdown'.
        index (Optional[int]): The 0-based index at which to insert the new cell.
            If None or invalid, the cell is appended to the end. Defaults to None.

    Returns:
        str: A confirmation message indicating the type of cell added and its
             final index (e.g., "Code cell added at index 5."), or an error
             message string starting with "[Error: ...]".
    """
    
    global logger, SERVER_URL, TOKEN, NOTEBOOK_PATH # Add needed globals

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

        # --- Prepare cell dictionary with EXPLICIT Yjs types ---
        new_cell_pre_ymap: Dict[str, Any] = {}
        new_cell_pre_ymap["cell_type"] = cell_type
        new_cell_pre_ymap["source"] = YText(content) # Create YText directly

        if cell_type == "code":
            # Use nbformat defaults for metadata, execution_count if desired
            base_cell = nbformat.v4.new_code_cell(source="") # Use nbformat for structure/defaults
            new_cell_pre_ymap["metadata"] = base_cell.metadata
            new_cell_pre_ymap["outputs"] = YArray() # Explicitly create YArray
            new_cell_pre_ymap["execution_count"] = None
        else: # markdown
            base_cell = nbformat.v4.new_markdown_cell(source="")
            new_cell_pre_ymap["metadata"] = base_cell.metadata
            # Markdown cells shouldn't have outputs/execution_count fields usually

        # --- End preparation ---

        with ydoc.ydoc.transaction():
            # Convert the dictionary (which now contains Yjs objects) to a YMap
            ycell_map = YMap(new_cell_pre_ymap)
            ycells.insert(insert_index, ycell_map)

        logger.info(f"Successfully inserted {cell_type} cell at index {insert_index}.")
        result_str = f"{cell_type.capitalize()} cell added at index {insert_index}."

        await asyncio.sleep(0.5) # Use the longer delay
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in add_cell (type: {cell_type}, index: {index}): {e}", exc_info=True)
        result_str = f"Error adding {cell_type} cell: {e}"
        # Cleanup happens in finally
        return result_str
    finally:
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try: await notebook.stop()
             except Exception as final_e: logger.error(f"Error stopping notebook in finally (add_cell): {final_e}")



# More stable than add_cell
@mcp.tool()
async def add_code_cell_on_bottom(cell_content: str) -> str:
    """Adds a code cell to the Jupyter notebook without executing it, on the bottom of the notebook.

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

# --- Tool: execute_cell (Modified to use asyncio.to_thread) ---
@mcp.tool()
async def execute_cell(cell_index: int) -> str:
    """
    Sends a request to execute a specific code cell by its index,
    running the request submission in a separate thread to potentially avoid
    blocking the main server loop during slow kernel interactions.
    Does NOT wait for kernel completion or guarantee execution success.

    Args:
        cell_index: The index of the cell to execute (0-based).

    Returns:
        str: Confirmation message that execution request was sent/dispatched, or an error message.
    """
    global kernel, logger, SERVER_URL, TOKEN, NOTEBOOK_PATH # Add needed globals

    logger.info(f"Executing execute_cell tool for cell index {cell_index} (using thread dispatch)")

    # Check/Restart Kernel if needed
    if not kernel or not kernel.is_alive():
        logger.warning("Kernel client not alive... Attempting restart.")
        try:
            kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
            kernel.start()
            logger.info("Kernel client restarted.")
            if not kernel.is_alive(): raise RuntimeError("Kernel restart failed")
        except Exception as kernel_err:
            logger.error(f"Failed to restart kernel client: {kernel_err}", exc_info=True)
            return f"[Error: Kernel client connection failed. Cannot send execution request for cell {cell_index}.]"

    notebook: NbModelClient | None = None
    result_str = f"[Error sending execution request for cell {cell_index}: Unknown issue]" # Default error message
    try:
        # Need client briefly to get model and call execute
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        _try_set_awareness(notebook, "execute_cell") # Keep if needed
        await notebook.start()
        await notebook.wait_until_synced()

        # Validate index
        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells) # Define num_cells here
        if not (0 <= cell_index < num_cells):
            result_str = f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"
            logger.warning(result_str)
            # No need to stop client here, finally block will handle it
            return result_str

        # Validate cell type
        cell_data_check = ycells[cell_index]
        if cell_data_check.get("cell_type") != "code":
             result_str = f"[Error: Cell index {cell_index} is not a code cell]"
             logger.warning(result_str)
             # No need to stop client here, finally block will handle it
             return result_str

        # --- Execute the potentially blocking call in a separate thread ---
        try:
            logger.info(f"Dispatching execution request for cell {cell_index} to thread...")
            # Run the synchronous 'notebook.execute_cell' in asyncio's default thread pool
            # Pass the function and its arguments to to_thread
            await asyncio.to_thread(
                 notebook.execute_cell, # The function to run in thread
                 cell_index,            # First argument to notebook.execute_cell
                 kernel                 # Second argument to notebook.execute_cell
            )
            # If to_thread completes without error, the request was *sent* successfully
            logger.info(f"Execution request successfully dispatched via thread for cell {cell_index}.")
            result_str = f"Execution request sent for cell at index {cell_index}."
        except Exception as exec_dispatch_err:
             # Catch errors that happen *during the call* within the thread
             logger.error(f"Error dispatching execute_cell to thread: {exec_dispatch_err}", exc_info=True)
             result_str = f"[Error dispatching execution request for cell {cell_index}: {exec_dispatch_err}]"
        # --- End threaded execution call ---

        # Stop the client connection shortly after *dispatching* the request
        # The actual kernel execution happens independently in the kernel process.
        await asyncio.sleep(0.1)
        # Ensure notebook client was successfully created before trying to stop
        if notebook:
            await notebook.stop()
            notebook = None # Mark as stopped *after* successful stop

        return result_str

    except Exception as e:
        logger.error(f"Error during execute_cell setup/connection for index {cell_index}: {e}", exc_info=True)
        # Ensure result_str reflects the outer error if it happens before dispatch attempt
        if "Unknown issue" in result_str:
            result_str = f"[Error in execute_cell setup for cell {cell_index}: {e}]"
        # Cleanup happens in finally
        return result_str
    finally:
        # Ensure stop is attempted if client was created and might still be running
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
             try:
                 logger.warning("Stopping notebook client in finally block (execute_cell)")
                 await notebook.stop()
             except Exception as final_e:
                 logger.error(f"Error stopping notebook in finally (execute_cell): {final_e}")

# --- Tool: execute_all_cells ---
@mcp.tool()
async def execute_all_cells() -> str:
    """
    Sends execution requests sequentially for all code cells found in the notebook.
    Does NOT wait for completion or report kernel-side errors/timeouts.
    """
    global kernel, logger, SERVER_URL, TOKEN, NOTEBOOK_PATH # Add needed globals
    logger.info(f"Executing execute_all_cells tool.")

    # Check/Restart Kernel if needed
    if not kernel or not kernel.is_alive():
        logger.warning("Kernel client not alive. Attempting restart...")
        try:
            kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
            kernel.start()
            logger.info("Kernel client restarted.")
            if not kernel.is_alive(): raise RuntimeError("Kernel restart failed")
        except Exception as kernel_err:
            logger.error(f"Failed to restart kernel client: {kernel_err}", exc_info=True)
            return f"[Error: Kernel client connection failed. Cannot execute cells.]"

    notebook: NbModelClient | None = None
    total_code_cells = 0
    cells_requested = 0
    request_error = None # To store error during request sending

    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells
        num_cells = len(ycells)
        logger.info(f"Found {num_cells} cells. Iterating to send execution requests for code cells.")

        for i, cell_data in enumerate(ycells):
            # Check cell type inside the loop
            if cell_data.get("cell_type") == "code":
                total_code_cells += 1
                try:
                    logger.info(f"Sending execution request for code cell {i}/{num_cells-1}...")
                    # Directly call the non-awaitable method
                    notebook.execute_cell(i, kernel)
                    cells_requested += 1
                except Exception as send_err:
                    # Catch errors ONLY during the *sending* of the request itself
                    logger.error(f"Error sending execution request for cell {i}: {send_err}", exc_info=True)
                    request_error = f"Error sending request for cell {i}: {send_err}"
                    break # Stop trying to send more requests if one fails

        # --- Loop finished or broken ---
        result_str: str
        if request_error:
            result_str = f"Stopped sending requests due to error: {request_error}. Sent requests for {cells_requested}/{total_code_cells} code cells found."
        else:
            result_str = f"Successfully sent execution requests for all {total_code_cells} code cells found."

        # Stop the client connection shortly after sending the last request
        await asyncio.sleep(0.1) # Keep small delay maybe?
        await notebook.stop()
        notebook = None # Mark as stopped
        logger.info(f"execute_all_cells tool completed sending requests. Preparing to return: {result_str}")
        return result_str

    except Exception as e:
        logger.error(f"Error during setup or iteration for execute_all_cells: {e}", exc_info=True)
        return f"Error during setup or iteration for execute_all_cells: {e}"
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
    """
    Moves a cell from from_index to to_index by copying data, deleting original,
    and inserting a new, correctly structured cell.
    """
    global logger, SERVER_URL, TOKEN, NOTEBOOK_PATH

    logger.info(f"Executing move_cell tool from {from_index} to {to_index} (robust copy method)")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue in move_cell from {from_index} to {to_index}]"
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells # This is the pycrdt.Array
        num_cells = len(ycells)

        # --- Validation ---
        if not (0 <= from_index < num_cells):
            result_str = f"[Error: from_index {from_index} is out of bounds (0-{num_cells-1})]"
        elif not (0 <= to_index <= num_cells): # Allow moving to the very end
             result_str = f"[Error: to_index {to_index} is out of bounds (0-{num_cells})]"
        elif from_index == to_index:
             result_str = f"Cell is already at index {from_index}." # No move needed
        else:
            # --- Perform Copy / Delete / Create / Insert ---
            logger.info(f"Attempting robust move: Copying {from_index}, Deleting {from_index}, Inserting new at {to_index}")
            try:
                # Perform operations within a single transaction
                with ydoc.ydoc.transaction():
                    # 1. Copy data from source cell, converting Yjs types to Python types
                    source_cell_y = ycells[from_index]
                    # Safely get data, providing defaults for missing keys
                    copied_data = {
                        "cell_type": source_cell_y.get("cell_type", "code"), # Default to code if missing? Or error?
                        "source": str(source_cell_y.get("source", "")),
                        "metadata": dict(source_cell_y.get("metadata", YMap())),
                    }
                    cell_type = copied_data["cell_type"]
                    if cell_type == "code":
                        copied_data["outputs"] = list(source_cell_y.get("outputs", YArray()))
                        copied_data["execution_count"] = source_cell_y.get("execution_count") # Can be None

                    # 2. Delete the original cell (use pop for potential direct object reuse if needed later, but del is fine)
                    del ycells[from_index]
                    # item_to_move_data = ycells.pop(from_index) # Alternative to del

                    # 3. Prepare the dictionary for the NEW YMap, converting back to Yjs types
                    new_cell_pre_ymap: Dict[str, Any] = {}
                    new_cell_pre_ymap["cell_type"] = cell_type
                    new_cell_pre_ymap["source"] = YText(copied_data["source"])
                    new_cell_pre_ymap["metadata"] = YMap(copied_data.get("metadata", {}))

                    if cell_type == "code":
                         # Convert Python list of outputs back to YArray
                         new_cell_pre_ymap["outputs"] = YArray(copied_data.get("outputs", []))
                         # Add execution_count only if it was present and not None in the copy
                         exec_count = copied_data.get("execution_count")
                         if exec_count is not None:
                              new_cell_pre_ymap["execution_count"] = exec_count
                    # else: # Markdown - Ensure outputs/count are not present if strict schema needed
                    #    if "outputs" in new_cell_pre_ymap: del new_cell_pre_ymap["outputs"]
                    #    if "execution_count" in new_cell_pre_ymap: del new_cell_pre_ymap["execution_count"]


                    # 4. Insert the NEW cell YMap at the target index
                    # Adjust insertion index if moving item to later position
                    # Example: move 0 -> 2 in [a,b,c]. del ycells[0] -> [b,c]. insert at 2 -> [b,c,a]. Correct.
                    # Example: move 2 -> 0 in [a,b,c]. del ycells[2] -> [a,b]. insert at 0 -> [c,a,b]. Correct.
                    # It seems direct insertion at to_index works correctly after deletion.
                    ycells.insert(to_index, YMap(new_cell_pre_ymap))

                logger.info(f"Robust move successful: Cell from {from_index} inserted at {to_index}.")
                result_str = f"Cell moved from index {from_index} to {to_index}."
            except KeyError as ke:
                 # Catch errors if expected keys are missing during copy (e.g., missing cell_type?)
                 logger.error(f"Missing key during cell copy/move from {from_index}: {ke}", exc_info=True)
                 result_str = f"[Error: Cell structure invalid during move - missing key {ke}]"
            except Exception as move_e:
                 # Catch other errors specifically during the move transaction
                 logger.error(f"Error during robust move transaction from {from_index} to {to_index}: {move_e}", exc_info=True)
                 result_str = f"[Error during move operation: {move_e}]"
            # --- End Copy / Delete / Create / Insert ---

        # Handle validation or transaction errors before stopping client
        # Check result_str which might have been updated by exception handling
        if "Error" in result_str or "already at index" in result_str:
             logger.warning(f"Move cell result: {result_str}")
             if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                   try: await notebook.stop()
                   except Exception: pass
                   notebook = None
             return result_str

        # Success case
        await asyncio.sleep(0.5) # Keep delay after modification
        await notebook.stop()
        notebook = None
        return result_str

    except Exception as e:
        # Catch errors during setup/connection
        logger.error(f"Error in move_cell setup from {from_index} to {to_index}: {e}", exc_info=True)
        result_str = f"Error moving cell from {from_index} to {to_index}: {e}"
        # Cleanup happens in finally
        return result_str
    finally:
        # Ensure stop is attempted if client was created and might still be running
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (move_cell): {final_e}")

            
@mcp.tool()
async def move_cell(from_index: int, to_index: int) -> str:
    """
    Moves a cell from from_index to to_index by deleting the original reference
    and re-inserting it at the target index within a transaction.
    """
    global logger, SERVER_URL, TOKEN, NOTEBOOK_PATH # Add needed globals

    logger.info(f"Executing move_cell tool from {from_index} to {to_index} (simple del/insert method)")
    notebook: NbModelClient | None = None
    result_str = f"[Error: Unknown issue in move_cell from {from_index} to {to_index}]"
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        # No awareness patch needed here usually for move
        await notebook.start()
        await notebook.wait_until_synced()

        ydoc = notebook._doc
        ycells = ydoc._ycells # This is the pycrdt.Array
        num_cells = len(ycells)

        # --- Validation ---
        if not (0 <= from_index < num_cells):
            result_str = f"[Error: from_index {from_index} is out of bounds (0-{num_cells-1})]"
        elif not (0 <= to_index <= num_cells): # Allow moving to the very end (index num_cells)
             result_str = f"[Error: to_index {to_index} is out of bounds (0-{num_cells})]"
        elif from_index == to_index:
             result_str = f"Cell is already at index {from_index}." # No move needed
        else:
            # --- Perform simple Delete / Insert of same reference ---
            logger.info(f"Attempting simple move: Deleting {from_index}, Inserting reference at {to_index}")
            try:
                # Perform operations within a single transaction
                with ydoc.ydoc.transaction():
                    # 1. Get the item reference (this is a YMap)
                    item_to_move = ycells[from_index]

                    # 2. Delete from the original position
                    del ycells[from_index]

                    # 3. Insert the same item reference at the target position
                    # pycrdt's insert should handle index adjustments correctly within transaction.
                    ycells.insert(to_index, item_to_move)

                logger.info(f"Simple move successful: Cell from {from_index} moved to {to_index}.")
                result_str = f"Cell moved from index {from_index} to {to_index}."
            except Exception as move_e:
                 # Catch errors specifically during the move transaction
                 logger.error(f"Error during simple move transaction from {from_index} to {to_index}: {move_e}", exc_info=True)
                 result_str = f"[Error during move operation: {move_e}]"
            # --- End Delete / Insert block ---

        # Handle validation or transaction errors before stopping client
        if "Error" in result_str or "already at index" in result_str:
             logger.warning(f"Move cell result: {result_str}")
             if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                   try: await notebook.stop()
                   except Exception: pass # Ignore stop errors if only validation failed
                   notebook = None
             return result_str

        # Success case
        await asyncio.sleep(0.5) # Keep delay after modification
        await notebook.stop()
        notebook = None
        return result_str

    except Exception as e:
        # Catch errors during setup/connection
        logger.error(f"Error in move_cell setup from {from_index} to {to_index}: {e}", exc_info=True)
        result_str = f"Error moving cell from {from_index} to {to_index}: {e}"
        # Cleanup happens in finally
        return result_str
    finally:
        # Ensure stop is attempted if client was created and might still be running
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (move_cell): {final_e}")
            
@mcp.tool()
async def search_notebook_cells(search_string: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Searches through all cells in the current notebook for a given string in their source code.

    Args:
        search_string: The string of code or text to search for within cell sources.
        case_sensitive: Set to True for a case-sensitive search (default is False, ignoring case).

    Returns:
        A list of dictionaries, where each dictionary represents a cell containing
        the search string. Each dictionary includes 'index', 'cell_type', and 'source'.
        Returns an empty list if the string is not found or if there's an error reading cells.
        Example return: [{'index': 0, 'cell_type': 'code', 'source': 'import pandas as pd\nprint("hello")'}]
    """
    logger.info(f"Executing search_notebook_cells tool for: '{search_string}' (case_sensitive={case_sensitive})")
    matches = []

    try:
        # 1. Get all cell data using the existing tool
        # Assumes get_all_cells() returns a list of dicts like [{'index': 0, 'cell_type': 'code', 'source': '...', 'execution_count': None}, ...]
        # Or returns [{'error': '...'}] on failure
        all_cells = await get_all_cells()

        # 2. Check for errors from get_all_cells
        # Handle potential error format like [{"error": "..."}]
        if not all_cells or (isinstance(all_cells, list) and len(all_cells) > 0 and isinstance(all_cells[0].get("error"), str)):
             logger.error(f"Failed to retrieve cells from get_all_cells. Response: {all_cells}")
             # Return empty list on error to indicate no matches found due to cell access failure
             return []
        # Handle case where get_all_cells didn't return a list (unexpected)
        if not isinstance(all_cells, list):
            logger.error(f"Unexpected response type from get_all_cells: {type(all_cells)}")
            return []

        # 3. Iterate and search through the cells
        search_term = search_string if case_sensitive else search_string.lower()

        for cell in all_cells:
            # Ensure source exists and is a string before searching
            source = cell.get("source") # get_all_cells should return source as string
            if not isinstance(source, str):
                # Log a warning if source is not a string, skip this cell
                logger.warning(f"Cell {cell.get('index')} has non-string source: {type(source)}. Skipping.")
                continue

            source_to_search = source if case_sensitive else source.lower()

            if search_term in source_to_search:
                logger.info(f"Found search string in cell index {cell.get('index')}")
                # Append relevant info for the matched cell
                matches.append({
                    "index": cell.get("index"),           # Get cell index
                    "cell_type": cell.get("cell_type"),   # Get cell type
                    "source": source                      # Return the original source content
                })

        logger.info(f"Search complete. Found {len(matches)} matches.")
        return matches

    except Exception as e:
        logger.error(f"Unexpected error in search_notebook_cells: {e}", exc_info=True)
        # Return empty list on any unexpected error during search processing
        return []

@mcp.tool()
async def split_cell(cell_index: int, line_number: int) -> str:
    """
    Splits a cell at a specific line number (1-based), ensuring correct Yjs types.
    """
    global logger, SERVER_URL, TOKEN, NOTEBOOK_PATH # Add needed globals

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

        # --- Validation ---
        if not (0 <= cell_index < num_cells):
            result_str = f"[Error: Cell index {cell_index} out of bounds (0-{num_cells-1})]"
            logger.warning(result_str)
            if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                 try: await notebook.stop()
                 except Exception: pass
                 notebook = None
            return result_str

        cell_data_read = ycells[cell_index]
        original_source = str(cell_data_read.get("source", "")) # Read as string
        original_cell_type = cell_data_read.get("cell_type", "code") # Default to code if missing? Safer.
        source_lines = original_source.splitlines(True) # Keep line endings

        # Validate line_number (allow splitting *after* last line -> creates empty second cell)
        if not (1 <= line_number <= len(source_lines) + 1):
            result_str = f"[Error: Line number {line_number} is out of range (1-{len(source_lines) + 1})]"
            logger.warning(result_str)
            if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
                 try: await notebook.stop()
                 except Exception: pass
                 notebook = None
            return result_str
        # --- End Validation ---

        new_cell_index = cell_index + 1
        with ydoc.ydoc.transaction():
            # 1. Calculate source parts
            source_part1 = "".join(source_lines[:line_number-1])
            source_part2 = "".join(source_lines[line_number-1:])

            # 2. Update original cell (index `cell_index`)
            cell_data_write = ycells[cell_index]
            source_obj = cell_data_write.get("source")
            # Update source using del slice / insert
            if isinstance(source_obj, YText):
                 existing_content = str(source_obj)
                 if existing_content:
                     del source_obj[0 : len(existing_content)] # Use slice deletion
                 source_obj.insert(0, source_part1) # Use insert
            else:
                 # If source wasn't YText or didn't exist, create it
                 cell_data_write["source"] = YText(source_part1)

            # Clean up outputs/count if it was a code cell
            if original_cell_type == "code":
                # Ensure outputs field is an empty YArray
                outputs_obj = cell_data_write.get("outputs")
                if isinstance(outputs_obj, YArray):
                    outputs_obj.clear() # Clear existing YArray
                else:
                    # If not YArray or doesn't exist, create/replace it
                    # Log warning only if it existed but wasn't YArray
                    if outputs_obj is not None:
                         logger.warning(f"Outputs field in split cell {cell_index} was not YArray ({type(outputs_obj)}). Replacing.")
                    cell_data_write["outputs"] = YArray()

                # Ensure execution_count is None
                cell_data_write["execution_count"] = None

            # 3. Prepare the dictionary for the NEW cell with explicit Yjs types
            new_cell_pre_ymap: Dict[str, Any] = {}
            new_cell_pre_ymap["cell_type"] = original_cell_type
            new_cell_pre_ymap["source"] = YText(source_part2) # Create YText directly

            if original_cell_type == "code":
                # Add default code cell metadata and ensure YArray for outputs
                base_code_cell = nbformat.v4.new_code_cell(source="") # Use nbformat just for defaults
                new_cell_pre_ymap["metadata"] = YMap(base_code_cell.metadata)
                new_cell_pre_ymap["outputs"] = YArray() # Explicitly create YArray
                new_cell_pre_ymap["execution_count"] = None
            else: # markdown
                base_md_cell = nbformat.v4.new_markdown_cell(source="")
                new_cell_pre_ymap["metadata"] = YMap(base_md_cell.metadata)
                # Ensure no outputs/execution_count for markdown
                if "outputs" in new_cell_pre_ymap: del new_cell_pre_ymap["outputs"]
                if "execution_count" in new_cell_pre_ymap: del new_cell_pre_ymap["execution_count"]

            # 4. Convert the prepared dictionary to YMap and insert
            logger.info(f"Splitting cell {cell_index}. Inserting new {original_cell_type} cell at index {new_cell_index}")
            ycells.insert(new_cell_index, YMap(new_cell_pre_ymap))

        result_str = f"Cell {cell_index} split at line {line_number}. New cell created at index {new_cell_index}."

        await asyncio.sleep(0.5) # Keep delay after modification
        await notebook.stop()
        notebook = None # Mark as stopped
        return result_str

    except Exception as e:
        logger.error(f"Error in split_cell for index {cell_index} at line {line_number}: {e}", exc_info=True)
        result_str = f"Error splitting cell {cell_index}: {e}"
        # Cleanup happens in finally
        return result_str
    finally:
        # Ensure stop is attempted if client was created and might still be running
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
    global logger, SERVER_URL, TOKEN, NOTEBOOK_PATH # Add needed globals

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
                    # --- Use Slice Deletion ---
                    existing_content = str(source_obj)
                    if existing_content:
                        del source_obj[0 : len(existing_content)] # Corrected delete
                    source_obj.insert(0, new_content) # Keep insert
                    # --- End Correction ---
                else:
                    # If it's not YText (or doesn't exist), replace/create it
                    # Ensure YText is imported: from pycrdt import Text as YText
                    cell_data["source"] = YText(new_content)
            logger.info(f"Updated source for cell index {cell_index}.")
            result_str = f"Source updated for cell at index {cell_index}."
        else:
            result_str = f"[Error: Cell index {cell_index} is out of bounds (0-{len(ycells)-1})]"
            logger.warning(result_str)
            # No need to stop client here if error is just validation before return

        await asyncio.sleep(0.5) # Keep delay after modification
        await notebook.stop()
        notebook = None
        return result_str
    except Exception as e:
        logger.error(f"Error in edit_cell_source for index {cell_index}: {e}", exc_info=True)
        result_str = f"Error editing cell {cell_index}: {e}"
        # Cleanup happens in finally
        return result_str
    finally:
        # Ensure stop is attempted if client was created and might still be running
        if notebook and notebook._NbModelClient__run and not notebook._NbModelClient__run.done():
            try: await notebook.stop()
            except Exception as final_e: logger.error(f"Error stopping notebook in finally (edit_cell_source): {final_e}")

@mcp.tool()
async def get_kernel_variables(wait_seconds: int = 2) -> str:
    """
    Lists variables currently defined in the Jupyter kernel's interactive namespace
    by executing the '%whos' magic command in a temporary cell.

    NOTE: This provides a snapshot of the kernel's state at the time of execution.

    Args:
        wait_seconds: Seconds to wait after starting execution before getting output. Default 2.

    Returns:
        str: The output from the %whos command (a table of variables, types, and info),
             or an error message if execution failed.
    """
    logger.info("Executing get_kernel_variables tool")
    code_content = "%whos"  # IPython magic command to list variables
    cell_output = "[Error: Failed to retrieve output for variable listing]" # Default error
    cell_index: int | None = None

    try:
        # 1. Add cell
        add_result = await add_code_cell(code_content)
        cell_index = _parse_index_from_message(add_result) # Assumes _parse_index_from_message exists
        if cell_index is None:
            logger.error(f"Failed to add cell for variable listing. Result: {add_result}")
            return f"[Error adding cell for variable listing: {add_result}]"
        logger.info(f"Variable listing cell added at index {cell_index}.")

        # 2. Execute cell
        exec_result = await execute_cell(cell_index)
        if "Error" in exec_result:
            logger.error(f"Failed to start execution for variable listing cell {cell_index}. Result: {exec_result}")
            # Try to delete the cell anyway before returning error
            try:
                logger.warning(f"Attempting cleanup delete for cell {cell_index} after execution error.")
                await delete_cell(cell_index)
            except Exception as del_err:
                 logger.error(f"Cleanup delete failed for cell {cell_index}: {del_err}")
            return f"[Error starting execution for variable listing: {exec_result}]"
        logger.info(f"Execution started for variable listing cell {cell_index}. Waiting {wait_seconds}s...")

        # 3. Wait
        await asyncio.sleep(wait_seconds)
        logger.info(f"Finished waiting for variable listing cell {cell_index}.")

        # 4. Get Output (use wait_seconds=0 in get_cell_output as we already waited)
        cell_output = await get_cell_output(cell_index, wait_seconds=0)
        logger.info(f"Output received for variable listing cell {cell_index}.")

        # 5. Delete Cell (cleanup - best effort in finally block)
        # Deletion moved to finally block for robustness

        # 6. Return kernel output
        return cell_output

    except Exception as e:
        logger.error(f"Error during get_kernel_variables orchestration: {e}", exc_info=True)
        # Error message returned, cleanup happens in finally
        return f"[Error listing kernel variables: {e}]"

    finally:
        # Ensure cleanup happens even if errors occur after cell creation
        if cell_index is not None:
            try:
                logger.info(f"Attempting cleanup delete for variable listing cell {cell_index} in finally block.")
                delete_result = await delete_cell(cell_index)
                logger.info(f"Deletion result for cell {cell_index} in finally: {delete_result}")
            except Exception as final_del_e:
                logger.error(f"Cleanup delete failed for cell {cell_index} in finally: {final_del_e}")


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
                   
                 
# --- Main async function ---
async def main():
    global kernel, logger, mcp # Ensure access to mcp instance

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.info(f"Starting Jupyter MCP Server main async function...")
    logger.info(f"Target notebook: {NOTEBOOK_PATH} on {SERVER_URL} [Log Level: {log_level}]")

    # Ensure kernel is started (assuming sync start before main)
    if not kernel or not kernel.is_alive():
         logger.error("Kernel object not initialized or not alive before starting server.")
         if not kernel: return # Exit if no kernel object
         # Optionally try to start kernel here if needed
         try:
             logger.info("Attempting kernel start within main...")
             kernel.start()
             if not kernel.is_alive():
                 logger.error("Kernel failed to start within main.")
                 return # Exit if kernel start fails
             logger.info("Kernel started successfully within main.")
         except Exception as start_err:
             logger.error(f"Error starting kernel within main: {start_err}", exc_info=True)
             return

    # No monitor task needed now

    try:
        logger.info("Starting MCP server async stdio run...")
        await mcp.run_stdio_async()
        logger.info("MCP server async stdio run finished.")
    # --- Add specific ExceptionGroup handling ---
    except ExceptionGroup as eg:
        logger.error(f"Caught ExceptionGroup in main run: {eg}", exc_info=False) # Log the group message
        for i, exc in enumerate(eg.exceptions):
             # Log each sub-exception WITH its traceback
             logger.error(f"--- Sub-exception {i+1} ---", exc_info=exc)
    # --- End Add ---
    except Exception as e: # Keep general handler
        logger.error(f"Caught general exception in main run: {e}", exc_info=True)
    finally:
        # --- Make SURE this uses kernel.stop(shutdown_kernel=False) ---
        logger.info("Starting cleanup...")
        if kernel and kernel.is_alive():
             logger.info("Stopping kernel client connection (leaving kernel process running)...")
             try:
                 kernel.stop(shutdown_kernel=False) # Ensure this is the call being made
                 logger.info("Kernel client connection stopped.")
             except AttributeError as ae:
                  logger.error(f"AttributeError during kernel stop: {ae}. Method missing?")
             except Exception as stop_e:
                  logger.error(f"Error stopping kernel client connection: {stop_e}", exc_info=True)
        else:
             logger.info("Kernel client already stopped or not started.")
        logger.info("Cleanup finished.")

# --- Entry point ---
if __name__ == "__main__":
    # Any purely synchronous setup can go here
    # e.g., basic logging config that doesn't need the loop
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Initialize kernel synchronously here if possible/needed
    try:
        logger.info(f"Initializing KernelClient for {SERVER_URL}...")
        kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
        kernel.start() # Start it synchronously
        logger.info("KernelClient started synchronously.")
    except Exception as e:
        logger.error(f"Failed to initialize KernelClient at startup: {e}", exc_info=True)
        kernel = None # Ensure kernel is None if start fails

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
        # This catches errors during asyncio.run itself OR reflects the unhandled ones from main
        logger.error(f"Unhandled exception at top level: {e}", exc_info=True)
