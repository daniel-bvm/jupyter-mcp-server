podman build -t jupyter_mcp_server . --pull

podman run --rm -it -p 34587:34587 -p 8000:80 \
    -e ETERNALAI_MCP_PROXY_URL="$ETERNALAI_MCP_PROXY_URL" \
    -e DUNE_API_KEY="$DUNE_API_KEY" \
    -e LLM_BASE_URL="$LLM_BASE_URL" \
    -e LLM_API_KEY="$LLM_API_KEY" \
    -e LLM_MODEL_ID="$LLM_MODEL_ID" \
    jupyter_mcp_server
