from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
import uvicorn
from app import run_memory_chat
import asyncio


app = FastAPI()

# ✅ MCP-exposed endpoint
@app.get("/api/mcp-tool", operation_id="mcp_tool")
async def mcp_tool():
          await run_memory_chat()
          return {"message": "This endpoint is exposed as an MCP tool."}

# ✅ MCP server setup (expose only mcp_tool)
mcp = FastApiMCP(
    app,
    name="Selective MCP Server",
    description="Exposes only the mcp_tool endpoint as an MCP tool.",
    base_url="http://localhost:8000",
    include_operations=["mcp_tool"],
)

mcp.mount()  # Mount at /mcp

# ✅ Middleware to restrict MCP access to localhost only
@app.middleware("http")
async def restrict_mcp_access(request: Request, call_next):
    if request.url.path.startswith("/mcp"):
        client_ip = request.client.host
        if client_ip not in ("127.0.0.1", "::1"):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied: MCP endpoint is restricted to localhost."},
            )
    return await call_next(request)

# ✅ Run with Python directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
