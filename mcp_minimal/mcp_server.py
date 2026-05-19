from mcp.server.fastmcp import FastMCP

# Initialize the server
mcp = FastMCP("MyMathServer")

# Define a tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")