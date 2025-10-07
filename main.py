from typing import Any
from mcp.server import MCPServer
# Importing the MCPServer class from mcp.server
# Importing FastMCP for a more efficient server implementation
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Smithery MCP Server Python templates" )

@mcp.command("hello")
def hello(name: str) -> str:
    """
    A simple command that returns a greeting message.
    
    :param name: The name of the person to greet.
    :return: A greeting message.
    """
    return f"Hello, {name}!"


@mcp.command("add")              
def add(a: int, b: int) -> int:
    """
    A command that adds two integers.
    
    :param a: The first integer.
    :param b: The second integer.
    :return: The sum of the two integers.
    """
    return a + b

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')