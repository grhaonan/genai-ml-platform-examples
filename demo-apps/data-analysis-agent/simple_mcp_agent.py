#!/usr/bin/env python3
"""
Simple MCP Agent following exact Strands documentation pattern
"""

from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient
from clickhouse_config import get_mcp_env

def create_clickhouse_agent():
    """Create ClickHouse agent with MCP tools using proper context manager."""
    mcp_env = get_mcp_env()
    
    # Create MCP client exactly as shown in documentation
    mcp_client = MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["mcp-clickhouse"],
            env=mcp_env
        )
    ))
    
    return mcp_client

# Usage example
async def main():
    mcp_client = create_clickhouse_agent()
    
    # Use within context manager as required
    with mcp_client:
        tools = mcp_client.list_tools_sync()
        agent = Agent(tools=tools)
        
        # Test the agent
        response = agent("What tables are available?")
        print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
