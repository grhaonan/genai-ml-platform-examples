#!/usr/bin/env python3
"""
Start MCP server with proper environment configuration
"""

import os
import subprocess
import sys
from clickhouse_config import get_mcp_env

def start_mcp_server():
    """Start the MCP server with proper environment variables."""
    # Get environment variables from config
    mcp_env = get_mcp_env()
    
    # Set environment variables
    env = os.environ.copy()
    env.update(mcp_env)
    
    # Start MCP server
    cmd = ["/opt/homebrew/bin/uvx", "mcp-clickhouse"]
    
    print("🚀 Starting MCP server with environment:")
    for key, value in mcp_env.items():
        if "PASSWORD" in key:
            print(f"   {key}=***")
        else:
            print(f"   {key}={value}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ MCP server failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 MCP server stopped")

if __name__ == "__main__":
    start_mcp_server()
