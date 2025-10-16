#!/usr/bin/env python3
"""
ClickHouse Cloud Configuration
Configuration settings for connecting to ClickHouse Cloud instance.
"""

# ClickHouse Cloud Configuration
CLICKHOUSE_CONFIG = {
    "host": "",  # Just hostname, no https://
    "port": 8443,
    "user": "default", 
    "password": "",
    "database": "default",
    "secure": True,  # Use HTTPS/TLS
    "verify": True   # Verify SSL certificates
}

# MCP Server Environment Variables
MCP_ENV = {
    "CLICKHOUSE_HOST": CLICKHOUSE_CONFIG["host"],
    "CLICKHOUSE_PORT": str(CLICKHOUSE_CONFIG["port"]),
    "CLICKHOUSE_USER": CLICKHOUSE_CONFIG["user"],
    "CLICKHOUSE_PASSWORD": CLICKHOUSE_CONFIG["password"],
    "CLICKHOUSE_DATABASE": CLICKHOUSE_CONFIG["database"],
    "CLICKHOUSE_SECURE": "true",  # Enable secure connection
    "CLICKHOUSE_VERIFY": "true"   # Verify SSL certificates
}

def get_clickhouse_config():
    """Get ClickHouse configuration dictionary."""
    return CLICKHOUSE_CONFIG.copy()

def get_mcp_env():
    """Get MCP environment variables dictionary."""
    return MCP_ENV.copy()

def test_connection():
    """Test connection to ClickHouse Cloud."""
    import urllib.request
    import urllib.error
    import base64
    
    try:
        # Create authentication header
        credentials = f"{CLICKHOUSE_CONFIG['user']}:{CLICKHOUSE_CONFIG['password']}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        # Create request
        url = f"https://{CLICKHOUSE_CONFIG['host']}:{CLICKHOUSE_CONFIG['port']}"
        req = urllib.request.Request(url, data=b"SELECT 1")
        req.add_header("Authorization", f"Basic {encoded_credentials}")
        
        # Test connection
        with urllib.request.urlopen(req, timeout=10) as response:
            result = response.read().decode()
            print(f"✅ ClickHouse Cloud connection successful: {result.strip()}")
            return True
            
    except Exception as e:
        print(f"❌ ClickHouse Cloud connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing ClickHouse Cloud Connection...")
    print(f"Host: {CLICKHOUSE_CONFIG['host']}")
    print(f"Port: {CLICKHOUSE_CONFIG['port']}")
    print(f"User: {CLICKHOUSE_CONFIG['user']}")
    print(f"Database: {CLICKHOUSE_CONFIG['database']}")
    print()
    
    test_connection()