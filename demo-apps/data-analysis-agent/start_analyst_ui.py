#!/usr/bin/env python3
"""
Startup script for ClickHouse Data Analyst UI
"""

import sys
import time
import webbrowser

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import boto3
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install fastapi uvicorn boto3")
        return False

def main():
    """Main startup function."""
    print("🚀 Starting ClickHouse Data Analyst UI (AgentCore Edition)")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ Dependencies OK")
    print(f"\n{'='*50}")
    print("🌐 Starting Web Server...")
    print("   URL: http://localhost:8001")
    print("   Press Ctrl+C to stop")
    print(f"{'='*50}")
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8001")
            print("🌐 Browser opened")
        except:
            print("   Please open http://localhost:8001 manually")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    try:
        import uvicorn
        uvicorn.run("analyst_web_server:app", host="0.0.0.0", port=8001, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()