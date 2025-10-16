#!/usr/bin/env python3
"""
Web UI Server for ClickHouse Data Analyst Agent
FastAPI-based web interface for data analysis and visualization.
"""

import json
import logging
import boto3
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ClickHouse Data Analyst UI",
    description="Web interface for ClickHouse data analysis and insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AgentCore configuration
AGENT_RUNTIME_ARN = ""
bedrock_client = None
connection_status = {"connected": False, "error": None}

# Pydantic models
class AnalysisRequest(BaseModel):
    query: str

class TableAnalysisRequest(BaseModel):
    table_name: str
    analysis_type: str  # 'full', 'numeric', 'text', 'insights'
    column_name: Optional[str] = None

class ComparisonRequest(BaseModel):
    table1: str
    table2: str

class QueryRequest(BaseModel):
    query: str

def generate_session_id():
    """Generate a unique session ID for AgentCore."""
    import random
    import string
    return 'session-' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=40))

def invoke_agentcore(prompt: str) -> str:
    """Invoke AgentCore runtime with a prompt."""
    session_id = generate_session_id()
    
    # AgentCore expects JSON payload with "prompt" key
    payload = json.dumps({"prompt": prompt})
    
    try:
        response = bedrock_client.invoke_agent_runtime(
            runtimeSessionId=session_id,
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            qualifier="DEFAULT",
            payload=payload.encode('utf-8')
        )
        
        # Handle streaming response
        if 'response' in response:
            response_stream = response['response']
            result = ""
            for event in response_stream:
                if 'chunk' in event:
                    chunk_data = event['chunk'].get('bytes', b'')
                    result += chunk_data.decode('utf-8')
            
            if result:
                # Try to parse JSON response
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and 'response' in parsed:
                        return parsed['response']
                    return result
                except json.JSONDecodeError:
                    return result
            
            return "No response from agent"
        
        return "No response from agent"
    
    except Exception as e:
        error_msg = str(e)
        if 'ThrottledException' in error_msg or 'Rate exceeded' in error_msg:
            return "⚠️ AgentCore Memory is currently rate-limited. Please wait a few seconds and try again."
        raise

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize AWS Bedrock client on startup."""
    global bedrock_client, connection_status
    try:
        bedrock_client = boto3.client('bedrock-agentcore', region_name='us-east-1')
        connection_status = {"connected": True, "error": None}
        logger.info("✅ AWS Bedrock AgentCore client initialized")
    except Exception as e:
        connection_status = {"connected": False, "error": str(e)}
        logger.error(f"❌ Failed to initialize client: {e}")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI."""
    try:
        with open("static/analyst_ui.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI file not found. Please ensure static/analyst_ui.html exists.</h1>")

@app.get("/api/status")
async def get_status():
    """Get connection status."""
    return connection_status

@app.post("/api/analyze")
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze data using natural language query."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        result = invoke_agentcore(request.query)
        
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "analysis_completed",
                "data": {
                    "query": request.query,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
        return {
            "success": True,
            "query": request.query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tables")
async def get_tables():
    """Get list of all tables."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        result = invoke_agentcore("List all tables in the database")
        
        return {
            "success": True,
            "tables": ["sales_data"],
            "analysis": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table/analyze")
async def analyze_table(request: TableAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze a specific table."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        if request.analysis_type == "full":
            prompt = f"Analyze the {request.table_name} table. Show its structure, row count, and sample data."
        elif request.analysis_type == "numeric" and request.column_name:
            prompt = f"Analyze the numeric column {request.column_name} in table {request.table_name}. Show statistics like average, min, max, and count."
        elif request.analysis_type == "text" and request.column_name:
            prompt = f"Analyze the text column {request.column_name} in table {request.table_name}. Show the most frequent values."
        elif request.analysis_type == "insights":
            prompt = f"Provide comprehensive insights about the {request.table_name} table including data patterns, quality, and recommendations."
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type or missing column name")
        
        result = invoke_agentcore(prompt)
        
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "table_analysis_completed",
                "data": {
                    "table_name": request.table_name,
                    "analysis_type": request.analysis_type,
                    "column_name": request.column_name,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
        return {
            "success": True,
            "table_name": request.table_name,
            "analysis_type": request.analysis_type,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table/compare")
async def compare_tables(request: ComparisonRequest, background_tasks: BackgroundTasks):
    """Compare two tables."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        result = invoke_agentcore(f"Compare tables {request.table1} and {request.table2}. Show their structures, row counts, and key differences.")
        
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "table_comparison_completed",
                "data": {
                    "table1": request.table1,
                    "table2": request.table2,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
        return {
            "success": True,
            "table1": request.table1,
            "table2": request.table2,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def execute_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Execute a SQL query."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        result = invoke_agentcore(f"Execute this SQL query: {request.query}")
        
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "query_executed",
                "data": {
                    "query": request.query,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
        return {
            "success": True,
            "query": request.query,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/table/{table_name}/columns")
async def get_table_columns(table_name: str):
    """Get columns for a specific table."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        result = invoke_agentcore(f"Describe the structure of table {table_name}")
        
        return {
            "success": True,
            "table_name": table_name,
            "columns": [],
            "raw_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights/{table_name}")
async def get_table_insights(table_name: str):
    """Get AI-generated insights for a table."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        query = f"Provide comprehensive insights and analysis for the {table_name} table including data quality, patterns, and recommendations"
        result = invoke_agentcore(query)
        
        return {
            "success": True,
            "table_name": table_name,
            "insights": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/info")
async def get_system_info():
    """Get system information."""
    if not bedrock_client or not connection_status["connected"]:
        raise HTTPException(status_code=503, detail="AgentCore not available")
    
    try:
        result = invoke_agentcore("Get the database version and list all available databases")
        
        return {
            "success": True,
            "system_info": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")