import base64
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from dotenv import load_dotenv
from strands import Agent, tool
from strands.telemetry import StrandsTelemetry
from strands_tools import image_reader

# Import MCP related modules
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

load_dotenv()

# Langfuse configuration (same as working version)
langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_AUTH = base64.b64encode(
    f"{langfuse_public_key}:{langfuse_secret_key}".encode()
).decode()
os.environ["LANGFUSE_PROJECT_NAME"] = "my_llm_project"
os.environ["DISABLE_ADOT_OBSERVABILITY"] = "true"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com") + "/api/public/otel"
)
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

for k in [
    "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
    "AGENT_OBSERVABILITY_ENABLED",
    "OTEL_PYTHON_DISTRO",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_PYTHON_CONFIGURATOR",
    "OTEL_PYTHON_EXCLUDED_URLS",
]:
    os.environ.pop(k, None)


app = BedrockAgentCoreApp()
MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Use cross-region inference



# Using the @tool python decorator, creating a custom tool that uses the AWS Health Dashboard RSS feed URL to check AWS service status
@tool
def aws_health_status_checker(
    region: str = "us-east-1", service_name: Optional[str] = None
) -> Dict:
    """
    Check the current operational status of AWS services in a specific region using the public RSS feed.

    Args:
        region: AWS region to check (e.g., us-east-1, us-west-2)
        service_name: Optional specific service to check (e.g., ec2, s3, lambda)
                     If not provided, will return status for all services

    Returns:
        Dictionary containing service health information
    """
    try:
        # AWS Health Dashboard RSS feed URL
        rss_url = "https://status.aws.amazon.com/rss/all.rss"

        # Get the RSS feed
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()

        # Parse the XML
        root = ET.fromstring(response.content)

        # Find all items (service events)
        items = root.findall(".//item")

        # Filter events by region and service
        events = []
        for item in items:
            title = item.find("title").text if item.find("title") is not None else ""
            description = (
                item.find("description").text
                if item.find("description") is not None
                else ""
            )
            pub_date = (
                item.find("pubDate").text if item.find("pubDate") is not None else ""
            )
            link = item.find("link").text if item.find("link") is not None else ""

            # Check if this event is for the requested region
            if region.lower() in title.lower():
                # If service_name is specified, check if this event is for that service
                if service_name is None or service_name.lower() in title.lower():
                    events.append(
                        {
                            "title": title,
                            "description": description,
                            "published_date": pub_date,
                            "link": link,
                        }
                    )

        if not events:
            return {
                "status": "healthy",
                "message": f"All services in {region} appear to be operating normally"
                if not service_name
                else f"{service_name} in {region} appears to be operating normally",
                "events": [],
            }

        return {
            "status": "service_disruption",
            "message": f"Service disruptions detected in {region}",
            "events": events,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking AWS service status: {str(e)}",
            "events": [],
        }


@app.entrypoint
def lab5_agent(payload):
    user_input = payload.get("prompt")
    print("LAB5: User input:", user_input)

    # Initialize Strands telemetry and setup OTLP exporter
    strands_telemetry = StrandsTelemetry()
    strands_telemetry.setup_otlp_exporter()

    # Initialize MCP clients for AWS services inside the function
    aws_docs_mcp = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx", 
                args=["awslabs.aws-documentation-mcp-server@latest"]
            )
        )
    )

    aws_pricing_mcp = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx", 
                args=["awslabs.aws-pricing-mcp-server@latest"]
            )
        )
    )

    # Use MCP clients within context manager
    with aws_docs_mcp, aws_pricing_mcp:
        # Get available MCP tools
        mcp_tools = aws_docs_mcp.list_tools_sync() + aws_pricing_mcp.list_tools_sync()
        
        agent = Agent(
            system_prompt="""You are an AWS expert assistant. Follow these rules strictly:

1. TOOL USAGE LIMITS:
   - Maximum 3 tool calls per response
   - For pricing questions: Use get_pricing_service_attributes ONCE, then answer completely
   - For documentation: Use search_documentation ONCE with specific keywords
   - NEVER make repetitive calls to the same tool

2. RESPONSE STRATEGY:
   - Provide comprehensive answers based on available tool results
   - If first tool call provides sufficient information, answer immediately
   - Do not search for additional information unless critically missing

3. EFFICIENCY REQUIREMENTS:
   - Combine multiple concepts in single tool calls
   - Use your existing knowledge to supplement tool results
   - Stop tool usage once you have enough information to answer

Answer user questions thoroughly but efficiently.""",
            tools=mcp_tools,
            model=MODEL,
            name="lab5_strands_agent_mcp_example",
            trace_attributes={
                "session.id": "aws-mcp-agent-demo-session",
                "user.id": "example-user@example.com",
                "langfuse.tags": [
                    "AWS-Strands-Agent",
                    "MCP-Tools",
                ],
                "metadata": {
                    "environment": "development",
                    "version": "1.0.0",
                    "query_type": "aws_assistance"
                }
            }
        )

        response = agent(user_input)
        response_text = response.message["content"][0]["text"]

        print(f"LAB5: Response preview: {response_text[:100]}...")

        return response_text


if __name__ == "__main__":
    app.run()
