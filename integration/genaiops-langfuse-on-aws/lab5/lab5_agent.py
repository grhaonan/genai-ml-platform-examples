import os
import base64
from dotenv import load_dotenv
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands_tools import image_reader
from strands.telemetry import StrandsTelemetry

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
    os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    + "/api/public/otel"
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
MODEL = "global.anthropic.claude-sonnet-4-20250514-v1:0"


@app.entrypoint
def lab5_agent(payload):
    user_input = payload.get("prompt")
    print("LAB5: User input:", user_input)

    # Initialize Strands telemetry and setup OTLP exporter
    strands_telemetry = StrandsTelemetry()
    strands_telemetry.setup_otlp_exporter()

    agent = Agent(
        tools=[image_reader],
        model=MODEL,
        name="lab5-agentcore-agent",
        trace_attributes={
            "session.id": "lab5-agentcore-session",
            "user.id": "lab5-user",
            "langfuse.tags": ["Lab5", "AgentCore", "Built-In-Tool"],
            "metadata": {
                "environment": "agentcore",
                "version": "1.0.0",
                "lab": "lab5",
            },
        },
    )

    response = agent(user_input)
    response_text = response.message["content"][0]["text"]

    print(f"LAB5: Response preview: {response_text[:100]}...")

    return response_text


if __name__ == "__main__":
    app.run()
