import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

load_dotenv()

# Setup
boto_session = Session()
region = boto_session.region_name
agentcore_runtime = Runtime()
# agent_name = "lab5_strands_agent"
agent_name="lab5_strands_agent_custom_tool_example"

# Get absolute path to agent file
script_dir = Path(__file__).parent
# agent_file = script_dir / "lab5_agent.py"
agent_file = script_dir / "lab5_agent_tools.py"

# Configure
print(f"Configuring agent: {agent_name}")
print(f"Agent file: {agent_file}")
configure_response = agentcore_runtime.configure(
    entrypoint=str(agent_file),
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file=str(script_dir / "requirements.txt"),
    region=region,
    agent_name=agent_name,
)

# Modify Dockerfile to remove opentelemetry-instrument
# The Dockerfile is generated in the current working directory
dockerfile_path = Path("Dockerfile")
if dockerfile_path.exists():
    with open(dockerfile_path, 'r') as f:
        dockerfile_content = f.read()
    
    # Replace the CMD instruction
    dockerfile_content = dockerfile_content.replace(
        'CMD ["opentelemetry-instrument", "python", "-m", "lab5_agent_tools"]',
        'CMD ["python", "-m", "lab5_agent_tools"]'
    )
    
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile modified to disable opentelemetry-instrument")
else:
    print("⚠️ Dockerfile not found in current directory")

# Prepare environment variables
env_vars = {
    "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST"),
    "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY"),
    "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY"),
    "PYTHONUNBUFFERED": "1",
}

# Deploy
print("Deploying to AgentCore...")
launch_result = agentcore_runtime.launch(env_vars=env_vars, auto_update_on_conflict=True)
print(f"Agent deployed: {launch_result.agent_arn}")
print("Use invoke.py to test the agent")
