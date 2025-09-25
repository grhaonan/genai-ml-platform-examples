import boto3
import json
from pathlib import Path

# Use control client to list agents and data client to invoke
control_client = boto3.client('bedrock-agentcore-control')
data_client = boto3.client('bedrock-agentcore')

# Get agent ARN from deploy output or list agents
response = control_client.list_agent_runtimes()
agents = response.get('agentRuntimes', [])
lab5_agent = next((a for a in agents if 'lab5' in a['agentRuntimeName']), None)

if not lab5_agent:
    print("No lab5 agent found. Available agents:")
    for agent in agents:
        print(f"  - {agent['agentRuntimeName']}: {agent['agentRuntimeArn']}")
    exit(1)

agent_arn = lab5_agent['agentRuntimeArn']
print(f"Found agent: {agent_arn}")

# Test with simple calculation first
print("\nTesting with calculation...")
response = data_client.invoke_agent_runtime(
    agentRuntimeArn=agent_arn,
    qualifier="DEFAULT",
    payload=json.dumps({"prompt": "What is 25 * 47 + 123?"})
)

# Parse response
if "text/event-stream" in response.get("contentType", ""):
    content = []
    for line in response["response"].iter_lines(chunk_size=1):
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                content.append(line[6:])
    response_text = "\n".join(content)
else:
    events = []
    for event in response.get("response", []):
        events.append(event)
    response_text = events[0].decode("utf-8")
print(f"Response: {response_text}")

# Test with general question
print("\nTesting with general question...")
response = data_client.invoke_agent_runtime(
    agentRuntimeArn=agent_arn,
    qualifier="DEFAULT",
    payload=json.dumps({"prompt": "Explain what AWS S3 is used for"})
)

if "text/event-stream" in response.get("contentType", ""):
    content = []
    for line in response["response"].iter_lines(chunk_size=1):
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                content.append(line[6:])
    response_text = "\n".join(content)
else:
    events = []
    for event in response.get("response", []):
        events.append(event)
    response_text = events[0].decode("utf-8")
print(f"Response: {response_text}")