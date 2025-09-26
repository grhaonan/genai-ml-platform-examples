import boto3
import json
import base64
from pathlib import Path
import uuid

def parse_response(response):
    """Parse agent response and return clean text"""
    try:
        # AgentCore returns response as bytes in 'response' field
        if 'response' in response:
            response_bytes = response['response']
            if hasattr(response_bytes, 'read'):
                # If it's a stream, read all content
                return response_bytes.read().decode('utf-8')
            elif isinstance(response_bytes, bytes):
                # If it's bytes, decode directly
                return response_bytes.decode('utf-8')
            else:
                # If it's already a string
                return str(response_bytes)
        return "No response content found"
    except Exception as e:
        return f"Error parsing response: {e}"

def invoke_agent(agent_arn, payload_data, session_id=None):
    """Invoke agent with payload data and optional session ID"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    try:
        response = data_client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=session_id,
            payload=json.dumps(payload_data).encode()
        )
        return parse_response(response)
    except Exception as e:
        return f"Error invoking agent: {e}"

# Setup clients
control_client = boto3.client('bedrock-agentcore-control')
data_client = boto3.client('bedrock-agentcore')

# Find agent
response = control_client.list_agent_runtimes()
agents = response.get('agentRuntimes', [])
print(f"Found {len(agents)} agents")
lab5_agent = next((a for a in agents if 'lab5' in a['agentRuntimeName']), None)
# lab5_agent_tool = next((a for a in agents if 'lab5_strands_agent_custom_tool_example' in a['agentRuntimeName']), None)
# lab5_agent_mcp = next((a for a in agents if 'lab5_strands_agent_mcp_example' in a['agentRuntimeName']), None)

if not lab5_agent:
    print("No lab5 agent found. Available agents:")
    for agent in agents:
        print(f"  - {agent['agentRuntimeName']}: {agent['agentRuntimeArn']}")
    exit(1)

agent_arn = lab5_agent['agentRuntimeArn']
print(f"Found agent: {agent_arn}")

# Test 1: Weather question
print("\n=== Test 1: Weather ===\n")
invoke_response = invoke_agent(agent_arn, {"prompt": "How is the weather outside?"})
print(invoke_response)

# Test 2: Math calculation
print("\n=== Test 2: Math ===\n")
invoke_response = invoke_agent(agent_arn, {"prompt": "How much is 2X5?"})
print(invoke_response)

# # Test 4: General question
# print("\n=== Test 4: General Question ===")
# response = invoke_agent(agent_arn, {"prompt": "What is AWS Lambda used for? Keep it brief."})
# print(response)

# # Test 5: Image analysis (Method 1 - Multi-modal payload)
# print("\n=== Test 5: Image Analysis ===")
# image_path = "../lab4/image/architecture.png"
# if Path(image_path).exists():
#     with open(image_path, "rb") as f:
#         image_data = base64.b64encode(f.read()).decode('utf-8')
    
#     payload = {
#         "prompt": "Describe what you see in this image",
#         "media": {
#             "type": "image",
#             "format": "png",
#             "data": image_data
#         }
#     }
    
#     response = invoke_agent(agent_arn, payload)
#     print(response)
# else:
#     print(f"Image file not found at {image_path}. Please provide a valid image path.")

# # Test 6: AWS Service Health
# print("\n=== Test 6: AWS Service Health ===")
# response = invoke_agent(agent_arn, {"prompt": "Check the AWS service health for any disruptions in the us-east-1 region."})
# print(response)

# # Test 7: MCP Interaction: AWS Documentation and Pricing
# print("\n=== Test 7: MCP Interaction ===")
# response = invoke_agent(agent_arn, {"prompt": "What is the pricing model for Amazon S3?"})
# print(response)