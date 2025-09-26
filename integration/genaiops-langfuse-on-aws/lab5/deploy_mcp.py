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
agent_name = "lab5_strands_agent_mcp_example"

# Create execution role with pricing permissions
import boto3
import json
from datetime import datetime

iam = boto3.client('iam')
sts = boto3.client('sts')
account_id = sts.get_caller_identity()['Account']

role_name = f"BedrockAgentCore-MCP-ExecutionRole-{region}"
execution_role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

# Check if role exists, create if not
try:
    iam.get_role(RoleName=role_name)
    print(f"✅ Using existing role: {role_name}")
except iam.exceptions.NoSuchEntityException:
    # Create role with pricing permissions
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Official AgentCore execution role policy with pricing permissions added
    execution_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ECRImageAccess",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer"
                ],
                "Resource": [f"arn:aws:ecr:{region}:{account_id}:repository/*"]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:DescribeLogStreams",
                    "logs:CreateLogGroup"
                ],
                "Resource": [f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"]
            },
            {
                "Effect": "Allow",
                "Action": ["logs:DescribeLogGroups"],
                "Resource": [f"arn:aws:logs:{region}:{account_id}:log-group:*"]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": [f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"]
            },
            {
                "Sid": "ECRTokenAccess",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets"
                ],
                "Resource": ["*"]
            },
            {
                "Effect": "Allow",
                "Resource": "*",
                "Action": "cloudwatch:PutMetricData",
                "Condition": {
                    "StringEquals": {
                        "cloudwatch:namespace": "bedrock-agentcore"
                    }
                }
            },
            {
                "Sid": "GetAgentAccessToken",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId"
                ],
                "Resource": [
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default/workload-identity/{agent_name}-*"
                ]
            },
            {
                "Sid": "BedrockModelInvocation",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream"
                ],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:{region}:{account_id}:*"
                ]
            },
            {
                "Sid": "AWSPricingAccess",
                "Effect": "Allow",
                "Action": [
                    "pricing:DescribeServices",
                    "pricing:GetAttributeValues",
                    "pricing:GetProducts"
                ],
                "Resource": "*"
            }
        ]
    }
    
    # Create role
    iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Execution role for Bedrock AgentCore with AWS Pricing permissions"
    )
    
    # Attach execution policy with pricing permissions
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName='BedrockAgentCoreExecutionPolicy',
        PolicyDocument=json.dumps(execution_policy)
    )
    
    print(f"✅ Created role with pricing permissions: {role_name}")

# Get absolute path to agent file
script_dir = Path(__file__).parent
agent_file = script_dir / "lab5_agent_mcp.py"

# Configure
print(f"Configuring agent: {agent_name}")
print(f"Agent file: {agent_file}")
configure_response = agentcore_runtime.configure(
    entrypoint=str(agent_file),
    execution_role=execution_role_arn,
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
        'CMD ["opentelemetry-instrument", "python", "-m", "lab5_agent_mcp"]',
        'CMD ["python", "-m", "lab5_agent_mcp"]'
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
    "PATH": "/usr/local/bin:/usr/bin:/bin",
}

# Deploy
print("Deploying to AgentCore...")
launch_result = agentcore_runtime.launch(env_vars=env_vars, auto_update_on_conflict=True)
print(f"Agent deployed: {launch_result.agent_arn}")

print(f"✅ Agent deployed with pricing permissions: {launch_result.agent_arn}")

print("Use invoke.py to test the agent")
