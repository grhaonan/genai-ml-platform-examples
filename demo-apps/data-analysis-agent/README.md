# SRE Helper Agent Setup

This guide explains how to set up an AI-powered SRE helper agent that
- Connects to ClickHouse for real-time log/trace query, detects errors/performance issues live
- Develops an autonomous AI agent using open-source framework (Strands Agent SDK and MCP) that automatically detects and analyzes system issues and errors, sends notifications when problems are identified, and executes corrective actions through REST API calls.
- Deploys the agent to AWS Bedrock AgentCore with production-grade reliability. security and observability

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SRE Helper    │    │   AgentCore     │    │   AgentCore     │    │   Lambda +      │
│   Agent         │◄──►│   Runtime       │◄──►│   Gateway       │◄──►│   SNS Topic     │
│                 │    │                 │    │   (OAuth)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   ClickHouse    │
│   MCP Server    │
│                 │
└─────────────────┘
```

## Components

1. **ClickHouse MCP Server**: Provides database query tools (stdio connection) - hosted within AgentCore Runtime
2. **AgentCore Gateway**: Exposes Lambda function as MCP tool (HTTP connection)
3. **Notification Lambda Function**: Sends analysis summaries to SNS topic
4. **Private REST API Lambda Function**: Run private REST API to fix issue
5. **SNS Topic**: Delivers notifications to subscribers

## Deployment Steps

### Prerequisites
1. Python 3.10+
2. AWS account and credentials configured
3. AWS CLI configured with appropriate permissions
4. AWS Account with Amazon Bedrock, Amazon Bedrock Agentcore, Lambda, SNS access
   
### 1. Deploy the Gateway and Lambda

```bash
# Make sure you're in the project directory
cd genai-ml-platform-examples/demo-apps/data-analysis-agent

# Activate virtual environment
python3 -m venv .venv
source .venv/bin/activate 

# Install the dependency 
pip install -r requirements.txt 

# Deploy Gateway, Lambda, and SNS topic
python deploy_clickhouse_gateway.py
```

This will:
- Create an SNS topic for notifications
- Deploy the Lambda function with SNS permissions
- Create an AgentCore Gateway with OAuth authentication
- Save configuration to `clickhouse_gateway_config.json`

### 2. Subscribe to SNS Topic

```bash
# Subscribe your email to receive notifications
aws sns subscribe \
  --topic-arn $(jq -r '.sns_topic_arn' clickhouse_gateway_config.json) \
  --protocol email \
  --notification-endpoint your-email@example.com

# Confirm the subscription via email
```

### 3. Deploy/Update AgentCore Runtime

```bash
# Configure the agent (if not already done)
agentcore configure -e agentcore_clickhouse_analyst.py

# Deploy or update the runtime
agentcore launch
```

The agent will automatically detect the Gateway configuration and include the notification tool.

### 4. Test the Integration

```bash
# Test the agent with a query that triggers notification
agentcore invoke '{"prompt": "there is a outage happened in recommedationservice, summerize the potantial root cause and send me a notification"}'
```

## Gateway Configuration

The `clickhouse_gateway_config.json` file contains:

```json
{
  "sns_topic_arn": "arn:aws:sns:...",
  "lambda_function_arn": "arn:aws:lambda:...",
  "gateway_id": "...",
  "gateway_url": "https://...",
  "cognito_user_pool_id": "...",
  "cognito_client_id": "...",
  "cognito_client_secret": "...",
  "discovery_url": "https://...",
  "access_token": "..."
}
```

## Available Tool

The Gateway exposes one tool to the agent:

### `send_analysis_summary`

Sends a ClickHouse analysis summary to the SNS topic.

**Parameters:**
- `summary` (required): The analysis summary text
- `table_name` (optional): Name of the table analyzed
- `analysis_type` (optional): Type of analysis performed

**Example usage in agent:**
```python
# The agent can call this tool automatically
"After analyzing the data, send a summary notification with the key findings"
```

## How It Works

1. **Agent receives query**: User asks for analysis with notification
2. **ClickHouse analysis**: Agent uses ClickHouse MCP tools to query data
3. **Generate summary**: Agent creates a summary of findings
4. **Call Gateway tool**: Agent invokes `send_analysis_summary` via Gateway
5. **Lambda execution**: Gateway forwards request to Lambda function
6. **SNS notification**: Lambda publishes message to SNS topic
7. **Email delivery**: SNS sends email to subscribers

## Troubleshooting

### Gateway not found
- Ensure `clickhouse_gateway_config.json` exists
- Redeploy using `python deploy_clickhouse_gateway.py`

### Authentication errors
- Token may have expired, redeploy to get fresh credentials
- Check Cognito configuration in AWS Console

### Lambda errors
- Check Lambda logs: `aws logs tail /aws/lambda/clickhouse-notification-lambda --follow`
- Verify SNS topic permissions

### No notifications received
- Confirm SNS subscription via email
- Check SNS topic has correct permissions
- Verify Lambda has SNS publish permissions

## Cleanup

To remove all resources:

```bash
python cleanup.py
```

## References

- [AgentCore Gateway Documentation](https://aws.github.io/bedrock-agentcore-starter-toolkit/examples/gateway-integration.md)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Strands SDK](https://strandsagents.com/)
