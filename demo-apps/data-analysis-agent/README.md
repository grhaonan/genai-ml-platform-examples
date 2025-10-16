# ClickHouse Data Analyst with Strands SDK

A powerful ClickHouse database analyst built with the [Strands SDK](https://strandsagents.com/) that connects to ClickHouse Cloud through the [Model Context Protocol (MCP)](https://github.com/ClickHouse/mcp-clickhouse). Now deployed on [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) for scalable, serverless execution.

## 🌟 Features

- 🔌 **MCP Integration**: Seamless connection to ClickHouse Cloud via MCP server
- 🤖 **Strands Data Analyst**: Built with official Strands SDK patterns
- ☁️ **Bedrock AgentCore**: Deployed on AWS for serverless, scalable execution
- 💬 **Conversational Interface**: Natural language database interactions
- 🛠️ **Comprehensive Analysis**: Full database analysis capabilities
- 📊 **Rich Formatting**: Beautiful query result display with insights
- 🌐 **Web UI**: Interactive web interface for data analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   Data Analyst  │    │   ClickHouse    │
│   (FastAPI)     │◄──►│   Agent         │◄──►│   Cloud         │
│                 │    │ (Bedrock        │    │   (MCP)         │
│                 │    │  AgentCore)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Configure ClickHouse Connection

First, update `clickhouse_config.py` with your ClickHouse Cloud credentials:

```python
CLICKHOUSE_CONFIG = {
    "host": "your-host.clickhouse.cloud",
    "port": 8443,
    "user": "default", 
    "password": "your-password",
    "database": "default",
    "secure": True,
    "verify": True
}
```

### 2. Deploy to Amazon Bedrock AgentCore

#### Prerequisites
- AWS CLI configured (`aws configure`)
- Python 3.10 or newer
- Amazon Bedrock model access enabled for Claude 3.7 Sonnet

#### Installation & Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

#### Configure and Deploy
```bash
# Configure the agent for AgentCore deployment
agentcore configure -e agentcore_clickhouse_analyst.py

# Deploy to AgentCore (creates runtime, memory, and observability)
agentcore launch

# Test the deployed agent
agentcore invoke '{"prompt": "Analyze my ClickHouse database"}'
```

#### Monitor Deployment
```bash
# Check deployment status
agentcore status

# View logs
agentcore logs

# Access CloudWatch dashboard for observability
# URL provided in agentcore status output
```

The web interface will be available at: http://localhost:8001

## 📁 Project Structure

```
clickhouse-mcp-agent/
├── clickhouse_config.py         # ClickHouse Cloud configuration
├── data_analyst_agent.py        # Main data analyst agent
├── simple_mcp_agent.py          # MCP client helper
├── analyst_web_server.py        # FastAPI web server
├── start_analyst_ui.py          # Application startup script
├── start_mcp_server.py          # MCP server startup script
├── static/                      # Web UI files
│   ├── analyst_ui.html          # Main UI template
│   └── analyst_ui.js            # Frontend JavaScript
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Agent Capabilities

The ClickHouse Data Analyst provides:

### Core Database Operations
- **Database Analysis** - Comprehensive database overview
- **Table Exploration** - Detailed table structure and content analysis
- **Data Quality Assessment** - Completeness, duplicates, anomalies detection
- **Performance Insights** - Query optimization recommendations

### Advanced Analytics
- **Statistical Analysis** - Mean, median, quartiles, distributions
- **Pattern Recognition** - Trend identification and data patterns
- **Comparative Analysis** - Table and dataset comparisons
- **Business Insights** - Actionable recommendations

### Natural Language Interface
- **Conversational Queries** - Ask questions in plain English
- **Smart Responses** - Context-aware analysis and explanations
- **Rich Formatting** - Beautiful, structured output with insights

## 💡 Usage Examples

### Web Interface
1. Access the deployed AgentCore endpoint
2. View automatic database analysis
3. Ask questions like "Analyze the sales data table"
4. Execute custom SQL queries
5. Compare different tables

### Programmatic Usage

```python
# Test deployed agent via AgentCore CLI
agentcore invoke '{"prompt": "What insights can you provide about the sales data?"}'

# Or use AWS SDK to invoke the runtime directly
import boto3
client = boto3.client('bedrock-agentcore')
response = client.invoke_runtime(
    runtimeId='your-runtime-id',
    payload='{"prompt": "Analyze my database"}'
)
```

## 🔧 Configuration

### ClickHouse Configuration

Update `clickhouse_config.py` with your ClickHouse Cloud credentials:

```python
CLICKHOUSE_CONFIG = {
    "host": "your-host.clickhouse.cloud",
    "port": 8443,
    "user": "default", 
    "password": "your-password",
    "database": "default",
    "secure": True,
    "verify": True
}
```

### AgentCore Environment Variables

The agent automatically uses these environment variables when deployed on AgentCore:

```bash
BEDROCK_AGENTCORE_MEMORY_ID=auto-configured
AWS_REGION=your-deployment-region
```


## 🔍 Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify ClickHouse Cloud credentials in `clickhouse_config.py`
   - Check network connectivity to ClickHouse Cloud

2. **MCP Server Issues**
   - Ensure UV is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Check MCP server logs for connection errors

3. **Port Already in Use**
   - Kill existing process: `lsof -ti :8001 | xargs kill -9`
   - Or use a different port in the startup script

### Debug Mode

Enable detailed logging:

```bash
export STRANDS_LOG_LEVEL=DEBUG
python start_analyst_ui.py
```

## 📚 References

- [Strands SDK Documentation](https://strandsagents.com/)
- [ClickHouse MCP Server](https://github.com/ClickHouse/mcp-clickhouse)
- [ClickHouse Cloud Documentation](https://clickhouse.com/docs)
- [Model Context Protocol](https://modelcontextprotocol.io)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

Built with ❤️ using [Strands SDK](https://strandsagents.com/) and [ClickHouse MCP](https://github.com/ClickHouse/mcp-clickhouse)