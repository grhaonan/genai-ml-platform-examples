# GenAIOps - Observability for GenAI Applications with Amazon Bedrock and Langfuse

## Welcome

This hands-on workshop teaches you how to implement comprehensive observability for GenAI applications using Amazon Bedrock and Langfuse. As organizations deploy generative AI into production, robust observability, monitoring, and evaluation become critical for understanding model behavior, identifying performance bottlenecks, and ensuring consistent quality.

## What You Will Learn

Through five progressive labs, you will gain practical experience with GenAI observability—from tracing basic interactions to monitoring production-grade agents. You'll learn to trace LLM interactions, manage prompts effectively, evaluate model outputs systematically, and monitor AI agents with confidence.

## Key Technologies

- **Amazon Bedrock**: Fully managed service providing access to foundation models through a unified API, including Nova Pro, Lite, and Micro models, with built-in capabilities for knowledge bases, guardrails, and agents
- **Langfuse**: Open-source LLM engineering platform providing comprehensive observability, tracing, prompt management, and evaluation capabilities for AI applications
- **RAGAS**: Evaluation framework for assessing RAG pipelines using LLM-based metrics without requiring manually-annotated ground truth data
- **Strands Agents**: Open-source SDK for building AI agents with a model-driven approach, allowing developers to define prompts and tools while the LLM handles planning and orchestration
- **Model Context Protocol (MCP)**: Open standard providing a universal interface for connecting AI systems to external data sources and tools
- **Amazon Bedrock AgentCore**: Infrastructure for deploying and operating AI agents with serverless runtime, automatic scaling, session isolation, and enterprise security controls

## Prerequisites

- AWS Account with appropriate permissions for Amazon Bedrock and related services (temporary accounts may be provided by workshop instructors)
- Basic understanding of Python programming
- Python 3.9+
- Langfuse instance (self-hosted or cloud)
- Model access enabled for:
  - Amazon Nova models (Pro, Lite, Micro)
  - Cohere Embed models
  - Claude models (optional)

**Estimated time to complete**: 3-4 hours for all five labs
## Setup

### 1. Environment Configuration

Copy the example environment file and configure your Langfuse credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Langfuse credentials:

```
LANGFUSE_PUBLIC_KEY=pk-lf-your-actual-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-actual-secret-key
LANGFUSE_HOST=https://your-langfuse-instance.com
```

> **Important**: When copying credentials from the Langfuse portal, note that newer versions of Langfuse display `LANGFUSE_BASE_URL`. However, these labs use an older version of the Langfuse library that expects `LANGFUSE_HOST`. Please use `LANGFUSE_HOST` instead of `LANGFUSE_BASE_URL` in your `.env` file. The value (URL) remains the same—only the variable name differs.

### 2. Install Dependencies

Each lab has its own requirements. Install dependencies as instructed in the Jupyter Notebook.

## Labs

### Lab 1: Tracing and Prompt Management with Amazon Bedrock and Langfuse
**Location**: `lab1/lab1-langfuse-basics.ipynb`

Focus on essential aspects of tracing and prompt management using Langfuse with Amazon Bedrock:
- Trace and debug LLM-based workflows with comprehensive instrumentation
- Manage and optimize prompts for Bedrock Nova models
- Monitor prompt performance and output quality with built-in metrics
- Single-turn and multi-turn conversations
- Multi-modal capabilities (text + images)
- Tool use with function calling
- Prompt versioning and management

**What You'll Build**: Understanding of prompt design, management, and LLM interaction tracing

### Lab 2: Retrieval-Augmented Generation with Amazon Bedrock and Langfuse
**Location**: `lab2/lab2-rag-langfuse.ipynb`

Explore RAG techniques with Bedrock LLMs for contextual awareness and enhanced knowledge:
- Set up a RAG pipeline using Bedrock Knowledge Bases
- Monitor and trace RAG pipeline performance with Langfuse
- Evaluate using RAGAS metrics (Faithfulness, Response Relevancy, Context Precision)
- Associate evaluation results with traces
- Implement scoring strategies: per-trace vs. sampling

**What You'll Build**: Complete RAG pipeline with monitoring and evaluation capabilities

### Lab 3: Generative AI Evaluation with Amazon Bedrock Guardrails and Langfuse
**Location**: `lab3/`

Dive into assessment and evaluation of generative AI applications:
- **Lab 3.1**: Implement metric-based and LLM-as-a-judge evaluation approaches
- **Lab 3.2**: Integrate Bedrock Guardrails for ethical and safety standards
  - Content filtering
  - PII detection and redaction
  - Prompt injection prevention
  - Denied topics enforcement
- Monitor and analyze performance with Langfuse
- Track key quality indicators

**What You'll Build**: Evaluation framework with safety guardrails

### Lab 4: Observability for Strands Agent with Langfuse
**Location**: `lab4/lab4-langfuse-strands-mcp.ipynb`

Explore agentic AI with comprehensive observability using Strands Agent framework:
- Build an intelligent AWS assistant connecting foundation models to AWS services
- Implement end-to-end observability for agentic AI systems
- Configure agent types: built-in tools, custom tools, and MCP tools
- Monitor, debug, and optimize agent performance through comprehensive tracing

**What You'll Build**: Agentic AI with Langfuse observability

### Lab 5: Production Agent Deployment with AgentCore
**Location**: `lab5/lab5-langfuse-agentcore.ipynb`

Deploy AI agents to Amazon Bedrock AgentCore Runtime through three progressive exercises:
- Deploy agents using AgentCore's serverless runtime with automatic scaling
- Configure dual observability: AgentCore Observability and Langfuse integration
- Develop custom tools using Strands framework
- Integrate external services through Model Context Protocol
- Monitor performance through CloudWatch GenAI Observability dashboard and Langfuse traces

**What You'll Build**: Enterprise-grade agent deployment with comprehensive observability and security controls

## Region Configuration

All labs use `us-west-2` region. Nova models are accessed via Cross-Region Inference (CRIS) with the `us.` prefix (e.g., `us.amazon.nova-pro-v1:0`).

## Troubleshooting

### Langfuse Connection Issues
- Verify credentials in `.env` file
- Check `langfuse.auth_check()` returns `True`
- Ensure Langfuse host URL is accessible

### Model Access Errors
- Verify model access in [Bedrock Model Access console](https://console.aws.amazon.com/bedrock/home#/modelaccess)
- Check region is set to `us-west-2`
- Confirm Nova models are available via `list_inference_profiles()`

### Knowledge Base Setup
- Ensure IAM permissions for OpenSearch Serverless
- Verify S3 bucket permissions
- Check data source sync status

## What You Will Build

By completing this workshop, you will have hands-on experience building complete GenAI applications with production-ready observability:
- Tracing that captures every step of AI application execution
- Systematic prompt management with versioning and performance tracking
- Model output evaluation using automated metrics and LLM-as-judge approaches
- Safety guardrails to prevent harmful outputs
- Autonomous agents handling complex, multi-step tasks
- Production agent deployment with enterprise-grade security and monitoring

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [Strands Agents GitHub](https://github.com/awslabs/strands)
- [Model Context Protocol](https://modelcontextprotocol.io/)

---

**Ready to begin?** Start with Lab 1 to learn the fundamentals of tracing and prompt management.
