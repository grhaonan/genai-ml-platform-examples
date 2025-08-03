# AWS GenAI ML Platform Examples

This repository contains a comprehensive collection of examples and resources for building, training, and deploying Generative AI and Machine Learning models at scale using AWS services. The platform demonstrates best practices across data management, governance, infrastructure, integration, and operations for GenAI/ML workflows.

The repository is organized into specialized domains (`operations/`, `infrastructure/`, `integration/`, `demo-apps/`) with each containing focused examples and implementations.

## Build & Commands

- Clone repository: `git clone https://github.com/aws-samples/genai-ml-platform-examples.git`
- Navigate to project: `cd genai-ml-platform-examples`
- Explore subdirectories: Each contains specific README.md and AGENT.md files
- Install dependencies: Follow individual project requirements (pip, uv, npm, etc.)
- Run examples: Execute notebooks, scripts, or applications per subdirectory instructions
- Deploy infrastructure: Use CDK, CloudFormation, or Terraform templates as provided

### Development Environment

- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- Python >=3.8 (most projects require >=3.13 for modern features)
- Jupyter Notebook environment (SageMaker Studio, local, or cloud)
- Docker for containerized deployments
- Node.js for web applications and CDK deployments
- Package managers: pip, uv, npm, pnpm as specified per project

## Code Style

- Python: Type hints with modern Python features, snake_case naming
- Use descriptive variable/function names following language conventions
- Import organization: Standard library, third-party, local imports
- Error handling: Implement proper exception handling for AWS services
- Configuration: Use environment variables and AWS parameter store
- Documentation: Comprehensive docstrings and README files
- Line length: Follow language-specific guidelines (PEP 8 for Python)
- Use f-strings for Python string formatting
- Prefer pathlib for file operations in Python
- NEVER hardcode AWS credentials or sensitive data

## Testing

- Interactive testing through Jupyter notebooks
- Unit testing with pytest for Python projects
- Integration testing with AWS services in development environments
- End-to-end testing for complete workflows
- Performance testing for ML model inference
- Security testing for deployed applications
- Mock external dependencies appropriately
- Use AWS testing best practices (LocalStack, moto, etc.)

## Architecture

The repository follows a domain-driven structure:

- **Operations**: MLOps and GenAI operations workflows
  - SageMaker MLflow integration examples
  - RAG pipeline automation
  - Agent tracing and evaluation
  - Model registry and lifecycle management
- **Infrastructure**: AWS resource provisioning and management
  - Cost-efficient model inference solutions
  - EKS-based deployments
  - Serverless architectures
  - Auto-scaling configurations
- **Integration**: Third-party and open-source integrations
  - LangFuse integration for observability
  - MCP (Model Context Protocol) implementations
  - Custom framework integrations
- **Demo Applications**: End-to-end application examples
  - Health and travel applications
  - Full-stack implementations with AI/ML backends

## Security

- Use AWS IAM roles and policies for service access
- Store sensitive configuration in AWS Systems Manager Parameter Store
- Never commit AWS credentials, API keys, or secrets to repository
- Use AWS SDK credential chain for authentication
- Validate all user inputs in applications and pipelines
- Follow AWS security best practices for all services
- Use least privilege principle for IAM permissions
- Regular dependency updates for security patches
- Implement proper data encryption at rest and in transit
- Use AWS KMS for key management

## Project Navigation

### Operations (MLOps/GenAI Operations)
Each operations project has its own AGENT.md file with specific guidance:

- **[sagemaker-mlflow-trace-evaluate-langgraph-agent](operations/sagemaker-mlflow-trace-evaluate-langgraph-agent/AGENT.md)**: LangGraph agent tracing and evaluation with MLflow
- **[sagemaker-mlflow-s3-vector-rag](operations/sagemaker-mlflow-s3-vector-rag/AGENT.md)**: RAG implementation using S3 Vectors and MLflow
- **[sagemaker-mlflow-rag-pipeline-automation](operations/sagemaker-mlflow-rag-pipeline-automation/AGENT.md)**: Automated RAG pipeline with SageMaker Pipelines
- **sagemaker-unified-model-registry**: Unified model registry examples
- **sagemaker-mlflow-model-registry**: MLflow model registry integration

### Infrastructure
- **cost-efficient-model-inference-sagemaker-graviton**: Cost-optimized inference on Graviton processors
- **efficient-model-inference**: Various efficient inference patterns
- **inference-component-scale-to-zero**: Auto-scaling inference components
- **notebooklm-with-bedrock-and-amazon-eks**: NotebookLM implementation on EKS
- **train-openclip-with-hyperpod**: OpenCLIP training with SageMaker HyperPod

### Integration
- **genaiops-langfuse-on-aws**: LangFuse observability platform integration
- **langfuse**: LangFuse examples and configurations
- **MCP**: Model Context Protocol implementations

### Demo Applications
- **health-app**: Healthcare AI application with full-stack implementation
- **travel-app**: Travel planning application with AI recommendations

## Configuration Management

When working with projects in this repository:

1. **Environment Variables**: Each project uses `.env` files or AWS parameter store
2. **AWS Configuration**: Ensure proper AWS CLI configuration and credentials
3. **Service Dependencies**: Check individual project requirements for AWS services
4. **Resource Limits**: Be aware of AWS service limits and quotas
5. **Cost Management**: Monitor AWS costs, especially for ML training and inference
6. **Regional Considerations**: Some services may not be available in all regions

## Getting Started Workflow

1. **Choose Your Domain**: Navigate to the relevant directory (operations, infrastructure, etc.)
2. **Read Project Documentation**: Each subdirectory has comprehensive README.md files
3. **Check AGENT.md Files**: Operations projects have specific AGENT.md guidance
4. **Set Up Prerequisites**: Install required tools and configure AWS access
5. **Follow Project Instructions**: Execute notebooks, deploy infrastructure, or run applications
6. **Experiment and Customize**: Modify examples to fit your specific use cases

## Contributing Guidelines

- Follow the existing project structure and naming conventions
- Include comprehensive documentation for new examples
- Add AGENT.md files for complex operational projects
- Test all examples thoroughly before submission
- Follow AWS security best practices
- Update this root AGENT.md when adding new major components
- See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines

## Support and Resources

- **AWS Documentation**: Reference official AWS service documentation
- **AWS Samples**: This repository is part of the AWS Samples collection
- **Community**: Engage with the AWS developer community
- **Issues**: Report bugs or request features through GitHub issues
- **Security**: Report security issues following the guidelines in CONTRIBUTING.md

All projects in this repository are licensed under the MIT-0 License. See individual LICENSE files for details.
