# SageMaker MLflow RAG Pipeline Automation Project

This project provides tools and utilities for automating RAG (Retrieval-Augmented Generation) pipeline operations using Amazon SageMaker AI. It features a scalable RAG pipeline with MLflow experiment tracking integrated at every stage, automated using SageMaker Pipelines and GitOps CI/CD practices for seamless promotion from development to production environments.

The core functionality includes pipeline orchestration (`sagemaker_pipeline/`), experiment tracking notebooks (`notebooks/`), and GitHub Actions automation (`.github/workflows/`).

## Build & Commands

- Install dependencies: `uv pip sync pyproject.toml`
- Create virtual environment: `uv venv`
- Run single-step pipeline: `python sagemaker_pipeline/run_pipeline.py -n pipeline_single`
- Run multi-step pipeline: `python sagemaker_pipeline/run_pipeline.py -n pipeline_multi`
- Start Jupyter notebooks: `jupyter notebook notebooks/`
- Get pipeline definition: `python sagemaker_pipeline/get_pipeline_definition.py`
- Trigger GitHub Actions: Push changes to `sagemaker_pipeline/**`

### Development Environment

- Python version: >=3.13
- Package manager: uv (preferred) or pip
- SageMaker AI Studio integration
- MLflow tracking server: Amazon SageMaker managed MLflow
- GitHub Actions for CI/CD automation
- OpenSearch domain for vector storage
- SageMaker endpoints for embedding and text generation

## Code Style

- Python: Type hints with modern Python features (>=3.13)
- Use descriptive variable/function names following snake_case convention
- Import organization: Standard library, third-party, local imports
- Error handling: Implement proper exception handling for AWS services
- Configuration: Use environment variables and GitHub secrets/variables
- Documentation: Use docstrings for functions and classes
- Line length: Follow PEP 8 guidelines (79-88 characters)
- Use f-strings for string formatting
- Prefer pathlib for file operations
- NEVER hardcode AWS credentials or sensitive data

## Testing

- Interactive testing through Jupyter notebooks
- SageMaker Pipeline validation and execution
- MLflow experiment tracking for performance evaluation
- GitHub Actions workflow testing
- End-to-end RAG pipeline validation
- Multi-step vs single-step pipeline comparison
- Integration testing with AWS services (SageMaker, OpenSearch, S3)

## Architecture

- **Pipeline Orchestration**: SageMaker Pipelines for workflow automation
- **Experiment Tracking**: SageMaker managed MLflow for comprehensive tracking
- **CI/CD**: GitHub Actions with OIDC authentication
- **Vector Storage**: Amazon OpenSearch for document embeddings
- **Model Endpoints**: SageMaker endpoints for embedding and text generation
- **Artifact Storage**: Amazon S3 for datasets and model artifacts
- **Infrastructure**: Infrastructure-as-Code (IaC) with GitOps practices
- **Processing**: SageMaker Processing jobs for data ingestion and chunking

## Security

- Use AWS IAM roles and OIDC for GitHub Actions authentication
- Store sensitive configuration in GitHub secrets and variables
- Never commit AWS credentials or API keys to repository
- Use AWS SDK credential chain for authentication
- Validate all inputs in pipeline steps
- Follow AWS security best practices for SageMaker, S3, and OpenSearch
- Use least privilege principle for IAM permissions
- Regular dependency updates for security patches
- Secure artifact storage with appropriate S3 bucket policies

## GitHub Actions Setup

Configure the following GitHub secrets and variables:

### Secrets
- `SAGEMAKER_PIPELINE_ROLE_ARN`: IAM role for SageMaker Pipeline execution
- `PIPELINE_EXECUTION_ROLE_ARN`: IAM role for GitHub Actions AWS access
- `ARTIFACT_BUCKET`: S3 bucket for storing artifacts

### Variables
- `MLFLOW_URI`: SageMaker MLflow tracking server ARN
- `AWS_REGION`: AWS region for resource deployment
- `EMBEDDING_ENDPOINT_NAME`: SageMaker endpoint for embedding generation
- `TEXT_ENDPOINT_NAME`: SageMaker endpoint for text generation
- `DOMAIN_NAME`: OpenSearch domain endpoint
- `PROCESSING_INSTANCE_TYPE`: SageMaker instance type for processing
- `PROCESSING_INSTANCE_COUNT`: Number of processing instances

## Pipeline Components

### Single-Step Pipeline
- **Location**: `sagemaker_pipeline/steps/single_step_pipeline/`
- **Script**: `single-step-pipeline.py`
- **Function**: Complete RAG workflow in one processing step
- **Use Case**: Simple experimentation and rapid prototyping

### Multi-Step Pipeline
- **Location**: `sagemaker_pipeline/steps/multi_step_pipeline/`
- **Components**:
  - `data_ingestion.py`: Document ingestion and preprocessing
  - `data_preparation.py`: Data preparation and validation
  - `data_chunking.py`: Text chunking strategies
  - `rag_retrieval.py`: Vector retrieval and similarity search
  - `rag_evaluation.py`: Performance evaluation and metrics
- **Use Case**: Production workflows with granular control and monitoring

## MLflow Integration

- Experiment organization for RAG pipeline components
- Parameter tracking for chunking, retrieval, and generation configurations
- Metric logging for evaluation results
- Artifact storage for processed datasets and model outputs
- Model versioning and comparison capabilities
- Integration with SageMaker Pipeline execution tracking
- Dashboard visualization for experiment analysis

## Project Structure

```
sagemaker-mlflow-rag-pipeline-automation/
├── .github/workflows/
│   └── build_sagemaker_pipeline.yml    # GitHub Actions CI/CD
├── notebooks/
│   ├── sagemaker-mlflow-experiment-agenticrag.ipynb
│   ├── sagemaker-pipeline-building.ipynb
│   └── utils.py
├── sagemaker_pipeline/
│   ├── pipeline_modules/               # Pipeline creation modules
│   ├── steps/                         # Pipeline step implementations
│   ├── run_pipeline.py               # Pipeline execution script
│   ├── get_pipeline_definition.py    # Pipeline definition utility
│   └── utils.py                      # Common utilities
├── images/                           # Architecture diagrams
├── screenshots/                      # Documentation screenshots
└── pyproject.toml                   # Project dependencies
```

## Dependencies

Key dependencies include:
- `sagemaker >= 2.210.0`: AWS SageMaker SDK
- `mlflow == 2.22.1`: Experiment tracking and model management
- `opensearch-py >= 2.8.0`: OpenSearch client for vector operations
- `langchain-text-splitters >= 0.3.8`: Text chunking utilities
- `datasets >= 3.6.0`: Dataset handling and processing
- `requests-aws4auth >= 1.3.1`: AWS authentication for requests

## Troubleshooting

Common issues and solutions:
- **ModuleNotFoundError**: Run `uv pip sync pyproject.toml` to install dependencies
- **GitHub Actions authentication**: Verify OIDC setup and IAM role permissions
- **SageMaker Pipeline failures**: Check CloudWatch logs and pipeline execution details
- **MLflow connection issues**: Verify tracking server ARN and network connectivity
- **OpenSearch access**: Ensure proper security group and IAM permissions

## Configuration Management

When adding new configuration options:
1. Update GitHub secrets/variables as needed
2. Modify pipeline parameter definitions in `pipeline_modules/`
3. Update documentation in README.md and this AGENT.md
4. Test configuration changes in development environment
5. Validate through GitHub Actions workflow

All configuration keys use consistent naming and MUST be documented in the appropriate sections above.
