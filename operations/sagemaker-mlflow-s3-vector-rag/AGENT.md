# SageMaker MLflow S3 Vector RAG Project

This project demonstrates how to build a cost-effective, enterprise-scale RAG (Retrieval Augmented Generation) application using Amazon S3 Vectors, SageMaker AI for scalable inference, and SageMaker managed MLflow for experiment tracking and evaluation. The system answers questions about Amazon financials using annual reports, shareholder letters, and 10-K filings as the knowledge base.

The core functionality includes document processing, vector storage in S3, retrieval logic, and response generation using LangGraph, with comprehensive evaluation through MLflow.

## Build & Commands

- Run main experiment: Open and execute `experiment-tracking-with-mlflow.ipynb`
- Install dependencies: `pip install -r requirements.txt` (if requirements file exists)
- Start Jupyter: `jupyter notebook experiment-tracking-with-mlflow.ipynb`
- Process documents: Execute document ingestion cells in notebook
- Run evaluations: Execute MLflow evaluation cells
- View results: Check SageMaker MLflow tracking server

### Development Environment

- Primary interface: Jupyter Notebook (`experiment-tracking-with-mlflow.ipynb`)
- SageMaker AI Studio domain required
- SageMaker managed MLflow tracking server
- Amazon Bedrock access (Claude 3 Sonnet for LLM-as-a-judge)
- S3 Vectors for vector storage
- AWS account with appropriate permissions

## Code Style

- Python: Type hints with modern Python features
- Use descriptive variable/function names following snake_case convention
- Class documentation: Use comprehensive docstrings with attributes and methods
- Import organization: Standard library, third-party (boto3, langchain), local imports
- Error handling: Implement proper exception handling for AWS services
- Configuration: Use environment variables or notebook parameters
- Line length: Follow PEP 8 guidelines (79-88 characters)
- Use f-strings for string formatting
- Prefer pathlib for file operations
- NEVER hardcode AWS credentials or sensitive data

## Testing

- Interactive testing through Jupyter notebook cells
- MLflow evaluation metrics for RAG performance assessment
- Ground truth datasets: `amazon_10k_eval_dataset.jsonl`
- Document processing validation with sample files
- Vector similarity testing for retrieval accuracy
- End-to-end RAG pipeline testing
- LLM-as-a-judge evaluation using Bedrock models
- Experiment comparison through MLflow UI

## Architecture

- **Document Processing**: LangChain document loaders for PDF processing
- **Text Chunking**: Multiple strategies (fixed-size, recursive) with configurable parameters
- **Vector Embedding**: SageMaker deployed embedding models
- **Vector Storage**: Amazon S3 Vectors with metadata support
- **Retrieval**: Semantic search using vector similarity
- **Generation**: SageMaker deployed LLMs with retrieved context
- **Orchestration**: LangGraph for retrieval and generation workflow
- **Evaluation**: SageMaker managed MLflow for experiment tracking
- **LLM Provider**: Amazon Bedrock (Claude 3 Sonnet)

## Security

- Use AWS IAM roles and policies for service access
- Store sensitive configuration in environment variables or SageMaker parameters
- Never commit AWS credentials to repository
- Use AWS SDK credential chain for authentication
- Validate all user inputs in RAG pipeline
- Follow AWS security best practices for S3, SageMaker, and Bedrock
- Use least privilege principle for AWS permissions
- Regularly update dependencies for security patches
- Secure vector storage with appropriate S3 bucket policies

## Experiment Configuration

Key experimental parameters to configure:

1. **Chunking Parameters**:
   - `chunk_size`: Size of text chunks (default: 600)
   - `chunk_overlap`: Overlap between chunks (default: 100)
   - Chunking strategy: Fixed-size vs Recursive

2. **Model Configuration**:
   - Embedding model: SageMaker JumpStart model ID and version
   - Generation model: SageMaker endpoint configuration
   - Bedrock model: Claude 3 Sonnet for evaluation

3. **Retrieval Parameters**:
   - Vector similarity threshold
   - Number of retrieved documents
   - Metadata filtering options

4. **MLflow Tracking**:
   - Experiment names and run parameters
   - Evaluation metrics and artifacts
   - Model versioning and comparison

## RAG Components

- **Document Ingestion**: `utils.py` - Document processing utilities
- **Text Chunking**: `FixedSizeChunker` and `RecursiveCharacterTextSplitter` classes
- **Vector Operations**: S3 Vectors SDK integration
- **Retrieval Logic**: Semantic search implementation
- **Generation Pipeline**: LangGraph-based RAG workflow
- **Evaluation**: MLflow metrics and ground truth comparison
- **Data Sources**: JSON files with document metadata and evaluation datasets

## MLflow Integration

- Experiment tracking with SageMaker managed MLflow
- Parameter logging for chunking and model configurations
- Metric tracking for RAG performance evaluation
- Artifact storage for processed documents and results
- Model comparison and versioning capabilities
- Integration with ground truth datasets for evaluation
- LLM-as-a-judge evaluation metrics

## Data Files

- `amazon_10k_eval_dataset.jsonl`: Ground truth evaluation dataset
- `source_files_short_version.json`: Short document metadata
- `source_files_long_version.json`: Extended document metadata
- `s3-vectors-buckets-arch.png`: Architecture diagram
- `mlflow-experiment-run.png`: MLflow experiment visualization

## Prerequisites Setup

1. AWS account with billing enabled
2. SageMaker AI Studio domain configured
3. SageMaker managed MLflow tracking server running
4. Amazon Bedrock access enabled (Claude 3 Sonnet)
5. Appropriate IAM permissions for S3, SageMaker, and Bedrock
6. S3 bucket configured for vector storage

All configuration should be documented and environment-specific settings managed through SageMaker parameters or environment variables.
