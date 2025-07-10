# Running RAG experiments using S3 Vectors and MLflow

In this post, we show how to build a cost-effective, enterprise-scale RAG application using Amazon S3 Vectors, SageMaker AI for scalable inference, and SageMaker managed MLflow for experiment tracking and evaluation, making sure the responses meet enterprise standards. We demonstrate this by building a RAG system that answers questions about Amazon financials using annual reports, shareholder letters, and 10-K filings as the knowledge base.

This sample consists of the following components:

- Document ingestion – Process PDF documents using LangChain’s document loaders.
- Text chunking – Experiment with different chunking strategies (for example, fixed-size or recursive) and configurations (such as chunk size and overlap).
- Vector embedding – Generate embeddings using SageMaker deployed embedding LLM models.
- Vector storage – Store vectors in Amazon S3 Vectors with associated metadata (such as the type of document). You can also store the entire text chunk in the vector metadata for simple retrieval.
- Retrieval logic – Implement semantic search using vector similarity.
- Response generation – Create responses using retrieved context and a SageMaker deployed LLM. The retrieval and generation steps are used in a LangGraph graph.
- Evaluation – Assess performance using ground truth datasets and SageMaker managed MLflow metrics.

### Code Contains:

The blog includes practical code snippets demonstrating:

- Invoking embedding and text generation models using SageMaker endpoints and the SageMaker SDK.
- Using the S3 Vectors SDK to put vectors into a vector index.
- Creating a RAG application using LangGraph.
- Evaluating the performance of the RAG application and tracking the experiments with managed MLflow.

![LangGraph RAG with S3 Vectors](s3-vectors-buckets-arch.png?raw=true "LangGraph RAG with S3 Vectors")

## Prerequisites
- An AWS account with billing enabled.
- A SageMaker AI Studio domain. For more information, refer to Use quick setup for Amazon SageMaker AI.
- Access to a running SageMaker AI managed MLflow tracking server in Amazon SageMaker Studio. For more information, refer to the instructions for setting up a new MLflow tracking server.
- Enable access to an Amazon Bedrock foundation model (FM) to use in LLM-as-a-judge. In this sample, we use Anthropic’s Claude 3 Sonnet.

## Executio

The code is self contained in the `experiment-tracking-with-mlflow.ipynb` file.

# License
This library is licensed under the MIT-0 License. See the LICENSE file.