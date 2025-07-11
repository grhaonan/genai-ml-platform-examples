
import boto3
import sagemaker
import time
import logging
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger
)
import os

logger = logging.getLogger(__name__)

def get_session(region, default_bucket):

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    
    session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )
    
    return session

def get_pipeline(
    role=None,
    default_bucket=None,
    pipeline_name="rag-multi-step-pipeline",
    base_job_prefix="rag-multi",
    github_repo=None,
    github_action=None,
    github_workflow_id=None
):

    region = os.environ.get("AWS_REGION", "us-west-2")

    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    pipeline_session = PipelineSession(
        boto_session=sagemaker_session.boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        default_bucket=default_bucket
    )
    
    # Generate timestamp for unique naming
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    
    embedding_endpoint_name = ParameterString(
        name="EmbeddingEndpointName",
        default_value=os.environ.get("EMBEDDING_ENDPOINT_NAME")
    )

    text_endpoint_name = ParameterString(
        name="TextEndpointName",
        default_value=os.environ.get("TEXT_ENDPOINT_NAME")
    )

    domain_name = ParameterString(
        name="DomainName",
        default_value=os.environ.get("DOMAIN_NAME")
    )

    index_name = ParameterString(
        name="IndexName",
        default_value="ragops-exp-index"
    )

    embedding_model_id = ParameterString(
        name="EmbeddingModelId",
        default_value= "huggingface-textembedding-all-MiniLM-L6-v2"
    )

    text_model_id = ParameterString(
        name="TextModelId",
        default_value= "deepseek-llm-r1-distill-qwen-7b"
    )

    chunking_strategy = ParameterString(
        name="ChunkingStrategy",
        default_value="RecursiveChunker"
    )

    chunk_size = ParameterString(
        name="ChunkSize",
        default_value= "500"
    )

    chunk_overlap = ParameterString(
        name="ChunkOverlap",
        default_value="200"
    )

    context_retrieval_size = ParameterString(
        name="ContextRetrievalSize",
        default_value="3"
    )

    model_dimensions = ParameterString(
        name="ModelDimensions",
        default_value="384"
    )

    mlflow_tracking_uri = ParameterString(
        name="MLflowTrackingUri",
        default_value=os.environ.get("MLFLOW_URI")
    )


    experiment_name = ParameterString(
        name="ExperimentName",
        default_value="rag-experiment-test"
    )

    instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value=os.environ.get("PROCESSING_INSTANCE_TYPE", "ml.m5.4xlarge")
    )

    instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=int(os.environ.get("PROCESSING_INSTANCE_COUNT", "1"))
    )

    llm_evaluator = ParameterString(
        name="LLMEvaluator",
        default_value="bedrock:/anthropic.claude-3-haiku-20240307-v1:0"
    )

    parent_run_name = ParameterString(
        name="ParentRunName",
        default_value="rag-experiment-run"
    )


    github_repository = ParameterString(
        name="GitHubRepository",
        default_value=github_repo
    )

    github_action_name = ParameterString(
        name="GitHubActionName",
        default_value = github_action
    )
    
    github_workflow_id = ParameterString(
        name="GitHubWorkflowId",
        default_value = github_workflow_id
    )

    # Output S3 location
    experiment_name = f"rag-pipeline-{timestamp}"
    
    # S3 paths for each step
    base_s3_uri = f"s3://{default_bucket}/rag-pipeline/{timestamp}"
    data_prep_output_s3_uri = f"{base_s3_uri}/data-prep"
    chunking_output_s3_uri = f"{base_s3_uri}/chunking"
    ingestion_output_s3_uri = f"{base_s3_uri}/ingestion"
    retrieval_output_s3_uri = f"{base_s3_uri}/retrieval"
    evaluation_output_s3_uri = f"{base_s3_uri}/evaluation"
    
    # Path to your scripts
    source_dir = './sagemaker_pipeline/steps/multi_step_pipeline'
    
    # Create the processor
    processor = PyTorchProcessor(
        framework_version="2.5",
        role=role,
        py_version="py311",
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name=f"{base_job_prefix}-{timestamp}",
        sagemaker_session=pipeline_session
    )
    
    # Step 1: Data Preparation
    data_prep_step_args = processor.run(
        outputs=[
            ProcessingOutput(
                output_name="data_prep_output",
                source="/opt/ml/processing/output/",
                destination=data_prep_output_s3_uri
            )
        ],
        code="data_preparation.py",
        source_dir=source_dir,
        arguments=[
            "--experiment-name", experiment_name,
            "--mlflow-tracking-uri", mlflow_tracking_uri,
            "--output-data-path", "/opt/ml/processing/output",
            "--parent-run-name", parent_run_name,
            "--github-repository", github_repository,
            "--github-action-name", github_action_name,
            "--github-workflow-id", github_workflow_id,
            "--sagemaker-pipeline-name", pipeline_name
        ]
    )

    data_prep_step = ProcessingStep(
        name="DataPrepStep",
        step_args=data_prep_step_args
    )

    # Step 2: Data Chunking
    chunking_step_args = processor.run(
        inputs=[
            ProcessingInput(
                source=data_prep_step.properties.ProcessingOutputConfig.Outputs["data_prep_output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="chunking_output",
                source="/opt/ml/processing/output/",
                destination=chunking_output_s3_uri
            )
        ],
        code="data_chunking.py",
        source_dir=source_dir,
        arguments=[
            "--experiment-name", experiment_name,
            "--mlflow-tracking-uri", mlflow_tracking_uri,
            "--input-data-path", "/opt/ml/processing/input",
            "--output-data-path", "/opt/ml/processing/output",
            "--chunking-strategy", chunking_strategy,
            "--chunk-size", chunk_size,
            "--chunk-overlap", chunk_overlap
        ]
    )

    data_chunking_step = ProcessingStep(
        name="DataChunkingStep",
        step_args=chunking_step_args
    )

    # Step 3: Data Ingestion
    ingestion_step_args = processor.run(
        inputs=[
            ProcessingInput(
                source=data_chunking_step.properties.ProcessingOutputConfig.Outputs["chunking_output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="ingestion_output",
                source="/opt/ml/processing/output/",
                destination=ingestion_output_s3_uri
            )
        ],
        code="data_ingestion.py",
        source_dir=source_dir,
        arguments=[
            "--experiment-name", experiment_name,
            "--mlflow-tracking-uri", mlflow_tracking_uri,
            "--input-data-path", "/opt/ml/processing/input",
            "--output-data-path", "/opt/ml/processing/output",
            "--embedding-endpoint-name", embedding_endpoint_name,
            "--domain-name", domain_name,
            "--index-name", index_name,
            "--model-dimensions", model_dimensions,
            "--embedding-model-id", embedding_model_id
        ]
    )

    data_ingestion_step = ProcessingStep(
        name="DataIngestionStep",
        step_args=ingestion_step_args
    )

    # Step 4: RAG Retrieval
    retrieval_step_args = processor.run(
        inputs=[
            ProcessingInput(
                source=data_ingestion_step.properties.ProcessingOutputConfig.Outputs["ingestion_output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="retrieval_output",
                source="/opt/ml/processing/output/",
                destination=retrieval_output_s3_uri
            )
        ],
        code="rag_retrieval.py",
        source_dir=source_dir,
        arguments=[
            "--experiment-name", experiment_name,
            "--mlflow-tracking-uri", mlflow_tracking_uri,
            "--input-data-path", "/opt/ml/processing/input",
            "--output-data-path", "/opt/ml/processing/output",
            "--embedding-endpoint-name", embedding_endpoint_name,
            "--text-endpoint-name", text_endpoint_name,
            "--domain-name", domain_name,
            "--index-name", index_name,
            "--context-retrieval-size", context_retrieval_size,
            "--embedding-model-id", embedding_model_id,
            "--text-model-id", text_model_id
        ]
    )

    rag_retrieval_step = ProcessingStep(
        name="RAGRetrievalStep",
        step_args=retrieval_step_args
    )

    # Step 5: RAG Evaluation
    evaluation_step_args = processor.run(
        inputs=[
            ProcessingInput(
                source=rag_retrieval_step.properties.ProcessingOutputConfig.Outputs["retrieval_output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_output",
                source="/opt/ml/processing/output/",
                destination=evaluation_output_s3_uri
            )
        ],
        code="rag_evaluation.py",
        source_dir=source_dir,
        arguments=[
            "--experiment-name", experiment_name,
            "--mlflow-tracking-uri", mlflow_tracking_uri,
            "--input-data-path", "/opt/ml/processing/input",
            "--output-data-path", "/opt/ml/processing/output",
            "--embedding-endpoint-name", embedding_endpoint_name,
            "--text-endpoint-name", text_endpoint_name,
            "--domain-name", domain_name,
            "--embedding-model-id", embedding_model_id,
            "--text-model-id", text_model_id,
            "--chunking-strategy", chunking_strategy,
            "--chunk-size", chunk_size,
            "--chunk-overlap", chunk_overlap,
            "--role-arn", role,
            "--llm-evaluator", llm_evaluator
        ]
    )

    rag_evaluation_step = ProcessingStep(
        name="RAGEvaluationStep",
        step_args=evaluation_step_args
    )
    
    # Create the pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            embedding_endpoint_name,
            text_endpoint_name,
            domain_name,
            index_name,
            embedding_model_id,
            text_model_id,
            chunking_strategy,
            chunk_size,
            chunk_overlap,
            context_retrieval_size,
            model_dimensions,
            mlflow_tracking_uri,
            instance_type,
            instance_count,
            parent_run_name,
            llm_evaluator,
            github_repository,
            github_action_name,
            github_workflow_id,
        ],
        steps=[
            data_prep_step, 
            data_chunking_step, 
            data_ingestion_step, 
            rag_retrieval_step, 
            rag_evaluation_step
        ],
        sagemaker_session=pipeline_session
    )
    
    return pipeline

def get_pipeline_custom_tags(tags, region, project_arn):
    """Gets custom tags for the pipeline.
    
    Args:
        tags: Existing tags
        region: AWS region
        project_arn: SageMaker project ARN
        
    Returns:
        List of tags to attach to the pipeline
    """
    tags = tags or []
    
    # Add project tag if part of a SageMaker project
    if project_arn:
        tags.append({"Key": "sagemaker:project-name", "Value": project_arn})
        
    # Add region tag
    tags.append({"Key": "Region", "Value": region})
    
    return tags