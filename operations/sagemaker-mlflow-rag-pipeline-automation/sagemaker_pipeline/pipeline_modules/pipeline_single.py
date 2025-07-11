import os
import boto3
import sagemaker
import time
import logging
from sagemaker.processing import ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger
)

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
    pipeline_name="rag-single-step-pipeline",
    base_job_prefix="rag-single",
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
    
        # Define pipeline parameters
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
    output_s3_uri = f"s3://{default_bucket}/rag-pipeline/output/{timestamp}/"
    experiment_name = f"rag-pipeline-{timestamp}"
    
    # Path to your existing script
    script_path = "single-step-pipeline.py"
    source_dir = './sagemaker_pipeline/steps/single_step_pipeline'
    
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
    
    processing_step_args = processor.run(
        code=script_path,
        source_dir=source_dir,
        outputs=[
            ProcessingOutput(
                output_name="rag_output",
                source="/opt/ml/processing/output/",
                destination=output_s3_uri
            )
        ],
        arguments=[
            "--experiment-name", experiment_name,
            "--mlflow-tracking-uri", mlflow_tracking_uri,
            "--embedding-endpoint-name", embedding_endpoint_name,
            "--text-endpoint-name", text_endpoint_name,
            "--domain-name", domain_name,
            "--index-name", index_name,
            "--chunking-strategy", chunking_strategy,
            "--chunk-size", chunk_size,
            "--chunk-overlap", chunk_overlap,
            "--context-retrieval-size", context_retrieval_size,
            "--embedding-model-id", embedding_model_id,
            "--text-model-id", text_model_id,
            "--model-dimensions", model_dimensions,
            "--output-data-path", "/opt/ml/processing/output",
            "--role-arn", role,
            "--llm-evaluator", llm_evaluator,
            "--github-repository", github_repository,
            "--github-action-name", github_action_name,
            "--github-workflow-id", github_workflow_id,
            "--sagemaker-pipeline-name", pipeline_name
        ],
        logs=True   
    )
    
    processing_step = ProcessingStep(
        name="SingleStepRagPipeline",
        step_args=processing_step_args
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
            llm_evaluator,
            github_repository,
            github_action_name,
            github_workflow_id,
        ],
        steps=[processing_step],
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
    
    if project_arn:
        tags.append({"Key": "sagemaker:project-name", "Value": project_arn})
        
    # Add region tag
    tags.append({"Key": "Region", "Value": region})
    
    return tags