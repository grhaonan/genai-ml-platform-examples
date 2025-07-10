# Trace and evaluate agents using Amazon SageMaker MLFlow
This post explores how to effectively trace and evaluate LangChain LangGraph LLM agents using Amazon SageMaker MLflow. LangGraph, a powerful framework for building stateful, multi-actor applications, can be seamlessly integrated with MLflow on Amazon SageMaker for enhanced GenAI Agent experimentation, management, observibility and evaluation. Furthermore, It shows how to integrate with other valuable libraries like RAGAS: Expanding the evaluation metrics using RAGAS gives concrete ways to measure LLM agent's peformance using the exhaustive evaluation metrics and other relevant tools.

### Key Topics Covered:
- Introduction to LangGraph and its capabilities for creating cyclical flows and agentic architectures
- Setting up MLflow on Amazon SageMaker for LangGraph agent tracking
- Implementing tracing for LangGraph agents using MLflow
- Evaluating LangGraph agent performance with MLflow LLM evaluation metrics
- Leveraging Amazon SageMaker features for operationalizing agents (AgentOps)
- MLFlow Prompt registry a powerful tool that streamlines prompt engineering and management in your Generative AI (GenAI) applications. It enables you to version, track, and reuse prompts across your organization, helping maintain consistency and improving collaboration in prompt development (MLFlow Prompt registry is supported in MLFlow version 2.22 or higher.)

### Code Contains:
The blog includes practical code snippets demonstrating:
- Creating a LangGraph agent (Sample). The agent is a ReAct type financial assistant agent to fetch data from datastore. See below for graph visualization (`graph_diagram.png`).
- Logging the agent to SageMaker MLflow
- Tracing agent execution
- Evaluating agent performance

![Alt text](graph_diagram.png?raw=true "ReAct LangGraph agent")

### Benefits:
- Gain insights into agent behavior and decision-making processes
- Improve agent performance through systematic evaluation
- Streamline the deployment and scaling of LangGraph agents on AWS infrastructure
- Operationalize GenAI agents using GenAI models on AWS

## Prerequisites
- AWS Account 
- SageMakerAI Studio - [AWS Setup instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)
- SageMaker MLFlow tracking server - [AWS Setup instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server.html) . Note: For MLFlow Prompt Registry you will need MLFlow version that support advanced GenAI features.
- Access to Bedrock LLM model. This code was tested with bedrock model with inference profile `us.anthropic.claude-3-5-haiku-20241022-v1:0`.
- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver (Preferred)

## Getting started
1. Change directory into the root folder to find `run.py`
2. Run the uv commands `uv venv` and `uv pip sync pyproject.toml`. See the next section if you want to use pip instead of uv
4. Create environmental file `.env`. (Or you can also rename example in file `.env_sample`)
5. Set all the variable in the `.env` file. See example in file `.env_sample`

        ```
        PROJECT="<ENTER-YOUR-VALUE>"
        MLFLOW_URI_SMAI = "arn:aws:sagemaker:<ENTER-YOUR-VALUE-SAGEMAKERAI-MLFLOW-TRACKING-SERVER-ARN>"
        MLFLOW_EXPERIMENT_ID = "<ENTER-YOUR-VALUE>"
        VERSION = "<ENTER-YOUR-VALUE>"
        MODELID = '<ENTER-BEDROCK-MODELID-VALUE>'
        AWS_REGION = '<ENTER-YOUR-REGION>' 
        ```
5. Activate your AWS credentials
6. Execute command `uv run run.py` for running the agent and the MLFlow evaluation with tracing.

### Steps to get started with pip instead of uv
1. Install and activate your python virtual environment `virtualenv venv && . venv/bin/activate`
2. Intall the python packages `pip install --upgrade pip` and 
`pip install -r requirements.txt`

### Execution for pip 
 - ```python run.py``` For running the agent and the MLFlow evaluation with tracing.
 - Use the notebook for running the agent evaluation using RAGAS and tracking with MLFlow.

## Troubleshooting
Ignore the following errors 
- ``` ERROR mlflow.tracking.fluent: Failed to start system metrics monitoring: ROCm library not found.```
- ``` AttributeError: 'NoneType' object has no attribute 'rsmi_shut_down'```

# License
This library is licensed under the MIT-0 License. See the LICENSE file.