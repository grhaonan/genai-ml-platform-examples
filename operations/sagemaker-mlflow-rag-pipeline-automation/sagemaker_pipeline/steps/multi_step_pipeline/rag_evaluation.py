import argparse
import os
import mlflow
import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import sagemaker
from datasets import load_from_disk
import pandas as pd
from langchain import hub
from langchain_core.messages import BaseMessage
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph
from typing import List, Dict, Union, Any
from typing_extensions import TypedDict
from mlflow.entities import SpanType

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--embedding-endpoint-name", type=str, required=True)
    parser.add_argument("--text-endpoint-name", type=str, required=True)
    parser.add_argument("--domain-name", type=str, required=True)
    parser.add_argument("--embedding-model-id", type=str, required=True)
    parser.add_argument("--text-model-id", type=str, required=True)
    parser.add_argument("--chunking-strategy", type=str, default="RecursiveChunker")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--llm-evaluator",type=str,default="bedrock:/anthropic.claude-3-haiku-20240307-v1:0")
    return parser.parse_args()

def setup_opensearch_client(domain_name):
    host = domain_name
    port = 443
    
    # Authentication
    service = "es"
    region = os.environ.get('AWS_REGION')
    print(region)
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )
    
    # Initialize OpenSearch client
    opensearch_client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    return opensearch_client



def get_parent_id(input_path,output_path):
    parent_run_id_file = os.path.join(input_path, "parent_run_id.txt")
    if os.path.exists(parent_run_id_file):
        with open(parent_run_id_file, "r") as f:
            parent_run_id = f.read().strip()
        print(f"Found parent run ID: {parent_run_id}")
        
        # Copy parent run ID to output for next step
        # This will be uploaded to S3 for the next step to use
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "parent_run_id.txt"), "w") as f:
            f.write(parent_run_id)

        return parent_run_id

    else:
        print("Warning: Parent run ID file not found at", parent_run_id_file)
        print("Directory contents:", os.listdir(input_path))
        return


def rag_evaluation(input_path, output_path, opensearch_client, llm_evaluator, embedding_endpoint_name, 
                  text_endpoint_name, embedding_model_id, text_model_id,
                  chunking_strategy, chunk_size, chunk_overlap):

    with mlflow.start_run(run_id=get_parent_id(input_path, output_path)) as run:
        run_id = run.info.run_id
        print("mlflow_run", run_id)

        with mlflow.start_run(run_name="RAGEvaluation", nested=True):
            # Load dataset from previous step
            source_dataset = load_from_disk(os.path.join(input_path, "original_dataset"))
            
            # Load graph configuration
            with open(os.path.join(input_path, "graph_config.json"), "r") as f:
                graph_config = json.load(f)
            
            index_name = graph_config["index_name"]
            
            # Load the pre-computed retrieval results
            with open(os.path.join(input_path, "all_retrieval_results.json"), "r") as f:
                all_retrieval_results = json.load(f)
            
            # Convert to pandas DataFrame
            df = source_dataset['train'].to_pandas()
            
            # Merge 'final_decision' and 'long_answer' into a new column 'merged_answer'
            df['merged_answer'] = df['final_decision'].astype(str) + ', ' + df['long_answer'].astype(str)
            
            # Drop unnecessary columns
            df = df.drop(columns=['final_decision', 'long_answer', 'context', 'pubid'])
    
            # Create mini evaluation dataset - use the same questions that were used in retrieval
            questions = [result["question"] for result in all_retrieval_results]
            eval_df_mini = df[df['question'].isin(questions)].head(10)
    
            print(eval_df_mini)
            
            # Set up the model function that will use the pre-computed results
            def model_for_eval(input_df):
                results = []
                for index, row in input_df.iterrows():
                    question = row["question"]
                    
                    # Find the pre-computed result for this question
                    result = next((r for r in all_retrieval_results if r["question"] == question), None)
                    
                    if result:
                        # Use the pre-computed result
                        answer = result.get("answer", "")
                        context = result.get("context", "")
                        results.append({
                            "answer": answer,
                            "context": context
                        })
                    else:
                        # Fallback if not found (shouldn't happen)
                        print(f"Warning: No pre-computed result found for question: {question}")
                        results.append({
                            "answer": "No pre-computed result found",
                            "context": ""
                        })
                
                return pd.DataFrame(results)
 
            # Set up evaluation metrics
            answer_similarity_aws = mlflow.metrics.genai.answer_similarity(
                model=llm_evaluator,
                parameters={
                    "temperature": 0,
                    "max_tokens": 256,
                    "anthropic_version": "bedrock-2023-05-31",
                },
            )
    
            answer_correctness_aws = mlflow.metrics.genai.answer_correctness(
                model=llm_evaluator,
                parameters={
                    "temperature": 0,
                    "max_tokens": 256,
                    "anthropic_version": "bedrock-2023-05-31",
                },
            )
            
            answer_relevance_aws = mlflow.metrics.genai.answer_relevance(
                model=llm_evaluator,
                parameters={
                    "temperature": 0,
                    "max_tokens": 256,
                    "anthropic_version": "bedrock-2023-05-31",
                },
            )
            
            answer_faithfulness_aws = mlflow.metrics.genai.faithfulness(
                model=llm_evaluator,
                parameters={
                    "temperature": 0,
                    "max_tokens": 256,
                    "anthropic_version": "bedrock-2023-05-31",
                },
            )
            
            # Basic metrics
            from mlflow.metrics import rouge1, rougeL, token_count, latency, toxicity
            
            metrics_genai_only = [
                answer_correctness_aws, answer_similarity_aws, 
                answer_relevance_aws, answer_faithfulness_aws
            ]
            
            try:
                # Run evaluation
                results = mlflow.evaluate(
                    model=model_for_eval,
                    data=eval_df_mini,  # evaluation DataFrame
                    predictions="answer",
                    extra_metrics=metrics_genai_only,
                    evaluator_config={
                        "col_mapping": {
                            "inputs": "question",
                            "context": "context",
                            "targets": "merged_answer"
                        }
                    }
                )
    
                # Log metrics
                mlflow.log_metrics(results.metrics)
                
                # Log parameters
                mlflow_run_log_params = {
                    "database_name": opensearch_client.transport.hosts[0]['host'],
                    "database_index": index_name,
                    "embedding_model_id": embedding_model_id,
                    "text_model_id": text_model_id,
                    "embedding_sagemaker_endpoint": embedding_endpoint_name,
                    "text_sagemaker_endpoint": text_endpoint_name,
                    "chunking_strategy_type": chunking_strategy,
                    "chunking_strategy_chunk_size": chunk_size,
                    "chunking_strategy_chunk_overlap": chunk_overlap,
                    "model_type": "rag_pipeline",
                    "llm_as_judge": llm_evaluator,
                    "eval_dataset_size": len(eval_df_mini)
                }
                mlflow.log_params(mlflow_run_log_params)
                
                # Save evaluation results to output path
                if output_path:
                    os.makedirs(output_path, exist_ok=True)
                    results_path = os.path.join(output_path, "evaluation_results.json")
                    with open(results_path, "w") as f:
                        json.dump({k: str(v) for k, v in results.metrics.items()}, f, indent=2)
                    
                    # Also save a sample of predictions
                    predictions = model_for_eval(eval_df_mini)
                    predictions_path = os.path.join(output_path, "sample_predictions.csv")
                    predictions.to_csv(predictions_path, index=False)
                    
                    print(f"Evaluation results saved to {output_path}")
                    
                return results
                
            except Exception as e:
                print(f"Evaluation failed: {str(e)}")
                mlflow.log_param("evaluation_error", str(e))
                raise

def main():

    args = parse_args()
    os.environ["AWS_ROLE_ARN"] = args.role_arn

    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    
    # Set up OpenSearch client
    opensearch_client = setup_opensearch_client(args.domain_name)
    
    # Run RAG evaluation
    rag_evaluation(
        args.input_data_path,
        args.output_data_path,
        opensearch_client,
        args.llm_evaluator,
        args.embedding_endpoint_name,
        args.text_endpoint_name,
        args.embedding_model_id,
        args.text_model_id,
        args.chunking_strategy,
        args.chunk_size,
        args.chunk_overlap
    )

if __name__ == "__main__":
    main()



