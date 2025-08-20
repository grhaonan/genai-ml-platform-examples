import argparse
import os
import shutil
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
    parser.add_argument("--index-name", type=str, default="ragops-exp-index")
    parser.add_argument("--context-retrieval-size", type=int, default=3)
    parser.add_argument("--embedding-model-id", type=str, required=True)
    parser.add_argument("--text-model-id", type=str, required=True)
    return parser.parse_args()

def setup_opensearch_client(domain_name):
    """Set up and return an OpenSearch client."""
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

def setup_model_predictor(endpoint_name):
    region = os.environ.get('AWS_REGION')
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    return Predictor(
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        sagemaker_session=sagemaker_session
    )

def setup_prompt():
    return hub.pull("rlm/rag-prompt")



def lc_to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Converts LangChain messages into OpenAI-compatible format.
    """
    formatted_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported message type: {type(m)}")

        # Normalize content
        if isinstance(m.content, str):
            content = m.content
        else:
            raise ValueError(f"Unsupported content format: {m.content}")

        formatted_messages.append({"role": role, "content": content})
    return formatted_messages

# Define application state structure
class State(TypedDict):
    question: str
    context: List[str]
    answer: str

def setup_graph(opensearch_client, embedding_model_predictor, text_generation_predictor, prompt, index_name, context_retrieval_size=3):
    """Set up and return the RAG graph and helper functions."""
    
    # Define retrieve_context function
    def retrieve_context(query):
        embedding_response = embedding_model_predictor.predict({'inputs': query})
        vector = embedding_response[0]
        
        search_query = {
            "size": context_retrieval_size,
            "query": {
                "knn": {
                    "vector": {
                        "vector": vector,
                        "k": context_retrieval_size
                    }
                }
            }
        }
        
        response = opensearch_client.search(
            body=search_query,
            index=index_name
        )
        
        context_texts = []
        for hit in response['hits']['hits']:
            context_texts.append(hit['_source']['text'])
            
        return context_texts
    
    # Define the state workflow functions that will be used in the graph
    @mlflow.trace(attributes={"workflow": "agentrag_retrieve_node"}, span_type=SpanType.AGENT)
    def retrieve(state: State) -> Dict[str,str]:
        """Retrieves relevant documents for a given question using vector search."""
        try:
            document_chunks = retrieve_context(query=state["question"])
            return {"context": document_chunks}
        except Exception as e:
            raise RuntimeError(f"Document retrieval failed: {e}")
    
    @mlflow.trace(attributes={"workflow": "agentrag_generate_node"}, span_type=SpanType.AGENT)
    def generate(state: State) -> Dict[str,str]:
        """Generates an answer using retrieved context and user question."""
        try:
            docs_content = "\n\n".join(doc for doc in state["context"])
            
            # Generate LangChain messages and convert to OpenAI format
            lc_messages = prompt.invoke({
                "question": state["question"],
                "context": docs_content
            }).to_messages()
            openai_messages = lc_to_openai_messages(lc_messages)
            
            request = {
                "messages": openai_messages,
                "temperature": 0.2,
                "max_new_tokens": 512,
            }
            
            response = text_generation_predictor.predict(request)
            
            # Validate response structure
            if ("choices" not in response 
                    or not response["choices"] 
                    or "message" not in response["choices"][0]
                    or "content" not in response["choices"][0]["message"]):
                raise ValueError("Unexpected response format from text generation predictor.")
            
            return {
                "answer": response["choices"][0]["message"]["content"],
                "context": docs_content
            }
        
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")
    
    # Create and compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    
    
    # Return both the graph and the retrieve_context function
    return graph_builder

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


def rag_retrieval(input_path, output_path, opensearch_client, embedding_model_predictor, 
                 text_generation_predictor, embedding_model_id, text_model_id, 
                 embedding_endpoint_name, text_endpoint_name, index_name, context_retrieval_size):
    """RAG retrieval setup with its own MLflow run."""

    mlflow.langchain.autolog()
    with mlflow.start_run(run_id=get_parent_id(input_path, output_path)) as run:
        run_id = run.info.run_id
        print("mlflow_run", run_id)

        with mlflow.start_run(run_name="RAGRetrieval",nested=True):
            mlflow.log_param("database_type", "opensearch")
            mlflow.log_param("database_name", opensearch_client.transport.hosts[0]['host'])
            mlflow.log_param("database_index", index_name)
            mlflow.log_param("embedding_model_id", embedding_model_id)
            mlflow.log_param("text_model_id", text_model_id)
            mlflow.log_param("embedding_sagemaker_endpoint", embedding_endpoint_name)
            mlflow.log_param("text_sagemaker_endpoint", text_endpoint_name)
            mlflow.log_param("RAG_context_size", context_retrieval_size)
            
            # Load dataset from previous step
            source_dataset = load_from_disk(os.path.join(input_path, "original_dataset"))
            
            # Set up the RAG pipeline components
            prompt = setup_prompt()
            graph_builder = setup_graph(
                opensearch_client, 
                embedding_model_predictor, 
                text_generation_predictor, 
                prompt,
                index_name,
                context_retrieval_size
            )
            
            # Source test queries
            TEST_SAMPLE_SIZE = 10  # First 10 questions
            sample_questions = [q for q in source_dataset["train"]["question"][:TEST_SAMPLE_SIZE]]
            mlflow.log_metrics({
                "RAG_query_size": len(sample_questions)
            })
            
            print(opensearch_client)

            # Test connection by listing indices
            try:
                response = opensearch_client.indices.get_alias(index="*")
                print("Indices:", response)
            except Exception as e:
                print("Error connecting to OpenSearch:", str(e))
            
            # Test index
            print(opensearch_client.indices.exists(index=index_name))
                        
            # Initialize metrics collection
            retrieval_metrics = {
                "total_queries": len(sample_questions),
                "contexts_per_query": [],
                "missing_context_count": 0,
                "context_lengths": []
            }
            query_examples = []

            graph_with_context = graph_builder.compile()

            # Store all retrieval results for evaluation
            all_retrieval_results = []
            
            # Test retrieval for each sample question
            for idx, question in enumerate(sample_questions):
                response = graph_with_context.invoke({"question": question})
                contexts = response.get("context", [])
                answer = response.get("answer", "")
                
                # Collect metrics
                num_contexts = len(contexts.split('\n\n')) if isinstance(contexts, str) else 0
                retrieval_metrics["contexts_per_query"].append(num_contexts)
                
                if isinstance(contexts, str):
                    retrieval_metrics["context_lengths"].extend([len(c) for c in contexts.split('\n\n')])
                
                if num_contexts == 0:
                    retrieval_metrics["missing_context_count"] += 1
                
                # Store examples for artifact logging
                query_example = {
                    "question": question,
                    "context_count": num_contexts,
                    "contexts_sample": contexts[:1000] if isinstance(contexts, str) else "",  # Truncate for logging
                    "answer": answer
                }
                query_examples.append(query_example)
                
                # Store full results for evaluation
                all_retrieval_results.append({
                    "question": question,
                    "context": contexts,
                    "answer": answer
                })
            
            # Calculate derived metrics
            import numpy as np
            avg_contexts = np.mean(retrieval_metrics["contexts_per_query"]) if retrieval_metrics["contexts_per_query"] else 0
            missing_context_rate = retrieval_metrics["missing_context_count"] / retrieval_metrics["total_queries"]
            avg_context_length = np.mean(retrieval_metrics["context_lengths"]) if retrieval_metrics["context_lengths"] else 0
            print(f"Avg contexts: {avg_contexts}, Missing rate: {missing_context_rate}, Avg length: {avg_context_length}")
    
            # Log metrics to MLflow
            mlflow.log_metrics({
                "avg_contexts_per_query": avg_contexts,
                "missing_context_rate": missing_context_rate,
                "avg_context_length": avg_context_length,
                "total_missing_context": retrieval_metrics["missing_context_count"]
            })
            
            # Save results for next step
            os.makedirs(output_path, exist_ok=True)
            
            # Log sample results as artifact
            with open(os.path.join(output_path, "retrieval_samples.json"), "w") as f:
                json.dump(query_examples, f, indent=2)
            mlflow.log_artifact(os.path.join(output_path, "retrieval_samples.json"))
            
            # Save the full retrieval results for evaluation
            with open(os.path.join(output_path, "all_retrieval_results.json"), "w") as f:
                json.dump(all_retrieval_results, f, indent=2)
            
            # Save the graph configuration for the evaluation step
            graph_config = {
                "index_name": index_name,
                "context_retrieval_size": context_retrieval_size
            }
            with open(os.path.join(output_path, "graph_config.json"), "w") as f:
                json.dump(graph_config, f)
            
            # Copy the original dataset for next steps
            # os.system(f"cp -r {input_path}/original_dataset {output_path}/")
            # Validate paths first
            if not os.path.exists(input_path):
                raise ValueError(f"Input path does not exist: {input_path}")

            # Perform the copy operation safely
            shutil.copytree(
                os.path.join(input_path, "original_dataset"),
                os.path.join(output_path, "original_dataset")
            )
            
            print(f"Retrieval results saved to {output_path}")
            return query_examples


def main():
    args = parse_args()
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    
    # Set up OpenSearch client
    opensearch_client = setup_opensearch_client(args.domain_name)
    
    # Set up predictors
    embedding_model_predictor = setup_model_predictor(args.embedding_endpoint_name)
    text_generation_predictor = setup_model_predictor(args.text_endpoint_name)
    
    # Run RAG retrieval
    rag_retrieval(
        args.input_data_path,
        args.output_data_path,
        opensearch_client,
        embedding_model_predictor,
        text_generation_predictor,
        args.embedding_model_id,
        args.text_model_id,
        args.embedding_endpoint_name,
        args.text_endpoint_name,
        args.index_name,
        args.context_retrieval_size
    )

if __name__ == "__main__":
    main()