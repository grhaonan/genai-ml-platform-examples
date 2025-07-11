# Standard library imports
import argparse
import os
import json
from datetime import datetime
from time import gmtime, strftime
from typing import List, Dict, Union, Any
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import mlflow
from mlflow.entities import SpanType
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError
from requests_aws4auth import AWS4Auth
from datasets import load_dataset, Dataset

# LangChain imports
from langchain import hub
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# SageMaker imports
import sagemaker
from sagemaker.serializers import JSONSerializer, IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--embedding-endpoint-name", type=str, required=True)
    parser.add_argument("--text-endpoint-name", type=str, required=True)
    parser.add_argument("--domain-name", type=str, required=True)
    parser.add_argument("--index-name", type=str, default="ragops-exp-index")
    parser.add_argument("--chunking-strategy", type=str, default="RecursiveChunker")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--context-retrieval-size", type=int, default=3)
    parser.add_argument("--embedding-model-id", type=str)
    parser.add_argument("--text-model-id", type=str)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--model-dimensions", type=int, required=True)
    parser.add_argument("--llm-evaluator", type=str, default= "bedrock:/anthropic.claude-3-haiku-20240307-v1:0")
    return parser.parse_args()


class FixedSizeChunker:
    """
    A class that divides a given text into chunks of fixed size with overlap.
    This class uses the `CharacterTextSplitter` to split text at paragraph boundaries, ensuring that
    each chunk has the specified size and overlap.

    Attributes:
        chunk_size (int): The desired size of each chunk (default is 1000 characters).
        chunk_overlap (int): The number of characters to overlap between consecutive chunks (default is 200).
        text_splitter (CharacterTextSplitter): An instance of the CharacterTextSplitter to handle the splitting process.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 300):
        from langchain_text_splitters.character import CharacterTextSplitter
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )

    def chunk(self, texts: list) -> List[Dict[str, str]]:
        chunks = self.text_splitter.create_documents(texts)
        return [{"chunk_text": chunk.page_content} for chunk in chunks]

class RecursiveChunker:
    """
    A class that divides a given text into chunks using recursive splitting, 
    with the specified size and overlap between chunks. This class uses the 
    `RecursiveCharacterTextSplitter` to break down the text recursively, ensuring 
    that the chunk sizes are respected while splitting at logical points.

    Attributes:
        chunk_size (int): The desired size of each chunk (default is 4000 characters).
        chunk_overlap (int): The number of characters to overlap between consecutive chunks (default is 200).
        text_splitter (RecursiveCharacterTextSplitter): An instance of the RecursiveCharacterTextSplitter to handle the splitting process.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 300):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "."],  # Initial splitting: paragraphs-> lines -> line
            keep_separator=True,  
            is_separator_regex=False,
        )

    def chunk(self, texts: list) -> List[Dict[str, str]]:
        chunks = self.text_splitter.create_documents(texts)
        return [{"chunk_text": chunk.page_content} for chunk in chunks]


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
    # Get the current region from the session
    region = os.environ.get('AWS_REGION') 
    # Create a SageMaker session with the region
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    # Now create predictor with the session
    return Predictor(
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        sagemaker_session=sagemaker_session
    )

def setup_prompt():
    return hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def setup_graph(opensearch_client, embedding_model_predictor, text_generation_predictor, prompt, index_name, context_retrieval_size=3):
    """Set up and return the RAG graph and helper functions."""
    global graph_with_context, retrieve_context
    
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
    def retrieve(state):
        """Retrieves relevant documents for a given question using vector search."""
        try:
            document_chunks = retrieve_context(query=state["question"])
            return {"context": document_chunks}
        except Exception as e:
            raise RuntimeError(f"Document retrieval failed: {e}")
    
    @mlflow.trace(attributes={"workflow": "agentrag_generate_node"}, span_type=SpanType.AGENT)
    def generate(state):
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
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    graph_with_context = graph_builder.compile()
    
    # Return both the graph and the retrieve_context function
    return graph_with_context, retrieve_context

def ingest_documents_to_opensearch(documents, opensearch_client, embedding_model_predictor, index_name):
    """Helper function to ingest documents to OpenSearch with connection retry."""
    ingested_count = 0
    total_documents = len(documents)
    domain_name = opensearch_client.transport.hosts[0]['host']
    
    for doc_id, text_dict in enumerate(documents):
        try:
            # Get embedding
            embedding_response = embedding_model_predictor.predict({'inputs': text_dict["chunk_text"]})
            document = {
                "text": text_dict['chunk_text'],
                "vector": embedding_response[0]
            }
            
            # Try to index the document
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    opensearch_client.index(
                        index=index_name,
                        id=doc_id,
                        body=document
                    )
                    break  # Success, exit retry loop
                    
                except (ConnectionTimeout, ConnectionError) as e:
                    retry_count += 1
                    print(f"Connection error for doc {doc_id}: {str(e)}. Retry {retry_count}/{max_retries}")
                    
                    if retry_count < max_retries:
                        # Reestablish connection
                        print("Reestablishing connection...")
                        opensearch_client = setup_opensearch_client(domain_name)
                    else:
                        # Max retries reached, re-raise the exception
                        raise
            
            ingested_count += 1
            if ingested_count % 10 == 0:
                print(f"ingested {ingested_count} chunks")
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            # Continue with next document instead of stopping
            continue
    
    try:
        opensearch_client.indices.refresh(index=index_name)
    except Exception as e:
        print(f"Warning: Error refreshing index: {str(e)}")
    
    success_message = f"Successfully ingested {ingested_count} out of {total_documents} documents."
    return {"status": "success", "message": success_message, "ingested_count": ingested_count}



def lc_to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Converts LangChain messages into OpenAI-compatible format.

    Args:
        messages (List[BaseMessage]): A list of LangChain message objects 
                                      (HumanMessage, AIMessage, SystemMessage).

    Returns:
        List[Dict[str, Any]]: A list of OpenAI-style message dictionaries with 
                              'role' and 'content' fields.

    Raises:
        ValueError: If a message type is not supported.
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
            content = [{"type": "text", "text": m.content}]
        elif isinstance(m.content, list):
            content = m.content
        else:
            raise ValueError(f"Unsupported content format: {m.content}")

        formatted_messages.append({"role": role, "content": content})
    return formatted_messages


def data_preparation():
    """Data preparation step with its own MLflow run."""
    with mlflow.start_run(run_name="DataPreparation", nested=True):
        mlflow.log_param("data_type", "medical")
        mlflow.log_param("data_pii", "False")
        mlflow.log_param("data_classification", "Public")
        mlflow.log_param("data_source", "HuggingFace Hub")
        
        # Load dataset
        data_path = "qiaojin/PubMedQA"
        data_name = "pqa_artificial"
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("data_name", data_name)
        source_dataset = load_dataset(data_path, data_name)
        
        # Dataset Lineage Tracking
        hf_dataset = mlflow.data.from_pandas(
            df=source_dataset['train'].to_pandas(),
        )
        mlflow.log_input(
            hf_dataset, 
            context="DataPreprocessing",
            tags={
                "task": "medical_qa",
                "split": "train",
                "version": "1.0.0"
            }
        )
        
        # Calculate and log statistics
        stats = {}
        stats['total_entries'] = len(source_dataset['train'])
        stats['unique_questions'] = len(set(source_dataset['train']["question"]))
        
        context_lengths = [len(str(context).split()) 
                          for item in source_dataset['train']["context"]
                          for context in item['contexts']] 
        stats.update({
            'avg_context_length': np.mean(context_lengths),
        })
        
        decisions = Counter(source_dataset['train']["final_decision"])
        stats.update({
            'yes_decisions': decisions.get('yes', 0),
            'no_decisions': decisions.get('no', 0),
            'maybe_decisions': decisions.get('maybe', 0)
        })
        print("stats:", stats)
        mlflow.log_metrics({
            "total_qa_pairs": stats['total_entries'],
            "unique_questions": stats['unique_questions'],
            "avg_context_length": stats['avg_context_length'],
            "decision_yes": stats['yes_decisions'],
            "decision_no": stats['no_decisions'],
            "decision_maybe": stats['maybe_decisions']
        })
        
        # Log distribution artifacts
        plt.figure(figsize=(10,6))
        plt.hist([len(str(c).split()) for c in source_dataset['train']["context"]], bins=50)
        plt.title("Context Length Distribution")
        plt.savefig("context_len_dist.png")
        mlflow.log_artifact("context_len_dist.png")
        
        return source_dataset

def data_chunking(source_dataset, chunking_strategy="RecursiveChunker", 
                 chunk_size=500, chunk_overlap=200):
    """Data chunking step with its own MLflow run."""
    chunking_stage_stats = {}
    with mlflow.start_run(run_name="DataChunking", nested=True):
        mlflow.log_param("chunking_strategy_type", chunking_strategy)
        mlflow.log_param("chunking_strategy_chunk_size", chunk_size)
        mlflow.log_param("chunking_strategy_chunk_overlap", chunk_overlap)
        
        source_contexts = source_dataset['train']['context']
        
        if chunking_strategy=="RecursiveChunker":
            chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunking_strategy=="FixedSizeChunker":
            chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        source_contexts_chunked_count = 0
        source_contexts_chunked = []
        for context_documents in source_contexts:
            recursive_chunker_chunks = chunker.chunk(context_documents['contexts'])
            source_contexts_chunked_count = source_contexts_chunked_count + len(recursive_chunker_chunks)
            source_contexts_chunked.extend(recursive_chunker_chunks)
 
        chunking_stage_stats['total_source_contexts_entries'] = len(source_contexts)
        chunking_stage_stats['total_contexts_chunked'] = source_contexts_chunked_count
        unique_chunks = len({tuple(chunk.items()) for chunk in source_contexts_chunked})
        chunking_stage_stats['total_unique_chunks_final'] = unique_chunks
        print(f"stats: {chunking_stage_stats}")
        mlflow.log_metrics({
            "total_source_contexts_entries": chunking_stage_stats['total_source_contexts_entries'],
            "total_contexts_chunked": chunking_stage_stats['total_contexts_chunked'],
            "total_unique_chunks_final": chunking_stage_stats['total_unique_chunks_final'],
        })

        return source_contexts_chunked

def data_ingestion(source_contexts_chunked, opensearch_client,model_dimensions, embedding_model_predictor, 
                  embedding_model_id, text_model_id, embedding_endpoint_name, text_endpoint_name,
                  index_name="ragops-exp-index" ):
    """Data ingestion step with its own MLflow run."""
    with mlflow.start_run(run_name="DataIngestion", nested=True):
        mlflow.log_param("database_type", "opensearch")
        mlflow.log_param("database_name", opensearch_client.transport.hosts[0]['host'])
        mlflow.log_param("database_index", index_name)
        
        if opensearch_client.indices.exists(index=index_name):
            mlflow.log_param("database_index_creation", "False")
        else:
            mlflow.log_param("database_index_creation", "True")
            # Create index with appropriate settings
            index_body = {
                "settings": {
                    "index.knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": model_dimensions  # Match embedding model dimension
                        }
                    }
                }
            }
            opensearch_client.indices.create(
                index=index_name,
                body=index_body
            )

        mlflow.log_param("embedding_model_id", embedding_model_id)
        mlflow.log_param("text_model_id", text_model_id)
        mlflow.log_param("embedding_sagemaker_endpoint", embedding_endpoint_name)
        mlflow.log_param("text_sagemaker_endpoint", text_endpoint_name)

        result = ingest_documents_to_opensearch(
            source_contexts_chunked[:100], 
            opensearch_client, 
            embedding_model_predictor, 
            index_name
        )
        print(result)
        
        if "ingested_count" in result:
            mlflow.log_metric("ingested_count", result["ingested_count"])
        
        return index_name

def rag_retrieval(source_dataset, graph_with_context, opensearch_client, 
                  embedding_model_id, text_model_id, 
                  embedding_endpoint_name, text_endpoint_name,
                  index_name, context_retrieval_size=3):
    """RAG retrieval setup with its own MLflow run."""
    with mlflow.start_run(run_name="RAGRetrieval", nested=True):
        mlflow.log_param("database_type", "opensearch")
        mlflow.log_param("database_name", opensearch_client.transport.hosts[0]['host'])
        mlflow.log_param("database_index", index_name)
        mlflow.log_param("embedding_model_id", embedding_model_id)
        mlflow.log_param("text_model_id", text_model_id)
        mlflow.log_param("embedding_sagemaker_endpoint", embedding_endpoint_name)
        mlflow.log_param("text_sagemaker_endpoint", text_endpoint_name)
        mlflow.log_param("RAG_context_size", context_retrieval_size)
        
        # Source test queries
        TEST_SAMPLE_SIZE = 10  # First 10 questions
        sample_questions = [q for q in source_dataset["train"]["question"][:TEST_SAMPLE_SIZE]]
        mlflow.log_metrics({
            "RAG_query_size": len(sample_questions)
        })
        
        # Initialize metrics collection
        retrieval_metrics = {
            "total_queries": len(sample_questions),
            "contexts_per_query": [],
            "missing_context_count": 0,
            "context_lengths": []
        }
        query_examples = []
        
        # Store complete results for reuse in evaluation
        complete_results = []
        
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
                "contexts_sample": contexts[:1000] if isinstance(contexts, str) else "", 
                "answer": answer  
            }
            query_examples.append(query_example)
            
            # Store complete results for evaluation
            complete_results.append({
                "question": question,
                "context": contexts,
                "answer": answer
            })
        
        # Calculate derived metrics
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
        
        # Log sample results as artifact
        with open("retrieval_samples.json", "w") as f:
            json.dump(query_examples, f, indent=2)
        mlflow.log_artifact("retrieval_samples.json")
        
        return complete_results  

def rag_evaluation(source_dataset, retrieval_results, output_path, 
                  opensearch_client, llm_evaluator,embedding_model_id, text_model_id,
                  chunking_strategy, chunk_size, chunk_overlap,
                  embedding_endpoint_name, text_endpoint_name, 
                  index_name):
    """RAG evaluation with its own MLflow run."""
    # 1. Load the dataset - using the source_dataset we already have
    eval_org_dataset = source_dataset
    
    # 2. Convert to pandas DataFrame
    df = eval_org_dataset['train'].to_pandas()
    
    # 3. Merge 'final_decision' and 'long_answer' into a new column 'merged_answer'
    df['merged_answer'] = df['final_decision'].astype(str) + ', ' + df['long_answer'].astype(str)
    
    # 4. Drop unnecessary columns
    df = df.drop(columns=['final_decision', 'long_answer', 'context', 'pubid'])

    # Create mini evaluation datasets - use only the questions we have results for
    eval_df_mini = df.head(len(retrieval_results))

    
    # Create a model function that uses the pre-computed results instead of invoking the graph
    def model_for_eval(input_df):
        results = []
        for index, row in input_df.iterrows():
            # Find the matching result from retrieval_results
            matching_result = next(
                (r for r in retrieval_results if r["question"] == row["question"]), 
                {"answer": "No matching result found", "context": ""}
            )
            results.append({
                "answer": matching_result["answer"],
                "context": matching_result["context"]
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
    
    with mlflow.start_run(run_name="RAGEvaluation", nested=True):
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
    experiment_suffix = strftime('%d', gmtime())
    experiment_name = f"{experiment_suffix}-{args.experiment_name}"
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id = experiment.experiment_id
    
    print(f"Running RAG pipeline with experiment ID: {experiment_id}")

    embedding_endpoint_name = args.embedding_endpoint_name
    text_endpoint_name = args.text_endpoint_name
    index_name = args.index_name
    embedding_model_id = args.embedding_model_id
    text_model_id = args.text_model_id
    context_retrieval_size = args.context_retrieval_size
    chunking_strategy = args.chunking_strategy
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    model_dimensions = args.model_dimensions
    llm_evaluator = args.llm_evaluator

    # Set up OpenSearch client
    opensearch_client = setup_opensearch_client(args.domain_name)
    
    # Set up predictors
    embedding_model_predictor = setup_model_predictor(embedding_endpoint_name)
    text_generation_predictor = setup_model_predictor(text_endpoint_name)

    #setup autologging for langgraph traces
    mlflow.langchain.autolog()

    # Create a parent run
    with mlflow.start_run(run_name=f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as main_run:
        main_run_id = main_run.info.run_id
        print("mlflow_run", main_run_id)
        mlflow.log_param("pipeline_start_time", datetime.now().isoformat())
        
        try:
            # Execute pipeline steps
            source_dataset = data_preparation()
            
            source_contexts_chunked = data_chunking(
                source_dataset, 
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            data_ingestion(
                source_contexts_chunked, 
                opensearch_client, 
                model_dimensions,
                embedding_model_predictor,
                embedding_model_id,
                text_model_id,
                embedding_endpoint_name,
                text_endpoint_name,
                index_name,

            )
            
            # Set up the RAG pipeline components
            prompt = setup_prompt()
            graph_with_context, retrieve_context = setup_graph(
                opensearch_client, 
                embedding_model_predictor, 
                text_generation_predictor, 
                prompt,
                index_name,
                context_retrieval_size
            )
            
            # Run retrieval and store results
            retrieval_results = rag_retrieval(
                source_dataset,
                graph_with_context,
                opensearch_client,
                embedding_model_id,
                text_model_id,
                embedding_endpoint_name,
                text_endpoint_name,
                index_name,
                context_retrieval_size
            )
            
            # Pass retrieval results to evaluation
            rag_evaluation(
                source_dataset,
                retrieval_results, 
                args.output_data_path,
                opensearch_client,
                llm_evaluator,
                embedding_model_id,
                text_model_id,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
                embedding_endpoint_name,
                text_endpoint_name,
                index_name
            )
            
            # Log successful completion in parent run
            mlflow.log_param("pipeline_status", "completed")
            mlflow.log_param("pipeline_end_time", datetime.now().isoformat())
            
            print("RAG pipeline completed successfully!")
            
        except Exception as e:
            # Log failure in parent run
            mlflow.log_param("pipeline_status", "failed")
            mlflow.log_param("error_message", str(e))
            print(f"Pipeline failed: {str(e)}")
            raise
if __name__ == "__main__":
    main()