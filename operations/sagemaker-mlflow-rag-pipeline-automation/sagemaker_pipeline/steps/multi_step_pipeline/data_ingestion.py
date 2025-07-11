import argparse
import os
import mlflow
import boto3
import pickle
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import sagemaker
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--embedding-endpoint-name", type=str, required=True)
    parser.add_argument("--domain-name", type=str, required=True)
    parser.add_argument("--index-name", type=str, default="ragops-exp-index")
    parser.add_argument("--embedding-model-id", type=str, required=True)
    parser.add_argument("--model-dimensions", type=int, required=True)
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
        
def data_ingestion(input_path, output_path, opensearch_client, embedding_model_predictor, 
                  embedding_model_id, embedding_endpoint_name, 
                   model_dimensions,index_name):
    """Data ingestion step with its own MLflow run."""
    with mlflow.start_run(run_id=get_parent_id(input_path, output_path)) as run:
        run_id = run.info.run_id
        print("mlflow_run", run_id)

        #Start data ingestion step
        with mlflow.start_run(run_name="DataIngestion",nested=True):
            mlflow.log_param("database_type", "opensearch")
            mlflow.log_param("database_name", opensearch_client.transport.hosts[0]['host'])
            mlflow.log_param("database_index", index_name)
            
            # Load chunked data from previous step
            with open(os.path.join(input_path, "chunked_contexts.pkl"), "rb") as f:
                source_contexts_chunked = pickle.load(f)
            
            if opensearch_client.indices.exists(index=index_name):
                mlflow.log_param("database_index_creation", "False")
            else:
                mlflow.log_param("database_index_creation", "True")
                # Create index with appropriate settings
                index_body = {
                    "settings": {
                        "index.knn": True,
                        "number_of_shards": 1,
                        "number_of_replicas": 2
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
            mlflow.log_param("embedding_sagemaker_endpoint", embedding_endpoint_name)
    
            # Ingest documents - using only first 100 chunks to save time for testing
            result = ingest_documents_to_opensearch(
                source_contexts_chunked[:100], 
                opensearch_client, 
                embedding_model_predictor, 
                index_name
            )
            print(result)
            
            if "ingested_count" in result:
                mlflow.log_metric("ingested_count", result["ingested_count"])
            
            # Save ingestion results
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "ingestion_results.json"), "w") as f:
                json.dump(result, f)
            
            # Copy the original dataset for next steps
            os.system(f"cp -r {input_path}/original_dataset {output_path}/")
            
            print(f"Ingestion results saved to {output_path}")
            return index_name

def main():
    args = parse_args()
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    
    # Set up OpenSearch client
    opensearch_client = setup_opensearch_client(args.domain_name)
    
    # Set up embedding model predictor
    embedding_model_predictor = setup_model_predictor(args.embedding_endpoint_name)
    
    # Run data ingestion
    data_ingestion(
        args.input_data_path,
        args.output_data_path,
        opensearch_client,
        embedding_model_predictor,
        args.embedding_model_id,
        args.embedding_endpoint_name,
        args.model_dimensions,
        args.index_name
    )

if __name__ == "__main__":
    main()