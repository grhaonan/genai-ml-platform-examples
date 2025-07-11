import os
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import time
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel

# Initialize SageMaker session
session = boto3.Session()
sm_client = session.client('sagemaker')
region = session.region_name
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

def create_opensearch_domain(domain_name):
    """
    Create a simple OpenSearch domain using the SageMaker execution role.
    """
    # Get SageMaker execution role
    from sagemaker import get_execution_role
    role_arn = get_execution_role()
    
    # Get region and account ID
    session = boto3.session.Session()
    region = session.region_name
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    
    # Create OpenSearch client
    client = boto3.client('opensearch', region_name=region)
    
    # Policy that allows access from SageMaker role
    access_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "AWS": role_arn
                },
                "Action": "es:*",
                "Resource": f"arn:aws:es:{region}:{account_id}:domain/{domain_name}/*"
            }
        ]
    }
    
    try:
        print(f"Creating OpenSearch domain '{domain_name}'...")
        response = client.create_domain(
            DomainName=domain_name,
            EngineVersion='OpenSearch_2.5',
            ClusterConfig={
                'InstanceType': 't3.small.search',
                'InstanceCount': 1,
                'DedicatedMasterEnabled': False
            },
            EBSOptions={
                'EBSEnabled': True,
                'VolumeType': 'gp3',
                'VolumeSize': 10
            },
            AccessPolicies=json.dumps(access_policy),
            EncryptionAtRestOptions={'Enabled': True},
            NodeToNodeEncryptionOptions={'Enabled': True},
            DomainEndpointOptions={'EnforceHTTPS': True},
            AdvancedSecurityOptions={'Enabled': False}  # Using IAM auth instead
        )
        
        print("Domain creation initiated. This will take 15-20 minutes.")
        print("Run the next cell for status updates.")
        
        # Return basic info
        return {
            "domain_name": domain_name,
            "region": region,
            "role_arn": role_arn
        }
        
    except client.exceptions.ResourceAlreadyExistsException:
        print(f"Domain '{domain_name}' already exists.")
        return {
            "domain_name": domain_name,
            "region": region,
            "role_arn": role_arn,
            "status": "already_exists"
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def get_domain_status(domain_name):
    """Check the status of an OpenSearch domain"""
    session = boto3.session.Session()
    region = session.region_name
    client = boto3.client('opensearch', region_name=region)
    
    try:
        response = client.describe_domain(DomainName=domain_name)
        status = response['DomainStatus']
        
        if 'Endpoint' in status:
            print(f"\nDomain '{domain_name}' is active!")
            print(f"Endpoint: https://{status['Endpoint']}")
            print(f"Dashboard: https://{status['Endpoint']}/_dashboards/")
            print("Authentication: Using SageMaker execution role")
        else:
            print(f"Domain '{domain_name}' is still being created.")
        
        return status
    except Exception as e:
        print(f"Error getting domain status: {str(e)}")
        return None

def check_endpoint_status(endpoint_name):
    """Check the status of a SageMaker endpoint"""
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"Endpoint {endpoint_name} status: {status}")
        return status
    except Exception as e:
        print(f"Error checking endpoint status: {str(e)}")
        return None


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

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, content: str) -> List[Dict[str, str]]:

        chunks = self.text_splitter.split_documents(content)

        return [{"chunk": chunk.page_content} for chunk in chunks]

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

    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, content: str) -> List[Dict[str, str]]:

        chunks = self.text_splitter.split_documents(content)

        return [{"chunk": chunk.page_content} for chunk in chunks]

"""
Helper functions
"""

def deploy_jumpstart_model(model_id: str,
                 instance_type: str,
                 endpoint_name_base: str,
                 model_version: str = "*") -> str:
    """
    Deploys a SageMaker JumpStart model.

    Args:
        model_id (str): The JumpStart model ID.
        instance_type (str): The SageMaker instance type.
        endpoint_name_base (str): Prefix for the endpoint name.
        model_version (str): The version of the model to deploy.

    Returns:
        str: Name of the deployed real-time endpoint
    """

    try:
        # Check if endpoint already exists
        sm_client.describe_endpoint(EndpointName=endpoint_name_base)
        print(f"Endpoint {endpoint_name_base} already exists.")
        return endpoint_name_base
    except sm_client.exceptions.ClientError:
        # Endpoint doesn't exist, proceed with creation
        pass
        
    try:
        endpoint_name = sagemaker.utils.unique_name_from_base(endpoint_name_base)
        model = JumpStartModel(
            model_id=model_id,
            model_version=model_version,
            instance_type=instance_type
        )
        model.deploy(
            accept_eula=True,
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        return endpoint_name
            
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        raise
    



from urllib.request import urlretrieve

def download_pdfs(urls: List[str], filenames: List[str]):
    """
    Download PDF files from urls
    """
    local_data_path = "./data/"
    os.makedirs(local_data_path, exist_ok=True)
    
    for idx, url in enumerate(urls):
        file_path = os.path.join(local_data_path, filenames[idx])
        urlretrieve(url, file_path)

from langchain_core.documents.base import Document
from typing import List, Dict
from utils import FixedSizeChunker, RecursiveChunker
from botocore.config import Config

def chunk_langchain_document(documents: List[Document], chunking: str = "fixed") -> List[Dict[str, str]]:
    """
    Retrieves a text object from an S3 bucket, processes it, and returns a list of text chunks 
    based on the specified chunking strategy.

    This function supports two chunking strategies: 
    - "fixed": Chunks the text into fixed-size chunks with overlap using the FixedSizeChunker.
    - "recursive": Recursively splits the text into chunks with overlap using the RecursiveChunker.

    Args:
        documents (List[str]): The list of LangChain documents to chunk.
        chunking (str): The chunking strategy to use ("fixed" or "recursive"). Defaults to "fixed".

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing a chunk of the text.
        If an invalid chunking strategy is provided, an error message is returned in a dictionary.

    Raises:
        ValueError: If an invalid chunking strategy is provided.
        Exception: If any error occurs while interacting with the S3 bucket.
    """
    MAX_RETRIES = 5
    retry_config = Config(
        retries={
            "max_attempts": MAX_RETRIES,
            "mode": "standard"
        }
    )

    if chunking == "fixed":
        try:
            fixed_size_chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)
            chunks = fixed_size_chunker.chunk(documents)
            return chunks
        except Exception as e:
            raise Exception(f"Error during fixed-size chunking: {e}")

    elif chunking == "recursive":
        try:
            recursive_chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
            chunks = recursive_chunker.chunk(documents)
            return chunks
        except Exception as e:
            raise Exception(f"Error during recursive chunking: {e}")
    
    else:
        raise ValueError(f"Invalid chunking strategy: '{chunking}'. Must be 'fixed' or 'recursive'.")

from langchain_core.documents.base import Document
from langchain.document_loaders import PyPDFLoader

def process_pdf(filename: str, chunking: str = "fixed") -> List[Dict[str, str]]:
    """
    Process a PDF file to extract a list of pages using LangChain's PyPDFLoader
    """
    local_data_path = "./data/"
    filepath = os.path.join(local_data_path, filename)
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    chunks = chunk_langchain_document(documents=pages, chunking=chunking)

    return chunks

def delete_s3_prefix(bucket_name, prefix):
    """
    Deletes all objects under the specified prefix in the given S3 bucket.

    Parameters:
    - bucket_name: str, name of the S3 bucket
    - prefix: str, prefix (folder path) inside the bucket to delete
    """
    s3 = boto3.client('s3')

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    delete_us = dict(Objects=[])

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                delete_us['Objects'].append({'Key': obj['Key']})

                if len(delete_us['Objects']) >= 1000:
                    s3.delete_objects(Bucket=bucket_name, Delete=delete_us)
                    delete_us = dict(Objects=[])

    if delete_us['Objects']:
        s3.delete_objects(Bucket=bucket_name, Delete=delete_us)

    print(f"All objects under prefix '{prefix}' in bucket '{bucket_name}' have been deleted.")

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from typing import List, Dict, Union, Any

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