import os
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):

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

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):

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

def chunk_langchain_document(documents: List[Document], chunking: str = "fixed", chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, str]]:
    """
    Retrieves a text object, processes it, and returns a list of text chunks 
    based on the specified chunking strategy.

    This function supports two chunking strategies: 
    - "fixed": Chunks the text into fixed-size chunks with overlap using the FixedSizeChunker.
    - "recursive": Recursively splits the text into chunks with overlap using the RecursiveChunker.

    Args:
        documents (List[str]): The list of LangChain documents to chunk.
        chunking (str): The chunking strategy to use ("fixed" or "recursive"). Defaults to "fixed".
        chunk_size (int): Chunk size.
        chunk_overlap (int): Chunk overlap.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing a chunk of the text.
        If an invalid chunking strategy is provided, an error message is returned in a dictionary.

    Raises:
        ValueError: If an invalid chunking strategy is provided.
        Exception: If any other error occurs during chunking.
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
            fixed_size_chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = fixed_size_chunker.chunk(documents)
            return chunks
        except Exception as e:
            raise Exception(f"Error during fixed-size chunking: {e}")

    elif chunking == "recursive":
        try:
            recursive_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = recursive_chunker.chunk(documents)
            return chunks
        except Exception as e:
            raise Exception(f"Error during recursive chunking: {e}")
    
    else:
        raise ValueError(f"Invalid chunking strategy: '{chunking}'. Must be 'fixed' or 'recursive'.")

from langchain_core.documents.base import Document
from langchain.document_loaders import PyPDFLoader

def process_pdf(filename: str, chunking: str = "fixed", chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, str]]:
    """
    Process a PDF file to extract a list of pages using LangChain's PyPDFLoader
    """
    local_data_path = "./data/"
    filepath = os.path.join(local_data_path, filename)
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    chunks = chunk_langchain_document(
        documents=pages,
        chunking=chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return chunks

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from typing import List, Dict, Union, Any

def langchain_to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
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