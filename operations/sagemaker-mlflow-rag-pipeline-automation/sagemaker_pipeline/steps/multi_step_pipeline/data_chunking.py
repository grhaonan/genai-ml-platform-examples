import argparse
import os
import mlflow
import numpy as np
from datasets import load_from_disk
import json
import pickle
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

class FixedSizeChunker:

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 300):
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

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 300):
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--chunking-strategy", type=str, default="RecursiveChunker")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    return parser.parse_args()

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

def data_chunking(input_path, output_path, chunking_strategy, 
                 chunk_size, chunk_overlap):
    """Data chunking step with its own MLflow run."""

    with mlflow.start_run(run_id=get_parent_id(input_path, output_path)) as run:
        run_id = run.info.run_id
        print("mlflow_run", run_id)
    
        # dataset stats calculation 
        chunking_stage_stats = {}
        with mlflow.start_run(run_name="DataChunking",nested=True):
            mlflow.log_param("chunking_strategy_type", chunking_strategy)
            mlflow.log_param("chunking_strategy_chunk_size", chunk_size)
            mlflow.log_param("chunking_strategy_chunk_overlap", chunk_overlap)
            
            # Load dataset from previous step
            source_dataset = load_from_disk(os.path.join(input_path, "dataset"))
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
    
            # Save chunked data for next step
            os.makedirs(output_path, exist_ok=True)
            
            # Save the original dataset for later use in evaluation
            source_dataset.save_to_disk(os.path.join(output_path, "original_dataset"))
            
            # Save the chunked contexts
            with open(os.path.join(output_path, "chunked_contexts.pkl"), "wb") as f:
                pickle.dump(source_contexts_chunked, f)
            
            # Save stats
            with open(os.path.join(output_path, "chunking_stats.json"), "w") as f:
                json.dump(chunking_stage_stats, f)
            
            print(f"Chunked data saved to {output_path}")
            return source_contexts_chunked

def main():
    args = parse_args()
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    
    # Run data chunking
    data_chunking(
        args.input_data_path, 
        args.output_data_path,
        args.chunking_strategy,
        args.chunk_size,
        args.chunk_overlap
    )

if __name__ == "__main__":
    main()