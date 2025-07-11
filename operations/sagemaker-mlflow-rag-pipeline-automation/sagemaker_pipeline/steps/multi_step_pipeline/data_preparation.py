import argparse
import os
import mlflow
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_dataset
from datetime import datetime
from time import gmtime, strftime
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--parent-run-name", type=str, required=True)
    parser.add_argument("--github-repository", type=str)
    parser.add_argument("--github-action-name", type=str)
    parser.add_argument("--github-workflow-id", type=str)
    parser.add_argument("--sagemaker-pipeline-name", type=str)

    return parser.parse_args()

def data_preparation(output_path, parent_run_name, github_repository, github_action_name, github_workflow_id,
                     sagemaker_pipeline_name):

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"Created parent MLflow run with ID: {parent_run_id}")
        
        # Save the run ID to a file in the output directory
        # This directory will be uploaded to S3 as specified in ProcessingOutput
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "parent_run_id.txt"), "w") as f:
            f.write(parent_run_id)
        
        # Log basic parameters in parent run
        mlflow.log_param("pipeline_start_time", datetime.now().isoformat())
        mlflow.log_param("pipeline_status", "started")
        mlflow.log_param("github_repository", github_repository)
        mlflow.log_param("github_action_name", github_action_name)
        mlflow.log_param("github_workflow_id", github_workflow_id)
        mlflow.log_param("sagemaker_pipeline_name", sagemaker_pipeline_name)
        mlflow.log_param("pipeline_type", "multi")

        #Start the Data preparation step
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
            
            # Save dataset to output path
            os.makedirs(output_path, exist_ok=True)
            source_dataset.save_to_disk(os.path.join(output_path, "dataset"))
            
            # Save stats for next step
            with open(os.path.join(output_path, "dataset_stats.json"), "w") as f:
                json.dump(stats, f)
            
            print(f"Dataset saved to {output_path}")
            return source_dataset

def main():
    args = parse_args()

    env_vars = dict(sorted(os.environ.items()))
    
    # Print each variable
    for key, value in env_vars.items():
        # Truncate very long values for readability
        print(f"{key}: {value}")
    
    experiment_suffix = strftime('%d', gmtime())

    # Set up MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    mlflow.set_experiment(experiment_name=f"multi-{experiment_suffix}-{args.experiment_name}")
    
    # Run data preparation
    data_preparation(args.output_data_path, args.parent_run_name,
                     args.github_repository, args.github_action_name,
                     args.github_workflow_id, args.sagemaker_pipeline_name)


if __name__ == "__main__":
    main()