"""
Status Tracker Module

This module provides a high-level interface for tracking file processing status
using DynamoDB operations. It wraps the DynamoDBOperations class to provide
specific functionality for file status management and filtering.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from dynamodb_operations import DynamoDBOperations


class StatusTracker:
    """
    High-level interface for file status tracking and management.
    
    This class provides methods for filtering new files, initializing records,
    and managing file processing status using DynamoDB as the backend.
    """
    
    def __init__(self, dynamodb_operations: DynamoDBOperations):
        """
        Initialize StatusTracker with DynamoDB operations.
        
        Args:
            dynamodb_operations: Instance of DynamoDBOperations for database access
        """
        self.dynamodb_operations = dynamodb_operations
        self.logger = logging.getLogger(__name__)
    
    def filter_new_files(self, table_name: str, file_paths: List[str]) -> Tuple[List[str], int]:
        """
        Filter out files that already exist in DynamoDB or need reprocessing.
        
        Files with 'error' or 'failed' status are considered for reprocessing.
        
        Args:
            table_name: Name of the DynamoDB table
            file_paths: List of S3 file paths to check
            
        Returns:
            Tuple of (new_files_list, existing_files_count)
        """
        self.logger.debug(f"Filtering {len(file_paths)} files for new/reprocessable items")
        
        # Get existing records from DynamoDB
        existing_records = self.dynamodb_operations.batch_get_files(
            table_name=table_name,
            file_paths=file_paths
        )
        
        new_files = []
        existing_count = 0
        
        for file_path in file_paths:
            if file_path not in existing_records:
                # File doesn't exist, add to new files
                new_files.append(file_path)
            else:
                record = existing_records[file_path]
                status = record.get('status', '')
                
                # Allow reprocessing of files with error or failed status
                if status in ['error', 'failed']:
                    new_files.append(file_path)
                    self.logger.debug(f"File {file_path} has status '{status}', allowing reprocessing")
                else:
                    existing_count += 1
                    self.logger.debug(f"File {file_path} already exists with status '{status}', skipping")
        
        self.logger.info(f"Filtered files: {len(new_files)} new/reprocessable, {existing_count} existing")
        return new_files, existing_count
    
    def initialize_file_records(self, table_name: str, file_paths: List[str], bucket_uri: str) -> int:
        """
        Initialize DynamoDB records for new files with 'processing' status.
        
        Args:
            table_name: Name of the DynamoDB table
            file_paths: List of S3 file paths to initialize
            bucket_uri: S3 bucket URI for the files
            
        Returns:
            Number of successfully processed files
        """
        self.logger.debug(f"Initializing {len(file_paths)} file records")
        
        processed_count = 0
        
        for file_path in file_paths:
            success, error = self.dynamodb_operations.insert_file_record(
                table_name=table_name,
                file_path=file_path,
                bucket_url=bucket_uri,
                status="processing"
            )
            
            if success:
                processed_count += 1
                self.logger.debug(f"Successfully initialized record for {file_path}")
            else:
                self.logger.error(f"Failed to initialize record for {file_path}: {error}")
        
        self.logger.info(f"Initialized {processed_count}/{len(file_paths)} file records")
        return processed_count
    
    def get_file_status(self, table_name: str, file_path: str) -> Dict[str, Any]:
        """
        Get the current status record for a specific file.
        
        Args:
            table_name: Name of the DynamoDB table
            file_path: S3 file path to check
            
        Returns:
            Dictionary containing file record or empty dict if not found
        """
        record = self.dynamodb_operations.check_file_exists(table_name, file_path)
        return record if record is not None else {}
    
    def get_files_by_status(self, table_name: str, file_paths: List[str], target_status: str) -> List[str]:
        """
        Filter files by their current status in DynamoDB.
        
        Args:
            table_name: Name of the DynamoDB table
            file_paths: List of S3 file paths to check
            target_status: Status to filter by (e.g., 'error', 'submitted')
            
        Returns:
            List of file paths that have the target status
        """
        self.logger.debug(f"Filtering {len(file_paths)} files by status: {target_status}")
        
        existing_records = self.dynamodb_operations.batch_get_files(
            table_name=table_name,
            file_paths=file_paths
        )
        
        matching_files = []
        for file_path, record in existing_records.items():
            if record.get('status') == target_status:
                matching_files.append(file_path)
        
        self.logger.debug(f"Found {len(matching_files)} files with status '{target_status}'")
        return matching_files
    
    def update_file_status(self, table_name: str, file_path: str, status: str, **kwargs) -> bool:
        """
        Update the status of a file record in DynamoDB.
        
        Args:
            table_name: Name of the DynamoDB table
            file_path: S3 file path to update
            status: New status value
            **kwargs: Additional parameters to pass to update_file_status
            
        Returns:
            True if update was successful, False otherwise
        """
        success, error = self.dynamodb_operations.update_file_status(
            table_name=table_name,
            file_path=file_path,
            status=status,
            **kwargs
        )
        
        if not success:
            self.logger.error(f"Failed to update status for {file_path}: {error}")
        
        return success