"""
End-to-End Tests for DynamoDB Operations

This module contains integration tests that make real calls to DynamoDB API
to test the batch_get_files functionality and other DynamoDB operations.

Requirements:
- AWS credentials configured
- DynamoDB table created (or will be created during test)
- Proper IAM permissions for DynamoDB operations
"""

import pytest
import boto3
import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from botocore.exceptions import ClientError

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lambda', 'shared'))
from dynamodb_operations import DynamoDBOperations


def delete_all_dynamodb_table_data(table_name: str, region_name: str = 'us-east-1') -> bool:
    """
    Standalone utility function to delete all data from a DynamoDB table.
    
    This function can be imported and used independently for cleanup operations.
    
    Args:
        table_name (str): Name of the DynamoDB table to clear
        region_name (str): AWS region name (default: 'us-east-1')
        
    Returns:
        bool: True if successful, False if any errors occurred
        
    Example:
        from tests.lambda.e2e.test_dynamodb_operations import delete_all_dynamodb_table_data
        success = delete_all_dynamodb_table_data('my-test-table', 'us-west-2')
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create DynamoDB client
        dynamodb_client = boto3.client('dynamodb', region_name=region_name)
        
        logger.info(f"Starting to delete all data from table: {table_name}")
        
        # Scan the table to get all items
        paginator = dynamodb_client.get_paginator('scan')
        page_iterator = paginator.paginate(TableName=table_name)
        
        total_deleted = 0
        
        for page in page_iterator:
            items = page.get('Items', [])
            
            if not items:
                continue
            
            # Prepare batch delete requests
            delete_requests = []
            for item in items:
                # Extract the primary key (file_path in our case)
                delete_requests.append({
                    'DeleteRequest': {
                        'Key': {
                            'file_path': item['file_path']
                        }
                    }
                })
            
            # Process deletes in batches of 25 (DynamoDB limit)
            batch_size = 25
            for i in range(0, len(delete_requests), batch_size):
                batch = delete_requests[i:i + batch_size]
                
                try:
                    response = dynamodb_client.batch_write_item(
                        RequestItems={
                            table_name: batch
                        }
                    )
                    
                    # Handle unprocessed items
                    unprocessed = response.get('UnprocessedItems', {})
                    if unprocessed:
                        logger.warning(f"Some items were not processed in batch delete: {len(unprocessed)}")
                        # Could implement retry logic here if needed
                    
                    total_deleted += len(batch)
                    
                except ClientError as e:
                    logger.error(f"Error in batch delete: {e}")
                    return False
        
        logger.info(f"Successfully deleted {total_deleted} items from table: {table_name}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"DynamoDB error deleting all data ({error_code}): {error_message}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error deleting all table data: {str(e)}", exc_info=True)
        return False


class TestDynamoDBOperationsE2E:
    """End-to-end tests for DynamoDB operations with real AWS API calls."""
    
    @pytest.fixture(scope="class")
    def aws_region(self):
        """Get AWS region from environment or use default."""
        return os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    
    @pytest.fixture(scope="class")
    def account_id(self):
        """Get AWS account ID."""
        try:
            sts_client = boto3.client('sts')
            response = sts_client.get_caller_identity()
            return response['Account']
        except Exception:
            return "111111111"  # Fallback for testing
    
    @pytest.fixture(scope="class")
    def dynamodb_client(self, aws_region):
        """Create DynamoDB client for test setup/teardown."""
        return boto3.client('dynamodb', region_name=aws_region)
    
    @pytest.fixture
    def db_operations(self, aws_region, account_id):
        """Create DynamoDBOperations instance."""
        return DynamoDBOperations(region_name=aws_region, account_id=account_id)
    
    @pytest.fixture
    def logger(self):
        """Create logger for test methods."""
        return logging.getLogger(__name__)
    
    def delete_all_table_data(self, table_name: str, dynamodb_client, logger) -> bool:
        """
        Delete all data from a DynamoDB table.
        
        This function scans the entire table and deletes all items in batches.
        Useful for test cleanup and data reset operations.
        
        Args:
            table_name (str): Name of the DynamoDB table to clear
            dynamodb_client: Boto3 DynamoDB client
            
        Returns:
            bool: True if successful, False if any errors occurred
        """
        try:
            logger.info(f"Starting to delete all data from table: {table_name}")
            
            # Scan the table to get all items
            paginator = dynamodb_client.get_paginator('scan')
            page_iterator = paginator.paginate(TableName=table_name)
            
            total_deleted = 0
            
            for page in page_iterator:
                items = page.get('Items', [])
                
                if not items:
                    continue
                
                # Prepare batch delete requests
                delete_requests = []
                for item in items:
                    # Extract the primary key (file_path in our case)
                    delete_requests.append({
                        'DeleteRequest': {
                            'Key': {
                                'file_path': item['file_path']
                            }
                        }
                    })
                
                # Process deletes in batches of 25 (DynamoDB limit)
                batch_size = 25
                for i in range(0, len(delete_requests), batch_size):
                    batch = delete_requests[i:i + batch_size]
                    
                    try:
                        response = dynamodb_client.batch_write_item(
                            RequestItems={
                                table_name: batch
                            }
                        )
                        
                        # Handle unprocessed items
                        unprocessed = response.get('UnprocessedItems', {})
                        if unprocessed:
                            logger.warning(f"Some items were not processed in batch delete: {len(unprocessed)}")
                            # Could implement retry logic here if needed
                        
                        total_deleted += len(batch)
                        
                    except ClientError as e:
                        logger.error(f"Error in batch delete: {e}")
                        return False
            
            logger.info(f"Successfully deleted {total_deleted} items from table: {table_name}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"DynamoDB error deleting all data ({error_code}): {error_message}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error deleting all table data: {str(e)}", exc_info=True)
            return False

    # @pytest.mark.skip(reason="Large batch test - skip for regular test runs")
    def test_delete_all_table_data(self, db_operations, dynamodb_client, logger):
        test_table = os.environ.get('TEST_DYNAMODB_TABLE', 'AwsBlogSagemakerStack-FileProcessingStatusTable3F2FBEB5-NCWLYUKVC0WJ')
        # Delete all data
        success = self.delete_all_table_data(test_table, dynamodb_client, logger)
        assert success, "Failed to delete all table data"

    def test_select_data_by_inference_id(self, db_operations: DynamoDBOperations, logger):
        """
        Test selecting DynamoDB data by specific inference_id.
        
        This test queries for records with inference_id: 26a22a80-1078-4404-84d9-3b6f63a6dea6
        """
        test_table = os.environ.get('TEST_DYNAMODB_TABLE', 'AwsBlogSagemakerStack-FileProcessingStatusTable3F2FBEB5-NCWLYUKVC0WJ')
        target_inference_id = "55b3947c-4385-4db5-9c7b-f1acaa9c50e4"

        logger.info(f"Testing query for inference_id: {target_inference_id}")
        
        # Test find_record_by_inference_id method (returns first match)
        record = db_operations.find_record_by_inference_id(test_table, target_inference_id)
        logger.info(f"Found record by inference_id: {record}")

    def test_update_job_completion_status(self, db_operations: DynamoDBOperations, logger, aws_region, account_id):
        # get TEST_DYNAMODB_TABLE from env
        test_dynamodb_table = os.environ.get('TEST_DYNAMODB_TABLE', '')

        result = db_operations.update_job_completion_status(
            table_name=test_dynamodb_table,
            file_path=f"s3://sagemaker-{aws_region}-{account_id}/parakeet-asr/single-file-folder/sample_000000.wav",
            status="completed",
            original_content='original_content',
            summarized_content='summarized_content'
        )

        print(result)


        

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v -s"])