"""
DynamoDB Operations

This module handles all DynamoDB-related operations including client initialization,
record management, and status tracking with graceful degradation.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class DynamoDBOperations:
    """Handles DynamoDB operations for file status tracking."""
    
    def __init__(self, region_name: str, account_id: str):
        self.region_name = region_name
        self.account_id = account_id
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    def initialize_client(self) -> Dict[str, Any]:
        """
        Initialize DynamoDB client with proper error handling.
        
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        try:
            session = boto3.Session()
            self._client = session.client('dynamodb', region_name=self.region_name)
            self.logger.info(f"DynamoDB client initialized successfully for account {self.account_id} in region {self.region_name}")
            return {
                "is_success": True,
                "data": self._client,
                "error_code": None,
                "error_message": None
            }
        except NoCredentialsError:
            error_msg = "AWS credentials not found. Please configure AWS credentials."
            self.logger.error(error_msg)
            return {
                "is_success": False,
                "data": None,
                "error_code": "CREDENTIALS_ERROR",
                "error_message": error_msg
            }
        except Exception as e:
            error_msg = f"Failed to initialize DynamoDB client: {str(e)}"
            self.logger.error(error_msg)
            return {
                "is_success": False,
                "data": None,
                "error_code": "INITIALIZATION_ERROR",
                "error_message": error_msg
            }
    
    @property
    def client(self):
        """Get DynamoDB client, initializing if necessary."""
        if self._client is None:
            result = self.initialize_client()
            if not result["is_success"]:
                # Log error but return None to allow graceful degradation
                self.logger.error(f"Failed to initialize DynamoDB client: {result['error_message']}")
                return None
        return self._client
    
    def check_file_exists(self, table_name: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Check if a file path exists in the DynamoDB table and return its status.
        
        Args:
            table_name (str): Name of the DynamoDB table
            file_path (str): S3 file path to check
            
        Returns:
            Optional[Dict[str, Any]]: Record data if exists, None if not found
            
        Note:
            This function implements graceful degradation - if DynamoDB operations fail,
            it logs the error and returns None to allow processing to continue.
        """
        self.logger.debug(f"Checking if file exists in DynamoDB: {file_path}")
        
        try:
            response = self.client.get_item(
                TableName=table_name,
                Key={
                    'file_path': {'S': file_path}
                }
            )
            
            if 'Item' in response:
                # Convert DynamoDB item format to Python dict
                item = {}
                for key, value in response['Item'].items():
                    if 'S' in value:
                        item[key] = value['S']
                    elif 'N' in value:
                        item[key] = value['N']
                    # Add other type conversions as needed
                
                self.logger.debug(f"File found in DynamoDB with status: {item.get('status', 'unknown')}")
                return item
            else:
                self.logger.debug(f"File not found in DynamoDB: {file_path}")
                return None
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            error_msg = f"DynamoDB get_item error ({error_code}): {error_message} for file: {file_path}"
            self.logger.error(error_msg)
            # Graceful degradation - return None to allow processing to continue
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error checking file in DynamoDB: {str(e)} for file: {file_path}"
            self.logger.error(error_msg, exc_info=True)
            # Graceful degradation - return None to allow processing to continue
            return None
    
    def insert_file_record(self, table_name: str, file_path: str, bucket_url: str, 
                          status: str = "processing", inference_id: Optional[str] = None,
                          output_location: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Insert a new file record in DynamoDB with the specified status.
        
        Args:
            table_name (str): Name of the DynamoDB table
            file_path (str): S3 file path to insert (primary key)
            bucket_url (str): S3 bucket URL where the file is located
            status (str): Initial status (default: "processing")
            inference_id (Optional[str]): SageMaker inference ID if available
            output_location (Optional[str]): SageMaker output location if available
            
        Returns:
            tuple[bool, Optional[str]]: (success, error_message) - True/None if successful, 
                                       False/error_message if failed (for graceful degradation)
        """
        self.logger.debug(f"Inserting file record in DynamoDB: {file_path} with status: {status}")
        
        try:
            current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            # Build the item with required fields
            item = {
                'file_path': {'S': file_path},
                'bucket_url': {'S': bucket_url},
                'status': {'S': status},
                'created_at': {'S': current_time},
                'updated_at': {'S': current_time}
            }
            
            # Add optional fields if provided
            if inference_id:
                item['inference_id'] = {'S': inference_id}
            if output_location:
                item['output_location'] = {'S': output_location}
            
            response = self.client.put_item(
                TableName=table_name,
                Item=item,
                # Use condition to prevent overwriting existing records
                ConditionExpression='attribute_not_exists(file_path)'
            )
            
            self.logger.debug(f"Successfully inserted file record: {file_path}")
            return True, None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            if error_code == 'ConditionalCheckFailedException':
                # Record already exists - this is expected in some cases
                self.logger.debug(f"File record already exists in DynamoDB: {file_path}")
                return True, None  # Consider this a success for idempotency
            else:
                error_msg = f"DynamoDB put_item error ({error_code}): {error_message} for file: {file_path}"
                self.logger.error(error_msg)
                return False, error_msg  # Graceful degradation
                
        except Exception as e:
            error_msg = f"Unexpected error inserting file record in DynamoDB: {str(e)} for file: {file_path}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg  # Graceful degradation
    
    def update_file_status(self, table_name: str, file_path: str, status: str, 
                          error_message: Optional[str] = None,
                          inference_id: Optional[str] = None,
                          output_location: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Update the status of a file record in DynamoDB.
        
        Args:
            table_name (str): Name of the DynamoDB table
            file_path (str): S3 file path to update
            status (str): New status ("submitted", "error", "bucket_error")
            error_message (Optional[str]): Error details if status is error
            inference_id (Optional[str]): SageMaker inference ID if available
            output_location (Optional[str]): SageMaker output location if available
            
        Returns:
            tuple[bool, Optional[str]]: (success, error_message) - True/None if successful, 
                                       False/error_message if failed (for graceful degradation)
        """
        self.logger.debug(f"Updating file status in DynamoDB: {file_path} to status: {status}")
        
        try:
            current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            # Build update expression and attribute values
            update_expression = "SET #status = :status, updated_at = :updated_at"
            expression_attribute_names = {'#status': 'status'}
            expression_attribute_values = {
                ':status': {'S': status},
                ':updated_at': {'S': current_time}
            }
            
            # Add error message if provided
            if error_message:
                update_expression += ", error_message = :error_message"
                expression_attribute_values[':error_message'] = {'S': error_message}
            
            # Add inference ID if provided
            if inference_id:
                update_expression += ", inference_id = :inference_id"
                expression_attribute_values[':inference_id'] = {'S': inference_id}
            
            # Add output location if provided
            if output_location:
                update_expression += ", output_location = :output_location"
                expression_attribute_values[':output_location'] = {'S': output_location}
            
            response = self.client.update_item(
                TableName=table_name,
                Key={
                    'file_path': {'S': file_path}
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                # Ensure the record exists before updating
                ConditionExpression='attribute_exists(file_path)'
            )
            
            self.logger.debug(f"Successfully updated file status: {file_path} to {status}")
            return True, None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message_db = e.response['Error']['Message']
            
            if error_code == 'ConditionalCheckFailedException':
                error_msg = f"File record does not exist in DynamoDB for update: {file_path}"
                self.logger.error(error_msg)
            else:
                error_msg = f"DynamoDB update_item error ({error_code}): {error_message_db} for file: {file_path}"
                self.logger.error(error_msg)
            
            return False, error_msg  # Graceful degradation
            
        except Exception as e:
            error_msg = f"Unexpected error updating file status in DynamoDB: {str(e)} for file: {file_path}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg  # Graceful degradation
    
    def find_record_by_inference_id(self, table_name: str, inference_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a record by inference_id using GSI query operation for optimal performance.
        
        Args:
            table_name (str): Name of the DynamoDB table
            inference_id (str): SageMaker inference ID to search for
            
        Returns:
            Optional[Dict[str, Any]]: Record data if found, None if not found
            
        Note:
            This function uses the InferenceIdIndex GSI for efficient querying.
        """
        self.logger.info(f"Querying for record by inference_id: {inference_id}")
        
        try:
            response = self.client.query(
                TableName=table_name,
                IndexName='InferenceIdIndex',
                KeyConditionExpression='inference_id = :inference_id',
                ExpressionAttributeValues={
                    ':inference_id': {'S': inference_id}
                }
            )
            
            if 'Items' in response and len(response['Items']) > 0:
                # Convert DynamoDB item format to Python dict
                item = {}
                for key, value in response['Items'][0].items():
                    if 'S' in value:
                        item[key] = value['S']
                    elif 'N' in value:
                        item[key] = value['N']
                    # Add other type conversions as needed
                
                self.logger.info(f"Found record by inference_id: {inference_id}")
                return item
            else:
                self.logger.info(f"No record found for inference_id: {inference_id}")
                return None
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            error_msg = f"DynamoDB query error ({error_code}): {error_message} for inference_id: {inference_id}"
            self.logger.error(error_msg)
            # Graceful degradation - return None to allow processing to continue
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error querying record by inference_id: {str(e)} for inference_id: {inference_id}"
            self.logger.error(error_msg, exc_info=True)
            # Graceful degradation - return None to allow processing to continue
            return None
    
    def update_job_completion_status(self, table_name: str, file_path: str, status: str,
                                   completion_timestamp: Optional[str] = None,
                                   sagemaker_job_id: Optional[str] = None,
                                   failure_reason: Optional[str] = None,
                                   original_content: Optional[str] = None,
                                   summarized_content: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Update job status to completed or failed with completion timestamp and content.
        
        Args:
            table_name (str): Name of the DynamoDB table
            file_path (str): S3 file path to update (primary key)
            status (str): Final status ("completed" or "failed")
            completion_timestamp (Optional[str]): When the job completed (ISO format)
            sagemaker_job_id (Optional[str]): SageMaker job identifier
            failure_reason (Optional[str]): Reason for failure if status is "failed"
            original_content (Optional[str]): Original inference output content
            summarized_content (Optional[str]): AI-generated summary of the content
            
        Returns:
            tuple[bool, Optional[str]]: (success, error_message) - True/None if successful, 
                                       False/error_message if failed (for graceful degradation)
        """
        self.logger.debug(f"Updating job completion status in DynamoDB: {file_path} to status: {status}")
        
        try:
            current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            completion_time = completion_timestamp or current_time
            
            # Build update expression and attribute values
            update_expression = "SET #status = :status, updated_at = :updated_at, completion_timestamp = :completion_timestamp"
            expression_attribute_names = {'#status': 'status'}
            expression_attribute_values = {
                ':status': {'S': status},
                ':updated_at': {'S': current_time},
                ':completion_timestamp': {'S': completion_time}
            }
            
            # Add SageMaker job ID if provided
            if sagemaker_job_id:
                update_expression += ", sagemaker_job_id = :sagemaker_job_id"
                expression_attribute_values[':sagemaker_job_id'] = {'S': sagemaker_job_id}
            
            # Add failure reason if provided and status is failed
            if failure_reason and status == "failed":
                update_expression += ", failure_reason = :failure_reason"
                expression_attribute_values[':failure_reason'] = {'S': failure_reason}
            
            # Add original content if provided
            if original_content is not None:
                # Validate content size (DynamoDB item size limit is 400KB)
                if len(original_content.encode('utf-8')) > 250000:  # Leave buffer for other attributes
                    self.logger.warning(f"Original content too large ({len(original_content)} chars), truncating")
                    original_content = original_content[:250000] + "... [TRUNCATED]"
                
                update_expression += ", original_content = :original_content"
                expression_attribute_values[':original_content'] = {'S': original_content}
            
            # Add summarized content if provided
            if summarized_content is not None:
                # Validate content size
                if len(summarized_content.encode('utf-8')) > 150000:  # Reasonable limit for summaries
                    self.logger.warning(f"Summarized content too large ({len(summarized_content)} chars), truncating")
                    summarized_content = summarized_content[:150000] + "... [TRUNCATED]"
                
                update_expression += ", summarized_content = :summarized_content"
                expression_attribute_values[':summarized_content'] = {'S': summarized_content}
            
            response = self.client.update_item(
                TableName=table_name,
                Key={
                    'file_path': {'S': file_path}
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                # Ensure the record exists before updating
                ConditionExpression='attribute_exists(file_path)'
            )
            
            self.logger.info(f"Successfully updated job completion status: {file_path} to {status}")
            return True, None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message_db = e.response['Error']['Message']
            
            if error_code == 'ConditionalCheckFailedException':
                error_msg = f"File record does not exist in DynamoDB for completion update: {file_path}"
                self.logger.error(error_msg)
            else:
                error_msg = f"DynamoDB update_item error ({error_code}): {error_message_db} for file: {file_path}"
                self.logger.error(error_msg)
            
            return False, error_msg  # Graceful degradation
            
        except Exception as e:
            error_msg = f"Unexpected error updating job completion status in DynamoDB: {str(e)} for file: {file_path}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg  # Graceful degradation
    
    def validate_content_size(self, content: str, content_type: str = "content") -> Tuple[bool, Optional[str]]:
        """
        Validate content size against DynamoDB limits and return validation result.
        
        Args:
            content (str): Content to validate
            content_type (str): Type of content for logging ("original" or "summarized")
            
        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message) - True/None if valid,
                                       False/error_message if invalid
        """
        if not content:
            return True, None
        
        try:
            content_bytes = len(content.encode('utf-8'))
            
            # DynamoDB item size limit is 400KB, but we need buffer for other attributes
            max_size = 350000 if content_type == "original" else 50000
            
            if content_bytes > max_size:
                error_msg = f"{content_type.capitalize()} content size ({content_bytes} bytes) exceeds limit ({max_size} bytes)"
                self.logger.error(error_msg)
                return False, error_msg
            
            self.logger.debug(f"{content_type.capitalize()} content size validation passed: {content_bytes} bytes")
            return True, None
            
        except Exception as e:
            error_msg = f"Error validating {content_type} content size: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def batch_get_files(self, table_name: str, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve multiple file records from DynamoDB in a single batch operation.
        
        Args:
            table_name (str): Name of the DynamoDB table
            file_paths (list[str]): List of S3 file paths to retrieve (max 100)
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping file_path to record data
            
        Note:
            This function implements graceful degradation - if DynamoDB operations fail,
            it logs the error and returns an empty dict to allow processing to continue.
        """
        if not file_paths:
            return {}
        
        if len(file_paths) > 100:
            self.logger.warning(f"BatchGetItem supports max 100 items, got {len(file_paths)}. Taking first 100.")
            file_paths = file_paths[:100]
        
        self.logger.debug(f"Batch getting {len(file_paths)} files from DynamoDB")
        
        try:
            # Build the request items
            keys = [{'file_path': {'S': file_path}} for file_path in file_paths]
            
            response = self.client.batch_get_item(
                RequestItems={
                    table_name: {
                        'Keys': keys
                    }
                }
            )
            
            results = {}
            
            # Process returned items
            if 'Responses' in response and table_name in response['Responses']:
                for item in response['Responses'][table_name]:
                    # Convert DynamoDB item format to Python dict
                    converted_item = {}
                    for key, value in item.items():
                        if 'S' in value:
                            converted_item[key] = value['S']
                        elif 'N' in value:
                            converted_item[key] = value['N']
                        # Add other type conversions as needed
                    
                    file_path = converted_item.get('file_path')
                    if file_path:
                        results[file_path] = converted_item
            
            # Handle unprocessed keys (due to throttling or other issues)
            if 'UnprocessedKeys' in response and response['UnprocessedKeys']:
                unprocessed_count = len(response['UnprocessedKeys'].get(table_name, {}).get('Keys', []))
                self.logger.warning(f"BatchGetItem had {unprocessed_count} unprocessed keys. Consider retry logic.")
            
            self.logger.debug(f"Successfully retrieved {len(results)} files from DynamoDB")
            return results
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            error_msg = f"DynamoDB batch_get_item error ({error_code}): {error_message}"
            self.logger.error(error_msg)
            return {}  # Graceful degradation
            
        except Exception as e:
            error_msg = f"Unexpected error in batch get files: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {}  # Graceful degradation

    def query_records_by_inference_id(self, table_name: str, inference_id: str) -> List[Dict[str, Any]]:
        """
        Query all records with a specific inference_id using GSI.
        
        Args:
            table_name (str): Name of the DynamoDB table
            inference_id (str): SageMaker inference ID to search for
            
        Returns:
            List[Dict[str, Any]]: List of records matching the inference_id
            
        Note:
            This function uses the InferenceIdIndex GSI for efficient querying.
            Returns all matching records, not just the first one.
        """
        self.logger.debug(f"Querying all records by inference_id: {inference_id}")
        
        try:
            response = self.client.query(
                TableName=table_name,
                IndexName='InferenceIdIndex',
                KeyConditionExpression='inference_id = :inference_id',
                ExpressionAttributeValues={
                    ':inference_id': {'S': inference_id}
                }
            )
            
            results = []
            if 'Items' in response:
                for item in response['Items']:
                    # Convert DynamoDB item format to Python dict
                    converted_item = {}
                    for key, value in item.items():
                        if 'S' in value:
                            converted_item[key] = value['S']
                        elif 'N' in value:
                            converted_item[key] = value['N']
                        # Add other type conversions as needed
                    
                    results.append(converted_item)
            
            self.logger.debug(f"Found {len(results)} records for inference_id: {inference_id}")
            return results
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            error_msg = f"DynamoDB query error ({error_code}): {error_message} for inference_id: {inference_id}"
            self.logger.error(error_msg)
            # Graceful degradation - return empty list to allow processing to continue
            return []
            
        except Exception as e:
            error_msg = f"Unexpected error querying records by inference_id: {str(e)} for inference_id: {inference_id}"
            self.logger.error(error_msg, exc_info=True)
            # Graceful degradation - return empty list to allow processing to continue
            return []

    def insert_bucket_error_record(self, table_name: str, bucket_uri: str, error_message: str) -> Tuple[bool, Optional[str]]:
        """
        Insert a bucket-level error record in DynamoDB for tracking bucket access failures.
        
        Args:
            table_name (str): Name of the DynamoDB table
            bucket_uri (str): S3 bucket URI that failed
            error_message (str): Error details
            
        Returns:
            tuple[bool, Optional[str]]: (success, error_message) - True/None if successful, 
                                       False/error_message if failed (for graceful degradation)
        """
        self.logger.debug(f"Inserting bucket error record in DynamoDB: {bucket_uri}")
        
        try:
            current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            response = self.client.put_item(
                TableName=table_name,
                Item={
                    'file_path': {'S': bucket_uri},  # Use bucket URI as the key
                    'status': {'S': 'bucket_error'},
                    'created_at': {'S': current_time},
                    'updated_at': {'S': current_time},
                    'error_message': {'S': error_message}
                }
            )
            
            self.logger.info(f"Successfully inserted bucket error record: {bucket_uri}")
            return True, None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message_db = e.response['Error']['Message']
            error_msg = f"DynamoDB put_item error ({error_code}): {error_message_db} for bucket: {bucket_uri}"
            self.logger.error(error_msg)
            return False, error_msg  # Graceful degradation
            
        except Exception as e:
            error_msg = f"Unexpected error inserting bucket error record in DynamoDB: {str(e)} for bucket: {bucket_uri}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg  # Graceful degradation