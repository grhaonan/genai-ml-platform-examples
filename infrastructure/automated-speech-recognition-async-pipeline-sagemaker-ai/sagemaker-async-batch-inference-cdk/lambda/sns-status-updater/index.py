"""
SNS Status Updater Lambda Function

This Lambda function processes SNS messages from SageMaker async inference completion
events and updates the corresponding DynamoDB records with completion status.
It provides real-time status tracking for ML inference jobs.
"""

import json
import logging
import sys
import os
import time
import random
from typing import Dict, Any, Optional

# Add shared directory to path for imports
# In Lambda runtime, shared directory is copied to the same level as function code
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, 'shared')

# For development, also try the relative path
if not os.path.exists(shared_path):
    shared_path = os.path.join(current_dir, '..', 'shared')

if os.path.exists(shared_path) and shared_path not in sys.path:
    sys.path.insert(0, shared_path)

# Import modules
from sns_config import ConfigManager
from sns_message_processor import SNSMessageProcessor
from logger_setup import LoggerSetup
from dynamodb_operations import DynamoDBOperations
from s3_operations import S3Operations
from bedrock_operations import BedrockOperations


class SNSStatusUpdater:
    """Main processor class that handles SNS messages and updates DynamoDB status."""
    
    def __init__(self):
        # Initialize logger
        self.logger = LoggerSetup.setup_logging('sns-status-updater')
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.sns_processor = SNSMessageProcessor()
        
        # Initialize DynamoDB operations with config
        self.dynamodb_operations = DynamoDBOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )
        
        # Initialize S3 operations for content retrieval
        self.s3_operations = S3Operations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )
        
        # Initialize Bedrock operations for content summarization
        self.bedrock_operations = BedrockOperations()
    
    def process_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Process the Lambda request with comprehensive error handling.
        
        Args:
            event (Dict[str, Any]): Lambda event containing SNS message
            context (Any): Lambda context object
            
        Returns:
            Dict[str, Any]: Response with processing results
        """
        self.logger.info(f"SNS Status Updater Lambda function started. Request ID: {context.aws_request_id}")
        self.logger.debug(f"Event received: {json.dumps(event, default=str)}")
        
        try:
            # Validate environment variables
            env_result = self.config_manager.validate_environment_variables()
            if not env_result["is_success"]:
                error_msg = env_result["error_message"]
                self.logger.error(error_msg)
                return self._create_error_response(
                    400, 'ValidationError', error_msg, context.aws_request_id
                )
            
            env_vars = env_result["data"]
            self.logger.info("Environment variables validation completed")
            
            # Process SNS records from the event
            if 'Records' not in event:
                error_msg = "No SNS records found in event"
                self.logger.error(error_msg)
                return self._create_error_response(
                    400, 'ValidationError', error_msg, context.aws_request_id
                )
            
            records = event['Records']
            self.logger.info(f"Processing {len(records)} SNS records")
            
            # Process each SNS record with detailed tracking
            processing_results = self._process_sns_records(records, env_vars, context)
            
            # Prepare comprehensive response
            response = {
                'statusCode': 200,
                'body': {
                    'message': 'SNS message processing completed',
                    'total_records': len(records),
                    'processed_records': processing_results['processed_count'],
                    'failed_records': processing_results['failed_count'],
                    'skipped_records': processing_results['skipped_count'],
                    'database_updates': processing_results['database_updates'],
                    'request_id': context.aws_request_id
                }
            }
            
            self.logger.info(f"SNS Status Updater Lambda function completed successfully. Request ID: {context.aws_request_id}")
            return response
            
        except Exception as e:
            error_msg = f"Error processing SNS message: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Determine error code based on exception type
            if isinstance(e, ValueError):
                error_code = 'ValidationError'
                status_code = 400
            else:
                error_code = 'InternalError'
                status_code = 500
                
            return self._create_error_response(
                status_code, error_code, error_msg, context.aws_request_id
            )
    
    def _process_sns_records(self, records: list, env_vars: Dict[str, str], 
                           context: Any) -> Dict[str, int]:
        """
        Process all SNS records with comprehensive error handling and tracking.
        
        Args:
            records (list): List of SNS records from the event
            env_vars (Dict[str, str]): Validated environment variables
            context (Any): Lambda context object
            
        Returns:
            Dict[str, int]: Processing statistics
        """
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        database_updates = 0
        
        for record_index, record in enumerate(records):
            try:
                self.logger.debug(f"Processing SNS record {record_index + 1}/{len(records)}")
                
                # Process individual record with retry logic
                result = self._process_sns_record_with_retry(record, env_vars, context)
                
                if result['success']:
                    processed_count += 1
                    if result['database_updated']:
                        database_updates += 1
                    self.logger.debug(f"Successfully processed SNS record {record_index + 1}")
                elif result['skipped']:
                    skipped_count += 1
                    self.logger.info(f"Skipped SNS record {record_index + 1}: {result['reason']}")
                else:
                    failed_count += 1
                    self.logger.error(f"Failed to process SNS record {record_index + 1}: {result['error']}")
                    
            except Exception as record_error:
                failed_count += 1
                self.logger.error(f"Failed to process SNS record {record_index + 1}: {str(record_error)}", exc_info=True)
        
        return {
            'processed_count': processed_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'database_updates': database_updates
        }
    
    def _process_sns_record_with_retry(self, record: Dict[str, Any], env_vars: Dict[str, str], 
                                     context: Any) -> Dict[str, Any]:
        """
        Process a single SNS record with retry logic for DynamoDB operations.
        
        Args:
            record (Dict[str, Any]): SNS record from the event
            env_vars (Dict[str, str]): Validated environment variables
            context (Any): Lambda context object
            
        Returns:
            Dict[str, Any]: Processing result with success/failure details
        """
        try:
            # Validate and parse SNS message using the processor
            validation_result = self.sns_processor.validate_and_parse_sns_record(record)
            if not validation_result["is_success"]:
                return {
                    'success': False,
                    'database_updated': False,
                    'skipped': False,
                    'reason': None,
                    'error': validation_result["error_message"]
                }
            
            validated_data = validation_result["data"]
            
            # Extract details for database operations
            inference_details = self.sns_processor.extract_inference_details(validated_data)
            
            self.logger.info(f"Processing SNS message for inference ID: {inference_details['inference_id']}, invocationStatus: {inference_details['invocationStatus']}")
            
            # Process content if job completed successfully
            original_content = None
            summarized_content = None
            
            if inference_details['invocationStatus'] == 'Completed' and inference_details.get('output_location'):
                content_result = self._process_inference_content(inference_details, context)
                if content_result['success']:
                    original_content = content_result['original_content']
                    summarized_content = content_result['summarized_content']
                    self.logger.info(f"Successfully processed content for inference_id: {inference_details['inference_id']}")
                else:
                    # Log content processing failure but continue with status update
                    self.logger.warning(f"Content processing failed for inference_id: {inference_details['inference_id']}: {content_result['error']}")
            
            # Update DynamoDB with retry logic, including content if available
            database_result = self._update_database_with_retry(
                inference_details, env_vars['DYNAMODB_TABLE_NAME'], context,
                original_content=original_content, summarized_content=summarized_content
            )
            
            if database_result['success']:
                return {
                    'success': True,
                    'database_updated': database_result['updated'],
                    'skipped': False,
                    'reason': None,
                    'error': None
                }
            elif database_result['record_not_found']:
                # This is a warning, not an error - record might not exist yet due to eventual consistency
                self.logger.warning(f"Record not found for inference_id: {inference_details['inference_id']}. This may be due to eventual consistency.")
                return {
                    'success': False,
                    'database_updated': False,
                    'skipped': True,
                    'reason': 'Record not found - possible eventual consistency issue',
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'database_updated': False,
                    'skipped': False,
                    'reason': None,
                    'error': database_result['error']
                }
                
        except Exception as processing_error:
            error_msg = f"Processing error: {str(processing_error)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'database_updated': False,
                'skipped': False,
                'reason': None,
                'error': error_msg
            }
    
    def _process_inference_content(self, inference_details: Dict[str, Any], 
                                 context: Any) -> Dict[str, Any]:
        """
        Process inference output content by retrieving from S3 and generating summary.
        
        Args:
            inference_details (Dict[str, Any]): Extracted inference details from SNS message
            context (Any): Lambda context object for timeout checking
            
        Returns:
            Dict[str, Any]: Processing result with original and summarized content
        """
        try:
            output_location = inference_details.get('output_location')
            if not output_location:
                return {
                    'success': False,
                    'original_content': None,
                    'summarized_content': None,
                    'error': 'No output location provided'
                }
            
            self.logger.info(f"Processing content from S3 location: {output_location}")
            
            # Check remaining time to avoid timeout
            remaining_time = context.get_remaining_time_in_millis() / 1000.0
            if remaining_time < 30.0:  # Need at least 30 seconds for content processing
                self.logger.warning(f"Insufficient time remaining ({remaining_time:.1f}s) for content processing")
                return {
                    'success': False,
                    'original_content': None,
                    'summarized_content': None,
                    'error': 'Insufficient time remaining for content processing'
                }
            
            # Extract S3 output location from inference details
            extract_result = self.s3_operations.extract_s3_output_location(inference_details)
            if not extract_result['is_success']:
                return {
                    'success': False,
                    'original_content': None,
                    'summarized_content': None,
                    'error': f"Failed to extract S3 location: {extract_result['error_message']}"
                }
            
            s3_location = extract_result['output_location']
            
            # Read inference output file from S3
            read_result = self.s3_operations.read_inference_output_file(s3_location)
            if not read_result['is_success']:
                return {
                    'success': False,
                    'original_content': None,
                    'summarized_content': None,
                    'error': f"Failed to read S3 file: {read_result['error_message']}"
                }
            
            # Use decoded content if available, otherwise use raw content
            original_content = read_result['decoded_content'] or read_result['content']
            
            if not original_content:
                return {
                    'success': False,
                    'original_content': None,
                    'summarized_content': None,
                    'error': 'No content found in S3 file'
                }
            
            self.logger.info(f"Successfully retrieved {len(original_content)} characters of content from S3")
            
            # Check remaining time before Bedrock processing
            remaining_time = context.get_remaining_time_in_millis() / 1000.0
            if remaining_time < 20.0:  # Need at least 20 seconds for Bedrock processing
                self.logger.warning(f"Insufficient time remaining ({remaining_time:.1f}s) for Bedrock processing")
                # Return with original content only
                return {
                    'success': True,
                    'original_content': original_content,
                    'summarized_content': None,
                    'error': None
                }
            
            # Generate summary using Bedrock API
            summary_success, summarized_content, summary_error = self.bedrock_operations.process_inference_output(original_content)
            
            if summary_success:
                self.logger.info(f"Successfully generated summary ({len(summarized_content)} characters)")
                return {
                    'success': True,
                    'original_content': original_content,
                    'summarized_content': summarized_content,
                    'error': None
                }
            else:
                # Log Bedrock failure but continue with original content
                self.logger.warning(f"Bedrock summarization failed: {summary_error}")
                return {
                    'success': True,
                    'original_content': original_content,
                    'summarized_content': None,
                    'error': None
                }
                
        except Exception as e:
            error_msg = f"Unexpected error processing inference content: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'original_content': None,
                'summarized_content': None,
                'error': error_msg
            }
    
    def _update_database_with_retry(self, inference_details: Dict[str, Any], 
                                  table_name: str, context: Any,
                                  original_content: Optional[str] = None,
                                  summarized_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Update DynamoDB with exponential backoff retry logic.
        
        Args:
            inference_details (Dict[str, Any]): Extracted inference details from SNS message
            table_name (str): DynamoDB table name
            context (Any): Lambda context object for timeout checking
            original_content (Optional[str]): Original inference output content
            summarized_content (Optional[str]): AI-generated summary of the content
            
        Returns:
            Dict[str, Any]: Update result with success/failure details
        """
        max_retries = 3
        base_delay = 1.0  # Base delay in seconds
        max_delay = 10.0  # Maximum delay in seconds
        
        for attempt in range(max_retries + 1):
            # Check remaining time to avoid timeout
            remaining_time = context.get_remaining_time_in_millis() / 1000.0
            if remaining_time < 5.0:  # Need at least 5 seconds for operation
                self.logger.warning(f"Insufficient time remaining ({remaining_time:.1f}s) for database operation")
                return {
                    'success': False,
                    'updated': False,
                    'record_not_found': False,
                    'error': 'Insufficient time remaining for database operation'
                }
            
            # First, try to find the record by inference_id
            record = self.dynamodb_operations.find_record_by_inference_id(
                table_name, inference_details['inference_id']
            )
            
            if record is None:
                self.logger.warning(f"No record found for inference_id: {inference_details['inference_id']}")
                return {
                    'success': False,
                    'updated': False,
                    'record_not_found': True,
                    'error': None
                }
            
            # Update the record with completion status
            file_path = record['file_path']
            status = inference_details['invocationStatus'].lower()
            
            success, error_message = self.dynamodb_operations.update_job_completion_status(
                table_name=table_name,
                file_path=file_path,
                status=status,
                completion_timestamp=inference_details.get('completion_time'),
                sagemaker_job_id=inference_details['inference_id'],
                failure_reason=inference_details.get('failure_reason') if status == "failed" else None,
                original_content=original_content,
                summarized_content=summarized_content
            )
            
            if success:
                self.logger.info(f"Successfully updated database record for file: {file_path}, status: {status}")
                return {
                    'success': True,
                    'updated': True,
                    'record_not_found': False,
                    'error': None
                }
            else:
                # Check if this is a retryable error
                if self._is_retryable_error(error_message) and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    self.logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {error_message}")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"Database operation failed after {attempt + 1} attempts: {error_message}")
                    return {
                        'success': False,
                        'updated': False,
                        'record_not_found': False,
                        'error': error_message
                    }
                        
        
        # This should not be reached, but included for completeness
        return {
            'success': False,
            'updated': False,
            'record_not_found': False,
            'error': 'Maximum retry attempts exceeded'
        }
    
    def _is_retryable_error(self, error_message: str) -> bool:
        """
        Determine if an error is retryable based on error message patterns.
        
        Args:
            error_message (str): Error message to analyze
            
        Returns:
            bool: True if error is retryable, False otherwise
        """
        if not error_message:
            return False
        
        error_lower = error_message.lower()
        
        # Retryable error patterns
        retryable_patterns = [
            'throttling',
            'provisionedthroughputexceeded',
            'serviceunavailable',
            'internalservererror',
            'requesttimeout',
            'timeout',
            'connection',
            'network',
            'temporary'
        ]
        
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _create_error_response(self, status_code: int, error_type: str, 
                             message: str, request_id: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            status_code (int): HTTP status code
            error_type (str): Type of error
            message (str): Error message
            request_id (str): Lambda request ID
            
        Returns:
            Dict[str, Any]: Standardized error response
        """
        return {
            'statusCode': status_code,
            'body': {
                'error': error_type,
                'message': message,
                'request_id': request_id,
                'timestamp': time.time()
            }
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler function for SNS status updater.
    
    Args:
        event (Dict[str, Any]): Lambda event containing SNS message
        context (Any): Lambda context object
        
    Returns:
        Dict[str, Any]: Response with processing results
    """
    # Initialize processor instance inside the handler to avoid import-time issues
    processor = SNSStatusUpdater()
    processor.logger.info(f"Received event: {event}")
    return processor.process_request(event, context)