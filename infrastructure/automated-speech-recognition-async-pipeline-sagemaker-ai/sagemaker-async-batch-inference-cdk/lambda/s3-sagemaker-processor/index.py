"""
S3 SageMaker Processor Lambda Function

This Lambda function processes files from an S3 bucket and submits them to a SageMaker
asynchronous inference endpoint. It tracks processing status in DynamoDB to ensure
idempotency and prevent duplicate processing.
"""

import json
import logging
import sys
import os
from typing import Dict, Any

# Add shared directory to path for imports
# In Lambda runtime, shared directory is copied to the same level as function code
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, 'shared')

# For development, also try the relative path
if not os.path.exists(shared_path):
    shared_path = os.path.join(current_dir, '..', 'shared')

if os.path.exists(shared_path) and shared_path not in sys.path:
    sys.path.insert(0, shared_path)

# Import from current directory
from config import ConfigManager
from event_validator import EventValidator
from sagemaker_operations import SageMakerOperations

# Import from shared directory
from logger_setup import LoggerSetup
from s3_operations import S3Operations
from dynamodb_operations import DynamoDBOperations


class S3SageMakerProcessor:
    """Main processor class that orchestrates S3 and DynamoDB operations."""
    
    def __init__(self):
        # Initialize logger
        self.logger = LoggerSetup.setup_logging('s3-sagemaker-processor')
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.event_validator = EventValidator()
        
        # Initialize AWS operations with config
        self.s3_operations = S3Operations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )
        self.dynamodb_operations = DynamoDBOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )
        self.sagemaker_operations = SageMakerOperations(
            region_name=ConfigManager.AWS_REGION,
            account_id=ConfigManager.AWS_ACCOUNT_ID
        )
    
    def process_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Process the Lambda request with proper error handling.
        
        Args:
            event (Dict[str, Any]): Lambda event containing bucket_uri
            context (Any): Lambda context object
            
        Returns:
            Dict[str, Any]: Response with processing results
        """
        self.logger.info(f"Lambda function started. Request ID: {context.aws_request_id}")
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
            
            # Validate and extract bucket URI from event
            event_result = self.event_validator.validate_lambda_event(event)
            if not event_result["is_success"]:
                error_msg = event_result["error_message"]
                self.logger.error(error_msg)
                return self._create_error_response(
                    400, 'ValidationError', error_msg, context.aws_request_id
                )
            
            bucket_uri = event_result["data"]
            self.logger.info(f"Processing bucket URI: {bucket_uri}")
            
            # Parse bucket URI to extract bucket name and prefix
            parse_result = self.event_validator.parse_s3_bucket_uri(bucket_uri)
            if not parse_result["is_success"]:
                error_msg = parse_result["error_message"]
                self.logger.error(error_msg)
                return self._create_error_response(
                    400, 'ValidationError', error_msg, context.aws_request_id
                )
            
            bucket_name, prefix = parse_result["data"]
            self.logger.info(f"Extracted bucket: {bucket_name}, prefix: {prefix}")
            
            # Initialize S3 client and list files
            list_result = self.s3_operations.list_objects(bucket_name, prefix)
            if not list_result['is_success']:
                # Handle S3-specific errors
                error_msg = f"S3 operation error: {list_result['error_message']}"
                self.logger.error(error_msg)
                
                return {
                    'statusCode': 500,
                    'body': {
                        'error': 'S3Error',
                        'message': error_msg,
                        'request_id': context.aws_request_id
                    }
                }
            
            files = list_result['objects']
            self.logger.info(f"Found {len(files)} files in bucket {bucket_name}")
            
            # Validate SageMaker endpoint configuration
            endpoint_name = env_vars['SAGEMAKER_ENDPOINT_NAME']
            endpoint_result = self.sagemaker_operations.validate_endpoint_configuration(endpoint_name)
            if not endpoint_result["is_success"]:
                error_msg = f"SageMaker endpoint error: {endpoint_result['error_message']}"
                self.logger.error(error_msg)
                return self._create_error_response(
                    500, 'SageMakerError', error_msg, context.aws_request_id
                )
            
            self.logger.info(f"SageMaker endpoint validated: {endpoint_name}")
            
            # Main processing orchestration logic
            table_name = env_vars['DYNAMODB_TABLE_NAME']
            endpoint_name = env_vars['SAGEMAKER_ENDPOINT_NAME']
            
            # Process files with status checking and batch submission
            processing_results = self._process_files_with_status_tracking(
                files, bucket_uri, table_name, endpoint_name
            )
            
            # Prepare response with processing summary
            response = {
                'statusCode': 200,
                'body': {
                    'message': 'File processing completed',
                    'bucket_name': bucket_name,
                    'prefix': prefix,
                    'files_found': len(files),
                    'processed_files': processing_results['processed_count'],
                    'skipped_files': processing_results['skipped_count'],
                    'failed_files': processing_results['failed_count'],
                    'successful_submissions': processing_results['successful_count'],
                    'request_id': context.aws_request_id
                }
            }
            
            self.logger.info(f"Lambda function completed successfully. Request ID: {context.aws_request_id}")
            return response
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'statusCode': 500,
                'body': {
                    'error': 'InternalError',
                    'message': error_msg,
                    'request_id': context.aws_request_id
                }
            }
    
    def _process_files_with_status_tracking(self, files: list, bucket_uri: str, 
                                           table_name: str, endpoint_name: str) -> Dict[str, int]:
        """
        Process files with DynamoDB status tracking and batch submission to SageMaker.
        
        Args:
            files (list): List of S3 file paths to process
            bucket_uri (str): Original bucket URI for reference
            table_name (str): DynamoDB table name for status tracking
            endpoint_name (str): SageMaker endpoint name
            
        Returns:
            Dict[str, int]: Processing statistics
        """
        self.logger.info(f"Starting file processing with status tracking for {len(files)} files")
        
        # Prepare S3 URIs and check existing records
        full_s3_uris = self._construct_s3_uris(files, bucket_uri)
        existing_records = self._get_existing_records(table_name, full_s3_uris)
        
        # Filter new files and insert initial records
        new_files = self._filter_new_files(full_s3_uris, existing_records)
        processed_count = self._insert_initial_records(new_files, table_name, bucket_uri)
        
        # Early return if no new files to process
        if not new_files:
            return self._create_no_files_response(len(files), len(existing_records))
        
        # Process files in batches
        batch_results = self._process_files_in_batches(
            new_files, table_name, endpoint_name
        )
        
        # Combine results
        return {
            'total_files_found': len(files),
            'existing_files': len(existing_records),
            'processed_count': processed_count,
            'skipped_count': batch_results['skipped_count'],
            'failed_count': batch_results['failed_count'],
            'successful_count': batch_results['successful_count']
        }
    
    def _construct_s3_uris(self, files: list, bucket_uri: str) -> list:
        """Construct full S3 URIs from file paths."""
        parse_result = self.event_validator.parse_s3_bucket_uri(bucket_uri)
        if not parse_result["is_success"]:
            raise ValueError(f"Failed to parse bucket URI: {parse_result['error_message']}")
        
        bucket_name, _ = parse_result["data"]
        
        full_s3_uris = []
        for file_path in files:
            full_s3_uri = f"s3://{bucket_name}/{file_path}"
            self.logger.debug(f"Constructed S3 URI: {full_s3_uri}")
            full_s3_uris.append(full_s3_uri)
        
        self.logger.info(f"Constructed {len(full_s3_uris)} full S3 URIs for processing")
        return full_s3_uris
    
    def _get_existing_records(self, table_name: str, full_s3_uris: list) -> Dict:
        """Get existing records from DynamoDB."""
        existing_records = self.dynamodb_operations.batch_get_files(
            table_name=table_name,
            file_paths=full_s3_uris
        )
        self.logger.info(f"Found {len(existing_records)} existing records in DynamoDB")
        return existing_records
    
    def _filter_new_files(self, full_s3_uris: list, existing_records: Dict) -> list:
        """Filter out files that already exist in DynamoDB."""
        new_files = []
        
        for uri in full_s3_uris:
            if uri not in existing_records:
                new_files.append(uri)
            else:
                record = existing_records[uri]
                status = record.get('status', 'unknown')
                if status in ['error', 'bucket_error', 'failed']:
                    self.logger.info(f"Will reprocess file with error status: {uri} (status: {status})")
                    new_files.append(uri)
                else:
                    self.logger.debug(f"Skipping existing file: {uri} (status: {status})")
        
        self.logger.info(f"Identified {len(new_files)} new files to process")
        return new_files
    
    def _insert_initial_records(self, new_files: list, table_name: str, bucket_uri: str) -> int:
        """Insert initial processing records for new files."""
        processed_count = 0
        
        for full_s3_uri in new_files:
            success, error_message = self.dynamodb_operations.insert_file_record(
                table_name=table_name,
                file_path=full_s3_uri,
                bucket_url=bucket_uri,
                status="processing"
            )
            if success:
                processed_count += 1
            else:
                self.logger.warning(f"Failed to insert record for {full_s3_uri}: {error_message}")
        
        return processed_count
    
    def _create_no_files_response(self, total_files: int, existing_files: int) -> Dict[str, Any]:
        """Create response when no new files need processing."""
        self.logger.info("No new files to process - all files already exist in DynamoDB")
        return {
            'total_files_found': total_files,
            'existing_files': existing_files,
            'processed_count': 0,
            'skipped_count': 0,
            'failed_count': 0,
            'successful_count': 0
        }
    
    def _process_files_in_batches(self, files: list, table_name: str, endpoint_name: str) -> Dict[str, int]:
        """Process files in batches and return processing statistics."""
        batch_size = 10
        total_batches = (len(files) + batch_size - 1) // batch_size
        
        # Initialize counters
        skipped_count = 0
        failed_count = 0
        successful_count = 0
        
        self.logger.info(f"Processing {len(files)} files in {total_batches} batches of {batch_size}")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(files))
            batch_files = files[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch_files)} files")
            
            batch_results = self._process_single_batch(
                batch_files, table_name, endpoint_name, batch_num + 1
            )
            
            # Accumulate results
            skipped_count += batch_results['skipped_count']
            failed_count += batch_results['failed_count']
            successful_count += batch_results['successful_count']
        
        self.logger.info(f"Batch processing completed. Skipped: {skipped_count}, "
                        f"Failed: {failed_count}, Successful: {successful_count}")
        
        return {
            'skipped_count': skipped_count,
            'failed_count': failed_count,
            'successful_count': successful_count
        }
    
    def _process_single_batch(self, batch_files: list, table_name: str, 
                             endpoint_name: str, batch_num: int) -> Dict[str, int]:
        """Process a single batch of files."""
        skipped_count = 0
        failed_count = 0
        successful_count = 0
        
        try:
            # Submit batch to SageMaker
            batch_result = self.sagemaker_operations.batch_submit_files_for_inference(
                endpoint_name=endpoint_name,
                s3_urls=batch_files,
                batch_count=len(batch_files)
            )
            
            if not batch_result["is_success"]:
                self.logger.error(f"Batch submission failed: {batch_result['error_message']}")
                # Mark all files as failed
                for file_path in batch_files:
                    failed_count += self._handle_failed_submission(
                        {'error': batch_result['error_message'], 's3_url': file_path}, 
                        table_name, file_path
                    )
                return skipped_count, failed_count, successful_count
            
            batch_results = batch_result["data"]
            
            # Process individual results
            for result in batch_results:
                file_path = result['s3_url']
                
                if result.get('skipped', False):
                    skipped_count += 1
                    self.logger.debug(f"Skipped existing file: {file_path}")
                    continue
                
                if result['success']:
                    successful_count += self._handle_successful_submission(
                        result, table_name, file_path
                    )
                else:
                    failed_count += self._handle_failed_submission(
                        result, table_name, file_path
                    )
            
            self.logger.info(f"Completed batch {batch_num}")
            
        except Exception as batch_error:
            failed_count += self._handle_batch_error(
                batch_files, table_name, batch_error, batch_num
            )
        
        return {
            'skipped_count': skipped_count,
            'failed_count': failed_count,
            'successful_count': successful_count
        }
    
    def _handle_successful_submission(self, result: Dict, table_name: str, file_path: str) -> int:
        """Handle successful SageMaker submission."""
        inference_result = result['result']
        success, error_message = self.dynamodb_operations.update_file_status(
            table_name=table_name,
            file_path=file_path,
            status="submitted",
            inference_id=inference_result.get('inference_id'),
            output_location=inference_result.get('output_location')
        )
        
        if success:
            self.logger.debug(f"Successfully processed file: {file_path}")
            return 1
        else:
            self.logger.error(f"Failed to update DynamoDB for successful submission: {file_path}: {error_message}")
            return 0
    
    def _handle_failed_submission(self, result: Dict, table_name: str, file_path: str) -> int:
        """Handle failed SageMaker submission."""
        error_message = result.get('error', 'Unknown error during SageMaker submission')
        success, db_error_message = self.dynamodb_operations.update_file_status(
            table_name=table_name,
            file_path=file_path,
            status="error",
            error_message=error_message
        )
        
        self.logger.error(f"Failed to process file {file_path}: {error_message}")
        
        if not success:
            self.logger.error(f"Failed to update DynamoDB for failed submission: {file_path}: {db_error_message}")
        
        return 1
    
    def _handle_batch_error(self, batch_files: list, table_name: str, 
                           batch_error: Exception, batch_num: int) -> int:
        """Handle batch-level processing errors."""
        error_message = f"Batch processing error: {str(batch_error)}"
        self.logger.error(f"Batch {batch_num} failed: {error_message}")
        
        failed_count = 0
        for file_path in batch_files:
            success, db_error_message = self.dynamodb_operations.update_file_status(
                table_name=table_name,
                file_path=file_path,
                status="error",
                error_message=error_message
            )
            failed_count += 1
            
            if not success:
                self.logger.error(f"Failed to update DynamoDB for batch error: {file_path}: {db_error_message}")
        
        return failed_count
    
    def _create_error_response(self, status_code: int, error_type: str, 
                              error_message: str, request_id: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            status_code (int): HTTP status code
            error_type (str): Type of error
            error_message (str): Error message
            request_id (str): AWS request ID
            
        Returns:
            Dict[str, Any]: Standardized error response
        """
        return {
            'statusCode': status_code,
            'body': {
                'error': error_type,
                'message': error_message,
                'request_id': request_id
            }
        }


# Initialize processor instance
processor = S3SageMakerProcessor()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler function for S3 SageMaker processor.
    
    Args:
        event (Dict[str, Any]): Lambda event containing bucket_uri
        context (Any): Lambda context object
        
    Returns:
        Dict[str, Any]: Response with processing results
    """
    return processor.process_request(event, context)