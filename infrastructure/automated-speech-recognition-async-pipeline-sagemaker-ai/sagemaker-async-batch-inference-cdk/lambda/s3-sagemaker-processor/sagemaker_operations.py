"""
SageMaker Operations Module

This module handles SageMaker asynchronous inference operations including
endpoint invocation and error handling.
"""

import os
import boto3
import logging
import uuid
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError, BotoCoreError


class SageMakerOperations:
    """Handles SageMaker asynchronous inference operations."""
    
    def __init__(self, region_name: str = None, account_id: str = None):
        """
        Initialize SageMaker operations client.
        
        Args:
            region_name (str): AWS region name (defaults to environment variable)
            account_id (str): AWS account ID (defaults to environment variable)
        """
        self.region_name = region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        self.account_id = account_id or os.environ.get('AWS_ACCOUNT_ID', '')
        self.logger = logging.getLogger(__name__)
        self.sagemaker_runtime = None
        self.initialization_error = None
        
        try:
            self.sagemaker_runtime = boto3.client(
                'sagemaker-runtime',
                region_name=region_name
            )
            self.logger.info(f"SageMaker runtime client initialized for region: {region_name}")
        except Exception as e:
            error_msg = f"SageMaker client initialization failed: {str(e)}"
            self.logger.error(f"Failed to initialize SageMaker runtime client: {str(e)}")
            self.initialization_error = error_msg
    
    def generate_inference_id(self) -> str:
        """
        Generate a random string for inference_id.
        
        Returns:
            str: Random UUID string for inference identification
        """
        inference_id = str(uuid.uuid4())
        self.logger.debug(f"Generated inference ID: {inference_id}")
        return inference_id
    
    def submit_file_for_inference(
        self,
        endpoint_name: str,
        input_location: str,
        content_type: str = "application/octet-stream",
        accept: str = "application/json"
    ) -> Dict[str, Any]:
        """
        Submit a file to SageMaker async inference endpoint.
        
        Args:
            endpoint_name (str): SageMaker endpoint name/ARN
            input_location (str): S3 URI of the input file
            content_type (str): Content type of the input file
            accept (str): Accept header for response format
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        # Check if client was initialized successfully
        if self.initialization_error:
            return {
                "is_success": False,
                "data": None,
                "error_code": "INITIALIZATION_ERROR",
                "error_message": self.initialization_error
            }
        
        inference_id = self.generate_inference_id()
        
        try:
            self.logger.info(f"Submitting file for inference: {input_location}")
            self.logger.debug(f"Endpoint: {endpoint_name}, Inference ID: {inference_id}")
            
            response = self.sagemaker_runtime.invoke_endpoint_async(
                EndpointName=endpoint_name,
                InferenceId=inference_id,
                InputLocation=input_location,
                ContentType=content_type,
                Accept=accept
            )
            
            # Extract relevant information from response
            result = {
                'inference_id': inference_id,
                'output_location': response.get('OutputLocation'),
                'response_metadata': response.get('ResponseMetadata', {}),
                'input_location': input_location
            }
            
            self.logger.info(f"Successfully submitted file for inference. Inference ID: {inference_id}")
            self.logger.debug(f"Output location: {result['output_location']}")
            
            return {
                "is_success": True,
                "data": result,
                "error_code": None,
                "error_message": None
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            self.logger.error(f"SageMaker ClientError - Code: {error_code}, Message: {error_message}")
            self.logger.error(f"Failed to submit file: {input_location}")
            
            # Handle specific error cases
            if error_code == 'ValidationException':
                return {
                    "is_success": False,
                    "data": None,
                    "error_code": "VALIDATION_ERROR",
                    "error_message": f"Invalid endpoint configuration or input: {error_message}"
                }
            elif error_code == 'ModelError':
                return {
                    "is_success": False,
                    "data": None,
                    "error_code": "MODEL_ERROR",
                    "error_message": f"Model error during inference: {error_message}"
                }
            elif error_code == 'InternalFailure':
                return {
                    "is_success": False,
                    "data": None,
                    "error_code": "SAGEMAKER_INTERNAL_ERROR",
                    "error_message": f"SageMaker internal failure: {error_message}"
                }
            else:
                return {
                    "is_success": False,
                    "data": None,
                    "error_code": "SAGEMAKER_ERROR",
                    "error_message": f"SageMaker endpoint error ({error_code}): {error_message}"
                }
                
        except BotoCoreError as e:
            self.logger.error(f"BotoCoreError during SageMaker invocation: {str(e)}")
            return {
                "is_success": False,
                "data": None,
                "error_code": "NETWORK_ERROR",
                "error_message": f"Network or configuration error: {str(e)}"
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error during SageMaker invocation: {str(e)}")
            return {
                "is_success": False,
                "data": None,
                "error_code": "UNEXPECTED_ERROR",
                "error_message": f"Unexpected SageMaker error: {str(e)}"
            }
    
    def validate_endpoint_configuration(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Validate that the SageMaker endpoint exists and is accessible.
        
        Args:
            endpoint_name (str): SageMaker endpoint name/ARN
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        try:
            # Create a SageMaker client to describe the endpoint
            sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
            
            self.logger.debug(f"Validating endpoint: {endpoint_name}")
            
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = response.get('EndpointStatus')
            
            if endpoint_status == 'InService':
                self.logger.info(f"Endpoint {endpoint_name} is in service and ready")
                return {
                    "is_success": True,
                    "data": True,
                    "error_code": None,
                    "error_message": None
                }
            else:
                self.logger.warning(f"Endpoint {endpoint_name} status: {endpoint_status}")
                return {
                    "is_success": False,
                    "data": False,
                    "error_code": "ENDPOINT_NOT_READY",
                    "error_message": f"Endpoint {endpoint_name} is not in service (status: {endpoint_status})"
                }
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'ValidationException':
                return {
                    "is_success": False,
                    "data": False,
                    "error_code": "VALIDATION_ERROR",
                    "error_message": f"Invalid endpoint name: {endpoint_name}"
                }
            else:
                return {
                    "is_success": False,
                    "data": False,
                    "error_code": "ENDPOINT_VALIDATION_ERROR",
                    "error_message": f"Endpoint validation failed ({error_code}): {error_message}"
                }
                
        except Exception as e:
            self.logger.error(f"Unexpected error during endpoint validation: {str(e)}")
            return {
                "is_success": False,
                "data": False,
                "error_code": "UNEXPECTED_ERROR",
                "error_message": f"Endpoint validation error: {str(e)}"
            }
    
    def batch_submit_files_for_inference(
        self,
        endpoint_name: str,
        s3_urls: list,
        batch_count: int = 10,
        content_type: str = "application/octet-stream",
        accept: str = "application/json"
    ) -> Dict[str, Any]:
        """
        Submit multiple files to SageMaker async inference endpoint concurrently.
        
        Args:
            endpoint_name (str): SageMaker endpoint name/ARN
            s3_urls (list): List of S3 URIs of the input files
            batch_count (int): Maximum number of concurrent requests (default: 10)
            content_type (str): Content type of the input files
            accept (str): Accept header for response format
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        import concurrent.futures
        import time
        
        if not s3_urls:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "s3_urls list cannot be empty"
            }
        
        if batch_count <= 0:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "batch_count must be greater than 0"
            }
        
        self.logger.info(f"Starting batch submission of {len(s3_urls)} files with batch size {batch_count}")
        
        def submit_single_request(request_id, s3_url):
            """Submit a single inference request."""
            result = self.submit_file_for_inference(
                endpoint_name=endpoint_name,
                input_location=s3_url,
                content_type=content_type,
                accept=accept
            )
            
            if result["is_success"]:
                return {
                    'request_id': request_id,
                    'success': True,
                    'result': result["data"],
                    's3_url': s3_url
                }
            else:
                self.logger.error(f"Request {request_id} failed for {s3_url}: {result['error_message']}")
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': result["error_message"],
                    's3_url': s3_url
                }
        
        start_time = time.time()
        results = []
        
        try:
            # Use ThreadPoolExecutor for concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_count) as executor:
                # Submit all requests for all S3 URLs
                future_to_request = {
                    executor.submit(submit_single_request, i, s3_url): i 
                    for i, s3_url in enumerate(s3_urls, 1)
                }
                
                # Collect all results
                for future in concurrent.futures.as_completed(future_to_request):
                    request_id = future_to_request[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.debug(f"Request {request_id} completed: {result['success']}")
                    except Exception as exc:
                        self.logger.error(f"Request {request_id} generated an exception: {exc}")
                        results.append({
                            'request_id': request_id,
                            'success': False,
                            'error': str(exc),
                            's3_url': s3_urls[request_id - 1] if request_id <= len(s3_urls) else 'unknown'
                        })
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Log summary
            successful_requests = [r for r in results if r['success']]
            failed_requests = [r for r in results if not r['success']]
            
            self.logger.info(f"Batch submission completed in {total_time:.2f} seconds")
            self.logger.info(f"Total: {len(results)}, Successful: {len(successful_requests)}, Failed: {len(failed_requests)}")
            
            if failed_requests:
                self.logger.warning(f"Failed requests: {[r['request_id'] for r in failed_requests]}")
            
            # Sort results by request_id to maintain order
            results.sort(key=lambda x: x['request_id'])
            
            return {
                "is_success": True,
                "data": results,
                "error_code": None,
                "error_message": None
            }
            
        except Exception as e:
            self.logger.error(f"Batch submission failed: {str(e)}")
            return {
                "is_success": False,
                "data": None,
                "error_code": "BATCH_SUBMISSION_ERROR",
                "error_message": f"Batch submission failed: {str(e)}"
            }