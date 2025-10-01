"""
S3 Operations

This module handles all S3-related operations including client initialization,
object listing with pagination support, and content retrieval for SageMaker inference results.
"""

import logging
import base64
import json
from typing import List, Dict, Any, Tuple, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError


class S3Operations:
    """Handles S3 operations including client initialization and object listing."""
    
    def __init__(self, region_name: str, account_id: str, session: Optional[boto3.Session] = None):
        self.region_name = region_name
        self.account_id = account_id
        self.session = session
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    def initialize_client(self) -> Dict[str, Any]:
        """
        Initialize S3 client with proper error handling.
        
        Returns:
            Dict[str, Any]: Result with is_success, client, error_code, and error_message
        """
        try:
            if self.session:
                self._client = self.session.client('s3', region_name=self.region_name)
            else:
                self._client = boto3.client('s3', region_name=self.region_name)
            self.logger.info(f"S3 client initialized successfully for account {self.account_id} in region {self.region_name}")
            return {
                'is_success': True,
                'client': self._client,
                'error_code': None,
                'error_message': None
            }
        except NoCredentialsError:
            error_msg = "AWS credentials not found. Please configure AWS credentials."
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'client': None,
                'error_code': 'NO_CREDENTIALS',
                'error_message': error_msg
            }
        except Exception as e:
            error_msg = f"Failed to initialize S3 client: {str(e)}"
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'client': None,
                'error_code': 'INITIALIZATION_FAILED',
                'error_message': error_msg
            }
    
    def get_client(self) -> Dict[str, Any]:
        """
        Get S3 client, initializing if necessary.
        
        Returns:
            Dict[str, Any]: Result with is_success, client, error_code, and error_message
        """
        if self._client is None:
            return self.initialize_client()
        
        return {
            'is_success': True,
            'client': self._client,
            'error_code': None,
            'error_message': None
        }
    
    def list_objects(self, bucket_name: str, prefix: str = "") -> Dict[str, Any]:
        """
        List all objects in S3 bucket with pagination support.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            prefix (str): Optional prefix to filter objects
            
        Returns:
            Dict[str, Any]: Result with is_success, objects, error_code, and error_message
        """
        self.logger.info(f"Listing objects in bucket '{bucket_name}' with prefix '{prefix}'")
        
        # Get S3 client
        client_result = self.get_client()
        if not client_result['is_success']:
            return {
                'is_success': False,
                'objects': [],
                'error_code': client_result['error_code'],
                'error_message': client_result['error_message']
            }
        
        client = client_result['client']
        
        try:
            objects = []
            paginator = client.get_paginator('list_objects_v2')
            
            # Configure pagination parameters
            page_iterator_params = {
                'Bucket': bucket_name,
                'PaginationConfig': {
                    'MaxItems': None,  # No limit on total items
                    'PageSize': 1000   # Process 1000 objects per page
                }
            }
            
            # Add prefix if provided
            if prefix:
                page_iterator_params['Prefix'] = prefix
            
            page_iterator = paginator.paginate(**page_iterator_params)
            
            # Iterate through all pages
            total_objects = 0
            for page in page_iterator:
                if 'Contents' in page:
                    page_objects = [obj['Key'] for obj in page['Contents']]
                    objects.extend(page_objects)
                    total_objects += len(page_objects)
                    self.logger.debug(f"Processed page with {len(page_objects)} objects. Total so far: {total_objects}")
            
            # Handle empty bucket case
            if not objects:
                self.logger.info(f"No objects found in bucket '{bucket_name}' with prefix '{prefix}'")
                return {
                    'is_success': True,
                    'objects': [],
                    'error_code': None,
                    'error_message': None
                }
            
            # Handle nested folder structures - filter out directory markers
            file_objects = []
            for obj_key in objects:
                # Skip directory markers (objects ending with '/')
                if not obj_key.endswith('/'):
                    file_objects.append(obj_key)
                else:
                    self.logger.debug(f"Skipping directory marker: {obj_key}")
            
            self.logger.info(f"Successfully listed {len(file_objects)} files from bucket '{bucket_name}'")
            return {
                'is_success': True,
                'objects': file_objects,
                'error_code': None,
                'error_message': None
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            if error_code == 'NoSuchBucket':
                error_msg = f"Bucket '{bucket_name}' does not exist"
                self.logger.error(error_msg)
                return {
                    'is_success': False,
                    'objects': [],
                    'error_code': 'NO_SUCH_BUCKET',
                    'error_message': error_msg
                }
            elif error_code == 'AccessDenied':
                error_msg = f"Access denied to bucket '{bucket_name}'. Check IAM permissions."
                self.logger.error(error_msg)
                return {
                    'is_success': False,
                    'objects': [],
                    'error_code': 'ACCESS_DENIED',
                    'error_message': error_msg
                }
            elif error_code == 'InvalidBucketName':
                error_msg = f"Invalid bucket name '{bucket_name}'"
                self.logger.error(error_msg)
                return {
                    'is_success': False,
                    'objects': [],
                    'error_code': 'INVALID_BUCKET_NAME',
                    'error_message': error_msg
                }
            else:
                error_msg = f"S3 client error ({error_code}): {error_message}"
                self.logger.error(error_msg)
                return {
                    'is_success': False,
                    'objects': [],
                    'error_code': error_code,
                    'error_message': error_msg
                }
                
        except BotoCoreError as e:
            error_msg = f"Network or connectivity error accessing S3: {str(e)}"
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'objects': [],
                'error_code': 'NETWORK_ERROR',
                'error_message': error_msg
            }
            
        except Exception as e:
            error_msg = f"Unexpected error listing S3 objects: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'is_success': False,
                'objects': [],
                'error_code': 'UNEXPECTED_ERROR',
                'error_message': error_msg
            }
    
    def delete_all_files_in_folder(self, s3_path: str) -> Dict[str, Any]:
        """
        Delete all files in an S3 folder path.
        
        Args:
            s3_path (str): S3 path in format 's3://bucket-name/folder/path/' or 'bucket-name/folder/path/'
            
        Returns:
            Dict[str, Any]: Result with is_success, deleted_count, error_code, and error_message
        """
        self.logger.info(f"Starting deletion of all files in S3 path: {s3_path}")
        
        # Parse S3 path
        parse_result = self._parse_s3_path(s3_path)
        if not parse_result['is_success']:
            return {
                'is_success': False,
                'deleted_count': 0,
                'error_code': parse_result['error_code'],
                'error_message': parse_result['error_message']
            }
        
        bucket_name = parse_result['bucket_name']
        prefix = parse_result['prefix']
        
        try:
            # List all objects in the folder
            list_result = self.list_objects(bucket_name, prefix)
            if not list_result['is_success']:
                return {
                    'is_success': False,
                    'deleted_count': 0,
                    'error_code': list_result['error_code'],
                    'error_message': list_result['error_message']
                }
            
            objects_to_delete = list_result['objects']
            
            if not objects_to_delete:
                self.logger.info(f"No files found to delete in path: {s3_path}")
                return {
                    'is_success': True,
                    'deleted_count': 0,
                    'error_code': None,
                    'error_message': None
                }
            
            self.logger.info(f"Found {len(objects_to_delete)} files to delete")
            
            # Delete objects in batches (S3 delete_objects supports up to 1000 objects per request)
            deleted_count = 0
            batch_size = 1000
            
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i:i + batch_size]
                batch_result = self._delete_objects_batch(bucket_name, batch)
                
                if not batch_result['is_success']:
                    return {
                        'is_success': False,
                        'deleted_count': deleted_count,
                        'error_code': batch_result['error_code'],
                        'error_message': batch_result['error_message']
                    }
                
                deleted_count += batch_result['deleted_count']
            
            self.logger.info(f"Successfully deleted {deleted_count} files from {s3_path}")
            return {
                'is_success': True,
                'deleted_count': deleted_count,
                'error_code': None,
                'error_message': None
            }
            
        except Exception as e:
            error_msg = f"Failed to delete files in S3 path '{s3_path}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'is_success': False,
                'deleted_count': 0,
                'error_code': 'UNEXPECTED_ERROR',
                'error_message': error_msg
            }
    
    def _parse_s3_path(self, s3_path: str) -> Dict[str, Any]:
        """
        Parse S3 path to extract bucket name and prefix.
        
        Args:
            s3_path (str): S3 path in various formats
            
        Returns:
            Dict[str, Any]: Result with is_success, bucket_name, prefix, error_code, and error_message
        """
        if not s3_path or not isinstance(s3_path, str):
            error_msg = "S3 path must be a non-empty string"
            return {
                'is_success': False,
                'bucket_name': None,
                'prefix': None,
                'error_code': 'INVALID_PATH',
                'error_message': error_msg
            }
        
        # Remove leading/trailing whitespace
        s3_path = s3_path.strip()
        
        # Handle s3:// format
        if s3_path.startswith('s3://'):
            path_without_protocol = s3_path[5:]  # Remove 's3://'
        else:
            path_without_protocol = s3_path
        
        # Split into bucket and prefix
        path_parts = path_without_protocol.split('/', 1)
        
        if not path_parts[0]:
            error_msg = f"Invalid S3 path format: {s3_path}"
            return {
                'is_success': False,
                'bucket_name': None,
                'prefix': None,
                'error_code': 'INVALID_PATH_FORMAT',
                'error_message': error_msg
            }
        
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ""
        
        # Ensure prefix ends with '/' if it's not empty (for folder operations)
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        self.logger.debug(f"Parsed S3 path '{s3_path}' -> bucket: '{bucket_name}', prefix: '{prefix}'")
        return {
            'is_success': True,
            'bucket_name': bucket_name,
            'prefix': prefix,
            'error_code': None,
            'error_message': None
        }
    
    def _delete_objects_batch(self, bucket_name: str, object_keys: List[str]) -> Dict[str, Any]:
        """
        Delete a batch of objects from S3.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            object_keys (List[str]): List of object keys to delete
            
        Returns:
            Dict[str, Any]: Result with is_success, deleted_count, error_code, and error_message
        """
        if not object_keys:
            return {
                'is_success': True,
                'deleted_count': 0,
                'error_code': None,
                'error_message': None
            }
        
        # Get S3 client
        client_result = self.get_client()
        if not client_result['is_success']:
            return {
                'is_success': False,
                'deleted_count': 0,
                'error_code': client_result['error_code'],
                'error_message': client_result['error_message']
            }
        
        client = client_result['client']
        
        try:
            # Prepare delete request
            delete_request = {
                'Objects': [{'Key': key} for key in object_keys],
                'Quiet': False  # Return information about deleted objects
            }
            
            self.logger.debug(f"Deleting batch of {len(object_keys)} objects from bucket '{bucket_name}'")
            
            # Execute batch delete
            response = client.delete_objects(
                Bucket=bucket_name,
                Delete=delete_request
            )
            
            # Count successful deletions
            deleted_count = len(response.get('Deleted', []))
            
            # Log any errors
            if 'Errors' in response and response['Errors']:
                for error in response['Errors']:
                    self.logger.warning(f"Failed to delete object '{error['Key']}': {error['Message']}")
            
            self.logger.debug(f"Successfully deleted {deleted_count} objects in batch")
            return {
                'is_success': True,
                'deleted_count': deleted_count,
                'error_code': None,
                'error_message': None
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            error_msg = f"S3 batch delete error ({error_code}): {error_message}"
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'deleted_count': 0,
                'error_code': error_code,
                'error_message': error_msg
            }
            
        except Exception as e:
            error_msg = f"Unexpected error during batch delete: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'is_success': False,
                'deleted_count': 0,
                'error_code': 'UNEXPECTED_ERROR',
                'error_message': error_msg
            }
    
    def extract_s3_output_location(self, sns_message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract S3 output location from SNS message payload.
        
        Args:
            sns_message_data (Dict[str, Any]): Validated SNS message data
            
        Returns:
            Dict[str, Any]: Result with is_success, output_location, error_code, and error_message
        """
        self.logger.debug("Extracting S3 output location from SNS message data")
        
        if not isinstance(sns_message_data, dict):
            error_msg = "SNS message data must be a dictionary"
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'output_location': None,
                'error_code': 'INVALID_INPUT',
                'error_message': error_msg
            }
        
        output_location = sns_message_data.get('output_location')
        
        if not output_location:
            error_msg = "No output_location found in SNS message data"
            self.logger.warning(error_msg)
            return {
                'is_success': False,
                'output_location': None,
                'error_code': 'NO_OUTPUT_LOCATION',
                'error_message': error_msg
            }
        
        if not isinstance(output_location, str):
            error_msg = "Output location must be a string"
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'output_location': None,
                'error_code': 'INVALID_OUTPUT_LOCATION',
                'error_message': error_msg
            }
        
        output_location = output_location.strip()
        if not output_location.startswith('s3://'):
            error_msg = f"Invalid S3 output location format: '{output_location}'. Must start with 's3://'"
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'output_location': None,
                'error_code': 'INVALID_S3_URI',
                'error_message': error_msg
            }
        
        self.logger.info(f"Successfully extracted S3 output location: {output_location}")
        return {
            'is_success': True,
            'output_location': output_location,
            'error_code': None,
            'error_message': None
        }
    
    def read_inference_output_file(self, s3_output_location: str) -> Dict[str, Any]:
        """
        Read and decode SageMaker inference output file from S3.
        
        Args:
            s3_output_location (str): S3 URI of the output file
            
        Returns:
            Dict[str, Any]: Result with is_success, content, decoded_content, error_code, and error_message
        """
        self.logger.info(f"Reading inference output file from: {s3_output_location}")
        
        # Parse S3 path
        parse_result = self._parse_s3_path(s3_output_location)
        if not parse_result['is_success']:
            return {
                'is_success': False,
                'content': None,
                'decoded_content': None,
                'error_code': parse_result['error_code'],
                'error_message': parse_result['error_message']
            }
        
        bucket_name = parse_result['bucket_name']
        object_key = parse_result['prefix'].rstrip('/')  # Remove trailing slash for file operations
        
        if not object_key:
            error_msg = f"Invalid S3 output location: '{s3_output_location}'. No object key found."
            self.logger.error(error_msg)
            return {
                'is_success': False,
                'content': None,
                'decoded_content': None,
                'error_code': 'INVALID_OBJECT_KEY',
                'error_message': error_msg
            }
        
        # Get S3 client
        client_result = self.get_client()
        if not client_result['is_success']:
            return {
                'is_success': False,
                'content': None,
                'decoded_content': None,
                'error_code': client_result['error_code'],
                'error_message': client_result['error_message']
            }
        
        client = client_result['client']
        
        try:
            self.logger.debug(f"Attempting to read object: bucket='{bucket_name}', key='{object_key}'")
            
            # Read the object from S3
            response = client.get_object(Bucket=bucket_name, Key=object_key)
            
            # Read the content
            raw_content = response['Body'].read()
            
            # Convert bytes to string
            if isinstance(raw_content, bytes):
                content = raw_content.decode('utf-8')
            else:
                content = str(raw_content)
            
            self.logger.info(f"Successfully read {len(content)} characters from S3 object")
            
            # Attempt to decode base64 content
            decode_result = self._decode_base64_content(content)
            
            return {
                'is_success': True,
                'content': content,
                'decoded_content': decode_result['decoded_content'],
                'error_code': None,
                'error_message': None
            }
            
        except Exception as e:
            error_msg = f"Error reading inference output file: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'is_success': False,
                'content': None,
                'error_code': 'S3_READ_ERROR',
                'error_message': error_msg
            }
    
    def _decode_base64_content(self, content: str) -> Dict[str, Any]:
        """
        Attempt to decode base64 content if present.
        
        Args:
            content (str): Raw content from S3 file
            
        Returns:
            Dict[str, Any]: Result with decoded_content (None if not base64 or decode failed)
        """
        if not content or not isinstance(content, str):
            return {'decoded_content': None}
        
        content = content.strip()
        
        # Check if content looks like base64 (basic heuristic)
        if not self._is_likely_base64(content):
            self.logger.debug("Content does not appear to be base64 encoded")
            return {'decoded_content': None}
        
        try:
            # Attempt to decode base64
            decoded_bytes = base64.b64decode(content, validate=True)
            decoded_content = decoded_bytes.decode('utf-8')
            
            self.logger.info(f"Successfully decoded base64 content ({len(decoded_content)} characters)")
            return {'decoded_content': decoded_content}
            
        except (base64.binascii.Error, UnicodeDecodeError, ValueError) as e:
            self.logger.debug(f"Failed to decode as base64: {str(e)}")
            return {'decoded_content': None}
        except Exception as e:
            self.logger.warning(f"Unexpected error during base64 decode: {str(e)}")
            return {'decoded_content': None}
    
    def _is_likely_base64(self, content: str) -> bool:
        """
        Check if content is likely to be base64 encoded.
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content looks like base64
        """
        if not content:
            return False
        
        # Remove whitespace
        content = content.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')
        
        # Check length (base64 length should be multiple of 4)
        if len(content) % 4 != 0:
            return False
        
        # Check if all characters are valid base64 characters
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        if not all(c in base64_chars for c in content):
            return False
        
        # Check for reasonable length (avoid trying to decode very short strings)
        if len(content) < 4:
            return False
        
        return True