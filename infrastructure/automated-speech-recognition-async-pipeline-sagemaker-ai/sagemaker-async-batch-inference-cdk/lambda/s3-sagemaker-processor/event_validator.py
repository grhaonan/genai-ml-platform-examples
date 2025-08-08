"""
Event Validation and Parsing

This module handles Lambda event validation and S3 URI parsing.
"""

import re
import logging
from typing import Dict, Any, Tuple
from urllib.parse import urlparse


class EventValidator:
    """Handles Lambda event validation and S3 URI parsing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_s3_bucket_uri(self, bucket_uri: str) -> Dict[str, Any]:
        """
        Parse and validate S3 bucket URI format.
        
        Args:
            bucket_uri (str): S3 bucket URI in format s3://bucket-name/path/
            
        Returns:
            Dict[str, Any]: Response with is_success, data (tuple of bucket_name, prefix), error_code, and error_message
        """
        if not bucket_uri:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Bucket URI cannot be empty"
            }
        
        # Parse the URI
        try:
            parsed = urlparse(bucket_uri)
        except Exception as e:
            return {
                "is_success": False,
                "data": None,
                "error_code": "URI_PARSE_ERROR",
                "error_message": f"Invalid URI format: {str(e)}"
            }
        
        # Validate scheme
        if parsed.scheme != 's3':
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Invalid URI scheme '{parsed.scheme}'. Expected 's3'"
            }
        
        # Validate bucket name
        bucket_name = parsed.netloc
        if not bucket_name:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Bucket name is required in URI"
            }
        
        # Validate bucket name format (AWS S3 bucket naming rules)
        bucket_pattern = r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$'
        if not re.match(bucket_pattern, bucket_name):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Invalid bucket name format: {bucket_name}"
            }
        
        if len(bucket_name) < 3 or len(bucket_name) > 63:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Bucket name must be between 3 and 63 characters: {bucket_name}"
            }
        
        # Extract prefix (remove leading slash if present)
        prefix = parsed.path.lstrip('/')
        
        self.logger.info(f"Parsed S3 URI - Bucket: {bucket_name}, Prefix: {prefix}")
        return {
            "is_success": True,
            "data": (bucket_name, prefix),
            "error_code": None,
            "error_message": None
        }
    
    def validate_lambda_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Lambda event structure and extract bucket URI.
        
        Args:
            event (Dict[str, Any]): Lambda event payload
            
        Returns:
            Dict[str, Any]: Response with is_success, data (bucket_uri), error_code, and error_message
        """
        if not isinstance(event, dict):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Event must be a dictionary"
            }
        
        bucket_uri = event.get('bucket_uri')
        if not bucket_uri:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Missing required field 'bucket_uri' in event"
            }
        
        if not isinstance(bucket_uri, str):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Field 'bucket_uri' must be a string"
            }
        
        # Validate the bucket URI format
        parse_result = self.parse_s3_bucket_uri(bucket_uri)
        if not parse_result["is_success"]:
            return parse_result
        
        return {
            "is_success": True,
            "data": bucket_uri,
            "error_code": None,
            "error_message": None
        }