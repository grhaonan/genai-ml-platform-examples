"""
SNS Message Processing and Validation

This module handles parsing and validation of SNS messages from SageMaker
async inference completion events.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime


class SNSMessageProcessor:
    """Handles SNS message parsing and validation for SageMaker async inference events."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_and_parse_sns_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and parse an SNS record from Lambda event.
        
        Args:
            record (Dict[str, Any]): SNS record from Lambda event
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        self.logger.debug(f"Validating SNS record: {json.dumps(record, default=str)}")
        
        # Validate SNS record structure
        if not isinstance(record, dict):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "SNS record must be a dictionary"
            }
        
        if 'Sns' not in record:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Invalid SNS record: missing 'Sns' field"
            }
        
        sns_message = record['Sns']
        if not isinstance(sns_message, dict):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "SNS message must be a dictionary"
            }
        
        # Validate required SNS fields
        required_sns_fields = ['Message', 'MessageId', 'TopicArn', 'Timestamp']
        for field in required_sns_fields:
            if field not in sns_message:
                return {
                    "is_success": False,
                    "data": None,
                    "error_code": "VALIDATION_ERROR",
                    "error_message": f"Invalid SNS message: missing '{field}' field"
                }
        
        # Extract and validate message content
        message_body = sns_message['Message']
        if not isinstance(message_body, str):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "SNS message body must be a string"
            }
        
        # Parse the message JSON
        try:
            message_data = json.loads(message_body)
        except json.JSONDecodeError as e:
            return {
                "is_success": False,
                "data": None,
                "error_code": "JSON_PARSE_ERROR",
                "error_message": f"Invalid JSON in SNS message body: {str(e)}"
            }
        
        # Validate SageMaker message format
        validation_result = self._validate_sagemaker_message(message_data)
        if not validation_result["is_success"]:
            return validation_result
        
        validated_data = validation_result["data"]
        self.logger.info("validated_data: %s", json.dumps(validated_data, default=str))
        
        # Add SNS metadata
        validated_data['sns_metadata'] = {
            'message_id': sns_message['MessageId'],
            'topic_arn': sns_message['TopicArn'],
            'timestamp': sns_message['Timestamp']
        }
        
        self.logger.info(f"Successfully validated SNS message for inference ID: {validated_data.get('inference_id', 'unknown')}")
        return {
            "is_success": True,
            "data": validated_data,
            "error_code": None,
            "error_message": None
        }
    
    def _validate_sagemaker_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate SageMaker async inference completion message format.
        
        Args:
            message_data (Dict[str, Any]): Parsed message data from SNS
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        if not isinstance(message_data, dict):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "SageMaker message data must be a dictionary"
            }
        
        # Required fields for SageMaker async inference completion
        required_fields = ['inferenceId', 'invocationStatus']
        for field in required_fields:
            if field not in message_data:
                return {
                    "is_success": False,
                    "data": None,
                    "error_code": "VALIDATION_ERROR",
                    "error_message": f"Missing required field in SageMaker message: '{field}'"
                }
        
        # Validate inference ID
        inference_id = message_data['inferenceId']
        if not isinstance(inference_id, str) or not inference_id.strip():
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "inferenceId must be a non-empty string"
            }
        
        # Validate optional fields if present
        completion_time = message_data.get("completionTime")
        creation_time = message_data.get("creationTime")
        
        # Extract locations from nested parameters
        locations_result = self._extract_locations(message_data)
        if not locations_result["is_success"]:
            return locations_result
        
        input_location = locations_result["data"]["input_location"]
        output_location = locations_result["data"]["output_location"]
        
        failure_result = self._validate_failure_reason(message_data.get('failureReason'), message_data.get('invocationStatus'))
        if not failure_result["is_success"]:
            return failure_result
        failure_reason = failure_result["data"]
        
        validated_data = {
            'inference_id': inference_id.strip(),
            'invocationStatus': message_data.get('invocationStatus'),
            'completion_time': completion_time,
            'creation_time': creation_time,
            'input_location': input_location,
            'output_location': output_location,
            'failure_reason': failure_reason,
        }
        
        self.logger.debug(f"Validated SageMaker message data: {json.dumps(validated_data, default=str)}")
        return {
            "is_success": True,
            "data": validated_data,
            "error_code": None,
            "error_message": None
        }
    
    def _validate_timestamp(self, timestamp: Optional[str]) -> Dict[str, Any]:
        """
        Validate timestamp format.
        
        Args:
            timestamp (Optional[str]): Timestamp string to validate
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        if timestamp is None:
            return {
                "is_success": True,
                "data": None,
                "error_code": None,
                "error_message": None
            }
        
        if not isinstance(timestamp, str):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Timestamp must be a string"
            }
        
        # Try to parse ISO format timestamp
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return {
                "is_success": True,
                "data": timestamp,
                "error_code": None,
                "error_message": None
            }
        except ValueError:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Invalid timestamp format: '{timestamp}'. Expected ISO format."
            }
    
    def _validate_s3_location(self, location: Optional[str]) -> Dict[str, Any]:
        """
        Validate S3 location format.
        
        Args:
            location (Optional[str]): S3 location to validate
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        if location is None:
            return {
                "is_success": True,
                "data": None,
                "error_code": None,
                "error_message": None
            }
        
        if not isinstance(location, str):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "S3 location must be a string"
            }
        
        location = location.strip()
        if not location:
            return {
                "is_success": True,
                "data": None,
                "error_code": None,
                "error_message": None
            }
        
        # Basic S3 URI validation
        if not location.startswith('s3://'):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Invalid S3 location format: '{location}'. Must start with 's3://'"
            }
        
        # Check for minimum valid S3 URI structure
        if len(location) < 6:  # s3://x
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Invalid S3 location format: '{location}'. Too short."
            }
        
        return {
            "is_success": True,
            "data": location,
            "error_code": None,
            "error_message": None
        }
    
    def _extract_locations(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate S3 locations from message data"""
        try:
            self.logger.info("Extracting S3 locations from message data")
            
            # Extract input location from requestParameters
            request_params = message_data.get('requestParameters', {})
            input_location_raw = request_params.get('inputLocation')
            
            if not input_location_raw:
                return {
                    "is_success": False,
                    "error_message": "Input location not found in requestParameters",
                    "error_code": "MISSING_INPUT_LOCATION"
                }
            
            s3_result = self._validate_s3_location(input_location_raw)
            if not s3_result["is_success"]:
                return s3_result
            input_location = s3_result["data"]
            
            # Extract output location from responseParameters
            response_params = message_data.get('responseParameters', {})
            output_location_raw = response_params.get('outputLocation')
            
            if not output_location_raw:
                return {
                    "is_success": False,
                    "error_message": "Output location not found in responseParameters",
                    "error_code": "MISSING_OUTPUT_LOCATION"
                }
            
            s3_result = self._validate_s3_location(output_location_raw)
            if not s3_result["is_success"]:
                return s3_result
            output_location = s3_result["data"]
            
            return {
                "is_success": True,
                "data": {
                    "input_location": input_location,
                    "output_location": output_location
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to extract locations: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "is_success": False,
                "error_message": error_msg,
                "error_code": "LOCATION_EXTRACTION_ERROR"
            }

    def _validate_failure_reason(self, failure_reason: Optional[str], status: str) -> Dict[str, Any]:
        """Validate failure reason based on status.
        
        Args:
            failure_reason (Optional[str]): Failure reason to validate
            status (str): Job status
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        if failure_reason is None:
            return {
                "is_success": True,
                "data": None,
                "error_code": None,
                "error_message": None
            }
        
        if not isinstance(failure_reason, str):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Failure reason must be a string"
            }
        
        failure_reason = failure_reason.strip()
        if not failure_reason:
            return {
                "is_success": True,
                "data": None,
                "error_code": None,
                "error_message": None
            }
        
        # Log warning if failure reason is provided for successful jobs
        if status == 'COMPLETED' and failure_reason:
            self.logger.warning(f"Failure reason provided for completed job: '{failure_reason}'")
        
        return {
            "is_success": True,
            "data": failure_reason,
            "error_code": None,
            "error_message": None
        }
    
    def extract_inference_details(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key details for DynamoDB update operations.
        
        Args:
            validated_data (Dict[str, Any]): Validated message data
            
        Returns:
            Dict[str, Any]: Extracted details for database operations
        """
        return {
            'inference_id': validated_data['inference_id'],
            'invocationStatus': validated_data['invocationStatus'],  # Convert to lowercase for consistency
            'completion_timestamp': validated_data.get('completion_time'),
            'input_location': validated_data.get('input_location'),
            'output_location': validated_data.get('output_location'),
            'failure_reason': validated_data.get('failure_reason'),
            'sns_message_id': validated_data.get('sns_metadata', {}).get('message_id')
        }