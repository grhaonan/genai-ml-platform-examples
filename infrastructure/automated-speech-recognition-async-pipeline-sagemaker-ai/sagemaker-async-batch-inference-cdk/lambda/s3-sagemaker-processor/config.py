"""
Configuration and Environment Management

This module handles environment variable validation and AWS configuration.
"""

import os
import logging
from typing import Dict, Any


class ConfigManager:
    """Manages environment variables and AWS configuration."""
    
    # AWS Configuration from environment variables
    AWS_ACCOUNT_ID = os.environ.get('AWS_ACCOUNT_ID', '')
    AWS_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_environment_variables(self) -> Dict[str, Any]:
        """
        Validate that all required environment variables are set.
        
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        required_vars = {
            'SAGEMAKER_ENDPOINT_NAME': 'SageMaker endpoint name/ARN',
            'DYNAMODB_TABLE_NAME': 'DynamoDB table name for status tracking'
        }
        
        env_vars = {}
        missing_vars = []
        
        for var_name, description in required_vars.items():
            value = os.environ.get(var_name)
            if not value:
                missing_vars.append(f"{var_name} ({description})")
            else:
                env_vars[var_name] = value
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            self.logger.error(error_msg)
            return {
                "is_success": False,
                "data": None,
                "error_code": "MISSING_ENV_VARS",
                "error_message": error_msg
            }
        
        self.logger.info("Environment variables validated successfully")
        return {
            "is_success": True,
            "data": env_vars,
            "error_code": None,
            "error_message": None
        }