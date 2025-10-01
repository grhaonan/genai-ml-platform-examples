"""
Shared Configuration Management

This module provides centralized configuration and environment variable management
for Lambda functions.
"""

import os
import logging
from typing import Dict, List, Optional, Any


class BaseConfigManager:
    """Base configuration manager with common functionality."""
    
    # AWS Configuration constants
    AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID', '')
    AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
    
    def __init__(self, service_name: str = None):
        self.service_name = service_name or 'unknown-service'
        self.logger = logging.getLogger(self.service_name)
    
    def validate_environment_variables(self, required_vars: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate that all required environment variables are set.
        
        Args:
            required_vars (Dict[str, str]): Dictionary mapping variable names to descriptions
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
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
    
    def get_optional_env_var(self, var_name: str, default_value: str = None) -> Optional[str]:
        """
        Get an optional environment variable with a default value.
        
        Args:
            var_name (str): Environment variable name
            default_value (str, optional): Default value if not set
            
        Returns:
            Optional[str]: Environment variable value or default
        """
        value = os.environ.get(var_name, default_value)
        if value:
            self.logger.debug(f"Environment variable {var_name} = {value}")
        return value
    
    def validate_aws_region(self, region: str = None) -> Dict[str, Any]:
        """
        Validate AWS region configuration.
        
        Args:
            region (str, optional): Region to validate, defaults to AWS_REGION
            
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        region = region or self.AWS_REGION
        
        # Basic region format validation
        if not region or not isinstance(region, str):
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": "AWS region must be a non-empty string"
            }
        
        # Check for basic AWS region format (e.g., us-west-2)
        if len(region.split('-')) < 3:
            return {
                "is_success": False,
                "data": None,
                "error_code": "VALIDATION_ERROR",
                "error_message": f"Invalid AWS region format: {region}"
            }
        
        return {
            "is_success": True,
            "data": region,
            "error_code": None,
            "error_message": None
        }