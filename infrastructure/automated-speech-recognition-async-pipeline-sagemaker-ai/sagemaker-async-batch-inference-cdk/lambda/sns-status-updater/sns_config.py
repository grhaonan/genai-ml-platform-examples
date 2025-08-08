"""
Configuration and Environment Management for SNS Status Updater

This module handles environment variable validation and AWS configuration
for the SNS status updater Lambda function.
"""

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

from config_manager import BaseConfigManager


class ConfigManager(BaseConfigManager):
    """Manages environment variables and AWS configuration for SNS status updater."""
    
    def __init__(self):
        super().__init__('sns-status-updater')
    
    def validate_environment_variables(self) -> Dict[str, Any]:
        """
        Validate that all required environment variables are set for SNS status updater.
        
        Returns:
            Dict[str, Any]: Response with is_success, data, error_code, and error_message
        """
        required_vars = {
            'DYNAMODB_TABLE_NAME': 'DynamoDB table name for status tracking'
        }
        
        return super().validate_environment_variables(required_vars)