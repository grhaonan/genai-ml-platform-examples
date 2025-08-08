"""
Unit tests for BedrockOperations class.

Tests AWS Bedrock API integration and content summarization functionality
with comprehensive error handling scenarios.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError
import sys
import os

# Add the lambda directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../lambda/sns-status-updater'))

from bedrock_operations import BedrockOperations


class TestBedrockOperations:
    """Test cases for BedrockOperations class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.bedrock_ops = BedrockOperations()
        
    def test_generate_summary_success(self):
        """Test successful content summarization."""
        test_content = "Thousands of Catholic faithful are expected to descend upon Rome in the coming weeks, as cardinals from around the world gather at the Vatican to elect a new pope.A new pontiff is expected to be chosen within three days of the conclave, which is set to begin on May 7, and large crowds will likely wait outside Saint Peter's Basilica in anticipation.But for many Catholics, this is a particularly profound time to travel to Rome, regardless of the conclave.This year is a Jubilee year, during which Catholics the world over are encouraged to make a pilgrimage to Rome to reaffirm their faith."
        success, summary, error_msg = self.bedrock_ops.generate_summary(test_content)
        print(summary)
        
        assert success is True


if __name__ == '__main__':
    pytest.main([__file__])