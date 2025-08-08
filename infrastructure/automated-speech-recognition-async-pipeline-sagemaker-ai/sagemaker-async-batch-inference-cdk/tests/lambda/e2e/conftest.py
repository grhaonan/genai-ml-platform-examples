"""
Pytest configuration for E2E tests.
"""

import pytest
import sys
import os

def pytest_configure(config):
    """Configure pytest for E2E tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add e2e marker to all tests in this directory
        item.add_marker(pytest.mark.e2e)
        
        # Add integration marker to tests with 'integration' in name
        if 'integration' in item.name or 'workflow' in item.name:
            item.add_marker(pytest.mark.integration)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment for E2E tests."""
    # Ensure the lambda directory is in the path
    lambda_dir = os.path.join(os.path.dirname(__file__), '../../../lambda/s3-sagemaker-processor')
    if lambda_dir not in sys.path:
        sys.path.insert(0, lambda_dir)
    
    yield
    
    # Cleanup if needed
    pass