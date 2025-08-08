# Shared Lambda Libraries

This directory contains shared utilities that can be used across multiple Lambda functions in the project.

## Structure

```
lambda/shared/
├── __init__.py              # Package initialization
├── logger_setup.py          # Centralized logging configuration
├── config_manager.py        # Base configuration management
└── README.md               # This file
```

## Usage

### Logger Setup

```python
from shared.logger_setup import LoggerSetup

# Initialize logger with service name
logger = LoggerSetup.setup_logging('my-service')

# Or use the convenience method
logger = LoggerSetup.get_logger('my-service')
```

### Config Manager

```python
from shared.config_manager import BaseConfigManager

class MyConfigManager(BaseConfigManager):
    def __init__(self):
        super().__init__('my-service')
    
    def validate_environment_variables(self):
        required_vars = {
            'MY_VAR': 'Description of my variable'
        }
        return super().validate_environment_variables(required_vars)
```

## CDK Integration

The CDK stack automatically bundles these shared libraries with each Lambda function during deployment. The bundling process:

1. Copies the Lambda function code
2. Installs Python dependencies from `requirements.txt`
3. Copies the shared libraries to the function package

## Testing

When writing tests, make sure to add the shared library path to `sys.path`:

```python
import sys
import os

# Add lambda directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../lambda/sns-status-updater'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../lambda'))
```

## Adding New Shared Utilities

1. Create the new module in `lambda/shared/`
2. Import and use it in your Lambda functions
3. Update tests to include the shared path
4. The CDK will automatically bundle it with all Lambda functions