"""
Configuration settings for SageMaker Migration Advisor Streamlit Application
"""

import os

# Application Settings
APP_TITLE = "SageMaker Migration Advisor"
APP_ICON = "🚀"
PAGE_LAYOUT = "wide"

# Streamlit Server Settings
DEFAULT_PORT = 8501
DEFAULT_HOST = "localhost"

# AWS Bedrock Model Configuration
BEDROCK_MODELS = [
    {
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "name": "Claude 4.5 Sonnet",
        "region": "us-west-2"
    },
    {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "name": "Claude 4 Sonnet",
        "region": "us-west-2"
    },
    {
        "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "name": "Claude 3.7 Sonnet",
        "region": "us-west-2"
    }
]

# File Paths
OUTPUT_DIR = "generated-diagrams"
INTERACTION_LOG_FILE = "streamlit_agent_interactions.txt"
TEMP_UPLOAD_DIR = "temp_uploads"

# UI Configuration
WORKFLOW_STEPS = [
    ('input', 'Architecture Input', '📋'),
    ('description', 'Architecture Analysis', '🔍'),
    ('qa', 'Clarification Q&A', '❓'),
    ('sagemaker', 'SageMaker Design', '🚀'),
    ('diagram', 'Diagram Generation', '📊'),
    ('tco', 'TCO Analysis', '💰'),
    ('navigator', 'Migration Roadmap', '🗺️')
]

# Supported file types for diagram upload
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'gif', 'bmp']

# Agent Configuration
CONVERSATION_WINDOW_SIZE = 20
MODEL_TEMPERATURE = 0.0

# Error Recovery Settings
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

# Logging Configuration
LOG_LEVEL = os.environ.get('STREAMLIT_LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Custom CSS Styles
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin: 1rem 0;
        font-weight: 600;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .agent-response {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .workflow-step {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
    }
    .step-completed {
        background-color: #d4edda;
        color: #155724;
    }
    .step-current {
        background-color: #fff3cd;
        color: #856404;
    }
    .step-pending {
        background-color: #e2e3e5;
        color: #6c757d;
    }
    .sidebar-section {
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #f8f9fa;
    }
</style>
"""

# Feature Flags
ENABLE_DIAGRAM_GENERATION = True
ENABLE_TCO_ANALYSIS = True
ENABLE_CLOUDFORMATION = False  # Optional CloudFormation generation
ENABLE_DEBUG_MODE = os.environ.get('STREAMLIT_DEBUG', 'false').lower() == 'true'

# Performance Settings
MAX_UPLOAD_SIZE_MB = 10
MAX_TEXT_LENGTH = 10000
AGENT_TIMEOUT_SECONDS = 300

# Export Settings
EXPORT_FORMATS = ['json', 'txt']
INCLUDE_TIMESTAMPS = True
INCLUDE_METADATA = True