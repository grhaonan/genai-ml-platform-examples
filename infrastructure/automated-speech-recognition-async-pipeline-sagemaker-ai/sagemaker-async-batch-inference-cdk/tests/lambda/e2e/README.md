# End-to-End (E2E) Tests

This directory contains true end-to-end tests for the Lambda functions using **real AWS services**. These tests demonstrate real-world usage scenarios with actual AWS S3, DynamoDB, and SageMaker services.

## ⚠️ Prerequisites

### AWS Credentials
These tests require valid AWS credentials configured via:
- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- IAM roles (when running on EC2/Lambda)
- AWS SSO profiles

### Required AWS Permissions
The tests need permissions for:
- **S3**: `s3:ListBucket`, `s3:GetObject` on the test bucket
- **DynamoDB**: `dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:UpdateItem` (when implemented)
- **SageMaker**: `sagemaker:InvokeEndpointAsync` (when implemented)

## Test Structure

- `test_real_usage_demo.py`: Comprehensive E2E tests using real AWS services
- `conftest.py`: Pytest configuration and fixtures for E2E tests
- `__init__.py`: Package initialization

## Running the Tests

### Run all E2E tests
```bash
python -m pytest tests/lambda/e2e/ -v
```

### Run with detailed output (including print statements)
```bash
python -m pytest tests/lambda/e2e/ -v -s
```

### Run only integration tests
```bash
python -m pytest tests/lambda/e2e/ -v -m integration
```

### Run a specific test
```bash
python -m pytest tests/lambda/e2e/test_real_usage_demo.py::TestLambdaRealUsageDemo::test_successful_lambda_invocation -v -s
```

### Skip tests if AWS not available
```bash
python -m pytest tests/lambda/e2e/ -v --tb=short
```

## Test Categories

### Markers
- `@pytest.mark.e2e`: All tests in this directory (automatically applied)
- `@pytest.mark.integration`: Tests that use multiple AWS services together

### Test Types
1. **Real AWS Service Tests**: Test actual AWS S3, DynamoDB, SageMaker operations
2. **Error Handling Tests**: Test various error scenarios with real services
3. **Service Availability Tests**: Verify AWS services are accessible
4. **Integration Tests**: Test full workflow with real AWS infrastructure

## Environment Setup

The tests use:
- **Real AWS clients** (boto3) - no mocking
- Required environment variables for Lambda configuration
- Actual S3 bucket: `s3://sagemaker-async-input/20250730/`
- Real DynamoDB tables and SageMaker endpoints (when configured)

## Test Behavior

### Successful Tests
- Connect to real AWS services
- List actual files from S3 bucket
- Demonstrate real Lambda execution flow
- Show actual AWS operation results

### Skipped Tests
Tests are automatically skipped when:
- AWS credentials are not available (`NoCredentialsError`)
- AWS permissions are insufficient (`AccessDenied`)
- AWS resources don't exist (`NoSuchBucket`, `NoSuchKey`)
- Network/infrastructure issues occur

### Error Tests
- Test validation with real AWS error responses
- Demonstrate error handling with actual AWS exceptions
- Verify graceful degradation patterns

## Sample S3 URL

The tests use the real S3 URL: `s3://sagemaker-async-input/20250730/`

This bucket and prefix must exist and be accessible for successful test execution.

## Expected Output

When running the tests, you'll see:
- Real AWS service connections and operations
- Actual files discovered in S3 bucket
- Live AWS API responses and status codes
- Real error handling with AWS exceptions
- Performance metrics from actual AWS calls

## Cost Considerations

These E2E tests make real AWS API calls which may incur small costs:
- S3 LIST operations: ~$0.0004 per 1,000 requests
- DynamoDB operations: Varies by table configuration
- SageMaker inference: Varies by endpoint configuration

## Future Enhancements

These E2E tests can be extended to include:
- Multi-region testing with real AWS regions
- Performance testing with actual AWS latency
- Load testing with real AWS service limits
- Cost optimization testing with real AWS pricing
- Disaster recovery testing with real AWS failover scenarios