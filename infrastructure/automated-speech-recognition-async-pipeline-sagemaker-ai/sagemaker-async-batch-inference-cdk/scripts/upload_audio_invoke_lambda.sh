#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <LAMBDA_ARN> <S3_BUCKET_URI> [AWS_PROFILE]"
    echo ""
    echo "Arguments:"
    echo "  LAMBDA_ARN       Lambda function ARN"
    echo "  S3_BUCKET_URI    S3 bucket URI (e.g., s3://my-bucket/path/)"
    echo "  AWS_PROFILE      AWS profile to use (optional, defaults to 'default')"
    echo ""
    echo "Example:"
    echo "  $0 arn:aws:lambda:us-west-2:123456789012:function:MyFunction s3://my-audio-bucket/uploads/"
    echo "  $0 arn:aws:lambda:us-west-2:123456789012:function:MyFunction s3://my-audio-bucket/uploads/ my-profile"
    exit 1
}

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: LAMBDA_ARN and S3_BUCKET_URI are required"
    usage
fi

# Configuration
LAMBDA_ARN="$1"
S3_BUCKET="$2"
AWS_PROFILE="${3:-default}"
CLOUDFRONT_URL="https://d1r3o5mtjsxzvl.cloudfront.net/sample_000000.wav"
LAMBDA_INPUT="{\"bucket_uri\": \"$S3_BUCKET\"}"

# Extract bucket name and path from S3_BUCKET for reuse
S3_BUCKET_NAME=$(echo "$S3_BUCKET" | sed 's|s3://||' | cut -d'/' -f1)
S3_BUCKET_PATH=$(echo "$S3_BUCKET" | sed 's|s3://[^/]*/||')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed"
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        print_error "AWS credentials not configured or invalid for profile '$AWS_PROFILE'"
        exit 1
    fi
    
    print_status "All dependencies are available"
}

# Download file from CloudFront
download_file() {
    local filename="sample_000000.wav"
    
    print_status "Downloading file from CloudFront..."
    
    if curl -L -o "$filename" "$CLOUDFRONT_URL" >/dev/null 2>&1; then
        print_status "File downloaded successfully: $filename"
        echo "$filename"
    else
        print_error "Failed to download file from $CLOUDFRONT_URL"
        exit 1
    fi
}

# Upload file to S3
upload_to_s3() {
    local filename="$1"
    
    print_status "Uploading file to S3..."
    
    # Check if file exists locally first
    if [ ! -f "$filename" ]; then
        print_error "Local file '$filename' does not exist"
        exit 1
    fi
    
    # Upload file to S3
    local upload_output
    if upload_output=$(aws s3 cp "$filename" "$S3_BUCKET" --profile "$AWS_PROFILE" 2>&1); then
        print_status "File uploaded successfully to $S3_BUCKET"
        print_status "Upload details: $upload_output"
    else
        print_error "Failed to upload file to S3"
        print_error "Error details: $upload_output"
        exit 1
    fi
}

# Invoke Lambda function
invoke_lambda() {
    print_status "Invoking Lambda function..."
    
    local response_file="lambda_response.json"
    
    if aws lambda invoke \
        --function-name "$LAMBDA_ARN" \
        --cli-binary-format raw-in-base64-out \
        --payload "$LAMBDA_INPUT" \
        --profile "$AWS_PROFILE" \
        "$response_file"; then
        
        print_status "Lambda function invoked successfully"
        print_status "Response saved to: $response_file"
        
        # Display response content
        if [ -f "$response_file" ]; then
            print_status "Lambda response:"
            cat "$response_file" | jq '.' 2>/dev/null || cat "$response_file"
        fi
    else
        print_error "Failed to invoke Lambda function"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    local filename="$1"
    
    print_status "Cleaning up temporary files..."
    
    if [ -f "$filename" ]; then
        rm "$filename"
        print_status "Removed local file: $filename"
    fi
    
    if [ -f "lambda_response.json" ]; then
        print_warning "Lambda response file kept: lambda_response.json"
    fi
}

# Validate Lambda ARN format
validate_lambda_arn() {
    if [[ ! "$LAMBDA_ARN" =~ ^arn:aws:lambda:[a-z0-9-]+:[0-9]+:function:[a-zA-Z0-9-_]+$ ]]; then
        print_error "Invalid Lambda ARN format: $LAMBDA_ARN"
        print_error "Expected format: arn:aws:lambda:region:account-id:function:function-name"
        exit 1
    fi
    print_status "Lambda ARN validated: $LAMBDA_ARN"
}

# Validate S3 bucket URI format
validate_s3_uri() {
    if [[ ! "$S3_BUCKET" =~ ^s3://[a-zA-Z0-9._-]+(/.*)?$ ]]; then
        print_error "Invalid S3 bucket URI format: $S3_BUCKET"
        print_error "Expected format: s3://bucket-name/path/"
        exit 1
    fi
    print_status "S3 bucket URI validated: $S3_BUCKET"
}

# Main execution
main() {
    print_status "Starting download, upload, and Lambda invocation process..."
    print_status "Using Lambda ARN: $LAMBDA_ARN"
    print_status "Using S3 bucket: $S3_BUCKET"
    print_status "Using AWS profile: $AWS_PROFILE"
    
    # Validate Lambda ARN
    validate_lambda_arn
    
    # Validate S3 URI
    validate_s3_uri
    
    # Check dependencies
    check_dependencies
    
    # Download file
    local filename
    filename=$(download_file)
    
    # Upload to S3
    upload_to_s3 "$filename"
    
    # Invoke Lambda
    invoke_lambda
    
    # Cleanup
    cleanup "$filename"
    
    print_status "Process completed successfully!"
}

# Error handling
set -e
trap 'print_error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"