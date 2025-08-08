import * as cdk from 'aws-cdk-lib';

/**
 * Configuration interface for the S3 SageMaker Processor stack
 */
export interface S3SageMakerProcessorStackProps extends cdk.StackProps {
  /**
   * Configuration for the Lambda function
   */
  lambdaConfig?: LambdaFunctionConfig;
  
  /**
   * Configuration for the DynamoDB table
   */
  dynamoDbConfig?: DynamoDbConfig;
  
  /**
   * Configuration for SageMaker integration
   */
  sageMakerConfig?: SageMakerConfig;
}

/**
 * Configuration interface for the Lambda function
 */
export interface LambdaFunctionConfig {
  /**
   * Lambda function timeout in minutes (default: 15)
   */
  timeoutMinutes?: number;
  
  /**
   * Lambda function memory allocation in MB (default: 512)
   */
  memorySize?: number;
  
  /**
   * Python runtime version (default: PYTHON_3_9)
   */
  runtime?: string;
  
  /**
   * Log level for the Lambda function (default: INFO)
   */
  logLevel?: string;
  
  /**
   * Path to the Lambda function code directory
   */
  codePath?: string;
}

/**
 * Configuration interface for DynamoDB table
 */
export interface DynamoDbConfig {
  /**
   * DynamoDB table name (if not provided, will be auto-generated)
   */
  tableName?: string;
  
  /**
   * Billing mode for the DynamoDB table (default: ON_DEMAND)
   */
  billingMode?: string;
  
  /**
   * Whether to enable point-in-time recovery (default: false)
   */
  pointInTimeRecovery?: boolean;
  
  /**
   * Whether to create the InferenceIdIndex GSI (default: true)
   */
  enableInferenceIdIndex?: boolean;
}

/**
 * Configuration interface for SageMaker integration
 */
export interface SageMakerConfig {
  /**
   * SageMaker endpoint name or ARN
   * This will be passed as an environment variable to the Lambda function
   */
  endpointName?: string;
  
  /**
   * Whether to create IAM permissions for SageMaker access (default: true)
   */
  enableSageMakerAccess?: boolean;
}

/**
 * Interface for Lambda function environment variables
 */
export interface LambdaEnvironmentVariables {
  /**
   * DynamoDB table name for status tracking
   */
  DYNAMODB_TABLE_NAME: string;
  
  /**
   * SageMaker endpoint name/ARN for async inference
   */
  SAGEMAKER_ENDPOINT_NAME?: string;
  
  /**
   * Logging level for the Lambda function
   */
  LOG_LEVEL: string;
  
  /**
   * Index signature to allow additional environment variables
   */
  [key: string]: string | undefined;
}

/**
 * Interface for IAM policy configuration
 */
export interface IAMPolicyConfig {
  /**
   * S3 bucket ARNs that the Lambda function should have access to
   */
  s3BucketArns?: string[];
  
  /**
   * Whether to allow access to all S3 buckets (default: false)
   * Use with caution in production environments
   */
  allowAllS3Access?: boolean;
  
  /**
   * Additional IAM policy statements to attach to the Lambda role
   */
  additionalPolicyStatements?: any[];
}