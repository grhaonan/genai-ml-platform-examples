import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as snsSubscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import { NagSuppressions } from 'cdk-nag';

import { Construct } from 'constructs';
import {
  S3SageMakerProcessorStackProps,
  LambdaFunctionConfig,
  DynamoDbConfig,
  SageMakerConfig
} from './types';

export class AwsBlogSagemakerStack extends cdk.Stack {
  public readonly s3ProcessorFunction: lambda.Function;
  public readonly snsStatusUpdaterFunction: lambda.Function;
  public readonly dynamoDbTable: dynamodb.Table;
  public readonly snsStatusTopic: sns.ITopic;
  public readonly s3ProcessorRole: iam.Role;
  public readonly snsUpdaterRole: iam.Role;

  constructor(scope: Construct, id: string, props?: S3SageMakerProcessorStackProps) {
    super(scope, id, props);

    // Extract configuration with defaults
    const lambdaConfig: LambdaFunctionConfig = {
      timeoutMinutes: 15,
      memorySize: 512,
      runtime: 'python3.13',
      logLevel: 'INFO',
      codePath: 'lambda/s3-sagemaker-processor',
      ...props?.lambdaConfig
    };

    const dynamoConfig: DynamoDbConfig = {
      billingMode: 'ON_DEMAND',
      pointInTimeRecovery: true,
      enableInferenceIdIndex: true,
      ...props?.dynamoDbConfig
    };

    const sageMakerConfig: SageMakerConfig = {
      enableSageMakerAccess: true,
      ...props?.sageMakerConfig
    };

    // DynamoDB table for tracking file processing status
    this.dynamoDbTable = new dynamodb.Table(this, 'FileProcessingStatusTable', {
      tableName: dynamoConfig.tableName,
      partitionKey: {
        name: 'file_path',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamoConfig.billingMode === 'ON_DEMAND'
        ? dynamodb.BillingMode.PAY_PER_REQUEST
        : dynamodb.BillingMode.PROVISIONED,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: dynamoConfig.pointInTimeRecovery || false
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY, // For development/testing
    });

    // Add Global Secondary Index for inference_id lookups (if enabled)
    if (dynamoConfig.enableInferenceIdIndex) {
      this.dynamoDbTable.addGlobalSecondaryIndex({
        indexName: 'InferenceIdIndex',
        partitionKey: {
          name: 'inference_id',
          type: dynamodb.AttributeType.STRING
        },
        projectionType: dynamodb.ProjectionType.ALL, // Project all attributes for full record access
      });
    }

    // Create IAM role for S3 SageMaker Processor Lambda function with required permissions
    this.s3ProcessorRole = new iam.Role(this, 'S3SageMakerProcessorRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'IAM role for S3 SageMaker Processor Lambda function',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')
      ]
    });

    // Add S3 permissions - restrict to SageMaker bucket pattern for improved security
    this.s3ProcessorRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:ListBucket',
        's3:GetObject'
      ],
      resources: [
        `arn:aws:s3:::sagemaker-${this.region}-${this.account}`,
        `arn:aws:s3:::sagemaker-${this.region}-${this.account}/*`
      ]
    }));

    // Add DynamoDB permissions for the status table
    this.s3ProcessorRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:PutItem',
        'dynamodb:UpdateItem',
        'dynamodb:Query',
        'dynamodb:BatchGetItem'
      ],
      resources: [
        this.dynamoDbTable.tableArn,
        `${this.dynamoDbTable.tableArn}/index/*` // Allow access to all GSIs
      ]
    }));

    // Add SageMaker permissions for async inference
    if (sageMakerConfig.enableSageMakerAccess) {
      const sageMakerEndpointArn = sageMakerConfig.endpointName 
        ? `arn:aws:sagemaker:${this.region}:${this.account}:endpoint/${sageMakerConfig.endpointName}`
        : `arn:aws:sagemaker:${this.region}:${this.account}:endpoint/*`;
        
      this.s3ProcessorRole.addToPolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'sagemaker:InvokeEndpointAsync',
          'sagemaker:DescribeEndpoint'
        ],
        resources: [sageMakerEndpointArn]
      }));
    }

    // Add CloudWatch Logs permissions (additional to basic execution role)
    this.s3ProcessorRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents'
      ],
      resources: ['*']
    }));

    // Import existing SNS topic for status updates
    this.snsStatusTopic = sns.Topic.fromTopicArn(this, 'SageMakerStatusTopic',
      `arn:aws:sns:${this.region}:${this.account}:success-inf`
    );

    // Import external SNS topic for failed inference notifications
    const failedInferenceTopic = sns.Topic.fromTopicArn(this, 'FailedInferenceTopic',
      `arn:aws:sns:${this.region}:${this.account}:failed-inf`
    );

    // Add SNS permissions for publishing status updates (after topic creation)
    this.s3ProcessorRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'sns:Publish'
      ],
      resources: [this.snsStatusTopic.topicArn]
    }));

    // Add CDK-nag suppressions for S3 Processor Role
    NagSuppressions.addResourceSuppressions(
      this.s3ProcessorRole,
      [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'AWSLambdaBasicExecutionRole is required for Lambda functions to write logs to CloudWatch',
          appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Lambda needs access to all DynamoDB GSI indexes for querying by inference_id',
          appliesTo: [`Resource::<FileProcessingStatusTable3F2FBEB5.Arn>/index/*`]
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'S3 object access is restricted to SageMaker bucket pattern for security',
          appliesTo: [
            `Resource::arn:aws:s3:::sagemaker-<AWS::Region>-<AWS::AccountId>/*`,
            `Resource::arn:aws:s3:::sagemaker-${this.region}-${this.account}/*`
          ]
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'CloudWatch Logs permissions require wildcard for log group creation',
          appliesTo: ['Resource::*']
        },

      ],
      true // Apply to child constructs (DefaultPolicy)
    );

    // Create environment variables for Lambda function
    const environmentVariables = this.createLambdaEnvironmentVariables(
      sageMakerConfig.endpointName,
      lambdaConfig.logLevel
    );

    // Create S3 SageMaker Processor Lambda function
    this.s3ProcessorFunction = this.createLambdaFunction(
      'S3SageMakerProcessor',
      'lambda/s3-sagemaker-processor',
      this.s3ProcessorRole,
      environmentVariables,
      'Lambda function to process S3 files through SageMaker async inference'
    );



    // Create IAM role for SNS Status Updater Lambda function
    this.snsUpdaterRole = new iam.Role(this, 'SNSStatusUpdaterRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'IAM role for SNS Status Updater Lambda function',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')
      ]
    });

    // Add DynamoDB permissions for SNS updater
    this.snsUpdaterRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:UpdateItem',
        'dynamodb:Query',
        'dynamodb:BatchGetItem'
      ],
      resources: [
        this.dynamoDbTable.tableArn,
        `${this.dynamoDbTable.tableArn}/index/*` // Allow access to all GSIs
      ]
    }));

    // Add S3 permissions - restrict to SageMaker bucket pattern for improved security
    this.snsUpdaterRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:ListBucket',
        's3:GetObject'
      ],
      resources: [
        `arn:aws:s3:::sagemaker-${this.region}-${this.account}`,
        `arn:aws:s3:::sagemaker-${this.region}-${this.account}/*`
      ]
    }));

    // Add Bedrock permissions for AI/ML processing in status updates
    this.snsUpdaterRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream'
      ],
      resources: [
        'arn:aws:bedrock:*::foundation-model/*' // Allow access to all foundation models
      ]
    }));

    // Add CDK-nag suppressions for SNS Updater Role
    NagSuppressions.addResourceSuppressions(
      this.snsUpdaterRole,
      [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'AWSLambdaBasicExecutionRole is required for Lambda functions to write logs to CloudWatch',
          appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Lambda needs access to all DynamoDB GSI indexes for querying',
          appliesTo: [`Resource::<FileProcessingStatusTable3F2FBEB5.Arn>/index/*`]
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'S3 object access is restricted to SageMaker bucket pattern for security',
          appliesTo: [
            `Resource::arn:aws:s3:::sagemaker-<AWS::Region>-<AWS::AccountId>/*`,
            `Resource::arn:aws:s3:::sagemaker-${this.region}-${this.account}/*`
          ]
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Bedrock access to foundation models is required for AI processing',
          appliesTo: ['Resource::arn:aws:bedrock:*::foundation-model/*']
        }
      ],
      true // Apply to child constructs (DefaultPolicy)
    );

    // Create environment variables for SNS Status Updater
    const snsEnvironmentVariables = {
      DYNAMODB_TABLE_NAME: this.dynamoDbTable.tableName,
      SNS_TOPIC_ARN: this.snsStatusTopic.topicArn,
      LOG_LEVEL: lambdaConfig.logLevel!
    };

    // Create SNS Status Updater Lambda function
    this.snsStatusUpdaterFunction = this.createLambdaFunction(
      'SNSStatusUpdater',
      'lambda/sns-status-updater',
      this.snsUpdaterRole,
      snsEnvironmentVariables,
      'Lambda function to process SNS status updates from SageMaker'
    );

    // Grant SNS topic permission to invoke the status updater function
    this.snsStatusTopic.addSubscription(
      new snsSubscriptions.LambdaSubscription(this.snsStatusUpdaterFunction)
    );

    // Add subscription to external failed inference topic
    failedInferenceTopic.addSubscription(
      new snsSubscriptions.LambdaSubscription(this.snsStatusUpdaterFunction)
    );

    // Stack outputs
    new cdk.CfnOutput(this, 'DynamoDbTableName', {
      value: this.dynamoDbTable.tableName,
      description: 'DynamoDB table name for file processing status'
    });

    new cdk.CfnOutput(this, 'S3ProcessorFunctionArn', {
      value: this.s3ProcessorFunction.functionArn,
      description: 'Lambda function ARN for S3 SageMaker processor'
    });

    new cdk.CfnOutput(this, 'S3ProcessorFunctionName', {
      value: this.s3ProcessorFunction.functionName,
      description: 'Lambda function name for S3 SageMaker processor'
    });

    new cdk.CfnOutput(this, 'SNSStatusUpdaterFunctionArn', {
      value: this.snsStatusUpdaterFunction.functionArn,
      description: 'Lambda function ARN for SNS status updater'
    });

    new cdk.CfnOutput(this, 'SNSTopicArn', {
      value: this.snsStatusTopic.topicArn,
      description: 'SNS topic ARN for SageMaker status updates'
    });

    // Output GSI name if enabled
    if (dynamoConfig.enableInferenceIdIndex) {
      new cdk.CfnOutput(this, 'InferenceIdIndexName', {
        value: 'InferenceIdIndex',
        description: 'DynamoDB Global Secondary Index name for inference_id lookups'
      });
    }
  }

  /**
   * Helper method to create Lambda functions with shared libraries
   */
    private createLambdaFunction(
    functionId: string,
    codePath: string,
    role: iam.Role,
    environment: { [key: string]: string },
    description: string
  ): lambda.Function {
    return new lambda.Function(this, functionId, {
      runtime: lambda.Runtime.PYTHON_3_13,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromAsset('.', {
        exclude: [
          'node_modules',
          'cdk.out',
          'dist',
          'tests',
          '.git',
          '.venv',
          '*.md',
          'tsconfig.json',
          'package.json',
          'package-lock.json',
          'jest.config.js',
          'cdk.json'
        ],
        bundling: {
          image: lambda.Runtime.PYTHON_3_9.bundlingImage,
          command: [
            'bash', '-c',
            [
              `cd ${codePath}`,
              'pip install -r requirements.txt -t /asset-output || true',
              'cp -r . /asset-output',
              // Copy shared libraries from the project root
              'cp -r ../shared /asset-output/shared'
            ].join(' && ')
          ],
        },
      }),
      role: role,
      timeout: cdk.Duration.minutes(15),
      memorySize: 512,
      environment: environment,
      description: description,
      functionName: `${this.stackName}-${functionId.toLowerCase()}`
    });
  }

  /**
   * Helper method to create environment variables for Lambda function
   */
  private createLambdaEnvironmentVariables(
    sageMakerEndpoint?: string,
    logLevel: string = 'INFO'
  ): { [key: string]: string } {
    const envVars: { [key: string]: string } = {
      DYNAMODB_TABLE_NAME: this.dynamoDbTable.tableName,
      LOG_LEVEL: logLevel
    };

    if (sageMakerEndpoint) {
      envVars.SAGEMAKER_ENDPOINT_NAME = sageMakerEndpoint;
    }

    return envVars;
  }
}
