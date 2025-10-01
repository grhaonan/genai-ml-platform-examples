import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import * as AwsBlogSagemaker from '../lib/aws-blog-sagemaker-stack';
import { S3SageMakerProcessorStackProps } from '../lib/types';

describe('AwsBlogSagemakerStack', () => {
    let app: cdk.App;
    let stack: AwsBlogSagemaker.AwsBlogSagemakerStack;
    let template: Template;

    beforeEach(() => {
        app = new cdk.App();
    });

    test('Stack can be created with default configuration', () => {
        // WHEN
        stack = new AwsBlogSagemaker.AwsBlogSagemakerStack(app, 'TestStack');
        template = Template.fromStack(stack);

        // THEN
        expect(stack).toBeDefined();
        expect(template).toBeDefined();
    });

    test('Stack can be created with custom configuration', () => {
        // GIVEN
        const props: S3SageMakerProcessorStackProps = {
            lambdaConfig: {
                timeoutMinutes: 10,
                memorySize: 256,
                logLevel: 'DEBUG'
            },
            dynamoDbConfig: {
                tableName: 'custom-table-name',
                pointInTimeRecovery: true
            },
            sageMakerConfig: {
                endpointName: 'test-endpoint',
                enableSageMakerAccess: true
            }
        };

        // WHEN
        stack = new AwsBlogSagemaker.AwsBlogSagemakerStack(app, 'TestStackWithProps', props);
        template = Template.fromStack(stack);

        // THEN
        expect(stack).toBeDefined();
        expect(template).toBeDefined();
    });

    // Additional tests will be added as resources are implemented in subsequent tasks
    test('Stack structure is properly initialized', () => {
        // WHEN
        stack = new AwsBlogSagemaker.AwsBlogSagemakerStack(app, 'TestStack');

        // THEN
        expect(stack.dynamoDbTable).toBeDefined();
        expect(stack.s3ProcessorFunction).toBeDefined();
        expect(stack.snsStatusUpdaterFunction).toBeDefined();
        expect(stack.s3ProcessorRole).toBeDefined();
        expect(stack.snsUpdaterRole).toBeDefined();
        expect(stack.snsStatusTopic).toBeDefined();
    });
});