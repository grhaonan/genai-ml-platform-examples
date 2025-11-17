#!/usr/bin/env python3
"""
Deploy ClickHouse Notification Lambda and create AgentCore Gateway
"""

import boto3
import json
import time
import zipfile
import os
from pathlib import Path
from dotenv import load_dotenv
from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

load_dotenv(dotenv_path=Path('.') / '.env', override=True)

REGION = os.getenv("AWS_REGION", "us-east-1")
LAMBDA_FUNCTION_NAME = 'clickhouse-notification-lambda'
FIX_ISSUE_LAMBDA_NAME = 'clickhouse-fix-issue-lambda'
SNS_TOPIC_NAME = 'clickhouse-analysis-notifications'
GATEWAY_NAME = f'ClickHouseNotificationGateway-{int(time.time())}'

def create_sns_topic():
    """Create SNS topic for notifications"""
    sns = boto3.client('sns', region_name=REGION)
    
    try:
        response = sns.create_topic(Name=SNS_TOPIC_NAME)
        topic_arn = response['TopicArn']
        print(f"✓ Created SNS topic: {topic_arn}")
        return topic_arn
    except Exception as e:
        # Topic might already exist
        topics = sns.list_topics()
        for topic in topics['Topics']:
            if SNS_TOPIC_NAME in topic['TopicArn']:
                print(f"✓ Using existing SNS topic: {topic['TopicArn']}")
                return topic['TopicArn']
        raise e

def create_lambda_zip():
    """Create ZIP file for Lambda deployment"""
    zip_path = 'clickhouse_notification_lambda.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('clickhouse_notification_lambda.py', 'clickhouse_notification_lambda.py')
    
    print(f"✓ Created Lambda ZIP: {zip_path}")
    
    # Create fix_issue Lambda ZIP
    fix_zip_path = 'fix_issue_lambda.zip'
    with zipfile.ZipFile(fix_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('fix_issue_lambda.py', 'fix_issue_lambda.py')
    
    print(f"✓ Created fix_issue Lambda ZIP: {fix_zip_path}")
    
    return zip_path, fix_zip_path

def create_lambda_role(topic_arn):
    """Create IAM role for Lambda function"""
    iam = boto3.client('iam', region_name=REGION)
    role_name = f'{LAMBDA_FUNCTION_NAME}-role'
    
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": "arn:aws:logs:*:*:*"
            },
            {
                "Effect": "Allow",
                "Action": ["sns:Publish"],
                "Resource": topic_arn
            }
        ]
    }
    
    try:
        role_response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description='IAM role for ClickHouse Notification Lambda'
        )
        
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='ClickHouseNotificationPolicy',
            PolicyDocument=json.dumps(policy_document)
        )
        
        print(f"✓ Created IAM role: {role_name}")
        return role_response['Role']['Arn']
        
    except iam.exceptions.EntityAlreadyExistsException:
        role_response = iam.get_role(RoleName=role_name)
        print(f"✓ Using existing IAM role: {role_name}")
        return role_response['Role']['Arn']

def deploy_lambda_function(role_arn, zip_path, topic_arn):
    """Deploy Lambda function"""
    lambda_client = boto3.client('lambda', region_name=REGION)
    
    with open(zip_path, 'rb') as zip_file:
        zip_content = zip_file.read()
    
    try:
        response = lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime='python3.12',
            Role=role_arn,
            Handler='clickhouse_notification_lambda.lambda_handler',
            Code={'ZipFile': zip_content},
            Description='ClickHouse analysis notification tool',
            Timeout=30,
            MemorySize=256,
            Environment={'Variables': {'SNS_TOPIC_ARN': topic_arn}}
        )
        
        print(f"✓ Created Lambda function: {LAMBDA_FUNCTION_NAME}")
        return response['FunctionArn']
        
    except lambda_client.exceptions.ResourceConflictException:
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME,
            ZipFile=zip_content
        )
        
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Environment={'Variables': {'SNS_TOPIC_ARN': topic_arn}}
        )
        
        response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        print(f"✓ Updated existing Lambda function: {LAMBDA_FUNCTION_NAME}")
        return response['Configuration']['FunctionArn']

def deploy_fix_issue_lambda(role_arn, zip_path):
    """Deploy fix_issue Lambda function"""
    lambda_client = boto3.client('lambda', region_name=REGION)
    
    with open(zip_path, 'rb') as zip_file:
        zip_content = zip_file.read()
    
    try:
        response = lambda_client.create_function(
            FunctionName=FIX_ISSUE_LAMBDA_NAME,
            Runtime='python3.12',
            Role=role_arn,
            Handler='fix_issue_lambda.lambda_handler',
            Code={'ZipFile': zip_content},
            Description='Fix issues tool - demo only',
            Timeout=30,
            MemorySize=256
        )
        
        print(f"✓ Created Lambda function: {FIX_ISSUE_LAMBDA_NAME}")
        return response['FunctionArn']
        
    except lambda_client.exceptions.ResourceConflictException:
        lambda_client.update_function_code(
            FunctionName=FIX_ISSUE_LAMBDA_NAME,
            ZipFile=zip_content
        )
        
        response = lambda_client.get_function(FunctionName=FIX_ISSUE_LAMBDA_NAME)
        print(f"✓ Updated existing Lambda function: {FIX_ISSUE_LAMBDA_NAME}")
        return response['Configuration']['FunctionArn']

def create_agentcore_gateway(lambda_arn, fix_lambda_arn):
    """Create AgentCore Gateway using Starter Toolkit"""
    print("\n🌐 Creating AgentCore Gateway...")
    
    client = GatewayClient(region_name=REGION)
    
    # Create OAuth authorization server
    print("🔒 Creating OAuth authorization server...")
    cognito_response = client.create_oauth_authorizer_with_cognito(GATEWAY_NAME)
    print("✓ Created Cognito authorizer")
    
    # Create the gateway
    print("🌉 Creating Gateway...")
    gateway = client.create_mcp_gateway(
        name=GATEWAY_NAME,
        role_arn=None,
        authorizer_config=cognito_response["authorizer_config"],
        enable_semantic_search=True,
    )
    print(f"✓ Created Gateway: {gateway['gatewayId']}")
    
    # Create Lambda target with notification tool
    print("🛠️ Creating Lambda target...")
    
    # Add Lambda permission for Gateway to invoke it
    lambda_client = boto3.client('lambda', region_name=REGION)
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId='AllowAgentCoreGatewayInvoke',
            Action='lambda:InvokeFunction',
            Principal='bedrock-agentcore.amazonaws.com',
            SourceArn=gateway.get('gatewayArn', f"arn:aws:bedrock-agentcore:{REGION}:*:gateway/*")
        )
        print("✓ Added Lambda invoke permission for Gateway")
    except lambda_client.exceptions.ResourceConflictException:
        print("✓ Lambda permission already exists")
    
    lambda_target = client.create_mcp_gateway_target(
        gateway=gateway,
        name="ClickHouseNotificationTarget",
        target_type="lambda",
        target_payload={
            "lambdaArn": lambda_arn,
            "toolSchema": {
                "inlinePayload": [
                    {
                        "name": "send_analysis_summary",
                        "description": "Send ClickHouse analysis summary to SNS topic for notifications",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "The analysis summary text to send"
                                },
                                "table_name": {
                                    "type": "string",
                                    "description": "Name of the table analyzed"
                                },
                                "analysis_type": {
                                    "type": "string",
                                    "description": "Type of analysis performed"
                                }
                            },
                            "required": ["summary"]
                        }
                    }
                ]
            }
        },
        credentials=None,  # Uses GATEWAY_IAM_ROLE by default
    )
    print(f"✓ Created notification target: {lambda_target['targetId']}")
    
    # Add Lambda invoke permission for fix_issue Lambda
    try:
        lambda_client.add_permission(
            FunctionName=fix_lambda_arn,
            StatementId=f'AllowGatewayInvoke-{int(time.time())}',
            Action='lambda:InvokeFunction',
            Principal='bedrock-agentcore.amazonaws.com',
            SourceArn=gateway.get('gatewayArn', f"arn:aws:bedrock-agentcore:{REGION}:*:gateway/*")
        )
        print("✓ Added Lambda invoke permission for fix_issue Lambda")
    except lambda_client.exceptions.ResourceConflictException:
        print("✓ Fix_issue Lambda permission already exists")
    
    # Create fix_issue target
    fix_target = client.create_mcp_gateway_target(
        gateway=gateway,
        name="ClickHouseFixIssueTarget",
        target_type="lambda",
        target_payload={
            "lambdaArn": fix_lambda_arn,
            "toolSchema": {
                "inlinePayload": [
                    {
                        "name": "fix_issue",
                        "description": "Attempt to fix identified issues in the database or system",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "issue_description": {
                                    "type": "string",
                                    "description": "Description of the issue to fix"
                                },
                                "table_name": {
                                    "type": "string",
                                    "description": "Name of the table with the issue"
                                }
                            },
                            "required": []
                        }
                    }
                ]
            }
        },
        credentials=None,
    )
    print(f"✓ Created fix_issue target: {fix_target['targetId']}")
    
    # Get access token
    print("🎫 Getting access token...")
    access_token = client.get_access_token_for_cognito(cognito_response["client_info"])
    print("✓ Access token obtained")
    
    return {
        'gateway_id': gateway["gatewayId"],
        'gateway_url': gateway["gatewayUrl"],
        'gateway_arn': gateway.get("gatewayArn", ""),
        'target_id': lambda_target["targetId"],
        'fix_target_id': fix_target["targetId"],
        'cognito_response': cognito_response,
        'access_token': access_token
    }

def main():
    """Main deployment function"""
    print("🚀 Deploying ClickHouse Notification Gateway")
    print("=" * 60)
    
    # Step 1: Create SNS topic
    print("\n📧 Creating SNS topic...")
    topic_arn = create_sns_topic()
    
    # Step 2: Create Lambda ZIP
    zip_path, fix_zip_path = create_lambda_zip()
    
    # Step 3: Create Lambda role and function
    print("\n📦 Creating Lambda functions...")
    lambda_role_arn = create_lambda_role(topic_arn)
    time.sleep(10)
    lambda_arn = deploy_lambda_function(lambda_role_arn, zip_path, topic_arn)
    fix_lambda_arn = deploy_fix_issue_lambda(lambda_role_arn, fix_zip_path)
    
    # Step 4: Create Gateway
    print("\n🌐 Creating AgentCore Gateway...")
    gateway_config = create_agentcore_gateway(lambda_arn, fix_lambda_arn)
    
    # Step 5: Output configuration
    print("\n" + "=" * 60)
    print("🎉 Deployment Complete!")
    print("=" * 60)
    
    config = {
        "sns_topic_arn": topic_arn,
        "lambda_function_arn": lambda_arn,
        "fix_lambda_function_arn": fix_lambda_arn,
        "gateway_id": gateway_config['gateway_id'],
        "gateway_url": gateway_config['gateway_url'],
        "gateway_arn": gateway_config['gateway_arn'],
        "target_id": gateway_config['target_id'],
        "fix_target_id": gateway_config['fix_target_id'],
        "cognito_user_pool_id": gateway_config['cognito_response']['client_info']['user_pool_id'],
        "cognito_client_id": gateway_config['cognito_response']['client_info']['client_id'],
        "cognito_client_secret": gateway_config['cognito_response']['client_info']['client_secret'],
        "discovery_url": gateway_config['cognito_response']['client_info'].get('discovery_url') or 
                        f"https://cognito-idp.{REGION}.amazonaws.com/{gateway_config['cognito_response']['client_info']['user_pool_id']}/.well-known/openid-configuration",
        "access_token": gateway_config['access_token']
    }
    
    # Save configuration
    with open('clickhouse_gateway_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"SNS Topic ARN: {topic_arn}")
    print(f"Lambda Function ARN: {lambda_arn}")
    print(f"Gateway URL: {gateway_config['gateway_url']}")
    print(f"Gateway ID: {gateway_config['gateway_id']}")
    print(f"\n📄 Configuration saved to: clickhouse_gateway_config.json")
    
    # Cleanup
    os.remove(zip_path)
    
    # Step 6: Test Bedrock invocation
    print("\n🤖 Testing Bedrock Claude Sonnet 4.5...")
    bedrock = boto3.client('bedrock-runtime', region_name=REGION)
    
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say hello in one sentence"}]
        })
    )
    
    result = json.loads(response['body'].read())
    #print(f"✓ Bedrock response: {result['content'][0]['text']}")
    
    return config

if __name__ == "__main__":
    main()
