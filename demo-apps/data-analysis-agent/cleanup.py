#!/usr/bin/env python3
"""
Cleanup script to delete AgentCore, Gateway, Lambdas, and SNS topic
"""

import boto3
import json
import os
import time
from pathlib import Path

REGION = os.getenv("AWS_REGION", "us-east-1")
CONFIG_FILE = 'clickhouse_gateway_config.json'

def load_config():
    """Load configuration from file"""
    if not Path(CONFIG_FILE).exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        return None
    
    with open(CONFIG_FILE) as f:
        return json.load(f)

def delete_agentcore_runtime():
    """Delete AgentCore runtime"""
    try:
        agentcore_client = boto3.client('bedrock-agentcore-control', region_name=REGION)
        
        response = agentcore_client.list_agent_runtimes()
        runtimes = response.get('agentRuntimes', [])
        
        for runtime in runtimes:
            runtime_id = runtime['agentRuntimeId']
            print(f"🗑️  Deleting AgentCore runtime: {runtime_id}")
            agentcore_client.delete_agent_runtime(agentRuntimeId=runtime_id)
            print(f"✓ Deleted runtime: {runtime_id}")
        
        if not runtimes:
            print("ℹ️  No AgentCore runtimes found")
            
    except Exception as e:
        print(f"⚠️  Error deleting AgentCore runtime: {e}")

def delete_gateway(config):
    """Delete Gateway and its targets"""
    try:
        agentcore_client = boto3.client('bedrock-agentcore-control', region_name=REGION)
        gateway_id = config['gateway_id']
        
        print(f"🗑️  Deleting Gateway targets first...")
        
        # Delete targets first
        try:
            response = agentcore_client.list_gateway_targets(gatewayIdentifier=gateway_id)
            targets = response.get('targets', [])
            
            for target in targets:
                target_id = target['targetId']
                print(f"  Deleting target: {target_id}")
                agentcore_client.delete_gateway_target(
                    gatewayIdentifier=gateway_id,
                    targetIdentifier=target_id
                )
                print(f"  ✓ Deleted target: {target_id}")
        except Exception as e:
            print(f"  ⚠️  Error deleting targets: {e}")
        
        # Delete gateway
        print(f"🗑️  Deleting Gateway: {gateway_id}")
        agentcore_client.delete_gateway(gatewayIdentifier=gateway_id)
        print(f"✓ Deleted Gateway: {gateway_id}")
        
    except Exception as e:
        print(f"⚠️  Error deleting Gateway: {e}")

def delete_lambda_function(function_name):
    """Delete Lambda function"""
    try:
        lambda_client = boto3.client('lambda', region_name=REGION)
        
        print(f"🗑️  Deleting Lambda: {function_name}")
        lambda_client.delete_function(FunctionName=function_name)
        print(f"✓ Deleted Lambda: {function_name}")
        
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"ℹ️  Lambda not found: {function_name}")
    except Exception as e:
        print(f"⚠️  Error deleting Lambda {function_name}: {e}")

def delete_sns_topic(config):
    """Delete SNS topic"""
    try:
        sns_client = boto3.client('sns', region_name=REGION)
        topic_arn = config['sns_topic_arn']
        
        print(f"🗑️  Deleting SNS topic: {topic_arn}")
        sns_client.delete_topic(TopicArn=topic_arn)
        print(f"✓ Deleted SNS topic")
        
    except Exception as e:
        print(f"⚠️  Error deleting SNS topic: {e}")

def delete_cognito_resources(config):
    """Delete Cognito user pool and client"""
    try:
        cognito_client = boto3.client('cognito-idp', region_name=REGION)
        user_pool_id = config['cognito_user_pool_id']
        
        # Delete domain first
        try:
            print(f"🗑️  Deleting Cognito domain...")
            response = cognito_client.describe_user_pool(UserPoolId=user_pool_id)
            domain = response['UserPool'].get('Domain')
            if domain:
                cognito_client.delete_user_pool_domain(
                    Domain=domain,
                    UserPoolId=user_pool_id
                )
                print(f"  ✓ Deleted domain: {domain}")
        except Exception as e:
            print(f"  ⚠️  Error deleting domain: {e}")
        
        print(f"🗑️  Deleting Cognito user pool: {user_pool_id}")
        cognito_client.delete_user_pool(UserPoolId=user_pool_id)
        print(f"✓ Deleted Cognito user pool")
        
    except Exception as e:
        print(f"⚠️  Error deleting Cognito resources: {e}")

def delete_iam_role():
    """Delete IAM role"""
    try:
        iam_client = boto3.client('iam', region_name=REGION)
        role_name = 'clickhouse-notification-lambda-role'
        
        print(f"🗑️  Deleting IAM role: {role_name}")
        
        # Detach policies first
        try:
            response = iam_client.list_attached_role_policies(RoleName=role_name)
            for policy in response['AttachedPolicies']:
                iam_client.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy['PolicyArn']
                )
        except:
            pass
        
        # Delete inline policies
        try:
            response = iam_client.list_role_policies(RoleName=role_name)
            for policy_name in response['PolicyNames']:
                iam_client.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name
                )
        except:
            pass
        
        # Delete role
        iam_client.delete_role(RoleName=role_name)
        print(f"✓ Deleted IAM role: {role_name}")
        
    except iam_client.exceptions.NoSuchEntityException:
        print(f"ℹ️  IAM role not found: {role_name}")
    except Exception as e:
        print(f"⚠️  Error deleting IAM role: {e}")

def delete_config_file():
    """Delete configuration file"""
    try:
        if Path(CONFIG_FILE).exists():
            os.remove(CONFIG_FILE)
            print(f"✓ Deleted config file: {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️  Error deleting config file: {e}")

def delete_all_gateways():
    """Delete all gateways with ClickHouse in name"""
    try:
        agentcore_client = boto3.client('bedrock-agentcore-control', region_name=REGION)
        
        response = agentcore_client.list_gateways()
        gateways = response.get('items', [])
        
        for gateway in gateways:
            gateway_id = gateway['gatewayId']
            gateway_name = gateway.get('name', '')
            
            if 'clickhouse' in gateway_name.lower() or 'notification' in gateway_name.lower():
                print(f"🗑️  Found Gateway: {gateway_name} ({gateway_id})")
                
                # Delete targets first
                try:
                    targets_response = agentcore_client.list_gateway_targets(gatewayIdentifier=gateway_id)
                    targets = targets_response.get('items', [])
                    
                    for target in targets:
                        target_id = target['targetId']
                        print(f"  Deleting target: {target_id}")
                        agentcore_client.delete_gateway_target(
                            gatewayIdentifier=gateway_id,
                            targetId=target_id
                        )
                        print(f"  ✓ Deleted target: {target_id}")
                    
                    if targets:
                        print(f"  Waiting for targets to be deleted...")
                        time.sleep(5)
                        
                except Exception as e:
                    print(f"  ⚠️  Error deleting targets: {e}")
                
                # Delete gateway
                agentcore_client.delete_gateway(gatewayIdentifier=gateway_id)
                print(f"✓ Deleted Gateway: {gateway_id}")
        
        if not gateways:
            print("ℹ️  No gateways found")
            
    except Exception as e:
        print(f"⚠️  Error deleting gateways: {e}")

def delete_all_sns_topics():
    """Delete SNS topics with clickhouse in name"""
    try:
        sns_client = boto3.client('sns', region_name=REGION)
        
        response = sns_client.list_topics()
        topics = response.get('Topics', [])
        
        for topic in topics:
            topic_arn = topic['TopicArn']
            if 'clickhouse' in topic_arn.lower():
                print(f"🗑️  Deleting SNS topic: {topic_arn}")
                sns_client.delete_topic(TopicArn=topic_arn)
                print(f"✓ Deleted SNS topic")
        
    except Exception as e:
        print(f"⚠️  Error deleting SNS topics: {e}")

def delete_all_cognito_pools():
    """Delete Cognito user pools created by gateway"""
    try:
        cognito_client = boto3.client('cognito-idp', region_name=REGION)
        
        response = cognito_client.list_user_pools(MaxResults=60)
        pools = response.get('UserPools', [])
        
        for pool in pools:
            pool_id = pool['Id']
            pool_name = pool['Name']
            
            if 'gateway' in pool_name.lower() or 'clickhouse' in pool_name.lower():
                print(f"🗑️  Found Cognito pool: {pool_name} ({pool_id})")
                
                # Delete domain first
                try:
                    pool_details = cognito_client.describe_user_pool(UserPoolId=pool_id)
                    domain = pool_details['UserPool'].get('Domain')
                    if domain:
                        cognito_client.delete_user_pool_domain(
                            Domain=domain,
                            UserPoolId=pool_id
                        )
                        print(f"  ✓ Deleted domain: {domain}")
                except Exception as e:
                    print(f"  ⚠️  Error deleting domain: {e}")
                
                # Delete pool
                cognito_client.delete_user_pool(UserPoolId=pool_id)
                print(f"✓ Deleted Cognito pool: {pool_id}")
        
    except Exception as e:
        print(f"⚠️  Error deleting Cognito pools: {e}")

def main():
    """Main cleanup function"""
    print("🧹 Starting cleanup of ClickHouse Gateway resources")
    print("=" * 60)
    
    config = load_config()
    
    print("\n1️⃣  Deleting AgentCore Runtime...")
    delete_agentcore_runtime()
    
    print("\n2️⃣  Deleting Gateways...")
    if config:
        delete_gateway(config)
    else:
        delete_all_gateways()
    
    print("\n3️⃣  Deleting Lambda functions...")
    delete_lambda_function('clickhouse-notification-lambda')
    delete_lambda_function('clickhouse-fix-issue-lambda')
    
    print("\n4️⃣  Deleting SNS topics...")
    if config:
        delete_sns_topic(config)
    else:
        delete_all_sns_topics()
    
    print("\n5️⃣  Deleting Cognito resources...")
    if config:
        delete_cognito_resources(config)
    else:
        delete_all_cognito_pools()
    
    print("\n6️⃣  Deleting IAM role...")
    delete_iam_role()
    
    print("\n7️⃣  Deleting config file...")
    delete_config_file()
    
    print("\n" + "=" * 60)
    print("🎉 Cleanup complete!")

if __name__ == "__main__":
    main()
