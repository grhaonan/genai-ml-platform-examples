#!/usr/bin/env python3
"""
Test script to verify Streamlit SageMaker Migration Advisor setup
"""

import sys
import os
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'streamlit',
        'strands',
        'strands_tools', 
        'boto3',
        'PIL',
        'pandas',
        'numpy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_files():
    """Test if all required files exist"""
    print("\n📁 Testing required files...")
    
    required_files = [
        'sagemaker_migration_advisor.py',
        'prompts.py',
        'logger_config.py',
        'tools/__init__.py',
        'tools/user_prompt.py',
        'advisor_config.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_aws_config():
    """Test AWS configuration"""
    print("\n🔐 Testing AWS configuration...")
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Try to get credentials from any source (env vars, profile, SSO, etc.)
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            print("  ✅ AWS credentials found and accessible")
            
            # Check region
            region = session.region_name
            if region:
                print(f"  ✅ AWS region: {region}")
            else:
                print("  ⚠️  No default region set, will use us-west-2")
            
            # Check credential source
            aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            aws_profile = os.environ.get('AWS_PROFILE')
            
            if aws_access_key:
                print("  ℹ️  Using environment variable credentials")
            elif aws_profile:
                print(f"  ℹ️  Using AWS profile: {aws_profile}")
            else:
                print("  ℹ️  Using AWS SSO or other credential source")
            
            return True
        else:
            print("  ❌ No AWS credentials found")
            print("     Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            print("     Or configure AWS CLI with 'aws configure'")
            return False
            
    except Exception as e:
        print(f"  ❌ Error checking AWS credentials: {e}")
        return False

def test_bedrock_access():
    """Test Bedrock model access"""
    print("\n🤖 Testing Bedrock access...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        # Try to create a Bedrock client
        bedrock = boto3.client('bedrock', region_name='us-west-2')
        
        # Try to list foundation models (this requires minimal permissions)
        try:
            response = bedrock.list_foundation_models()
            print("  ✅ Bedrock access confirmed")
            
            # Check for Claude models
            claude_models = [
                model for model in response.get('modelSummaries', [])
                if 'claude' in model.get('modelId', '').lower()
            ]
            
            if claude_models:
                print(f"  ✅ Found {len(claude_models)} Claude models")
            else:
                print("  ⚠️  No Claude models found - check model access")
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                print("  ❌ Access denied to Bedrock")
                print("     Check your AWS permissions for Bedrock")
            else:
                print(f"  ❌ Bedrock error: {error_code}")
            return False
            
    except NoCredentialsError:
        print("  ❌ No AWS credentials available")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_advisor_config():
    """Test advisor configuration"""
    print("\n⚙️  Testing advisor configuration...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit imported successfully")
        
        # Test if we can import our config
        try:
            import advisor_config
            print("  ✅ Advisor config loaded")
        except ImportError:
            print("  ⚠️  advisor_config.py not found (optional)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit configuration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 SageMaker Migration Advisor - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Test", test_files),
        ("AWS Config Test", test_aws_config),
        ("Bedrock Access Test", test_bedrock_access),
        ("Advisor Config Test", test_advisor_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run the Streamlit app.")
        print("   Run: python run_sagemaker_migration_advisor.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("   Check the installation guide in README.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)