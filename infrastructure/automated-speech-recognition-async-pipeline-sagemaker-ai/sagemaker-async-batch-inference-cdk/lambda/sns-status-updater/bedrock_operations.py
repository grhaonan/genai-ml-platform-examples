"""
AWS Bedrock Operations for content summarization using Claude Sonnet 3.7.

This module provides functionality to interact with AWS Bedrock API for
AI-powered content summarization of SageMaker inference results.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import boto3
from botocore.exceptions import ClientError, BotoCoreError


class BedrockOperations:
    """
    Handles AWS Bedrock API operations for content summarization.
    
    This class provides methods to initialize Bedrock runtime client and
    generate intelligent summaries using Claude Sonnet 3.7 model.
    """
    
    def __init__(self, region_name: str = None):
        """
        Initialize BedrockOperations with specified region.
        
        Args:
            region_name (str): AWS region for Bedrock service (defaults to environment variable)
        """
        self.region_name = region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        self.logger = logging.getLogger(__name__)
        self.bedrock_client = None
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        
    def initialize_client(self) -> Tuple[bool, str]:
        """
        Initialize Bedrock runtime client with appropriate region configuration.
        
        Returns:
            Tuple[bool, str]: (success_status, error_message)
        """
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.region_name
            )
            self.logger.info(f"Bedrock client initialized successfully in region {self.region_name}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Failed to initialize Bedrock client: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def generate_summary(self, content: str) -> Tuple[bool, str, str]:
        """
        Generate intelligent summary of inference output using Bedrock API.
        
        Args:
            content (str): Original content to summarize
            
        Returns:
            Tuple[bool, str, str]: (success_status, summary_content, error_message)
        """
        if not self.bedrock_client:
            init_success, init_error = self.initialize_client()
            if not init_success:
                return False, "", init_error
        
        if not content or not content.strip():
            error_msg = "Content is empty or None"
            self.logger.warning(error_msg)
            return False, "", error_msg
            
        try:
            # Prepare the prompt for Claude Sonnet 3.7
            system_prompt = (
                "You are an AI assistant that creates concise, intelligent summaries. Focus on key insights,"
                "findings, and actionable information. Keep summaries under 500 words."
            )
            
            user_prompt = f"""
            Please analyze and summarize the following content:

            {content}

            Provide a concise summary that highlights:
            1. Key findings or results
            2. Important patterns or insights
            3. Any notable anomalies or significant data points
            4. Actionable conclusions if applicable

            Keep the summary professional and focused on the most important aspects.
            """
            
            # Prepare the request body for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            # Make the API call to Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and len(response_body['content']) > 0:
                summary = response_body['content'][0]['text']
                self.logger.info("Successfully generated content summary using Bedrock API")
                return True, summary, ""
            else:
                error_msg = "No content returned from Bedrock API"
                self.logger.error(error_msg)
                return False, "", error_msg
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = f"Bedrock API ClientError ({error_code}): {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, "", error_msg
            
        except BotoCoreError as e:
            error_msg = f"Bedrock API BotoCoreError: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, "", error_msg
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Bedrock API response: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, "", error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during Bedrock API processing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, "", error_msg
    
    def process_inference_output(self, inference_content: str) -> Tuple[bool, str, str]:
        """
        Process SageMaker inference output and generate summary.
        
        This method handles the complete workflow of processing inference results
        and generating intelligent summaries with proper error handling.
        
        Args:
            inference_content (str): Raw inference output content
            
        Returns:
            Tuple[bool, str, str]: (success_status, summary_content, error_message)
        """
        try:
            # Validate input content
            if not inference_content:
                error_msg = "Inference content is empty or None"
                self.logger.warning(error_msg)
                return False, "", error_msg
            
            # Truncate content if too long to avoid API limits
            max_content_length = 50000  # Conservative limit for Claude
            if len(inference_content) > max_content_length:
                truncated_content = inference_content[:max_content_length] + "... [content truncated]"
                self.logger.info(f"Content truncated from {len(inference_content)} to {len(truncated_content)} characters")
                inference_content = truncated_content
            
            # Generate summary using Bedrock API
            success, summary, error_msg = self.generate_summary(inference_content)
            
            if success:
                self.logger.info("Successfully processed inference output and generated summary")
                return True, summary, ""
            else:
                self.logger.error(f"Failed to generate summary: {error_msg}")
                return False, "", error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error processing inference output: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, "", error_msg