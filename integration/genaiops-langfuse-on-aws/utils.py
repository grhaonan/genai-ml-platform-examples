import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
import requests
from langfuse.decorators import langfuse_context, observe
from langfuse.model import PromptClient

from config import AWS_REGION

# used to invoke the Bedrock Converse API
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

# In case the input message is not in the Bedrock Converse API format,
# for example it follow openAI format, we need to convert it to the Bedrock Converse API format.
def convert_to_bedrock_messages(
    messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Convert message to Bedrock Converse API format"""
    bedrock_messages = []

    # Extract system messages first
    system_prompts = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompts.append({"text": msg["content"]})
        else:
            # Handle user/assistant messages
            content_list = []

            # If content is already a list, process each item
            if isinstance(msg["content"], list):
                for content_item in msg["content"]:
                    if content_item["type"] == "text":
                        content_list.append({"text": content_item["text"]})
                    elif content_item["type"] == "image_url":
                        # Get image format from URL
                        if "url" not in content_item["image_url"]:
                            raise ValueError("Missing required 'url' field in image_url")
                        url = content_item["image_url"]["url"]
                        if not url:
                            raise ValueError("URL cannot be empty")
                        parsed_url = urlparse(url)
                        if not parsed_url.scheme or not parsed_url.netloc:
                            raise ValueError("Invalid URL format")
                        image_format = parsed_url.path.split(".")[-1].lower()
                        # Convert jpg to jpeg for Bedrock compatibility
                        if image_format == "jpg":
                            image_format = "jpeg"

                        # Download and encode image
                        response = requests.get(url)
                        image_bytes = response.content

                        content_list.append(
                            {
                                "image": {
                                    "format": image_format,
                                    "source": {"bytes": image_bytes},
                                }
                            }
                        )
            else:
                # If content is just text
                content_list.append({"text": msg["content"]})

            bedrock_messages.append({"role": msg["role"], "content": content_list})

    return system_prompts, bedrock_messages

# region Converse API Wrapper for Chat
@observe(as_type="generation", name="Bedrock Converse")
def converse(
    messages: List[Dict[str, Any]],
    model_id: str = "us.amazon.nova-pro-v1:0",
    prompt: Optional[PromptClient] = None,
    metadata: Dict[str, Any] = {},
    **kwargs,
) -> Optional[str]:
    # 1. extract model metadata
    kwargs_clone = kwargs.copy()
    model_parameters = {
        **kwargs_clone.pop("inferenceConfig", {}),
        **kwargs_clone.pop("additionalModelRequestFields", {}),
        **kwargs_clone.pop("guardrailConfig", {}),
    }
    langfuse_context.update_current_observation(
        input=messages,
        model=model_id,
        model_parameters=model_parameters,
        prompt=prompt,
    )

    # Convert messages to Bedrock format
    system_prompts, messages = convert_to_bedrock_messages(messages)

    # 2. model call with error handling
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompts,
            messages=messages,
            **kwargs,
        )
    except (ClientError, Exception) as e:
        error_message = f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
        langfuse_context.update_current_observation(
            level="ERROR", status_message=error_message
        )
        print(error_message)
        return

    # 3. extract response metadata
    response_text = response["output"]["message"]["content"][0]["text"]
    langfuse_context.update_current_observation(
        output=response_text,
        usage={
            "input": response["usage"]["inputTokens"],
            "output": response["usage"]["outputTokens"],
            "total": response["usage"]["totalTokens"],
        },
        metadata={
            "ResponseMetadata": response["ResponseMetadata"],
            **metadata,
        },
    )

    return response_text
# endregion

# region Converse API Wrapper for Tool Use
@observe(as_type="generation", name="Bedrock Converse Tool Use")
def converse_tool_use(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, str]],
    tool_choice: str = "auto",
    model_id: str = "us.amazon.nova-pro-v1:0",
    prompt: Optional[PromptClient] = None,
    metadata: Dict[str, Any] = {},
    **kwargs,
) -> Optional[List[Dict]]:
    # 1. extract model metadata
    kwargs_clone = kwargs.copy()
    model_parameters = {
        **kwargs_clone.pop("inferenceConfig", {}),
        **kwargs_clone.pop("additionalModelRequestFields", {}),
        **kwargs_clone.pop("guardrailConfig", {}),
    }

    langfuse_context.update_current_observation(
        input={"messages": messages, "tools": tools, "tool_choice": tool_choice},
        model=model_id,
        model_parameters=model_parameters,
        prompt=prompt,
    )

    # Convert messages to Bedrock format
    system_prompts, messages = convert_to_bedrock_messages(messages)

    # 2. Convert tools to Bedrock format
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "inputSchema": {"json": tool["function"]["parameters"]},
                }
            }
            for tool in tools
            if tool["type"] == "function"
        ]
    }

    # Add toolChoice configuration based on input
    if tool_choice != "auto":
        tool_config["toolChoice"] = {
            "any": {} if tool_choice == "any" else None,
            "auto": {} if tool_choice == "auto" else None,
            "tool": (
                {"name": tool_choice} if not tool_choice in ["any", "auto"] else None
            ),
        }

    # 3. model call with error handling
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompts,
            messages=messages,
            toolConfig=tool_config,
            **kwargs,
        )
    except (ClientError, Exception) as e:
        error_message = f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
        langfuse_context.update_current_observation(
            level="ERROR", status_message=error_message
        )
        print(error_message)
        return

    # 4. Handle tool use flow if needed
    output_message = response["output"]["message"]

    tool_calls = []
    if response["stopReason"] == "tool_use":
        for content in output_message["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                tool_calls.append(
                    {
                        "index": len(tool_calls),
                        "id": tool["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "arguments": json.dumps(tool["input"]),
                        },
                    }
                )

    # 5. Update Langfuse with response metadata
    langfuse_context.update_current_observation(
        output=tool_calls,
        usage={
            "input": response["usage"]["inputTokens"],
            "output": response["usage"]["outputTokens"],
            "total": response["usage"]["totalTokens"],
        },
        metadata={
            "ResponseMetadata": response["ResponseMetadata"],
            **metadata,
        },
    )

    return tool_calls
# endregion
