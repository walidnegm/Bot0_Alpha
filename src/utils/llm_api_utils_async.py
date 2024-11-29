"""
Filename: llm_api_utils_async.py
Last updated: 2024 Nov 1

This module provides asynchronous functions to interact with various large language models (LLMs), 
including OpenAI, Claude, and LLaMA. Each function enables API calls to these LLM providers, 
handling prompt-based requests and returning responses in specified formats 
(e.g., JSON, tabular, code, or plain text), with robust error handling and logging.
"""

import os
import asyncio
import logging
import logging_config
import httpx
import aiohttp
import json
from pydantic import ValidationError
import pandas as pd
from io import StringIO
from typing import Union, Optional, cast
from dotenv import load_dotenv

from anthropic import Anthropic, AsyncAnthropic
import openai
from openai import OpenAI, AsyncOpenAI
import ollama

# Import from internal
from utils.llm_api_utils import (
    get_openai_api_key,
    get_claude_api_key,
    clean_and_extract_json,
    validate_response_type,
    validate_json_response,
)
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
)
from models.thought_models import (
    IdeaClusterJSONModel,
    IdeaJSONModel,
    ThoughtJSONModel,
)
from models.evaluation_models import EvaluationJSONModel

# Setup logger
logger = logging.getLogger(__name__)


async def call_api_async(
    llm_provider: str,
    model_id: str,
    prompt: str,
    expected_res_type: str,
    json_type: Optional[str],
    client: Optional[Union[AsyncOpenAI, AsyncAnthropic]],
    temperature: float,
    max_tokens: int,
) -> Union[
    JSONResponse,
    TabularResponse,
    CodeResponse,
    TextResponse,
    IdeaJSONModel,
    IdeaClusterJSONModel,
    ThoughtJSONModel,
    EvaluationJSONModel,
]:
    """
    Unified function to handle async API calls for different language models like
    OpenAI, Claude, and Llama.

    Args:
        - client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]): Async client instance
        for the specific LLM provider.
        - model_id (str): Model identifier for the API call.
        - prompt (str): Prompt to be passed to the API.
        - expected_res_type (str): Expected format of the response ('json', 'tabular',
        'code', or 'str').
        - temperature (float): Degree of randomness in the model's output (0 to 1.0).
        - max_tokens (int): Maximum tokens to generate in response.
        - llm_provider (str): The LLM provider ("openai", "claude", "llama3").

    Returns:
        - Union[JSONResponse, TabularResponse, CodeResponse, TextResponse, etc.]:
        Model-specific response object based on `expected_res_type`.

    Raises:
        - ValueError: If an unsupported response type is specified or if the response
        content is empty.
        - Exception: Logs and raises any other exception encountered during API call.
    """
    try:
        logger.info(
            f"Making async API call with expected response type: {expected_res_type}"
        )

        # Make provider-specific API call
        if llm_provider == "openai":
            openai_client = cast(AsyncOpenAI, client)
            response = await openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who adheres to instructions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_content = response.choices[0].message.content

        elif llm_provider == "claude":
            claude_client = cast(AsyncAnthropic, client)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )
            response = await claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": system_instruction + prompt}],
                temperature=temperature,
            )
            # Extract content from Claude's multi-block response
            if hasattr(response, "content") and isinstance(response.content, list):
                # Concatenate text blocks into a single string
                response_content = "".join(
                    block.text if hasattr(block, "text") else str(block)
                    for block in response.content
                )
            elif hasattr(response, "content"):
                response_content = response.content
            else:
                raise ValueError(f"Unexpected response format from Claude: {response}")

            logger.info(f"Extracted Claude Response: {response_content}")

        elif llm_provider == "llama3":
            options = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_size": 10,
                "retry_enabled": True,
            }
            response = await ollama.generate(model=model_id, prompt=prompt, options=options)  # type: ignore
            response_content = response["response"]

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        # Validate response is NOT empty
        if not response_content:
            raise ValueError(f"Received an empty response from {llm_provider} API.")

        # Validate response type (json, text, code, tabular)
        validated_response_content = validate_response_type(
            response_content, expected_res_type
        )

        # For JSON responses, perform additional validation if needed
        if expected_res_type == "json" and isinstance(
            validated_response_content, JSONResponse
        ):
            # Additional JSON validation logic if necessary
            logger.debug("Performing additional JSON validation.")

            validated_response_content = validate_json_response(
                validated_response_content.data, json_type=json_type
            )

        logger.info(f"Validated response content: {validated_response_content}")
        return validated_response_content

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


async def call_openai_api_async(
    prompt: str,
    model_id: str = "gpt-4-turbo",
    expected_res_type: str = "str",  # Set to str (-> text response)
    json_type: Optional[str] = None,
    client: Optional[AsyncOpenAI] = None,
    temperature: float = 0.4,
    max_tokens: int = 1056,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    IdeaClusterJSONModel,
    IdeaJSONModel,
    ThoughtJSONModel,
    EvaluationJSONModel,
]:
    """
    Handles async API call to OpenAI to generate responses based on a given prompt and expected response type.

    Args:
        client (Optional[OpenAI]): OpenAI API client instance (optional).
        model_id (str): Model ID to use for the OpenAI API call.
        prompt (str): The prompt to send to the API.
        expected_res_type (str): Expected type of response from the API ('str', 'json', 'tabular', or 'code').
        temperature (float): Controls creativity (0 to 1.0).
        max_tokens (int): Maximum tokens for generated response.

    Returns:
        Union[JSONResponse, TabularResponse, TextResponse, CodeResponse]: Formatted response object.
    """
    openai_client = client if client else AsyncOpenAI(api_key=get_openai_api_key())

    logger.info("OpenAI client ready for async API call.")

    validated_response_content = await call_api_async(
        llm_provider="openai",
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        client=openai_client,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info("call_openai_api_async - return validated response content.")

    return validated_response_content


async def call_claude_api_async(
    prompt: str,
    model_id: str = "claude-3-5-sonnet-20241022",
    expected_res_type: str = "str",
    json_type: Optional[str] = None,
    client: Optional[AsyncAnthropic] = None,
    temperature: float = 0.4,
    max_tokens: int = 1056,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    IdeaClusterJSONModel,
    IdeaJSONModel,
    ThoughtJSONModel,
    EvaluationJSONModel,
]:
    """
    Handles async API call to Claude to generate responses based on a given prompt and expected response type.

    Args:
        client (Optional[Anthropic]): Claude API client instance (optional).
        model_id (str): Model ID to use for the Claude API call.
        prompt (str): The prompt to send to the API.
        expected_res_type (str): Expected type of response from the API ('str', 'json', 'tabular', or 'code').
        context_type (str): Specifies the context model, e.g., "editing" or "job_site".
        temperature (float): Controls creativity (0 to 1.0).
        max_tokens (int): Maximum tokens for generated response.

    Returns:
        Union[JSONResponse, TabularResponse, TextResponse, CodeResponse]: Formatted response object.
    """
    claude_client = client if client else AsyncAnthropic(api_key=get_claude_api_key())

    logger.info("Claude client ready for async API call.")

    return await call_api_async(
        llm_provider="claude",
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        client=claude_client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def call_llama3_async(
    prompt: str,
    model_id: str = "llama3",
    expected_res_type: str = "str",
    json_type: str = None,
    temperature: float = 0.4,
    max_tokens: int = 1056,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    IdeaClusterJSONModel,
    IdeaClusterJSONModel,
    ThoughtJSONModel,
    EvaluationJSONModel,
]:
    """
    Handles async API call to LLaMA 3 to generate responses based on a given prompt and expected response type.

    Args:
        model_id (str): Model ID to use for the LLaMA 3 API call.
        prompt (str): The prompt to send to the API.
        expected_res_type (str): Expected response format ('str', 'json', 'tabular', or 'code').
        json_type (str): Specifies types of JSON response (pydantic model).
        temperature (float): Controls creativity (0 to 1.0).
        max_tokens (int): Maximum tokens for generated response.

    Returns:
        Union[JSONResponse, TabularResponse, TextResponse, CodeResponse, etc.]: Formatted response object.
    """
    return await call_api_async(
        llm_provider="llama3",
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        client=None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
