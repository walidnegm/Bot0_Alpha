# Utils classes/methods for data extraction, parsing, and manipulation

# External libraries
import os
import re
import json
import logging
from io import StringIO
from typing import Optional, Union, Dict, List, Any, cast
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, ValidationError

from openai import OpenAI
from anthropic import Anthropic
from anthropic.types import ContentBlock
import ollama

from models.llm_response_base_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
)
from models.thought_models import (
    IdeaClusterJSONModel,
    IdeaJSONModel,
    ThoughtJSONModel,
    EvalJSONModel,
)


# from models.openai_claude_llama_response_basemodels import (
#     OpenAITextResponse,
#     OpenAIJSONResponse,
#     OpenAITabularResponse,
#     OpenAICodeResponse,
#     ClaudeTextResponse,
#     ClaudeJSONResponse,
#     ClaudeTabularResponse,
#     ClaudeCodeResponse,
#     LlamaTextResponse,
#     LlamaJSONResponse,
#     LlamaTabularResponse,
#     LlamaCodeResponse,
# )

from config import (
    GPT_35_TURBO,  # "gpt-3.5-turbo-0125"
    GPT_4,  # "gpt-4"
    GPT_4_TURBO,  # "gpt-4-0125-preview"
    CLAUDE_HAIKU,  # "claude-3-haiku-20240307"
    CLAUDE_SONNET,  # "claude-3-sonnet-20240229"
    CLAUDE_OPUS,  # "claude-3-opus-20240229"
)

# Logger setup
logger = logging.getLogger(__name__)

json_model_mapping = {
    "thought_json": ThoughtJSONModel,
    "idea_json": IdeaJSONModel,
    "cluster_json": IdeaClusterJSONModel,
    "eval_json": EvalJSONModel,
    # Additional mappings as needed
}


# Utility Functions
def clean_and_extract_json(
    response_content: Any,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extracts, cleans, and parses JSON content from the API response.
    Strips out any non-JSON content like extra text before the JSON block.
    Also removes JavaScript-style comments and trailing commas.

    Args:
        response_content (Any): The full response content as a string, potentially
        containing JSON.

    Returns:
        Optional[Union[Dict[str, Any], List[Any]]]: Parsed JSON data as a dictionary or list,
        or None if parsing fails.
    """
    try:
        # Extract JSON-like block by matching the first `{` and the last `}`
        match = re.search(r"{.*}", response_content, re.DOTALL)
        if not match:
            logger.error("No valid JSON content found in response.")
            return None

        raw_json_string = match.group(0)

        # Remove JavaScript-style single-line comments (// ...) but retain valid JSON
        cleaned_json_string = re.sub(r"\s*//[^\n]*", "", raw_json_string)

        # Remove trailing commas before closing braces or brackets
        cleaned_json_string = re.sub(r",\s*([\]}])", r"\1", cleaned_json_string)

        # Remove leading commas or commas before opening braces or brackets
        cleaned_json_string = re.sub(r"([{\[])\s*,", r"\1", cleaned_json_string)

        # Check for mismatched braces or brackets and attempt to correct
        if cleaned_json_string.count("{") != cleaned_json_string.count("}"):
            logger.warning("Mismatched braces detected. Attempting to correct.")
            # Ensure we have balanced braces by truncating any extraneous braces
            open_count = cleaned_json_string.count("{")
            close_count = cleaned_json_string.count("}")
            if open_count > close_count:
                cleaned_json_string = cleaned_json_string.rstrip("}")
            elif close_count > open_count:
                cleaned_json_string = "{" + cleaned_json_string.lstrip("{")

        # Parse JSON string into a Python object
        return json.loads(cleaned_json_string)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return None


def get_claude_api_key() -> str:
    """Retrieves the Claude API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        logger.error("Claude API key not found. Please set it in the .env file.")
        raise EnvironmentError("Claude API key not found.")
    return api_key


def get_openai_api_key() -> str:
    """Retrieves the OpenAI API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set it in the .env file.")
        raise EnvironmentError("OpenAI API key not found.")
    return api_key


# Validation Functions


def validate_json_response(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], json_type: str
) -> Union[
    EvalJSONModel, IdeaClusterJSONModel, IdeaJSONModel, ThoughtJSONModel, JSONResponse
]:
    """
    Validates JSON data against a specific Pydantic model based on `json_type`.

    Args:
        - data (Union[Dict[str, Any], List[Dict[str, Any]]]): The raw JSON data
        to validate.
        - json_type (str): The type of JSON model to use for validation.

    Returns:
        Union[EvalJSONResponse, ThoughtJSONResponse]:
        An instance of the validated Pydantic model.
    """
    # Get the specific model from the mapping
    model = json_model_mapping.get(json_type)

    if not model:
        raise ValueError(f"Unsupported json_type: {json_type}")

    # Use model-specific instantiation
    return model(**data) if model != JSONResponse else JSONResponse(data=data)


def validate_response_type(
    response_content: Union[str, Any], expected_res_type: str
) -> Union[
    JSONResponse,
    TabularResponse,
    CodeResponse,
    TextResponse,
]:
    """
    Validates the response content and returns a Pydantic model based on
    the expected response type.

    Args:
        - response_content (str): The raw response content from the LLM API.
        - expected_res_type (str): The expected type of the response
        (e.g., "str", "json", "tabular", "code").

    Returns:
        Union[CodeResponse, JSONResponse, TabularResponse, TextResponse]:
            The validated and structured response as a Pydantic model instance.
            - CodeResponse: Returned when expected_res_type is "code", wraps code content.
            - JSONResponse, JobSiteResponseModel, or EditingResponseModel:
              Returned when expected_res_type is "json", based on context_type.
            - TabularResponse: Returned when expected_res_type is "tabular", wraps a DataFrame.
            - TextResponse: Returned when expected_res_type is "str", wraps plain text content.
    """
    if expected_res_type == "json":
        # Check if response_content is a string that needs parsing
        if isinstance(response_content, str):
            # Only parse if it's a string
            cleaned_content = clean_and_extract_json(response_content)
            if cleaned_content is None:
                raise ValueError("Failed to extract valid JSON from the response.")
        else:
            # If it's already a dict or list, use it directly
            cleaned_content = response_content

        # Create a JSONResponse instance with the cleaned content
        if isinstance(cleaned_content, (dict, list)):
            return JSONResponse(data=cleaned_content)
        else:
            raise TypeError(
                f"Expected dict or list for JSON response, got {type(cleaned_content)}"
            )

    elif expected_res_type == "tabular":
        try:
            df = pd.read_csv(StringIO(response_content))
            return TabularResponse(data=df)
        except Exception as e:
            logger.error(f"Error parsing tabular data: {e}")
            raise ValueError("Response is not valid tabular data.")

    elif expected_res_type == "code":
        if isinstance(response_content, str):
            return CodeResponse(code=response_content)
        else:
            raise TypeError(
                f"Expected str for code response, got {type(response_content)}"
            )

    elif expected_res_type == "str":
        if isinstance(response_content, str):
            return TextResponse(content=response_content)
        else:
            raise TypeError(
                f"Expected str for text response, got {type(response_content)}"
            )

    else:
        raise ValueError(f"Unsupported response type: {expected_res_type}")


# API Calling Functions
def call_api(
    client: Optional[Union[OpenAI, Anthropic]],
    model_id: str,
    prompt: str,
    expected_res_type: str,
    temperature: float,
    max_tokens: int,
    llm_provider: str,
    json_type: Optional[str] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    CodeResponse,
    TextResponse,
]:
    """Unified function to handle API calls for OpenAI, Claude, and Llama."""
    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")
        if llm_provider == "openai":
            openai_client = cast(OpenAI, client)
            response = openai_client.chat.completions.create(
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
            claude_client = cast(Anthropic, client)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )
            response = claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": system_instruction + prompt}],
                temperature=temperature,
            )

            # Need to add an extra step to extract content from response object's TextBlocks
            # (Unlike GPT and LlaMA, Claude uses multi-blocks in its responses:
            # The content attribute of Message is a list of TextBlock objects,
            # whereas others wrap everything into a single block.)
            response_content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else response.content[0]
            )

            # or
            # response_content = response.content[0]

            # if isinstance(response_content, ContentBlock):
            #     response_content = response_content.text
            # else:
            #     response_content = str(response_content)

            logger.info(response_content)

        elif llm_provider == "llama3":
            options = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_size": 10,
                "retry_enabled": True,
            }
            response = ollama.generate(model=model_id, prompt=prompt, options=options)  # type: ignore
            response_content = response["response"]

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        # Validate response is not empty
        if not response_content:
            raise ValueError(f"Received an empty response from {llm_provider} API.")

        # Validate response type (generic text, JSON, tabular, and code)
        validated_response_content = validate_response_type(
            response_content, expected_res_type
        )

        # Further validate JSON responses
        if expected_res_type == "json":
            validated_response_content = validate_json_response(
                validated_response_content.data, json_type=json_type
            )

        # Log and return the validated response
        logger.info(f"Validated response content: \n{validated_response_content}")

        return validated_response_content

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


def call_openai_api(
    prompt: str,
    model_id: str = "gpt-4-turbo",
    expected_res_type: str = "str",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[OpenAI] = None,
    json_type: str = "thought_json",
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
]:
    """
    Calls OpenAI API and parses response.

    Args:
        prompt (str): The prompt to send to the API.
        client (Optional[OpenAI]): An OpenAI client instance. If None, a new client is instantiated.
        model_id (str): Model ID to use for the API call.
        expected_res_type (str): The expected type of response from the API ('str', 'json',
        'tabular', or 'code').
        temperature (float): Controls the creativity of the response.
        max_tokens (int): Maximum number of tokens for the response.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse]:
        The structured response from the API.
    """
    openai_client = client if client else OpenAI(api_key=get_openai_api_key())
    logger.info("OpenAI client ready for API call.")

    return call_api(
        client=openai_client,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="openai",
        json_type=json_type,
    )


def call_claude_api(
    prompt: str,
    model_id: str = "claude-3-5-sonnet-20241022",
    expected_res_type: str = "str",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[Anthropic] = None,
    json_type: str = "thought_json",
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
]:
    """
    Calls the Claude API to generate responses based on a given prompt and expected response type.

    Args:
        prompt (str): The prompt to send to the API.
        model_id (str): Model ID to use for the Claude API call.
        expected_res_type (str): The expected type of response from the API ('str', 'json', 'tabular', or 'code').
        temperature (float): Controls the creativity of the response.
        max_tokens (int): Maximum number of tokens for the response.
        client (Optional[Anthropic]): A Claude client instance.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse]: The structured response from the API.
    """
    claude_client = client if client else Anthropic(api_key=get_claude_api_key())
    logger.info("Claude client ready for API call.")
    return call_api(
        client=claude_client,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="claude",
        json_type=json_type,
    )


def call_llama3(
    prompt: str,
    model_id: str = "llama3",
    expected_res_type: str = "str",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    json_type: str = "thought_json",
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
]:
    """Calls the Llama 3 API and parses response."""
    return call_api(
        client=None,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="llama3",
        json_type=json_type,
    )
