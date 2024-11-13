"""
Filename: openai_claude_llama3_basemodels.py
Module to define base Pydantic models for OpenAI, Claude, and Llama3
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError, constr, Field
import pandas as pd
import logging
import logging_config


# Set up logger
logger = logging.getLogger(__name__)


# Define Data Classes for Each Response Type by Provider


# Define the sub-concept model
class SubConcept(BaseModel):
    name: str
    description: str


# OpenAI response type classes
class OpenAITextResponse(BaseModel):
    """
    Pydantic model for OpenAI text response.

    Attributes:
        text (str): Generated text content from OpenAI.

    Example Usage:
        openai_response = OpenAITextResponse(text="Generated text from OpenAI")
    """

    text: str


class OpenAIJSONResponse(BaseModel):
    """
    Pydantic model for OpenAI JSON response.

    Attributes:
        data (Dict[str, Any]): JSON data returned by OpenAI, structured in a dictionary.

    Example Usage:
        openai_json_response = OpenAIJSONResponse(data={"key": "value"})
    """

    data: Dict[str, Any]


class OpenAITabularResponse(BaseModel):
    """
    Pydantic model for OpenAI tabular response.

    Attributes:
        rows (List[Dict[str, str]]): List of dictionaries representing rows in a table,
                                     with column names as keys.

    Methods:
        to_dataframe: Converts rows to a Pandas DataFrame for tabular data manipulation.

    Example JSON response:
        tabular_json_response = {
            "rows": [
                {"Name": "Alice", "Age": "30", "City": "New York"},
                {"Name": "Bob", "Age": "25", "City": "Los Angeles"}
            ]
        }
    """

    rows: List[Dict[str, str]]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class OpenAICodeResponse(BaseModel):
    """
    Pydantic model for OpenAI code response.

    Attributes:
        code (str): Code snippet generated by OpenAI.
        language (Optional[str]): Programming language of the code, if specified.
        explanation (Optional[str]): Optional explanation or description of the code.

    Example Usage:
        openai_code_response = OpenAICodeResponse(
            code="def add(a, b): return a + b",
            language="python",
            explanation="This function adds two numbers."
        )
    """

    code: str
    language: Optional[str] = None
    explanation: Optional[str] = None


# Claude response type classes
class ClaudeTextResponse(BaseModel):
    """
    Pydantic model for Claude text response.

    Attributes:
        content (str): Generated text content from Claude.

    Example Usage:
        claude_response = ClaudeTextResponse(content="Generated text from Claude")
    """

    content: str


class ClaudeJSONResponse(BaseModel):
    """
    Pydantic model for Claude JSON response.

    Attributes:
        data (Dict[str, Any]): JSON data returned by Claude, structured in a dictionary.

    Example Usage:
        claude_json_response = ClaudeJSONResponse(data={"key": "value"})
    """

    data: Dict[str, Any]


class ClaudeTabularResponse(BaseModel):
    """
    Pydantic model for Claude tabular response.

    Attributes:
        rows (List[Dict[str, str]]): List of dictionaries representing rows in a table,
                                     with column names as keys.

    Methods:
        to_dataframe: Converts rows to a Pandas DataFrame for tabular data manipulation.

    Example JSON response:
        tabular_json_response = {
            "rows": [
                {"Name": "Alice", "Age": "30", "City": "New York"},
                {"Name": "Bob", "Age": "25", "City": "Los Angeles"}
            ]
        }
    """

    rows: List[Dict[str, str]]

    def to_dataframe(self) -> pd.DataFrame:
        """Method to generate pandas df"""
        return pd.DataFrame(self.rows)


class ClaudeCodeResponse(BaseModel):
    """
    Pydantic model for Claude code response.

    Attributes:
        code (str): Code snippet generated by Claude.
        language (Optional[str]): Programming language of the code, if specified.
        explanation (Optional[str]): Optional explanation or description of the code.

    Example Usage:
        claude_code_response = ClaudeCodeResponse(
            code="def add(a, b): return a + b",
            language="python",
            explanation="This function adds two numbers."
        )
    """

    code: str
    language: Optional[str] = None
    explanation: Optional[str] = None


# Llama response type classes
class LlamaTextResponse(BaseModel):
    """
    Pydantic model for Llama text response.

    Attributes:
        result (str): Generated text content from Llama.

    Example Usage:
        llama_response = LlamaTextResponse(result="Generated text from Llama")
    """

    result: str


class LlamaJSONResponse(BaseModel):
    """
    Pydantic model for Llama JSON response.

    Attributes:
        data (Dict[str, Any]): JSON data returned by Llama, structured in a dictionary.

    Example Usage:
        llama_json_response = LlamaJSONResponse(data={"key": "value"})
    """

    data: Dict[str, Any]


class LlamaTabularResponse(BaseModel):
    """
    Pydantic model for Llama tabular response.

    Attributes:
        rows (List[Dict[str, str]]): List of dictionaries representing rows in a table,
                                     with column names as keys.

    Methods:
        to_dataframe: Converts rows to a Pandas DataFrame for tabular data manipulation.

    Example JSON response:
        tabular_json_response = {
            "rows": [
                {"Name": "Alice", "Age": "30", "City": "New York"},
                {"Name": "Bob", "Age": "25", "City": "Los Angeles"}
            ]
        }
    """

    rows: List[Dict[str, str]]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class LlamaCodeResponse(BaseModel):
    """
    Pydantic model for Llama code response.

    Attributes:
        code (str): Code snippet generated by Llama.
        language (Optional[str]): Programming language of the code, if specified.
        explanation (Optional[str]): Optional explanation or description of the code.

    Example Usage:
        llama_code_response = LlamaCodeResponse(
            code="def add(a, b): return a + b",
            language="python",
            explanation="This function adds two numbers."
        )
    """

    code: str
    language: Optional[str] = None
    explanation: Optional[str] = None
