"""
TBA
"""

# Dependencies
from typing import Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)  # n error handling and retry strategy library that makes API calls more robust.
import logging
import logging_config

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from models.llm_response_models import TextResponse
from utils.llm_api_utils_async import (
    call_claude_api_async,
    call_openai_api_async,
    call_llama3_async,
)
from prompts.evaluation_prompt_templates import (
    INITIAL_QUESTION_GENERATION_PROMPT,
    FOLLOWUP_QUESTION_GENERATION_PROMPT,
)


logger = logging.getLogger(__name__)


class QuestionGeneratorAsync:
    """
    Utility class for generating intelligent and contextual questions using an LLM.

    Supports generating initial and follow-up questions with configurable parameters
    and built-in error handling.
    """

    def __init__(
        self,
        llm_provider: str,
        model_id,
        llm_client: Optional[Union[AsyncOpenAI, AsyncAnthropic]],
        temperature: float,
        max_tokens: int,
    ):
        self.llm_provider = llm_provider
        self.llm_client = llm_client
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def complexity(complexity: str) -> str:
        """
        Map complexity level to descriptive text.
        """
        complexity_map = {
            "simple": "straightforward and easy to understand",
            "moderate": "thought-provoking and requiring some reflection",
            "advanced": "complex and requiring deep analytical thinking",
        }
        return complexity_map.get(complexity, complexity_map["moderate"])

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_initial_question(
        self,
        topic_name: str,
        context_text: Optional[str] = None,
        complexity: str = "moderate",
    ) -> str:
        """
        Generate an initial discussion question about a topic.

        Args:
            topic_name: Primary subject of the question.
            context: Optional detailed context.
            complexity: Difficulty level of the question ('simple', 'moderate', 'advanced').

        Returns:
            A generated open-ended question as a string.
        """
        complexity_level = self.complexity(complexity)
        context_prompt = (
            f"Context:\n{context_text}"
            if context_text
            else "No additional context provided."
        )

        prompt = INITIAL_QUESTION_GENERATION_PROMPT.format(
            complexity_level=complexity_level,
            topic_name=topic_name,
            context=context_prompt,
        )

        logger.info(f"Initial question generation prompt: {prompt}")  # Debugging

        return await self.call_llm_async(prompt)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_followup_question(
        self, evaluation_text: str, context_text: Optional[str] = None
    ) -> TextResponse:
        """
        Generate a strategic follow-up question based on previous evaluation.

        Args:
            evaluation_text: Context or previous evaluation to base follow-up on.
            context_text: Optional context to incorporate into the question.

        Returns:
            A generated follow-up question as a TextResponse object.
        """
        context_prompt = (
            f"Context:\n{context_text}"
            if context_text
            else "No additional context provided."
        )

        prompt = FOLLOWUP_QUESTION_GENERATION_PROMPT.format(
            evaluation=evaluation_text, context=context_prompt
        )

        logger.info(f"Followup question generation prompt: {prompt}")
        return await self.call_llm_async(prompt)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def call_llm_async(
        self,
        prompt: str,
        expected_res_type: str = "str",
    ) -> TextResponse:
        """
        Call the LLM API asynchronously.

        Args:
            prompt: The input prompt for the LLM.
            expected_res_type: Expected response type, default is 'str'.

        Returns:
            A TextResponse object containing the LLM response.
        """
        try:
            if self.llm_provider == "openai":
                response_model = await call_openai_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.llm_client,
                )
            elif self.llm_provider == "claude":
                response_model = await call_claude_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.llm_client,
                )
            elif self.llm_provider == "llama3":
                response_model = await call_llama3_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

            logger.info(f"Response data type: {type(response_model)}\n{response_model}")
            return response_model

        except Exception as e:
            logger.error(f"Error calling LLM '{self.llm_provider}': {e}")
            raise
