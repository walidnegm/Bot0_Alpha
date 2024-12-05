"""
Evaluator Agent Module

This module defines the EvaluatorAgent class, which evaluates a user’s response 
to a question based on predefined criteria. It supports multiple LLM providers 
(OpenAI, Claude, LLaMA 3) and can either instantiate an API client or use an inherited one.

Example Usage:
    evaluator = EvaluatorAgent(api_key='your_key', llm_provider='openai', model_id='gpt-4')
    question = "What are the main differences between microcontrollers and microprocessors?"
    answer = "Microcontrollers integrate memory, I/O peripherals, and a CPU on a single chip, \
        whereas microprocessors focus on processing tasks and rely on external memory \
            and peripherals."
    evaluation, evaluation_text = await evaluator.evaluate(question, answer)
    composite_score = evaluator.calculate_composite_score(evaluation.criteria)
"""

import re
import json
import logging
from typing import Dict, Tuple, Union, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError
import aiohttp

# from LLM APIs
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# from internal modules
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TextResponse,
    TabularResponse,
)
from models.indexed_thought_models import IndexedIdeaJSONModel
from models.evaluation_models import (
    EvaluationCriteria,
    QuestionAnswerPair,
    EvaluationJSONModel,
)
from prompts.evaluation_prompt_templates import QUESTION_ANSWER_EVAL_PROMPT
from utils.llm_api_utils_async import (
    call_openai_api_async,
    call_claude_api_async,
    call_llama3_async,
    get_claude_api_key,
    get_openai_api_key,
)


# Initialize logger
logger = logging.getLogger(__name__)

# Define LLM model options for different providers
LLM_MODELS = {
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "claude": ["claude-3", "claude-3-5"],
    "llama3": ["llama3-turbo"],
}


class EvaluatorAgentAsync:
    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_provider: str = "openai",
        model_id: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 1056,
        client: Optional[
            Union[call_openai_api_async, call_claude_api_async, call_llama3_async]
        ] = None,
    ):
        """
        Initialize the EvaluatorAgentAsync.

        Args:
            - api_key (Optional[str]): The API key for the LLM provider (optional if a client
            is inherited).
            - llm_provider (str): The LLM provider to use (default is 'openai').
            - model_id (str): The model ID to use for the LLM.
            - temperature (float): Sampling temperature.
            - max_tokens (int): Maximum number of tokens in the response.
            - client (Optional[Callable]): Optionally inherit an API client for
            OpenAI, Claude, or LLaMA 3.
        """
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Instantiate the API client based on llm_provider
        if llm_provider == "openai":
            self.client = AsyncOpenAI(api_key=get_openai_api_key())
        elif llm_provider == "claude":
            self.client = AsyncAnthropic(api_key=get_claude_api_key())
        else:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")
        self.client = client

        logger.info(
            f"Evaluator Agent (Async) instantiated with provider '{llm_provider}', model '{model_id}'."
        )

    async def __aenter__(self):
        """
        Asynchronous context manager entry. Initializes the HTTP session.
        """
        self.session = aiohttp.ClientSession()
        logger.info("EvaluatorAgentAsync HTTP session started.")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Asynchronous context manager exit. Closes the HTTP session.
        """
        if self.session:
            await self.session.close()
            logger.info("EvaluatorAgentAsync HTTP session closed.")

    async def evaluate_async(
        self, question: str, answer: str, idea: str, thought: str
    ) -> Tuple[EvaluationJSONModel, str]:
        """
        Evaluate the answer to a question based on defined criteria.

        Args:
            question (str): The question posed to the user.
            answer (str): The user's answer.
            idea (str): Additional context idea.
            thought (str): Additional context thought.

        Returns:
            Tuple[EvaluationJSONModel, str]:
            - the full eval model, and
            - the raw evaluation JSON.
        """
        # qa_pair = QuestionAnswerPair(question=question, answer=answer)
        # qa_pair_json = qa_pair.model_dump_json(indent=2)  # Nicely formatted JSON string

        prompt = QUESTION_ANSWER_EVAL_PROMPT.format(
            question=question, answer=answer, idea=idea, thought=thought
        )

        logger.info(f"Evaluation prompt: \n{prompt}")

        evaluation_model = await self._call_llm_async(
            prompt=prompt,
            llm_provider=self.llm_provider,
            expected_res_type="json",
            validation_model="eval_json",
        )

        # TODO: debugging, to be deleted later
        logger.debug(f"Evaluation model: {evaluation_model}")
        logger.debug(f"Criteria: {evaluation_model.evaluation.criteria}")
        logger.debug(f"Explanations: {evaluation_model.evaluation.explanations}")
        logger.debug(f"Total score: {evaluation_model.evaluation.total_score}")

        # Convert EvaluationJSONModel to EvaluationCriteria for further processing
        evaluation_criteria = evaluation_model.evaluation

        logger.info(f"Evaluation Criteria: {evaluation_criteria}")

        # Access the raw JSON if needed
        raw_evaluation_json = evaluation_model.model_dump()

        logger.info(
            f"Raw JSON output of evaluation_model:\n{raw_evaluation_json}"
        )  # TODO: debudding, delete later

        return (
            evaluation_model,
            raw_evaluation_json,
        )  # return both the model (EvaluationJSONModel) and raw json

    async def _call_llm_async(
        self,
        prompt: str,
        llm_provider: str,
        expected_res_type: str = "json",
        validation_model: str = "eval_json",
        # Default to str; the response is expected to be just a question, which is just text/str
    ) -> EvaluationJSONModel:
        # Route the API call to the specified LLM provider and return the response
        try:
            if llm_provider == "openai":
                response_model = await call_openai_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    json_type=validation_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.client,
                )

            elif llm_provider == "claude":
                response_model = await call_claude_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    json_type=validation_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.client,
                )
            elif llm_provider == "llama3":
                response_model = await call_llama3_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    json_type=validation_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")

            logger.info(f"Returning response_model: \n{response_model}")

            return response_model  # Expected to return EvaluationJSONModel
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            raise
        except ValidationError as ve:
            logger.error(f"ValidationError during response model conversion: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM call or processing: {e}")
            raise

    def calculate_composite_score(self, criteria: Dict[str, int]) -> float:
        """
        Calculate a composite score as the average of all criteria scores.

        Args:
            criteria (Dict[str, int]): The dictionary of criteria scores.

        Returns:
            float: The composite score rounded to two decimal places.
        """
        if not criteria:
            return 0.0
        total_score = sum(criteria.values())
        composite = round(total_score / len(criteria), 2)

        logger.info(f"Composite Score Calculated: {composite}")  # debugging

        return composite

    def meets_threshold(self, criteria: EvaluationCriteria, threshold: int = 4) -> bool:
        """
        Check if the scores in evaluation criteria meet the given threshold.

        Args:
            criteria (EvaluationCriteria)): Scores for evaluation criteria.
            threshold (int): Minimum acceptable score for "correctness" (default: 3).

        Returns:
            bool: True if correctness score meets the threshold, False otherwise.
        """
        meets = criteria.criteria.get("correctness") >= threshold
        logger.info(f"Meets Threshold ({threshold}): {meets}")
        return meets


# Example usage
async def main():
    api_key = "API_Key"
    evaluator = EvaluatorAgentAsync(
        api_key=api_key, llm_provider="openai", model_id="gpt-3.5-turbo"
    )

    question = (
        "What are the main differences between microcontrollers and microprocessors?"
    )
    answer = "Microcontrollers integrate memory, I/O peripherals, and a CPU on a single chip, whereas microprocessors focus on processing tasks and rely on external memory and peripherals."

    evaluation, evaluation_text = await evaluator.evaluate_async(question, answer)
    composite_score = evaluator.calculate_composite_score(evaluation.criteria)

    print(f"Evaluation: {evaluation}")
    print(f"Evaluation Text: {evaluation_text}")
    print(f"Composite Score: {composite_score}")


# Run the example
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
