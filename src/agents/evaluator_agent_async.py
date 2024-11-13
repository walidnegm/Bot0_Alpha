"""
Evaluator Agent Module

This module defines the EvaluatorAgent class, which evaluates a user’s response 
to a question based on predefined criteria. It supports multiple LLM providers 
(OpenAI, Claude, LLaMA 3) and can either instantiate an API client or use an inherited one.

Example Usage:
    evaluator = EvaluatorAgent(api_key='your_key', llm_provider='openai', model_id='gpt-4')
    question = "What are the main differences between microcontrollers and microprocessors?"
    answer = "Microcontrollers integrate memory, I/O peripherals, and a CPU on a single chip, \
        whereas microprocessors focus on processing tasks and rely on external memory and peripherals."
    evaluation, evaluation_text = await evaluator.evaluate(question, answer)
    composite_score = evaluator.calculate_composite_score(evaluation.criteria)
"""

import re
import json
import logging
from typing import Dict, Tuple, Union, Optional
from pydantic import BaseModel, Field, field_validator
from utils.llm_api_utils_async import (
    call_openai_api_async,
    call_claude_api_async,
    call_llama3_async,
)

# Initialize logger
logger = logging.getLogger(__name__)

# Define LLM model options for different providers
LLM_MODELS = {
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "claude": ["claude-3", "claude-3-5"],
    "llama3": ["llama3-turbo"],
}

# Scoring criteria
CRITERIA_KEYS = {"relevance", "correctness", "specificity", "clarity"}


class EvaluationCriteria(BaseModel):
    criteria: Dict[str, int] = Field(..., min_items=4, max_items=4)
    explanations: Dict[str, str] = Field(..., min_items=4, max_items=4)

    @field_validator("criteria")
    def check_criteria_keys(cls, v):
        if set(v.keys()) != CRITERIA_KEYS:
            raise ValueError(
                f"Criteria must include exactly these keys: {CRITERIA_KEYS}"
            )
        return v

    @field_validator("explanations")
    def check_explanations_keys(cls, v):
        if set(v.keys()) != CRITERIA_KEYS:
            raise ValueError(
                f"Explanations must include exactly these keys: {CRITERIA_KEYS}"
            )
        return v


class QuestionAnswerPair(BaseModel):
    question: str
    answer: str

    @field_validator("question", "answer")
    def check_non_empty(cls, v):
        if not v.strip():
            raise ValueError("Question and answer cannot be empty")
        return v


class EvaluatorAgentAsync:
    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_provider: str = "openai",
        model_id: str = "gpt-3.5-turbo",
        client: Optional[
            Union[call_openai_api_async, call_claude_api_async, call_llama3_async]
        ] = None,
    ):
        """
        Initialize the EvaluatorAgent.

        Args:
            api_key (Optional[str]): The API key for the LLM provider (optional if a client is inherited).
            llm_provider (str): The LLM provider to use (default is 'openai').
            model_id (str): The model ID to use for the LLM.
            client (Optional[Callable]): Optionally inherit an API client for OpenAI, Claude, or LLaMA 3.
        """
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.client = client

        logger.info(
            f"Evaluator Agent (Async) instantiated with provider '{llm_provider}', model '{model_id}'."
        )

    async def evaluate_async(
        self, question: str, answer: str
    ) -> Tuple[EvaluationCriteria, str]:
        """
        Evaluate the answer to a question based on defined criteria.

        Args:
            question (str): The question posed to the user.
            answer (str): The user's answer.

        Returns:
            Tuple[EvaluationCriteria, str]: The evaluation criteria with scores and explanations, and the raw evaluation text.
        """
        qa_pair = QuestionAnswerPair(question=question, answer=answer)

        prompt = f"""
        Question: {qa_pair.question}
        Answer: {qa_pair.answer}

        Please evaluate the given answer based on the following criteria:
        1. Relevance: How relevant is the answer to the question?
        2. Correctness: Is the answer factually correct?
        3. Specificity: Does the answer provide enough detail and specificity?
        4. Clarity: Is the answer clearly and well-written?

        Provide a score from 1 to 5 for each criterion and a brief explanation for each score.
        """

        response_text = await self.call_llm_async(prompt=prompt)
        criteria, explanations = self.parse_evaluation(response_text)

        evaluation = EvaluationCriteria(criteria=criteria, explanations=explanations)
        return evaluation, response_text

    async def call_llm_async(self, prompt: str) -> str:
        """
        Route API call to the specified LLM provider and return the response text.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response text from the LLM.

        Raises:
            ValueError: If an unsupported LLM provider is specified.
        """
        if self.llm_provider == "openai":
            response = await call_openai_api_async(
                prompt=prompt, model_id=self.model_id, client=self.client
            )
        elif self.llm_provider == "claude":
            response = await call_claude_api_async(
                prompt=prompt, model_id=self.model_id, client=self.client
            )
        elif self.llm_provider == "llama3":
            response = await call_llama3_async(prompt=prompt, model_id=self.model_id)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        return response.content.strip()

    def parse_evaluation(
        self, evaluation_text: str
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Parse the LLM's evaluation text to extract scores and explanations.

        Args:
            evaluation_text (str): The raw evaluation response from the LLM.

        Returns:
            Tuple[Dict[str, int], Dict[str, str]]: Parsed scores and explanations.
        """
        lines = evaluation_text.split("\n")
        criteria = {}
        explanations = {}
        for line in lines:
            if "Relevance" in line:
                criteria["relevance"], explanations["relevance"] = (
                    self.extract_score_and_explanation(line)
                )
            elif "Correctness" in line:
                criteria["correctness"], explanations["correctness"] = (
                    self.extract_score_and_explanation(line)
                )
            elif "Specificity" in line:
                criteria["specificity"], explanations["specificity"] = (
                    self.extract_score_and_explanation(line)
                )
            elif "Clarity" in line:
                criteria["clarity"], explanations["clarity"] = (
                    self.extract_score_and_explanation(line)
                )
        return criteria, explanations

    def extract_score_and_explanation(self, line: str) -> Tuple[int, str]:
        """
        Extract the score and explanation from a line of evaluation text.

        Args:
            line (str): The line of text containing a criterion score and explanation.

        Returns:
            Tuple[int, str]: The score as an integer and the explanation as a string.
        """
        score_part, explanation = line.split(":", 1)
        score = int(re.search(r"\d+", score_part).group())
        return score, explanation.strip()

    def calculate_composite_score(self, criteria: Dict[str, int]) -> float:
        """
        Calculate a composite score as the average of all criteria scores.

        Args:
            criteria (Dict[str, int]): The dictionary of criteria scores.

        Returns:
            float: The composite score.
        """
        total_score = sum(criteria.values())
        return round(total_score / len(criteria), 2)


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
