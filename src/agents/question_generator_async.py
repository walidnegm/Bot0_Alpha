"""
TBA
"""

# Dependencies
from typing import Optional, Union, List, Dict
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
from models.evaluation_models import EvaluationCriteria
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
    Generate a follow-up question based on evaluation, context, and scoped logs.

    Args:
        evaluation (EvaluationCriteria): Evaluation data for the user's response.
        idea (str): The overarching idea or context.
        thought (str): The main thought being discussed.
        sub_thought_description (str): The sub-thought description for context.
        context_logs (List[Dict]): Scoped logs for the current sub-thought.

    Returns:
        str: The generated follow-up question.
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
        self,
        evaluation: EvaluationCriteria,
        idea: str,
        thought: str,
        sub_thought_description: str,
        context_logs: List[Dict],
    ) -> TextResponse:
        """
        Generate a follow-up question based on evaluation, context, and scoped logs.

        Args:
            - evaluation (EvaluationCriteria): Evaluation data (pydantic model) for the user's response.
            - idea (str): The overarching idea or context.
            - thought (str): The main thought being discussed.
            - sub_thought_description (str): The sub-thought description for context.
            - context_logs (List[Dict]): Scoped logs for the current sub-thought.

        Returns:
            str: The generated follow-up question.

        >>> Example:
        *Input (Mocked):
        - evaluation:
            relevance: 4/5, "Addresses key aspects but misses technical depth."
            correctness: 5/5, "All details are accurate."
            specificity: 3/5, "Could include more specific use cases."
            clarity: 5/5, "The response is clear and easy to follow."
            total_score: 4.25
        - idea: "Embedded Software Development for Aerospace"
        - thought: "Safety-Critical Requirements in Embedded Systems"
        - sub_thought_description: "Determining software safety levels (DAL)"
        - context_logs:
            [
                {"role": "agent", "message": "What are the key factors for determining software \
                    safety levels?"},
                {"role": "user", "message": "Software safety levels depend on system criticality \
                    and failure impact."},
            ]
        
        *Generated Prompt for LLM (Mocked)
        Given the following context, evaluation, and conversation so far:

        Context:
        - Idea: Embedded Software Development for Aerospace
        - Main Thought: Safety-Critical Requirements in Embedded Systems
        - Sub-thought: Determining software safety levels (DAL) for certification compliance

        Evaluation:
        - Relevance: 4/5, The response is mostly relevant but misses key technical details.
        - Correctness: 3/5, Some factual inaccuracies were observed.
        - Specificity: 2/5, The response is too generic and lacks depth.
        - Clarity: 5/5, The response is well-structured and easy to follow.

        Conversation Context:
        Agent: What are the key factors for determining software safety levels?
        User: Software safety levels depend on system criticality and failure impact.
        Agent: Can you provide an example of how failure impact influences safety levels?

        Generate an insightful follow-up question that:
        - Builds upon the previous discussion
        - Probes deeper into the underlying concepts
        - Encourages further critical analysis
        - Is precise and thought-provoking.

        *Output (Mocked):
        "What specific techniques or frameworks can be used to assess failure impact in \
            safety-critical systems, and how do they influence the determination of DAL \
                in aerospace applications?"

        """
        # Serialize evaluation scores and explanations
        evaluation_scores_and_explanations = "\n".join(
            f"- {criterion.capitalize()}: {score}/5, {evaluation.explanations[criterion]}"
            for criterion, score in evaluation.criteria.items()
        )

        # Format conversation context
        conversation_context = "\n".join(
            f"{log['role'].capitalize()}: {log['message']}" for log in context_logs
        )

        # Build the prompt
        prompt = FOLLOWUP_QUESTION_GENERATION_PROMPT.format(
            idea=idea,
            main_thought=thought,
            sub_thought_description=sub_thought_description,
            evaluation_scores_and_explanations=evaluation_scores_and_explanations,
            conversation_context=conversation_context,
        )

        logger.info(f"Follow-up question generation prompt: {prompt}")

        return await self.call_llm_async(prompt)  # Returns a pydantic obj

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
