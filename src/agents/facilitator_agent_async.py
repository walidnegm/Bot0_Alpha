"""
Facilitator Agent Module

This module defines the FacilitatorAgentAsync class, which facilitates an interactive 
conversation with a user about a main concept and its related sub-concepts.

The agent leverages language models (LLMs) to generate dynamic questions, 
evaluate user responses, and maintain a conversation memory for context
and future reference.

The agent supports multiple LLM providers (e.g., OpenAI, Claude, LLaMA 3) 
and can route API calls to the specified provider using the call_llm_async method.

Example usage:
    # Run the facilitator agent
    asyncio.run(main())


TODO: Needed to handle "I don't know type of question"
TODO: Need to add reflective agent
TODO: Need to define roles of different agents and leverage it to dynamically create prompts
"""

import json
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Dict, List, Union, Optional
from pydantic import ValidationError

import logging
import aiofiles
import aioconsole

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

import logging_config
from agents.evaluator_agent_async import EvaluatorAgentAsync
from agents.reflective_agent_async import ReflectiveAgentAsync
from agents.state_transition_machine import TopicExhaustionService
from agents.state_management import StateManager
from agents.question_generator_async import QuestionGeneratorAsync
from models.llm_response_models import TextResponse
from models.indexed_thought_models import (
    IndexedIdeaJSONModel,
    IndexedSubThoughtJSONModel,
)
from models.evaluation_models import EvaluationCriteria
from utils.llm_api_utils_async import (
    call_openai_api_async,
    call_claude_api_async,
    call_llama3_async,
)
from utils.llm_api_utils import get_openai_api_key, get_claude_api_key


# Setup logger
logger = logging.getLogger(__name__)


class UserExitException(Exception):
    """Custom exception to handle user termination of the conversation."""


class FacilitatorAgentAsync:
    """
    FacilitatorAgentAsync Class

    Facilitates an interactive discussion based on a structured model of ideas, thoughts, and sub-thoughts.
    This class handles question generation, response evaluation, and memory management.

    Attributes:
        user_id (str): Identifier for the user.
        memory_file (Union[Path, str]): Path to the file where conversation history is stored.
        idea_data (IndexedIdeaJSONModel): Data model containing the main idea, thoughts, and sub-thoughts.
        state_manager (StateManager): Instance to track user progress and state.
        llm_provider (str): Provider of the LLM (e.g., "openai").
        model_id (str): Specific model ID for the LLM.
        temperature (float): Temperature for controlling randomness in LLM responses.
        max_tokens (int): Maximum token limit for responses.
        client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]): LLM client instance.
        scoped_logs (List[Dict]): Scoped logs for the current sub-thought.
        conversation_memory (List[Dict]): History of the conversation.

    Methods:
        - `coordinate_conversation`: Orchestrates the flow of the conversation.
        - `discuss_thought`: Discusses all sub-thoughts of a given main thought.
        - `discuss_single_sub_thought`: Handles interaction for a single sub-thought.
        - `log_exchange`: Logs a single exchange between user and agent.
        - `persist_chat_history_to_disk`: Saves the full conversation memory to disk.
        - `evaluate_response`: Evaluates user responses using the evaluator agent.
    """

    SAFE_WORD = "exit"  # Safe word to terminate the conversation

    def __init__(
        self,
        user_id: str,
        memory_file: Union[Path, str],
        idea_data: IndexedIdeaJSONModel,
        state_manager: StateManager,
        llm_provider: str = "openai",
        model_id: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 1056,
        client: Optional[Union[Any, Any]] = None,
    ):
        """
        Initialize the FacilitatorAgentAsync instance.

        Args:
            user_id (str): User identifier.
            memory_file (Union[Path, str]): File to persist conversation memory.
            idea_data (IndexedIdeaJSONModel): Idea data containing thoughts and sub-thoughts.
            state_manager (StateManager): Manages user state and progress.
            llm_provider (str): Language model provider (default: "openai").
            model_id (str): LLM model ID (default: "gpt-4-turbo").
            temperature (float): Temperature for randomness in responses (default: 0.3).
            max_tokens (int): Maximum tokens for LLM response (default: 1056).
            client (Optional[Any]): Optional LLM client instance.
        """
        self.user_id = user_id
        self.memory_file = memory_file
        self.idea_data = idea_data
        self.state_manager = state_manager
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or self._initialize_client()
        self.scoped_logs = []
        self.conversation_memory = []
        self.agent_index = {}
        self._initialize_agents()

    def _initialize_client(self) -> Any:
        """Initialize the LLM client based on the provider."""
        if self.llm_provider == "openai":
            return AsyncOpenAI(api_key=get_openai_api_key())
        elif self.llm_provider == "claude":
            return AsyncAnthropic(api_key=get_claude_api_key())
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _initialize_agents(self):
        """Dynamically initialize supporting agents."""
        self.evaluator_agent = EvaluatorAgentAsync(
            llm_provider=self.llm_provider,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            client=self.client,
        )
        self.question_generator = QuestionGeneratorAsync(
            llm_provider=self.llm_provider,
            model_id=self.model_id,
            llm_client=self.client,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.topic_exhaustion_service = TopicExhaustionService()

    async def coordinate_conversation(
        self, thought_indexes: Optional[List[int]] = None
    ) -> None:
        """
        Orchestrate the flow of the conversation.

        Progress through each thought and its sub-thoughts while maintaining state.
        """
        state = self.state_manager.get_state(self.user_id)
        agent_reply = f"The main topic is {self.idea_data.idea}."
        print(f"Agent: {agent_reply}")
        await self.log_exchange(role="agent", message=agent_reply)

        # Determine which thoughts to process
        if thought_indexes is None:
            thought_indexes = range(state.thought_index, len(self.idea_data.thoughts))
        else:
            thought_indexes = [
                idx
                for idx in thought_indexes
                if state.thought_index <= idx < len(self.idea_data.thoughts)
            ]

        for thought_index in thought_indexes:
            thought = self.idea_data.thoughts[thought_index]
            try:
                await self.discuss_thought(
                    thought=thought,
                    thought_index=thought_index,
                    sub_thought_index=state.sub_thought_index,
                )
            except UserExitException:
                print("Conversation terminated by the user. Goodbye!")
                return

        agent_reply = "We've completed all thoughts. Thank you for the discussion!"
        print(f"Agent: {agent_reply}")
        await self.log_exchange(role="agent", message=agent_reply)

    async def discuss_thought(
        self, thought: IndexedIdeaJSONModel, thought_index: int, sub_thought_index: int
    ) -> None:
        """
        Discuss a main thought and its associated sub-thoughts.

        Args:
            thought (IndexedIdeaJSONModel): Current thought being discussed.
            thought_index (int): Index of the thought.
            sub_thought_index (int): Starting sub-thought index.
        """
        print(f"Agent: Now discussing '{thought.thought}'.")
        for sub_index in range(sub_thought_index, len(thought.sub_thoughts)):
            sub_thought = thought.sub_thoughts[sub_index]
            await self.discuss_single_sub_thought_async(thought, sub_thought, 5)
            self.state_manager.update_state(
                self.user_id,
                thought_index=thought_index,
                sub_thought_index=sub_index + 1,
            )

    async def discuss_single_sub_thought_async(
        self,
        thought: IndexedIdeaJSONModel,
        sub_thought: IndexedSubThoughtJSONModel,
        max_no_questions: int = 100,  # default to a very high number -> no upper limit
    ) -> None:
        """
        Discuss a single sub-thought.

        Args:
            - thought (IndexedIdeaJSONModel): Current thought being discussed.
            - sub_thought (IndexedSubThoughtJSONModel): Specific sub-thought to discuss.
            - max_no_questions (int): Maximum number of follow-up questions to ask.
                                    Defaults to 100 (effectively no limit).
        """
        # Initialize scoped logging for the sub-thought
        self.start_scoped_logging()

        print(f"\nLet's talk about '{sub_thought.name}'.")
        question = await self.question_generator.generate_initial_question(
            topic_name=sub_thought.name, context_text=sub_thought.description
        )

        # Ensure question content is a string
        if not isinstance(question, TextResponse):
            raise TypeError(f"Expected TextResponse, got {type(question).__name__}")
        question: str = question.content  # Explicitly extract content

        print(f"Agent: {question}")
        await self.log_exchange(role="agent", message=question, scoped=True)

        self.topic_exhaustion_service.reset()

        # Initialize follow-up question counter
        follow_up_count = 0
        max_follow_up_questions = (
            max_no_questions - 1
        )  # Set the maximum number of follow-up questions

        while True:
            # Get user response
            user_response = await aioconsole.ainput("You: ")
            await self.log_exchange(role="user", message=user_response, scoped=True)

            # Check for SAFE_WORD
            if user_response.strip().lower() == self.SAFE_WORD:
                raise UserExitException()

            try:
                # Evaluate user response
                evaluation = await self.evaluate_response(
                    question, user_response, thought
                )

                # Update state with evaluation results
                try:
                    self.state_manager.update_state(
                        self.user_id, current_evaluation=evaluation
                    )
                except Exception as e:
                    logger.error(f"Failed to update state with evaluation: {e}")

                # Check if response meets threshold
                meets_threshold = self.evaluator_agent.meets_threshold(
                    criteria=evaluation, threshold=4.5
                )
                if meets_threshold:
                    agent_reply = (
                        "That's correct! Great job. Let's move on to the next topic."
                    )
                    print(f"Agent: {agent_reply}")
                    await self.log_exchange(
                        role="agent", message=agent_reply, scoped=True
                    )
                    break
                else:
                    # Generate follow-up question
                    followup_question = (
                        await self.question_generator.generate_followup_question(
                            evaluation=evaluation,
                            idea=self.idea_data.idea,
                            thought=thought.thought,
                            sub_thought_description=sub_thought.description,
                            context_logs=self.get_scoped_logs_for_sub_thought(),
                        )
                    )

                    # Ensure question content is a string
                    if not isinstance(followup_question, TextResponse):
                        raise TypeError(
                            f"Expected TextResponse, got {type(question).__name__}"
                        )
                    followup_question: str = (
                        followup_question.content
                    )  # Explicitly extract content

                    # Increment follow-up question counter
                    follow_up_count += 1
                    if follow_up_count > max_follow_up_questions:
                        print(
                            "Agent: Let's move on, as we've had enough discussion on this topic."
                        )
                        logger.info(
                            f"Follow-up question limit of {max_no_questions} reached for sub-thought: {sub_thought.name}."
                        )
                        break

                    agent_reply = (
                        f"Your response is only partially correct. Here's a follow-up question: "
                        f"{followup_question}"
                    )
                    print(f"Agent: {agent_reply}")
                    await self.log_exchange(
                        role="agent", message=agent_reply, scoped=True
                    )

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                print("An error occurred during evaluation. Let's move on.")
                break

            # Check for topic exhaustion (if redundancy or lack of new info)
            self.topic_exhaustion_service.set_scoped_logs(
                self.get_scoped_logs_for_sub_thought()
            )

            exhaustion_result = self.topic_exhaustion_service.is_topic_exhausted(
                answer=user_response
            )
            if exhaustion_result["is_exhausted"]:
                agent_reply = (
                    f"We've covered '{sub_thought.name}' sufficiently. Let's move on."
                )
                print(f"Agent: {agent_reply}")
                await self.log_exchange(role="agent", message=agent_reply, scoped=True)
                break

    async def evaluate_response(
        self, question: str, user_response: str, thought: IndexedIdeaJSONModel
    ) -> EvaluationCriteria:
        """
        Evaluate a user's response based on a given question and thought context.

        This method integrates with the evaluator agent to analyze the user's response
        and validate the evaluation result. It performs the following transformations:

        1. Input Parameters:
        - `question` (str): The question posed to the user.
        - `user_response` (str): The user's answer to the question.
        - `thought` (IndexedIdeaJSONModel): The context of the current thought being
        discussed.

        2. Call to 'EvaluatorAgentAsync.evaluate_async':
        - Input:
            - 'question': The question posed to the user.
            - 'user_response': The user's answer to the question.
            - `idea` (str): The overarching idea, extracted as `self.idea_data.idea`.
            - `thought` (str): The specific thought being evaluated,
            extracted as `thought.thought`.
        - Output:
            - `evaluation_model` (EvaluationJSONModel): A structured evaluation model
            containing the evaluation results as a nested `evaluation` field of
            type `EvaluationCriteria`.

        3. Validation and Parsing:
        - The `evaluation` field from `EvaluationJSONModel` is validated as
        an `EvaluationCriteria` object using `EvaluationCriteria.model_validate`.
        - This ensures the data conforms to the expected structure.

        4. Model Differences:
        - `EvaluationJSONModel`:
            - Represents the complete evaluation response returned from the evaluator agent.
            - Contains metadata and other fields beyond just the evaluation criteria.
            - Includes the `evaluation` field, which holds the evaluation details.
        - `EvaluationCriteria`:
            - A subset of the `EvaluationJSONModel`, representing only the evaluation details.
            - Includes specific scores (`criteria`), explanations (`explanations`),
            and a `total_score`.

        5. Output:
        - Returns a validated `EvaluationCriteria` object containing:
            - `criteria` (Dict[str, int]): Scoring for specific evaluation dimensions
            (e.g., relevance, clarity).
            - `explanations` (Dict[str, str]): Textual explanations for the scores.
            - `total_score` (float): An aggregated score across all dimensions.

        Raises:
            ValidationError: If the evaluation model fails validation.

        Returns:
            EvaluationCriteria: The validated evaluation criteria model.
        """
        # Call evaluator agent's evaluation_async method to return a EvaluationJSONModel model
        evaluation_model, _ = await self.evaluator_agent.evaluate_async(
            question=question,
            answer=user_response,
            idea=self.idea_data.idea,  # from pyd model -> attribut (str)
            thought=thought.thought,  # from pyd model -> attribute (str)
        )  # The 2nd parameter returned is not used in this module

        # Log raw evaluation data for debugging
        logger.debug(
            f"EvolutionCriteria data to validate: {evaluation_model.evaluation}"
        )  # TODO: Debugging; delete later
        # Parse EvaluationJSONModel -> EvaluationCriteria Validate the evaluation model
        # model validate, and then return EvalCriteria model
        try:
            # Parse model and validate
            evaluation_criteria_model = EvaluationCriteria.model_validate(
                evaluation_model.evaluation
            )

            logger.info("Evaluation criteria model validated and returned.")

            return evaluation_criteria_model  # Returning validated EvaluationCriteria
        except ValidationError as e:
            logger.error(f"Validation failed for evaluation model: {e}")
            raise

    async def log_exchange(self, role: str, message: str, scoped: bool = False) -> None:
        """Log an interaction and optionally store it in scoped logs."""
        log_entry = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
        }
        async with aiofiles.open(self.memory_file, "w") as f:
            self.conversation_memory.append(log_entry)
            await f.write(json.dumps(self.conversation_memory, indent=2))

        if scoped:
            self.scoped_logs.append(log_entry)

    def start_scoped_logging(self) -> None:
        """Initialize a new scoped log."""
        self.scoped_logs = []

    def get_scoped_logs_for_sub_thought(self) -> List[Dict]:
        """Retrieve logs specific to the current sub-thought."""
        return self.scoped_logs
