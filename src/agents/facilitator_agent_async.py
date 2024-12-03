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
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Dict, List, Union, Optional

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

    async def coordinate_conversation(self) -> None:
        """
        Orchestrate the flow of the conversation.

        Progress through each thought and its sub-thoughts while maintaining state.
        """
        state = self.state_manager.get_state(self.user_id)
        print(f"Agent: Let's start discussing {self.idea_data.idea}.")

        for thought_index in range(state.thought_index, len(self.idea_data.thoughts)):
            thought = self.idea_data.thoughts[thought_index]
            try:
                await self.discuss_thought(
                    thought, thought_index, state.sub_thought_index
                )
            except UserExitException:
                print("Conversation terminated by the user. Goodbye!")
                return

        print("Agent: We've completed all thoughts. Thank you for the discussion!")

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
            await self.discuss_single_sub_thought_async(thought, sub_thought)
            self.state_manager.update_state(
                self.user_id,
                thought_index=thought_index,
                sub_thought_index=sub_index + 1,
            )

    async def discuss_single_sub_thought_async(
        self, thought: IndexedIdeaJSONModel, sub_thought: IndexedSubThoughtJSONModel
    ) -> None:
        """
        Discuss a single sub-thought.

        Args:
            thought (IndexedIdeaJSONModel): Current thought being discussed.
            sub_thought (IndexedSubThoughtJSONModel): Specific sub-thought to discuss.
        """
        # Initialize scoped logging for the sub-thought
        self.start_scoped_logging()

        print(f"\nLet's talk about '{sub_thought.name}'.")
        question = await self.question_generator.generate_initial_question(
            topic_name=sub_thought.name, context_text=sub_thought.description
        )
        print(f"Agent: {question}")
        await self.log_exchange(role="agent", message=question, scoped=True)

        self.topic_exhaustion_service.reset()

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
                    agent_reply = f"Your response is only partially correct. Here's a follow-up question: \
                        {followup_question}"
                    print(f"Agent: {agent_reply}")
                    await self.log_exchange(
                        role="agent", message=agent_reply, scoped=True
                    )

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                print("An error occurred during evaluation. Let's move on.")
                break

            # Check for topic exhaustion
            self.topic_exhaustion_service.set_scoped_logs(
                self.get_scoped_logs_for_sub_thought()
            )

            exhaustion_result = self.topic_exhaustion_service.is_topic_exhausted(
                question, user_response
            )
            if exhaustion_result["is_exhausted"]:
                agent_reply = (
                    f"We've covered '{sub_thought.name}' sufficiently. Let's move on."
                )
                print(f"Agent: {agent_reply}")
                await self.log_exchange(role="agent", message=agent_reply, scoped=True)
                break

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

    async def evaluate_response(
        self, question: str, user_response: str, thought: IndexedIdeaJSONModel
    ) -> EvaluationCriteria:
        """Evaluate a user's response."""
        evaluation, _ = await self.evaluator_agent.evaluate_async(
            question=question,
            answer=user_response,
            idea=self.idea_data.idea,
            thought=thought.thought,
        )  # The 2nd parameter returned is not used in this module
        return evaluation.evaluation

    def start_scoped_logging(self) -> None:
        """Initialize a new scoped log."""
        self.scoped_logs = []

    def get_scoped_logs_for_sub_thought(self) -> List[Dict]:
        """Retrieve logs specific to the current sub-thought."""
        return self.scoped_logs
