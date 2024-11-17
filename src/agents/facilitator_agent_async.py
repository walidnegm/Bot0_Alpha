"""
Facilitator Agent Module

This module defines the FacilitatorAgentAsync class, which conducts an interactive 
conversation with a user about a main concept and its related sub-concepts. 

The agent leverages language models (LLMs) to generate dynamic questions, 
evaluate user responses, and maintain a conversation memory for context
and future reference.

The agent supports multiple LLM providers (e.g., OpenAI, Claude, LLaMA 3) 
and can route API calls to the specified provider using the call_llm method.

Example usage:
    # Run the facilitator agent
    asyncio.run(main())
"""

import json
import asyncio
import aioconsole  # For asynchronous console input/output
import aiohttp  # For making asynchronous HTTP requests
import logging
from typing import Any, Dict, List, Union, Optional

from openai import OpenAI, AsyncOpenAI  # Adjust according to your module's structure
from anthropic import (
    Anthropic,
    AsyncAnthropic,
)  # Adjust according to your module's structure

# Import internal modules
from agents.evaluator_agent_async import EvaluatorAgentAsync
from agents.state_transition_machine import TopicExhaustionService
from agents.state_management import StateManager
from models.llm_response_base_models import TextResponse
from models.indexed_thought_models import IndexedSubThoughtJSONModel
from utils.llm_api_utils import get_claude_api_key, get_openai_api_key
from utils.llm_api_utils_async import (
    call_openai_api_async,
    call_claude_api_async,
    call_llama3_async,
)


# Setup logger
logger = logging.getLogger(__name__)


class FacilitatorAgentAsync:
    """
    FacilitatorAgentAsync Class

    This class facilitates an interactive educational conversation with a user about
    a main concept and its sub-concepts.

    It generates questions, evaluates user responses, and transitions to the next sub-concept
    when the current topic is exhausted.

    Attributes:
        -concept (str): The main concept to discuss.
        -sub_concepts (List[Dict[str, Any]]): A list of sub-concepts related to the main concept.
        -llm_provider (str): The LLM provider to use (e.g., "openai", "claude", "llama3").
        -model_id (str): The model ID to use for the LLM.
        -temperature (float): The temperature setting for the LLM.
        -max_tokens (int): The maximum number of tokens to generate.
        -client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]): The API client instance.
        -conversation_memory (List[Dict[str, Any]]): Memory of the conversation for context and future reference.
        -current_sub_concept_index (int): Index to keep track of progression through sub-concepts.
        -evaluation_agent (EvaluatorAgent): The agent used to evaluate user responses.
        -topic_exhaustion_service (TopicExhaustionService): Service to determine if
        the topic is exhausted based on conversation metrics.
    """

    def __init__(
        self,
        user_id: str,
        idea_data: Dict[str, Any],
        state_manager: StateManager,
        llm_provider: str = "openai",
        model_id: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 1056,
        client: Optional[Union[AsyncOpenAI, AsyncAnthropic]] = None,
    ):
        """
        Initialize the FacilitatorAgentAsync.

        Args:
            data (Dict[str, Any]): A dictionary containing the main concept and sub-concepts.
            llm_provider (str): The LLM provider to use (default is "openai").
            model_id (str): The model ID to use for the LLM (default is "gpt-4-turbo").
            temperature (float): The temperature setting for the LLM (default is 0.3).
            max_tokens (int): The maximum number of tokens to generate (default is 1056).
            client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]): An optional API client instance.

        Raises:
            ValueError: If an unsupported LLM provider is specified and no client is provided.
        """
        self.user_id = user_id
        self.idea_data = idea_data
        self.state_manager = state_manager
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or self._initialize_client()

        self.conversation_memory = []

        # Initialize evaluation and transition agents (placeholders for modularity)
        self.evaluation_agent = EvaluatorAgentAsync()
        self.topic_exhaustion_service = TopicExhaustionService()

        # Log instantiation
        logger.info(
            f"FacilitatorAgentAsync instantiated with provider '{llm_provider}', model '{model_id}', "
            f"temperature {temperature}, max_tokens {max_tokens}."
        )

    def _initialize_client(self):
        """Initialize the LLM client based on the provider."""
        if self.llm_provider == "openai":
            api_key = get_openai_api_key()
            return OpenAI(api_key=api_key)
        elif self.llm_provider == "claude":
            api_key = get_claude_api_key()
            return Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    async def begin_conversation_async(self) -> None:
        """Start the conversation by retrieving or initializing user state."""
        state = self.state_manager.get_state(self.user_id)
        current_thought_index = state.get("thought_index", 0)
        current_sub_thought_index = state.get("sub_thought_index", 0)

        if current_thought_index >= len(self.idea_data.thoughts or []):
            print("All thoughts have been discussed. Thank you!")
            return

        print(f"Today, let's discuss {self.idea_data.idea}.")
        await self.discuss_next_sub_thought_async(
            current_thought_index, current_sub_thought_index
        )

    async def discuss_next_sub_thought_async(
        self, thought_index: int, sub_thought_index: int
    ):
        """
        Discuss the next sub-thought based on the current user state.

        Args:
            thought_index (int): Current thought index.
            sub_thought_index (int): Current sub-thought index within the thought.
        """
        thoughts = self.idea_data.thoughts or []
        if thought_index >= len(thoughts):
            print("We've covered all the main thoughts. Thank you for the discussion!")
            return

        current_thought = thoughts[thought_index]
        sub_thoughts = current_thought.sub_thoughts or []

        while sub_thought_index < len(sub_thoughts):
            current_sub_thought = sub_thoughts[sub_thought_index]
            await self.handle_focused_discussion_async(
                current_thought, current_sub_thought
            )

            sub_thought_index += 1
            self.state_manager.update_state(
                self.user_id,
                thought_index=thought_index,
                sub_thought_index=sub_thought_index,
            )

        thought_index += 1
        sub_thought_index = 0
        self.state_manager.update_state(
            self.user_id,
            thought_index=thought_index,
            sub_thought_index=sub_thought_index,
        )
        await self.discuss_next_sub_thought_async(thought_index, sub_thought_index)

    async def handle_focused_discussion_async(self, thought, sub_thought):
        """
        Conduct a focused discussion on a sub-thought.

        Args:
            thought (IndexedThoughtJSONModel): Current thought being discussed.
            sub_thought (IndexedSubThoughtJSONModel): Current sub-thought.
        """
        print(f"\nLet's talk about {sub_thought.name}.")
        question = await self.generate_question_async(
            sub_thought.name, sub_thought.description
        )
        print(f"Agent: {question}")

        self.topic_exhaustion_service.reset()
        while True:
            user_response = await aioconsole.ainput("You: ")
            try:
                evaluation = await self.evaluation_agent.evaluate_async(
                    correct_answer=sub_thought.description, user_response=user_response
                )
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                print("An error occurred during evaluation. Let's move on.")
                break

            exhaustion_result = self.topic_exhaustion_service.is_topic_exhausted(
                question, user_response
            )
            if exhaustion_result["is_exhausted"]:
                print(
                    f"Agent: We've covered {sub_thought.name} sufficiently "
                    f"(Redundancy={exhaustion_result['redundancy_score']:.2f})."
                )
                break

            if evaluation["is_correct"] or evaluation["should_move_on"]:
                agent_reply = await self.generate_conciliatory_reply_async(evaluation)
                print(f"Agent: {agent_reply}")
                break
            else:
                followup_question = await self.generate_followup_question_async(
                    evaluation
                )
                print(f"Agent: {followup_question}")

    async def generate_question_async(self, name, description=None):
        """Generate a discussion question about the sub-thought."""
        prompt = (
            f"Based on the description: {description}, generate an open-ended question about '{name}'."
            if description
            else f"Generate an open-ended question about '{name}'."
        )
        response = await self.call_llm_async(prompt, self.llm_provider, "str")
        return response.content.strip()

    async def generate_conciliatory_reply_async(
        self, evaluation: Dict[str, Any]
    ) -> str:
        # Generate a conciliatory reply based on the evaluation
        return (
            "That's correct! Great job."
            if evaluation["is_correct"]
            else "Thank you for your thoughts. Let's move on to the next topic."
        )

    async def generate_followup_question_async(
        self,
        evaluation: Dict[str, Any],
        llm_provider: Optional[str] = None,
    ) -> str:
        # Generate a follow-up question to guide the user towards the correct understanding
        prompt: str = f"""Based on the user's previous response and your evaluation:
        {evaluation['evaluation_text']}
        Generate a helpful follow-up question to guide the user towards the correct understanding. Provide only the question and no additional text.
        """
        if llm_provider is None:
            llm_provider = self.llm_provider

        try:
            response: TextResponse = await self.call_llm_async(
                prompt=prompt,
                llm_provider=llm_provider,
                expected_res_type="str",
            )
        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            return "Could you elaborate further?"

        return response.content.strip()

    async def call_llm_async(
        self,
        prompt: str,
        llm_provider: str,
        expected_res_type: str = "str",
        # Default to str; the response is expected to be just a question, which is just text/str
    ) -> TextResponse:
        # Route the API call to the specified LLM provider and return the response
        try:
            if llm_provider == "openai":
                response = await call_openai_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.client,
                )
            elif llm_provider == "claude":
                response = await call_claude_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.client,
                )
            elif llm_provider == "llama3":
                response = await call_llama3_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            return response
        except Exception as e:
            logger.error(f"Error calling LLM '{llm_provider}': {e}")
            raise

    async def persist_to_memory(
        self, memory_json_file: str = "conversation_history.json"
    ) -> None:
        # Save the conversation memory to a JSON file
        try:
            with open(memory_json_file, "w") as f:
                json.dump(self.conversation_memory, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving conversation memory: {e}")
            raise


# Main function to run the agent
async def main():
    # Load data from JSON file
    try:
        with open("concepts.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading concepts file: {e}")
        return

    agent = FacilitatorAgentAsync(data)

    await agent.begin_conversation_async()
    await agent.persist_to_memory()


# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"An error occurred: {e}")
