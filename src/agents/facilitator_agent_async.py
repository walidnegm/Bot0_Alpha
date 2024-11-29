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
import aiofiles
import logging
from typing import Any, Dict, List, Union, Optional

from openai import OpenAI, AsyncOpenAI  # Adjust according to your module's structure
from anthropic import (
    Anthropic,
    AsyncAnthropic,
)  # Adjust according to your module's structure

# Import internal modules
from agents.evaluator_agent_async import EvaluatorAgentAsync
from agents.reflective_agent_async import ReflectiveAgentAsync
from agents.state_transition_machine import TopicExhaustionService
from agents.state_management import StateManager
from models.llm_response_models import TextResponse
from models.indexed_thought_models import (
    IndexedIdeaJSONModel,
    IndexedSubThoughtJSONModel,
)
from utils.generic_utils import read_from_json_file, save_to_json_file
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
    a main concept and its sub-concepts. It leverages structured data models and state
    management to ensure a consistent and progressive discussion experience.

    Key Features:
        - **Interactive Discussions**: Guides users through a series of thoughts and sub-thoughts
          using data from the `IndexedIdeaJSONModel`.
        - **State Management**: Integrates with `StateManager` to track user progress (e.g.,
          thought index and sub-thought index) and ensure continuity across sessions.
        - **Asynchronous Operations**: Utilizes asynchronous methods for efficient handling of
          user input, API calls, and file I/O.
        - **Flexible Integration**: Modular design supports the addition of other agents
          (e.g., ReflectiveAgent, EvaluatorAgent) and state transition machines.
        - **Persistence**: Periodically or on-demand saves conversation history to a JSON file,
          ensuring data is not lost during a session.

    Attributes:
        - user_id (str): Identifier for the user.
        - idea_data (IndexedIdeaJSONModel): The main concept and associated thoughts/sub-thoughts.
        - state_manager (StateManager): Tracks and persists user progress across conversations.
        - llm_provider (str): The LLM provider to use (e.g., "openai", "claude").
        - model_id (str): The specific model ID for the LLM (e.g., "gpt-4-turbo").
        - temperature (float): Temperature setting for LLM responses, controlling randomness.
        - max_tokens (int): Maximum number of tokens allowed in LLM responses.
        - client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]): API client instance for
          LLM calls.
        - conversation_memory (List[Dict]): A list of exchanges (agent-user interactions)
          for persistence.
        - memory_lock (asyncio.Lock): Ensures safe access to shared resources during
          asynchronous tasks.
        - evaluation_agent (EvaluatorAgentAsync): An agent for evaluating user responses.
        - topic_exhaustion_service (TopicExhaustionService): Determines when a topic
          is exhausted.

    Key Methods:
        - `begin_conversation_async`: Starts the conversation, progressing through thoughts
          and sub-thoughts.
        - `discuss_next_sub_thought_async`: Manages the sequential discussion of thoughts and
          sub-thoughts.
        - `handle_focused_discussion_async`: Conducts multi-turn discussions on a specific sub-thought.
        - `persist_to_memory`: Saves the conversation history to a JSON file (asynchronous).
        - `update_memory_and_persist`: Updates conversation memory and persists changes asynchronously.
        - `generate_question_async`: Produces discussion questions using an LLM.
        - `generate_conciliatory_reply_async`: Generates replies to acknowledge correct or
          sufficient responses.
        - `generate_followup_question_async`: Produces follow-up questions based on user responses.
        - `call_llm_async`: Routes API calls to the specified LLM provider.

    Usage:
        1. Instantiate the class with a user ID, structured idea data (`IndexedIdeaJSONModel`),
           and a `StateManager`.
        2. Call `begin_conversation_async` to start the discussion.
        3. The agent will progress through the main thoughts and sub-thoughts, evaluating
           user responses and transitioning topics as necessary.
        4. Use `persist_to_memory` to save conversation history periodically or at the end of
           the session.
    """

    SAFE_WORD = "exit"  # *Define the safe word

    def __init__(
        self,
        user_id: str,
        idea_data: IndexedIdeaJSONModel,
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
            user_id (str): Identifier for the user.
            idea_data (IndexedIdeaJSONModel): Data model containing main and sub-thoughts.
            state_manager (StateManager): Instance of StateManager for state tracking.
            llm_provider (str): The LLM provider to use (default is "openai").
            model_id (str): The model ID to use for the LLM (default is "gpt-4-turbo").
            temperature (float): The temperature setting for the LLM (default is 0.3).
            max_tokens (int): The maximum number of tokens to generate (default is 1056).
            client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]): API client instance.

        Raises:
            ValueError: If an unsupported LLM provider is specified and no client is provided.
        """
        logger.debug("Initializing agents ...")

        self.user_id = user_id
        self.idea_data = idea_data
        self.state_manager = state_manager
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or self._initialize_client()

        self.conversation_memory = []
        self.memory_lock = (
            asyncio.Lock()
        )  # Lock for ensuring memory operations are ordered

        # Initialize evaluation, topic exhaustion, and reflective agents
        self.agent_index = {}  # Centralized index for agents
        self._initialize_agents()

    def _initialize_client(self):
        """Initialize the LLM client based on the provider."""
        if self.llm_provider == "openai":
            api_key = get_openai_api_key()
            return AsyncOpenAI(api_key=api_key)
        elif self.llm_provider == "claude":
            api_key = get_claude_api_key()
            return AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _initialize_agents(self):
        """Dynamically initialize agents and services."""
        self.evaluator_agent = EvaluatorAgentAsync(
            llm_provider=self.llm_provider,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            client=self.client,
        )
        self.topic_exhaustion_service = TopicExhaustionService()
        self.reflective_agent = ReflectiveAgentAsync()

        # # TODO: The agent index approach: implement later
        # self.agent_index = {
        #     "evaluator_agent": EvaluatorAgentAsync(
        #         llm_provider=self.llm_provider,
        #         model_id=self.model_id,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens,
        #         client=self.client,
        #     ),
        #     "topic_exhaustion_service": TopicExhaustionService(),
        #     "reflective_agent": ReflectiveAgentAsync(),
        # }

        # # *Set attributes for the agents
        # self.evaluator_agent = self.agent_index["evaluator_agent"]
        # self.topic_exhaustion_service = self.agent_index["topic_exhaustion_service"]
        # self.reflective_agent = self.agent_index["reflective_agent"]

        logger.debug(f"Agent index after initialization: {self.agent_index}")
        logger.debug(
            f"Initialized agents: evaluator_agent={self.evaluator_agent}, "
            f"topic_exhaustion_service={self.topic_exhaustion_service}, "
            f"reflective_agent={self.reflective_agent}"
        )

    def get_agent(self, agent_name: str):
        """Retrieve an agent or service by its name."""
        agent = self.agent_index.get(agent_name)
        if agent is None:
            logger.error(f"Agent '{agent_name}' not found in agent_index.")
        else:
            logger.debug(f"Retrieved agent '{agent_name}': {agent}")
        return agent

    async def begin_conversation_async(
        self, thought_index: Optional[int] = None
    ) -> None:
        """
        Start the conversation by retrieving or initializing user state, or explicitly
        targeting a specific thought index.

        Args:
            thought_index (Optional[int]): The specific thought index to start with. If None,
            the agent will default to the user's state from the StateManager.
        """
        logger.debug(
            f"Starting conversation for user {self.user_id} with thought_index={thought_index}"
        )  # Debugging

        # Retrieve the user's current state as a Pydantic model
        state = self.state_manager.get_state(self.user_id)
        self.state_manager.get_state(self.user_id)
        if thought_index is not None:
            current_thought_index = thought_index
            current_sub_thought_index = 0

        else:
            #  Default to the user's progress stored in StateManager
            current_thought_index = state.thought_index
            current_sub_thought_index = state.sub_thought_index

        if current_thought_index >= len(self.idea_data.thoughts or []):
            print("All thoughts have been discussed. Thank you!")
            return

        logger.debug(
            f"Starting discussion on thought index {current_thought_index}, sub-thought index {current_sub_thought_index}."
        )  # TODO: debugging / to be deleted later

        print(f"Today, let's discuss {self.idea_data.idea}.")
        await self.discuss_next_sub_thought_async(
            current_thought_index, current_sub_thought_index
        )

    async def discuss_next_sub_thought_async(
        self, thought_index: int, sub_thought_index: int
    ):
        """
        Discuss the next sub-thought based on the provided thought index and the user's state.

        Args:
            thought_index (int): The index of the thought to discuss.
            sub_thought_index (int): The starting sub-thought index within the thought.
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

        # Default to the next thought only if not explicitly provided
        if thought_index == self.state_manager.get_state(self.user_id).thought_index:
            thought_index += 1
            sub_thought_index = 0
            self.state_manager.update_state(
                self.user_id,
                thought_index=thought_index,
                sub_thought_index=sub_thought_index,
            )
            await self.discuss_next_sub_thought_async(thought_index, sub_thought_index)

    async def handle_focused_discussion_async(
        self, thought: IndexedIdeaJSONModel, sub_thought: IndexedSubThoughtJSONModel
    ):
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
            # Get response from the user
            user_response = await aioconsole.ainput("You: ")
            logger.info(f"user_response: {user_response}")

            # Check if the user used the safe word
            if user_response.strip().lower() == self.SAFE_WORD:
                print("Conversation ended by the user. Goodbye!")
                return  # Exit the discussion gracefully

            # TODO: debugging; remove later!
            logger.debug(f"Question: {question}")
            logger.debug(f"Answer: {user_response}")
            logger.debug(f"Idea: {self.idea_data.idea}")
            logger.debug(f"Thought: {thought.thought}")

            # *MAIN CONVERSATION PROCESS
            try:
                # Pass response to the evaluator agent for eval
                evaluation_model, raw_evaluation_json = (
                    await self.evaluator_agent.evaluate_async(
                        question=question,
                        answer=user_response,
                        idea=self.idea_data.idea,
                        thought=str(
                            thought.thought
                        ),  # thought is a pyd model (need to call its thought attribute)
                    )
                )
                evaluation = evaluation_model.evaluation  # extract evaluation

                # Update state manager with eval data
                try:
                    self.state_manager.update_state(
                        self.user_id, current_evaluation=evaluation
                    )
                except Exception as e:
                    logger.error(f"Failed to update state with evaluation: {e}")

                # Check if user response's eval score meets the "correctness" threshold
                meets = self.evaluator_agent.meets_threshold(
                    criteria=evaluation, threshold=4
                )
                if meets:
                    print(f"Agent: Your response is mostly correct.")
                    # Move to the next sub-thought or topic
                else:
                    print(
                        f"Ageent: Your response is only partially correct. Let me clarify further."
                    )
                    followup_question = await self.generate_followup_question_async(
                        evaluation
                    )
                    print(f"Agent: {followup_question}")

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                logger.info("An error occurred during evaluation. Let's move on.")
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

        # Append the discussion to memory and persist it
        await self.update_memory_and_persist(
            {
                "thought": thought.thought,
                "sub_thought": sub_thought.name,
                "user_response": user_response,
                "evaluation": evaluation,
            }
        )

    async def update_memory_and_persist(self, data: Dict):
        """
        Update conversation memory and persist changes asynchronously.

        Args:
            data (Dict): Data to append to the conversation memory.
        """
        async with self.memory_lock:  # Lock ensures exclusive access
            self.conversation_memory.append(data)
            await self.persist_to_memory()

    async def persist_to_memory(self, memory_json_file: str) -> None:
        """
        Persist the conversation memory to a JSON file.

        Args:
            memory_json_file (str): Path to the JSON file for saving the conversation memory.
        """
        async with self.memory_lock:  # Lock ensures no updates occur during persistence
            try:
                async with aiofiles.open(memory_json_file, "w") as f:
                    await f.write(json.dumps(self.conversation_memory, indent=2))
            except IOError as e:
                logger.error(f"Error saving conversation memory: {e}")
                raise

    async def generate_question_async(self, name, description=None):
        """Generate a discussion question about the sub-thought."""
        prompt = (
            f"Based on the description: {description}, generate an open-ended question about '{name}'."
            if description
            else f"Generate an open-ended question about '{name}'."
        )

        try:
            response_model = await self.call_llm_async(prompt, self.llm_provider, "str")

            question = (
                response_model.content.strip()
            )  # Extract text from TextResponseModel

            logger.info(f"Generated question: {response_model.content.strip()}")

            return response_model.content.strip()
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return "What are your thoughts on this topic?"

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
            response_model: TextResponse = await self.call_llm_async(
                prompt=prompt,
                llm_provider=llm_provider,
                expected_res_type="str",
            )
        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            return "Could you elaborate further?"

        return response_model.content.strip()

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
                response_model = await call_openai_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.client,
                )

            elif llm_provider == "claude":
                response_model = await call_claude_api_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    client=self.client,
                )
            elif llm_provider == "llama3":
                response_model = await call_llama3_async(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type=expected_res_type,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")

            logger.info(f"Response data type: {type(response_model)}\n{response_model}")

            return response_model
        except Exception as e:
            logger.error(f"Error calling LLM '{llm_provider}': {e}")
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
