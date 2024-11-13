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
from utils.llm_api_utils import get_claude_api_key, get_openai_api_key
from utils.llm_api_utils_async import (
    call_openai_api_async,
    call_claude_api_async,
    call_llama3_async,
)
from models.llm_response_base_models import TextResponse


# Setup logger
logger = logging.getLogger(__name__)


class FacilitatorAgentAsync:
    """
    FacilitatorAgentAsync Class

    This class facilitates an interactive educational conversation with a user about a main concept and its sub-concepts.
    It generates questions, evaluates user responses, and transitions to the next sub-concept when the current topic is exhausted.

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
        data: Dict[str, Any],
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
        self.concept = data["concept"]
        self.sub_concepts = data["sub_concepts"]
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client

        self.conversation_memory = []
        self.current_sub_concept_index = 0

        # Log instantiation
        logger.info(
            f"FacilitatorAgentAsync instantiated with provider '{llm_provider}', model '{model_id}', "
            f"temperature {temperature}, max_tokens {max_tokens}."
        )

        # Initialize API client
        if self.llm_provider == "openai" and not self.client:
            api_key = get_openai_api_key()
            self.client = OpenAI(api_key=api_key)

            logger.info(f"{self.llm_provider} API instantiated.")

        elif self.llm_provider == "claude" and not self.client:
            api_key = get_claude_api_key()
            self.client = Anthropic(api_key=api_key)

            logger.info(f"{self.llm_provider} API instantiated.")

        elif self.llm_provider == "llama3":
            logger.info("Using LLaMA3 model.")
        elif not self.client:
            raise ValueError(f"Unsupported model: {self.llm_provider}")

        # Initialize EvaluationAgent and TopicExhaustionService
        self.evaluation_agent = EvaluatorAgentAsync()
        self.topic_exhaustion_service = TopicExhaustionService()

    async def start_conversation(self) -> None:
        """
        Start the conversation by introducing the main concept and initiating the discussion of sub-concepts.
        """
        print(f"Today, we'll discuss about {self.concept}.")
        await self.discuss_next_sub_concept()

    async def discuss_next_sub_concept(self) -> None:
        """
        Manage the sequential discussion of sub-concepts, transitioning when a topic is exhausted.
        """
        while self.current_sub_concept_index < len(self.sub_concepts):
            sub_concept: Dict[str, Any] = self.sub_concepts[
                self.current_sub_concept_index
            ]
            await self.handle_focused_discussion(sub_concept)
            self.current_sub_concept_index += 1

        print("We've covered all the topics. Thank you for the discussion!")

    async def handle_focused_discussion(self, sub_concept: Dict[str, Any]) -> None:
        """
        Conduct a focused, multi-turn discussion on a specific sub-concept, transitioning when the topic is exhausted.

        Args:
            sub_concept (Dict[str, Any]): A dictionary containing 'name' and 'description' of the sub-concept.
        """
        exchanges: List[Dict[str, str]] = []
        sub_concept_name: str = sub_concept["name"]
        sub_concept_description: str = sub_concept["description"]

        print(f"\nLet's talk about {sub_concept_name}.")
        question: str = await self.generate_question(
            sub_concept_name, sub_concept_description
        )
        print(f"Agent: {question}")
        exchange: Dict[str, str] = {"agent": question}

        self.topic_exhaustion_service.reset()

        while True:
            user_response: str = await aioconsole.ainput("You: ")
            exchange["user"] = user_response

            # Evaluate user's response
            try:
                evaluation: Dict[str, Any] = await self.evaluation_agent.evaluate_async(
                    correct_answer=sub_concept_description, user_response=user_response
                )
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                print(
                    "An error occurred during evaluation. Let's move on to the next topic."
                )
                break

            # Check for topic exhaustion and get scores
            exhaustion_result = self.topic_exhaustion_service.is_topic_exhausted(
                question, user_response
            )

            if exhaustion_result["is_exhausted"]:
                print(
                    f"Agent: Great! We've covered the topic of {sub_concept_name} sufficiently "
                    f"(Redundancy={exhaustion_result['redundancy_score']:.2f}, "
                    f"Coverage={exhaustion_result['coverage_score']:.2f}, "
                    f"New Info={exhaustion_result['new_info_score']:.2f})."
                )
                exchanges.append(exchange)
                break

            if evaluation["is_correct"] or evaluation["should_move_on"]:
                agent_reply: str = await self.generate_conciliatory_reply(evaluation)
                print(f"Agent: {agent_reply}")
                exchanges.append(exchange)
                break
            else:
                agent_reply: str = await self.generate_followup_question(evaluation)
                print(f"Agent: {agent_reply}")
                exchanges.append(exchange)
                exchange = {"agent": agent_reply}

        self.conversation_memory.append(
            {"sub_concept": sub_concept_name, "exchanges": exchanges}
        )

    async def generate_question(
        self,
        sub_concept_name: str,
        sub_concept_description: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> str:
        # Generate an open-ended question about the sub-concept
        if sub_concept_description:
            prompt: str = (
                f"Based on the following description:\n\n{sub_concept_description}\n\n"
                f"Generate one open-ended question that encourages discussion about this concept. "
                f"Provide only the question and no additional text."
            )
        else:
            prompt: str = (
                f"Generate one clear, open-ended question about '{sub_concept_name}'. "
                f"The question should encourage discussion and cannot be answered with a simple 'yes' or 'no'. "
                f"Provide only the question and no additional text."
            )

        if llm_provider is None:
            llm_provider = self.llm_provider

        try:
            response: TextResponse = await self.call_llm(
                prompt=prompt,
                llm_provider=llm_provider,
                expected_res_type="str",
            )
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return "Could you tell me more about this topic?"

        return response.content.strip()

    async def generate_conciliatory_reply(self, evaluation: Dict[str, Any]) -> str:
        # Generate a conciliatory reply based on the evaluation
        return (
            "That's correct! Great job."
            if evaluation["is_correct"]
            else "Thank you for your thoughts. Let's move on to the next topic."
        )

    async def generate_followup_question(
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
            response: TextResponse = await self.call_llm(
                prompt=prompt,
                llm_provider=llm_provider,
                expected_res_type="str",
            )
        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            return "Could you elaborate further?"

        return response.content.strip()

    async def call_llm(
        self,
        prompt: str,
        llm_provider: str,
        expected_res_type: str,
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

    async def save_conversation_memory(
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

    await agent.start_conversation()
    await agent.save_conversation_memory()


# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"An error occurred: {e}")
