import asyncio
import os
from dotenv import load_dotenv
from agents.question_generator_async import QuestionGeneratorAsync


# Centralized configuration for testing
EVALUATION_TEXT = """
Evaluation:
- Previous Question: How can the process of identifying and analyzing system resources, such as memory, processing power, and peripheral interfaces, be optimized to enhance system performance and reliability?
- Previous Answer: To optimize system performance and reliability, implement resource monitoring tools and analyze usage patterns to identify bottlenecks. Adjust resource allocation, upgrade or add resources as needed, and ensure regular system maintenance to ensure optimal system performance.

- Relevance: 4/5, the answer addresses system performance optimization but lacks depth in the embedded systems context.
- Correctness: 5/5, the information provided is accurate, with no misleading or incorrect details.
- Specificity: 3/5, The answer is somewhat generic; more detailed strategies or examples could enhance it.
- Clarity: 5/5, The answer is clear, well-structured, and easy to understand, with a logical flow.
- Overall Score: 4.25/5
"""

CONTEXT_TEXT_0 = "embedded software development in aerospace; Safety and Reliability."
CONTEXT_TEXT_1 = "embedded software development in aerospace; Identification and analysis of safety-critical requirements according to standards like DO-178C and ARP4754A. This involves determining the software level (DAL) and establishing compliance criteria for the development process."

DUMMY_CONFIG = {
    "topic_name": "Requirements Analysis and Safety Standards",
    "context_text": "embedded software development in aerospace; Safety and Reliability.",
    "complexity": "advanced",
    "evaluation_text": EVALUATION_TEXT,
    "followup_context_text": CONTEXT_TEXT_1,
    "llm_provider": "openai",
    "model_id": "gpt-4-turbo",
    "temperature": 0.7,
    "max_tokens": 1056,
}

DUMMY_CONFIG_ALT = {
    "topic_name": "Requirements Analysis and Safety Standards",
    "context_text": "embedded software development in aerospace; Safety and Reliability.",
    "complexity": "advanced",
    "evaluation_text": EVALUATION_TEXT,
    "followup_context_text": CONTEXT_TEXT_1,
    "llm_provider": "claude",
    "model_id": "claude-3-5-haiku-20241022",
    "temperature": 0.7,
    "max_tokens": 100,
}
# Choose a configuration (default: DUMMY_CONFIG)
CONFIG_DICT = DUMMY_CONFIG


def create_real_agent(config):
    """
    Create a QuestionGeneratorAsync instance with a real OpenAI client.
    """
    from openai import AsyncOpenAI

    # Load API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Create the OpenAI client
    openai_client = AsyncOpenAI(api_key=openai_api_key)

    # Create the QuestionGeneratorAsync instance
    return QuestionGeneratorAsync(
        llm_provider=config["llm_provider"],
        model_id=config["model_id"],
        llm_client=openai_client,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )


async def test_generate_initial_question(agent, config):
    """
    Test the generate_initial_question method with real API.
    """
    print("\n--- Testing generate_initial_question ---")
    question = await agent.generate_initial_question(
        topic_name=config["topic_name"],
        context_text=config["context_text"],
        complexity=config["complexity"],
    )
    print("Generated Question:", question)


async def test_generate_followup_question(agent, config):
    """
    Test the generate_followup_question method with real API.
    """
    print("\n--- Testing generate_followup_question ---")
    followup_question = await agent.generate_followup_question(
        evaluation_text=config["evaluation_text"],
        context_text=config["followup_context_text"],
    )
    print("Generated Follow-up Question:", followup_question.content)


async def test_call_llm_async(agent):
    """
    Test the call_llm_async method with real API.
    """
    prompt = "What are the implications of AI in healthcare?"

    print("\n--- Testing call_llm_async ---")
    response = await agent.call_llm_async(prompt)
    print("LLM Response:", response.content)


async def main():
    """
    Main testing function to test with real API.
    """
    # Create the agent with a real API client
    agent = create_real_agent(CONFIG_DICT)

    # Run tests
    # await test_generate_initial_question(agent, CONFIG_DICT)
    await test_generate_followup_question(agent, CONFIG_DICT)
    # await test_call_llm_async(agent)


if __name__ == "__main__":
    asyncio.run(main())
