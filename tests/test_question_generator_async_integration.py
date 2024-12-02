import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
import os
import sys

# Setup PYTHONPATH for tests
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from agents.question_generator_async import QuestionGeneratorAsync
from models.llm_response_models import TextResponse

print("sys.path:", sys.path)

# Dummy data for testing
DUMMY_TOPIC_NAME = "climate change"
DUMMY_CONTEXT = "Its impact on global food security."
DUMMY_COMPLEXITY = "advanced"
DUMMY_EVALUATION_TEXT = (
    "The initial response was thorough but lacked focus on renewable energy."
)
DUMMY_CONTEXT_TEXT = (
    "Consider the importance of renewable energy in combating climate change."
)
DUMMY_EXPECTED_INITIAL_PROMPT = f"""
Create a complex and requiring deep analytical thinking open-ended discussion question about '{DUMMY_TOPIC_NAME}'. 
Incorporate the following contextual details: {DUMMY_CONTEXT}
Ensure the question:
- Encourages critical thinking
- Is clear and specific
- Invites multiple perspectives
"""
DUMMY_EXPECTED_FOLLOWUP_PROMPT = f"""
Given the following evaluation and context (if any):
-Evaluation: {DUMMY_EVALUATION_TEXT}
- Context: Incorporate the following contextual details: {DUMMY_CONTEXT_TEXT}

Generate an insightful follow-up question that:
- Builds upon the previous discussion
- Probes deeper into the underlying concepts
- Encourages further critical analysis
- Is precise and thought-provoking
"""


@pytest.fixture
def mock_generator():
    """
    Fixture to provide a mock QuestionGeneratorAsync instance.
    """
    mock_client = AsyncMock()
    generator = QuestionGeneratorAsync(
        llm_provider="openai",
        model_id="gpt-4",
        llm_client=mock_client,
        temperature=0.7,
        max_tokens=1056,
    )
    return generator, mock_client


@pytest.mark.asyncio
@patch(
    "agents.question_generator_async.INITIAL_QUESTION_GENERATION_PROMPT",
    DUMMY_EXPECTED_INITIAL_PROMPT,
)
async def test_generate_initial_question(mock_generator):
    """
    Test the `generate_initial_question` method.
    """
    generator, mock_client = mock_generator
    mock_client.return_value = TextResponse(
        content="What are the most critical challenges to global food security due to \
            climate change?"
    )

    question = await generator.generate_initial_question(
        topic_name=DUMMY_TOPIC_NAME,
        context_text=DUMMY_CONTEXT,
        complexity=DUMMY_COMPLEXITY,
    )

    assert isinstance(question, str)
    mock_client.assert_awaited_once()
    assert "global food security" in question


@pytest.mark.asyncio
@patch(
    "agents.question_generator_async.FOLLOWUP_QUESTION_GENERATION_PROMPT",
    DUMMY_EXPECTED_FOLLOWUP_PROMPT,
)
async def test_generate_followup_question(mock_generator):
    """
    Test the `generate_followup_question` method.
    """
    generator, mock_client = mock_generator
    mock_client.return_value = TextResponse(
        content="How can renewable energy solutions directly address the challenges \
            posed by climate change?"
    )

    followup_question = await generator.generate_followup_question(
        evaluation_text=DUMMY_EVALUATION_TEXT,
        context_text=DUMMY_CONTEXT_TEXT,
    )

    assert isinstance(followup_question, str)
    mock_client.assert_awaited_once()
    assert "renewable energy" in followup_question


@pytest.mark.asyncio
@patch("utils.llm_api_utils_async.call_openai_api_async", autospec=True)
async def test_call_llm_async(mock_call_openai_api_async, mock_generator):
    """
    Test the `call_llm_async` method.
    """
    generator = mock_generator[0]

    # Mock the API response
    mock_response = TextResponse(content="This is a mocked response from the LLM.")
    mock_call_openai_api_async.return_value = mock_response

    # Run the method
    prompt = "Test prompt for LLM."
    response = await generator.call_llm_async(prompt)

    # Assertions
    assert isinstance(response, TextResponse)
    assert response.text == "This is a mocked response from the LLM."
