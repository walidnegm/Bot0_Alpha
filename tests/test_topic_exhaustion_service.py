import sys
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import pytest

# Setup PYTHONPATH for tests
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from agents.state_transition_machine import TopicExhaustionService, ConversationMetrics

import pytest


@pytest.fixture
def topic_exhaustion_service():
    """Fixture to create a fresh instance of TopicExhaustionService."""
    return TopicExhaustionService()


def generate_mock_log(role, message, timestamp=None):
    """Utility to generate a mock log entry."""
    return {
        "message_id": str(uuid.uuid4()),
        "timestamp": (timestamp or datetime.now()).isoformat(),
        "role": role,
        "message": message,
    }


def test_set_scoped_logs(topic_exhaustion_service):
    """Test that set_scoped_logs correctly initializes logs and user responses."""
    scoped_logs = [
        generate_mock_log("agent", "What are the benefits of AI?"),
        generate_mock_log("user", "It improves efficiency."),
        generate_mock_log("user", "AI automates repetitive tasks."),
    ]
    topic_exhaustion_service.set_scoped_logs(scoped_logs)

    assert topic_exhaustion_service.scoped_logs == scoped_logs
    assert topic_exhaustion_service.previous_responses == [
        "It improves efficiency.",
        "AI automates repetitive tasks.",
    ]


def test_calculate_redundancy(topic_exhaustion_service):
    """Test redundancy calculation with mocked user responses."""
    topic_exhaustion_service.previous_responses = [
        "AI helps with efficiency.",
        "AI automates repetitive tasks.",
    ]
    redundancy_score = topic_exhaustion_service._calculate_redundancy(
        "AI makes tasks more efficient."
    )

    # Expect some overlap since "efficiency" and "tasks" are common
    assert 0.0 < redundancy_score < 1.0


def test_calculate_new_info(topic_exhaustion_service):
    """Test new information metric calculation."""
    topic_exhaustion_service.keywords = {"ai", "tasks"}
    new_info_score = topic_exhaustion_service._calculate_new_info(
        "AI improves efficiency with new methods."
    )

    # "efficiency" and "methods" are new keywords
    assert 0.0 < new_info_score <= 1.0


def test_is_topic_exhausted(topic_exhaustion_service):
    """Test whether the topic is marked as exhausted based on thresholds."""
    topic_exhaustion_service.thresholds = {
        "redundancy": 0.5,
        "new_info": 0.3,
    }

    now = datetime.now()
    scoped_logs = [
        generate_mock_log(
            "agent", "What are the benefits of AI?", now - timedelta(minutes=2)
        ),
        generate_mock_log(
            "user", "It improves efficiency.", now - timedelta(minutes=1)
        ),
    ]
    topic_exhaustion_service.set_scoped_logs(scoped_logs)

    user_response = "AI automates tasks and improves efficiency."
    result = topic_exhaustion_service.is_topic_exhausted(user_response)

    assert isinstance(result, dict)
    assert "is_exhausted" in result
    assert result["redundancy_score"] >= 0.0
    assert result["new_info_score"] >= 0.0

    # Check exhaustion status
    assert result["is_exhausted"] is False  # Adjust based on mock thresholds
