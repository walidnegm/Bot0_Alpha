"""Pydantic models to validate user_states in the interview conversation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import logging_config
from datetime import datetime


from models.evaluation_models import EvaluationCriteria

# Set logger
logger = logging.getLogger(__name__)


class SessionMetadata(BaseModel):
    total_interactions: int = Field(
        0, description="Total number of interactions in the session"
    )
    average_response_time: Optional[float] = Field(
        None, description="Average response time per interaction in seconds"
    )


class UserState(BaseModel):
    """
    Represents the state of a user in a conversational agent system.

    Attributes:
        thought_index (int): Current thought index in the conversation. Default is 0.
        sub_thought_index (int): Current sub-thought index within the thought. Default is 0.
        current_question (Optional[str]): The current question being discussed. Default is None.
        last_updated (datetime): Timestamp of the last update. Default is the current time.
        skills (Optional[List[str]]): List of user skills or expertise. Default is an empty list.
        session_metadata (Optional[SessionMetadata]): Metadata related to the session.
            Defaults to an instance of `SessionMetadata` with `session_id='default_session'`.

    **Example:**

    >>> Example:
    # Create a default user state
    user1 = UserState()
    print(user1)

    # Output:
    # thought_index=0
    # sub_thought_index=0
    # current_question=None
    # last_updated=datetime.datetime(2024, 11, 24, 17, 0, 0, 123456)
    # skills=[]
    # session_metadata=SessionMetadata(session_id='default_session')

    # Create a user state with a custom question
    user2 = UserState(current_question="What is Python?")
    print(user2)

    # Output:
    # thought_index=0
    # sub_thought_index=0
    # current_question="What is Python?"
    # last_updated=datetime.datetime(2024, 11, 24, 17, 1, 0, 123456)
    # skills=[]
    # session_metadata=SessionMetadata(session_id='default_session')

    # Modify skills for one user
    user1.skills.append("Python")
    print(user1.skills)  # ['Python']
    print(user2.skills)  # []  (remains unchanged)

    >>> JSON output (raw output) example:
    {
        "user_123": {
            "thought_index": 1,
            "sub_thought_index": 2,
            "current_question": "What is the importance of embedded systems in aerospace?",
            "last_updated": "2024-11-24T19:30:00.123456",
            "skills": ["Python", "Systems Design"],
            "session_metadata": {
                "session_id": "default_session"
            },
            "current_evaluation": {
                "criteria": {
                    "relevance": 4,
                    "correctness": 5,
                    "specificity": 3,
                    "clarity": 4
                },
                "explanations": {
                    "relevance": "The answer aligns with the question but lacks direct examples specific to embedded systems.",
                    "correctness": "The answer correctly identifies relevant methodologies and tools without factual errors.",
                    "specificity": "Mentions general methodologies and tools but lacks detail on how they apply to embedded systems.",
                    "clarity": "The answer is clear and well-structured, though it could integrate more specific examples."
                },
                "total_score": 4.0
            },
            "evaluations": [
                {
                    "criteria": {
                        "relevance": 5,
                        "correctness": 4,
                        "specificity": 4,
                        "clarity": 5
                    },
                    "explanations": {
                        "relevance": "The response directly addressed the question with comprehensive details.",
                        "correctness": "Most methodologies mentioned were accurate, though one was less applicable.",
                        "specificity": "The answer provided specific examples and steps for implementation.",
                        "clarity": "The explanation was clear and concise."
                    },
                    "total_score": 4.5
                },
                {
                    "criteria": {
                        "relevance": 3,
                        "correctness": 3,
                        "specificity": 2,
                        "clarity": 4
                    },
                    "explanations": {
                        "relevance": "The answer partially addressed the question but missed key elements.",
                        "correctness": "Some information was accurate, but some points were misleading.",
                        "specificity": "The response lacked depth and specific examples.",
                        "clarity": "The answer was generally clear but could benefit from better organization."
                    },
                    "total_score": 3.0
                }
            ]
        }
    }

    """

    thought_index: int = Field(
        0, description="Current thought index in the conversation"
    )
    sub_thought_index: int = Field(
        0, description="Current sub-thought index within the thought"
    )
    current_question: Optional[str] = Field(
        None, description="Current questin being discussed"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Timestamp of the last update"
    )
    skills: Optional[List[str]] = Field(
        default_factory=list, description="List of user skills or expertise"
    )
    session_metadata: Optional[SessionMetadata] = Field(
        default_factory=SessionMetadata, description="Session-specific metadata"
    )

    # Re-add evaluation-related fields
    current_evaluation: Optional[EvaluationCriteria] = Field(
        None, description="Evaluation for the current question"
    )
    evaluations: List[EvaluationCriteria] = Field(
        default_factory=list, description="List of past evaluations"
    )  # *List of dicts rather than dicts of dicts b/c order matters + allow it to have duplicate values


# Top level model
class Session(BaseModel):
    """
    Pydantic model to validate top-level data structure for interview conversations.

    Example `user_states` structure (model attributes):
        user_states = {
            "user123": UserSession(
                user_id="user123",
                session_state=UserSessionState(
                    thought_index=0,
                    sub_thought_index=0,
                    llm_provider="openai",
                    model_id="gpt-4-turbo",
                    skills=["Intermediate Machine Learning", "Basic NLP"],
                    session_metadata=SessionMetadata(
                        total_interactions=3, average_response_time=1.5
                    )
                )
            ),
            "user456": UserSession(
                user_id="user456",
                session_state=UserSessionState(
                    thought_index=1,
                    sub_thought_index=2,
                    llm_provider="anthropic",
                    model_id="claude-1",
                    skills=["Advanced Data Science", "AI Research"],
                    session_metadata=SessionMetadata(
                        total_interactions=5, average_response_time=1.8
                    )
                )
            )
        }

    Example `user_states` data structure (actual output):
        user_states = {
            "user123": {
                "user_id": "user123",
                "session_state": {
                    "thought_index": 0,
                    "sub_thought_index": 0,
                    "llm_provider": "openai",
                    "model_id": "gpt-4-turbo",
                    "skills": ["Intermediate Machine Learning", "Basic NLP"],
                    "session_metadata": {
                        "total_interactions": 3,
                        "average_response_time": 1.5
                    }
                }
            },
            "user456": {
                "user_id": "user456",
                "session_state": {
                    "thought_index": 1,
                    "sub_thought_index": 2,
                    "llm_provider": "anthropic",
                    "model_id": "claude-1",
                    "skills": ["Advanced Data Science", "AI Research"],
                    "session_metadata": {
                        "total_interactions": 5,
                        "average_response_time": 1.8
                    }
                }
            }
        }
    """

    user_id: str = Field(..., description="Unique identifier for the user")
    session_state: UserState = Field(
        ..., description="Current session state for the user"
    )
