"""Pydantic models to validate user_states in the interview conversation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import logging_config

# Set logger
logger = logging.getLogger(__name__)


class SessionMetadata(BaseModel):
    total_interactions: int = Field(
        0, description="Total number of interactions in the session"
    )
    average_response_time: Optional[float] = Field(
        None, description="Average response time per interaction in seconds"
    )


class UserSessionState(BaseModel):
    thought_index: int = Field(
        0, description="Current thought index in the conversation"
    )
    sub_thought_index: int = Field(
        0, description="Current sub-thought index within the thought"
    )
    llm_provider: str = Field(
        ..., description="LLM provider being used (e.g., 'openai', 'anthropic')"
    )
    model_id: str = Field(..., description="Model ID of the LLM (e.g., 'gpt-4-turbo')")
    skills: Optional[List[str]] = Field(
        default_factory=list, description="List of user skills or expertise"
    )
    session_metadata: Optional[SessionMetadata] = Field(
        default_factory=SessionMetadata, description="Session-specific metadata"
    )


# Top level model
class UserSession(BaseModel):
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
    session_state: UserSessionState = Field(
        ..., description="Current session state for the user"
    )
