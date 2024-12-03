"""
Filename: state_transition_machine.

The module manages conversational turns of conversations of other agents.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Union
from enum import Enum
import logging
from datetime import datetime
import logging
import logging_config


# Set up logger
logger = logging.getLogger(__name__)


class TopicState(Enum):
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    TRANSITIONING = "transitioning"


@dataclass
class ConversationMetrics:
    redundancy_score: float = 0.0
    coverage_score: float = 0.0
    new_info_score: float = 0.0
    exchange_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    def is_exhausted(self, threshold: dict) -> bool:
        return (
            self.redundancy_score > threshold["redundancy"]
            and self.coverage_score > threshold["coverage"]
            and self.new_info_score < threshold["new_info"]
        )


class TopicExhaustionService:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the service with thresholds for redundancy, coverage, and new info scores.

        Args:
            thresholds (Optional[Dict[str, float]]): Threshold values for exhaustion.
        """
        self.logger = logging.getLogger(__name__)
        self.thresholds = thresholds or {
            "redundancy": 0.7,
            "coverage": 0.8,
            "new_info": 0.2,
        }
        self.reset()

    def reset(self):
        """Reset the service state for a new topic."""
        self.keywords: Set[str] = set()
        self.previous_responses: List[str] = []
        self.metrics = ConversationMetrics()
        self.scoped_logs: List[Dict[str, str]] = (
            []
        )  # To store scoped logs for this topic

    def set_scoped_logs(self, scoped_logs: List[Dict[str, str]]) -> None:
        """
        Set scoped logs for the current topic.

        Args:
            scoped_logs (List[Dict[str, str]]): Scoped logs for the current sub-thought.
        """
        self.scoped_logs = scoped_logs
        self.previous_responses = [
            log["message"] for log in self.scoped_logs if log["role"] == "user"
        ]

    def is_topic_exhausted(
        self, question: str, answer: str
    ) -> Dict[str, Union[bool, float]]:
        """
        Analyze if the current topic is exhausted based on the latest exchange.

        Args:
            question (str): The latest question asked.
            answer (str): The user's latest response.

        Returns:
            Dict[str, Union[bool, float]]: Dictionary containing the exhaustion
            status and each score.
        """
        # Update metrics
        self.metrics.exchange_count += 1
        self.metrics.last_update = datetime.now()

        # Calculate scores
        self.metrics.redundancy_score = self._calculate_redundancy(answer)
        self.metrics.coverage_score = self._calculate_coverage(question, answer)
        self.metrics.new_info_score = self._calculate_new_info(answer)

        # Log metrics for monitoring
        self.logger.debug(
            f"Exchange {self.metrics.exchange_count} Metrics: "
            f"Redundancy={self.metrics.redundancy_score:.2f}, "
            f"Coverage={self.metrics.coverage_score:.2f}, "
            f"New Info={self.metrics.new_info_score:.2f}"
        )

        exhausted = self.metrics.is_exhausted(self.thresholds)
        return {
            "is_exhausted": exhausted,
            "redundancy_score": self.metrics.redundancy_score,
            "coverage_score": self.metrics.coverage_score,
            "new_info_score": self.metrics.new_info_score,
        }

    def _calculate_redundancy(self, answer: str) -> float:
        """
        Calculate redundancy based on overlap with all previous user responses.
        """
        if not self.previous_responses:
            self.previous_responses.append(answer)
            return 0.0

        current_words = set(answer.lower().split())
        total_overlap = 0

        for prev_response in self.previous_responses:
            prev_words = set(prev_response.lower().split())
            overlap = len(current_words.intersection(prev_words))
            total_overlap += overlap / len(current_words) if current_words else 0

        self.previous_responses.append(answer)
        return total_overlap / len(self.previous_responses)

    def _calculate_coverage(self, question: str, answer: str) -> float:
        """
        Calculate how well the answer covers the question and previous agent questions.
        """
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Combine all agent questions from scoped logs for broader coverage evaluation
        agent_questions = " ".join(
            log["message"] for log in self.scoped_logs if log["role"] == "agent"
        )
        agent_words = set(agent_questions.lower().split())

        overlap = len(answer_words.intersection(agent_words.union(question_words)))
        return overlap / len(agent_words.union(question_words)) if agent_words else 0

    def _calculate_new_info(self, answer: str) -> float:
        """
        Calculate the amount of new information in the response based on all logs.
        """
        words = set(answer.lower().split())
        new_words = words - self.keywords
        self.keywords.update(words)

        return len(new_words) / len(words) if words else 0
