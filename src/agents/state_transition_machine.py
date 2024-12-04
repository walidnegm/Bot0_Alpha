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


# class TopicState(Enum):
#     ACTIVE = "active"
#     EXHAUSTED = "exhausted"
#     TRANSITIONING = "transitioning"


@dataclass
class ConversationMetrics:
    redundancy_score: float = 0.0
    new_info_score: float = 0.0
    exchange_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    def is_exhausted(self, threshold: dict) -> bool:
        return (
            self.redundancy_score > threshold["redundancy"]
            and self.new_info_score < threshold["new_info"]
        )


class TopicExhaustionService:
    """
    A service to evaluate whether a conversational topic is exhausted based on
    redundancy and new information metrics.

    This service tracks user responses during a discussion and analyzes whether
    further follow-up questions are likely to yield meaningful new insights.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
                thresholds (Optional[Dict[str, float]]): A dictionary of thresholds used
                    to determine exhaustion. Defaults are:
                    - "redundancy": The threshold above which the conversation is
                    considered repetitive (default: 0.7).
                    - "new_info": The threshold below which the conversation is
                    considered uninformative (default: 0.2).

            Attributes:
                - thresholds (dict): Dictionary storing the thresholds for redundancy and
                new information.
                - scoped_logs (List[Dict[str, str]]): Stores all user and agent logs
                for the current topic.
                - previous_responses (List[str]): List of all user messages logged so far.
                - keywords (Set[str]): Tracks unique words used during the conversation.
                - metrics (ConversationMetrics): Tracks the scores and state of
                the current topic.

        Metrics Used
        1. Redundancy: Evaluates how much of the current user response overlaps with all
        previous responses for that discussion (sub_thought discussion).
        - Uses the entire conversation log: Specifically, all log["message"] entries
        where log["role"] == "user".

        2. New Information: Measures how much new information the current user response
        introduces, relative to all prior exchanges.
        - Uses the entire conversation log

        """
        self.logger = logging.getLogger(__name__)
        self.thresholds = thresholds or {
            "redundancy": 0.7,
            "new_info": 0.2,
        }
        self.reset()

    def reset(self):
        """
        Reset the service to its initial state, clearing all logs, metrics, and keywords.

        This method is used to start fresh analysis for a new topic or sub-thought.
        """
        self.keywords: Set[str] = set()
        self.previous_responses: List[str] = []
        self.metrics = ConversationMetrics()
        self.scoped_logs: List[Dict[str, str]] = (
            []
        )  # To store scoped logs for this topic

    def set_scoped_logs(self, scoped_logs: List[Dict[str, str]]) -> None:
        """
        Set the scoped logs for the current topic. Scoped logs include all agent
        and user exchanges for the specific topic under discussion.

        Args:
            scoped_logs (List[Dict[str, str]]): A list of log dictionaries. Each log
                contains the following fields:
                - "message_id" (str): Unique identifier for the log entry.
                - "timestamp" (str): ISO 8601 formatted timestamp of the log entry.
                - "role" (str): Role of the participant ("agent" or "user").
                - "message" (str): The content of the message.

        Effects:
            - Updates the `scoped_logs` attribute with the provided logs.
            - Extracts and stores all user responses in `previous_responses`.
        """
        if not isinstance(scoped_logs, list):
            self.logger.error("Scoped logs must be a list of dictionaries.")
            raise ValueError("Scoped logs must be a list of dictionaries.")
        if not all(isinstance(log, dict) for log in scoped_logs):
            self.logger.error("Each log entry in scoped logs must be a dictionary.")
            raise ValueError("Each log entry in scoped logs must be a dictionary.")

        self.scoped_logs = scoped_logs
        self.previous_responses = [
            log["message"] for log in self.scoped_logs if log["role"] == "user"
        ]

    def is_topic_exhausted(self, answer: str) -> Dict[str, Union[bool, float]]:
        """
        Evaluate whether the current topic is exhausted based on redundancy and
        new information metrics.

        Args:
            answer (str): The user's latest response to evaluate.

        Returns:
            Dict[str, Union[bool, float]]:
                A dictionary containing:
                - "is_exhausted" (bool): Whether the topic is exhausted.
                - "redundancy_score" (float):
                Measure of repetition in user responses (0.0 to 1.0).
                - "new_info_score" (float):
                Measure of unique information introduced in the latest response
                (0.0 to 1.0).

        Evaluation Criteria:
            - A topic is considered "exhausted" if:
                - Redundancy exceeds the "redundancy" threshold.
                - New information falls below the "new_info" threshold.

        Example Output:
            {
                "is_exhausted": True,
                "redundancy_score": 0.8,
                "new_info_score": 0.1,
            }
        """
        try:
            # Update metrics
            self.metrics.exchange_count += 1
            self.metrics.last_update = datetime.now()

            # Calculate scores
            self.metrics.redundancy_score = self._calculate_redundancy(answer)
            self.metrics.new_info_score = self._calculate_new_info(answer)

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}", exc_info=True)
            raise RuntimeError("Failed to calculate topic exhaustion metrics.") from e

        # Log metrics for monitoring
        self.logger.debug(
            f"Exchange {self.metrics.exchange_count} Metrics: "
            f"Redundancy={self.metrics.redundancy_score:.2f}, "
            f"New Info={self.metrics.new_info_score:.2f}"
        )

        try:
            # *Determine exhaustion status
            exhausted = self.metrics.is_exhausted(self.thresholds)
        except KeyError as e:
            self.logger.error(f"Missing threshold key: {e}", exc_info=True)
            raise ValueError(f"Thresholds must include keys: {e}") from e

        return {
            "is_exhausted": exhausted,
            "redundancy_score": self.metrics.redundancy_score,
            "new_info_score": self.metrics.new_info_score,
        }

    def _calculate_redundancy(self, answer: str) -> float:
        """
        Calculate redundancy based on the overlap between the user's latest response
        and all previous user responses.

        Args:
            answer (str): The user's latest response.

        Returns:
            float: Redundancy score (0.0 to 1.0). A higher score indicates more repetition
            in user responses.

        Logic:
            - Splits the latest response into words.
            - Compares the words in the latest response to all previous user responses.
            - Calculates the average overlap across all prior responses.

        Example:
            Previous Responses: ["AI improves efficiency", "AI automates tasks"]
            Latest Response: "AI improves automation"
            Redundancy Score: Calculated based on shared words {"AI", "improves"}.
        """
        try:
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

        except Exception as e:
            self.logger.error(f"Error calculating resundancy: {e}", exc_info=True)
            raise

    def _calculate_new_info(self, answer: str) -> float:
        """
        Calculate the amount of new information introduced in the user's latest response.

        Args:
            answer (str): The user's latest response.

        Returns:
            float: New information score (0.0 to 1.0). A higher score indicates more
            unique content relative to prior exchanges.

        Logic:
            - Splits the latest response into words.
            - Identifies words not previously seen in the conversation.
            - Computes the proportion of new words in the latest response.

        Example:
            Keywords Seen So Far: {"AI", "tasks"}
            Latest Response: "AI improves efficiency"
            New Info Score: Calculated based on new words {"improves", "efficiency"}.
        """
        words = set(answer.lower().split())
        new_words = words - self.keywords
        self.keywords.update(words)

        return len(new_words) / len(words) if words else 0
