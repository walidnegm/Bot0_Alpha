import json
import logging
from typing import Dict
from pydantic import BaseModel, Field, field_validator, ValidationError, ValidationInfo

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define the required criteria keys
CRITERIA_KEYS = {"relevance", "correctness", "specificity", "clarity"}


class EvaluationCriteria(BaseModel):
    """
    Represents the evaluation criteria and explanations.

    *Example:
    ```json
    {
        "criteria": {
            "relevance": 5,
            "correctness": 4,
            "specificity": 5,
            "clarity": 5
        },
        "explanations": {
            "relevance": "The answer directly addresses the question with comprehensive details.",
            "correctness": "All factual information is accurate and well-supported.",
            "specificity": "The answer provides in-depth explanations with precise details.",
            "clarity": "The response is well-structured, clear, and easy to understand."
        }
    }
    ```
    """

    criteria: Dict[str, int] = Field(..., min_items=4, max_items=4)
    explanations: Dict[str, str] = Field(..., min_items=4, max_items=4)
    total_score: float

    @field_validator("criteria")
    def validate_criteria(cls, v):
        logger.debug(f"Validating criteria: type(v)={type(v)}, value={v}")

        # Validate if it's a dictionary
        if not isinstance(v, dict):
            raise TypeError(f"Expected a dictionary for criteria, but got {type(v)}.")

        logger.debug(f"Validating criteria: {v}")

        # Step 1: Ensure the keys in `criteria` match the required `CRITERIA_KEYS`
        if set(v.keys()) != CRITERIA_KEYS:
            missing = CRITERIA_KEYS - set(v.keys())
            extra = set(v.keys()) - CRITERIA_KEYS
            error_msg = ""
            if missing:
                error_msg += f"Missing criteria keys: {missing}. "
            if extra:
                error_msg += f"Unexpected criteria keys: {extra}."
            raise ValueError(error_msg.strip())

        # Step 2: Ensure each score in `criteria` is valid
        for key, score in v.items():
            if not isinstance(score, int):
                raise TypeError(f"Score for '{key}' must be an integer.")
            if not 1 <= score <= 5:
                raise ValueError(
                    f"Score for '{key}' must be between 1 and 5. Got {score}."
                )
        return v

    @field_validator("explanations")
    def validate_explanations(cls, v):
        logger.debug(f"Validating explanations: type(v)={type(v)}, value={v}")

        if not isinstance(v, dict):
            raise TypeError(
                f"Expected a dictionary for explanations, but got {type(v)}."
            )

        if set(v.keys()) != CRITERIA_KEYS:
            missing = CRITERIA_KEYS - set(v.keys())
            extra = set(v.keys()) - CRITERIA_KEYS
            error_msg = ""
            if missing:
                error_msg += f"Missing explanation keys: {missing}. "
            if extra:
                error_msg += f"Unexpected explanation keys: {extra}."
            raise ValueError(error_msg.strip())

        for key, explanation in v.items():
            if not isinstance(explanation, str):
                raise TypeError(f"Explanation for '{key}' must be a string.")
            word_count = len(explanation.split())
            if word_count > 50:
                raise ValueError(
                    f"Explanation for '{key}' exceeds 50 words (currently {word_count} words)."
                )
        return v

    # @field_validator("total_score", mode="after")
    # def validate_total_score(cls, v, values):
    #     logger.debug(f"Validating total_score: {v}")
    #     # criteria = values.get("criteria", {})
    #     criteria = values["criteria"] if "criteria" in values else None

    #     # Extra validation: check for if it's dict
    #     if not criteria or not isinstance(criteria, dict):
    #         raise ValueError(
    #             "Criteria scores are missing, cannot validate total_score."
    #         )

    #     expected_total = sum(criteria.values()) / len(criteria)
    #     logger.debug(
    #         f"Expected total score: {expected_total:.2f}, Provided total score: {v}"
    #     )
    #     if abs(v - expected_total) > 0.01:
    #         raise ValueError(
    #             f"Total score {v} does not match the average of criteria scores {expected_total:.2f}."
    #         )
    #     return v


class QuestionAnswerPair(BaseModel):
    """
    Represents a pair of question and answer.

    **Example:**
    ```json
    {
        "question": "What are the main differences between microcontrollers and microprocessors?",
        "answer": "Microcontrollers integrate memory, I/O peripherals, and a CPU on a single chip, \
            whereas microprocessors focus on processing tasks and rely on external memory \
                and peripherals."
    }
    ```
    """

    question: str
    answer: str

    @field_validator("question", "answer")
    def validate_non_empty(cls, v, info: ValidationInfo):
        field_name = info.field_name  # Adjust to access the correct field name
        logger.debug(f"Validating {field_name}: '{v}'")
        if not v:
            raise ValueError(f"{field_name} cannot be empty")
        return v


class EvaluationJSONModel(BaseModel):
    """
    Represents the evaluation response in JSON format.

    **Example Structure:**
    ```json
    {
        "evaluation": {
            "criteria": {
                "relevance": 5,
                "correctness": 4,
                "specificity": 5,
                "clarity": 5
            },
            "explanations": {
                "relevance": "The answer directly addresses the question with comprehensive details.",
                "correctness": "All factual information is accurate and well-supported.",
                "specificity": "The answer provides in-depth explanations with precise details.",
                "clarity": "The response is well-structured, clear, and easy to understand."
            },
            "total_score": 4.75
        }
    }
    ```
    """

    evaluation: EvaluationCriteria

    @field_validator("evaluation")
    def validate_evaluation(cls, v):
        # This validator can be used for additional top-level validations if needed
        logger.debug(f"Validating evaluation: {v}")
        return v
