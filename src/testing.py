from agents.evaluator_agent_async import EvaluatorAgentAsync
from agents.facilitator_agent_async import FacilitatorAgentAsync
from models.evaluation_models import EvaluationJSONModel, QuestionAnswerPair
import asyncio
from prompts.evaluation_prompt_templates import QUESTION_ANSWER_EVAL_PROMPT


question = "How can the systematic identification and cataloging of system resources, such as memory and processing power, impact the optimization and efficiency of a computing system?"
answer = "Requirements Analysis, System Modeling, Performance Analysis. Tools: System Simulation Tools, Performance Analysis Tools, Debugging and Profiling Tools, Resource Monitoring Tools."
idea = "embedded software development"
thought = "System Resources and Constraints"


prompt = QUESTION_ANSWER_EVAL_PROMPT.format(
    question=question, answer=answer, idea=idea, thought=thought
)

print(prompt)
