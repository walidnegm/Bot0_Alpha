"""Prompt templates for question and answer evaluation (load into the EvaluatorAgent)"""

QUESTION_ANSWER_EVAL_PROMPT = """
Question: {question}
Answer: {answer}

Additional Context: 
Idea: {idea}
Thought: {thought}

Carefully evaluate the given answer using a rigorous, systematic approach:

Evaluation Criteria:
1. Relevance: Assess how directly and comprehensively the answer addresses the specific question asked, ESPECIALLY in light of the provided context.
2. Correctness: Verify the factual accuracy and absence of misleading or incorrect information.
3. Specificity: Examine the depth, precision, and level of detail in the answer.
4. Clarity: Evaluate the answer's linguistic quality, logical flow, and ease of understanding.

Scoring Guidelines:
- Score Range: 1-5 (1 = Very Poor, 5 = Excellent)
- Consider multiple dimensions within each criterion
- Provide concise, objective explanations
- Explicitly consider the question context in your evaluation

Special Handling for Insufficient Responses:
- If the answer appears to be a minimal, non-informative response that fails to engage with the question, evaluate the response critically.
(examples: "I don't know", "Not sure", etc.), 
- Consider:
  1. Is this a genuine acknowledgment of limited knowledge?
  2. Does the response show any attempt to provide context or partial understanding?
  3. Could the response be expanded or clarified?
- Adjust scores accordingly, potentially lowering scores for relevance, specificity, and clarity if the response is truly uninformative.

Produce a STRICTLY FORMATTED JSON response with the following requirements:
{{
    "evaluation": {{
        "criteria": {{
            "relevance": <integer_score_1_to_5>,
            "correctness": <integer_score_1_to_5>,
            "specificity": <integer_score_1_to_5>,
            "clarity": <integer_score_1_to_5>
        }},
        "explanations": {{
            "relevance": "<precise_explanation_max_50_words>",
            "correctness": "<precise_explanation_max_50_words>", 
            "specificity": "<precise_explanation_max_50_words>",
            "clarity": "<precise_explanation_max_50_words>"
        }},
        "total_score": <calculated_average_of_criteria_scores>
    }}
}}

IMPORTANT CONSTRAINTS:
- Keys in the 'criteria' and 'explanations' objects MUST be UNIQUE.
- JSON MUST be valid and well-formed.
"""


INITIAL_QUESTION_GENERATION_PROMPT = """
Create a {complexity_level} open-ended discussion question about '{topic_name}'. 

{context}

Ensure the question:
- Encourages critical thinking
- Is clear and specific
- Invites multiple perspectives
"""

FOLLOWUP_QUESTION_GENERATION_PROMPT = """
Given the following context, evaluation, and conversation so far:

Context:
- Idea: {idea}
- Main Thought: {main_thought}
- Sub-thought: {sub_thought_description}

Evaluation:
{evaluation_scores_and_explanations}

Conversation Context:
{conversation_context}

Generate an insightful follow-up question that:
- Builds upon the previous discussion
- Probes deeper into the underlying concepts
- Encourages further critical analysis
- Is precise and thought-provoking.
"""


# TODO: old version; delete later
# QUESTION_ANSWER_EVAL_PROMPT = """
# Question: {question}
# Answer: {answer}

# Additional Context:
# Idea: {idea}
# Thought: {thought}

# Carefully evaluate the given answer using a rigorous, systematic approach:

# Evaluation Criteria:
# 1. Relevance: Assess how directly and comprehensively the answer addresses the specific question asked, ESPECIALLY in light of the provided context.
# 2. Correctness: Verify the factual accuracy and absence of misleading or incorrect information.
# 3. Specificity: Examine the depth, precision, and level of detail in the answer.
# 4. Clarity: Evaluate the answer's linguistic quality, logical flow, and ease of understanding.

# Scoring Guidelines:
# - Score Range: 1-5 (1 = Very Poor, 5 = Excellent)
# - Consider multiple dimensions within each criterion
# - Provide concise, objective explanations
# - Explicitly consider the question context in your evaluation

# Produce a STRICTLY FORMATTED JSON response with the following requirements:
# {{
#     "evaluation": {
#         "criteria": {
#             "relevance": <integer_score_1_to_5>,
#             "correctness": <integer_score_1_to_5>,
#             "specificity": <integer_score_1_to_5>,
#             "clarity": <integer_score_1_to_5>
#         },
#         "explanations": {
#             "relevance": "<precise_explanation_max_50_words>",
#             "correctness": "<precise_explanation_max_50_words>",
#             "specificity": "<precise_explanation_max_50_words>",
#             "clarity": "<precise_explanation_max_50_words>"
#         },
#         "total_score": <calculated_average_of_criteria_scores>
#     }
# }}

# IMPORTANT CONSTRAINTS:
# - JSON MUST be PERFECTLY VALID
# - All placeholders MUST be replaced with actual data
# - Explanations must be objective and factual
# - Scores must be integers between 1-5
# """
# TODO Origin recommended prompt from claude;
# TODO it's more thorough than my existing version; update later
# prompt = f"""\
# Question: {qa_pair.question}
# Answer: {qa_pair.answer}

# Carefully evaluate the given answer using a rigorous, systematic approach:

# Evaluation Criteria:
# 1. Relevance: Assess how directly and comprehensively the answer addresses the specific question asked.
# 2. Correctness: Verify the factual accuracy and absence of misleading or incorrect information.
# 3. Specificity: Examine the depth, precision, and level of detail in the answer.
# 4. Clarity: Evaluate the answer's linguistic quality, logical flow, and ease of understanding.

# Scoring Guidelines:
# - Score Range: 1-5 (1 = Very Poor, 5 = Excellent)
# - Consider multiple dimensions within each criterion
# - Provide concise, objective explanations

# Produce a STRICTLY FORMATTED JSON response with the following requirements:
# {{
#     "version": "1.0",
#     "timestamp": "<current_iso8601_timestamp>",
#     "evaluation": {{
#         "criteria": {{
#             "relevance": <integer_score_1_to_5>,
#             "correctness": <integer_score_1_to_5>,
#             "specificity": <integer_score_1_to_5>,
#             "clarity": <integer_score_1_to_5>
#         }},
#         "explanations": {{
#             "relevance": "<precise_explanation_max_50_words>",
#             "correctness": "<precise_explanation_max_50_words>",
#             "specificity": "<precise_explanation_max_50_words>",
#             "clarity": "<precise_explanation_max_50_words>"
#         }},
#         "total_score": <calculated_average_of_criteria_scores>
#     }},
#     "validation": {{
#         "json_schema_version": "draft-07",
#         "is_valid": true
#     }}
# }}

# IMPORTANT CONSTRAINTS:
# - JSON MUST be PERFECTLY VALID
# - All placeholders MUST be replaced with actual data
# - Explanations must be objective and factual
# - Scores must be integers between 1-5
# """
