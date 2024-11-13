# Append this after the task part of the prompt
# Need the {{}} (also called escape {}) because you expect the returned content to have a {} format.

# *Here: thought always represents the top level, which is "idea",
# and sub_thought represents lower levels (idea -> thought -> sub_thought)
THOUGHT_GENERATION_PROMPT = """
You are an expert in organizing complex topics into distinct, high-level thoughts.

Perform the following tasks:
1. Analyze the given overarching theme or idea: "{idea}"
2. Identify and break down this idea into {num_sub_thoughts} key thoughts. 
   Ensure each thought is distinct from the others and represents a significant aspect of the idea.
3. For each thought, provide a brief description (1-2 sentences) that captures its essence and 
   importance within the idea.

Return your analysis in the following JSON format:

{{
  "idea": "{idea}",
  "thoughts": [
    {{
      "thought": "Thought 1",
      "description": "Brief description of Thought 1"
    }},
    # Additional thoughts
  ]
}}

Ensure your response is a valid JSON object with the exact structure provided above, 
without any additional text, explanations, or markdown syntax.
"""

# Need the double curly braces {{ }}
SUB_THOUGHT_GENERATION_PROMPT = """
You are an expert in breaking down large and complex topics or thoughts into simpler parts.

Perform the following tasks:
1. Analyze the given topic or thought: {thought}
2. Break down this thought into {num_sub_thoughts} key sub-thoughts. Ensure that each sub-thought is 
   distinct from the others.
3. For each sub-thought, provide a very brief description (1-2 sentences).

Return your analysis in the following JSON format:

{{
  "idea": "{idea}",
  "thought": "{thought}",
  "sub_thoughts": [
    {{
      "name": "Sub-thought 1",
      "description": "Brief description of Sub-thought 1"
    }},
    # Additional sub-thoughts
  ]
}}

Ensure your response is a valid JSON object with the exact structure provided above, 
without any additional text, explanations, or markdown syntax.
"""

# TODO: Incorporate this later into the standard prompt
# TODO: (it's well structured but has a metadata section,
# TODO: which is more standard industry practice but not for this phase)
SUB_THOUGHT_GENERATION_PROMPT_CLAUDE_EDIT_VERSION = """
You are an expert in breaking down large and complex topics or thoughts into simpler, meaningful components.

Context:
- Input thought: {thought}
- Requested number of sub-thoughts: {num_sub_thoughts}
- Record ID: {idea}

Task Overview:
1. Analyze the given thought
2. Break it down into {num_sub_thoughts} key sub-thoughts
3. Provide a concise description for each sub-thought

Requirements for Sub-Thoughts:
- Must be directly related to the main thought
- Should be at a similar level of abstraction
- Should be clearly distinct from each other
- Collectively should cover the core aspects of the main thought
- If the topic is too narrow for {num_sub_thoughts} sub-thoughts, \
  generate the maximum reasonable number and explain in the "metadata" field

Requirements for Descriptions:
- Maximum 30 words per description
- Focus on unique, defining characteristics
- Avoid redundancy with the sub-thought name

Return your analysis in the following JSON format:

{{
  "metadata": {{
    "record_id": "{idea}",
    "requested_sub_thoughts": {num_sub_thoughts},
    "actual_sub_thoughts": "<number_generated>",
    "notes": "<optional: explanation if number differs from requested>"
  }},
  "thought": "{thought}",
  "sub_thoughts": [
    {{
      "name": "Sub-thought name",
      "description": "Brief description (max 30 words)",
      "relevance_score": "<float between 0-1 indicating importance to main thought>"
    }},
    # Additional sub-thoughts...
  ]
}}

Important:
- Ensure the response is a valid JSON object
- Do not include any additional text or markdown syntax
- All fields are required except "metadata.notes"
- Numbers should be numeric (not strings)
- Relevance scores should sum to 1.0 across all sub-thoughts
"""

RECLUSTER_AND_PICK_TOP_SUB_THOUGHTS_PROMPT = """
You are an expert in organizing complex ideas into distinct, well-defined clusters.

Task:
You are given a list of sub-thoughts related to the main thought: {thought}.
Each sub-thought represents a distinct aspect of this thought, and your task is to re-organize 
these sub-thoughts into the {num_clusters} most meaningful and distinct clusters.

Instructions:
1. Carefully analyze the sub-thoughts and group them based on their thematic or conceptual \
  similarities. 
2. Provide a name for each cluster that best represents the overall idea of \
  the grouped sub-thoughts.
3. Select and return the top {top_n} clusters that are most distinct from each other and \
  provide a very brief 
   description (1-2 sentences) of each cluster's key ideas.

Sub-Thoughts:
{sub_thoughts_list}

Ensure your response is a valid JSON object with the following format:

{{
  "idea": "{idea}",
  "thought": "{thought}",
  "sub_thoughts": [
    {{
      "name": "Cluster 1",
      "description": "Brief description of Cluster 1"
    }},
    ...
  ]
}}

Ensure your response is a valid JSON object with the exact structure provided above, 
without any additional text, explanations, or markdown syntax.
"""

RECLUSTER_AND_PICK_TOP_CLUSTER_PROMPT = """
You are an expert in organizing complex ideas into distinct, well-defined clusters.

Task:
You are given a list of thoughts related to the main idea: "{idea}".
Each thought represents a distinct aspect of this idea. Your task is to re-organize these thoughts 
into {num_clusters} meaningful and distinct clusters.

Instructions:
1. Carefully analyze each thought and group them based on thematic or conceptual similarities. 
2. Provide a name for each cluster that best represents the overall theme of the grouped thoughts.
3. Select and return the top {top_n} clusters that are most distinct from each other and 
   provide a brief description (1-2 sentences) of each cluster's key themes.

Thoughts List:
{thoughts_list}

Ensure your response is a valid JSON object with the following format:

{{
  "idea": "{idea}",
  "clusters": [
    {{
      "name": "Cluster 1",
      "description": "Brief description of Cluster 1",
      "thoughts": ["Thought 1", "Thought 2", ...]
    }},
    # Additional clusters
  ]
}}

Ensure your response is a valid JSON object with the exact structure provided above, 
without any additional text, explanations, or markdown syntax.
"""

VERTICAL_SUB_THOUGHT_GENERATION_PROMPT = """
You are an expert in the field of {idea}.
Your goal is to provide structured, step-by-step explanations that break down complex topics 
within these domains. You should always ensure that the explanation builds on foundational knowledge 
and leads to a deeper understanding of advanced concepts.

Task:
Break down and explain the key aspects of a main thought following a {progression_type} progression.

Main Thought: {thought}

Progression Types:
1. "simple_to_complex": Start with the most basic concepts and gradually introduce more advanced ideas.
2. "implementation_steps": Describe the key areas or steps one would need to consider when implementing or 
   working with the main thought.
3. "chronological": If applicable, explain the evolution or historical development of the main thought.
4. "problem_solution": Introduce problems or challenges related to the main thought, followed by 
   their solutions or approaches.

Instructions:
1. Based on the specified progression type **{progression_type}**, determine the most appropriate 
   starting point for explaining {thought}.
2. For each subsequent step, explain the sub-thought that logically follows from the previous one, 
   adhering to the chosen progression type.
3. For each step, provide:
   - A clear explanation of the sub-thought
   - Why it's important in the context of {thought}
   - How it connects to the next step (except for the final step)
4. Provide a total of {num_sub_thoughts} steps in your explanation.
5. **For the final step, omit the "connection_to_next" field entirely. \
  Do not include any explanation 
   or comment about the omission. Simply leave it out.**

Your response must be a valid JSON object with the following format:

{{
  "idea": "{idea}",
  "thought": "{thought}",
  "progression_type": "{progression_type}",
  "sub_thoughts": [
    {{
      "name": "Name of the sub-thought",
      "description": "Succinct explanation of the sub-thought (2-3 sentences)",
      "importance": "Why this sub-thought is important",
      "connection_to_next": "How this sub-thought leads to the next one"
    }}
  ]
}}

IMPORTANT: Ensure your response is a valid JSON object with the exact structure \
  provided above, 
without any additional text, comments, explanations, or markdown syntax.
"""

SIMPLE_TO_COMPLEX_SUB_THOUGHT_GENERATION_PROMPT = """
You are an expert in explaining complex topics in a structured, step-by-step manner.

Task:
Explain a given main thought following a progression from simple to complex sub-thoughts. 
Start with the most basic ideas and gradually introduce more advanced concepts.
**For the final step, omit the "connection_to_next" field entirely. Do not include \
  any explanation 
  or comment about the omission. Simply leave it out.**

Main Thought: {thought}

Instructions:
1. Begin with the most fundamental sub-thought related to the main thought.
2. For each subsequent step, introduce a more advanced sub-thought that builds upon \
  previous knowledge.
3. For each step, provide:
   - A clear explanation of the sub-thought
   - Why it's important in understanding the main thought
   - How it leads to or connects with more advanced sub-thoughts \
    (except for the final step)
4. Include a total of {num_steps} steps in your explanation.

Your response must be a valid JSON object with the following format:

{{
  "idea": "{idea}",
  "thought": "{thought}",
  "steps": [
    {{
      "step_number": 1,
      "name": "Name of the sub-thought",
      "explanation": "Clear explanation of the sub-thought",
      "importance": "Why this sub-thought is important",
      "leads_to": "How this sub-thought leads to more advanced ones"
    }},
    # Additional steps
  ]
}}

IMPORTANT: Ensure your response is a valid JSON object with the exact \
  structure provided above, without any additional text, comments, explanations, \
    or markdown syntax.
"""
