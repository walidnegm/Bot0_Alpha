"""
Filename: thought_generator_openai_self_attention.py
Author: Xiao-Fei Zhang
Last updated on: 

Description:
This module provides a framework for generating sub-thoughts using the OpenAI API and 
evaluating their similarity through self-attention mechanisms using a transformer model (BERT). 

The module identifies distinct sub-thoughts by:
- clustering similar concepts together, 
- reducing redundancy, and 
- selecting representative ideas. 

It includes functions to 
- generate thoughts, 
- compute similarity matrices, 
- perform clustering, and save results
(for tasks like brainstorming, breaking down complex topics, and knowledge organization.)

Key Functions:
1. 'generate_thoughts': Uses OpenAI API to generate sub-thoughts based on a given topic.
2. 'compute_similarity_and_cluster': Computes self-attention-based similarity and clusters 
distinct sub-thoughts.
3. 'save_results': Saves generated thoughts and similarity matrices to files.
4. 'generate_prompt_template': Generates a JSON prompt template for the OpenAI API call.
5. 'main': Main function for testing and demonstrating the use of the above functions.

Usage:
The module can be imported into other modules for generating and clustering sub-thoughts. 
Each function returns data that can be handled programmatically.
"""

import os
import logging
import logging_config
from pathlib import Path
from typing import List, Optional, Union, Dict
from pydantic import ValidationError
from dotenv import load_dotenv
import jsonschema
import json
import numpy as np
import torch

# LLM related
from openai import OpenAI
from anthropic import Anthropic

# Internal
from utils.llm_api_utils import (
    call_openai_api,
    call_claude_api,
    call_llama3,
    get_claude_api_key,
    get_openai_api_key,
)
from utils.generic_utils import save_to_json_file
from models.llm_response_base_models import JSONResponse
from models.json_validation_models import (
    ClusterJSONModel,
    IdeaJSONModel,
    ThoughtJSONModel,
    validate_thought_batch,
)
from prompts.thought_generation_prompt_templates import (
    SUB_THOUGHT_GENERATION_PROMPT,
    RECLUSTER_AND_PICK_TOP_CLUSTER_PROMPT,
    THOUGHT_GENERATION_PROMPT,
    VERTICAL_SUB_THOUGHT_GENERATION_PROMPT,
)

logger = logging.getLogger(__name__)

# JSON schema to validate sub-thought generation response
# (this can potentially affect which ones are going to be saved or returned)
LLM_RES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "concept": {"type": "string"},
        "sub_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "importance": {
                        "type": ["string", "null"]
                    },  # Allow optional importance
                    "connection_to_next": {
                        "type": ["string", "null"]
                    },  # Allow optional connection_to_next
                },
                "required": ["name", "description"],  # Only these fields are required
            },
            "minItems": 1,
        },
    },
    "required": ["concept", "sub_concepts"],
}


class ThoughtGenerator:
    """
    A class to generate thoughts and sub-thoughts using an LLM API, compute similarity,
    and perform clustering.

    This class supports generating both high-level thoughts (concepts) from an idea,
    as well as detailed sub-thoughts under each concept. It dynamically validates
    the generated responses using appropriate Pydantic models (`IdeaJSONModel` or
    'ThoughtJSONModel') based on the generation context (horizontal or vertical).

    Methods:
        - generate_parallell_thoughts: Generates high-level thoughts from a main idea
        (horizontal generation).
        - generate_vertical_thoughts: Generates sub-thoughts under a main concept
        (vertical generation).
        - compute_similarity_and_cluster: Computes self-attention-based similarity and
        clusters distinct sub-thoughts.
        - save_results: Saves generated thoughts and sub-thoughts to files.
        - generate_prompt_template: Formats the prompt for the API call based on
        input templates.
        - call_llm: Calls the LLM API with the prompt and validates response based on
        the specified model.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model_id: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1056,
    ):
        """
        Initializes the ThoughtGenerator with API details for LLM calls.

        Args:
            llm_provider (str): The LLM provider to use ("openai", "claude", etc.).
            model_id (str): The model ID to use for the LLM (e.g., "gpt-4-turbo").
            temperature (float): Temperature setting for response creativity.
            max_tokens (int): Maximum tokens for each response.

        Raises:
            ValueError: If the specified LLM provider is unsupported.
        """
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Instantiate the API client based on llm_provider
        if llm_provider == "openai":
            self.client = OpenAI(api_key=get_openai_api_key())
        elif llm_provider == "claude":
            self.client = Anthropic(api_key=get_claude_api_key())
        else:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    def create_prompt(self, prompt_template: str, **kwargs: Dict[str, str]) -> str:
        """
        Formats a prompt using the specified template and arguments.

        Args:
            prompt_template (str): The template to format.
            **kwargs: Keyword arguments to fill in the template placeholders.

        Returns:
            str: The formatted prompt.

        Raises:
            ValueError: If a required template placeholder is missing in the kwargs.
        """
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            logger.error(f"Error formatting prompt: Missing key {missing_key}")
            raise ValueError(f"Missing key in the template: {missing_key}")

    def call_llm(
        self,
        prompt: str,
        temperature: float = None,
        validation_model: str = "thought_json",
    ) -> Optional[Union[IdeaJSONModel, ThoughtJSONModel]]:
        """
        Call the specified LLM API (OpenAI, Claude, or LLaMA3) with the provided prompt
        and return the response as a validated ThoughtJSONResponse.

        Args:
            - prompt (str): The formatted prompt to send to the LLM API.
            - temperature (float, optional): Temperature setting for this specific API call.
            If None, uses the class-level temperature.
            - json_type (str): key value to specify which pydantic model to validate LLM response.
            Default to "thought_json" (ThoughtJSONModel)

        Returns:
            Optional[Union[IdeaJSONModel, ThoughtJSONResponse]]: The validation model to apply,
            either "idea_json" or "thought_json" model.


        Raises:
            ValueError: If the llm_provider is unsupported.
        """
        try:
            thought_response_model = None

            if self.llm_provider.lower() in {"openai", "gpt"}:
                # Call OpenAI API with the instantiated client
                thought_response_model = call_openai_api(
                    prompt=prompt,
                    client=self.client,
                    model_id=self.model_id,
                    expected_res_type="json",
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens,
                    json_type=validation_model,
                )

            elif self.llm_provider.lower() in {"anthropic", "claude"}:
                # Call Claude API with the instantiated client
                thought_response_model = call_claude_api(
                    prompt=prompt,
                    client=self.client,
                    model_id=self.model_id,
                    expected_res_type="json",
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens,
                    json_type=validation_model,
                )

            elif self.llm_provider.lower() in {"llama3", "llama"}:
                # Call LLaMA3 API without a client instance
                thought_response_model = call_llama3(
                    prompt=prompt,
                    model_id=self.model_id,
                    expected_res_type="json",
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens,
                    json_type=validation_model,
                )

            else:
                raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")

            logger.debug(
                f"Response type: {type(thought_response_model)}, Value: {thought_response_model}"
            )
            return thought_response_model

        except Exception as e:
            logger.error(
                f"Error calling LLM API for provider '{self.llm_provider}': {e}"
            )
            return None

    def generate_parallell_thoughts(
        self,
        thought: str,
        prompt_template: str,
        num_sub_thoughts: int = 10,
        temperature: float = None,
    ) -> IdeaJSONModel:
        """
        Generates high-level thoughts based on an idea using horizontal thought generation.

        Args:
            thought (str): The main idea or theme to break down into high-level concepts.
            prompt_template (str): The template to use for generating the prompt.
            num_sub_thoughts (int): Number of thoughts (concepts) to generate.
            temperature (float): Optional temperature setting for the API call.

        Returns:
            IdeaJSONModel: A validated model containing the main idea and generated
            high-level thoughts.

        Raises:
            ValueError: If the API fails to generate thoughts for the specified idea.
        """
        prompt = self.create_prompt(
            prompt_template=prompt_template,
            idea=thought,
            num_sub_thoughts=num_sub_thoughts,
        )
        thought_response_model = self.call_llm(
            prompt=prompt,
            temperature=temperature or self.temperature,
            validation_model="idea_json",
        )

        if thought_response_model is None:
            raise ValueError(
                f"Failed to generate parallel thoughts for thought {thought}."
            )

        return thought_response_model

    # Transform clusters to thoughts list if needed
    def flatten_clusters_to_thoughts(
        self, idea_data: ClusterJSONModel
    ) -> IdeaJSONModel:
        """
        Transform a ClusterJSONModel into an IdeaJSONModel, flattening clusters into thoughts.

        Args:
            idea_data (ClusterJSONModel): Clustered data to be flattened into a thoughts list.

        Returns:
            IdeaJSONModel: Model containing flattened thoughts list.
        """
        thoughts_list = []

        # Logging the clusters for verification
        logger.debug(f"Flattening clusters: {idea_data.clusters}")

        for cluster in idea_data.clusters:
            for thought_name in cluster.thoughts:

                # Logging each thought being added
                logger.debug(
                    f"Adding thought: {thought_name} with description: {cluster.description}"
                )

                # Append each thought with its cluster description as the thought's description
                thoughts_list.append(
                    ThoughtJSONModel(
                        thought=thought_name, description=cluster.description
                    )
                )

        return IdeaJSONModel(idea=idea_data.idea, thoughts=thoughts_list, clusters=None)

    def regroup_and_pick_top_thoughts(
        self,
        sub_thoughts: IdeaJSONModel,
        num_clusters: int,
        top_n: int,
    ) -> ClusterJSONModel:
        """
        Re-clusters and selects top sub-thoughts, returning an IdeaJSONModel with the top clusters.

        Args:
            sub_thoughts (IdeaJSONModel): Initial model containing thoughts.
            num_clusters (int): Number of clusters to form.
            top_n (int): Number of top clusters to select.

        Returns:
            IdeaJSONModel: A model with the top selected clusters or thoughts.
        """
        # Prepare thought data for clustering
        thought_descriptions = [
            {"thought": thought.thought, "description": thought.description}
            for thought in sub_thoughts.thoughts
        ]

        # Construct the prompt for clustering and selecting the most relevant thoughts
        prompt = self.create_prompt(
            prompt_template=RECLUSTER_AND_PICK_TOP_CLUSTER_PROMPT,
            idea=sub_thoughts.idea,
            thoughts_list=thought_descriptions,
            num_clusters=num_clusters,
            top_n=top_n,
        )

        # Generate top clusters
        thought_response_model = self.call_llm(
            prompt=prompt, validation_model="cluster_json"
        )

        if thought_response_model is None:
            raise ValueError(
                f"Failed to regroup and pick top clusters for idea '{sub_thoughts.idea}'."
            )

        # -> into IdeaJSONModel
        thought_response_model = self.flatten_clusters_to_thoughts(
            thought_response_model
        )

        if thought_response_model is None:
            raise ValueError(
                f"Failed to transform clusters into IdeaJSONModel for idea '{sub_thoughts.idea}'."
            )
        return thought_response_model

    def process_horizontal_thought_generation(
        self,
        thought: str,
        num_sub_thoughts: int = 10,
        num_clusters: int = 6,
        top_n: int = 4,
    ) -> IdeaJSONModel:
        """
        Orchestrates horizontal thought generation and clustering, breaking down a higher level
        concept into lower level concepts.

        This method first generates an initial set of lower level concepts using a language model (LLM).
        It then re-clusters them into larger groups, and then selects the top N clusters,
        returning the final set of re-clustered and selected lower level concepts.

        Args:
            -thought (str): top level concept (thought) to generate lower level concepts (sub-thoughts).
            -num_sub_thoughts (int, optional): The number of lower level concept (sub_thoughts) to
            generate initially. Default is 10.
            -num_clusters (int, optional): The number of clusters to form during re-clustering.
            Default is 6.
            -top_n (int, optional): The number of top clusters to select after re-clustering.
            Default is 4.

        Returns:
            IdeaJSONModel: A pydantic model instance with the final set of re-clustered and
            selected sub-thoughts.

        Example:
            >>> generator = ThoughtGenerator()
            >>> result = generator.process_horizontal_thought_generation("embedded systems", 5, 3, 2)
            >>> print(result.json(indent=4))

            {
                "idea": "embedded systems",
                "concepts": [
                    {
                        "concept": "embedded software development",
                        "description": "Focuses on creating and maintaining software for embedded systems."
                    },
                    {
                        "concept": "hardware design in embedded systems",
                        "description": "Covers the physical components and circuit design for embedded systems."
                    }
                ]
            }
        """
        init_thoughts = self.generate_parallell_thoughts(
            thought=thought,
            prompt_template=THOUGHT_GENERATION_PROMPT,  # use idea->thought prompt template
            num_sub_thoughts=num_sub_thoughts,
        )

        final_thoughts = self.regroup_and_pick_top_thoughts(
            sub_thoughts=init_thoughts,
            num_clusters=num_clusters,
            top_n=top_n,
        )

        return final_thoughts

    def generate_vertical_thoughts(
        self,
        thought: str,
        idea: str,  # high level concept input by user or system-record keeping in json
        progression_type: str = "implementation_steps",
        num_sub_thoughts: int = 7,
        temperature: int = None,
    ) -> ThoughtJSONModel:
        """
        Generates multiple vertical sub-thoughts based on the progression type, returning
        a Thought model.

        Args:
            - thought (str): The main topic to generate sub-thoughts for.
            * idea (str): The overarching idea or theme containing the highest level concept.
            - progression_type (str): The type of progression
            (e.g., "simple_to_complex", "implementation_steps").
                4 types of progression_type:
                    "simple_to_complex",
                    "implementation_steps",
                    "chronological",
                    "problem_solution",
            - num_sub_thoughts (int): The number of sub-thoughts to generate.
            - global_context (str): The domain or field for contextualizing
            the thought generation.

        Returns:
            ThoughtJSONModel: The main concept with a structured list of sub-thoughts.

        Raises:
            ValueError: If an invalid progression type is passed.

        Example:
            >>> generator = ThoughtGenerator()
            >>> result = generator.generate_vertical_thoughts("embedded software development", \
                "embedded systems", "simple_to_complex", 5)
            >>> print(result.json(indent=4))

            {
                "concept": "embedded software development",
                "sub_concepts": [
                    {
                        "name": "requirements analysis",
                        "description": "Define system requirements to guide software development."
                    },
                    {
                        "name": "architecture design",
                        "description": "Plan the software structure and component interactions."
                    },
                    {
                        "name": "implementation",
                        "description": "Write and test the code for system functionality."
                    }
                ]
            }
        """
        prompt = self.create_prompt(
            prompt_template=VERTICAL_SUB_THOUGHT_GENERATION_PROMPT,
            thought=thought,
            num_sub_thoughts=num_sub_thoughts,
            progression_type=progression_type,
            idea=idea,
        )
        thought_response_model = self.call_llm(
            prompt=prompt,
            temperature=temperature or self.temperature,
        )

        if thought_response_model is None:
            raise ValueError(
                f"Failed to generate vertical thoughts for thought '{thought}'."
            )
        return thought_response_model

    def generate_array_of_thoughts(
        self,
        input_data: Dict,
        progression_type: str = "implementation_steps",
        num_sub_thoughts: int = 5,
        temperature: int = None,
    ) -> IdeaJSONModel:
        """
        Generates sub-thoughts for each concept within a high-level idea.

        This function takes a dictionary with a high-level idea and multiple concepts, 
        each of which is expanded into detailed sub-thoughts. It uses vertical thought generation
        for each concept under the overarching idea.

        Args:
            - input_data (dict): A dictionary containing an "idea" and a list of "concepts".
                Example format:
                {
                    "idea": "embedded systems",
                    "thoughts": [
                        {
                            "name": "embedded software development",
                            "description": "Creating and managing software for embedded systems."
                        },
                        ...
                    ]
                }
            - progression_type (str): Type of progression to generate sub-thoughts 
            (e.g., "simple_to_complex").
            - num_sub_thoughts (int): The number of sub-thoughts to generate for each concept.
            temperature (float, optional): Optional temperature setting for the LLM call.

        Returns:
            List[ThoughtJSONModel]: A list of ThoughtJSONModel instances, each containing 
            generated sub-thoughts.

        Example:
            >>> generator = ThoughtGenerator()
            >>> input_data = {
                    "idea": "embedded systems",
                    "thoughts": [
                        {"name": "System Software Development", "description": \
                            "Development of low-level software..."},
                        {"name": "Application Software Development", "description": \
                            "Focuses on high-level applications for embedded systems"}
                    ]
                }
            >>> result = generator.generate_array_of_thoughts(input_data, "simple_to_complex", 3)
            >>> for thought in result:
                    print(thought.json(indent=4))

            [
                {
                    "concept": "System Software Development",
                    "sub_concepts": [
                        {
                            "name": "kernel programming",
                            "description": "Creating low-level code for managing hardware resources."
                        },
                        {
                            "name": "device drivers",
                            "description": "Interfaces between hardware components and the operating system."
                        }
                    ]
                },
                {
                    "concept": "Application Software Development",
                    "sub_concepts": [
                        {
                            "name": "UI/UX design",
                            "description": "Designing user interfaces for embedded applications."
                        },
                        {
                            "name": "middleware services",
                            "description": "Provides a layer for application components to interact."
                        }
                    ]
                }
            ]
        """
        idea = input_data.get("idea")
        thoughts = input_data.get("thoughts", [])
        if not idea or not thoughts:
            raise ValueError("Invalid input data.")

        all_thoughts = []
        for thought in thoughts:
            thought_name = thought.get("thought")
            thought_description = thought.get(
                "description", None
            )  # Get description, default to None if missing
            if not thought_name:
                logger.warning("Skipping sub-concept with missing 'name'.")
                continue

            try:
                thought_model = self.generate_vertical_thoughts(
                    thought=thought_name,
                    idea=idea,
                    progression_type=progression_type,
                    num_sub_thoughts=num_sub_thoughts,
                    temperature=temperature,
                )

                # Add description to the ThoughtJSONModel instance
                thought_model.description = thought_description

                # Append the model to all_thoughts[]
                all_thoughts.append(thought_model)
            except ValueError as e:
                logger.error(f"Failed to generate sub-thoughts for {thought_name}: {e}")

        # Validate the batch of generated thoughts (validate_thought_batch pyd model
        # handles any a high level to multiple lower level "thoughts")
        validated_thoughts = validate_thought_batch(
            [thought.model_dump() for thought in all_thoughts]
        )
        logger.info(
            f"Validated {len(validated_thoughts)} out of {len(all_thoughts)} generated thoughts."
        )

        # Load list of thought models into idea model
        validated_idea = IdeaJSONModel(idea=idea, thoughts=all_thoughts)

        return validated_idea

    def save_results(
        self,
        data: Union[ThoughtJSONModel, List[ThoughtJSONModel]],
        json_file: Union[Path, str],
    ):
        """
        Saves generated thoughts or sub-thoughts to a JSON file.

        Args:
            - data (Union[ThoughtJSONModel, List[ThoughtJSONModel]]): The data to save,
                either a single ThoughtJSONModel instance or a list of them.
            - json_file (Union[Path, str]): The path to save the JSON file.

        Raises:
            IOError: If an error occurs during the file saving process.
        """

        try:
            # If data is a list, convert each ThoughtJSONResponse to a dictionary
            if isinstance(data, list):
                data_dict = [item.model_dump() for item in data]

            else:
                # Convert the Pydantic model to a dictionary
                data_dict = data.model_dump()

            logger.info(f"Data before saving: \n{data_dict}")

            # Write the dictionary or list of dictionaries to a JSON file with indentation
            save_to_json_file(data=data_dict, file_path=json_file)

            logger.info(f"Data saved to {json_file}.")
        except Exception as e:
            logger.error(f"Error saving data to {json_file}: {e}")
