import os
import logging
import logging_config
from typing import Optional
from thought_generation.thought_generator import ThoughtGenerator

logger = logging.getLogger(__name__)


def vertical_thought_pipeline(
    concept: str,
    num_sub_concepts: int,
    context: str,
    json_file: Optional[str] = None,
):
    """
    Pipeline to generate "vertical thoughts" based on a main concept and
    save the results to a JSON file.

    This function generates a specified number of sub-thoughts for a given main concept following a
    vertical thought process. It ensures the output is saved to a specified JSON file. The function
    uses an external thought generation model (e.g., GPT-4 turbo) to create the sub-thoughts,
    considering the context provided, and checks for the existence of the target file directory
    before proceeding.

    Args:
        concept (str): The main concept or topic from which sub-thoughts will be generated.
        num_sub_concepts (int): The number of sub-thoughts to be generated.
        context (str): A global context that provides domain knowledge to guide thought generation.
        json_file (str): The path to the JSON file where the generated sub-thoughts will be saved.

    Raises:
        FileNotFoundError: If the specified directory for the JSON file does not exist.

    Example:
        vertical_thought_pipeline(
            concept="Artificial Intelligence",
            num_sub_concepts=5,
            context="machine learning, neural networks",
            json_file="output/sub_thoughts.json"
        )
    """

    # rename param names to be consistent, check json file path
    thought, num_sub_thoughts, global_context = concept, num_sub_concepts, context

    if json_file:
        directory = os.path.dirname(json_file)
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"The file or directory at {directory} does not exist."
            )

    # Process to generate sub-topics/concepts
    thought_generator = ThoughtGenerator(
        llm_provider="openai", model_id="gpt-4-turbo", temperature=0.8
    )
    sub_topics = thought_generator.generate_vertical_sub_thoughts(
        thought=thought,
        num_sub_thoughts=num_sub_thoughts,
        global_context=global_context,
    )

    print(sub_topics)

    # Save results to file if json_file is provided
    if json_file:
        thought_generator.save_results(sub_topics, json_file)
