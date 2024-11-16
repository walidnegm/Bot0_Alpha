import pytest
from thought_generation.thought_generator import ThoughtGenerator
from models.thought_models import IdeaJSONModel, IdeaClusterJSONModel
from prompts.thought_generation_prompt_templates import THOUGHT_GENERATION_PROMPT


def test_process_horizontal_thought_generation():
    """
    Test the horizontal thought generation process to ensure clusters are generated and validated.
    """
    # Initialize the ThoughtGenerator instance
    generator = ThoughtGenerator(
        llm_provider="openai",
        model_id="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=512,
    )

    # Test inputs
    test_idea = "Future of AI in Education"
    num_sub_thoughts = 10
    num_clusters = 4
    top_n = 4

    try:
        # Step 1: Generate initial thoughts (IdeaJSONModel)
        idea_model = generator.generate_parallell_thoughts(
            thought=test_idea,
            prompt_template=THOUGHT_GENERATION_PROMPT,
            num_sub_thoughts=num_sub_thoughts,
        )

        assert isinstance(
            idea_model, IdeaJSONModel
        ), f"Expected IdeaJSONModel, but got {type(idea_model).__name__}."
        assert (
            len(idea_model.thoughts) <= num_sub_thoughts
        ), f"Expected at most {num_sub_thoughts} thoughts, but got {len(idea_model.thoughts)}."

        # Step 2: Cluster and pick top clusters (IdeaClusterJSONModel)
        cluster_model = generator.cluster_and_pick_top_clusters(
            thoughts_to_group=idea_model,
            num_clusters=num_clusters,
            top_n=top_n,
        )

        assert isinstance(
            cluster_model, IdeaClusterJSONModel
        ), f"Expected IdeaClusterJSONModel, but got {type(cluster_model).__name__}."
        assert (
            len(cluster_model.clusters) == top_n
        ), f"Expected {top_n} clusters, but got {len(cluster_model.clusters)}."

        # Step 3: Convert clusters back into thoughts (IdeaJSONModel)
        final_idea_model = generator.convert_cluster_to_idea(cluster_model)

        assert isinstance(
            final_idea_model, IdeaJSONModel
        ), f"Expected IdeaJSONModel, but got {type(final_idea_model).__name__}."
        assert (
            len(final_idea_model.thoughts) == top_n
        ), f"Expected {top_n} thoughts, but got {len(final_idea_model.thoughts)}."

        print("Test passed!")
        print("Final Idea Model:")
        print(final_idea_model.model_dump())

    except (AssertionError, ValueError, TypeError) as e:
        print(f"Test failed: {e}")
        raise e


if __name__ == "__main__":
    test_process_horizontal_thought_generation()
