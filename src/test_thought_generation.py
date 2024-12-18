import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from thought_generation.thought_generator import ThoughtGenerator
from models.thought_models import IdeaJSONModel, IdeaClusterJSONModel
from prompts.thought_generation_prompt_templates import THOUGHT_GENERATION_PROMPT


# Fixture to mock the `input_output` directory
@pytest.fixture
def mock_input_output_dir(tmp_path):
    input_output_dir = tmp_path / "input_output"
    input_output_dir.mkdir()
    # Create mock files if necessary (adjust according to your use case)
    (input_output_dir / "mock_file.json").write_text('{"mock_key": "mock_value"}')
    return input_output_dir


# Mock file reading and writing functions
@patch("thought_generation.thought_generator.read_from_json_file")
@patch("thought_generation.thought_generator.save_to_json_file")
def test_process_horizontal_thought_generation(
    mock_save, mock_read, mock_input_output_dir
):
    """
    Test the horizontal thought generation process to ensure clusters are generated and validated.
    """
    # Mock the file read and save operations
    mock_read.return_value = {"mock_key": "mock_value"}
    mock_save.return_value = None

    # Initialize the ThoughtGenerator instance
    generator = ThoughtGenerator(
        llm_provider="openai",
        model_id="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=512,
    )

    # Ensure the generator uses the mocked `input_output` directory
    generator.input_output_dir = mock_input_output_dir

    # Test inputs
    test_idea = "Future of AI in Education"
    num_sub_thoughts = 10
    num_clusters = 4
    top_n = 4

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


if __name__ == "__main__":
    # For standalone testing
    pytest.main([__file__])


# !Another option below
# # Add the root directory to sys.path
# root_dir = Path(__file__).parent.parent
# sys.path.append(str(root_dir))

# # Ensure input_output is accessible
# input_output_dir = root_dir / "input_output"
# if not input_output_dir.exists():
#     raise FileNotFoundError(f"Required directory {input_output_dir} does not exist.")
