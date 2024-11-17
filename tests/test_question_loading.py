import os
import sys
from pathlib import Path
import unittest
from pathlib import Path
from datetime import datetime
import json


project_root = Path(__file__).resolve().parent.parent
src_dir = os.path.join(project_root, "src")
os.chdir(src_dir)
print(f"Current Working Directory: {os.getcwd()}")  # Optional, for debugging

if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import your modules and classes

from thought_generation.thought_reader import IndexedThoughtReader
from interviewagent_xf_edit_2 import StateManager, QuestionLoader, DialogueManager
from project_config import INTERVIEW_STATES_FILE

# Mock Data for Testing
MOCK_JSON_FILE = r"C:\github\Bot0_Alpha\input_output\thought_generation\openai_thought_generation\models_with_indexes\array_of_thoughts_output_with_index_embedded_software_development_openai.json"
MOCK_STATE_FILE = INTERVIEW_STATES_FILE

MOCK_DATA = {
    "idea": "embedded software development",
    "thoughts": [
        {
            "thought_index": 0,
            "thought": "System Essentials",
            "description": "Focuses on the core operating systems...",
            "sub_thoughts": [
                {
                    "sub_thought_index": 0,
                    "name": "Define System Requirements",
                    "description": "Identify the specific needs...",
                    "importance": "Defining system requirements is crucial...",
                    "connection_to_next": "Knowing the requirements helps...",
                },
                {
                    "sub_thought_index": 1,
                    "name": "Select Appropriate Hardware",
                    "description": "Choose suitable hardware components...",
                    "importance": "Selecting appropriate hardware is essential...",
                    "connection_to_next": "Once the hardware is selected...",
                },
            ],
        }
    ],
}


class TestQuestionLoading(unittest.TestCase):
    """Unit tests for loading questions, state management, and debugging."""

    @classmethod
    def setUpClass(cls):
        """Set up mock files for testing."""
        # Define paths for mock JSON and state files
        cls.mock_json_path = Path(
            "C:/github/Bot0_Alpha/input_output/thought_generation/openai_thought_generation/models_with_indexes/array_of_thoughts_output_with_index_embedded_software_development_openai.json"
        )
        cls.mock_state_path = (
            Path("C:/github/Bot0_Alpha/input_output/interview_states")
            / "test_states.json"
        )

        # Ensure the directory for the state file exists
        if not cls.mock_state_path.parent.exists():
            cls.mock_state_path.parent.mkdir(parents=True, exist_ok=True)

        # Create an empty state file or clear existing contents
        with cls.mock_state_path.open("w") as f:
            json.dump({}, f)  # Write an empty JSON object

        # Write mock JSON data
        with cls.mock_json_path.open("w") as f:
            json.dump(MOCK_DATA, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up mock files."""
        if cls.mock_json_path.exists():
            cls.mock_json_path.unlink()  # Delete the mock JSON file
        # if cls.mock_state_path.exists():
        #     cls.mock_state_path.unlink()  # Delete the mock state file

    def setUp(self):
        """Initialize components for each test."""
        self.reader = IndexedThoughtReader(self.mock_json_path)
        self.state_manager = StateManager(storage_path=str(self.mock_state_path))
        self.question_loader = QuestionLoader(data_file=self.mock_json_path)

    def test_load_json_data(self):
        """Test loading and validating the JSON data."""
        # Check the idea
        idea = self.reader.get_idea()
        self.assertEqual(idea, MOCK_DATA["idea"], "Idea does not match expected value.")

        # Check the thoughts
        thoughts = self.reader.get_thoughts()
        self.assertEqual(
            len(thoughts), len(MOCK_DATA["thoughts"]), "Thought count mismatch."
        )

    def test_state_management(self):
        """Test the StateManager class for state handling."""
        user_id = "test_user"
        initial_state = self.state_manager.get_state(user_id)

        # Verify initial state
        self.assertEqual(initial_state["thought_index"], 0)
        self.assertEqual(initial_state["sub_thought_index"], 0)

        # Update state
        self.state_manager.update_state(user_id, thought_index=1, sub_thought_index=2)
        updated_state = self.state_manager.get_state(user_id)

        # Verify updated state
        self.assertEqual(updated_state["thought_index"], 1)
        self.assertEqual(updated_state["sub_thought_index"], 2)

    def test_question_loading(self):
        """Test loading questions based on the user's state."""
        user_id = "test_user"

        # Initial question
        question_data = self.question_loader.get_next_question(
            user_id, self.state_manager
        )
        self.assertIn("thought", question_data, "Thought not found in question data.")
        self.assertIn(
            "sub_thought", question_data, "Sub-thought not found in question data."
        )

        # Verify first sub-thought
        sub_thought = question_data["sub_thought"]
        self.assertEqual(
            sub_thought["name"], MOCK_DATA["thoughts"][0]["sub_thoughts"][0]["name"]
        )

        # Update state and fetch next question
        self.state_manager.update_state(user_id, sub_thought_index=1)
        question_data = self.question_loader.get_next_question(
            user_id, self.state_manager
        )
        sub_thought = question_data["sub_thought"]

        # Verify second sub-thought
        self.assertEqual(
            sub_thought["name"], MOCK_DATA["thoughts"][0]["sub_thoughts"][1]["name"]
        )

    def test_reset_state(self):
        """Test resetting the user's state."""
        user_id = "test_user"

        # Update and reset state
        self.state_manager.update_state(user_id, thought_index=1, sub_thought_index=2)
        self.state_manager.reset_state(user_id)

        # Verify reset state
        reset_state = self.state_manager.get_state(user_id)
        self.assertEqual(reset_state["thought_index"], 0)
        self.assertEqual(reset_state["sub_thought_index"], 0)

    def test_full_workflow(self):
        """Test the full question-fetching and state management workflow."""
        user_id = "test_user"

        # Simulate fetching all questions
        while True:
            question_data = self.question_loader.get_next_question(
                user_id, self.state_manager
            )

            if "message" in question_data:
                # No more questions available
                self.assertEqual(
                    question_data["message"], "No more questions available."
                )
                break

            # Verify question data
            self.assertIn("thought", question_data)
            self.assertIn("sub_thought", question_data)

        # Verify that the state manager resets when no more questions are available
        state = self.state_manager.get_state(user_id)
        self.assertEqual(state["thought_index"], 0)
        self.assertEqual(state["sub_thought_index"], 0)


if __name__ == "__main__":
    unittest.main()
