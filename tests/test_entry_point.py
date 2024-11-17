import unittest
from pathlib import Path
import json

# Import modules directly from src
from src.thought_generation.thought_reader import IndexedThoughtReader
from src.interviewagent_xf_edit_2 import StateManager, QuestionLoader


class TestQuestionLoading(unittest.TestCase):
    def test_json_loading(self):
        """Test loading the thoughts JSON file."""
        test_file = Path("input_output/thoughts.json")
        reader = IndexedThoughtReader(test_file)
        self.assertEqual(reader.get_idea(), "embedded software development")
