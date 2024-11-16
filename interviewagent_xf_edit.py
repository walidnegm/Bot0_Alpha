"""
Refactored interview agent module with modular state management and dialogue handling.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from pydantic import BaseModel
from io import BytesIO
import base64
import openai
import configparser
import os
import logging
import json
from typing import Optional
from pydantic import ValidationError
from datetime import datetime
import getpass
from dotenv import load_dotenv

# from internal modules (ThoughtReader, data folder locations)
from src.thought_generation.thought_reader import IndexedThoughtReader
from src.config import (
    OPENAI_INDEXED_MODELS_DIR,
    CLAUDE_INDEXED_MODELS_DIR,
    INTERVIEW_STATES_FILE,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_dependencies():
    """
    Initialize and return core dependencies: state_manager, question_loader, dialogue_manager.
    This function is called as a dependency in FastAPI endpoints.
    """
    try:
        # Define the thought data and state file paths
        thought_data_file = OPENAI_INDEXED_MODELS_DIR / "thoughts.json"
        state_file = INTERVIEW_STATES_FILE

        # Load thoughts data
        json_content = load_thoughts_data(thought_data_file)

        # Initialize core components
        state_manager = StateManager(storage_path=state_file)
        question_loader = QuestionLoader(data_source=json_content)
        dialogue_manager = DialogueManager(llm_client=openai)

        return state_manager, question_loader, dialogue_manager
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {str(e)}")
        raise RuntimeError("Failed to initialize application dependencies") from e


# Load OpenAI API key
def get_xf_api_key():
    """Load the API key based on the user environment."""
    current_user = getpass.getuser()
    xf_username = "xzhan"
    if current_user == xf_username:
        # Load .env file and get the API key
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")
        logger.info(f"Loaded API key from .env for user: {xf_username}")
        return api_key

    return None


def get_openai_api_key():
    """Load the OpenAI API key, either from .env or a config file."""
    api_key = get_xf_api_key()
    if not api_key:
        config = configparser.ConfigParser()
        config.read("/root/backend/config.ini")
        api_key = config.get("settings", "OPENAI_API_KEY", fallback=None)

    if api_key:
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("OpenAI API key loaded.")
        return api_key
    else:
        raise ValueError("OpenAI API key not found.")


def load_thoughts_data(json_file: Path):
    """
    Use the ThoughtReader class to:
    Read data -> validate & load into pyd model
    -> print out for debugging and analysis
    -> and return full data dump
    """
    try:
        reader = IndexedThoughtReader(json_file)
        logger.info(f"Loaded main idea: {reader.get_idea()}")
        logger.info(f"Loaded thoughts: {reader.get_thoughts()}")
        return reader.dump_all()
    except FileNotFoundError:
        logger.error(f"Thoughts file not found: {json_file}")
        raise
    except ValidationError as e:
        logger.error(f"Validation error in thoughts data: {e.json()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during thoughts loading: {str(e)}")
        raise


# ================== Core Components ==================
class StateManager:
    """Manages user states for the interview bot."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.states = self._load_states()

    def _load_states(self):
        """Load states from a file if provided."""
        if self.storage_path:
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        return {}

    def persist(self):
        """Persist states to a file if storage path is provided."""
        if self.storage_path:
            with open(self.storage_path, "w") as f:
                json.dump(self.states, f)

    def get_state(self, user_id: str):
        """Retrieve or initialize state for a user."""
        if user_id not in self.states:
            self.states[user_id] = {
                "thought_index": 0,
                "sub_thought_index": 0,
                "current_question": None,
                "last_updated": datetime.utcnow().isoformat(),
            }
        return self.states[user_id]

    def update_state(self, user_id: str, **updates):
        """Update the user's state."""
        state = self.get_state(user_id)
        state.update(updates)
        state["last_updated"] = datetime.utcnow().isoformat()
        self.persist()

    def reset_state(self, user_id: str):
        """Reset the user's state."""
        if user_id in self.states:
            del self.states[user_id]
        self.persist()


class QuestionLoader:
    """Loads and manages questions from the hierarchical data."""

    def __init__(self, data_source):
        self.data_source = data_source

    def get_next_question(self, user_id: str, state_manager: StateManager):
        """Fetch the next question based on the user's current state."""
        state = state_manager.get_state(user_id)

        try:
            thought = self.data_source["thoughts"][state["thought_index"]]
            sub_thought = thought["sub_thoughts"][state["sub_thought_index"]]

            # Update state for the next question
            state_manager.update_state(
                user_id,
                thought_index=state["thought_index"],
                sub_thought_index=state["sub_thought_index"] + 1,
            )

            # Handle sub-thought overflow
            if state_manager.get_state(user_id)["sub_thought_index"] >= len(
                thought["sub_thoughts"]
            ):
                state_manager.update_state(
                    user_id,
                    thought_index=state["thought_index"] + 1,
                    sub_thought_index=0,
                )

            return {
                "thought": thought,
                "sub_thought": sub_thought,
            }
        except IndexError:
            state_manager.reset_state(user_id)
            return {"message": "No more questions available."}


class DialogueManager:
    """Handles interactions with the LLM to generate questions."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_question(
        self, description: str, importance: str, pleasantry: str = ""
    ):
        """Generate a question using the LLM."""
        prompt = (
            f"{pleasantry}Generate an interview question based on the following details:\n\n"
            f"Description: {description}\n"
            f"Importance: {importance}\n\n"
            "The question should be concise, relevant, and specific to the topic."
        )
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an interview agent."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# ================== FastAPI Endpoints ==================
@app.get("/")
async def read_root():
    """Root endpoint to verify the server is running."""
    return {"message": "FastAPI server is running!"}


@app.get("/get_question/{user_id}")
async def get_question_async(user_id: str, dependencies=Depends(get_dependencies)):
    """
    Retrieve the next question for a given user based on their state.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        dict: JSON response containing the generated question and metadata.
    """
    try:
        # Ensure global variables are available
        global state_manager, question_loader, dialogue_manager

        question_data = question_loader.get_next_question(user_id, state_manager)

        if "message" in question_data:
            return question_data

        thought = question_data["thought"]
        sub_thought = question_data["sub_thought"]

        pleasantry = (
            "Hello! Let's start with this: "
            if sub_thought["sub_thought_index"] == 0
            else ""
        )
        generated_question = dialogue_manager.generate_question(
            description=sub_thought["description"],
            importance=sub_thought["importance"],
            pleasantry=pleasantry,
        )

        return {
            "question": generated_question,
            "thought_index": state_manager.get_state(user_id)["thought_index"],
            "sub_thought_index": state_manager.get_state(user_id)["sub_thought_index"],
            "thought": thought["thought"],
            "sub_thought_name": sub_thought["name"],
            "description": sub_thought["description"],
            "importance": sub_thought["importance"],
            "connection_to_next": sub_thought.get("connection_to_next"),
        }
    except Exception as e:
        logger.error(f"Error retrieving question: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving question: {str(e)}"
        )


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/synthesize_speech")
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech and return the audio as a Base64-encoded string.
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        mp3_fp = BytesIO()
        tts = gTTS(text=request.text, lang=request.language, slow=False)
        tts.write_to_fp(mp3_fp)

        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")

        return {"audio": audio_base64, "content_type": "audio/mp3"}
    except Exception as e:
        logger.error(f"Error in TTS processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")


@app.get("/test")
async def test():
    """Test endpoint to verify the server is running."""
    return {"status": "Server is running"}


# ================== Setup & Execution ==================
def setup_components(thought_data_file: str, state_file: str = "user_states.json"):
    """
    Sets up the core components for the application, including state management,
    question loading, and dialogue management.

    Args:
        thought_data_file (str): Path to the JSON file containing thoughts data.
        state_file (str): Path to the file used for persisting user states.

    Returns:
        tuple: A tuple containing (state_manager, question_loader, dialogue_manager).

    Raises:
        FileNotFoundError: If the thoughts data file is not found.
        ValidationError: If the thoughts data fails schema validation.
        Exception: For any other unexpected errors.
    """
    try:
        # Load thoughts data into the proper format
        logger.info(f"Loading thoughts data from: {thought_data_file}")
        json_content = load_thoughts_data(thought_data_file)

        # Initialize core components
        state_manager = StateManager(storage_path=state_file)
        question_loader = QuestionLoader(data_source=json_content)
        dialogue_manager = DialogueManager(llm_client=openai)

        logger.info("Core components initialized successfully.")
        return state_manager, question_loader, dialogue_manager

    except FileNotFoundError as e:
        logger.error(f"Thoughts data file not found: {thought_data_file}")
        raise e
    except ValidationError as e:
        logger.error(f"Thoughts data validation error: {e.json()}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during component setup: {str(e)}")
        raise e


def main(thought_data_file: str, state_file: str = "user_states.json"):
    """
    Main execution function for setting up and running the application.

    Args:
        thought_data_file (str): Path to the JSON file containing thoughts data.
        state_file (str): Path to the file used for persisting user states.
    """
    try:
        # Set up core components
        state_manager, question_loader, dialogue_manager = setup_components(
            thought_data_file, state_file
        )

        # Initialize IndexedThoughtReader for detailed interactions
        thought_reader = IndexedThoughtReader(thought_data_file)

        # Retrieve and log the main idea
        idea = thought_reader.get_idea()
        logger.info(f"Main Idea: {idea}")

        # Retrieve and log indexed thoughts
        list_of_thoughts = thought_reader.get_thoughts()
        logger.info(f"Indexed Thoughts: {json.dumps(list_of_thoughts, indent=4)}")

        # Example of fetching a question for a user
        user_id = "example_user"
        question_data = question_loader.get_next_question(user_id, state_manager)

        if "message" in question_data:
            logger.info(f"No more questions for user {user_id}")
        else:
            thought = question_data["thought"]
            sub_thought = question_data["sub_thought"]
            logger.info(
                f"Next Question for {user_id}: Thought: {thought['thought']}, "
                f"Sub-Thought: {sub_thought['name']}"
            )

    except Exception as e:
        logger.error(f"Application failed to execute: {str(e)}")


if __name__ == "__main__":
    # Specify the file paths

    # !Customize thought data file here:
    # for loading topics to generate questions
    thought_data_file_name = (
        "array_of_thoughts_output_with_index_embedded_software_development_openai.json"
    )
    thought_data_dir = OPENAI_INDEXED_MODELS_DIR
    thought_data_file = Path(thought_data_dir / thought_data_file_name)

    # State file

    state_file = INTERVIEW_STATES_FILE  # Path to the state file

    # Run the main function
    main(thought_data_file, state_file)
