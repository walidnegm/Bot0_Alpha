"""
Refactored interview agent module with modular state management and dialogue handling.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from pydantic import BaseModel, ValidationError
from io import BytesIO
import base64
import openai
import configparser
import os
import logging
import logging_config
import json
from typing import Optional
from datetime import datetime
import getpass
from dotenv import load_dotenv
import uvicorn

# Import internal modules
from thought_generation.thought_reader import IndexedThoughtReader
from config import OPENAI_INDEXED_MODELS_DIR, INTERVIEW_STATES_FILE

# Set up logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== Helper Functions ==================
def get_openai_api_key():
    """Load the OpenAI API key from environment or configuration file."""
    current_user = getpass.getuser()
    xf_username = "xzhan"

    if current_user == xf_username:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("Loaded API key from .env")
            return api_key

    config = configparser.ConfigParser()
    config.read("/root/backend/config.ini")
    api_key = config.get("settings", "OPENAI_API_KEY", fallback=None)
    if api_key:
        logger.info("Loaded API key from config file")
        return api_key

    raise ValueError("OpenAI API key not found.")


# ================== Core Classes ==================
class StateManager:
    """Manages user states for the interview bot."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or INTERVIEW_STATES_FILE
        self.states = self._load_states()

    def _load_states(self):
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def persist(self):
        """Persist states to a file if a storage path is provided."""
        with open(self.storage_path, "w") as f:
            json.dump(self.states, f)

    def get_state(self, user_id: str):
        """Retrieve or initialize a user's state."""
        if user_id not in self.states:
            self.states[user_id] = {
                "thought_index": 0,
                "sub_thought_index": 0,
                "current_question": None,
                "last_updated": datetime.utcnow().isoformat(),
            }
        return self.states[user_id]

    def update_state(self, user_id: str, **updates):
        """Update the state of a user."""
        state = self.get_state(user_id)
        state.update(updates)
        state["last_updated"] = datetime.utcnow().isoformat()
        self.persist()

    def reset_state(self, user_id: str):
        """Reset a user's state."""
        if user_id in self.states:
            del self.states[user_id]
        self.persist()


class QuestionLoader:
    """Loads and manages questions from hierarchical data."""

    def __init__(self, data_file: Path):
        self.data_source = self._load_data(data_file)

    def _load_data(self, data_file: Path):
        reader = IndexedThoughtReader(data_file)
        return reader.dump_all()

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

            return {"thought": thought, "sub_thought": sub_thought}
        except IndexError:
            state_manager.reset_state(user_id)
            return {"message": "No more questions available."}


class DialogueManager:
    """Handles interactions with OpenAI to generate questions."""

    def __init__(self):
        api_key = get_openai_api_key()
        openai.api_key = api_key
        self.llm_client = openai

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
async def get_question_async(user_id: str):
    """
    Retrieve the next question for a user based on their state.

    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        dict: JSON response containing the generated question and metadata.
    """
    try:
        state_manager = StateManager()
        question_loader = QuestionLoader(
            data_file=Path(OPENAI_INDEXED_MODELS_DIR / "thoughts.json")
        )
        dialogue_manager = DialogueManager()

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
        logger.error(f"Error retrieving question for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving question: {str(e)}"
        )


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/synthesize_speech")
async def synthesize_speech(request: TTSRequest):
    """Convert text to speech and return the audio as a Base64-encoded string."""
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
        logger.error(f"Error in TTS processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")


@app.get("/test")
async def test():
    """Test endpoint to verify the server is running."""
    return {"status": "Server is running"}


if __name__ == "__main__":
    uvicorn.run(
        "interviewagent_xf_edit_2:app", host="127.0.0.1", port=8000, reload=True
    )
