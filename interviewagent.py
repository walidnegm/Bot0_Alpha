from fastapi import FastAPI, HTTPException
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from .config file
config = configparser.ConfigParser()
config.read("/root/backend/config.ini")
api_key = config.get("settings", "OPENAI_API_KEY")

if api_key:
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OpenAI API key loaded from config.")
else:
    raise ValueError("OpenAI API key not found in config file.")

# Initialize OpenAI client
client = openai

# Load JSON content from the file
json_file_path = "thought_generation/claude_output/array_of_thoughts_output_claude_embedded_software_development.json"
try:
    with open(json_file_path, "r") as json_file:
        json_content = json.load(json_file)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="JSON file not found.")
except json.JSONDecodeError:
    raise HTTPException(status_code=500, detail="Error decoding JSON file.")


@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}


# Track state for user questioning
user_states = {}


@app.get("/get_question/{user_id}")
async def get_question(user_id: str):
    # Initialize state if user not in user_states
    if user_id not in user_states:
        user_states[user_id] = {"thought_index": 0, "sub_thought_index": 0}

    state = user_states[user_id]
    try:
        # Log current state
        logger.info(f"User {user_id} state: {state}")

        # Get current thought and sub_thoughts
        if state["thought_index"] >= len(json_content["thoughts"]):
            # No more thoughts available
            user_states.pop(user_id, None)
            return {"message": "No more questions available."}

        thought = json_content["thoughts"][state["thought_index"]]
        sub_thoughts = thought.get("sub_thoughts", [])

        # Move to the next thought if sub_thoughts are exhausted
        if state["sub_thought_index"] >= len(sub_thoughts):
            state["thought_index"] += 1
            state["sub_thought_index"] = 0

            # Check again for thought index overflow
            if state["thought_index"] >= len(json_content["thoughts"]):
                user_states.pop(user_id, None)
                return {"message": "No more questions available."}

            thought = json_content["thoughts"][state["thought_index"]]
            sub_thoughts = thought.get("sub_thoughts", [])

        # Get current sub-thought details
        sub_thought = sub_thoughts[state["sub_thought_index"]]
        description = sub_thought["description"]
        importance = sub_thought["importance"]

        # Determine if it's the first or second question for pleasantries
        pleasantry = ""
        if state["sub_thought_index"] == 0:
            pleasantry = "Hello! Let's start with this: "
        elif state["sub_thought_index"] == 1:
            pleasantry = "Great, now let's dive a bit deeper: "

        # OpenAI context and prompt
        intent_context = (
            "You are an interview agent and you will evaluate the response, "
            "providing a score from 1 to 10 in terms of accuracy."
        )
        intent_prompt = f"{pleasantry}Generate an interview question based on this description and importance: {description}, {importance}"

        # Generate the question using OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": intent_context},
                {"role": "user", "content": intent_prompt},
            ],
        )
        generated_question = response.choices[0].message.content.strip()

        # Log generated question
        logger.info(f"Generated question: {generated_question}")

        # Move to the next sub_thought
        state["sub_thought_index"] += 1

        return {"question": generated_question}
    except IndexError as e:
        logger.error(f"Index error: {e}")
        raise HTTPException(
            status_code=404, detail="Thought or sub-thought index out of range."
        )
    except Exception as e:
        logger.error(f"Error retrieving question: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving question: {str(e)}"
        )


# TTS Endpoint
class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/synthesize_speech")
async def synthesize_speech(request: TTSRequest):
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


# Test endpoint
@app.get("/test")
async def test():
    return {"status": "Server is running"}
