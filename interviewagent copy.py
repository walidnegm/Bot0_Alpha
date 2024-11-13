from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from pydantic import BaseModel
import base64
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO  # This was missing

import openai
import configparser
import os
import logging
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from .config file
config = configparser.ConfigParser()
config.read('/root/backend/config.ini')
openai.api_key = config.get("settings", "OPENAI_API_KEY")

config = configparser.ConfigParser()
config.read("config.ini")
api_key = config.get("settings", "OPENAI_API_KEY")

print("API key from config file", api_key)
openai.api_key = api_key

os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("value in environment", os.environ["OPENAI_API_KEY"])

client = openai.Client()

# Load JSON content from the file
#json_file_path = "/root/my-react-app/src/rank_of_sub_thoughts_1.json"
json_file_path = "/root/thoughtgeneration/Bot0_Release1/backend/input_output/thought_generation/array_of_thoughts_output_1_.json"

#json_file_path = "/rank_of_sub_thoughts_1.json"
print ("json_file_path: ", json_file_path)

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

@app.get("/get_question/{index}")
async def get_question(index: int):
    if index < 0 or index >= len(json_content["sub_concepts"]):
        raise HTTPException(status_code=404, detail="Index out of range.")

    description = json_content["sub_concepts"][index]["description"]
    context = (
        "You are an interview agent and you will evaluate the response, "
        "providing a score from 1 to 10 in terms of accuracy."
    )

    logging.info(f"Generating question based on description: {description}")

    try:
        # Use OpenAI to generate a question based on the description
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": f"Generate an interview question based on this description, but do not state you are an interview, just state the question: {description}"}
            ],
        )
        generated_question = response.choices[0].message.content.strip()

        logging.info("Received response from OpenAI API.")
        return {"question": generated_question}
    except Exception as e:
        logging.error(f"Error generating question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/synthesize_speech")
async def synthesize_speech(request: TTSRequest):
    try:
        logger.info(f"Received TTS request for text: {request.text[:50]}...")  # Log first 50 chars
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Create an in-memory bytes buffer
        mp3_fp = BytesIO()
        
        # Create gTTS object and save to buffer
        logger.info("Creating gTTS object...")
        tts = gTTS(text=request.text, lang=request.language, slow=False)
        
        logger.info("Writing to buffer...")
        tts.write_to_fp(mp3_fp)
        
        # Get the value and encode to base64
        logger.info("Encoding to base64...")
        mp3_fp.seek(0)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        
        logger.info("Successfully created audio")
        return {
            "audio": audio_base64,
            "content_type": "audio/mp3"
        }
        
    except Exception as e:
        logger.error(f"Error in TTS processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")

# Add a test endpoint
@app.get("/test")
async def test():
    return {"status": "TTS server is running"}
