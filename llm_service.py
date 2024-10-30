from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import configparser
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load OpenAI API key from .config file
config = configparser.ConfigParser()
config.read('/root/backend/config.ini')
openai.api_key = config.get("settings", "OPENAI_API_KEY")

class LLMRequest(BaseModel):
    transcription_file: str  # Name of the file in the transcriptions folder
    
@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}

@app.post("/process_llm")
async def process_llm(request: LLMRequest):
    transcription_path = os.path.join("transcriptions", request.transcription_file)
    logging.info(f"Received request to process LLM for file: {request.transcription_file}")

    if not os.path.isfile(transcription_path):
        raise HTTPException(status_code=404, detail="Transcription file not found.")

    with open(transcription_path, "r") as file:
        transcription_text = file.read()

    logging.info("Transcription text loaded, sending to OpenAI API...")

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=transcription_text,
            max_tokens=100
        )
        llm_response = response.choices[0].text.strip()
        logging.info("Received response from OpenAI API.")

    except Exception as e:
        logging.error(f"Error processing LLM request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing LLM request: {str(e)}")

    return {"response": llm_response}
