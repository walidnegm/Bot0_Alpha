from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import configparser
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from configparser import ConfigParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from .config file
config = configparser.ConfigParser()
config.read('/root/backend/config.ini')
openai.api_key = config.get("settings", "OPENAI_API_KEY")


config = ConfigParser()
config.read("config.ini")
api_key = config.get("settings", "OPENAI_API_KEY")

print("API key from config file", api_key)
openai.api_key = api_key

os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("value in enviromnet", os.environ["OPENAI_API_KEY"])

client = openai.Client()



class LLMRequest(BaseModel):
    transcription_text: str  # Text content for processing instead of file name

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}

@app.post("/process_llm")
async def process_llm(request: LLMRequest):
    transcription_text = request.transcription_text
    print (transcription_text)
    logging.info(f"Received transcription text: {transcription_text}")

    logging.info("Sending transcription text to OpenAI API...")
    try:

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": transcription_text}
            ],
        )
        llm_response = model_response = response.choices[0].message.content

        logging.info("Received response from OpenAI API.")
    except Exception as e:
        logging.error(f"Error processing LLM request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing LLM request: {str(e)}")

    return {"response": llm_response}