from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import configparser
import os
import logging
import json
from PyPDF2 import PdfReader
# Create a FastAPI instance

#Refers to the Python file and the FastAPI app instance.
#If your Python file is named resume_service.py, it will look for the app object defined in that file.

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
openai_api_key = config.get("settings", "OPENAI_API_KEY")

config = configparser.ConfigParser()
config.read("config.ini")
api_key = config.get("settings", "OPENAI_API_KEY")

print("API key from config file", api_key)
os.environ["OPENAI_API_KEY"] = api_key
print("value in environment", os.environ["OPENAI_API_KEY"])

# Initialize OpenAI client
client = OpenAI()

class SkillsResponse(BaseModel):
    skills: dict
    skills_list: list

class SkillsResponse(BaseModel):
    skills: list

#Upload Resume Endpoint
#HTTP Method: POST API
#Description: Accepts a PDF file, extracts text using PdfReader, sends the text to OpenAI for skills extraction, and returns the extracted skills.
#uvicorn resume_service:app --host 0.0.0.0 --port 8002 &
#Sets the port number where the FastAPI app will listen for incoming requests.
#Port is the communication endpoint that allows devices and software to send and receive data over a network.


#Visiting http://localhost:8002/upload_resume means:
#IP address: example.com (or its equivalent IP).
#Port number: 8080.

@app.post("/upload_resume", response_model=SkillsResponse)
async def upload_resume(file: UploadFile = File(...)):
    try:
        # Read PDF content
        pdf_reader = PdfReader(file.file)
        pdf_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        if not pdf_text:
            raise ValueError("Failed to extract text from the uploaded PDF.")

        logger.info("Extracted text from PDF successfully.")

        # Query OpenAI to extract skills
        logger.info("Requesting skills extraction from OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract key skills from the following resume:"},
                {"role": "user", "content": pdf_text}
            ],
            max_tokens=150
        )
        llm_response_skills = response.choices[0].message.content.strip()

        # Log the response from OpenAI
        logger.info(f"OpenAI response: {llm_response_skills}")

        # Check if the response is empty
        if not llm_response_skills:
            raise ValueError("Received empty response from OpenAI.")

        # Handle the response as a list of skills
        skills_list = llm_response_skills.split("\n")
        skills_list = [skill.strip() for skill in skills_list if skill.strip()]

        if not skills_list:
            raise ValueError("No skills extracted from the OpenAI response.")

        return SkillsResponse(skills=skills_list)

    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")

#Test Endpoint
#Route: /test

@app.get("/test")
async def test():
    return {"status": "Resume microservice is running"}