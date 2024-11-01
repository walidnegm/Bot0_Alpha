from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
json_file_path = "/root/my-react-app/src/sub_thoughts_output_0.json"
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
