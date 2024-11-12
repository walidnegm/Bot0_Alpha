from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import configparser
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from configparser import ConfigParser

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from .config file
config = configparser.ConfigParser()
config.read('/root/backend/config.ini')
openai_api_key = config.get("settings", "OPENAI_API_KEY")

config = ConfigParser()
config.read("config.ini")
api_key = config.get("settings", "OPENAI_API_KEY")

print("API key from config file", api_key)
openai.api_key = api_key

os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("value in environment", os.environ["OPENAI_API_KEY"])

client = openai.Client()

class LLMRequest(BaseModel):
    transcription_text: str
    question: str
    question_id: int

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}

class QuestionContext:
    def __init__(self, question: str, question_id: int):
        self.question = question
        self.question_id = question_id
        self.clarification_count = 0
        self.repeat_count = 0
        self.previous_responses = []

# Store active question contexts
question_contexts = {}

@app.post("/process_llm")
async def process_llm(request: LLMRequest):
    transcription_text = request.transcription_text
    question = request.question
    question_id = request.question_id
    
    if question_id not in question_contexts:
        question_contexts[question_id] = QuestionContext(question, question_id)
    
    context = question_contexts[question_id]
    
    logging.info(f"Processing response for question ID: {question_id}")
    logging.info(f"Transcription text: {transcription_text}")

    try:
        # Modified intent detection to include "next question" requests
        intent_context = (
            "You are an interview agent. Determine if the interviewee's response is: "
            "1. An attempt to answer the question "
            "2. A request for clarification "
            "3. A request to repeat the question "
            "4. A request to move to the next question "
            "Consider the full context including previous clarifications and responses. "
            "Respond ONLY with either 'answer', 'clarification', 'repeat', or 'next'."
        )
        
        intent_prompt = (
            f"Question: {question}\n"
            f"Current response: {transcription_text}\n"
            f"Previous clarifications: {context.clarification_count}\n"
            f"Previous repeats: {context.repeat_count}\n"
            "What is the intent of this response?\n\n"
            "Note: If the response includes phrases like 'next question', 'move on', "
            "'go to the next one', or similar requests to proceed, respond with 'next'."
        )

        intent_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": intent_context},
                {"role": "user", "content": intent_prompt}
            ],
        )
        
        interaction_type = intent_response.choices[0].message.content.strip().lower()
        logging.info(f"Detected interaction type: {interaction_type}")

        # Handle "next" request
        if interaction_type == "next":
            return {
                "response": "next_question",  # Special response to trigger next question in frontend
                "message": "Moving to the next question."
            }

        # Update context based on interaction type
        if interaction_type == "clarification":
            context.clarification_count += 1
            response_context = (
                "You are an interview agent. The interviewee has asked for clarification. "
                f"This is clarification request #{context.clarification_count} for this question. "
                "Provide a progressively more detailed explanation while remaining fair. "
                "Do not give away answers."
            )
            prompt = (
                f"Original Question: {question}\n"
                f"Clarification Request: {transcription_text}\n"
                f"Previous clarifications given: {context.clarification_count - 1}\n"
                "Please provide an appropriate level of clarification."
            )
        
        elif interaction_type == "repeat":
            context.repeat_count += 1
            if context.repeat_count > 2:
                return {
                    "response": (
                        f"I'll repeat the question once more: {question}\n"
                        "If you're having trouble hearing, please let me know and we can try to adjust."
                    )
                }
            return {"response": f"Here's the question again: {question}"}
        
        else:  # answer
            context.previous_responses.append(transcription_text)
            response_context = (
                "You are an interview agent evaluating the interviewee's response. "
                "This may be a follow-up or modified answer to their previous responses. "
                "Consider the entire context when scoring. "
                "Provide a score from 1 to 10 and explain why you gave that score."
            )
            prompt = (
                f"Question: {question}\n"
                f"Current Answer: {transcription_text}\n"
                f"Previous Answers: {context.previous_responses[:-1]}\n"
                f"Number of clarifications requested: {context.clarification_count}\n"
                "Please evaluate this response considering the full context."
            )

        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": response_context},
                {"role": "user", "content": prompt}
            ],
        )
        
        return {"response": final_response.choices[0].message.content}
        
    except Exception as e:
        logging.error(f"Error in process_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to start a new question
@app.post("/new_question")
async def new_question(question_id: int):
    if question_id in question_contexts:
        del question_contexts[question_id]
    return {"status": "success", "message": "Question context reset"}