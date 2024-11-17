import requests
import openai
import openai
import configparser
import os
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from .config file
config = configparser.ConfigParser()
config.read("./config.ini")
api_key = config.get("settings", "OPENAI_API_KEY")

if api_key:
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OpenAI API key loaded from config.")
else:
    raise ValueError("OpenAI API key not found in config file.")

# Initialize OpenAI client
client = openai

class LLamaClientNV:
    def __init__(self, base_url):
        self.base_url = base_url
        self.messages = [
            {
                "role": "system",
                "content": """You are a vehicle diagnostics expert and expert summary creator.
                You must understand the results and the reasons for the query.
                Use your understanding to create an appropriate response summary based on the results for the query.
                The response must be a clear and meaningful report to the query then discard those results and provide your
                summarized output in 40 words or less characters.
                The response must not contain the text either from the query or results."""
            }
        ]
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )


    def generate_completion(self, user_message_result):
        # Append the "user" message to the conversation
        prompt_text = f"""
            "Query": {user_message},
            "Results":{results}
            """
        self.messages.append({"role": "user", "content": prompt_text})

        response = requests.post(
            f"{self.base_url}/api/chat/",
            json={"messages":self.messages, "model": "llama3.2", "stream": False}
        )

        if response.status_code == 200:
            print(response.json())
            assistant_message = response.json()["message"]["content"]
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        else:
            print("Error: " + str(response.status_code) + ": " + response.text)
            return None

    def generate_completion_llamaspp(self, user_message_result):
        # Append the "user" message to the conversation
        prompt_text = f"""
            "Query": {user_message},
            "Results": {results}
            """

        print(prompt_text)
        self.messages.append({"role": "user", "content": prompt_text})

        completion = self.client.chat_completion.create(
            model="llama3.2",
            messages=self.messages
        )

        self.messages.append({"role": "assistant", "content": completion.choices[0].message.content})
        return completion.choices[0].message.content