import requests
import openai
import configparser
from openai import OpenAI, OpenAIError
import openai
import configparser
import os
from configparser import ConfigParser


class LLamaClientNV:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": """You are a vehicle diagnostics expert and expert summary creator.
                You will be provided a querey and its results.
                Use your understanding to create an appropriate response summary based on the results for the query.
                The response has to be in english and replace bullet points or syntax with approprirate words.
                The response must contain 40 words or less characters.
                The response must not contain the text either from the query or results."""
            }
        ]


    def setup_openai(self):
        try:
            if self.client is None:  # Only set up if not already initialized
                self.api_key = self.config.get("settings", "OPENAI_API_KEY")
                openai.api_key = self.api_key  # Set globally for OpenAI library
                self.client = openai.Client()  # Initialize OpenAI client
                print("OpenAI client initialized.")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise ValueError(f"Failed to load OpenAI API key from config file: {e}")

    def generate_completion(self, user_message, result, mode):
        """
        Unified method to generate completion based on the specified mode.
        :param mode: 'llamacpp' for local server, 'openai' for OpenAI API.
        """
        if mode == "llamacpp":
            return self.generate_completion_llamacpp(user_message, result)
        elif mode == "openai":
            return self.generate_completion_openai(user_message, result)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'llamacpp' or 'openai'.")
        

    def generate_completion_llamacpp(self, user_message, result):
        # Append the "user" message to the conversation
       

        base_url = "http://127.0.0.1:8080"  # Static URL for the LlamaCpp server
    
        prompt_text = f"""
            "Query": {user_message},
            "Results": {result}
            """
        print(f"Sending to LlamaCpp: {prompt_text}")
        self.messages.append({"role": "user", "content": prompt_text})

        response = requests.post(
            f"{base_url}/api/chat/",
            json={
                "messages": self.messages,
                "model": "llama3.2",  # Replace with your model name if different
                "stream": False
            }
        )
       
        if response.status_code == 200:
            response_data = response.json()
            assistant_message = response_data["message"]["content"]
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
        
    def generate_completion_openai(self, user_message, result):
        """
        Generate completion using OpenAI API.
        """
         # Load OpenAI API key from .config file
        config = configparser.ConfigParser()
        config.read("config.ini")
    
        try:
            openai_api_key = config.get("settings", "OPENAI_API_KEY")
            api_key = config.get("settings", "OPENAI_API_KEY")

            print("API key from config file", api_key)
            openai.api_key = api_key

            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = os.getenv("OPENAI_API_KEY")
            print("value in environment", os.environ["OPENAI_API_KEY"])

            global_openai_client = openai.Client()

            print("Global OpenAI client initialized.")

        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise ValueError(f"Failed to load OpenAI API key from config file: {e}")      
        
        prompt_text = f"""
            "Query": {user_message},
            "Results": {result}
            """
        print(f"Sending to OpenAI: {prompt_text}")
        self.messages.append({"role": "user", "content": prompt_text})

        try:
            response = global_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Adjust the model if needed
                messages=self.messages
            )
            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            print (assistant_message)
            return assistant_message

        except OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return None
 