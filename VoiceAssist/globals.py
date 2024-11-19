import openai
import configparser
import os
from configparser import ConfigParser
global_openai_client = None

def setup_global_openai_client(config_path="./config.ini"):
    """
    Set up the global OpenAI client by loading the API key from the config file.
    """
    global global_openai_client
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