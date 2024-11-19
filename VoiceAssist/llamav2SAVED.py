import requests
import openai

class LLamaClientNV:
    def __init__(self, base_url):
        self.base_url = base_url
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
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="no key required"
        )


    def generate_completion(self, user_message, result):
        # Append the "user" message to the conversation
        
        prompt_text = f"""
            "Query": {user_message},
            "Results":{result}
            """
        self.messages.append({"role": "user", "content": prompt_text})

        response = requests.post(
            f"{self.base_url}/api/chat/",
            json={"messages":self.messages, "model": "llama3.2", "stream": False}
        )

        if response.status_code == 200:
            #print(response.json())
            assistant_message = response.json()['message']['content']
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        else:
            print("Error: {response.status_code} - {response.text}")
            return None

    def generate_completion_llamacpp(self, user_message, result):
        # Append the "user" message to the conversation
        
        prompt_text = f"""
            "Query": {user_message},
            "Results": {result}
            """
        print(prompt_text)
        self.messages.append({"role": "user", "content": prompt_text})

        completion = self.client.chat.completions.create(
            model="Llama3.2",
            messages=self.messages
       )
       
        self.messages.append({"role": "assistant", "content": completion.choices[0].message.content})
        return completion.choices[0].message.content

    def generate_completion_openai(self, user_message, result):
        # Construct the OpenAI API request
        prompt_text = f"""
            "Query": {user_message},
            "Results": {result}
            """
        print(f"Sending to OpenAI: {prompt_text}")

        self.messages.append({"role": "user", "content": prompt_text})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Adjust the model if needed
                messages=self.messages
            )

            assistant_message = response.choices[0].message["content"]
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return None


    #def generate_completion_llamacpp(self, user_message, result):
    # Create the prompt
    #    prompt_text = f"""
    #   "Query": {user_message},
    #   "Results": {result}
    #   """
    #   self.messages.append({"role": "user", "content": prompt_text})

    #    # Make a direct POST request to the local server
    #    response = requests.post(
    #        f"{self.base_url}/api/chat/",
    #       json={"messages": self.messages, "model": "llama3.2", "stream": False}
    #    )

    #    if response.status_code == 200:
    #       # Parse and return the response
    #       response_json = response.json()
    #       assistant_message = response_json["message"]["content"]
    #       self.messages.append({"role": "assistant", "content": assistant_message})
    #       return assistant_message
    #    else:
    #        print(f"Error {response.status_code}: {response.text}")
    #        return None