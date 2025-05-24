import os
import requests
from dotenv import load_dotenv

# Load the token from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file.")

MODEL_NAME = "google-t5/t5-base"
MODEL_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {hf_token}"}

class LLMModel:
    def __init__(self, 
                 model_name: str = MODEL_NAME, 
                 api_url: str = MODEL_API_URL, 
                 headers: dict = HEADERS):
        self.model_name = model_name
        self.api_url = api_url
        self.headers = headers

    def generate_response(self, query: str, temperature: float = 1, max_new_tokens: int = 100):
        payload = {
            "inputs": query,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        try:
            return clean_response(response.json())
        except:
            print("Failed to parse JSON. Error:", response.text)
            print("Response status code:", response.status_code)
            return None
        
def clean_response(response) -> str:
    """
    Cleans the response from the LLM to make it more readable.
    """
    if type(response) is list:
        response = response[0]
    if type(response) is dict:
        response = list(response.values())[0]
    return response

if __name__ == "__main__":
    llm_model = LLMModel()
    query = "A HTTP POST request is used to "
    response = llm_model.generate_response(query)
    print(response)