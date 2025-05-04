import google
from google import genai
from ollama import chat
from pprint import pprint

from typing import Optional
import json
import demjson3
import os


def clean_and_read_json(text: str) -> dict:

    ## Identify quotes which should be escaped and escape
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return demjson3.decode(text)
    except demjson3.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw text: {text}")
        raise e

class Gemini:
    def __init__(self):
        self.client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
                )
        self.models = [
                "gemini-2.5-flash-preview-04-17",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemma3",
                ]
        self.model_idx = 0


    def query_json(
            self, 
            user_prompt: str, 
            system_prompt: Optional[str] = None, 
            stream: bool = False,
            attempt_idx: int = 0,
            ) -> str:
        prompt = f"{user_prompt}\n"

        if system_prompt is not None:
            system_prompt += "\nPlease wrap the json data in <json_object></json_object> tags. YOU MUST INCLUDE THESE TAGS!"
        else:
            system_prompt = "Please wrap the json data in <json_object></json_object> tags. YOU MUST INCLUDE THESE TAGS!"

        try:
            response = self.client.models.generate_content(
                model=self.models[self.model_idx], 
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=[system_prompt],
                ),
            )
        except Exception as e:
            self.model_idx = (self.model_idx + 1) % len(self.models)
            print(f"This is the rate limit exception: {e}")
            if attempt_idx == 2:
                print(f"Tried 3 models, but all failed. Exiting.")
                raise RuntimeError(f"Rate limit error: {e}") from e

            raise RuntimeError(e) from e

            return self.query_json(user_prompt, system_prompt, stream, attempt_idx + 1)

        try:
            json_data = clean_and_read_json(
                    response.text.split("<json_object>")[-1].split("</json_object>")[0].strip()
                    )
        except json.JSONDecodeError as e:
            print(f"`query_json()` ASSUMES YOU ARE REQUESTING A JSON RESPONSE OBJECT")
            print(f"Raw response: {response}")
            raise e

        print(f"Response: {json.dumps(json_data, indent=2)}")
        return json_data 

    def query(
            self, 
            user_prompt: str, 
            system_prompt: Optional[str] = None, 
            stream: bool = False,
            attempt_idx: int = 0,
            ) -> str:
        prompt = f"{user_prompt}\n"

        try:
            if system_prompt:
                response = self.client.models.generate_content(
                    model=self.models[self.model_idx], 
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=[system_prompt],
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.models[self.model_idx], 
                    contents=prompt,
                )
        except Exception as e:
            self.model_idx = (self.model_idx + 1) % len(self.models)
            print(f"This is the rate limit exception: {e}")
            if attempt_idx == 2:
                print(f"Tried 3 models, but all failed. Exiting.")
                raise e

            return self.query_json(user_prompt, system_prompt, stream, attempt_idx + 1)

        return response


class Ollama:
    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    def query_json(self, user_prompt: str, system_prompt: Optional[str] = None, stream: bool = False) -> str:
        prompt = []

        if system_prompt is not None:
            system_prompt += "\nPlease wrap the json data in <json_object></json_object> tags. YOU MUST INCLUDE THESE TAGS!"
        else:
            system_prompt = "Please wrap the json data in <json_object></json_object> tags. YOU MUST INCLUDE THESE TAGS!"

        prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": user_prompt})
        print(f"Prompt: {prompt}")

        response = chat(model=self.model, messages=prompt, stream=stream)["message"].content

        try:
            json_data = clean_and_read_json(
                    response.split("<json_object>")[-1].split("</json_object>")[0].strip()
                    )

        except json.JSONDecodeError as e:
            print(f"`query_json()` ASSUMES YOU ARE REQUESTING A JSON RESPONSE OBJECT")
            print(f"Raw response: {response}")
            raise e

        print(f"Response: {json.dumps(json_data, indent=2)}")

        assert isinstance(json_data, dict), f"Expected a dictionary but got {type(json_data)}"

        return json_data 



if __name__ == "__main__":
    gemini = Gemini()
    ollama = Ollama()

    query = (
            "Explain how AI works in a few words. "
            "Please respond with a json object with a key 'response' and a value of the response."
            )
    response = gemini.query_json(query)
    response = ollama.query_json(query)
    print(response)
