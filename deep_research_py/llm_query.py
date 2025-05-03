import google
from google import genai
from ollama import chat

from typing import Optional
import json
import os



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

        response = response.text.replace("```json", "").replace("```", "").strip()

        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"`query_json()` ASSUMES YOU ARE REQUESTING A JSON RESPONSE OBJECT")
            print(f"Raw response: {response}")
            raise e

        return response

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
    def __init__(self, model: str = "gemma3"):
        self.model = model

    def query_json(self, user_prompt: str, system_prompt: Optional[str] = None, stream: bool = False) -> str:
        prompt = []

        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": user_prompt})

        response = chat(model=self.model, messages=prompt, stream=stream)

        result = response["message"].content
        result = result.replace("```json", "").replace("```", "").strip()

        try:
            json_result = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"`query_json()` ASSUMES YOU ARE REQUESTING A JSON RESPONSE OBJECT")
            print(f"Raw response: {result}")
            raise e

        return json_result



if __name__ == "__main__":
    gemini = Gemini()
    ollama = Ollama()

    query = "Explain how AI works in a few words. Please respond with a json objec with a key 'response' and a value of the response."
    response = gemini.query_json(query)
    response = ollama.query_json(query)
    print(response)
