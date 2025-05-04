from typing import List, Union
import json
from deep_research_py.prompt import system_prompt
from deep_research_py.ai.providers import get_client_response
from deep_research_py.deep_research import MODEL

from deep_research_py.llm_query import Gemini, Ollama


def generate_feedback(
        query: str,
        client: Union[Gemini, Ollama],
        ) -> List[str]:
    """Generates follow-up questions to clarify research direction."""

    '''
    response = chat(model=MODEL, messages=[
      {
        'role': 'system',
        'content': system_prompt(),
      },
    {
        "role": "user",
        "content": f"Given this research topic: {query}, generate 2-3 follow-up questions to better understand the user's research needs. Return the response as a JSON object with a 'questions' array field.",
    },
    ], format='json', stream=False)
    '''
    try:
        ## return json.loads(response["message"].content).get("questions", [])
        return client.query_json(
            user_prompt=f"Given this research topic: {query}, generate 2-3 follow-up questions to better understand the user's research needs. Return the response as a JSON object with a 'questions' array field.",
            system_prompt=system_prompt(),
        ).get("questions", [])

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return []


if __name__ == "__main__":
    query = "How does climate change affect marine biodiversity?"
    client = Ollama()
    questions = generate_feedback(query, client)
    print("Generated Feedback:")
    for question in questions:
        print(question)
