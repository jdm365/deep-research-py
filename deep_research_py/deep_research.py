from typing import List, Dict, TypedDict, Optional, Union
from dataclasses import dataclass
import asyncio
import openai
## from ollama import chat
from deep_research_py.llm_query import Gemini, Ollama

from deep_research_py.data_acquisition.services import search_service, DuckDuckGoService
from deep_research_py.ai.providers import trim_prompt, get_client_response
from deep_research_py.prompt import system_prompt
from tqdm import tqdm
import json

MODEL = "gemma3:12b"


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]


@dataclass
class SerpQuery:
    query: str
    research_goal: str

def generate_serp_queries_local(
    client: Union[Ollama, Gemini],
    query: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Generate SERP queries based on user input and previous learnings."""

    ## prompt = f"""Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a JSON object with a 'queries' array field containing {num_queries} queries (or less if the original prompt is clear). Each query object should have 'query' and 'research_goal' fields. Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>"""
    prompt = f"""Given the following facility from the user, generate a list of SERP queries to research the topic with the goal of finding likely suppliers, materials supplied, and transportation method. First identify likely input materials to their products, then search nearby for facilities which manufacture or supply those materials. Return a JSON object with a 'queries' array field containing {num_queries} queries (or less if the original prompt is clear). Each query object should have 'query' and 'research_goal' fields. Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>"""

    if learnings:
        prompt += f"\n\nHere are some learnings from previous research, use them to generate more specific queries: {' '.join(learnings)}"

    '''
    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        format="json",
        stream=False,
    )["message"].content

    queries = json.loads(response).get("queries", [])
    '''

    queries = client.query_json(
            user_prompt=prompt,
            system_prompt=system_prompt(),
            stream=False,
            )["queries"]

    queries = [
            {
                "query": q["query"],
                "research_goal": q["research_goal"],
            }
            for q in queries
            if q["query"] and q["research_goal"]
            ]
    return [SerpQuery(**q) for q in queries][:num_queries]



def process_serp_result_local(
    client: Union[Ollama, Gemini],
    query: str,
    search_result: List[Dict[str, str]],
    num_learnings: int = 2,
    num_follow_up_questions: int = 1,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    contents = [
        trim_prompt(item.get("content", ""), 25_000)
        for item in search_result
        if item.get("content")
    ]

    # Create the contents string separately
    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    prompt = (
        f"Given the following contents from a SERP search for the query <query>{query}</query>, "
        f"generate a list of learnings from the contents. Return a JSON object with 'learnings' "
        f"and 'followUpQuestions' keys with array of strings as values. Include up to {num_learnings} learnings and "
        f"{num_follow_up_questions} follow-up questions. The learnings should be unique, "
        "concise, and information-dense, including entities, metrics, numbers, and dates.\n\n"
        f"<contents>{contents_str}</contents>"
    )

    '''
    response = json.loads(chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        format="json",
        stream=False,
    )["message"].content)
    '''

    response = client.query_json(
            user_prompt=prompt,
            system_prompt=system_prompt(),
            stream=False,
            )

    return {
        "learnings": response["learnings"][:num_learnings],
        "followUpQuestions": response["followUpQuestions"][
            :num_follow_up_questions
        ],
    }

def get_predicted_facilities_local(
    client: Union[Ollama, Gemini],
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
) -> str:
    """Generate final report based on all research learnings."""

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )

    user_prompt = (
        f"Given the following facility provided by the user, provide at least 10 nearby facilities "
        f"which likely supply them materials. Return JSON objects with a 'facilities' array field"
        f"containing objects with fields 'name', 'address', 'materials', 'transportation method', and 'evidence/rationale'."

        f"from research:\n\n<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )

    '''
    response = json.loads(chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        stream=False,
    )["message"].content)
    '''

    response = client.query_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt(),
            stream=False,
            )

    try:
        candidate_facilities = response.get("facilities", [])

        return candidate_facilities
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return "Error generating report"

def write_final_report_local(
    client: Union[Ollama, Gemini],
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
) -> str:
    """Generate final report based on all research learnings."""

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )

    user_prompt = (
        f"Given the following prompt from the user, write a final report on the topic using "
        f"the learnings from research. Return a JSON object with a 'reportMarkdown' field "
        f"containing a detailed markdown report (aim for 3+ pages). Include ALL the learnings "
        f"from research:\n\n<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )

    '''
    response = json.loads(chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        stream=False,
    )["message"].content)
    '''

    response = client.query_json(
            user_prompt=prompt,
            system_prompt=system_prompt(),
            stream=False,
            )

    try:
        report = response.get("reportMarkdown", "")

        # Append sources
        urls_section = "\n\n## Sources\n\n" + "\n".join(
            [f"- {url}" for url in visited_urls]
        )
        return report + urls_section
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return "Error generating report"


def deep_research_local(
    gemini_client: Gemini,
    ollama_client: Ollama,
    query: str,
    breadth: int,
    depth: int,
    learnings: List[str] = [],
    visited_urls: List[str] = [],
) -> ResearchResult:
    """
    Main research function that recursively explores a topic.

    Args:
        query: Research query/topic
        breadth: Number of parallel searches to perform
        depth: How many levels deep to research
        learnings: Previous learnings to build upon
        visited_urls: Previously visited URLs
    """
    # Generate search queries
    serp_queries = generate_serp_queries_local(
        client=ollama_client,
        query=query,
        num_queries=breadth,
        learnings=learnings,
    )

    ddgs = DuckDuckGoService()
    def process_query(serp_query: SerpQuery) -> ResearchResult:
        # Search for content
        result = ddgs.search(serp_query.query, limit=5)

        # Collect new URLs
        new_urls = [
            item.get("url") for item in result if item.get("url")
        ]

        # Calculate new breadth and depth for next iteration
        new_breadth = max(1, breadth // 2)
        new_depth = depth - 1

        # Process the search results
        new_learnings = process_serp_result_local(
            client=ollama_client,
            query=serp_query.query,
            search_result=result,
            num_follow_up_questions=new_breadth,
        )

        all_learnings = learnings + new_learnings["learnings"]
        all_urls = visited_urls + new_urls

        # If we have more depth to go, continue research
        if new_depth > 0:
            print(
                f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}"
            )

            next_query = f"""
            Previous research goal: {serp_query.research_goal}
            Follow-up research directions: {" ".join(new_learnings["followUpQuestions"])}
            """.strip()

            return deep_research_local(
                gemini_client=gemini_client,
                ollama_client=ollama_client,
                query=next_query,
                breadth=new_breadth,
                depth=new_depth,
                learnings=all_learnings,
                visited_urls=all_urls,
            )

        return {"learnings": all_learnings, "visited_urls": all_urls}

    results = [process_query(query) for query in tqdm(serp_queries, desc="Processing queries")]

    all_learnings = list(
        set(learning for result in results for learning in result["learnings"])
    )

    all_urls = list(set(url for result in results for url in result["visited_urls"]))

    return {"learnings": all_learnings, "visited_urls": all_urls}



if __name__ == "__main__":
    # Example usage
    ## query = "What materials does Afton Chemical in Sauget / East St. Louis, IL use to manufacture their products? Where do they likely receive truck shipments from? What are the most common sic codes of companies that ship to them? What is afton's sic? Please provide a list of at least 5 nearby facilities which are likely candidates to directly ship materials to Afton Chemical. Please include their name, address, and the likely materials shipped."
    query = "Afton Chemical in Sauget / East St. Louis, IL"
    breadth = 2

    '''
    serp_queries = generate_serp_queries_local(
            query=query,
            num_queries=breadth,
            learnings=None,
            )
    print("Generated SERP Queries:")
    for serp_query in serp_queries:
        print(f"Query: {serp_query.query}, Research Goal: {serp_query.research_goal}")
    '''

    gemini_client = Gemini()
    ## ollama_client = Ollama(model="gemma3:12b")
    ollama_client = Ollama(model="qwen3:14b")

    # Example usage of deep_research
    depth = 2
    results = deep_research_local(
        gemini_client=gemini_client,
        ollama_client=ollama_client,
        query=query,
        breadth=breadth,
        depth=depth,
    )
    from pprint import pprint
    print("Research Results:")
    pprint(results)

    '''
    # Example usage of write_final_report
    md_result = write_final_report_local(
        prompt=query,
        learnings=results["learnings"],
        visited_urls=results["visited_urls"],
        client=None,
    )
    with open("report.md", "w") as f:
        f.write(md_result)
    print("Final Report:")
    print(md_result)
    '''

    # Example usage of get_predicted_facilities
    predicted_facilities = get_predicted_facilities_local(
        client=gemini_client,
        prompt=query,
        learnings=results["learnings"],
        visited_urls=results["visited_urls"],
    )
    print("Predicted Facilities:")
    for fac in predicted_facilities:
        print(json.dumps(fac, indent=2))
