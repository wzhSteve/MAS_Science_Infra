import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from .local_google_scraper import local_google_search
    
from MAS.epc_aw.tools.base import BaseTool
from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.tools.web_search.tool import Web_Search_Tool
import requests
from typing import List
import re

TOOL_NAME = "Google_Search_Tool"

LIMITATIONS = """
1. This tool is only suitable for general information search.
2. This tool contains less domain specific information.
3. This tools is not suitable for searching and analyzing videos at YouTube or other video platforms.
"""

BEST_PRACTICES = """
1. Choose this tool when you want to search general information about a topic or a keyward.
2. Choose this tool for question type of query, such as "What is the capital of France?" or "What is the capital of France?"
3. The tool will return a summarized information.
4. This tool is more suitable for definition, world knowledge, and general information search.
"""

# -------------------------------------------------------
# 新增：如果 LOCAL_SEARCH=1，则使用你的本地抓取器
# -------------------------------------------------------
USE_LOCAL = True

class Google_Search_Tool(BaseTool):
    def __init__(self, model_string=os.getenv("MODEL_Name")):
        super().__init__(
            tool_name=TOOL_NAME,
            # tool_description="A web search tool powered by Google's Gemini AI that provides real-time information from the internet with citation support.",
            tool_description="A broad-spectrum, Google-powered search tool for retrieving fresh, up-to-date information from across the internet. It accepts any type of query—keywords, topics, questions, or vague descriptions—and does not require URLs or entity grounding. This tool is the default choice for the first lookup (step_count = 1), where open-domain discovery is needed.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "add_citations": "bool - Whether to add citations to the results. If True, the results will be formatted with citations. By default, it is True.",
            },
            output_type="str - The search results of the query.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the capital of France?")',
                    "description": "Search for general information about the capital of France with default citations enabled."
                },
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?", add_citations=False)',
                    "description": "Search for information about Euro 2024 winner without citations."
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society article arXiv August 11, 2016", add_citations=True)',
                    "description": "Search for specific academic articles with citations enabled."
                }
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 5
        self.search_model = model_string
        self.web_rag_tool = Web_Search_Tool(model_string=model_string)
        if USE_LOCAL:
            # NOTE: deterministic mode
            self.client = create_llm_engine(
                model_string=self.model_string, 
                temperature=0.0, 
                top_p=1.0, 
                frequency_penalty=0.0, 
                presence_penalty=0.0
                )
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise Exception("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
            self.client = genai.Client(api_key=api_key)


    # ------------------------------------------------------
    # 新增：本地 Google 抓取执行
    # ------------------------------------------------------
    def _execute_local_search(self, query: str, add_citations_flag: bool = True):
        # Step 1: 原始 JSON（自然结果）
        raw_result = local_google_search(query)
        excluded_keywords = ["hugging", "openai", "grok", 'gaia', 'bamboogle']
        # excluded_keywords = ["openai", "grok"]
        
        page_list = [
            item 
            for item in raw_result.get("organic", [])
            # 使用 all() 和 generator expression 来确保链接不包含任何一个排除关键词
            if all(keyword not in item.get("link", "").lower() for keyword in excluded_keywords)
        ]
        for i in range(len(page_list)):
            url = page_list[i].get("link", "")
            abstract = self.web_rag_tool.execute(url=url, query=query)
            page_list[i]["abstract"] = abstract
        raw_result["organic"] = page_list

        # Step 2: 转成 JSON 供 LLM 输入
        json_text = json.dumps(raw_result, ensure_ascii=False, indent=2)

        # Step 3: 用 LLM 做总结 + citation
        prompt = f"""
You are a Search Result Summarization Model.

You will receive raw search results from a Google-style scraper in JSON format.
Each item includes:
	•	title
	•	link
	•	snippet

Your job is to produce an accurate, concise, citation-supported summary that answers the user query strictly using the information visible in the JSON.

Hard Rules (must follow exactly):
	1.	No hallucination.
Every statement must be directly supported by the text in title or snippet.
	2.	No external knowledge.
If the JSON does not contain the needed information, say so explicitly.
	3.	Citations required.
Use markdown citation style: [index](url)
	•	index = position in the JSON array, starting from 1
	•	Cite immediately after any statement supported by a title/snippet.
	4.	No rewriting of missing content.
If a snippet is vague or incomplete, summarize only what is present.
Do not infer causes, conclusions, motivations, or definitions that aren’t explicitly stated.
	5.	Natural language paragraph.
The final answer must be a short, coherent paragraph—not a list and not a bullet-point summary.
	6.	Duplicate links.
Treat repeated URLs as separate results unless the snippet text is identical.
	7.	Unanswerable queries.
If the JSON contains no relevant information for the query, respond:
“The search results do not provide information that answers this query.”

⸻

Required Output Format

Your output must include:
	1.	A direct answer to the user query (if possible from the JSON).
	2.	Citations immediately after each factual statement.
	3.	A single cohesive paragraph—no bullets, no headings.

⸻

User Query:

“{query}”

Raw Search Results (JSON):

{json_text}

⸻

Now produce the final summarized answer with citations.
                    """

        # Step 4: 调用 LLM
        summary = self.client(prompt)

        # summary = response.text

        # Step 5: 对 URL 做跳转清洗（与原版保持一致）
        try:
            summary = self.reformat_response(summary)
        except:
            pass

        # 增加引用
        indices = [int(x) for x in re.findall(r'\[(\d+)\]', summary)]
        unique_indices = set(indices)
        if add_citations_flag and unique_indices:
            citation_texts = []
            for idx in sorted(unique_indices):
                if 1 <= idx <= len(page_list):
                    item = page_list[idx - 1]
                    title = item.get("title", "No Title")
                    link = item.get("link", "No Link")
                    snippet = item.get("snippet", "No Snippet")
                    citation_texts.append(f"[{idx}] Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
            citation_section = "\n\nCitations:\n" + "\n".join(citation_texts)
            summary += citation_section
        
        return summary

    
    @staticmethod
    def get_real_url(url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
            return response.url
        except:
            return url

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        pattern = re.compile(r'\[\d+\]\((https?://[^\s)]+)\)')
        return pattern.findall(text)

    def reformat_response(self, response: str) -> str:
        urls = self.extract_urls(response)
        for url in urls:
            direct_url = self.get_real_url(url)
            response = response.replace(url, direct_url)
        return response

    @staticmethod
    def add_citations(response):
        text = response.text
        supports = response.candidates[0].grounding_metadata.grounding_supports
        chunks = response.candidates[0].grounding_metadata.grounding_chunks

        sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")
                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]
        return text


    def _execute_search(self, query: str, add_citations_flag: bool = True):
        # ------------------------------------------------------
        # 最小改动：如果 LOCAL_SEARCH=1，则直接返回本地结果
        # ------------------------------------------------------
        if USE_LOCAL:
            return self._execute_local_search(query, add_citations_flag)

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        response = None
        text = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.search_model,
                    contents=query,
                    config=config,
                )
                text = response.text
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Google Search failed after {self.max_retries} attempts. Error: {str(e)}"

        if response is None:
            return "Google Search failed to get valid response."

        if add_citations_flag:
            try:
                text = self.add_citations(response)
            except:
                pass

        try:
            text = self.reformat_response(text)
        except:
            pass

        return text


    def execute(self, query: str, add_citations: bool = True) -> str:
        return self._execute_search(query, add_citations)

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    """
    Test:
    cd agentflow/tools/google_search
    python tool.py
    """
    def print_json(result):
        import json
        print(json.dumps(result, indent=4))

    google_search = Google_Search_Tool()

    # Get tool metadata
    metadata = google_search.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)

    examples = [
        {'query': 'What is the capital of France?', 'add_citations': True},
        {'query': 'Who won the euro 2024?', 'add_citations': False},
        {'query': 'Physics and Society article arXiv August 11, 2016', 'add_citations': True},
    ]
    
    for example in examples:
        print(f"\nExecuting search: {example['query']}")
        try:
            result = google_search.execute(**example)
            print("Search Result:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)

    print("Done!")