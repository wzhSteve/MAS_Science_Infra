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

# LOCAL_SEARCH=1 (default) uses local Chrome scraper; set 0 to use Gemini API search.
USE_LOCAL = os.getenv("LOCAL_SEARCH", "1").strip().lower() not in ("0", "false", "no")

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
        self._gemini_client = None
        # NOTE: deterministic mode for local-search summarization
        self.client = create_llm_engine(
            model_string=self.model_string,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    def _get_gemini_client(self):
        if self._gemini_client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Google API key not found. Please set GOOGLE_API_KEY for Gemini search fallback."
                )
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client


    # ------------------------------------------------------
    # 新增：本地 Google 抓取执行
    # ------------------------------------------------------
    # Default exclusion list is intentionally minimal: only filter known
    # benchmark-leakage domains. Heavy filtering (openai/hugging/grok) was
    # found to suppress legitimate academic results on GAIA. Override via
    # env GOOGLE_SEARCH_EXCLUDED_KEYWORDS (comma-separated) when needed.
    DEFAULT_EXCLUDED_KEYWORDS = ["gaia", "bamboogle"]

    def _resolve_excluded_keywords(self) -> list:
        env_val = os.getenv("GOOGLE_SEARCH_EXCLUDED_KEYWORDS", "")
        if env_val:
            return [k.strip().lower() for k in env_val.split(",") if k.strip()]
        return list(self.DEFAULT_EXCLUDED_KEYWORDS)

    def _execute_local_search(self, query: str, add_citations_flag: bool = True):
        raw_result = local_google_search(query)
        excluded_keywords = self._resolve_excluded_keywords()

        page_list = [
            item
            for item in raw_result.get("organic", [])
            if all(keyword not in item.get("link", "").lower() for keyword in excluded_keywords)
        ]
        if not page_list:
            return (
                "Error: local Google search returned no organic results "
                f"for query={query!r}. This is a scraper/coverage failure, not a verified absence."
            )

        max_abstract_pages = int(os.getenv("GOOGLE_SEARCH_MAX_ABSTRACT_PAGES", "3"))
        for i, item in enumerate(page_list[:max_abstract_pages]):
            url = item.get("link", "")
            if not url:
                continue
            try:
                abstract = self.web_rag_tool.execute(url=url, query=query)
                page_list[i]["abstract"] = abstract
            except Exception as e:
                page_list[i]["abstract"] = f"[abstract_fetch_failed] {e}"
        raw_result["organic"] = page_list

        # Step 2: 转成 JSON 供 LLM 输入
        json_text = json.dumps(raw_result, ensure_ascii=False, indent=2)

        # Step 3: 用 LLM 做总结 + citation
        prompt = f"""
You are a Search Result Summarization Model.

You will receive raw search results from a Google-style scraper in JSON format.
Each item may include:
	•	title
	•	link
	•	snippet
	•	abstract (optional; page content extracted for the query)

Your job is to produce an accurate, concise, citation-supported summary that answers the user query using the information visible in the JSON.

Hard Rules (must follow exactly):
	1.	No hallucination.
Every statement must be directly supported by title, snippet, OR abstract.
	2.	Prefer concrete facts from abstract when present.
If a snippet is vague but the abstract contains the requested value (distance, count, date, name, etc.), use the abstract.
	3.	No external knowledge.
If the JSON does not contain the needed information, say so explicitly.
	4.	Citations required.
Use markdown citation style: [index](url)
	•	index = position in the JSON array, starting from 1
	•	Cite immediately after any statement supported by a title/snippet/abstract.
	5.	No rewriting of missing content.
If a snippet/abstract is vague or incomplete, summarize only what is present.
Do not infer causes, conclusions, motivations, or definitions that aren’t explicitly stated.
	6.	Natural language paragraph.
The final answer must be a short, coherent paragraph—not a list and not a bullet-point summary.
	7.	Duplicate links.
Treat repeated URLs as separate results unless the snippet/abstract text is identical.
	8.	Unanswerable queries.
Only if title, snippet, AND abstract (when present) contain no relevant information, respond exactly:
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

        # If the model still refuses but we already fetched usable abstracts,
        # surface the best abstract instead of a false ABSENCE signal.
        refuse = "the search results do not provide information that answers this query."
        if isinstance(summary, str) and summary.strip().lower().startswith(refuse):
            usable_abstracts = []
            for idx, item in enumerate(page_list, start=1):
                abstract = str(item.get("abstract") or "").strip()
                if not abstract or abstract.startswith("[abstract_fetch_failed]"):
                    continue
                if abstract.lower().startswith(("error", "[fetch_failed]")):
                    continue
                link = item.get("link", "")
                usable_abstracts.append(f"[{idx}]({link}) {abstract}")
            if usable_abstracts:
                summary = " ".join(usable_abstracts[:2])

        # Step 5: 对 URL 做跳转清洗（与原版保持一致）
        try:
            summary = self.reformat_response(summary)
        except Exception:
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


    def _execute_gemini_search(self, query: str, add_citations_flag: bool = True):
        client = self._get_gemini_client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        response = None
        text = None

        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
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
            except Exception:
                pass

        try:
            text = self.reformat_response(text)
        except Exception:
            pass

        return text

    def _execute_search(self, query: str, add_citations_flag: bool = True):
        if USE_LOCAL:
            try:
                return self._execute_local_search(query, add_citations_flag)
            except Exception as e:
                print(f"Local Google search failed: {e}")
                if os.getenv("GOOGLE_API_KEY"):
                    print("Falling back to Gemini Google Search API...")
                    return self._execute_gemini_search(query, add_citations_flag)
                return f"Error: local Google search failed: {e}"

        return self._execute_gemini_search(query, add_citations_flag)


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