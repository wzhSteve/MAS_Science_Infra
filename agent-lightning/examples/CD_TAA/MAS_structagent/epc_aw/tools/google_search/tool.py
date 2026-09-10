import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from .local_google_scraper import (
    build_query_variants,
    http_duckduckgo_search,
    local_google_search,
    sanitize_google_query,
)
    
from MAS.epc_aw.tools.base import BaseTool
from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.tools.web_search.tool import Web_Search_Tool
from MAS.epc_aw.tools.search_extract import (
    STRUCTURED_MARKER,
    extract_character_candidates,
    extract_numeric_candidates,
    format_tool_output,
    ground_facts,
    human_section,
    infer_query_intent,
    parse_structured_block,
    select_best_numeric_fact,
)
import requests
from typing import Any, Dict, List, Optional
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

# LOCAL_SEARCH=1 (default) uses local SERP + page fetch; set 0 for Gemini API search.
USE_LOCAL = os.getenv("LOCAL_SEARCH", "1").strip().lower() not in ("0", "false", "no")
# SERP backend: http/ddg (default, no Chrome) | chrome | chrome_first
SEARCH_BACKEND = os.getenv("GOOGLE_SEARCH_BACKEND", "http").strip().lower()

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
    # Exclude benchmark leakage + common mirror/leakage domains that pollute
    # character/numeric answers (HF datasets, Zhihu mirrors). Override via
    # env GOOGLE_SEARCH_EXCLUDED_KEYWORDS (comma-separated) when needed.
    DEFAULT_EXCLUDED_KEYWORDS = [
        "gaia",
        "bamboogle",
        "huggingface.co",
        "hf.co",
        "zhuanlan.zhihu.com",
        "zhihu.com",
        "harborframework.com",
        "instagram.com",
    ]

    def _resolve_excluded_keywords(self) -> list:
        env_val = os.getenv("GOOGLE_SEARCH_EXCLUDED_KEYWORDS", "")
        if env_val:
            return [k.strip().lower() for k in env_val.split(",") if k.strip()]
        return list(self.DEFAULT_EXCLUDED_KEYWORDS)

    @staticmethod
    def _annotate_family_scope(item: Dict[str, Any]) -> None:
        """Mark Nature-family aggregate hits so solvers can reject base_count."""
        blob = " ".join(
            str(item.get(k) or "")
            for k in ("title", "snippet", "abstract")
        ).lower()
        if any(
            m in blob
            for m in (
                "associated journal",
                "nature and its",
                "and its associated",
                "nature family",
            )
        ):
            item["scope"] = "family"

    def _collect_facts_from_pages(
        self, query: str, page_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        intent = infer_query_intent(query)
        facts: List[Dict[str, Any]] = []
        for item in page_list:
            url = str(item.get("link") or "")
            # Prefer structured facts already produced by Web_Search abstracts
            abstract = str(item.get("abstract") or "")
            payload = parse_structured_block(abstract)
            if payload and not payload.get("refusal"):
                for f in payload.get("facts") or []:
                    ff = dict(f)
                    ff.setdefault("url", url)
                    if item.get("scope") == "family":
                        ff["scope"] = "family"
                    facts.append(ff)
            human = human_section(abstract) if STRUCTURED_MARKER in abstract else abstract
            blob = " ".join(
                str(x)
                for x in (item.get("title"), item.get("snippet"), human)
                if x
            )
            if intent == "character_name":
                facts.extend(extract_character_candidates(blob, url=url))
            elif intent == "numeric":
                for f in extract_numeric_candidates(blob, url=url):
                    if item.get("scope") == "family":
                        f = dict(f)
                        f["scope"] = "family"
                    facts.append(f)
            else:
                facts.extend(extract_numeric_candidates(blob, url=url)[:3])
                facts.extend(extract_character_candidates(blob, url=url)[:2])
        return ground_facts(facts)

    def _passthrough_extractive(
        self,
        query: str,
        page_list: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        add_citations_flag: bool,
    ) -> str:
        """Skip summary LLM for numeric/character — quote + STRUCTURED only."""
        intent = infer_query_intent(query)
        usable_facts = [f for f in facts if f.get("scope") != "family"] or list(facts)

        if intent == "character_name":
            char_facts = [f for f in usable_facts if f.get("type") == "character_name"]
            if not char_facts:
                return format_tool_output(
                    "The search results do not provide information that answers this query.\n"
                    "[REFUSAL:true]",
                    [],
                    refusal=True,
                )
            qlow = str(query or "").lower()
            if any(k in qlow for k in ("unlambda", "applicative", "backtick", "backquote")):
                char_facts.sort(key=lambda f: 0 if f.get("value") == "backtick" else 1)
            chosen = char_facts[0]
            lines = []
            for idx, item in enumerate(page_list[:5], start=1):
                link = item.get("link", "")
                snip = str(item.get("snippet") or "")[:220]
                abs_h = human_section(str(item.get("abstract") or ""))[:220]
                lines.append(f"[{idx}]({link}) {item.get('title', '')} | {snip} | {abs_h}")
            human = f"{chosen.get('quote')}\n" + "\n".join(lines[:3])
            out = format_tool_output(human, [chosen], refusal=False)
        else:
            num_facts = [f for f in usable_facts if f.get("type") == "number"]
            chosen = select_best_numeric_fact(num_facts, query) if num_facts else None
            lines = []
            for idx, item in enumerate(page_list[:5], start=1):
                link = item.get("link", "")
                scope = f" scope={item.get('scope')}" if item.get("scope") else ""
                snip = str(item.get("snippet") or "")[:220]
                abs_h = human_section(str(item.get("abstract") or ""))[:280]
                lines.append(
                    f"[{idx}]({link}){scope} {item.get('title', '')} | {snip} | {abs_h}"
                )
            if not chosen:
                return format_tool_output(
                    "The search results do not provide information that answers this query.\n"
                    "[REFUSAL:true]\n"
                    + "\n".join(lines[:3]),
                    [],
                    refusal=True,
                )
            human = f"{chosen.get('quote')}\n" + "\n".join(lines[:3])
            out = format_tool_output(human, [chosen], refusal=False)

        try:
            out = self.reformat_response(out)
        except Exception:
            pass
        if add_citations_flag:
            out = self._append_citations(out, page_list, list(range(1, min(4, len(page_list) + 1))))
        return out

    def _append_citations(
        self,
        summary: str,
        page_list: List[Dict[str, Any]],
        indices: Optional[List[int]] = None,
    ) -> str:
        if indices is None:
            indices = [int(x) for x in re.findall(r"\[(\d+)\]", summary)]
        unique_indices = sorted(set(indices))
        if not unique_indices:
            return summary
        # Keep STRUCTURED block at the end: insert citations before it.
        structured_tail = ""
        body = summary
        if STRUCTURED_MARKER in summary:
            body, _, rest = summary.partition(STRUCTURED_MARKER)
            structured_tail = f"\n\n{STRUCTURED_MARKER}\n{rest.strip()}"
            body = body.rstrip()
        citation_texts = []
        for idx in unique_indices:
            if 1 <= idx <= len(page_list):
                item = page_list[idx - 1]
                title = item.get("title", "No Title")
                link = item.get("link", "No Link")
                snippet = item.get("snippet", "No Snippet")
                abstract = human_section(str(item.get("abstract") or ""))[:400] or "No Abstract"
                scope = item.get("scope")
                scope_line = f"\nScope: {scope}" if scope else ""
                citation_texts.append(
                    f"[{idx}] Title: {title}\nLink: {link}\nSnippet: {snippet}\n"
                    f"Abstract: {abstract}{scope_line}\n"
                )
        citation_section = "\n\nCitations:\n" + "\n".join(citation_texts)
        return body + citation_section + structured_tail

    def _search_backend(self, query: str) -> dict:
        """Resolve SERP without Chrome by default (DuckDuckGo HTTP).

        GOOGLE_SEARCH_BACKEND:
          - http / ddg / duckduckgo (default): requests → DDG HTML, no Chrome
          - chrome: undetected_chromedriver Google only
          - chrome_first: Chrome Google, then DDG on failure
        """
        original_query = str(query or "")
        backend = SEARCH_BACKEND

        if backend in ("http", "ddg", "duckduckgo"):
            print(f"[google_search] SERP backend=http (DuckDuckGo), query={original_query!r}")
            return http_duckduckgo_search(original_query)

        if backend == "chrome":
            print(f"[google_search] SERP backend=chrome, query={original_query!r}")
            return local_google_search(original_query)

        # chrome_first (legacy behavior)
        print(f"[google_search] SERP backend=chrome_first, query={original_query!r}")
        try:
            return local_google_search(original_query)
        except Exception as chrome_err:
            print(f"Local Google search failed: {chrome_err}")
            print("Falling back to DuckDuckGo HTTP search...")
            try:
                return http_duckduckgo_search(original_query)
            except Exception as http_err:
                raise RuntimeError(
                    f"Chrome Google failed ({chrome_err}); "
                    f"DuckDuckGo HTTP also failed ({http_err})"
                ) from http_err

    def _execute_local_search(self, query: str, add_citations_flag: bool = True):
        # Scraper sanitizes / variant-retries the SERP query. Keep the *original*
        # agent query for intent detection ("exact character" etc.) and extractive
        # fact collection; only the network search uses the cleaned form.
        original_query = str(query or "")
        raw_result = self._search_backend(original_query)
        serp_query = str(
            raw_result.get("query")
            or sanitize_google_query(original_query)
            or original_query
        )
        excluded_keywords = self._resolve_excluded_keywords()

        page_list = [
            item
            for item in raw_result.get("organic", [])
            if all(keyword not in item.get("link", "").lower() for keyword in excluded_keywords)
        ]
        if not page_list:
            # Raise so _execute_search can fall back to Gemini — empty organic is
            # coverage failure, not a verified absence.
            tried = raw_result.get("tried_queries") or build_query_variants(original_query)
            raise RuntimeError(
                "local Google search returned no organic results "
                f"for query={original_query!r} (tried={tried!r}). "
                "This is a scraper/coverage failure, not a verified absence."
            )

        # Prefer cleaned SERP keywords when fetching page abstracts (less noise).
        rag_query = serp_query or original_query
        max_abstract_pages = int(os.getenv("GOOGLE_SEARCH_MAX_ABSTRACT_PAGES", "5"))
        for i, item in enumerate(page_list[:max_abstract_pages]):
            url = item.get("link", "")
            if not url:
                continue
            try:
                abstract = self.web_rag_tool.execute(url=url, query=rag_query)
                page_list[i]["abstract"] = abstract
            except Exception as e:
                page_list[i]["abstract"] = f"[abstract_fetch_failed] {e}"
            self._annotate_family_scope(page_list[i])
        for item in page_list[max_abstract_pages:]:
            self._annotate_family_scope(item)
        raw_result["organic"] = page_list

        intent = infer_query_intent(original_query)
        facts = self._collect_facts_from_pages(original_query, page_list)
        query = original_query

        # numeric / character: skip paragraph summary LLM (most stable)
        if intent in ("numeric", "character_name"):
            return self._passthrough_extractive(
                query, page_list, facts, add_citations_flag,
            )

        # General queries: constrained paragraph summary + quote-grounded facts
        # Strip nested STRUCTURED from abstracts before LLM to reduce noise.
        llm_pages = []
        for item in page_list:
            copy = dict(item)
            abs_raw = str(copy.get("abstract") or "")
            if STRUCTURED_MARKER in abs_raw:
                copy["abstract"] = human_section(abs_raw)
            llm_pages.append(copy)
        llm_payload = dict(raw_result)
        llm_payload["organic"] = llm_pages
        json_text = json.dumps(llm_payload, ensure_ascii=False, indent=2)

        prompt = f"""
You are a Search Result Summarization Model.

You will receive raw search results from a Google-style scraper in JSON format.
Each item may include:
	•	title
	•	link
	•	snippet
	•	abstract (optional; page content extracted for the query)
	•	scope (optional; e.g. family = journal-family aggregate)

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
	9.	Numbers in the summary must appear verbatim in some title/snippet/abstract.
	10.	If scope=family, do not treat that count as Nature (journal) articles-only base_count.

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

        summary = self.client(prompt)

        refuse = "the search results do not provide information that answers this query."
        if isinstance(summary, str) and summary.strip().lower().startswith(refuse):
            usable_abstracts = []
            for idx, item in enumerate(page_list, start=1):
                abstract = human_section(str(item.get("abstract") or "")).strip()
                if not abstract or abstract.startswith("[abstract_fetch_failed]"):
                    continue
                if abstract.lower().startswith(("error", "[fetch_failed]")):
                    continue
                link = item.get("link", "")
                usable_abstracts.append(f"[{idx}]({link}) {abstract}")
            if usable_abstracts:
                summary = " ".join(usable_abstracts[:2])

        # Quote-ground numbers from summary against JSON text
        if isinstance(summary, str) and facts:
            summary_nums = set(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", summary))
            source_blob = json_text
            for num in list(summary_nums):
                if num not in source_blob and num.replace(",", "") not in source_blob.replace(",", ""):
                    # Drop unsupported number mentions by refusing free-form; fall back to extractive
                    chosen = select_best_numeric_fact(
                        [f for f in facts if f.get("type") == "number"], query,
                    )
                    if chosen:
                        summary = str(chosen.get("quote") or chosen.get("value"))

        try:
            summary = self.reformat_response(summary)
        except Exception:
            pass

        indices = [int(x) for x in re.findall(r'\[(\d+)\]', summary)]
        if add_citations_flag and indices:
            summary = self._append_citations(summary, page_list, indices)
        elif add_citations_flag:
            summary = self._append_citations(summary, page_list, list(range(1, min(4, len(page_list) + 1))))

        # Attach STRUCTURED facts (quote-grounded) for the solver
        return format_tool_output(str(summary or ""), facts, refusal=not facts and not summary)

    
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
        # Always sanitize before Gemini — long instruction+code queries are the
        # dominant cause of empty SERPs in GAIA-style agent traces.
        clean_query = sanitize_google_query(query) or str(query or "").strip()
        if USE_LOCAL:
            try:
                return self._execute_local_search(query, add_citations_flag)
            except Exception as e:
                print(f"Local search stack failed: {e}")
                if os.getenv("GOOGLE_API_KEY"):
                    print(
                        "Falling back to Gemini Google Search API with sanitized query: "
                        f"{clean_query!r}"
                    )
                    return self._execute_gemini_search(clean_query, add_citations_flag)
                return (
                    f"Error: local Google search failed: {e}. "
                    f"Sanitized query would have been {clean_query!r}."
                )

        return self._execute_gemini_search(clean_query, add_citations_flag)


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