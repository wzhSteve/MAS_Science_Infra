import os
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from MAS.epc_aw.tools.base import BaseTool
from MAS.epc_aw.engine.factory import create_llm_engine
from playwright.sync_api import sync_playwright
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import io
from collections import Counter
from typing import List
from PyPDF2 import PdfReader



def fetch_sciencedirect(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, timeout=20000)
        content = page.content()
        browser.close()
        return content
    
from bs4 import BeautifulSoup
# 如果要避免每次 import playwright 出现错误，这样写更稳健
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


load_dotenv()

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Web_Search_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations: 
1) To retrieve only the given URL, you should first search with Google or Wikipedia, and then use this web tool
2) Requires valid URLs that are accessible and contain text content. 
3) May not work with JavaScript-heavy websites or those requiring authentication. 
4) Performance depends on the quality and relevance of the website content. 
5) May return incomplete or inaccurate information if the website content is not comprehensive. 
6) Limited by the chunking and embedding process which may miss context. 
7) Requires OpenAI API access for embeddings and LLM generation.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1) Use specific, targeted queries rather than broad questions.
2) Ensure the URL is accessible and contains relevant information.
3) Prefer websites with well-structured, text-rich content.
4) For complex queries, break them down into smaller, specific questions.
5) Verify important information from multiple sources when possible.
6) Use it as part of a multi-step research process rather than a single source of truth.
7) It is highly recommended to use this tool after calling other web-based tools (e.g., Google_Search_Tool, Wiki_Search_Tool, etc.) to get the real, accessible URLs.
"""


SUMMARIZE_PROMPT_TEMPLATE = """
You are an expert AI assistant. Your task is to provide a clear, concise, and accurate answer to the user's query based **exclusively** on the provided reference information.

## Step-by-Step Instructions
1.  **Analyze the Query:** First, fully understand the user's query and identify the specific information being asked for.
2.  **Scan for Relevance:** Read through each numbered chunk in the reference information. Identify all chunks that contain information directly relevant to answering the query. A simple keyword match is not sufficient; the chunk must contain a substantive fact that helps answer the question.
3.  **Extract Key Facts & Synthesize:** From the relevant chunks, extract only the key facts and figures needed. Synthesize these extracted facts into a comprehensive, single-paragraph answer. Write the answer in your own words. **Do not** copy entire chunks.

## Output Format and Example

**IMPORTANT:** You must follow this format exactly.

### Example Input
- **User Query:** What were the key financial results for Q4 2023?
- **Reference Information:**
[1] The company's new "Project Starlight" initiative launched in January 2024.
[2] In Q4 2023, the company reported a total revenue of $5.2 million and a net profit of $800,000. This was a 15% increase in revenue compared to Q3 2023.
[3] Marketing spend in Q4 2023 was focused on digital channels, totaling $450,000.
[4] The CEO stated that the strong Q4 performance was driven by robust sales in the North American market.

### Example Output
Answer:
In the fourth quarter of 2023, the company achieved a total revenue of $5.2 million, which represented a 15% increase from the previous quarter, and a net profit of $800,000. The strong performance was attributed to robust sales in the North American market. The marketing expenditure for this period was $450,000.

---
## Your Turn

### User Query
{query}

### Reference Information
{reference_information}

### Output
"""

class Web_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string=os.getenv("MODEL_Name")):
        super().__init__(
            tool_name=TOOL_NAME,
            # tool_description="A specialized tool for answering questions by retrieving relevant information from a given website using RAG (Retrieval-Augmented Generation).",
            tool_description="A targeted web-retrieval tool for extracting and summarizing content strictly from a user-provided URL using Retrieval-Augmented Generation (RAG). The tool is activated only when a valid URL appears directly in the query. If the query does not contain a URL, this tool must never be called—no link inference, no guessing, no open-ended search.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for the website.",
                "url": "str - The URL of the website to retrieve information from.",
            },
            output_type="str - The answer to the user's query based on the information gathered from the website.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the exact mass in kg of the moon?", url="https://en.wikipedia.org/wiki/Moon")',
                    "description": "Retrieve information about the moon's mass from Wikipedia."
                },
                {
                    "command": 'execution = tool.execute(query="What are the main features of Python programming language?", url="https://www.python.org/about/apps/")',
                    "description": "Get information about Python features from the official website."
                }
            ],
            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )

        # self.model_string = "gpt-4o-mini" # NOTE: strong LLM for tool
        # self.model_string = "gemini-1.5-flash" # NOTE: weak 8B model for tool
        # self.model_string = "dashscope" # NOTE: weak Qwen2.5-7B model for tool

        self.model_string = model_string
        # print(f"Initializing Website RAG Tool with model: {self.model_string}")
        self.chunk_size = 200
        self.chunk_overlap = 20
        self.top_k = 10
        # self.embeddings_model = "text-embedding-3-large" # or "text-embedding-3-small" for efficiency
        self.embeddings_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        self.embeddings_model_api_key = os.getenv("EMBEDDING_MODEL_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.embeddings_model_url = os.getenv("EMBEDDING_MODEL_URL", None)
        self.max_window_size = 1000000
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
        # a small stopword set for token filtering
        self.stopwords = {
            "the", "a", "an", "and", "or", "of", "in", "on", "for", "with", "to",
            "is", "are", "was", "were", "that", "this", "it", "as", "by", "from",
            "be", "has", "have", "at", "which", "we", "their", "our", "can"
        }
        # NOTE: deterministic mode
        self.llm_engine = create_llm_engine(
            model_string=self.model_string, 
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )

    def _get_website_content(self, url: str, query: str, max_len: int = 1000) -> str:
        """
        Main entry:
          - url: web page or PDF link
          - query: user's query (used to find relevant content)
          - max_len: hard cap on returned characters (default 1000)
        Returns a short summary string (<= max_len chars).
        """
        max_len = int(max_len)
        max_len = min(max_len, 1000)  # hard cap as requested

        # Normalize arXiv pdf -> abs
        try:
            url = url.replace("arxiv.org/pdf", "arxiv.org/abs")
        except Exception:
            pass

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        # 1) HEAD to detect content-type if possible
        content_type = ""
        try:
            head = requests.head(url, headers=headers, timeout=8, allow_redirects=True)
            content_type = head.headers.get("Content-Type", "").lower()
        except Exception:
            content_type = ""

        # If PDF type or url endswith .pdf -> PDF flow
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                return self._pdf_flow(url, headers, query, max_len)
            except Exception:
                # fallback to HTML flow if PDF flow fails
                pass

        # 2) HTML flow via requests
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            html = resp.text or resp.content.decode("utf-8", errors="ignore")
        except Exception as e:
            # If requests failed in a way that indicates JS/captcha, try Playwright if available
            err = str(e).lower()
            need_browser = any(k in err for k in ("403", "captcha", "blocked", "cloudflare", "forbidden", "unsupported_browser"))
            if PLAYWRIGHT_AVAILABLE and need_browser:
                try:
                    return self._browser_flow(url, query, max_len)
                except Exception:
                    return f"[fetch_failed] {str(e)}"[:max_len]
            return f"[fetch_failed] {str(e)}"[:max_len]

        # 3) Try to extract summary/abstract from HTML
        try:
            # prioritize: meta description / open graph / twitter, then embedded JSON abstracts,
            # then site-specific selectors (ScienceDirect etc.), then query-driven extract & summarize.
            # 3a) meta descriptions
            meta_summary = self._extract_meta_description(html)
            if meta_summary:
                return self._safe_truncate(meta_summary, max_len)

            # 3b) embedded JSON abstracts (Next.js / PRELOADED_STATE)
            embedded = self._extract_embedded_abstract(html)
            if embedded:
                return self._safe_truncate(embedded, max_len)

            # 3c) site-specific quick selectors
            soup = BeautifulSoup(html, "html.parser")
            site_specific = self._extract_site_specific(soup, url)
            if site_specific:
                return self._safe_truncate(site_specific, max_len)

            # 3d) full-text paragraphs -> relevance selection -> extractive summary
            full_text = soup.get_text(separator="\n", strip=True)
            paragraphs = self._split_paragraphs(full_text)
            if not paragraphs:
                return self._safe_truncate(full_text, max_len)

            top_pars = self._select_relevant_paragraphs(paragraphs, query, top_k=6)
            if not top_pars:
                # fallback to the first N chars of the page
                return self._safe_truncate(full_text, max_len)

            summary = self._extractive_summary_from_paragraphs(top_pars, query, max_len)
            return self._safe_truncate(summary, max_len)
        except Exception as e:
            # Last-resort fallback: return beginning of page
            try:
                return self._safe_truncate(full_text, max_len)
            except Exception:
                return f"[processing_failed] {str(e)}"[:max_len]

    # ----------------- Helpers -----------------
    def _safe_truncate(self, s: str, n: int) -> str:
        if s is None:
            return ""
        s = s.strip()
        if len(s) <= n:
            return s
        # try to end at sentence boundary
        m = re.search(r"(.{0,%d}[\.。！？\?\n])" % (n-1), s)
        if m:
            return m.group(1).strip()[:n]
        return s[:n].rstrip()

    def _extract_meta_description(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        # common meta names
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return og["content"].strip()
        tw = soup.find("meta", property="twitter:description")
        if tw and tw.get("content"):
            return tw["content"].strip()
        # Dublin Core
        dc = soup.find("meta", attrs={"name": "dc.description"})
        if dc and dc.get("content"):
            return dc["content"].strip()
        return None

    def _extract_embedded_abstract(self, html: str):
        # try window.__PRELOADED_STATE__ first (ScienceDirect style)
        m = re.search(r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});', html, re.S)
        if m:
            js = m.group(1).strip()
            try:
                data = json.loads(js)
                # try a few plausible paths
                try:
                    return data["abstracts"]["content"][0]["$$"][0]["$$"][0]["_"]
                except Exception:
                    pass
                try:
                    return data["abstracts"]["content"][0]["$$"][0]["_"]
                except Exception:
                    pass
            except Exception:
                pass

        # try Next.js __NEXT_DATA__ common pattern
        m2 = re.search(r'__NEXT_DATA__"\s*type="application/json">\s*(\{.*?\})\s*</script>', html, re.S)
        if m2:
            js2 = m2.group(1).strip()
            try:
                data2 = json.loads(js2)
                try:
                    return data2["props"]["pageProps"]["article"]["abstracts"][0]["para"]
                except Exception:
                    pass
                try:
                    return data2["props"]["pageProps"]["coredata"]["dc:description"]
                except Exception:
                    pass
            except Exception:
                pass

        # generic JSON-LD
        for tag in BeautifulSoup(html, "html.parser").find_all("script", type="application/ld+json"):
            try:
                jd = json.loads(tag.string or "{}")
                if isinstance(jd, dict) and "description" in jd and jd["description"]:
                    return jd["description"].strip()
            except Exception:
                continue

        return None

    def _extract_site_specific(self, soup: BeautifulSoup, url: str):
        domain = urlparse(url).netloc.lower()
        # sciencedirect
        if "sciencedirect.com" in domain:
            block = soup.select_one("div.Abstracts p, div.Abstracts u-expandedtext p, div.abstracts p")
            if block:
                return block.get_text(" ", strip=True)
        # dl.acm.org
        if "dl.acm.org" in domain:
            block = soup.select_one("div.abstractSection, div.article__abstract p, div.citation__abstract")
            if block:
                return block.get_text(" ", strip=True)
        # springer
        if "springer" in domain:
            block = soup.select_one("section.Abstract p, div.Article__Abstract p")
            if block:
                return block.get_text(" ", strip=True)
        # ieee
        if "ieeexplore.ieee.org" in domain:
            meta = soup.find("meta", {"name": "description"})
            if meta and meta.get("content"):
                return meta["content"].strip()
        # arxiv
        if "arxiv.org" in domain:
            abs_block = soup.find("blockquote", class_="abstract")
            if abs_block:
                return abs_block.get_text(" ", strip=True).replace("Abstract:", "").strip()
        return None

    def _split_paragraphs(self, text: str) -> List[str]:
        # split on two or more newlines or long line breaks, then filter short lines
        parts = re.split(r'\n{2,}|\r\n{2,}', text)
        pars = []
        for p in parts:
            p = p.strip()
            if len(p) < 30:
                # but keep slightly longer single-line paragraphs
                continue
            pars.append(" ".join(p.split()))
        # fallback: further split if no paragraphs
        if not pars:
            # split sentences roughly
            sents = re.split(r'(?<=[\.。！？\?\!])\s+', text)
            pars = [s.strip() for s in sents if len(s.strip()) > 30]
        return pars

    def _normalize_tokens(self, s: str) -> List[str]:
        s = s.lower()
        # remove punctuation except intra-word hyphens
        s = re.sub(r'[^\w\s\-]', ' ', s)
        toks = [t for t in s.split() if t and t not in self.stopwords]
        return toks

    def _counter_from_text(self, s: str) -> Counter:
        toks = self._normalize_tokens(s)
        return Counter(toks)

    def _cosine_similarity_counters(self, a: Counter, b: Counter) -> float:
        # cosine similarity between two Counters (term freq vectors)
        if not a or not b:
            return 0.0
        # dot product
        dot = 0.0
        for k, v in a.items():
            dot += v * b.get(k, 0)
        norm_a = sum(v*v for v in a.values()) ** 0.5
        norm_b = sum(v*v for v in b.values()) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _select_relevant_paragraphs(self, paragraphs: List[str], query: str, top_k: int = 5) -> List[str]:
        qcnt = self._counter_from_text(query)
        scores = []
        for p in paragraphs:
            pcnt = self._counter_from_text(p)
            score = self._cosine_similarity_counters(qcnt, pcnt)
            scores.append((score, p))
        scores.sort(reverse=True, key=lambda x: x[0])
        top = [p for s, p in scores[:top_k] if s > 0]
        # if nothing scored >0, fallback to top-k by length
        if not top:
            top = [p for _, p in sorted(((len(p), p) for p in paragraphs), reverse=True)[:top_k]]
        return top

    def _extractive_summary_from_paragraphs(self, paragraphs: List[str], query: str, max_len: int) -> str:
        """
        Simple extractive summarizer:
          - split paragraphs into sentences
          - score sentences by similarity to query (same TF-cosine)
          - pick top sentences until length limit
        """
        qcnt = self._counter_from_text(query)
        sentences = []
        for p in paragraphs:
            sents = re.split(r'(?<=[\.。！？\?\!])\s+', p)
            for s in sents:
                s = s.strip()
                if len(s) < 20:
                    continue
                sentences.append(s)

        # score each sentence
        scored = []
        for s in sentences:
            sc = self._cosine_similarity_counters(qcnt, self._counter_from_text(s))
            # boost sentences appearing earlier in paragraphs slightly
            scored.append((sc, s))

        scored.sort(reverse=True, key=lambda x: x[0])

        selected = []
        cur_len = 0
        for sc, s in scored:
            if cur_len + len(s) + 1 > max_len:
                continue
            # avoid near-duplicates
            if any(self._jaccard_overlap(s, ex) > 0.6 for ex in selected):
                continue
            selected.append(s)
            cur_len += len(s) + 1
            if cur_len >= max_len:
                break

        if not selected:
            # fallback: join the top paragraphs truncated
            joinp = " ".join(paragraphs)
            return joinp[:max_len]
        return " ".join(selected)[:max_len]

    def _jaccard_overlap(self, a: str, b: str) -> float:
        ta = set(self._normalize_tokens(a))
        tb = set(self._normalize_tokens(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    # ----------------- PDF and Browser flows -----------------
    def _pdf_flow(self, url: str, headers: dict, query: str, max_len: int) -> str:
        """
        Download PDF and attempt to extract abstract from first pages.
        If no explicit abstract is found, behave like HTML flow: split into paragraphs,
        score by relevance to query, and extract summary.
        """
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            pdf_bytes = resp.content
            reader = PdfReader(io.BytesIO(pdf_bytes))
            full_text = ""
            # extract text from first 3 pages (abstract usually early)
            num_pages = min(len(reader.pages), 6)
            for i in range(num_pages):
                try:
                    page = reader.pages[i]
                    pt = page.extract_text() or ""
                    full_text += pt + "\n"
                except Exception:
                    continue

            # look for "Abstract" marker
            txt_lower = full_text.lower()
            idx = txt_lower.find("abstract")
            if idx != -1:
                # take a chunk after the marker
                snippet = full_text[idx: idx + 2000]
                # clean and return
                return self._safe_truncate(snippet.replace("\n", " "), max_len)

            # fallback: split into paragraphs & select relevant
            paragraphs = self._split_paragraphs(full_text)
            if paragraphs:
                top = self._select_relevant_paragraphs(paragraphs, query, top_k=6)
                summary = self._extractive_summary_from_paragraphs(top, query, max_len)
                return self._safe_truncate(summary, max_len)

            return "[PDF read but no abstract found]"[:max_len]
        except Exception as e:
            return f"[pdf_failed] {str(e)}"[:max_len]

    def _browser_flow(self, url: str, query: str, max_len: int) -> str:
        """
        Use Playwright to render JS-heavy pages, then reuse HTML flow logic.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                page.set_default_navigation_timeout(30000)
                page.goto(url, timeout=30000)
                html = page.content()
                try:
                    context.close()
                except Exception:
                    pass
                try:
                    browser.close()
                except Exception:
                    pass

            # reuse HTML processing
            meta_summary = self._extract_meta_description(html)
            if meta_summary:
                return self._safe_truncate(meta_summary, max_len)
            embedded = self._extract_embedded_abstract(html)
            if embedded:
                return self._safe_truncate(embedded, max_len)
            soup = BeautifulSoup(html, "html.parser")
            site_specific = self._extract_site_specific(soup, url)
            if site_specific:
                return self._safe_truncate(site_specific, max_len)
            full_text = soup.get_text(separator="\n", strip=True)
            paragraphs = self._split_paragraphs(full_text)
            top_pars = self._select_relevant_paragraphs(paragraphs, query, top_k=6)
            summary = self._extractive_summary_from_paragraphs(top_pars, query, max_len)
            return self._safe_truncate(summary, max_len)
        except Exception as e:
            return f"[browser_failed] {str(e)}"[:max_len]


    def _chunk_website_content(self, content):
        """
        Chunks the website content into smaller chunks based on the chunk size and overlap.
        Parameters:
            content (str): The website content to chunk.
        Returns:
            list: A list of chunks.
        """
        # Split the content string by whitespace characters
        words = content.split()
        ptr = 0
        chunks = []
        while True:
            start, end = ptr, min(ptr + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            ptr = end - self.chunk_overlap
        return chunks

    def _embed_strings(self, strings):
        """
        Embed the strings using OpenAI's embedding model.
        Parameters:
            strings (list): A list of strings to embed.
        Returns:
            list: A list of embeddings.
        """
        try:
            client = openai.OpenAI(api_key=self.embeddings_model_api_key, base_url=self.embeddings_model_url)
            embeddings = client.embeddings.create(
                input=strings,
                model=self.embeddings_model
            )
            res = [embedding.embedding for embedding in embeddings.data]
            return res
        except Exception as e:
            raise Exception(f"Error embedding strings: {str(e)}")

    def _cosine_similarity(self, a, b):
        """
        Calculate the cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _rank_chunks(self, query_embedding, chunk_embeddings):
        """
        Rank the chunks based on the query embedding.
        Parameters:
            query_embedding (list): The embedding of the query.
            chunk_embeddings (list): The embeddings of the chunks.
        Returns:
            list: The indices of the ranked chunks in descending order of similarity.
        """
        similarities = [self._cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
        return list(np.argsort(similarities)[::-1])

    def _concatenate_chunks(self, chunks):
        """
        Concatenate the chunks into a single string.
        """
        for i, chunk in enumerate(chunks):
            chunks[i] = f"Chunk [{i+1}]\n{chunk}"
        return "\n".join(chunks)

    def _construct_final_output(self, query, reference_information):
        """
        Construct the final output from the top chunks.
        """
        summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            query=query,
            reference_information=reference_information
        )
        
        summary = self.llm_engine(summary_prompt)
        return summary

    def execute(self, query, url):
        try:
            # step 1: get content from the website
            website_content = self._get_website_content(url=url, query=query)
            
            if website_content.startswith("Error"):
                return website_content

            # step 2: chunk the content
            chunks = self._chunk_website_content(website_content)
            
            if not chunks:
                return "Error: No content could be extracted from the website."

            # step 3: embed the chunks
            embeddings = self._embed_strings([query] + chunks)
            query_embedding = embeddings[0]
            chunk_embeddings = embeddings[1:]
            
            # step 4: rank the chunks
            ranked_chunks = self._rank_chunks(query_embedding, chunk_embeddings)
            top_chunks = [chunks[i] for i in ranked_chunks[:self.top_k]]

            # step 5: summarize the top chunks
            reference_string = self._concatenate_chunks(top_chunks)
            summary = self._construct_final_output(query, reference_string)

            return summary
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        # metadata['require_llm_engine'] = self.require_llm_engine
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/web_search
    python tool.py
    """

    import json

    # Example usage of the Web_Search_Tool
    tool = Web_Search_Tool(model_string=os.getenv("MODEL_Name")) # NOTE: strong LLM for tool
    # tool = Web_Search_Tool(model_string="gemini-1.5-flash") # NOTE: weak 8B model for tool
    # tool = Web_Search_Tool(model_string="dashscope") # NOTE: weak Qwen2.5-7B model for tool

    # Get tool metadata
    metadata = tool.get_metadata()
    # print("Tool Metadata:")
    # print(json.dumps(metadata, indent=4))

    examples = [
        {
            "query": "What is the exact mass in kg of the moon?", 
            "url": "https://en.wikipedia.org/wiki/Moon"
        },
        {
            "query": "What is the capital of France?", 
            "url": "https://en.wikipedia.org/wiki/France"
        },
        {
            "query": "What are the main features of Python programming language?", 
            "url": "https://www.python.org/about/apps/"
        }
    ]

    for example in examples:
        try:
            # Execute the tool with example query
            execution = tool.execute(**example)
            print("\nGenerated Response:")
            print(execution)
            print("\n")
        except Exception as e:
            print(f"Execution failed: {e}")


    print("\nDone!")




