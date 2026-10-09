"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


import random
import re

_LAST_ARXIV_TIME = 0.0


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=60.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None:
                delay = min(float(e.retry_after), cap)
            else:
                backoff = base * (2 ** attempt)
                jitter = random.uniform(0, 0.5 * backoff)
                delay = min(backoff + jitter, cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_TIME
    try:
        terms = re.findall(r'[a-zA-Z0-9]+', query)
        if not terms:
            return "NO RESULTS"

        search_q = " AND ".join(f"all:{t}" for t in terms)
        max_res = min(max(1, max_results), 30)

        now = time.time()
        if now - _LAST_ARXIV_TIME < 3.0:
            time.sleep(3.0 - (now - _LAST_ARXIV_TIME))
        _LAST_ARXIV_TIME = time.time()

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(
                        ARXIV_URL,
                        params={
                            "search_query": search_q,
                            "sortBy": "submittedDate",
                            "sortOrder": "descending",
                            "max_results": max_res,
                            "start": 0,
                        },
                    )
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra = resp.headers.get("Retry-After")
                        retry_after = float(ra) if ra and ra.isdigit() else None
                        raise RetryableError(f"arXiv HTTP {resp.status_code}", retry_after=retry_after)
                    resp.raise_for_status()
                    return resp.text
            except httpx.TransportError as te:
                raise RetryableError(f"arXiv Transport Error: {te}")

        xml_text = with_retry(_fetch, attempts=6, base=2.0, cap=60.0)

        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        records = []
        for entry in entries:
            id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
            pub_elem = entry.find("{http://www.w3.org/2005/Atom}published")
            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")

            if id_elem is None or id_elem.text is None:
                continue

            raw_id = id_elem.text.strip()
            clean_id = raw_id.split("/abs/")[-1]
            clean_id = re.sub(r"v\d+$", "", clean_id)
            url = f"https://arxiv.org/abs/{clean_id}"

            published = (pub_elem.text.strip() if pub_elem is not None and pub_elem.text else "")[:10]
            title = re.sub(r"\s+", " ", title_elem.text or "").strip() if title_elem is not None else ""
            summary = re.sub(r"\s+", " ", summary_elem.text or "").strip()[:600] if summary_elem is not None else ""

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {str(exc)}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        lim = min(max(1, limit), 100)
        params = {"limit": lim}
        if date and date.strip():
            params["date"] = date.strip()

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_DAILY_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra = resp.headers.get("Retry-After")
                        retry_after = float(ra) if ra and ra.isdigit() else None
                        raise RetryableError(f"HF Daily HTTP {resp.status_code}", retry_after=retry_after)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError as te:
                raise RetryableError(f"HF Daily Transport Error: {te}")

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            p = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(p, dict):
                continue
            paper_id = p.get("id")
            if not paper_id:
                continue

            url = f"https://huggingface.co/papers/{paper_id}"
            published = (p.get("publishedAt") or "")[:10]
            title = re.sub(r"\s+", " ", p.get("title") or "").strip()
            summary = re.sub(r"\s+", " ", p.get("summary") or "").strip()[:600]
            upvotes = p.get("upvotes") or 0
            github = p.get("githubRepo") or ""
            stars = p.get("githubStars") or 0

            rec = {
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            }

            if keyword and keyword.strip():
                kw = keyword.strip().lower()
                if kw not in title.lower() and kw not in summary.lower():
                    continue

            records.append(rec)

        records.sort(key=lambda x: x.get("upvotes", 0), reverse=True)

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {str(exc)}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        lim = min(max(1, limit), 50)
        params = {"q": query, "limit": lim}

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_SEARCH_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra = resp.headers.get("Retry-After")
                        retry_after = float(ra) if ra and ra.isdigit() else None
                        raise RetryableError(f"HF Search HTTP {resp.status_code}", retry_after=retry_after)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError as te:
                raise RetryableError(f"HF Search Transport Error: {te}")

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            p = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(p, dict):
                continue
            paper_id = p.get("id")
            if not paper_id:
                continue

            url = f"https://huggingface.co/papers/{paper_id}"
            published = (p.get("publishedAt") or "")[:10]
            title = re.sub(r"\s+", " ", p.get("title") or "").strip()
            summary_text = p.get("ai_summary") or p.get("summary") or ""
            summary = re.sub(r"\s+", " ", summary_text).strip()[:600]
            upvotes = p.get("upvotes") or 0
            github = p.get("githubRepo") or ""
            stars = p.get("githubStars") or 0

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {str(exc)}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(method_name, arguments):
    exa_key = os.getenv("EXA_API_KEY", "").strip()
    url = EXA_URL
    if exa_key:
        url = f"{EXA_URL}?exaApiKey={exa_key}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": method_name,
            "arguments": arguments,
        },
    }

    def _fetch():
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, headers=headers, json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    ra = resp.headers.get("Retry-After")
                    retry_after = float(ra) if ra and ra.isdigit() else None
                    raise RetryableError(f"Exa HTTP {resp.status_code}", retry_after=retry_after)
                resp.raise_for_status()

                resp_text = resp.text
                json_data = None
                for line in resp_text.splitlines():
                    if line.startswith("data:"):
                        json_str = line[5:].strip()
                        if json_str:
                            try:
                                json_data = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                pass
                if json_data is None:
                    try:
                        json_data = resp.json()
                    except json.JSONDecodeError:
                        raise RetryableError("Invalid JSON from Exa MCP")

                if "error" in json_data:
                    err_msg = str(json_data["error"])
                    if "rate limit" in err_msg.lower() or "429" in err_msg:
                        raise RetryableError(f"Exa Rate Limit: {err_msg}", retry_after=15.0)
                    raise RuntimeError(f"Exa Error: {err_msg}")

                result = json_data.get("result", {})
                if isinstance(result, dict):
                    meta = result.get("_meta", {})
                    if meta.get("rate_limit") or "rate limit" in str(result).lower():
                        raise RetryableError("Exa Rate Limited in _meta", retry_after=20.0)

                return result
        except httpx.TransportError as te:
            raise RetryableError(f"Exa Transport Error: {te}")

    return with_retry(_fetch, attempts=6, base=2.0, cap=60.0)


def _redact_key(text: str) -> str:
    exa_key = os.getenv("EXA_API_KEY", "").strip()
    if exa_key:
        text = text.replace(exa_key, "[REDACTED]")
    return text


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective if objective and objective.strip() else f"find comprehensive survey and information about {query}"
        args = {
            "query": query,
            "objective": obj,
            "numResults": min(max(1, num_results), 10),
        }
        result = _call_exa_mcp("web_search_exa", args)
        content_list = result.get("content", []) if isinstance(result, dict) else []
        texts = []
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text", "").strip()
                if txt:
                    texts.append(txt)

        final_text = "\n\n".join(texts).strip()
        if not final_text:
            return "NO RESULTS"
        return final_text[:12000]

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        args = {"urls": [url]}
        result = _call_exa_mcp("web_fetch_exa", args)
        content_list = result.get("content", []) if isinstance(result, dict) else []
        texts = []
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text", "").strip()
                if txt:
                    texts.append(txt)

        final_text = "\n\n".join(texts).strip()
        if not final_text:
            return "NO RESULTS"
        return final_text[:12000]

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
