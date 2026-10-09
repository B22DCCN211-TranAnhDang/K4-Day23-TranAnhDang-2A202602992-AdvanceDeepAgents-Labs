"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name."""
    if not topic or not topic.strip():
        return "topic"
    s = topic.strip().lower()
    s = re.sub(r"[^\w]+", "-", s).strip("-")
    s = s[:60].rstrip("-")
    return s if s else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please perform a deep research report on the topic: '{topic}'.\n"
        f"CRITICAL REQUIREMENTS (RUBRIC 2.1 & 2.2):\n"
        f"1. You MUST delegate to the researcher subagent at least 3 times using the task tool:\n"
        f"   - Task 1 to researcher: Search arXiv using arxiv_search for seminal and recent papers (label source: 'arxiv').\n"
        f"   - Task 2 to researcher: Search Hugging Face using hf_search_papers and hf_daily_papers for trending papers (label source: 'hf-search' or 'hf-daily').\n"
        f"   - Task 3 to researcher: Search the Web using web_search for survey blogs, benchmarks, or technical overview sites (label source: 'web').\n"
        f"2. Your sources.json MUST contain at least 3 distinct source families: 'arxiv', ('hf-search' or 'hf-daily'), AND 'web'.\n"
        f"3. In your report body, you MUST cite at least one [n] from EACH of the 3 source families (cite arXiv, cite Hugging Face, and cite Web).\n"
        f"4. Run finalize_citations.py and check_citations.py in the sandbox until OK is printed."
    )


def summarize(messages, elapsed, model_name):
    """Return summary dictionary of execution metrics."""
    tool_calls_counter = Counter()
    subagent_calls = 0
    total_input = 0
    total_output = 0

    for msg in messages:
        t_calls = getattr(msg, "tool_calls", None) or []
        for call in t_calls:
            c_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if c_name:
                tool_calls_counter[c_name] += 1
                if c_name == "task":
                    subagent_calls += 1

        meta = getattr(msg, "usage_metadata", None)
        if isinstance(meta, dict):
            total_input += meta.get("input_tokens", 0)
            total_output += meta.get("output_tokens", 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls_counter),
        "tokens": {"input": total_input, "output": total_output},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.decode("utf-8", errors="replace").strip():
        raise RuntimeError(f"Missing or empty report at {REPORT_PATH}")

    if not sources_bytes:
        raise RuntimeError(f"Missing sources file at {SOURCES_PATH}")

    try:
        raw_text = sources_bytes.decode("utf-8", errors="replace").strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
        # Find JSON list boundary if extra commentary surrounds it
        m = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
        if m:
            sources = json.loads(m.group(0))
        else:
            sources = json.loads(raw_text)
        if not isinstance(sources, list):
            raise ValueError("sources.json is not a JSON list")
    except Exception as exc:
        raise RuntimeError(f"Invalid sources.json: {exc}")

    n_sources = len(sources)
    source_families = sorted(list({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")}))

    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    meta_data = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": n_sources,
        "source_families": source_families,
    }

    report_path.write_bytes(report_bytes)
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    if not topic or not topic.strip():
        sys.stderr.write("Usage: python research.py \"<topic>\"\n")
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", getattr(model, "model", str(model)))
    start = time.monotonic()

    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            elapsed = time.monotonic() - start
            messages = result.get("messages", [])
            out_path = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {out_path}")
            return 0
    except RuntimeError as exc:
        sys.stderr.write(f"FAILED: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
