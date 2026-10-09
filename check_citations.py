"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    import re
    problems = []

    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    # 1. Validate sources.json
    seen_urls = set()
    sources_dict = {}

    for idx, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source item at index {idx} is not a dict")
            continue

        n = s.get("n")
        url = s.get("url")

        if not isinstance(n, int):
            problems.append(f"source item at index {idx} has non-integer 'n': {n!r}")
        else:
            if n in sources_dict:
                problems.append(f"duplicate source number [{n}] in sources.json")
            sources_dict[n] = url

        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid URL: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            seen_urls.add(url)

    families = {s.get("source") for s in sources if isinstance(s, dict) and s.get("source")}
    if len({"arxiv", "hf-daily", "hf-search", "web"} & families) < 3:
        problems.append(f"only {len({'arxiv', 'hf-daily', 'hf-search', 'web'} & families)} source families in sources.json: need at least 3 of ['arxiv', 'hf-daily', 'hf-search', 'web']")

    # 2. Split report at "## References"
    if "## References" not in report_text:
        problems.append("missing '## References' section in report")
        body = report_text
        ref_section = ""
    else:
        parts = report_text.split("## References", 1)
        body = parts[0]
        ref_section = parts[1]

    # 3. Extract citations [n] from body
    # Remove code blocks first
    body_clean = re.sub(r'```.*?```', '', body, flags=re.DOTALL)
    body_clean = re.sub(r'`[^`\n]+`', '', body_clean)
    # Remove standard markdown links like [text](url)
    body_clean = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', '', body_clean)

    # Match bracketed citation patterns like [1], [1, 2], [1-3], [1, 3-5]
    raw_citations = re.findall(r'\[([\d\s,\u2013\u2014\-]+)\]', body_clean)
    cited_numbers = set()

    for item in raw_citations:
        for part in item.split(','):
            part = part.strip()
            if not part:
                continue
            m_range = re.match(r'^(\d+)[\u2013\u2014\-]+(\d+)$', part)
            if m_range:
                start, end = int(m_range.group(1)), int(m_range.group(2))
                if start <= end:
                    for num in range(start, end + 1):
                        cited_numbers.add(num)
                else:
                    problems.append(f"invalid citation range [{part}] in body")
            elif part.isdigit():
                cited_numbers.add(int(part))

    # Check cited vs sources
    for n in sorted(cited_numbers):
        if n not in sources_dict:
            problems.append(f"[{n}] cited but missing from sources.json")

    for n in sorted(sources_dict.keys()):
        if n not in cited_numbers:
            problems.append(f"source [{n}] never cited")

    # 4. Check ## References section
    if ref_section:
        ref_lines = ref_section.strip().splitlines()
        ref_numbers_seen = []

        for line in ref_lines:
            line_str = line.strip()
            if not line_str:
                continue
            match = re.match(r'^\[(\d+)\]\s*(.*)', line_str)
            if not match:
                continue

            ref_n = int(match.group(1))
            ref_content = match.group(2)
            ref_numbers_seen.append(ref_n)

            if ref_n not in sources_dict:
                problems.append(f"reference line [{ref_n}] is not in sources.json")

            urls_in_line = re.findall(r'https?://[^\s>\]\)]+', ref_content)
            if not urls_in_line:
                problems.append(f"reference line [{ref_n}] has no valid URL")
            elif len(urls_in_line) > 1:
                problems.append(f"reference line [{ref_n}] bundles multiple URLs")
            else:
                line_url = urls_in_line[0].rstrip(".,;)")
                expected_url = sources_dict.get(ref_n)
                if expected_url and line_url != expected_url:
                    problems.append(f"reference line [{ref_n}] URL ({line_url}) does not match sources.json ({expected_url})")

        ref_counts = {}
        for num in ref_numbers_seen:
            ref_counts[num] = ref_counts.get(num, 0) + 1

        for num, count in ref_counts.items():
            if count > 1:
                problems.append(f"duplicate reference line for source [{num}]")

        for n in sorted(sources_dict.keys()):
            if n not in ref_counts:
                problems.append(f"missing reference line for source [{n}]")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
