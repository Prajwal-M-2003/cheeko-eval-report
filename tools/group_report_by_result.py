import argparse
import os
import re
from pathlib import Path


SECTION_STYLE = """
.result-sections { display:grid; gap:20px; }
.result-section {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: var(--r);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.result-section-head {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  padding:16px 20px;
  border-bottom:1px solid var(--border);
}
.result-section-head.fail {
  background: linear-gradient(135deg, #fff1f2, #ffe4e6);
}
.result-section-head.pass {
  background: linear-gradient(135deg, #ecfdf5, #d1fae5);
}
.result-section-title {
  font-size: 1rem;
  font-weight: 800;
  color: var(--navy);
}
.result-section-count {
  font-size: .78rem;
  font-weight: 700;
  color: var(--text2);
}
.result-jump {
  display:flex;
  justify-content:center;
  gap:12px;
  margin: 18px 0 0;
  flex-wrap: wrap;
}
.result-jump a {
  text-decoration:none;
  border:1px solid #cbd5e1;
  background:#fff;
  color: var(--navy);
  font-size:.78rem;
  font-weight:700;
  padding:8px 14px;
  border-radius:999px;
}
.result-jump a:hover {
  background:#eef2ff;
  border-color:#a5b4fc;
}
"""


def _replace_first(text: str, pattern: str, repl: str) -> str:
    return re.sub(pattern, repl, text, count=1, flags=re.DOTALL)


def _inject_section_style(html: str) -> str:
    if ".result-section {" in html:
        return html
    return html.replace("</style>", SECTION_STYLE + "\n</style>", 1)


def _inject_jump_links(html: str) -> str:
    jump = """
<div class="result-jump">
  <a href="#failed-cases">Jump to Failed Cases</a>
  <a href="#passed-cases">Jump to Passed Cases</a>
</div>
"""
    if 'href="#failed-cases"' in html:
        return html
    return html.replace('<div class="container">', '<div class="container">\n' + jump, 1)


def _extract_gcard_body(html: str) -> tuple[str, str, str]:
    marker = '<div class="gcard-body">'
    start = html.find(marker)
    if start == -1:
        raise RuntimeError("Could not find gcard-body in report.")
    body_start = start + len(marker)

    depth = 1
    i = body_start
    while i < len(html):
        next_open = html.find('<div', i)
        next_close = html.find('</div>', i)
        if next_close == -1:
            break
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open + 4
        else:
            depth -= 1
            if depth == 0:
                body_end = next_close
                return html[:body_start], html[body_start:body_end], html[body_end:]
            i = next_close + 6
    raise RuntimeError("Could not parse gcard-body boundaries.")


def _split_case_chunks(body_html: str) -> list[tuple[str, str]]:
    starts = list(re.finditer(r'<div class="trow (pass|fail)"', body_html))
    chunks: list[tuple[str, str]] = []
    for idx, match in enumerate(starts):
        status = match.group(1)
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(body_html)
        chunks.append((status, body_html[start:end].strip()))
    return chunks


def _build_grouped_body(body_html: str) -> str:
    chunks = _split_case_chunks(body_html)
    fail_chunks = [chunk for status, chunk in chunks if status == "fail"]
    pass_chunks = [chunk for status, chunk in chunks if status == "pass"]

    fail_html = "\n".join(fail_chunks) if fail_chunks else '<div class="trow"><div class="trow-main"><div class="trow-body"><div class="trow-q">No failed cases.</div></div></div></div>'
    pass_html = "\n".join(pass_chunks) if pass_chunks else '<div class="trow"><div class="trow-main"><div class="trow-body"><div class="trow-q">No passed cases.</div></div></div></div>'

    return f"""
            <div class="result-sections">
              <div class="result-section" id="failed-cases">
                <div class="result-section-head fail">
                  <div class="result-section-title">Failed Test Cases</div>
                  <div class="result-section-count">{len(fail_chunks)} failed</div>
                </div>
                <div class="result-section-body">
{fail_html}
                </div>
              </div>
              <div class="result-section" id="passed-cases">
                <div class="result-section-head pass">
                  <div class="result-section-title">Passed Test Cases</div>
                  <div class="result-section-count">{len(pass_chunks)} passed</div>
                </div>
                <div class="result-section-body">
{pass_html}
                </div>
              </div>
            </div>
"""


def build_grouped_report(input_path: Path, output_path: Path) -> None:
    html = input_path.read_text(encoding="utf-8", errors="ignore")
    html = _inject_section_style(html)
    html = _inject_jump_links(html)
    before, body, after = _extract_gcard_body(html)
    grouped_body = _build_grouped_body(body)
    html = before + grouped_body + after
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a grouped eval report with failed cases first.")
    parser.add_argument("--input", required=True, help="Path to existing eval report HTML")
    parser.add_argument("--output", help="Path to output grouped report HTML")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input report not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_grouped.html")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_grouped_report(input_path, output_path)
    print(f"Grouped report saved: {output_path}")


if __name__ == "__main__":
    main()
