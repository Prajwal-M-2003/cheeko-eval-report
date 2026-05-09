import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path


def load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def extract_id(question: str) -> str:
    m = re.search(r"\[([0-9a-fA-F]{8,40})\]\s*$", question or "")
    return m.group(1).lower() if m else ""


def shorten_question(question: str) -> str:
    return (question or "Unknown Question").strip()


def extract_user_and_assistant(actual: str) -> tuple[str, str]:
    text = actual or ""
    user_match = re.search(r"\[USER\]\s*(.*?)(?=\n\n---\n\n\[ASSISTANT\]|\n\[ASSISTANT\]|\Z)", text, flags=re.DOTALL)
    assistant_match = re.search(r"\[ASSISTANT\]\s*(.*?)(?=\n\n---\n\n\[USER\]|\n\[USER\]|\Z)", text, flags=re.DOTALL)
    user_text = user_match.group(1).strip() if user_match else ""
    assistant_text = assistant_match.group(1).strip() if assistant_match else text.strip()
    return user_text, assistant_text


def summarize_metrics(metrics: list[dict]) -> str:
    if not metrics:
        return "No metrics"
    parts = []
    for metric in metrics:
        name = re.sub(r"\s*\[.*?\]", "", metric.get("name", "")).strip()
        score = metric.get("score")
        success = metric.get("success")
        badge = "PASS" if success else "FAIL"
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        parts.append(f"{name}: {badge} ({score_text})")
    return " | ".join(parts)


def build_index(results: dict) -> dict:
    index = {}
    for _, item in results.items():
        question = item.get("question", "")
        qid = extract_id(question)
        if not qid:
            continue
        index[qid] = item
    return index


def ordered_ids_from_results(results: dict) -> list[str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for _, item in results.items():
        if not isinstance(item, dict):
            continue
        qid = extract_id(item.get("question", ""))
        if not qid or qid in seen:
            continue
        ordered_ids.append(qid)
        seen.add(qid)
    return ordered_ids


def esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_html(device: str, gemini_results: dict, xai_results: dict) -> str:
    gemini_index = build_index(gemini_results)
    xai_index = build_index(xai_results)
    common_id_set = set(gemini_index).intersection(xai_index)
    gemini_order = ordered_ids_from_results(gemini_results)
    common_ids = [qid for qid in gemini_order if qid in common_id_set]

    cards = []
    gemini_pass = 0
    xai_pass = 0

    for qid in common_ids:
        gem = gemini_index[qid]
        xai = xai_index[qid]

        question = shorten_question(gem.get("question") or xai.get("question"))
        user_text, gemini_response = extract_user_and_assistant(gem.get("actual", ""))
        _, xai_response = extract_user_and_assistant(xai.get("actual", ""))

        gem_ok = bool(gem.get("success"))
        xai_ok = bool(xai.get("success"))
        gemini_pass += 1 if gem_ok else 0
        xai_pass += 1 if xai_ok else 0

        gemini_metrics = summarize_metrics(gem.get("metrics", []))
        xai_metrics = summarize_metrics(xai.get("metrics", []))

        cards.append(
            f"""
            <section class="card">
              <div class="card-head">
                <div>
                  <div class="question">{esc(question)}</div>
                  <div class="qid">ID: {esc(qid)}</div>
                </div>
              </div>
              <div class="prompt-block">
                <div class="block-title">Prompt</div>
                <pre>{esc(user_text)}</pre>
              </div>
              <div class="compare-grid">
                <div class="provider gemini">
                  <div class="provider-head">
                    <span class="provider-name">Gemini</span>
                    <span class="status {'pass' if gem_ok else 'fail'}">{'PASS' if gem_ok else 'FAIL'}</span>
                  </div>
                  <div class="metric-line">{esc(gemini_metrics)}</div>
                  <pre>{esc(gemini_response)}</pre>
                </div>
                <div class="provider xai">
                  <div class="provider-head">
                    <span class="provider-name">xAI</span>
                    <span class="status {'pass' if xai_ok else 'fail'}">{'PASS' if xai_ok else 'FAIL'}</span>
                  </div>
                  <div class="metric-line">{esc(xai_metrics)}</div>
                  <pre>{esc(xai_metrics and xai_response or xai_response)}</pre>
                </div>
              </div>
            </section>
            """
        )

    total = len(common_ids)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CHEEKO Provider Comparison Report</title>
  <style>
    :root {{
      --navy:#172554;
      --ink:#0f172a;
      --muted:#64748b;
      --line:#dbe4f0;
      --card:#ffffff;
      --bg:#f8fafc;
      --gemini:#2563eb;
      --xai:#0f766e;
      --pass:#059669;
      --fail:#dc2626;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    .hero {{
      background:linear-gradient(135deg, #1e1b4b 0%, #1f3b4d 100%);
      color:#fff;
      padding:48px 20px 28px;
      text-align:center;
    }}
    .hero h1 {{ margin:0; font-size:56px; line-height:1; font-weight:900; }}
    .hero h1 .accent {{ color:#818cf8; }}
    .hero p {{ margin:14px 0 0; font-size:14px; color:#dbeafe; }}
    .wrap {{ max-width:1400px; margin:0 auto; padding:24px 20px 48px; }}
    .stats {{
      display:grid;
      gap:18px;
      grid-template-columns:repeat(3, minmax(0,1fr));
      margin-top:-56px;
      margin-bottom:24px;
    }}
    .stat {{
      background:#fff;
      border-radius:22px;
      padding:26px 20px;
      box-shadow:0 20px 40px rgba(15,23,42,.12);
      text-align:center;
    }}
    .stat .num {{ font-size:54px; font-weight:900; line-height:1; }}
    .stat .lbl {{ margin-top:8px; font-size:14px; font-weight:800; letter-spacing:.08em; color:#94a3b8; text-transform:uppercase; }}
    .stat.total .num {{ color:#2563eb; }}
    .stat.gemini .num {{ color:var(--gemini); }}
    .stat.xai .num {{ color:var(--xai); }}
    .note {{
      background:#fff;
      border:1px solid var(--line);
      border-radius:18px;
      padding:16px 18px;
      color:var(--muted);
      margin-bottom:22px;
    }}
    .card {{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:0 12px 26px rgba(15,23,42,.07);
      margin-bottom:20px;
      overflow:hidden;
    }}
    .card-head {{
      padding:20px 22px 14px;
      border-bottom:1px solid var(--line);
      background:#f8fbff;
    }}
    .question {{ font-size:18px; font-weight:900; color:var(--navy); }}
    .qid {{ margin-top:6px; font-size:13px; color:var(--muted); font-weight:700; }}
    .prompt-block {{ padding:18px 22px 0; }}
    .block-title {{ font-size:13px; font-weight:900; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; }}
    .compare-grid {{
      display:grid;
      grid-template-columns:repeat(2, minmax(0,1fr));
      gap:18px;
      padding:18px 22px 22px;
    }}
    .provider {{
      border:1px solid var(--line);
      border-radius:18px;
      overflow:hidden;
      background:#fff;
    }}
    .provider.gemini {{ border-top:5px solid var(--gemini); }}
    .provider.xai {{ border-top:5px solid var(--xai); }}
    .provider-head {{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      padding:14px 16px 8px;
    }}
    .provider-name {{ font-size:20px; font-weight:900; }}
    .metric-line {{ padding:0 16px 10px; font-size:13px; color:var(--muted); font-weight:700; }}
    .status {{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:66px;
      padding:8px 12px;
      border-radius:999px;
      font-size:12px;
      font-weight:900;
      color:#fff;
    }}
    .status.pass {{ background:var(--pass); }}
    .status.fail {{ background:var(--fail); }}
    pre {{
      margin:0;
      padding:0 16px 18px;
      white-space:pre-wrap;
      word-break:break-word;
      font:16px/1.55 Segoe UI, Arial, sans-serif;
      color:var(--ink);
    }}
    @media (max-width: 980px) {{
      .stats {{ grid-template-columns:1fr; margin-top:20px; }}
      .compare-grid {{ grid-template-columns:1fr; }}
      .hero h1 {{ font-size:42px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <h1><span class="accent">CHEEKO</span> Provider Comparison Report</h1>
    <p>Generated {esc(now)} | device={esc(device)} | same questions with Gemini and xAI responses</p>
  </header>
  <main class="wrap">
    <section class="stats">
      <div class="stat total">
        <div class="num">{total}</div>
        <div class="lbl">Common Questions</div>
      </div>
      <div class="stat gemini">
        <div class="num">{gemini_pass}</div>
        <div class="lbl">Gemini Passed</div>
      </div>
      <div class="stat xai">
        <div class="num">{xai_pass}</div>
        <div class="lbl">xAI Passed</div>
      </div>
    </section>
    <section class="note">
      This is a comparison-only report. Your normal Gemini and xAI reports stay separate and unchanged.
    </section>
    {''.join(cards) if cards else '<section class="note">No common questions found between Gemini and xAI state files.</section>'}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gemini vs xAI comparison report.")
    parser.add_argument("--device", default="v1")
    args = parser.parse_args()

    device = (args.device or "v1").strip().lower()
    root = Path(os.getcwd())
    state_dir = root / "state"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    gemini_path = state_dir / f"cheeko_{device}_google_executed_results.json"
    xai_path = state_dir / f"cheeko_{device}_xai_executed_results.json"

    html = build_html(
        device=device,
        gemini_results=load_results(gemini_path),
        xai_results=load_results(xai_path),
    )

    out_path = reports_dir / f"cheeko_{device}_provider_comparison_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
