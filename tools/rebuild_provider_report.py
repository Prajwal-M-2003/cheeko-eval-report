import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from report import build_html_report


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild provider HTML report from saved state.")
    parser.add_argument("--device", default="v1")
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()

    device = (args.device or "v1").strip().lower()
    provider = (args.provider or "google").strip().lower()

    root = ROOT_DIR
    state_path = root / "state" / f"cheeko_{device}_{provider}_executed_results.json"
    report_path = root / "reports" / f"cheeko_{device}_{provider}_eval_report.html"

    results = load_results(state_path)
    tests = list(results.values())
    groups = [{"name": "Group 6: Toxicity", "tests": tests}]

    html = build_html_report(groups, prompt_history=None, prompt_access_password="")
    report_path.write_text(html, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
