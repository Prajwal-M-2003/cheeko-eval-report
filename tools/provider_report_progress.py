import os
import re
import json
from pathlib import Path

import pandas as pd


DEFAULT_CSV = r"C:\Users\mpraj\Downloads\toxic-chat_annotation_all.csv"


def get_provider_report_path(device: str, provider: str) -> Path:
    return Path(os.getcwd()) / "reports" / f"cheeko_{device}_{provider}_eval_report.html"


def get_provider_state_path(device: str, provider: str) -> Path:
    return Path(os.getcwd()) / "state" / f"cheeko_{device}_{provider}_executed_results.json"


def _load_state_results(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def extract_report_total(report_path: Path) -> int:
    state_path = get_provider_state_path_from_report(report_path)
    state_results = _load_state_results(state_path)
    if state_results:
        return len(state_results)
    if not report_path.exists():
        return 0
    html = report_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<b>(\d+)</b>\s+test cases evaluated", html)
    return int(m.group(1)) if m else 0


def extract_report_ids(report_path: Path) -> set[str]:
    state_path = get_provider_state_path_from_report(report_path)
    state_results = _load_state_results(state_path)
    if state_results:
        ids = set()
        for key in state_results:
            ids.update(m.lower() for m in re.findall(r"\[([0-9a-fA-F]{8,})\]", str(key)))
        return ids
    if not report_path.exists():
        return set()
    html = report_path.read_text(encoding="utf-8", errors="ignore")
    return {m.lower() for m in re.findall(r"\[([0-9a-fA-F]{8,})\]", html)}


def get_provider_state_path_from_report(report_path: Path) -> Path:
    name = report_path.name
    m = re.match(r"cheeko_([a-z0-9_-]+)_([a-z0-9_-]+)_eval_report(?:_latest)?\.html$", name)
    if not m:
        return Path(os.getcwd()) / "state" / "missing_provider_state.json"
    return get_provider_state_path(m.group(1), m.group(2))


def load_toxic_csv_ids(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    toxic_df = df[df["toxicity"] == 1].copy()
    return toxic_df["conv_id"].astype(str).str.slice(0, 12).str.lower().tolist()
