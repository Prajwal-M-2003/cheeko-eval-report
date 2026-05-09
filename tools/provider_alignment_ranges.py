import argparse
import json
import re
from pathlib import Path

import pandas as pd

from provider_report_progress import DEFAULT_CSV, get_provider_state_path


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out: set[str] = set()
    if not isinstance(data, dict):
        return out
    for value in data.values():
        if not isinstance(value, dict):
            continue
        question = value.get("question", "")
        m = re.search(r"\[([0-9a-fA-F]{8,40})\]\s*$", question)
        if m:
            out.add(m.group(1).lower())
    return out


def to_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Show alignment ranges for Google and xAI.")
    parser.add_argument("--device", default="v1")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    args = parser.parse_args()

    device = (args.device or "v1").strip().lower()
    csv_path = args.csv

    df = pd.read_csv(csv_path)
    tox = df[df["toxicity"] == 1].copy().reset_index(drop=True)
    idx_by_id = {
        str(conv_id)[:12].lower(): idx
        for idx, conv_id in enumerate(tox["conv_id"].astype(str).tolist())
    }

    google_ids = load_ids(get_provider_state_path(device, "google"))
    xai_ids = load_ids(get_provider_state_path(device, "xai"))

    xai_missing_indices = [idx_by_id[i] for i in sorted(google_ids - xai_ids) if i in idx_by_id]
    google_missing_indices = [idx_by_id[i] for i in sorted(xai_ids - google_ids) if i in idx_by_id]

    payload = {
        "device": device,
        "google_total": len(google_ids),
        "xai_total": len(xai_ids),
        "common_total": len(google_ids & xai_ids),
        "xai_needs_ranges": to_ranges(xai_missing_indices),
        "google_needs_ranges": to_ranges(google_missing_indices),
    }

    if args.format == "text":
        print(f"Google total: {payload['google_total']}")
        print(f"xAI total: {payload['xai_total']}")
        print(f"Common total: {payload['common_total']}")
        print(f"xAI needs: {payload['xai_needs_ranges']}")
        print(f"Google needs: {payload['google_needs_ranges']}")
        return

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
