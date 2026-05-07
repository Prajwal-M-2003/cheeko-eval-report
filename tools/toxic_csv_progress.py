import argparse
import json
import os
import re

import pandas as pd


DEFAULT_CSV = r"C:\Users\mpraj\Downloads\toxic-chat_annotation_all.csv"


def _extract_short_ids_from_state(state_path: str) -> set[str]:
    if not os.path.exists(state_path):
        return set()
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    ids = set()
    for key in (data or {}).keys():
        # Example key:
        # Group 6: Toxicity::Toxic Prompt - ... [e167541c122e]
        if "Group 6: Toxicity::" not in key:
            continue
        # Strictly count only CSV-injected style titles.
        # This avoids mixing unrelated/manual/non-CSV toxicity tests.
        if "Group 6: Toxicity::Toxic Prompt - " not in key:
            continue
        m = re.search(r"\[([0-9a-fA-F]{8,})\]", key)
        if m:
            ids.add(m.group(1).lower())
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Show toxicity CSV execution progress.")
    parser.add_argument("--target-device", default=os.getenv("TARGET_DEVICE", "v1"), help="v1 or v2")
    parser.add_argument("--csv", default=os.getenv("TOXIC_CSV_PATH", DEFAULT_CSV), help="Path to toxicity CSV")
    args = parser.parse_args()

    device = (args.target_device or "v1").strip().lower()
    csv_path = args.csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    toxic_df = df[df["toxicity"] == 1].copy()
    toxic_df["conv_short"] = toxic_df["conv_id"].astype(str).str.slice(0, 12).str.lower()

    all_ids = toxic_df["conv_short"].tolist()
    all_id_set = set(all_ids)

    state_path = os.path.join(os.getcwd(), "state", f"cheeko_{device}_executed_results.json")
    done_ids = _extract_short_ids_from_state(state_path)
    done_in_csv = done_ids.intersection(all_id_set)

    total = len(toxic_df)
    done = len(done_in_csv)
    pending = max(total - done, 0)
    pct = (done / total * 100.0) if total else 0.0

    next_pending_index = None
    next_pending_id = None
    for idx, cid in enumerate(all_ids):
        if cid not in done_in_csv:
            next_pending_index = idx
            next_pending_id = cid
            break

    print("")
    print("=== Toxic CSV Progress ===")
    print(f"Device: {device}")
    print(f"CSV: {csv_path}")
    print(f"Total toxic rows: {total}")
    print(f"Done (executed in Group 6): {done}")
    print(f"Pending: {pending}")
    print(f"Progress: {pct:.1f}%")
    if next_pending_index is not None:
        print(f"Next pending start index: {next_pending_index} (conv_id short: {next_pending_id})")
    else:
        print("Next pending start index: none (all toxic rows completed)")
    print("==========================")
    print("")


if __name__ == "__main__":
    main()
