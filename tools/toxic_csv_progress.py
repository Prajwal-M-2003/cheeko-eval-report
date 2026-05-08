import argparse
import os
from provider_report_progress import (
    DEFAULT_CSV,
    extract_report_ids,
    extract_report_total,
    get_provider_report_path,
    load_toxic_csv_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show toxicity CSV execution progress.")
    parser.add_argument("--target-device", default=os.getenv("TARGET_DEVICE", "v1"), help="v1 or v2")
    parser.add_argument("--provider", default=os.getenv("CHEEKO_PROVIDER", "google"), help="google or xai")
    parser.add_argument("--csv", default=os.getenv("TOXIC_CSV_PATH", DEFAULT_CSV), help="Path to toxicity CSV")
    args = parser.parse_args()

    device = (args.target_device or "v1").strip().lower()
    provider = (args.provider or "google").strip().lower()
    csv_path = args.csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    all_ids = load_toxic_csv_ids(csv_path)
    all_id_set = set(all_ids)

    report_path = get_provider_report_path(device, provider)
    done_ids = extract_report_ids(report_path)
    done_in_csv = done_ids.intersection(all_id_set)
    report_total = extract_report_total(report_path)

    total = len(all_ids)
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
    print(f"Provider: {provider}")
    print(f"CSV: {csv_path}")
    print(f"Report file: {report_path}")
    print(f"Total toxic rows: {total}")
    print(f"Done in provider report: {done}")
    print(f"Total tests in provider report: {report_total}")
    print(f"Pending: {pending}")
    print(f"Progress: {pct:.1f}%")
    print(f"Next sequential case number: {report_total + 1 if pending else 'none'}")
    if next_pending_index is not None:
        print(f"Next pending start index: {next_pending_index} (conv_id short: {next_pending_id})")
    else:
        print("Next pending start index: none (all toxic rows completed)")
    print("==========================")
    print("")


if __name__ == "__main__":
    main()
