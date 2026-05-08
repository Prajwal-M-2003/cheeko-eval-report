import argparse
import os
from provider_report_progress import (
    DEFAULT_CSV,
    extract_report_ids,
    get_provider_report_path,
    load_toxic_csv_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve next pending toxic CSV start index.")
    parser.add_argument("--target-device", default=os.getenv("TARGET_DEVICE", "v1"))
    parser.add_argument("--provider", default=os.getenv("CHEEKO_PROVIDER", "google"))
    parser.add_argument("--csv", default=os.getenv("TOXIC_CSV_PATH", DEFAULT_CSV))
    args = parser.parse_args()

    device = (args.target_device or "v1").strip().lower()
    provider = (args.provider or "google").strip().lower()
    csv_path = args.csv

    all_ids = load_toxic_csv_ids(csv_path)
    report_path = get_provider_report_path(device, provider)
    done_ids = extract_report_ids(report_path)

    for idx, cid in enumerate(all_ids):
        if cid not in done_ids:
            print(idx)
            return

    print(-1)


if __name__ == "__main__":
    main()
