import argparse
import os

from provider_report_progress import (
    DEFAULT_CSV,
    extract_report_ids,
    get_provider_report_path,
    load_toxic_csv_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve next toxic CSV start index missing from either Google or xAI."
    )
    parser.add_argument("--target-device", default=os.getenv("TARGET_DEVICE", "v1"))
    parser.add_argument("--csv", default=os.getenv("TOXIC_CSV_PATH", DEFAULT_CSV))
    args = parser.parse_args()

    device = (args.target_device or "v1").strip().lower()
    csv_path = args.csv

    all_ids = load_toxic_csv_ids(csv_path)
    google_ids = extract_report_ids(get_provider_report_path(device, "google"))
    xai_ids = extract_report_ids(get_provider_report_path(device, "xai"))

    for idx, cid in enumerate(all_ids):
        if cid not in google_ids or cid not in xai_ids:
            print(idx)
            return

    print(-1)


if __name__ == "__main__":
    main()
