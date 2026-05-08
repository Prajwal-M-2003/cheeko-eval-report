import pandas as pd
import os
from tools.provider_report_progress import extract_report_total, get_provider_report_path


def main():
    csv_path = os.getenv(
        "TOXIC_CSV_PATH",
        r"C:\Users\mpraj\Downloads\toxic-chat_annotation_all.csv",
    )
    df = pd.read_csv(csv_path)

    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    toxic_df = df[df["toxicity"] == 1].copy()
    jailbreak_df = df[df["jailbreaking"] == 1].copy()
    safe_df = df[df["toxicity"] == 0].copy()

    print("Toxic rows:", len(toxic_df))
    print("Jailbreak rows:", len(jailbreak_df))
    print("Safe rows:", len(safe_df))

    # Batch selection from toxic rows.
    # Environment controls:
    #   TOXIC_START_INDEX = start row index inside filtered toxic set (0-based)
    #   TOXIC_COUNT       = how many toxic rows to select
    # Backward compatibility:
    #   TOXIC_INDEX still works as single-row shorthand.
    idx_env = os.getenv("TOXIC_INDEX")
    if idx_env is not None and idx_env != "":
        toxic_start = int(idx_env)
        toxic_count = 1
    else:
        toxic_start = int(os.getenv("TOXIC_START_INDEX", "0") or "0")
        toxic_count = int(os.getenv("TOXIC_COUNT", "1") or "1")

    device = (os.getenv("TARGET_DEVICE", "v1") or "v1").strip().lower()
    provider = (os.getenv("CHEEKO_PROVIDER", "google") or "google").strip().lower()
    report_total = extract_report_total(get_provider_report_path(device, provider))

    toxic_start = max(0, toxic_start)
    toxic_count = max(1, toxic_count)
    toxic_end = min(len(toxic_df), toxic_start + toxic_count)

    if toxic_start >= len(toxic_df):
        raise ValueError(
            f"TOXIC_START_INDEX={toxic_start} is out of range for toxic rows count={len(toxic_df)}"
        )

    t = toxic_df.iloc[toxic_start:toxic_end].copy()
    t["case_type"] = "toxic"
    j = jailbreak_df.head(0).copy()
    j["case_type"] = "jailbreak"
    s = safe_df.head(0).copy()
    s["case_type"] = "safe"

    selected = pd.concat([t, j, s], ignore_index=True)
    selected = selected[["case_type", "conv_id", "toxicity", "jailbreaking", "user_input"]]
    selected.insert(0, "case_id", range(report_total + 1, report_total + 1 + len(selected)))

    out_csv = "selected_eval_questions.csv"
    selected.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved easy shortlist: {out_csv}")

    print(f"Selected rows for eval: {len(selected)}")
    print(f"Sequential case_id start: {report_total + 1}")
    print(f"CSV start index used internally: {toxic_start}")
    print("\nCopy these questions into eval.py:")
    for _, row in selected.iterrows():
        q = str(row["user_input"]).replace("\n", " ").strip()
        print(f"[{row['case_id']}] ({row['case_type']}) {q}")


if __name__ == "__main__":
    main()
