param(
    [int]$ToxicIndex = 0,
    [string]$ConvId = "",
    [string]$TargetDevice = "v1",
    [string]$Provider = "xai"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$csvPath = if ($env:TOXIC_CSV_PATH -and $env:TOXIC_CSV_PATH.Trim() -ne "") {
    $env:TOXIC_CSV_PATH.Trim()
} else {
    "C:\Users\mpraj\Downloads\toxic-chat_annotation_all.csv"
}

$convIdNorm = $ConvId.Trim().ToLower()
if ($convIdNorm -ne "") {
    # Build selected_eval_questions.csv from exact conv_id match in toxic subset.
    $env:TOXIC_CSV_PATH = $csvPath
    $env:ONE_CONV_ID = $convIdNorm
    @'
import os
import sys
import pandas as pd

csv_path = os.getenv("TOXIC_CSV_PATH", r"C:\Users\mpraj\Downloads\toxic-chat_annotation_all.csv")
conv_id = (os.getenv("ONE_CONV_ID", "") or "").strip().lower()

if not conv_id:
    print("ERROR: ONE_CONV_ID is empty.")
    sys.exit(1)

df = pd.read_csv(csv_path)
tox = df[df["toxicity"] == 1].copy().reset_index(drop=True)
m = tox[tox["conv_id"].astype(str).str.lower().str.startswith(conv_id)]

if len(m) == 0:
    print(f"ERROR: conv_id '{conv_id}' not found in toxic subset.")
    sys.exit(2)

row = m.iloc[[0]].copy()
row["case_type"] = "toxic"
row.insert(0, "case_id", [int(m.index[0]) + 1])
out = row[["case_id", "case_type", "conv_id", "toxicity", "jailbreaking", "user_input"]]
out.to_csv("selected_eval_questions.csv", index=False, encoding="utf-8")

q = str(out.iloc[0]["user_input"]).replace("\n", " ").strip()
print("Selected rows for eval: 1")
print("Copy these questions into eval.py:")
print(f"[{int(out.iloc[0]['case_id'])}] (toxic) {q}")
'@ | python -
} else {
    $env:TOXIC_INDEX = "$ToxicIndex"
    python .\local_csv_preview.py
}

$env:TARGET_DEVICE = "$TargetDevice"
$env:CHEEKO_PROVIDER = "$Provider"
$env:USE_SELECTED_CSV_CASES = "false"

python .\tools\inject_csv_to_eval.py --max 1

# Force re-evaluation so fixes update existing case score in cumulative report.
Remove-Item ".\state\cheeko_${TargetDevice}_${Provider}_eval_checkpoint.json" -ErrorAction SilentlyContinue

python .\eval.py

Write-Host ""
Write-Host "Cumulative report (all executed so far):"
Write-Host "file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_${Provider}_eval_report.html)"
