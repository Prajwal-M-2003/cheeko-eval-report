import os

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


def main():
    # Requires Hugging Face access approval for allenai/wildguardmix.
    # Optional: set HF_TOKEN in environment if needed.
    token = os.getenv("HF_TOKEN")
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test", token=token)
        print("Rows:", len(ds))
        print("Columns:", ds.column_names)
        print("\nFirst row:\n", ds[0])
    except DatasetNotFoundError:
        print("\nAccess blocked: 'allenai/wildguardmix' is gated.")
        print("Do these 3 steps once:")
        print("1) Open: https://huggingface.co/datasets/allenai/wildguardmix")
        print("2) Log in and click 'Agree/Accept' for dataset access")
        print("3) Run: huggingface-cli login  (paste your HF token)")
        print("\nThen run this file again.")
    except Exception as e:
        print("\nUnexpected error while loading dataset:")
        print(type(e).__name__, "-", str(e))


if __name__ == "__main__":
    main()
