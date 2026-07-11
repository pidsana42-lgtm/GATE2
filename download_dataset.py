import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Download and save Typhoon tokenizer and Thai TNHC2 dataset.")
    parser.add_argument("--dataset_name", type=str, default="pythainlp/thai-tnhc2-books",
                        help="Hugging Face dataset name to download")
    parser.add_argument("--tokenizer_name", type=str, default="typhoon-ai/typhoon-7b",
                        help="Hugging Face tokenizer name to download")
    parser.add_argument("--dataset_dir", type=str, default="./dataset",
                        help="Local directory to save the dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="./Tokeniz",
                        help="Local directory to save the tokenizer")
    args = parser.parse_args()

    # 1. Download and save Tokenizer
    print(f"Downloading tokenizer '{args.tokenizer_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        os.makedirs(args.tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(args.tokenizer_dir)
        print(f"Successfully saved tokenizer to {args.tokenizer_dir}\n")
    except Exception as e:
        print(f"Error downloading tokenizer: {e}\n")

    # 2. Download and save Dataset
    print(f"Downloading dataset '{args.dataset_name}'...")
    try:
        dataset = load_dataset(args.dataset_name)
        os.makedirs(args.dataset_dir, exist_ok=True)
        dataset.save_to_disk(args.dataset_dir)
        print(f"Successfully saved dataset to {args.dataset_dir}\n")
    except Exception as e:
        print(f"Error downloading dataset: {e}\n")

if __name__ == "__main__":
    main()
