"""
Main method for testing 
"""

import argparse
from backend.preprocessing.pipeline import preprocess


def main():
    parser = argparse.ArgumentParser(description="AutoQuiz preprocessing test")
    parser.add_argument("file", help="Path to input document (.pdf, .pptx, or .txt)")
    parser.add_argument(
        "--strategy",
        choices=["raw", "clean", "summarise"],
        default="clean",
        help="Preprocessing strategy (default: clean)",
    )
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=40)
    args = parser.parse_args()

    print(f"Running preprocessing: strategy='{args.strategy}' on '{args.file}'\n")
    chunks = preprocess(args.file, strategy=args.strategy, max_words=args.max_words, min_words=args.min_words)

    print(f"Produced {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()


if __name__ == "__main__":
    main()
