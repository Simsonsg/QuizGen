"""
AutoQuiz — development entry point.

Usage:
    python main.py <file> [options]

Examples:
    python main.py data/input/lecture.pdf
    python main.py data/input/lecture.pdf --strategy clean --difficulty hard --cognitive analysis --candidates 4
"""

import argparse
from backend.preprocessing.pipeline import preprocess
from backend.generation import generate_all


def main():
    parser = argparse.ArgumentParser(description="AutoQuiz pipeline")
    parser.add_argument("file", help="Path to input document (.pdf, .pptx, or .txt)")
    parser.add_argument("--strategy", choices=["raw", "clean", "summarise"], default="clean")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--cognitive", choices=["recall", "comprehension", "application", "analysis"], default="recall")
    parser.add_argument("--candidates", type=int, default=3, help="Candidate questions per chunk")
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=40)
    args = parser.parse_args()

    print(f"[1/2] Preprocessing '{args.file}' (strategy={args.strategy})...")
    chunks = preprocess(args.file, strategy=args.strategy, max_words=args.max_words, min_words=args.min_words)
    print(f"      {len(chunks)} chunks produced.\n")

    print(f"[2/2] Generating questions (difficulty={args.difficulty}, cognitive={args.cognitive}, candidates={args.candidates})...")
    all_candidates = generate_all(chunks, difficulty=args.difficulty, cognitive_level=args.cognitive, n=args.candidates)

    total = sum(len(c) for c in all_candidates)
    print(f"      {total} candidate questions generated.\n")

    for chunk_i, candidates in enumerate(all_candidates, 1):
        print(f"=== Chunk {chunk_i} ===")
        for q_i, q in enumerate(candidates, 1):
            print(f"  Q{q_i} [{q.difficulty} / {q.cognitive_level}]: {q.question}")
            for letter, text in q.options.items():
                marker = " <--" if letter == q.answer else ""
                print(f"       {letter}) {text}{marker}")
        print()


if __name__ == "__main__":
    main()
