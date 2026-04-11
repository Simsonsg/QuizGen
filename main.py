"""
AutoQuiz — development entry point.

Usage:
    python main.py <file> [options]

Examples:
    python main.py data/input/lecture.pdf
    python main.py data/input/lecture.pdf --strategy clean --difficulty hard --cognitive analysis --candidates 4
    python main.py data/input/lecture.pdf --no-answerability-check
"""

import argparse
from backend.preprocessing.pipeline import preprocess
from backend.generation import generate_all
from backend.validation import filter_all
from backend.explanation import generate_and_attach


def main():
    parser = argparse.ArgumentParser(description="AutoQuiz pipeline")
    parser.add_argument("file", help="Path to input document (.pdf, .pptx, or .txt)")
    parser.add_argument("--strategy", choices=["raw", "clean", "summarise"], default="summarise")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--cognitive", choices=["recall", "comprehension", "application", "analysis"], default="recall")
    parser.add_argument("--candidates", type=int, default=3, help="Candidate questions per chunk")
    parser.add_argument("--similarity-threshold", type=float, default=0.25)
    parser.add_argument("--no-answerability-check", action="store_true")
    parser.add_argument("--no-explanations", action="store_true")
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=40)
    args = parser.parse_args()

    print(f"[1/4] Preprocessing '{args.file}' (strategy={args.strategy})...")
    chunks = preprocess(args.file, strategy=args.strategy, max_words=args.max_words, min_words=args.min_words)
    print(f"      {len(chunks)} chunks produced.\n")

    print(f"[2/4] Generating questions (difficulty={args.difficulty}, cognitive={args.cognitive}, candidates={args.candidates})...")
    all_candidates = generate_all(chunks, difficulty=args.difficulty, cognitive_level=args.cognitive, n=args.candidates)
    total_candidates = sum(len(c) for c in all_candidates)
    print(f"      {total_candidates} candidates generated.\n")

    check_answerability = not args.no_answerability_check
    print(f"[3/4] Validating (similarity_threshold={args.similarity_threshold}, answerability={check_answerability})...")
    validated = filter_all(all_candidates, similarity_threshold=args.similarity_threshold, check_answerability=check_answerability)
    print(f"      {len(validated)}/{total_candidates} questions passed validation.\n")

    if not args.no_explanations and validated:
        print(f"[4/4] Generating explanations for {len(validated)} questions...")
        generate_and_attach(validated)
        print("      Done.\n")
    else:
        print("[4/4] Skipping explanation generation.\n")

    print("=== Validated Questions ===\n")
    for i, q in enumerate(validated, 1):
        print(f"Q{i} [{q.difficulty} / {q.cognitive_level}]: {q.question}")
        for letter, text in q.options.items():
            marker = " <--" if letter == q.answer else ""
            print(f"     {letter}) {text}{marker}")
        if q.explanation:
            print(f"     Explanation: {q.explanation}")
        print()


if __name__ == "__main__":
    main()


#python main.py data/input/lecture.pdf --candidates 4 --no-answerability-check
#python main.py data/input/your_file.pdf --strategy clean
#python main.py data/input/your_file.pdf --strategy summarise

#python main.py data/input/lecture.pdf - full pipeline
#python main.py data/input/lecture.pdf --no-explanations --no-answerability-check
