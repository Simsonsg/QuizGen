"""
Compute average similarity score and explanation coverage for each session JSON.
"""

import csv
import json
import os

QUESTIONS_CSV = "data/output/annotation_questions.csv"
OUTPUT_DIR    = "data/output"


def load_session_labels():
    """Map question text -> session label (S1-S16) from annotation CSV."""
    q_to_session = {}
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q_to_session[row["question"].strip()] = row["session"]
    return q_to_session


def identify_session(questions, q_to_session):
    """Return S1-S16 label by majority vote across all questions in the file."""
    votes = {}
    for q in questions:
        label = q_to_session.get(q.get("question", "").strip())
        if label:
            votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get) if votes else None


def compute_stats(questions):
    n = len(questions)
    if n == 0:
        return None
    sim_scores = [q.get("similarity_score") for q in questions if q.get("similarity_score") is not None]
    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
    with_expl = sum(1 for q in questions if q.get("explanation", "").strip())
    expl_pct = with_expl / n * 100
    return {"n": n, "avg_sim": avg_sim, "with_expl": with_expl, "expl_pct": expl_pct}


def main():
    q_to_session = load_session_labels()

    results = {}
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        if not questions:
            continue
        label = identify_session(questions, q_to_session)
        if label:
            results[label] = compute_stats(questions)

    # Sort by session label S1..S16
    def sort_key(s):
        return int(s[1:])

    print(f"{'Session':<10} {'N':>4}  {'Avg Sim':>8}  {'With Expl':>10}  {'Expl %':>8}")
    print("-" * 50)
    for label in sorted(results, key=sort_key):
        s = results[label]
        print(f"{label:<10} {s['n']:>4}  {s['avg_sim']:>8.4f}  {s['with_expl']:>10}  {s['expl_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
