"""
Results analysis script for AutoQuiz evaluation.

Reads all session JSON files from data/output/ and produces summary
statistics for the dissertation results chapter.

Usage:
    python analyse_results.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("data/output")


def load_sessions() -> list[dict]:
    sessions = []
    for path in OUTPUT_DIR.glob("session_*.json"):
        with open(path, encoding="utf-8") as f:
            try:
                sessions.append(json.load(f))
            except json.JSONDecodeError:
                print(f"  [skip] could not parse {path.name}")
    return sessions


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def pipeline_vs_baseline(sessions: list[dict]):
    section("1. PIPELINE vs BASELINE")

    groups = defaultdict(list)
    for s in sessions:
        mode = s["config"].get("mode", "pipeline")
        groups[mode].append(s)

    for mode, group in sorted(groups.items()):
        sim_scores = [q.get("similarity_score", 0) for s in group for q in s["questions"] if q.get("similarity_score") is not None]
        expl_present = [1 for s in group for q in s["questions"] if q.get("explanation")]
        total_q = sum(len(s["questions"]) for s in group)
        pass_rates = [s["config"].get("validation_pass_rate", 1.0) for s in group]

        print(f"\n  Mode: {mode.upper()}  ({len(group)} sessions, {total_q} questions)")
        print(f"    Avg similarity score   : {avg(sim_scores)}")
        print(f"    Min / Max similarity   : {min(sim_scores):.4f} / {max(sim_scores):.4f}")
        print(f"    Explanation rate       : {len(expl_present)}/{total_q} ({100*len(expl_present)//total_q}%)")
        if mode == "pipeline":
            print(f"    Avg validation pass rate: {avg(pass_rates)}")


def preprocessing_comparison(sessions: list[dict]):
    section("2. PREPROCESSING STRATEGY COMPARISON")

    pipeline = [s for s in sessions if s["config"].get("mode") == "pipeline"]
    groups = defaultdict(list)
    for s in pipeline:
        groups[s["config"].get("strategy", "unknown")].append(s)

    for strategy in ["raw", "clean", "summarise"]:
        group = groups.get(strategy, [])
        if not group:
            print(f"\n  {strategy}: no sessions")
            continue
        sim_scores = [q.get("similarity_score") for s in group for q in s["questions"] if q.get("similarity_score") is not None]
        expl_present = [q for s in group for q in s["questions"] if q.get("explanation")]
        total_q = sum(len(s["questions"]) for s in group)
        print(f"\n  Strategy: {strategy.upper()}  ({len(group)} sessions, {total_q} questions)")
        print(f"    Avg similarity score : {avg(sim_scores)}")
        print(f"    Explanation rate     : {len(expl_present)}/{total_q}")
        print(f"    Pass rate (avg)      : {avg([s['config'].get('validation_pass_rate', 1.0) for s in group])}")


def difficulty_distribution(sessions: list[dict]):
    section("3. DIFFICULTY LABEL DISTRIBUTION")

    pipeline = [s for s in sessions if s["config"].get("mode") == "pipeline"]
    by_difficulty = defaultdict(list)
    for s in pipeline:
        for q in s["questions"]:
            if q.get("similarity_score") is not None:
                by_difficulty[q.get("difficulty", "?")].append(q["similarity_score"])

    for level in ["easy", "medium", "hard", "unspecified"]:
        scores = by_difficulty.get(level, [])
        if scores:
            print(f"  {level:12s}: {len(scores):3d} questions  |  avg sim = {avg(scores)}")


def cognitive_level_distribution(sessions: list[dict]):
    section("4. COGNITIVE LEVEL DISTRIBUTION")

    pipeline = [s for s in sessions if s["config"].get("mode") == "pipeline"]
    by_level = defaultdict(list)
    for s in pipeline:
        for q in s["questions"]:
            if q.get("similarity_score") is not None:
                by_level[q.get("cognitive_level", "?")].append(q["similarity_score"])

    for level in ["recall", "comprehension", "application", "analysis", "unspecified"]:
        scores = by_level.get(level, [])
        if scores:
            print(f"  {level:15s}: {len(scores):3d} questions  |  avg sim = {avg(scores)}")


def explanation_coverage(sessions: list[dict]):
    section("5. EXPLANATION COVERAGE")

    for mode in ["pipeline", "baseline"]:
        group = [s for s in sessions if s["config"].get("mode") == mode]
        if not group:
            continue
        total = sum(len(s["questions"]) for s in group)
        with_expl = sum(1 for s in group for q in s["questions"] if q.get("explanation", "").strip())
        without_expl = total - with_expl
        print(f"\n  {mode.upper()}")
        print(f"    With explanation    : {with_expl}/{total} ({100*with_expl//total if total else 0}%)")
        print(f"    Without explanation : {without_expl}/{total} (discarded by alignment filter)")


def per_file_breakdown(sessions: list[dict]):
    section("6. PER-FILE BREAKDOWN")

    by_file = defaultdict(list)
    for s in sessions:
        by_file[s["config"].get("filename", "unknown")].append(s)

    for filename, group in sorted(by_file.items()):
        sim_scores = [q["similarity_score"] for s in group for q in s["questions"] if q.get("similarity_score") is not None]
        total_q = sum(len(s["questions"]) for s in group)
        print(f"\n  {filename}  ({len(group)} sessions, {total_q} questions)")
        print(f"    Avg similarity: {avg(sim_scores)}")
        modes = sorted(m for m in set(s["config"].get("mode") for s in group) if m)
        strategies = sorted(t for t in set(s["config"].get("strategy") for s in group) if t)
        print(f"    Modes run     : {', '.join(modes) or 'unknown'}")
        print(f"    Strategies run: {', '.join(strategies) or 'unknown'}")


def sample_questions(sessions: list[dict], n: int = 3):
    section("7. SAMPLE QUESTIONS (highest similarity per mode)")

    for mode in ["pipeline", "baseline"]:
        group = [s for s in sessions if s["config"].get("mode") == mode]
        all_q = [(q, s["config"]) for s in group for q in s["questions"] if q.get("similarity_score") is not None]
        top = sorted(all_q, key=lambda x: x[0]["similarity_score"], reverse=True)[:n]
        print(f"\n  {mode.upper()} — top {n} by similarity:")
        for q, cfg in top:
            print(f"\n    [{q['similarity_score']:.4f}] [{cfg.get('strategy','?')}] [{q.get('difficulty','?')}] [{q.get('cognitive_level','?')}]")
            print(f"    Q: {q['question'][:120]}")
            print(f"    A: {q['answer']}) {q['options'].get(q['answer'], '')[:80]}")


def main():
    sessions = load_sessions()
    if not sessions:
        print("No session files found in data/output/")
        return

    print(f"\nLoaded {len(sessions)} sessions from {OUTPUT_DIR}/")
    total_questions = sum(len(s["questions"]) for s in sessions)
    print(f"Total questions across all sessions: {total_questions}")

    pipeline_vs_baseline(sessions)
    preprocessing_comparison(sessions)
    difficulty_distribution(sessions)
    cognitive_level_distribution(sessions)
    explanation_coverage(sessions)
    per_file_breakdown(sessions)
    sample_questions(sessions)

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
