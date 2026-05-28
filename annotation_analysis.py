"""
Annotation analysis scripts for dissertation results.

Sections:
  1. Session-level summary stats
  2. Cognitive-level aggregation table (Table 4.2)
  3. Structured error analysis (Option 3)
  4. Representative question extraction (Option 4)

Usage:
  python annotation_analysis.py
"""

import csv
from collections import Counter

QUESTIONS_CSV = "data/output/annotation_questions.csv"
SESSIONS_CSV  = "data/output/annotation_sessions.csv"


def load_questions():
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sessions():
    with open(SESSIONS_CSV, newline="", encoding="utf-8") as f:
        return {r["session"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# 1. Session-level summary
# ---------------------------------------------------------------------------
def session_summary():
    sessions = load_sessions()
    print("=== Session Summary ===")
    header = f"{'Session':<8} {'Source':<8} {'Strategy':<10} {'Diff':<8} {'Cog':<15} {'N':>3} {'AvgSim':>8} {'Pass%':>7} {'Expl%':>7} {'Clarity':>8} {'Corr%':>7} {'Dist':>6} {'CogMatch%':>10}"
    print(header)
    for s, r in sessions.items():
        print(
            f"{r['session']:<8} {r['source']:<8} {r['strategy']:<10} {r['difficulty']:<8} "
            f"{r['cognitive']:<15} {r['N']:>3} {float(r['avg_sim']):>8.4f} "
            f"{float(r['pass_rate'])*100:>6.1f}% {float(r['expl_rate'])*100:>6.1f}% "
            f"{float(r['clarity']):>8.2f} {float(r['correctness_pct']):>6.1f}% "
            f"{float(r['distractor']):>6.2f} {r['cog_match_pct']:>10}"
        )


# ---------------------------------------------------------------------------
# 2. Cognitive-level aggregation
# ---------------------------------------------------------------------------
def cog_level_table():
    sessions = load_sessions()

    groups = {
        "Recall":        ["S1", "S2", "S3", "S7", "S9", "S10", "S11", "S15"],
        "Comprehension": ["S4", "S12"],
        "Analysis":      ["S5", "S13"],
        "Application":   ["S6", "S14"],
    }

    print("\n=== Cognitive-Level Aggregation ===")
    print(f"{'Level':<15} {'Sessions':<40} {'N':>5} {'Match':>12} {'Rate':>7} {'AvgSim':>8}")
    for level, sess_list in groups.items():
        total_n, total_match, sim_sum = 0, 0, 0.0
        for s in sess_list:
            r = sessions[s]
            n = int(r["N"])
            pct = float(r["cog_match_pct"]) if r["cog_match_pct"] != "-1" else 0.0
            total_n   += n
            total_match += round(n * pct / 100)
            sim_sum   += float(r["avg_sim"]) * n
        rate = total_match / total_n * 100
        avg_sim = sim_sum / total_n
        print(
            f"{level:<15} {', '.join(sess_list):<40} {total_n:>5} "
            f"{total_match:>4}/{total_n:<4} ({rate:>5.1f}%) {avg_sim:>8.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Structured error analysis
# ---------------------------------------------------------------------------

# Manually classified defect types based on question review
CIRCULAR  = {("S3", "7")}
INVENTED  = {("S1", "10"), ("S15", "13")}


def classify_defects(rows):
    defects = {}
    for r in rows:
        key = (r["session"], r["q_num"])
        c, cl, d = int(r["correctness"]), int(r["clarity"]), int(r["distractor"])
        types = []
        if key in CIRCULAR:
            types.append("Circular")
        elif cl <= 2 and c == 1 and d == 1:
            types.append("Poorly worded")
        if c == 0:
            types.append("Invented content" if key in INVENTED else "Wrong answer")
        if types:
            defects[key] = {"row": r, "types": types}
    return defects


def error_analysis():
    rows = load_questions()
    defects = classify_defects(rows)

    print(f"\n=== Error Analysis ===")
    print(f"Total defective: {len(defects)} / {len(rows)} ({100*len(defects)/len(rows):.1f}%)\n")

    type_counter = Counter(t for v in defects.values() for t in v["types"])
    print("By defect type:")
    for t, n in type_counter.most_common():
        print(f"  {t:<25} {n}")

    print("\nBy difficulty:")
    for diff in ["easy", "medium", "hard", "—"]:
        total = sum(1 for r in rows if r["difficulty"] == diff)
        n_def = sum(1 for v in defects.values() if v["row"]["difficulty"] == diff)
        if total:
            print(f"  {diff:<10} {n_def:>2}/{total:<4} ({100*n_def/total:>5.1f}%)")

    print("\nBy strategy:")
    for strat in ["clean", "summarise", "baseline"]:
        total = sum(1 for r in rows if r["strategy"] == strat)
        n_def = sum(1 for v in defects.values() if v["row"]["strategy"] == strat)
        print(f"  {strat:<12} {n_def:>2}/{total:<4} ({100*n_def/total:>5.1f}%)")

    print("\nBy source:")
    for src in ["RL", "Ethics"]:
        total = sum(1 for r in rows if r["source"] == src)
        n_def = sum(1 for v in defects.values() if v["row"]["source"] == src)
        print(f"  {src:<10} {n_def:>2}/{total:<4} ({100*n_def/total:>5.1f}%)")

    print("\n--- Full listing ---")
    for (sess, q), v in sorted(defects.items()):
        r = v["row"]
        print(
            f"{sess} Q{int(q):>2} | {r['difficulty']:<6} | {r['strategy']:<9} | "
            f"{r['source']:<6} | {r['cognitive']:<13} | {', '.join(v['types'])}"
        )
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer']}")


# ---------------------------------------------------------------------------
# 4. Representative questions for qualitative analysis
# ---------------------------------------------------------------------------
def qualitative_examples():
    rows = {(r["session"], r["q_num"]): r for r in load_questions()}

    targets = [
        ("S7",  "15", "recall — well-formed"),
        ("S10", "2",  "recall — well-formed"),
        ("S6",  "2",  "application — conditioning hit"),
        ("S13", "15", "analysis — conditioning hit"),
        ("S5",  "12", "analysis — conditioning fail"),
        ("S12", "1",  "comprehension — conditioning fail"),
    ]

    print("\n=== Qualitative Analysis Examples ===")
    for (s, q, label) in targets:
        r = rows[(s, q)]
        print(f"\n--- {s} Q{q} [{label}] ---")
        print(f"  Source / Strategy / Difficulty / Cognitive: "
              f"{r['source']} / {r['strategy']} / {r['difficulty']} / {r['cognitive']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer']}")
        print(f"  clarity={r['clarity']}  correct={r['correctness']}  "
              f"distractor={r['distractor']}  cog_match={r['cog_match']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    session_summary()
    cog_level_table()
    error_analysis()
    qualitative_examples()
