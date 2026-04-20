"""
Question dataclass — the shared data structure used across generation, validation, and explanation.
"""

from dataclasses import dataclass, field


@dataclass
class Question:
    question: str
    options: dict[str, str]      # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str                  # "A", "B", "C", or "D"
    difficulty: str              # "easy" | "medium" | "hard"
    cognitive_level: str         # "recall" | "comprehension" | "application" | "analysis"
    source_chunk: str
    explanation: str = ""        # filled in by the explanation module
    similarity_score: float = 0.0  # set during validation — proxy for source grounding

    def correct_answer_text(self) -> str:
        return self.options.get(self.answer, "")
