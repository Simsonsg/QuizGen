"""
AutoQuiz web interface.
"""

import json
import os
import random
import shutil
import tempfile
import time
import uuid

import jinja2 as _jinja2

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.preprocessing.pipeline import preprocess
from backend.generation import generate_candidates, generate_all_baseline
from backend.validation import filter_candidates
from backend.validation.scorer import question_chunk_similarity
from backend.explanation import generate_and_attach

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="AutoQuiz")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


_env = _jinja2.Environment(
    loader=_jinja2.FileSystemLoader(TEMPLATES_DIR),
    autoescape=True,
    cache_size=0,  # disable cache 
)
_env.globals["zip"] = zip


def render(name: str, ctx: dict) -> HTMLResponse:
    template = _env.get_template(name)
    return HTMLResponse(template.render(**ctx))

# In-memory session store
_sessions: dict[str, dict] = {}




@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render("index.html", {})


@app.post("/")
async def index_post():
    return RedirectResponse("/", status_code=303)


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("pipeline"),
    strategy: str = Form("clean"),
    difficulty: str = Form("medium"),
    cognitive: str = Form("recall"),
    candidates: int = Form(3),
    max_questions: int = Form(10),
    check_answerability: bool = Form(False),
    check_cognitive: bool = Form(False),
):
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Derive max_chunks: need enough chunks to plausibly hit max_questions
    max_chunks = max(max_questions, 15)

    try:
        chunks = preprocess(tmp_path, strategy=strategy, max_chunks=max_chunks)

        total_candidates = 0
        if mode == "baseline":
            print(f"\n{'─' * 60}")
            print(f"  BASELINE MODE  |  single-pass, no conditioning, no filtering")
            print(f"{'─' * 60}\n")
            validated = generate_all_baseline(chunks, max_questions=max_questions)
            total_candidates = len(validated)
            if not validated:
                return render("index.html", {"error": "Baseline generation produced no questions. Try a different file."})
            for q in validated:
                q.similarity_score = round(question_chunk_similarity(q.question, q.source_chunk), 4)
        else:
            # Generate and validate chunk-by-chunk — stop as soon as we have
            # enough questions. Shuffle so coverage is spread across the
            # document rather than front-loaded.
            shuffled = chunks[:]
            random.shuffle(shuffled)
            total_chunks = len(shuffled)
            print(f"\n{'─' * 60}")
            print(f"  PIPELINE  |  {total_chunks} chunks  |  {candidates} candidates/chunk  |  target {max_questions} questions")
            print(f"{'─' * 60}")
            validated = []
            for chunk_i, chunk in enumerate(shuffled, 1):
                if len(validated) >= max_questions:
                    break
                print(f"\n  Chunk {chunk_i}/{total_chunks}")
                cands = generate_candidates(chunk, difficulty=difficulty, cognitive_level=cognitive, n=candidates)
                total_candidates += len(cands)
                passing = filter_candidates(cands, check_answerability=check_answerability, check_cognitive=check_cognitive)
                validated.extend(q for q, _ in passing)
            validated = validated[:max_questions]
            print(f"\n{'─' * 60}")
            print(f"  {len(validated)} questions validated from {total_candidates} candidates ({chunk_i} chunks processed)")
            print(f"{'─' * 60}\n")
            if not validated:
                return render("index.html", {"error": "No questions passed validation. Try a different file or lower the similarity threshold."})

        generate_and_attach(validated)
    finally:
        os.unlink(tmp_path)

    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "questions": [_question_to_dict(q) for q in validated],
        "answers": [None] * len(validated),
        "current": 0,
        "config": {
            "mode": mode,
            "strategy": strategy,
            "difficulty": difficulty if mode == "pipeline" else "unspecified",
            "cognitive": cognitive if mode == "pipeline" else "unspecified",
            "filename": file.filename,
            "candidates_per_chunk": candidates,
            "total_candidates_generated": total_candidates,
            "questions_passed_validation": len(validated),
            "validation_pass_rate": round(len(validated) / total_candidates, 4) if total_candidates else 0,
        },
        "feedback": {},
        "created_at": time.time(),
    }

    _log_session(sid, _sessions[sid])
    return RedirectResponse(f"/quiz/{sid}", status_code=303)



# Quiz


@app.get("/quiz/{sid}", response_class=HTMLResponse)
async def quiz(request: Request, sid: str):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    idx = session["current"]
    if idx >= len(session["questions"]):
        return RedirectResponse(f"/results/{sid}")

    q = session["questions"][idx]
    return render("quiz.html", {
        "sid": sid,
        "q": q,
        "idx": idx,
        "total": len(session["questions"]),
        "prev_answer": session["answers"][idx],
    })


@app.post("/quiz/{sid}/answer")
async def answer(sid: str, choice: str = Form(...)):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    idx = session["current"]
    session["answers"][idx] = choice
    return RedirectResponse(f"/quiz/{sid}/review", status_code=303)


@app.post("/quiz/{sid}/skip")
async def skip(sid: str):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    session["current"] += 1
    if session["current"] >= len(session["questions"]):
        return RedirectResponse(f"/results/{sid}", status_code=303)
    return RedirectResponse(f"/quiz/{sid}", status_code=303)


@app.post("/quiz/{sid}/prev")
async def prev_question(sid: str):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    if session["current"] > 0:
        session["current"] -= 1
    return RedirectResponse(f"/quiz/{sid}", status_code=303)


@app.get("/quiz/{sid}/review", response_class=HTMLResponse)
async def review(request: Request, sid: str):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    idx = session["current"]
    q = session["questions"][idx]
    user_answer = session["answers"][idx]
    correct = user_answer == q["answer"]

    return render("review.html", {
        "sid": sid,
        "q": q,
        "user_answer": user_answer,
        "correct": correct,
        "idx": idx,
        "total": len(session["questions"]),
        "is_last": idx == len(session["questions"]) - 1,
    })


@app.post("/quiz/{sid}/next")
async def next_question(sid: str):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    session["current"] += 1
    if session["current"] >= len(session["questions"]):
        return RedirectResponse(f"/results/{sid}", status_code=303)
    return RedirectResponse(f"/quiz/{sid}", status_code=303)



# Results + feedback


@app.get("/results/{sid}", response_class=HTMLResponse)
async def results(request: Request, sid: str):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    questions = session["questions"]
    answers = session["answers"]
    score = sum(1 for q, a in zip(questions, answers) if a == q["answer"])
    avg_similarity = round(sum(q["similarity_score"] for q in questions) / len(questions), 4) if questions else 0

    return render("results.html", {
        "sid": sid,
        "questions": questions,
        "answers": answers,
        "avg_similarity": avg_similarity,
        "score": score,
        "total": len(questions),
        "config": session["config"],
        "feedback_saved": session["feedback"].get("saved", False),
    })


@app.post("/results/{sid}/feedback")
async def feedback(
    sid: str,
    perceived_difficulty: str = Form(...),
    clarity_rating: int = Form(...),
    comments: str = Form(""),
):
    session = _get_session(sid)
    if session is None:
        return RedirectResponse("/")

    session["feedback"] = {
        "perceived_difficulty": perceived_difficulty,
        "clarity_rating": clarity_rating,
        "comments": comments,
        "saved": True,
    }

    _log_session(sid, session)
    return RedirectResponse(f"/results/{sid}", status_code=303)



# Helpers


def _get_session(sid: str) -> dict | None:
    return _sessions.get(sid)


def _question_to_dict(q) -> dict:
    return {
        "question": q.question,
        "options": q.options,
        "answer": q.answer,
        "difficulty": q.difficulty,
        "cognitive_level": q.cognitive_level,
        "explanation": q.explanation,
        "similarity_score": round(q.similarity_score, 4),
    }


def _log_session(sid: str, session: dict):
    path = os.path.join(OUTPUT_DIR, f"session_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
