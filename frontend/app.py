"""
AutoQuiz web interface.

Routes:
  GET  /                          — upload + config form
  POST /generate                  — run pipeline, create session
  GET  /quiz/{sid}                — current question
  POST /quiz/{sid}/answer         — submit answer, redirect to review
  GET  /quiz/{sid}/review         — show correct answer + explanation
  POST /quiz/{sid}/next           — advance to next question or results
  GET  /results/{sid}             — final score + feedback form
  POST /results/{sid}/feedback    — save feedback, log session to disk
"""

import json
import os
import shutil
import tempfile
import time
import uuid

import jinja2 as _jinja2

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.preprocessing.pipeline import preprocess
from backend.generation import generate_all
from backend.validation import filter_all
from backend.explanation import generate_and_attach

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="AutoQuiz")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Use Jinja2 directly (bypasses Starlette's wrapper which breaks on Python 3.14)
_env = _jinja2.Environment(
    loader=_jinja2.FileSystemLoader(TEMPLATES_DIR),
    autoescape=True,
    cache_size=0,  # disable cache — avoids Python 3.14 hash bug
)
_env.globals["zip"] = zip


def render(name: str, ctx: dict) -> HTMLResponse:
    template = _env.get_template(name)
    return HTMLResponse(template.render(**ctx))

# In-memory session store
_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Upload + generate
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render("index.html", {})


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(...),
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
        all_candidates = generate_all(chunks, difficulty=difficulty, cognitive_level=cognitive, n=candidates)
        validated = filter_all(all_candidates, check_answerability=check_answerability, check_cognitive=check_cognitive, max_questions=max_questions)

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
            "strategy": strategy,
            "difficulty": difficulty,
            "cognitive": cognitive,
            "filename": file.filename,
        },
        "feedback": {},
        "created_at": time.time(),
    }

    _log_session(sid, _sessions[sid])
    return RedirectResponse(f"/quiz/{sid}", status_code=303)


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Results + feedback
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
