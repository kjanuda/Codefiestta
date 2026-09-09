from pathlib import Path

from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from backend.app.api.ask import (
    router as ask_router,
)


ROOT_DIR = Path(
    __file__
).resolve().parents[2]

CORPUS_DIR = (
    ROOT_DIR
    / "corpus"
    / "Ashen_Era_Archive"
)

SAMPLE_QUESTIONS_PATH = (
    CORPUS_DIR
    / "sample_questions.json"
)


app = FastAPI(
    title="AshenLens API",
    description=(
        "Multimodal document assistant "
        "for the Ashen Era Archive."
    ),
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------
# Archive visual evidence
# -----------------------------------------

if CORPUS_DIR.exists():

    app.mount(
        "/archive",
        StaticFiles(
            directory=str(
                CORPUS_DIR
            )
        ),
        name="archive",
    )


# -----------------------------------------
# API routes
# -----------------------------------------

app.include_router(
    ask_router,
    prefix="/api",
    tags=[
        "Multimodal QA"
    ],
)


@app.get("/")
def root():

    return {
        "name":
            "AshenLens",

        "status":
            "running",
    }


@app.get("/api/health")
def health():

    return {
        "status":
            "ok",

        "corpus_exists":
            CORPUS_DIR.exists(),

        "sample_questions_exists":
            SAMPLE_QUESTIONS_PATH.exists(),
    }