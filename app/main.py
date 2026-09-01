from fastapi import FastAPI

from app.api import exercises, submissions
from app.database import Base, engine

app = FastAPI(title="Foosball Academy AI")

Base.metadata.create_all(bind=engine)

app.include_router(exercises.router)
app.include_router(submissions.router)


@app.get("/health")
def health():
    return {"status": "ok"}
