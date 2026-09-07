from fastapi import FastAPI

from app.api import detection, exercises, submissions
from app.database import Base, engine

app = FastAPI(title="Foosball Academy AI")

Base.metadata.create_all(bind=engine)

app.include_router(exercises.router)
app.include_router(submissions.router)
app.include_router(detection.router)


@app.get("/health")
def health():
    return {"status": "ok"}
