* [ ] 

# Foosball Academy AI

Exercise-validation service: players submit short training videos, the
backend analyzes them and either approves/rejects automatically or routes
them to a coach for review. See [docs/PLAN.md](docs/PLAN.md) for the full
architecture and [docs/FLOW.md](docs/FLOW.md) for the submission flow.

## Repository structure

```text
.
├── app/
│   ├── main.py           # FastAPI app
│   ├── config.py         # settings (env-driven)
│   ├── database.py       # SQLAlchemy engine/session
│   ├── models.py         # User, Exercise, Submission
│   ├── schemas.py        # Pydantic request/response models
│   ├── storage.py        # video storage (local disk for now)
│   ├── api/               # exercises.py, submissions.py routers
│   └── worker/            # celery_app.py, tasks.py
├── tests/
├── docs/
├── docker-compose.yml    # postgres + redis + api + worker
├── Dockerfile
└── requirements.txt
```

The CV pipeline (`field_detector`, `ball_tracker`, `rod_tracker`,
`event_analyzer`, `validators/`) isn't implemented yet — `app/worker/tasks.py`
currently only manages the submission status lifecycle as a placeholder.

## Local setup

### 1) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

### 3) Start Postgres + Redis

```bash
docker compose up -d postgres redis
```

### 4) Run the API

```bash
uvicorn app.main:app --reload
```

### 5) Run the worker (separate terminal)

```bash
celery -A app.worker.celery_app worker --loglevel=info
```

## Run everything with Docker Compose

```bash
docker compose up --build
```

## Tests

```bash
pytest
```
