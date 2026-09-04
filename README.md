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

## Tools

#### Table-detection annotations

These are development and regression-test tools. They are not used during
normal user video processing.

Run the interactive annotation tool from the repository root after activating
the virtual environment:

* [ ]
  ```bash
  python tools/annotate_table.py \
  	tests/table-detection_tests/table-detection_test-1.mp4 \
  	tests/fixtures/expected/table-detection_test-1.json
  ```

Left-click the four ordered field corners (`top_left`, `top_right`,
`bottom_right`, `bottom_left`), then click two endpoints for each rod. Right
click undoes the most recent input. Press `s` to save the JSON fixture and its
annotated PNG preview, or `q` to quit without saving.

To annotate a different startup frame, pass its zero-based frame index:

```bash
python tools/annotate_table.py \
	tests/table-detection_tests/table-detection_test-1.mp4 \
	tests/fixtures/expected/table-detection_test-1-frame-30.json \
	--frame-index 30
```

The default project dependency is `opencv-python-headless`, which is suitable
for the service but cannot open an interactive window. Run the interactive
command on a desktop session with a display and install the GUI build first:

* [ ] 
  ```bash
  pip install opencv-python
  ```

On a headless machine, export a frame for annotation in an external image
viewer instead:

```bash
python tools/annotate_table.py \
	tests/table-detection_tests/table-detection_test-1.mp4 \
	tests/fixtures/expected/table-detection_test-1.json \
	--export-frame /tmp/table-frame.png
```

The renderer is called automatically by the annotation tool. Run its focused
tests directly with:

```bash
pytest -q tests/unit/test_debug_renderer.py
```

Run the startup-frame reader tests with:

```bash
pytest -q tests/unit/test_startup_frames.py
```
