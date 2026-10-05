# Event Analysis Implementation Plan

## Summary

Implement a deterministic, independently testable CV layer that converts
calibrated ball-tracking observations into timestamped events. The first
increment will cover ball lost/reacquired transitions and goal-line crossings.
Ball proximity and sustained-control events will follow once rod/player
position tracking exposes a stable input contract. Exercise decisions remain
the responsibility of validators, not this package.

This plan follows the event-analysis boundary in [PLAN.md](PLAN.md),
[FLOW.md](FLOW.md), [DETECTION_PLAN.md](DETECTION_PLAN.md), and
[BALL_TRACKING_PLAN.md](BALL_TRACKING_PLAN.md). `app/ball_tracking/` already
provides immutable JSON-safe observations and confidence; `app/detection/`
provides calibrated field geometry and rods. No event-analysis package or
event-specific tests exist yet.

## Scope and Requirements

- [ ] **REQ-EVT-001 — Module boundary:** Event analysis accepts serializable
  tracking/calibration contracts and has no FastAPI, Celery, SQLAlchemy,
  database, or storage dependency.
- [ ] **REQ-EVT-002 — Event evidence:** Every emitted event records its type,
  source frame/timestamp evidence, relevant canonical coordinates, confidence,
  and diagnostics, and serializes to plain JSON-safe values.
- [ ] **REQ-EVT-003 — Ball transitions:** Detect ball-lost and ball-reacquired
  intervals from explicit `BallObservation` states. Preserve missing and
  uncertain evidence; do not interpolate measurements.
- [ ] **REQ-EVT-004 — Goal crossings:** Detect a ball crossing a configured goal
  mouth/line from consecutive supported observations, with side, direction,
  timing, confidence, and source frames. Do not treat every field exit as a
  goal.
- [ ] **REQ-EVT-005 — Player proximity:** When a separately produced player
  position track is available, derive ball-entered/left-proximity events and
  elapsed duration for the target player. This is an event signal only, not
  an exercise pass/fail result.
- [ ] **REQ-EVT-006 — Safe uncertainty:** Missing prerequisites, unsupported
  geometry, low-quality track evidence, and ambiguous transitions must be
  reported explicitly. They must not become confident negative exercise
  outcomes.
- [ ] **REQ-EVT-007 — Reproducibility:** Event thresholds and geometry assumptions
  are held in an immutable, versioned configuration and covered by synthetic
  unit tests and human-annotated supported-video regressions before tuning.

## Technical Context and Boundaries

- Language and runtime: Python, following the existing application packages.
- Inputs: `BallTrack` observations from `app/ball_tracking/contracts.py` and
  `FieldGeometry`/`TableCalibration` from `app/detection/contracts/`.
- Output: immutable event/result contracts with `to_dict()` methods; no
  OpenCV/NumPy values or frame images in persisted output.
- Proposed package:

  ```text
  app/event_analysis/
  ├── __init__.py
  ├── config.py
  ├── contracts.py
  └── analyzer.py
  ```
- Keep event analysis separate from static calibration, ball detection and
  tracking, exercise validation, and submission lifecycle management.
- Supported environment remains the calibrated Bonzini table and documented
  diagonal top-down camera. Do not imply support for other table/camera setups.
- Initial analysis is deterministic and inspectable. It does not infer
  unobserved ball positions or classify a player's exercise as successful.

## Decisions and Gating Questions

1. **Goal geometry and direction:** `FieldGeometry` identifies the playable
   field but does not define the opening/goal-mouth dimensions or which side is
   the player's attacking side. Before accepting goal-event thresholds, confirm
   the goal-mouth geometry and orientation convention against the supported
   Bonzini setup. Until then, goal events must be marked unavailable rather
   than approximated as any field-boundary exit.
2. **Player identity and proximity:** Existing `Rod` values describe rod lines,
   not individual player positions. Before implementing REQ-EVT-005, agree on
   the upstream player-position contract and what “under the player” means
   (figure bounds versus a calibrated distance). Keep this work out of the
   event package's detection responsibilities.
3. **Real event fixtures:** Obtain supported clips with manually reviewed event
   annotations before changing any CV threshold. Synthetic tracks can validate
   deterministic temporal logic, but cannot certify real-world event accuracy.

## Implementation Steps

### REQ-EVT-001 acceptance criteria

1. `app.event_analysis.EventAnalysisInput` takes the existing `BallTrack` and
   `TableCalibration` contracts directly, without copying CV data into
   infrastructure-specific models.
2. The input contract serializes using only the upstream contracts'
   `to_dict()` output and passes a JSON encode/decode round trip.
3. Importing `app.event_analysis` does not load FastAPI, Celery, SQLAlchemy,
   database, or storage modules; source imports are checked to prevent those
   dependencies from entering the package.
4. No event detection or exercise decision logic is added as part of this
   boundary-only increment.

### Step 1: Define event contracts and configuration

- **Requirements:** REQ-EVT-001, REQ-EVT-002, REQ-EVT-006, REQ-EVT-007
- Add `EventType`, immutable `Event`, and `EventAnalysisResult` contracts in
  `app/event_analysis/contracts.py`. Keep source frame indices, timestamps,
  evidence coordinates, confidence, diagnostics, warnings, and configuration
  version explicit.
- Add a frozen `EventAnalysisConfig` in `app/event_analysis/config.py`.
  Include versioned transition-gap, event-confidence, goal geometry, and
  proximity thresholds; do not place thresholds inline in analyzer logic.
- Make unsupported/missing inputs distinguishable from a valid analysis with
  zero events.

### Step 2: Analyze track-state transitions

- **Requirements:** REQ-EVT-002, REQ-EVT-003, REQ-EVT-006
- Implement `app/event_analysis/analyzer.py` to identify ball-lost intervals
  and subsequent reacquisition from the track's `DETECTED`, `MISSED`, and
  `UNCERTAIN` observations.
- Preserve original frame indices and timestamps. Do not fill gaps or turn
  `UNCERTAIN` into a measured position.
- Emit diagnostics when a gap is unresolved or confidence is insufficient to
  support an event boundary.

### Step 3: Add goal-crossing detection behind a geometry gate

- **Requirements:** REQ-EVT-002, REQ-EVT-004, REQ-EVT-006, REQ-EVT-007
- After the goal-mouth and orientation convention are confirmed, detect
  crossings in canonical field coordinates using consecutive supported
  observations.
- Require evidence to cross the configured mouth/line in the correct order;
  suppress duplicates from jitter/reversal and refuse to bridge unsupported
  observation gaps.
- Record the crossed side and direction without deciding whether it counts
  toward an exercise.

### Step 4: Add player-proximity events when upstream tracking is ready

- **Requirements:** REQ-EVT-002, REQ-EVT-005, REQ-EVT-006, REQ-EVT-007
- Consume a separately defined player-position track for the target player;
  do not infer figure position from the existing rod-line-only calibration
  contract.
- Emit proximity-entered and proximity-left events with their duration and
  supporting evidence. Keep the distance/stillness rule configurable and
  versioned.
- If player tracking is absent or ambiguous, report that this event family is
  unavailable/uncertain; never represent it as “no control event occurred.”

### Step 5: Integrate only after standalone validation

- **Requirements:** REQ-EVT-001, REQ-EVT-006
- Once the standalone contracts and regression suite are stable, add a thin
  call from `app/worker/tasks.py` after successful calibration and ball
  tracking. Persist only the JSON-safe event report in the existing metrics
  structure.
- Keep submissions pending review; event analysis must not approve/reject a
  submission or alter worker/API lifecycle policy.

## Task Breakdown

### Phase 1 — Foundational contracts and ball-state events

- [X] T001 [Plan:1.1] Create the `app/event_analysis/` package and
  `EventAnalysisInput` contract taking `BallTrack` and `TableCalibration`
  directly; export it from `app/event_analysis/__init__.py`.
- [X] T001a [Plan:1.1] Test input JSON serialization and enforce the
  infrastructure import boundary in `tests/unit/test_event_analysis_boundary.py`.
- [ ] T001b [Plan:1.1] Add event types and JSON-safe `Event` and
  `EventAnalysisResult` contracts in `app/event_analysis/contracts.py`.
- [ ] T002 [Plan:1.1] Add immutable, versioned event thresholds and validation
  in `app/event_analysis/config.py`.
- [ ] T003 [Plan:2.1] Implement deterministic lost/reacquired event extraction
  in `app/event_analysis/analyzer.py`.
- [ ] T004 [Plan:1.1,2.1] Add serialization, ordering, timestamp, missing-frame,
  uncertain-frame, and low-confidence tests in
  `tests/unit/test_event_analysis_contracts.py` and
  `tests/unit/test_event_analyzer.py`.

### Phase 2 — Goal-line crossing events

- [ ] T005 [Plan:3.1] Confirm supported goal-mouth geometry and side/direction
  conventions; record them as versioned configuration inputs.
- [ ] T006 [Plan:3.1] Add goal-crossing events, duplicate suppression, and
  gap-safe evidence handling to `app/event_analysis/analyzer.py`.
- [ ] T007 [Plan:3.1] Add synthetic goal-crossing tests and human-annotated
  Bonzini event regression cases under `tests/fixtures/expected/` and
  `tests/integration/`.

### Phase 3 — Player-proximity event signals

- [ ] T008 [Plan:4.1] Define/consume the upstream player-position contract
  without coupling event analysis to CV, API, or worker implementations.
- [ ] T009 [Plan:4.1] Implement proximity-entered/left events and elapsed
  duration using versioned thresholds in `app/event_analysis/`.
- [ ] T010 [Plan:4.1] Add synthetic and manually annotated proximity fixtures,
  including player ambiguity and ball-observation gaps.

### Phase 4 — Worker integration

- [ ] T011 [Plan:5.1] Persist the JSON-safe event-analysis report from
  `app/worker/tasks.py` after the standalone event suite passes.
- [ ] T012 [Plan:5.1] Extend `tests/unit/test_worker_tasks.py` to verify event
  reports are stored, analysis is skipped after review-required calibration or
  tracking, and no event outcome approves/rejects a submission.

## Testing and Acceptance

- Contract tests prove output round-trips through `json.dumps`/`json.loads`
  without OpenCV or NumPy values.
- Synthetic unit tests cover monotonic event ordering, source-frame/timestamp
  retention, gap boundaries, lost/reacquired transitions, goal-mouth crossing,
  direction, boundary jitter/reversal, and confidence/diagnostic propagation.
- Negative cases cover no observations, missed/uncertain intervals, low
  confidence, unsupported/missing goal geometry, ambiguous player position,
  and unobserved gaps at event boundaries.
- Fixture-based regressions use manually reviewed expected event annotations;
  never generate expected annotations from the analyzer itself. Do not tune
  thresholds until the corresponding supported-video fixture exists.
- CV modules remain independent of FastAPI, Celery, SQLAlchemy, and storage;
  worker integration is tested separately from analyzer behavior.
- Run focused event-analysis tests first, then the repository's full `pytest`
  suite after implementation.

**Completion criteria:** REQ-EVT-001 through REQ-EVT-004 have standalone
contracts, deterministic behavior, diagnostics, and focused regression
coverage. REQ-EVT-005 remains explicitly gated until player-position tracking
and its semantics are available. Worker integration is complete only when it
preserves the pending-review safety behavior.

## Requirement Mapping

| Requirement | Plan items              | Implementation evidence                                                    |
| ----------- | ----------------------- | -------------------------------------------------------------------------- |
| REQ-EVT-001 | 1.1, 5.1                | `app/event_analysis/`; `app/worker/tasks.py`                           |
| REQ-EVT-002 | 1.1, 2.1, 3.1, 4.1      | `app/event_analysis/contracts.py`; `app/event_analysis/analyzer.py`    |
| REQ-EVT-003 | 2.1                     | `app/event_analysis/analyzer.py`; `tests/unit/test_event_analyzer.py`  |
| REQ-EVT-004 | 3.1                     | `app/event_analysis/analyzer.py`; goal-crossing regression fixtures      |
| REQ-EVT-005 | 4.1                     | `app/event_analysis/analyzer.py`; player-proximity regression fixtures   |
| REQ-EVT-006 | 1.1, 2.1, 3.1, 4.1, 5.1 | event diagnostics and worker review-safe integration tests                 |
| REQ-EVT-007 | 1.1, 3.1, 4.1           | `app/event_analysis/config.py`; synthetic and annotated regression tests |
