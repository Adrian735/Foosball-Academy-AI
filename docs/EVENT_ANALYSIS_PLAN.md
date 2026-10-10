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

- [x] **REQ-EVT-001 — Module boundary:** Event analysis accepts serializable
  tracking/calibration contracts and has no FastAPI, Celery, SQLAlchemy,
  database, or storage dependency.
- [x] **REQ-EVT-002 — Event evidence:** Every emitted event records its type,
  source frame/timestamp evidence, relevant canonical coordinates, confidence,
  and diagnostics, and serializes to plain JSON-safe values.
- [x] **REQ-EVT-003 — Ball transitions:** Detect ball-lost and ball-reacquired
  intervals from explicit `BallObservation` states. Preserve missing and
  uncertain evidence; do not interpolate measurements.
- [ ] **REQ-EVT-003 acceptance criteria:**
  1. One `ball_lost` event is emitted at the first non-detected observation
     after a detected ball; repeated missed/uncertain frames do not duplicate
     the transition.
  2. A `ball_reacquired` event is emitted on the next detected observation and
     retains the prior detection, every missed/uncertain observation, and the
     reacquired observation with original frame indices/timestamps/states.
  3. No positions are synthesized for missed/uncertain observations.
  4. Leading unobserved frames do not assert a loss transition; trailing
     unresolved loss is represented by a result warning and no reacquired event.
  5. Existing track warnings and conservative confidence are retained; invalid
     observation ordering is rejected explicitly.
- [ ] **REQ-EVT-004 — Goal crossings:** Derive B90 goal-mouth geometry from
  static field calibration, extend ball tracking into estimated goal mouths,
  and identify a ball crossing a mouth between consecutive supported
  observations. Include side, direction, source frames/timestamps, confidence,
  and diagnostics; do not treat every field exit as a goal.
- [x] **REQ-EVT-005 — Player proximity:** When a separately produced player
  center track is available, derive entered/left events using the versioned
  B90-scaled physical distance threshold and elapsed duration. This is an
  event signal only, not an exercise pass/fail result.
- [ ] **REQ-EVT-006 — Safe uncertainty:** Missing prerequisites, unsupported
  geometry, low-quality track evidence, and ambiguous transitions must be
  reported explicitly. They must not become confident negative exercise
  outcomes.
- [ ] **REQ-EVT-007 — Reproducibility:** Event thresholds and geometry assumptions
  are held in an immutable, versioned configuration and covered by synthetic
  unit tests and human-annotated supported-video regressions before tuning.
- [ ] **REQ-EVT-002 acceptance criteria:**
  1. `EventEvidence`, `Event`, and `EventAnalysisResult` are immutable
     dataclasses; `EventType` is a string enum.
  2. Evidence retains source frame index and timestamp, with optional
     canonical ball and player coordinates when available.
  3. Events include type, evidence, unit-interval confidence, and diagnostics;
     results include ordered events, config version, and warnings.
  4. Serialization returns only native JSON-safe values and round-trips
     through `json.dumps`/`json.loads`.
  5. Invalid confidence, empty evidence, non-finite timestamps or coordinates,
     negative timestamps, and out-of-order evidence are rejected explicitly.

## Technical Context and Boundaries

- Language and runtime: Python, following the existing application packages.
- Inputs: `BallTrack` observations from `app/ball_tracking/contracts.py`,
  optional measured `PlayerPositionTrack` observations, and
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

1. **Goal geometry and direction:** Derive B90 goal bounds from the accepted
   field geometry and versioned B90 defaults. Label ends in canonical table
   coordinates, not player/team orientation; event analysis reports each end
   and direction without assigning an attack side.
2. **Player identity and proximity:** Existing `Rod` values describe rod lines,
   not individual player positions. REQ-EVT-005 consumes an externally
   measured player-center track in canonical coordinates; proximity uses
   physical Euclidean distance with a versioned 75 mm radius. Match ball and
   player observations by source frame only and never infer positions across
   gaps. This work does not detect player locations.
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

### REQ-EVT-004 acceptance criteria

1. Static calibration derives both goal mouths from detected field geometry
   and the supported Bonzini B90 defaults: a 1200 × 700 mm playfield, centered
   200 mm opening, and 150 mm mouth depth beyond each short end.
2. Goal geometry diagnostics clearly mark the mouth bounds as B90-derived
   estimates; field-calibration confidence remains attached to the estimate.
3. Ball detection/tracking accepts measurements inside configured goal mouths
   as well as the playable field; unsupported out-of-field regions remain
   masked.
4. A goal event requires a segment between consecutive, detected, sourced ball
   observations that intersects a calibrated aperture. Missed, uncertain, or
   non-adjacent frames cannot be bridged.
5. Event payload includes canonical end (`start` or `end`), inward/outward
   direction, source evidence, and confidence; duplicate jitter/reversal does
   not report repeated goals.
6. Synthetic field-to-goal geometry and ball-crossing tests pass; expected
   geometry is calculated from independently specified B90 dimensions.

### REQ-EVT-005 acceptance criteria

1. Event analysis accepts an immutable JSON-safe `PlayerPositionTrack` with
   a stable player identifier, source frame/timestamp, measured canonical
   center, state, confidence, and diagnostics.
2. Ball-to-player distance is computed as a physical Euclidean distance in
   millimetres using the supported B90 field dimensions; the versioned default
   proximity radius is 75 mm.
3. Entered/left events require measured ball and player positions from the
   same adjacent source frames. Missing, uncertain, low-confidence, unmatched,
   or non-adjacent evidence is never interpolated or bridged.
4. Events retain both measured centers, the player identifier, confidence,
   distance/radius details, and elapsed proximity duration on the leave event.
5. Missing or incomplete player-position prerequisites produce explicit
   warnings; event analysis never turns them into a negative exercise result.

### Step 3: Detect goal apertures and crossings

- **Requirements:** REQ-EVT-002, REQ-EVT-004, REQ-EVT-006, REQ-EVT-007
- Add a separate B90 goal-geometry estimator under `app/detection/` that
  scales the standard goal width/depth through detected field coordinates.
- Include estimated bounds in `TableCalibration`, preserving field confidence
  and explicit model-derived diagnostics. Do not change field/rod confidence.
- Extend the ball detector's allowed mask with only the calibrated aperture
  regions and retain canonical positions beyond the field polygon.
- Require a direct segment between adjacent detected observations to cross a
  detected mouth. Suppress duplicate crossings and report end/direction.
- Never infer a goal through missing or uncertain ball observations.

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
- Use a distinct immutable contract for externally measured player centers.
  The event analyzer consumes this contract but does not infer player
  positions from rods or image pixels.
- Compare canonical points after scaling each axis to B90 millimetres and use
  the configured 75 mm radius. Require same-frame evidence and adjacent frame
  transitions; gaps make an interval unresolved rather than interpolated.

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
- [x] T001b [Plan:1.1] Add `EventType`, immutable `EventEvidence`, `Event`,
  and `EventAnalysisResult` contracts in
  `app/event_analysis/contracts.py`.
- [x] T001c [Plan:1.1] Test event evidence, confidence and ordering
  validation, immutability, and JSON serialization in
  `tests/unit/test_event_analysis_contracts.py`.
- [ ] T002 [Plan:1.1] Add immutable, versioned event thresholds and validation
  in `app/event_analysis/config.py`.
- [x] T003a [Plan:2.1] Add transition, gap, unresolved, warning, confidence,
  and evidence-preservation tests in `tests/unit/test_event_analyzer.py`.
- [x] T003 [Plan:2.1] Implement deterministic lost/reacquired event extraction
  in `app/event_analysis/analyzer.py`.
- [x] T004 [Plan:1.1,2.1] Add serialization, ordering, timestamp, missing-frame,
  uncertain-frame, and low-confidence tests in
  `tests/unit/test_event_analysis_contracts.py` and
  `tests/unit/test_event_analyzer.py`.

### Phase 2 — Goal-line crossing events

- [x] T005 [Plan:3.1] Add immutable, serializable canonical `GoalMouth`
  geometry and separate diagnostics to
  `app/detection/contracts/table_contracts.py`.
- [x] T006 [Plan:3.1] Estimate both B90 goal mouths directly from the accepted
  field calibration in `app/detection/goal_geometry.py`.
- [x] T007 [Plan:3.1] Integrate detected goal geometry into both table
  calibration entry points and debug renderers without changing field/rod
  review gates.
- [x] T008 [Plan:3.1] Permit yellow-ball tracking only inside calibrated
  apertures in `app/ball_tracking/detector.py` and propagate geometry through
  `tracker.py`, `runner.py`, API, and worker callers.
- [x] T009 [Plan:3.1] Add goal crossings, explicit end/direction event data,
  adjacency and gap checks, and duplicate suppression in
  `app/event_analysis/analyzer.py`.
- [ ] T010 [Plan:3.1] Add synthetic geometry and trajectory tests plus manually
  reviewed Bonzini goal annotations under `tests/fixtures/expected/`.

REQ-EVT-004 remains gated on T010's manually reviewed real-video goal
annotations. Field-derived B90 geometry and crossing behavior have synthetic
coverage, but real-world accuracy claims still require that fixture.

### Phase 3 — Player-proximity event signals

- [x] T008 [Plan:4.1] Define/consume the upstream player-position contract
  without coupling event analysis to CV, API, or worker implementations.
- [x] T009 [Plan:4.1] Implement proximity-entered/left events and elapsed
  duration using versioned thresholds in `app/event_analysis/`.
- [x] T010a [Plan:4.1] Add synthetic proximity tests for transitions,
  confidence gating, unresolved intervals, and ball-observation gaps.
- [ ] T010b [Plan:4.1] Add manually annotated real-video proximity fixtures
  before tuning the 75 mm threshold.

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

**Completion criteria:** REQ-EVT-001 through REQ-EVT-005 have standalone
contracts, deterministic behavior, diagnostics, and focused synthetic
coverage. REQ-EVT-005 consumes externally measured player positions; visual
player tracking and real-video proximity annotation remain separate gates.
Worker integration is complete only when it preserves the pending-review
safety behavior.

## Requirement Mapping

| Requirement | Plan items              | Implementation evidence                                                    |
| ----------- | ----------------------- | -------------------------------------------------------------------------- |
| REQ-EVT-001 | 1.1, 5.1                | `app/event_analysis/`; `app/worker/tasks.py`                           |
| REQ-EVT-002 | 1.1, 2.1, 3.1, 4.1      | `app/event_analysis/contracts.py`; `app/event_analysis/analyzer.py`    |
| REQ-EVT-003 | 2.1                     | `app/event_analysis/analyzer.py`; `tests/unit/test_event_analyzer.py`  |
| REQ-EVT-004 | 3.1                     | `app/event_analysis/analyzer.py`; goal-crossing regression fixtures      |
| REQ-EVT-005 | 4.1                     | `app/event_analysis/contracts.py`, `config.py`, `analyzer.py`; `tests/unit/test_player_proximity.py` |
| REQ-EVT-006 | 1.1, 2.1, 3.1, 4.1, 5.1 | event diagnostics and worker review-safe integration tests                 |
| REQ-EVT-007 | 1.1, 3.1, 4.1           | `app/event_analysis/config.py`; synthetic and annotated regression tests |
