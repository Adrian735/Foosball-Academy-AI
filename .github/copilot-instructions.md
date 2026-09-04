# Foosball Academy AI Guidelines

## Scope and Architecture

- This is a Python/FastAPI service for validating short foosball exercise videos.
- Keep the processing pipeline modular and independently testable. Static table calibration must remain separate from ball tracking, event analysis, and exercise validation.
- Reuse techniques from the root `AI-bbf` POC only when needed. Do not import or copy from its nested `foosball-ai` project.
- Keep CV modules independent of FastAPI, Celery, PostgreSQL, and storage. Their public inputs/outputs must be plain, serializable contracts.
- The initial supported setup is a Bonzini table filmed diagonally from above. Do not silently claim support for other table layouts or camera configurations.
- See `docs/PLAN.md`, `docs/FLOW.md`, and `docs/DETECTION_PLAN.md` before changing related behavior.

## Code Quality

- Use type hints for all public functions, methods, and class attributes.
- Write a concise docstring for every class, public function, and public method. Explain its responsibility, inputs/outputs, and non-obvious constraints.
- Add short inline comments only for non-obvious CV thresholds, geometry, or algorithmic decisions.
- Prefer small classes with one responsibility and explicit dependencies over monolithic analyzer classes or hidden global state.
- Put configurable CV thresholds in a versioned configuration object; do not scatter magic numbers through detector code.
- Return structured diagnostics and confidence with detection results. Low-confidence calibration must route to review, never be interpreted as a failed player exercise.

## Feature Workflow

For every feature or behavior change:

1. Read the relevant architecture document and nearby implementation/tests.
2. Write or update a step-by-step plan in the relevant `docs/` document before implementation.
3. Define acceptance criteria and regression coverage before changing production code.
4. Implement the smallest coherent slice, preserving module boundaries.
5. Add or update focused unit tests and fixture-based regression tests when the change affects CV behavior.
6. Run the focused test suite after the feature is complete, then report the exact validation result.

## Testing and Fixtures

- Never change expected test annotations merely to make a detector test pass; fix the detector or explicitly re-annotate the fixture after human review.
- Each supported real-world table/camera/lighting scenario requires a manually verified expected annotation fixture before detection thresholds are changed.
- Include negative tests for unsupported framing, insufficient resolution/duration, blur, and failed calibration.
- Run `pytest` after completing a feature. If the test runner or dependency is missing, install the declared development dependency or state the exact blocker.

## Local Development

- Load `.env` configuration using project-specific environment variables such as `FOOSBALL_DATABASE_URL`; do not introduce generic names such as `DATABASE_URL`.
- Use Docker Compose for PostgreSQL and Redis. The local Postgres container is exposed on port `5433` to avoid the host PostgreSQL service on `5432`.
