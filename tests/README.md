# Tests

## What "E2E" means in this repo

An **E2E test** starts from a **screenshot** and exercises the full live
pipeline via HTTP, validating the JSON output at each deterministic stage:

```
screenshot → detector (Roboflow) → enricher → hand-state parser → decision engine
```

- **Detector stage** is non-deterministic (Roboflow bbox coordinates and
  confidence values vary per call), so it is **logged but not asserted**.
  Only its downstream effects are checked.
- **Enricher, hand-state parser, and decision engine** stages are
  deterministic given a detector output, so their JSON is asserted against
  canonical fixtures.

## What is excluded

- **Screen-monitor** — not part of this pipeline.
- **Submodule unit tests** — each submodule has its own `tests/` folder;
  those are not touched here.
- **Mocked / non-screenshot tests** — quarantined under `tests/_legacy_non_e2e/`.
  Run them manually with `uv run pytest tests/_legacy_non_e2e/ -v`.

## Prerequisites

1. **Services running** (one command):
   ```bash
   uv run python manage_services.py start
   ```
2. **Roboflow API key** set in `poker-vision-object-detector/.env`:
   ```
   ROBOFLOW_API_KEY=your-key-here
   ```

The suite fails fast with a clear message if services are down or the API
key is missing.

## Run commands

### Smoke test (fast sanity check)

Runs **one** happy-path scenario to confirm services + API key are wired up
before committing to a full run:

```bash
uv run pytest tests/e2e -m smoke -v
```

### Full E2E

Runs all enabled scenarios (currently 5):

```bash
uv run pytest tests/e2e -v
```

## Fixture structure

Canonical expected outputs live under `tests/fixtures/e2e/`:

```
tests/fixtures/e2e/
├── scenarios.json                          # manifest of E2E scenarios
├── raise_bb_limped_a9s/
│   ├── expected_enricher.json
│   ├── expected_hand_state.json
│   └── expected_decision.json
├── check_bb_unopened/
│   └── ...
└── ...
```

Each scenario folder has one `expected_*.json` per deterministic stage.

## Capturing / regenerating fixtures

When a stage's output legitimately changes (e.g. enricher logic updated),
regenerate the canonical fixtures from a live pipeline run:

```bash
# All enabled scenarios
uv run python tests/e2e/capture_fixtures.py

# Smoke scenario only
uv run python tests/e2e/capture_fixtures.py --smoke-only

# Single scenario
uv run python tests/e2e/capture_fixtures.py --scenario raise_bb_limped_a9s
```

**Review the diff carefully** before committing regenerated fixtures — a
change in expected output may indicate a regression, not an improvement.

## Adding a new scenario

1. Add the screenshot to `test-screenshots/`.
2. Add a row to `tests/fixtures/e2e/scenarios.json` with `id`,
   `screenshot_path`, and `enabled: true`. Set `smoke: true` if it should
   be the smoke-test scenario.
3. Run `uv run python tests/e2e/capture_fixtures.py --scenario <id>` to
   generate the expected fixtures.
4. Inspect the generated JSON under `tests/fixtures/e2e/<id>/`.
5. Run `uv run pytest tests/e2e -v` to confirm.

## Failure diagnostics

Each stage assertion names the exact stage that regressed:

```
[STAGE: hand_state] Scenario 'raise_bb_limped_a9s' regressed.
Expected: {...}
Actual:   {...}
```

This lets you pinpoint whether the break is in the enricher, parser, or
decision engine without reading the full trace.
