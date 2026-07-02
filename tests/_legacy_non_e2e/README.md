# Legacy non-E2E tests (quarantined)

These tests are **not** screenshot-first end-to-end tests and are intentionally
excluded from the default `pytest` run.

## Why they're here

| File | What it does | Why it's not E2E |
|---|---|---|
| `test_orchestrator.py` | Mocked-HTTP unit tests of the Flask orchestrator `/decide` endpoint (uses `responses` library) | Not screenshot-first (sends fake bytes); all downstream services are mocked |
| `test_pipeline_integration.py` | In-process tests from hand-built enriched payload → parser → decision engine | Starts from a hardcoded enriched-payload dict, skipping detector + enricher stages |
| `test_preflop_scenario_manifest.py` | Parametrized decision-engine-only tests from hand-state JSON fixtures | Decision-engine stage only; not screenshot-first |

## Running them manually

```bash
uv run pytest tests/_legacy_non_e2e/ -v
```

## Disposition

These are preserved for reference. They may be deleted once the E2E suite
under `tests/e2e/` fully covers the same scenarios from screenshot input.
