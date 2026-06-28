# Preflop Scenario Suite Workflow

This folder documents the screenshot-driven preflop scenario workflow.

## Goal

Validate MVP preflop decisions (`watching`, `check`, `call`, `raise`, `fold`) against realistic scenarios.

## Source of Truth

Scenario metadata lives in:

- `tests/fixtures/preflop_scenarios/manifest.json`

Each row points to one `hand_state_fixture` JSON and a screenshot path.

## Expected results workflow

Expected results are manual ground truth. Each screenshot-backed scenario should be annotated by a person who inspects the screenshot and records the correct:

- final decision action (`watching`, `check`, `call`, `raise`, `fold`)
- optional expected reason text
- hero hole cards
- hero position
- blinds and stacks
- pot and amount-to-call
- whether hero is to act
- whether hero has folded

This is why the current fixtures exist: the manifest points to a screenshot and its trusted hand-state fixture.

### Minimal annotation process

1. Capture a representative screenshot under `test-screenshots/preflop/<scenario-id>/`.
2. Create a hand-state fixture JSON in `tests/fixtures/preflop_scenarios/hand_states/`.
3. Add a manifest entry to `tests/fixtures/preflop_scenarios/manifest.json` with:
   - `id`
   - `enabled`
   - `expected_action`
   - `expected_reason_contains`
   - `screenshot_path`
   - `hand_state_fixture`
4. Verify the fixture by running the pipeline in diagnostic mode:
   - `uv run python pipeline_tester.py path/to/screenshot.png --verbose`
5. Set `enabled=true` once the screenshot and fixture are stable.

### What to record in a hand-state fixture

A fixture should include the same core state used by the decision engine:

- `hero_cards`
- `position`
- `hero_seat`
- `big_blind`
- `small_blind`
- `hero_stack`
- `pot`
- `amount_to_call`
- `action_on`
- `action_history`
- `is_hero_turn`
- `hero_folded`
- `hero_cards_visibility`
- `schema_version`

If the screenshot is not yet stable, keep `enabled=false` until the fixture is confirmed.

## Current test runner

- `tests/test_preflop_scenario_manifest.py`

The runner loads all enabled scenarios from `manifest.json`, builds `HandState`, and asserts:

1. expected action
2. reason contains expected phrase (optional)
3. amount contract:
   - `watching` / `check` / `fold` -> `None`
   - `call` / `raise` -> `> 0`

## How to add a new screenshot-backed scenario

1. Capture 3-5 screenshots for the scenario class.
2. Save files under `test-screenshots/preflop/<scenario-id>/`.
3. Build one representative hand-state fixture JSON under:
   - `tests/fixtures/preflop_scenarios/hand_states/`
4. Add a row to `manifest.json` with:
   - `id`
   - `expected_action`
   - `expected_reason_contains`
   - `screenshot_path`
   - `hand_state_fixture`
5. Keep `enabled=true` only when both are true:
   - the fixture is stable
   - `screenshot_path` is populated

For trial runs, treat `enabled=true` as screenshot-backed-only coverage.

## Suggested scenario IDs (MVP)

1. `watching_hero_folded`
2. `watching_waiting_for_cards`
3. `watching_waiting_other_players`
4. `check_bb_unopened`
5. `raise_bb_limped_a9s`
6. `call_facing_raise_medium`
7. `raise_btn_open_premium`
8. `fold_btn_open_weak`
9. `fold_facing_raise_weak`

## Run

```bash
c:/Users/Rajiv/OneDrive/Documents/Code/pokerProject/.venv/Scripts/python.exe -m pytest tests/test_preflop_scenario_manifest.py -q
```
