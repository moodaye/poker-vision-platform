# MVP0 Issues — Prioritized Backlog

> **Tracking date:** 2026-07-03
> **Source:** UAT real-time testing session + codebase research
> **Priority axis** = functional impact (does it cause wrong poker action?). P0 = wrong/missed action; P1 = degrades decision quality / observability; P2 = testing friction; P3 = cleanup.
> **Severity axis** = system reliability impact. Critical = blocks real-time play; High = unreliable; Medium = degraded; Low = cosmetic.
> **Sole P0 = Issue #1 (end-to-end time).** Resolving it reduces severity of issues 5, 8, 9, 3.

## Summary Table

| # | Issue | Priority | Severity | Tier | Status |
|---|---|---|---|---|---|
| 23 | Hero card misclassified (`Qd` instead of `Td`) | P1 | Critical | 1 | Open |
| 6 | Strong hands folded/checked | P1 | High | 2 | Open (known gap) |
| 21 | `action_on` stays `"none"` when `bet_box` detected | P1 | High | 2 | Open |
| 22 | `amount_to_call` always 0 (bet / call_button OCR disabled) | P1 | High | 2 | Open (blocked by OCR tradeoff) |
| 8 | Interval mode doesn't wait for response | P2 | High | 2 | Open |
| 3 | Tester blind during lag | P2 | Medium | 2 | Open |
| 4 | No request ID to tie logs | P2 | Medium | 2 | Open |
| 5 | Action taken on stale screen | P2 | Medium | 2 | Open |
| 15 | Logs tab in screen monitor | P2 | Medium | 2 | Open (new) |
| 2 | False "success" log in action executor | P2 | Medium | 3 | Open |
| 7 | Player name spaces removed | P2 | Medium | 3 | Open |
| 9 | Duplicate screenshots enter pipeline | P2 | Medium | 3 | Open (subsumed by #8) |
| 24 | Opponent `is_folded` always `null` — fold state not tracked | P2 | Medium | 3 | Open (needs detector support) |
| 25 | Per-seat current-round bet amounts not in hand state | P2 | Medium | 3 | Open (blocked by OCR tradeoff) |
| 14 | Ousted player shown as folded | P3 | Low | 3 | Open (needs investigation) |
| 19 | Investigate HTTP round-trip gap (enricher → classifier) | P2 | Medium | 4 | Open (new) |
| 16 | Persistent HTTP client on enricher (httpx.Client) | P2 | Low | 4 | Open (new) |
| 17 | Persistent HTTP session on orchestrator (requests.Session) | P2 | Low | 4 | Open (new) |
| 18 | Replace Flask dev server with Waitress on card classifier | P2 | Low | 4 | Open (new — deprioritized) |
| 10 | Clean up tests/ folder structure | P3 | Low | 4 | Open |
| 26 | `is_all_in` emitted as `null` instead of `false` when stack known | P3 | Low | 4 | Open |
| 27 | `position` field name should be `hero_position` | P3 | Low | 4 | Open (breaking change — defer) |
| 1 | End-to-end time too long (10s/15s budget) | P0 | Critical | 1 | Closed |
| 20 | Halo not consistently detected for determining hero turn | P1 | High | 1 | Closed |
| 11 | Config options hidden in manual mode | P2 | Low | 4 | Closed |
| 12 | Compact UI not minimal enough | P2 | Low | 4 | Closed |
| 13 | Hand state parser JSON not pretty | P2 | Low | 4 | Closed |

## Execution Order

- **Pre-Step:** Create this file (`ISSUES.md`) — DONE
- **Tier 1 (P0):** Issue #1
- **Tier 2 (P1, observability + functional):** Issues #4, #15, #3, #2, #5, #8, #6, #13
- **Tier 3 (P1/P2, data quality):** Issues #7, #14, #9
- **Tier 4 (P2/P3, UX/cleanup/latency refinements):** Issues #11, #12, #10, #16, #17, #18, #19

---

## Issue #1 — End-to-end time too long

**Priority:** P0 | **Severity:** Critical | **Tier:** 1

**Description:** The poker game allows 10s for pre-flop and 15s for post-flop decisions. The pipeline's end-to-end time exceeds this budget, causing the tester to take manual action before the pipeline completes. In manual testing mode, the tester ends up acting themselves.

**Root cause:**
- The pipeline is a **synchronous blocking chain** in `orchestrator.py` `POST /decide`: detector → enricher → parser → decision → executor, each a blocking `requests.post()`.
- Observed timings from `logs/orchestrator.log` and `logs/detection-enricher.log`:
  - Object detector (Roboflow API): **1.87s – 7.50s** (network call)
  - Detection enricher: **0.11s – 31.47s** (the 31s outlier was 17 card classifications at ~2s each via httpx to card-classifier)
  - Hand state parser: ~0.01–0.02s
  - Decision engine: ~0.02–0.03s
- The 10s/15s time budget is **NOT encoded anywhere** in the codebase — the system has no awareness of it.
- `pipeline_tester.py` has a self-imposed target of `<5.00s` but does not enforce it.
- Timeouts: `REQUEST_TIMEOUT_SECONDS = 30` (orchestrator), action executor hardcoded `timeout=5`.

**Fix approach (optimize first — no budget-abort yet):**
1. **Profile each stage** with timing instrumentation (enricher already has `[timing]` logs; add equivalent to orchestrator for detector/parser/decision).
2. **Optimize the enricher** — the biggest offender. Card classification (httpx to card-classifier) is the bottleneck at ~2s per card. Options: (a) batch classify cards in a single call instead of per-card; (b) parallelize the httpx calls; (c) increase card-classifier worker count.
3. **Optimize the detector** — Roboflow API call is 1.8–7.5s. Options: (a) local YOLO inference instead of API; (b) request smaller image to Roboflow; (c) cache results for identical screenshots (ties to issue #9).
4. **Add a timing summary** to the orchestrator response so the tester can see where time is spent.

**Files to modify:**
- `orchestrator.py` — add per-stage timing, log summary
- `poker-vision-detection-enricher/detection_enricher.py` — optimize card classification path
- `poker-vision-detection-enricher/api.py` — batch/parallel card classify
- `poker-vision-card-classifier/api.py` — support batch input if needed

**Dependencies:** None (this is the first P0).

**Verification:**
1. Run `pipeline_tester.py` against `tests/fixtures/e2e/raise_bb_limped_a9s` screenshot.
2. Confirm end-to-end time < 10s (preflop budget) consistently across 5 runs.
3. Check the timing summary in the orchestrator response shows no single stage > 5s.

---

## Issue #2 — False "success" log in action executor

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** The action executor logs that an action "executed successfully" even when the screen has already transitioned to the next screen. Need to clarify whether "success" means the correct button was clicked, or just that *a* button was clicked.

**Root cause:**
- `poker-vision-action-executor/executor.py` `execute()` returns `ActionResult(success=True)` **if and only if** it found the poker window, found a matching button (case-insensitive prefix match), and clicked it.
- **No post-click verification** — it does not re-scan the window, check that the action was registered by the poker client, or confirm the screen changed as expected.
- `success=True` means "I clicked the centre of a button whose label matched" — not "the correct action was taken on the correct screen."

**Fix approach:**
1. **Clarify the log message** — rename from "executed successfully" to "button clicked" or "click dispatched" to reflect what actually happened.
2. **Add post-click verification** (optional, harder) — after clicking, re-scan the window to confirm the action button is no longer visible (indicating the poker client registered the click). This is a follow-up; the log clarification is the quick win.
3. **Add the button text that was matched** to the log (already partially there via `logger.info("Button matched: hwnd=%d text=%r variant=%r")`).

**Files to modify:**
- `poker-vision-action-executor/executor.py` — clarify log message, add post-click check
- `poker-vision-action-executor/api.py` — update response message if needed

**Dependencies:** None.

**Verification:**
1. Trigger an action execution and check the log says "button clicked" (or similar), not "executed successfully."
2. If post-click verification added: trigger an action on a stale screen and confirm the log indicates the button was still visible after click (verification failed).

---

## Issue #3 — Tester blind during lag

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** Due to lag in the pipeline response, the screen moves on while the action is still being waited on. The tester clicks manual capture on the screen monitor and is then blind to what is actually happening in the pipeline.

**Root cause:**
- No real-time pipeline status is surfaced to the screen monitor UI.
- The tester can only see logs by opening individual log files — no aggregated view.
- The screen monitor UI shows capture status but not pipeline progress.

**Fix approach:**
- **Resolved by Issue #15 (Logs tab — Live Tail view).** The Live Tail SSE stream shows real-time pipeline activity, so the tester can see which stage is running and whether it's stuck.
- This issue is largely subsumed by #15. Track as a dependency.

**Files to modify:**
- See Issue #15.

**Dependencies:** Issue #15 (Logs tab).

**Verification:**
1. During a UAT run, open the Logs tab → Live Tail view.
2. Confirm the tester can see pipeline stages progressing in real time without opening log files.

---

## Issue #4 — No request ID to tie logs

**Priority:** P1 | **Severity:** Medium | **Tier:** 2

**Description:** There is only the timestamp to tie logs together across services. This is difficult. At minimum, we need a request ID that travels the pipeline to tie the end-to-end run together.

**Root cause:**
- **No request-id / correlation-id / trace-id mechanism exists** in the live pipeline.
- Searched the entire codebase for `request_id`, `correlation_id`, `trace_id`, `run_id`, `X-Request-Id`, `X-Correlation` — none found in the live path.
- The only `run_id` concept is in the offline training runner (`poker-vision-object-detector/poker_vision/detect/config.py`), not the live inference path.
- The screen monitor sends `metadata: {"source": "ScreenStream"}` in webhook payloads but no unique request identifier.

**Fix approach:**
1. **Screen monitor mints the ID** (confirmed decision) — when a screenshot is captured, generate a UUID (e.g., `str(uuid.uuid4())[:8]` for a short ID).
2. **Pass the ID in the webhook payload** — add `request_id` to the metadata/JSON sent to the orchestrator's `POST /decide`.
3. **Orchestrator propagates the ID** — read `request_id` from the incoming request, include it in every downstream `requests.post()` call (as a header `X-Request-Id` and/or in the JSON body).
4. **Each service logs the ID** — add `request_id` to every log line. For Flask services, use a logging filter or just include it in the log message. For the enricher (FastAPI), read the header and log it.
5. **Action executor** — receives the ID from the orchestrator and logs it.

**Files to modify:**
- `poker-vision-screen-monitor/screen_capture.py` — mint UUID at capture, add to webhook payload (`_send_to_external_systems`, `_send_image_to_url`)
- `orchestrator.py` — read `request_id` from request, propagate to all downstream calls (`call_object_detector`, `call_detection_enricher`, `call_hand_state_parser`, `call_decision_engine`, `call_action_executor`)
- `poker-vision-detection-enricher/api.py` — read `X-Request-Id` header, log it
- `poker-vision-hand-state-parser/api.py` — read header, log it
- `poker-vision-decision-engine/api.py` — read header, log it
- `poker-vision-action-executor/api.py` — read header, log it

**Dependencies:** None (foundational for #15 By Request ID view).

**Verification:**
1. Trigger a capture in manual mode.
2. Grep each of the 6 service log files for the same request ID.
3. Confirm the ID appears in: screen-monitor.log, orchestrator.log, detection-enricher.log, hand-state-parser.log, decision-engine.log, action-executor.log.

---

## Issue #5 — Action taken on stale screen

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** The action should only be taken if the screen is the same one as the screenshot was taken on. Currently, due to lag, the screen may have moved on by the time the executor clicks.

**Root cause:**
- `poker-vision-action-executor/executor.py` has **no screen-state comparison** before clicking.
- It operates purely against the live window's Win32 control tree at execution time — it does not receive or compare the original screenshot.
- The executor trusts that the orchestrator's decision is still valid for the current screen state.

**Fix approach:**
1. **Pass a screen fingerprint to the executor** — the orchestrator/screen-monitor computes a lightweight fingerprint of the screenshot (e.g., a hash of the image, or a hash of the detected button labels) and sends it with the action.
2. **Executor verifies before clicking** — before clicking, the executor re-reads the live window's button labels and compares to the fingerprint. If they differ, it aborts with `success=False, message="screen changed"`.
3. **Alternative (simpler):** pass the expected button label (e.g., "Fold") to the executor and have it verify that button is still present before clicking. If the button is gone (screen moved on), abort.

**Files to modify:**
- `orchestrator.py` — pass screen fingerprint or expected button to executor
- `poker-vision-action-executor/executor.py` — verify screen state before clicking
- `poker-vision-action-executor/models.py` — add fingerprint/expected_button to `ActionRequest`
- `poker-vision-action-executor/api.py` — accept new field

**Dependencies:** Less severe once Issue #1 (end-to-end time) is resolved — faster pipeline = less staleness. But the guard should still be added.

**Verification:**
1. Capture a screenshot, wait for the screen to change, then trigger an action.
2. Confirm the executor logs "screen changed, aborting" and does not click.

---

## Issue #6 — Strong hands folded/checked

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** Weak hands seem to be folded correctly, but strong hands are also folded and/or checked. Need to document the pre-flop decision logic.

**Root cause:**
- **The decision logic is correct.** PREMIUM and STRONG hands never fold in any situation handler (except short-stack shoves, which are intentional). See `poker-vision-decision-engine/decision_engine/preflop.py`.
- **The real cause is empty `action_history`.** The vision layer (object detector + enricher) does not yet emit `action`/`player` fields. Without action history, `classify_situation()` returns `UNOPENED` even when limps have occurred.
- Example: the `raise_bb_limped_a9s` e2e scenario expects a `raise` (iso-raise), but the actual decision is `check` because the parser sees `action_history == []` and `amount_to_call == 0` → `UNOPENED` → BB checks.
- **This was previously tracked** in `README.md` § "Remaining gaps" as "Action history is usually empty" — now migrated here.

**Fix approach:**
- **This is NOT a decision-engine bug.** Do not modify `preflop.py`.
- The fix is upstream: the vision layer must detect and emit action objects (fold/call/raise/bet) attributed to players. This is a larger effort (object detector training + enricher action-detection logic).
- **Immediate action:** Document the pre-flop decision logic clearly (it already exists in `poker-vision-decision-engine/mvp-specification.md`) and point testers to it. Add a note to this issue explaining the root cause.

**Files to reference (do not modify):**
- `poker-vision-decision-engine/decision_engine/preflop.py` — the pre-flop logic (correct)
- `poker-vision-decision-engine/mvp-specification.md` — documented strategy
- `poker-vision-decision-engine/decision_engine/hand_eval.py` — hand classification (5 buckets: PREMIUM/STRONG/MEDIUM/SPECULATIVE/WEAK)

**Dependencies:** Resolving this fully requires vision-layer action detection (large effort, post-MVP0). For now, document and defer.

**Verification:**
1. Confirm `mvp-specification.md` documents the pre-flop logic (it does).
2. Unit test: feed a HandState with `action_history=[{"action":"call","player":"BTN"}]` and confirm `classify_situation` returns `FACING_LIMP` (not `UNOPENED`).

---

## Issue #7 — Player name spaces removed

**Priority:** P1 | **Severity:** Medium | **Tier:** 3

**Description:** In the e2e test `tests/e2e/...raise_bb_limped...` scenario, spaces in player names are getting removed.

**Root cause:**
- **The bug is in `poker-vision-detection-enricher/ocr_module.py`, in the `run_ocr` function (~lines 150–158).**
- Two compounding problems:
  1. **`"".join(text_parts)`** concatenates OCR words with an **empty string** — not a space. So a name like `"mood ae"` (two words) becomes `"moodae"`.
  2. **The `player_name` Tesseract whitelist omits the space character.** Compare to `blinds` and `total_pot` profiles, which include a trailing space. The `player_name` whitelist is `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.` — no space.
- The parser itself only does `.strip()` (preserves internal spaces) — the space removal happens upstream in the enricher.
- Evidence: `tests/fixtures/e2e/raise_bb_limped_a9s/expected_hand_state.json` shows `"moodae"` (space-stripped).

**Fix approach:**
1. **Add a space to the `player_name` Tesseract whitelist** in `ocr_module.py`.
2. **Change `"".join(text_parts)` to `" ".join(text_parts)`** for the `player_name` profile (or conditionally).
3. **Add a `player_name` branch in `_clean_text_for_profile`** that normalizes whitespace via `re.sub(r"\s+", " ", cleaned).strip()` (like `blinds`/`total_pot` do).
4. **Regenerate the e2e fixtures** — the expected names in `expected_enricher.json` and `expected_hand_state.json` will change once spaces are preserved. Use `tests/e2e/capture_fixtures.py` to regenerate.

**Files to modify:**
- `poker-vision-detection-enricher/ocr_module.py` — fix whitelist + join + clean
- `tests/fixtures/e2e/raise_bb_limped_a9s/expected_enricher.json` — regenerate
- `tests/fixtures/e2e/raise_bb_limped_a9s/expected_hand_state.json` — regenerate

**Dependencies:** None.

**Verification:**
1. Run the enricher on the `raise_bb_limped_a9s` screenshot.
2. Confirm player names retain internal spaces (e.g., `"mood ae"` not `"moodae"`).
3. Regenerate fixtures and re-run e2e tests — all should pass.

---

## Issue #8 — Interval mode doesn't wait for response

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** In screen monitor interval mode, the monitor needs to wait for the pipeline response before sending the next screenshot. In manual mode, blocking the user from capturing is less important.

**Root cause:**
- `poker-vision-screen-monitor/screen_capture.py` `_send_to_external_systems()` sends the image on a **fire-and-forget daemon thread**.
- The capture loop (`_capture_loop`) captures, calls `_handle_captured_image`, then `time.sleep(self._config["interval"])` — it does **not** block on the orchestrator response.
- Result: in interval mode, the next screenshot can be captured (and a new webhook POST initiated) while the previous pipeline run is still in flight → overlapping runs.

**Fix approach:**
1. **In interval mode:** make `_send_to_external_systems` **synchronous** (block on the orchestrator response) OR use a lock/semaphore to prevent a new capture while a pipeline run is in flight.
2. **In manual mode:** optionally block the "Capture Now" button while a pipeline run is in progress (show a "processing..." state). Less important per user.
3. **Add a "pipeline busy" flag** to the screen monitor status so the UI can show when a run is in progress.

**Files to modify:**
- `poker-vision-screen-monitor/screen_capture.py` — serialize sends in interval mode (`_send_to_external_systems`, `_capture_loop`)
- `poker-vision-screen-monitor/static/js/monitor.js` — disable Capture Now button while busy (optional)

**Dependencies:** Less severe once Issue #1 (end-to-end time) is resolved — faster pipeline = less overlap window. But serialization should still be added.

**Verification:**
1. Set interval mode with a short interval (e.g., 2s).
2. Confirm no overlapping pipeline runs — check orchestrator logs for sequential (not interleaved) request IDs.
3. Confirm the "pipeline busy" flag shows in the UI during a run.

---

## Issue #9 — Duplicate screenshots enter pipeline

**Priority:** P2 | **Severity:** Medium | **Tier:** 3

**Description:** If the screen monitor sends the same screenshot (same hand state) as the one currently being processed, this request should be blocked from entering the pipeline.

**Root cause:**
- No deduplication mechanism. The screen monitor sends every capture regardless of whether the screen has changed.
- Subsumed by Issue #8 — serializing pipeline runs prevents simultaneous processing of the same screenshot.

**Fix approach:**
- **Addressed by Issue #8** (serialize pipeline runs). Once sends are serialized, duplicate/simultaneous processing is prevented.
- **Additional (optional):** add a content hash check — if the new screenshot's hash matches the last-processed screenshot, skip the send. This prevents reprocessing identical screens even in manual mode.

**Files to modify:**
- See Issue #8.
- Optional: `poker-vision-screen-monitor/screen_capture.py` — add content hash dedup.

**Dependencies:** Issue #8.

**Verification:**
1. Send the same screenshot twice in manual mode.
2. Confirm the second send is skipped (if hash dedup added) or queued (if serialization only).

---

## Issue #10 — Clean up tests/ folder structure

**Priority:** P3 | **Severity:** Low | **Tier:** 4

**Description:** Need to clean up folder structure, especially under `tests/`.

**Root cause:**
- The `tests/` folder is mostly clean already. `_legacy_non_e2e/` is quarantined with a README explaining why.
- Structure: `tests/e2e/`, `tests/fixtures/e2e/`, `tests/preflop_scenarios/`, `tests/_legacy_non_e2e/`, `tests/conftest.py`, `tests/README.md`.

**Fix approach:**
1. Review `tests/_legacy_non_e2e/` — confirm the tests are truly obsolete or decide to delete them.
2. Ensure consistent naming (e.g., all test files prefixed with `test_`).
3. Add a `tests/README.md` update if the structure changes.
4. Low priority — only do this if it blocks other work or if the folder grows messy.

**Files to modify:**
- `tests/` — review and clean

**Dependencies:** None.

**Verification:**
1. `pytest` still collects and runs all intended tests.
2. No broken imports after cleanup.

---

## Issue #11 — Config options hidden in manual mode

**Priority:** P2 | **Severity:** Low | **Tier:** 4

**Description:** Need config options visible in manual mode (screen monitor). Specific example: "send to external systems."

**Root cause:**
- The config panel exists in the full dashboard view but is hidden in compact/manual mode.
- The "send to external" toggle (`enable-external`) and webhook config are in the External Integration Panel, which is not visible in compact mode.

**Fix approach:**
1. **In compact mode:** add a minimal config dropdown or gear icon that exposes key toggles (especially `send_to_external`).
2. **In full manual mode:** ensure the config panel is accessible (it should be, since compact mode is a toggle, not a replacement).

**Files to modify:**
- `poker-vision-screen-monitor/templates/index.html` — add config access in compact panel
- `poker-vision-screen-monitor/static/js/monitor.js` — wire up config toggle in compact mode

**Dependencies:** None.

**Verification:**
1. Enter compact mode.
2. Confirm the "send to external" toggle is accessible and functional.

---

## Issue #12 — Compact UI not minimal enough

**Priority:** P2 | **Severity:** Low | **Tier:** 4

**Description:** Entering compact UI option should resize the window to a minimal level. All the user needs is a button to click that indicates a screenshot should be taken. Currently the button is very big (there are 2 buttons actually). Get to the smallest possible screen and position it to the top left of the monitor.

**Root cause:**
- `enterCompactMode()` (`monitor.js:358`) shows `#manual-compact-panel` — a 480px-wide card with "Capture Now" and "Return to Full Monitor" buttons.
- The "Capture Now" button uses `btn-lg` (large).
- No auto-resize or reposition of the browser window.

**Fix approach:**
1. **Shrink the compact panel** — remove `btn-lg`, reduce padding, make the panel ~200px wide.
2. **Auto-position the window** — use `window.moveTo(0, 0)` and `window.resizeTo(width, height)` to position top-left. (Note: some browsers restrict this for non-script-opened windows.)
3. **Reduce to one primary button** — "Capture Now" should be the only prominent button; "Return to Full Monitor" can be a small text link or icon.
4. **Hide all non-essential UI** — navbar, footer, other cards should be hidden in compact mode (some already are via `body.compact-manual-ui`).

**Files to modify:**
- `poker-vision-screen-monitor/static/js/monitor.js` — `enterCompactMode()`, add window positioning
- `poker-vision-screen-monitor/static/css/custom.css` — compact panel styles
- `poker-vision-screen-monitor/templates/index.html` — compact panel markup

**Dependencies:** None.

**Verification:**
1. Enter compact mode.
2. Confirm the window is small (~200px wide), positioned top-left, with a single small "Capture Now" button.
3. Confirm the button still triggers a capture.

---

## Issue #13 — Hand state parser logs should prettify JSON

**Priority:** P2 | **Severity:** Low | **Tier:** 4

**Description:** Hand state parser logs should prettify the JSON logs.

**Root cause:**
- `poker-vision-hand-state-parser/api.py` `_log_json()` uses `json.dumps(payload, sort_keys=True, default=str)` — **no `indent=` parameter**, so JSON is compact single-line.
- The resulting log lines are very long (truncated at 2000 chars by the log handler).

**Fix approach:**
1. **Add `indent=2`** to the `json.dumps` call in `_log_json()`.
2. **Or use a multi-line log format** — log each top-level key on its own line.
3. **Consider a config toggle** — `pretty_json_logs: true` in `config.yaml` (default true for readability; false for machine parsing).

**Files to modify:**
- `poker-vision-hand-state-parser/api.py` — `_log_json()` function (~line 62)
- `poker-vision-hand-state-parser/config.yaml` — add `pretty_json_logs` option (optional)

**Dependencies:** None.

**Verification:**
1. Trigger a `/parse` request.
2. Check `logs/hand-state-parser.log` — confirm the JSON is multi-line/indented and readable.

---

## Issue #14 — Ousted player shown as folded

**Priority:** P1 | **Severity:** Medium | **Tier:** 3

**Description:** If a player has been ousted from the tournament, the hand state parser indicates the player has folded.

**Root cause:**
- **The code DOES distinguish ousted from folded.** `_seat_status()` in `hand_state_parser.py` (~line 225) returns `"eliminated_tournament"` when `stack <= 0`, vs `"folded_this_hand"` for folded players.
- The `SeatStatus` type in `decision_engine/models.py` confirms: `deciding`, `waiting_turn`, `folded_this_hand`, `watching_hand`, `all_in`, `eliminated_tournament`, `unknown`.
- **The issue is likely a detection problem** — the player's stack is not being read as 0 (or not read at all), so the `stack <= 0` check doesn't trigger. Instead, the player's hidden cards trigger the `hidden_cards_post_blind_pot` heuristic → `folded_this_hand`.
- **Needs investigation** before coding a fix.

**Fix approach:**
1. **Investigate first** — feed a screenshot with an ousted player (zero stack) and check what the parser outputs. Is `stack` null or 0?
2. **If stack is null:** the parser can't distinguish ousted from folded. Add a fallback — if a player has no cards AND no stack detected AND no action history, mark as `eliminated_tournament` (or `unknown`).
3. **If stack is 0 but still marked folded:** debug the `_seat_status` logic path.
4. **Consider a vision signal** — if the poker client shows an "eliminated" badge, the detector could be trained to detect it.

**Files to modify (after investigation):**
- `poker-vision-hand-state-parser/hand_state_parser.py` — `_seat_status()`, fold detection logic

**Dependencies:** None (but investigation needed first).

**Verification:**
1. Feed a screenshot with an ousted player (zero stack, no cards).
2. Confirm `seat_status == "eliminated_tournament"` (not `folded_this_hand`).

---

## Issue #15 — Logs tab in screen monitor

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** Add a "Logs" tab to the screen monitor UI to view end-to-end logs by mint ID and by pipeline stage. Two views: (a) By Request ID — filter all 7 logs by request ID, grouped by stage; (b) Live Tail — SSE stream of all logs, color-coded by stage.

**Root cause:**
- No log-viewing capability exists in the screen monitor.
- The tester must open 7 individual log files to debug a run.
- No way to tie logs together by request ID (depends on Issue #4).

**Fix approach:**

### Architecture
- The screen monitor (Flask process) reads the 7 log files directly from `logs/` folder (all services on the same machine).
- SSE runs only between the browser and the screen monitor — the 6 other services just write log files as they do today.
- The screen monitor tails the files and pushes new lines via SSE.
- Reuses the existing `/api/image/stream` pattern (MJPEG generator) as a template for SSE.

### UI changes
1. **Add Bootstrap nav tabs** at the top of `#main-dashboard` in `index.html`: "Monitor" | "Logs".
2. **Logs tab content:**
   - **By Request ID view:** input field for mint ID → fetch `GET /api/logs?request_id=<id>` → render results grouped by stage (Capture → Detect → Enrich → Parse → Decide → Execute), each stage a collapsible card.
   - **Live Tail view:** `EventSource` connected to `GET /api/logs/stream` → render lines in real time, color-coded by source service.

### Backend changes
3. **New Flask routes in `app.py`:**
   - `GET /api/logs` — query param `request_id` (optional), `stage` (optional). Reads all 7 log files, filters by request_id, returns JSON `[{service, timestamp, level, message, request_id}, ...]`.
   - `GET /api/logs/stream` — SSE endpoint. Tails all 7 log files, pushes new lines as `data: {service, timestamp, level, message}\n\n`.
4. **Log file paths:** resolve from config (the `logs/` folder location, default `../logs/` relative to the screen monitor).

### JS changes
5. **New `static/js/logs.js`:**
   - `loadLogsByRequestId(id)` — fetch and render By Request ID view.
   - `startLiveTail()` / `stopLiveTail()` — manage `EventSource` lifecycle.
   - Color-code by service: screen-monitor (gray), orchestrator (blue), detection-enricher (yellow), hand-state-parser (green), decision-engine (purple), action-executor (red), card-classifier (orange).

### Phasing
- **Phase 1 (quick win for #3):** Live Tail view — works independently of #4. Build first.
- **Phase 2 (after #4 lands):** By Request ID view — requires mint ID in log lines.

**Files to modify:**
- `poker-vision-screen-monitor/app.py` — add `/api/logs` and `/api/logs/stream` routes
- `poker-vision-screen-monitor/templates/index.html` — add nav tabs + Logs tab content
- `poker-vision-screen-monitor/static/js/logs.js` — new file, logs view logic
- `poker-vision-screen-monitor/static/js/monitor.js` — add tab switching logic
- `poker-vision-screen-monitor/static/css/custom.css` — logs tab styles, color coding
- `poker-vision-screen-monitor/config.py` — add `logs_folder` config option

**Dependencies:**
- Phase 1 (Live Tail): None — build immediately.
- Phase 2 (By Request ID): Issue #4 (mint ID) must be implemented first.

**Verification:**
1. **Live Tail:** Open the Logs tab → Live Tail view. Trigger a capture. Confirm log lines appear in real time, color-coded by service.
2. **By Request ID:** After #4 is implemented, trigger a capture, note the mint ID, enter it in the By Request ID view. Confirm all 7 services' log lines for that ID appear, grouped by stage.
3. Confirm the SSE connection auto-reconnects if the browser tab is briefly backgrounded.

---

## Issue #16 — Persistent HTTP client on enricher (httpx.Client)

**Priority:** P2 | **Severity:** Low | **Tier:** 4

**Description:** The detection enricher uses `httpx.post()` (module-level function) for every card-classifier call, creating a throwaway client each time. This means a new TCP connection + handshake per call. A persistent `httpx.Client` would reuse the connection across calls.

**Root cause:**
- `detection_enricher.py` `_classify_snip()` calls `httpx.post(...)` directly — the module-level function creates a transient client with no connection pooling.
- `DetectionEnricher.__init__` stores `classifier_url` but never creates an `httpx.Client` instance.
- Per-call overhead: ~5–15ms for TCP handshake on localhost (small per call, but adds up with N cards).

**Fix approach:**
1. Create a persistent `httpx.Client` in `DetectionEnricher.__init__` (with appropriate timeout config).
2. Replace `httpx.post(...)` calls with `self._client.post(...)`.
3. Ensure the client is closed on shutdown (or rely on process lifetime).

**Files to modify:**
- `poker-vision-detection-enricher/detection_enricher.py` — add `self._client = httpx.Client(...)` in `__init__`, use in `_classify_snip` (or the batch call if Issue #1's batch endpoint is implemented)

**Dependencies:** None. Stacks on top of the batch endpoint (Issue #1 work) — if batching is implemented, the client is used for the single batch call; if not, it's used for N serial calls.

**Verification:**
1. Run the enricher and confirm no new TCP connections per card (check via `netstat` or logging).
2. Confirm a small timing improvement per call.

---

## Issue #17 — Persistent HTTP session on orchestrator (requests.Session)

**Priority:** P2 | **Severity:** Low | **Tier:** 4

**Description:** The orchestrator uses `requests.post()` (module-level function) for every downstream service call (detector, enricher, parser, decision, executor), creating a throwaway session each time. A persistent `requests.Session` would reuse connections across calls.

**Root cause:**
- `orchestrator.py` calls `requests.post(...)` in `call_object_detector()`, `call_detection_enricher()`, `call_hand_state_parser()`, `call_decision_engine()`, `call_action_executor()` — each uses the module-level function with no session reuse.
- Per-call overhead: ~5–15ms for TCP handshake on localhost (5 services × ~10ms = ~50ms per screenshot).

**Fix approach:**
1. Create a module-level `requests.Session()` (or on a class if the orchestrator is refactored to one).
2. Replace all `requests.post(...)` calls with `session.post(...)`.
3. The session automatically pools connections per host (each service is a different host:port, so each gets its own kept-alive connection).

**Files to modify:**
- `orchestrator.py` — create `session = requests.Session()`, replace `requests.post` with `session.post` in all 5 `call_*` functions

**Dependencies:** None. Independent of other issues.

**Verification:**
1. Run the orchestrator and confirm connections are reused (check via `netstat` — should see persistent connections to each service port).
2. Confirm a small timing improvement per pipeline run (~50ms).

---

## Issue #18 — Replace Flask dev server with Waitress on card classifier

**Priority:** P2 | **Severity:** Low | **Tier:** 4

**Description:** The card classifier service runs on Flask's built-in development server (`app.run()`). Originally hypothesized that the dev server added ~900ms per-request overhead. **Timing instrumentation (added during Issue #1 work) disproved this** — the Flask endpoint overhead is only ~1ms. This issue is deprioritized.

**Root cause (updated with measured data):**
- `poker-vision-card-classifier/api.py` uses `app.run(host="0.0.0.0", port=5001, debug=False)` — Flask's dev server (Werkzeug).
- **Measured breakdown (n=2 batch call):**
  - Classifier endpoint total: 0.288s (decode=0.018s, infer=0.269s, Flask overhead=~1ms)
  - Enricher HTTP round-trip: 1.564s (encode=0.003s, http=1.564s)
  - **Gap between enricher http and classifier total: ~1.276s** — this is NOT Flask dev server overhead
- The ~900ms estimate was wrong. Flask dev server overhead is negligible (~1ms).
- The 1.276s gap is under investigation as Issue #19.

**Fix approach (if pursued later):**
1. Add `waitress>=3.0.0` to `poker-vision-card-classifier/pyproject.toml` dependencies.
2. Replace `app.run(...)` with `waitress.serve(app, ...)` in `api.py`.
3. Expected saving: ~1ms (negligible). Only worth doing if concurrent request handling becomes needed.

**Files to modify:**
- `poker-vision-card-classifier/pyproject.toml` — add `waitress` dependency
- `poker-vision-card-classifier/api.py` — replace `app.run()` with `waitress.serve()`

**Dependencies:** None. Deprioritized — measure actual impact before implementing.

**Verification:**
1. Restart the card classifier service.
2. Compare timing logs before/after — expect negligible change (~1ms).

---

## Issue #19 — Investigate HTTP round-trip gap (enricher → classifier)

**Priority:** P2 | **Severity:** Medium | **Tier:** 4

**Description:** Timing instrumentation revealed a ~1.276s gap between the enricher's HTTP round-trip time and the classifier's endpoint total time. This gap is the largest remaining latency component in the card classification path and needs investigation before it can be optimized.

**Root cause (measured data from 2026-07-03):**
- For a batch of 2 cards:
  - Enricher `http` (full round-trip): **1.564s**
  - Classifier `total` (endpoint function): **0.288s**
  - Gap: **1.276s** (unaccounted for)
- The gap covers: TCP connection setup, HTTP request transmission, Werkzeug WSGI request parsing (before endpoint function), HTTP response transmission, httpx response parsing.
- On localhost, TCP setup should be <1ms and payload transmission <10ms — so 1.276s is **suspiciously high**.
- Possible culprits:
  1. **httpx client creation overhead** — `httpx.post()` (module-level) creates a new client + transport + connection per call. This may be more expensive than expected.
  2. **Werkzeug WSGI request parsing** — the dev server's request handling before the endpoint function runs (not captured by our timing).
  3. **Something in `@app.before_request`** — though it's just a None check.
  4. **Payload size** — base64 PNG encoding of card crops may be larger than expected, slowing transmission.

**Fix approach:**
1. **Isolate the gap** — add a `@app.before_request` timestamp to the classifier to measure WSGI overhead (time from request received to endpoint function start).
2. **Try persistent `httpx.Client`** (Issue #16) — if the gap shrinks, it's httpx client creation overhead.
3. **Log payload size** — measure the base64 payload size to rule out transmission time.
4. **Try `requests` instead of `httpx`** — if the gap disappears, it's an httpx-specific issue.

**Files to modify (investigation):**
- `poker-vision-card-classifier/api.py` — add `@app.before_request` timing
- `poker-vision-detection-enricher/detection_enricher.py` — log payload size

**Dependencies:** None. Related to Issues #16 and #18 but independent to investigate.

**Verification:**
1. Run a batch classify call and compare the before_request timestamp to the endpoint total.
2. Identify which component consumes the 1.276s gap.
3. Document findings and create a targeted fix issue if warranted.

---

## Issue #20 — Halo not consistently detected for determining hero turn | P1 | High | 1 | Open (new) |

**Priority:** P1 | **Severity:** High | **Tier:** 1

**Description:** Found at least one screenshot where halo is not detected for determining hero's turn. E.g., see capture_20260703_161846_678303

**Root cause (measured data from 2026-07-03):**
- TBD

**Fix approach:**

**Files to modify (investigation):**

**Dependencies:** 

**Verification:**

---

## Issue #21 — `action_on` stays `"none"` when `bet_box` detected

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** When the `bet_box` UI widget is present (the hero's action panel), `is_hero_turn` is correctly set to `true`, but `action_on` remains `"none"`. The two fields are inconsistent, and downstream logic that checks `action_on` for seat routing gets the wrong answer.

**Affected scenarios:** `check_bb_unopened`, `fold_btn_open_weak`.

**Root cause:**
- In `hand_state_parser.py`, the `bet_box` detection branch sets `is_hero_turn = True` but never sets `action_on`. That variable retains its initialised value of `"none"`.
- The halo-based branch (the `else`) correctly sets `action_on = active_seat`, but is skipped entirely when `bet_box` is present.

**Fix approach:**
1. Inside the `bet_box` branch, add `action_on = position` (the hero's seat label) immediately after setting `is_hero_turn = True`.

**Files to modify:**
- `poker-vision-hand-state-parser/hand_state_parser.py` — `bet_box` branch (~1 line)

**Dependencies:** None.

**Verification:**
1. Run the `check_bb_unopened` and `fold_btn_open_weak` e2e scenarios.
2. Confirm `action_on` matches `hero_seat` and `is_hero_turn == true`.

---

## Issue #22 — `amount_to_call` always 0 (bet / call_button OCR disabled)

**Priority:** P1 | **Severity:** High | **Tier:** 2

**Description:** `amount_to_call` is 0 in every scenario where a call is actually required. Affects `fold_btn_open_weak` (should be 20 — the BB amount the BTN must call) and `watching_waiting_other_players` (should be 10 — the outstanding SB-to-BB call). This causes the decision engine to treat every preflop situation as a check/no-call scenario.

**Root cause:**
- The parser derives `amount_to_call` from `bet`, `max_bet`, or `min_bet` enricher objects that carry OCR text.
- **OCR is intentionally disabled for `bet` and `call_button` objects** in the enricher's default processing config to reduce latency (see `poker-vision-detection-enricher/README.md` — *OCR Performance Tradeoffs*). These objects carry `processing: "none"` and no `ocr_text`.
- The `call_button` object (e.g., "Call 20") is detected with high confidence but never read. It is the most reliable source of the exact call amount.

**Fix approach (when latency budget allows):**
1. Add `"call_button": "ocr"` to the enricher processing map — this reads the "Call XX" button text directly.
2. Add `call_button` as the first candidate source in the parser's `amount_to_call` extraction loop (higher priority than `bet`).
3. Optionally also add `"bet": "ocr"` to read posted blind amounts.
4. Re-capture affected e2e fixtures after the change.

**⚠️ Blocked by OCR latency tradeoff** — do not enable until the latency budget (Issue #1 resolution) allows it. See enricher README for context.

**Files to modify:**
- `poker-vision-detection-enricher/api.py` — add `call_button` (and optionally `bet`) to default processing map
- `poker-vision-hand-state-parser/hand_state_parser.py` — add `call_button` to `amount_to_call` source priority list
- `tests/fixtures/e2e/*/expected_*.json` — re-capture affected fixtures

**Dependencies:** Issue #1 (latency budget must allow additional OCR calls).

**Verification:**
1. Feed a screenshot where hero must call (e.g., `screenshot_preflop_4`). Confirm `amount_to_call == 20`.
2. Re-run all e2e tests and confirm they pass.

---

## Issue #23 — Hero card misclassified (`Qd` instead of `Td`)

**Priority:** P1 | **Severity:** Critical | **Tier:** 1

**Description:** In the `check_bb_unopened` scenario, the hero is holding 5d and Td, but the hand state reports `[
"5d", "Qd"]`. The Ten of Diamonds is misclassified as the Queen of Diamonds.

**Root cause:**
- The card classifier model confuses `Td` and `Qd`. Both cards have a similar visual appearance at small crop sizes — the Q tail and the T crossbar are visually close in low-resolution crops from this poker client's UI.
- This is a **model quality / training data gap**, not a parser logic bug. The card classifier is in `poker-vision-card-classifier/`.

**Fix approach:**
1. Identify the `Td` crop from `screenshot_preflop_14.png` (the `check_bb_unopened` screenshot).
2. Add this crop (and similar `Td` crops from other screenshots) to the card classifier training dataset with the correct label.
3. Retrain the classifier and verify `Td` no longer misclassifies as `Qd`.
4. Re-capture the `check_bb_unopened` fixture after retraining.

**Files to modify:**
- `poker-vision-card-classifier/` — training data, retrain
- `tests/fixtures/e2e/check_bb_unopened/expected_*.json` — re-capture after fix

**Dependencies:** Requires card classifier retraining pipeline.

**Verification:**
1. Feed `screenshot_preflop_14.png` through the pipeline.
2. Confirm hero cards are reported as `["5d", "Td"]`.

---

## Issue #24 — Opponent `is_folded` always `null` — fold state not tracked

**Priority:** P2 | **Severity:** Medium | **Tier:** 3

**Description:** For all opponent seats, `is_folded` is always `null` and `status` defaults to `waiting_turn` regardless of whether the player has actually folded. This affects `watching_waiting_other_players` where at least one opponent should be marked as folded.

**Root cause:**
- The parser hardcodes `is_folded: null` for all non-hero seats. There is no mechanism for the enricher/detector to signal opponent fold state.
- `_seat_status()` returns `waiting_turn` for all opponents not matching `action_on`, with no fold distinction.
- The detector does not yet emit a `folded_player` class, and there is no enricher inference path for opponent folds.

**Fix approach:**
1. **Detector (long-term):** Train the object detector to recognise a `folded_player` class (e.g., greyed-out avatar, face-down cards removed, badge visible).
2. **Parser (medium-term):** Once the enricher can flag a `player_other` or `player_name` as folded via `spatial_info.is_folded`, propagate that to `SeatState.is_folded` and the status calculation.
3. **Short-term (heuristic):** If a player has no detected cards AND no visible chip stack AND is not the active player, infer `watching_hand` rather than `waiting_turn`.

**Files to modify (after investigation):**
- `poker-vision-hand-state-parser/hand_state_parser.py` — `_seat_status()`, seat assembly loop
- `poker-vision-detection-enricher/detection_enricher.py` — spatial/fold signal propagation

**Dependencies:** Detector retraining (long-term path).

**Verification:**
1. Feed a screenshot where one opponent has folded.
2. Confirm their `is_folded == true` and `status == "folded_this_hand"`.

---

## Issue #25 — Per-seat current-round bet amounts not in hand state

**Priority:** P2 | **Severity:** Medium | **Tier:** 3

**Description:** The hand state has no field for each player's bet already committed in the current round. For example, in `fold_btn_open_weak` (screenshot_preflop_4), Weave has 10 in the pot (SB), Donna1212 has 20 (BB), and the hero has 0 (has not acted yet). This information is needed for accurate `amount_to_call` derivation and future action history support.

**Root cause:**
- No `current_bet` or similar per-seat field exists in `SeatState`.
- `bet` objects in the enricher are detected spatially and owned via `spatial_info.owner_player`, but **OCR is disabled for `bet` objects** (latency tradeoff — see enricher README), so no numeric value is available.
- Without the per-player bet values, the parser cannot derive the call amount or understand the betting round state.

**Fix approach (when latency budget allows):**
1. Enable `"bet": "ocr"` in the enricher processing map to read the numeric bet amounts.
2. Add `current_bet: integer | null` to `SeatState` in the hand state spec and parser output.
3. Populate `current_bet` by matching `bet.spatial_info.owner_player` to a seat, same pattern as opponent stacks.
4. Use `current_bet` values to derive `amount_to_call` more precisely (max posted bet minus hero's current bet).

**⚠️ Blocked by OCR latency tradeoff** — do not enable until the latency budget (Issue #1 resolution) allows it. See enricher README for context.

**Files to modify:**
- `poker-vision-detection-enricher/api.py` — add `"bet": "ocr"` to processing map
- `HAND_STATE_INTERFACE_SPEC_V2.md` — add `current_bet` to `SeatState`
- `poker-vision-hand-state-parser/hand_state_parser.py` — populate `current_bet` per seat
- `tests/fixtures/e2e/*/expected_*.json` — re-capture affected fixtures

**Dependencies:** Issue #1 (latency), Issue #22 (amount_to_call — related fix).

**Verification:**
1. Feed `screenshot_preflop_4` through the pipeline.
2. Confirm `seats[SB].current_bet == 10`, `seats[BB].current_bet == 20`, `seats[BTN].current_bet == 0`.

---

## Issue #26 — `is_all_in` emitted as `null` instead of `false` when stack is known

**Priority:** P3 | **Severity:** Low | **Tier:** 4

**Description:** All non-all-in seats (hero and opponents) emit `is_all_in: null`. When a player has a detected positive stack, they are definitively not all-in, and `false` would be more accurate and useful to consumers.

**Root cause:**
- The parser emits `is_all_in = None` (→ JSON `null`) for any seat without an all-in signal, including seats with a known positive stack.
- The spec states `is_all_in: boolean | null` and "when no all-in signal is present: null" — the current behaviour matches the spec, but the spec is overly conservative.
- When `seat_stack` is a known positive integer, `is_all_in = False` is the correct inference.

**Fix approach:**
1. In the seat assembly loop, set `is_all_in = False` when the stack is a known positive integer and no all-in signal is present:
   `seat_is_all_in = True if in_all_in else (False if seat_stack else None)`
2. Update `HAND_STATE_INTERFACE_SPEC_V2.md` to reflect this rule.

**Files to modify:**
- `poker-vision-hand-state-parser/hand_state_parser.py` — seat assembly loop
- `HAND_STATE_INTERFACE_SPEC_V2.md` — update `is_all_in` semantics
- `tests/fixtures/e2e/*/expected_*.json` — re-capture affected fixtures

**Dependencies:** None. Low-risk, independent change.

**Verification:**
1. Confirm all seats with detected positive stacks show `is_all_in: false`.
2. Confirm seats with unknown stacks still show `is_all_in: null`.

---

## Issue #27 — `position` field name should be `hero_position`

**Priority:** P3 | **Severity:** Low | **Tier:** 4

**Description:** The hand state top-level field `position` (which holds the hero's seat label e.g. `"BTN"`) should be named `hero_position` for clarity, since `position` is ambiguous.

**Root cause / context:**
- `position` is explicitly documented in `HAND_STATE_INTERFACE_SPEC_V2.md` as a **legacy alias** for `hero_seat`. Both already exist in the output with the same value.
- Renaming `position` to `hero_position` is a **breaking change** — it would require updating the spec, the parser output dict, the decision engine input parsing, all existing tests, and all fixture files.
- Since `hero_seat` already provides the same information clearly, the practical value of this rename is low.

**Fix approach (if pursued):**
1. Add `hero_position` as an additional alias alongside `position` and `hero_seat` (non-breaking).
2. In a later release, deprecate `position` and remove it once all callers migrate.
3. Update `HAND_STATE_INTERFACE_SPEC_V2.md` and parser output.

**Files to modify:**
- `poker-vision-hand-state-parser/hand_state_parser.py`
- `HAND_STATE_INTERFACE_SPEC_V2.md`
- `poker-vision-decision-engine/api.py` (reads `position`)

**Dependencies:** Coordinate with all callers — breaking change.

**Recommendation:** Defer. `hero_seat` already exists and is unambiguous. Only pursue if a caller explicitly needs `hero_position`.

**Verification:**
1. Confirm `hero_position` emitted alongside existing `position` / `hero_seat`.
2. All e2e tests pass.

---

## Appendix: Migrated from README

The following items were previously tracked in `README.md` § "MVP P0 Status" → "Remaining gaps — by impact" and have been migrated here:

1. **Action history is usually empty** → Migrated to Issue #6 (strong hands folded/checked). The root cause is the same: no reliable enrichment path produces `action`/`player` fields, causing situation misclassification.
2. **Position defaults to BTN when `player_me` is not detected** → Retained in README as a detection-quality dependency (not a UAT testing issue). See `README.md` § "MVP P0 Status" table row for Hand state parser.
3. **SPECULATIVE hands have no dedicated rules** → Retained in README as a decision-engine enhancement (not a UAT testing issue). See `README.md` § "MVP P0 Status" table row for Decision engine.
