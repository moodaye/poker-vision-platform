"""Manage all pipeline services.

Services managed:
    detection-enricher   http://127.0.0.1:5004/health
    hand-state-parser    http://127.0.0.1:5003/health
    decision-engine      http://127.0.0.1:5002/health
    orchestrator         http://127.0.0.1:5100/health

PIDs are saved to .services.pids so the stop command can shut them down.
Logs are written to logs/<service-name>.log.

Usage:
    uv run python manage_services.py start   # start any services not already running
    uv run python manage_services.py stop    # stop all services started by this script
    uv run python manage_services.py status  # show health of all services
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent

SERVICES = [
    {
        "name": "detection-enricher",
        "cmd": ["uv", "run", "python", "poker-vision-detection-enricher/api.py"],
        "health_url": "http://127.0.0.1:5004/health",
        "port": 5004,
    },
    {
        "name": "hand-state-parser",
        "cmd": ["uv", "run", "python", "poker-vision-hand-state-parser/api.py"],
        "health_url": "http://127.0.0.1:5003/health",
        "port": 5003,
    },
    {
        "name": "decision-engine",
        "cmd": ["uv", "run", "python", "poker-vision-decision-engine/api.py"],
        "health_url": "http://127.0.0.1:5002/health",
        "port": 5002,
    },
    {
        "name": "orchestrator",
        "cmd": ["uv", "run", "python", "orchestrator.py"],
        "health_url": "http://127.0.0.1:5100/health",
        "port": 5100,
    },
]

PID_FILE = REPO_ROOT / ".services.pids"
LOG_DIR = REPO_ROOT / "logs"
HEALTH_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_healthy(url: str) -> bool:
    try:
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def wait_for_health(url: str, name: str, timeout: int = HEALTH_TIMEOUT_SECONDS) -> bool:
    print(f"  Waiting for {name} to be ready", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_healthy(url):
            print(" ready")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(" timed out")
    return False


def load_pids() -> dict[str, int]:
    if PID_FILE.exists():
        try:
            return json.loads(PID_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_pids(pids: dict[str, int]) -> None:
    PID_FILE.write_text(json.dumps(pids, indent=2))


def kill_pid(name: str, pid: int) -> None:
    if sys.platform == "win32":
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"[{name}] stopped (PID {pid})")
        else:
            stderr = result.stderr.strip()
            if "not found" in stderr.lower() or "invalid" in stderr.lower():
                print(f"[{name}] was not running (PID {pid})")
            else:
                print(f"[{name}] could not stop PID {pid}: {stderr}")
    else:
        import os
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[{name}] stopped (PID {pid})")
        except ProcessLookupError:
            print(f"[{name}] was not running (PID {pid})")
        except PermissionError as exc:
            print(f"[{name}] permission denied stopping PID {pid}: {exc}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_start() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    pids = load_pids()
    all_ok = True

    for svc in SERVICES:
        name = svc["name"]
        health_url = svc["health_url"]

        if is_healthy(health_url):
            print(f"[{name}] already running on port {svc['port']}")
            continue

        print(f"[{name}] starting on port {svc['port']} ...")
        log_path = LOG_DIR / f"{name}.log"

        with log_path.open("w") as log_file:
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                svc["cmd"],
                cwd=str(REPO_ROOT),
                stdout=log_file,
                stderr=log_file,
                **kwargs,
            )

        pids[name] = proc.pid
        save_pids(pids)

        if not wait_for_health(health_url, name):
            print(
                f"  ERROR: {name} did not become healthy within {HEALTH_TIMEOUT_SECONDS}s"
            )
            print(f"         Check logs/{name}.log for details")
            all_ok = False

    print()
    if all_ok:
        print("All services are running. You can now use the poker pipeline.py")
    else:
        print("One or more services failed to start. Check the log files in logs/")
        sys.exit(1)


def cmd_stop() -> None:
    if not PID_FILE.exists():
        print("No .services.pids file found — nothing to stop.")
        print("If services are running, stop them manually or restart your terminal.")
        return

    try:
        pids: dict[str, int] = json.loads(PID_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not read .services.pids: {exc}")
        sys.exit(1)

    if not pids:
        print("No services recorded in .services.pids.")
        PID_FILE.unlink(missing_ok=True)
        return

    for name, pid in pids.items():
        kill_pid(name, pid)

    PID_FILE.unlink(missing_ok=True)
    print("\nDone.")


def cmd_status() -> None:
    any_down = False
    for svc in SERVICES:
        name = svc["name"]
        url = svc["health_url"]
        port = svc["port"]
        if is_healthy(url):
            print(f"[{name}]  up    port {port}")
        else:
            print(f"[{name}]  DOWN  port {port}")
            any_down = True
    if any_down:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {"start": cmd_start, "stop": cmd_stop, "status": cmd_status}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python manage_services.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
