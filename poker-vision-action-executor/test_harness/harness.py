"""
test_harness/harness.py — Native Win32 poker action-bar simulator.

Creates a real Windows window containing native Win32 Button and Edit controls
that mirror a poker client's action bar.  Because the controls are genuine
Win32 controls the executor's ``EnumChildWindows``-based discovery works
against this harness exactly as it would against a real poker client.

Why native Win32 (not tkinter)?
--------------------------------
tkinter draws widgets inside a Tcl/Tk canvas and does NOT create separate
Win32 HWNDs for individual buttons.  ``EnumChildWindows`` would therefore find
no ``Button``-class children.  The native Win32 approach (via ctypes) creates
real ``HWND``s with class ``"Button"`` and ``"Edit"`` that the executor can
discover.

Layout
------
    [Fold]  [Call 50]  [Check]  [__bet__]  [Raise To]

Usage — standalone (manual visual test):
    python test_harness/harness.py

Usage — programmatic (from integration tests):
    from test_harness.harness import HarnessState, run_harness_in_thread

    state = HarnessState()
    harness, thread = run_harness_in_thread(state, auto_close_after=5.0)
    state.ready.wait(timeout=3)
    # ... drive the executor ...
    state.done.wait(timeout=6)
    print(state.clicked)          # e.g. ["Fold"]
    print(state.last_bet_value)   # e.g. "300"
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import threading
import time
from dataclasses import dataclass, field

# ── Win32 constants ────────────────────────────────────────────────────────────

WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_VISIBLE = 0x10000000
WS_CHILD = 0x40000000
WS_TABSTOP = 0x00010000
WS_BORDER = 0x00800000
BS_PUSHBUTTON = 0x00000000
ES_LEFT = 0x0000
PM_REMOVE = 0x0001
SW_SHOW = 5
WM_DESTROY = 0x0002
WM_COMMAND = 0x0111
WM_QUIT = 0x0012
WM_SETFONT = 0x0030
BN_CLICKED = 0
EN_CHANGE = 0x0300

# Control IDs
_ID_FOLD = 101
_ID_CALL = 102
_ID_CHECK = 103
_ID_RAISE = 104
_ID_BET_ENTRY = 105

# Maps control ID → button label (must match config.yaml variants)
BUTTON_ID_TO_LABEL: dict[int, str] = {
    _ID_FOLD: "Fold",
    _ID_CALL: "Call 50",
    _ID_CHECK: "Check",
    _ID_RAISE: "Raise To",
}

HARNESS_WINDOW_TITLE = "PokerTestHarness"
_HARNESS_CLASS = "PokerHarnessWndClass"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ── Win32 structures ───────────────────────────────────────────────────────────

WNDPROCTYPE = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.wintypes.HWND,
    ctypes.c_uint,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROCTYPE),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", ctypes.wintypes.HMODULE),
        ("hIcon", ctypes.wintypes.HANDLE),
        ("hCursor", ctypes.wintypes.HANDLE),
        ("hbrBackground", ctypes.wintypes.HANDLE),
        ("lpszMenuName", ctypes.wintypes.LPCWSTR),
        ("lpszClassName", ctypes.wintypes.LPCWSTR),
        ("hIconSm", ctypes.wintypes.HANDLE),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.wintypes.HWND),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.wintypes.WPARAM),
        ("lParam", ctypes.wintypes.LPARAM),
        ("time", ctypes.wintypes.DWORD),
        ("pt", ctypes.wintypes.POINT),
    ]


# ── Harness state ──────────────────────────────────────────────────────────────


@dataclass
class HarnessState:
    """Captures everything the harness records during a test run.

    Attributes:
        clicked:         Labels of buttons that were clicked, in order.
        last_bet_value:  Most recent text content of the bet-size Edit control.
        ready:           Set once the window is visible and accepting messages.
        done:            Set when the window has been destroyed.
    """

    clicked: list[str] = field(default_factory=list)
    last_bet_value: str = ""
    ready: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    # Internal: hwnd of the Edit control so tests can read its value directly.
    _edit_hwnd: int = 0


# ── WndProc (module-level so ctypes keeps a stable reference) ──────────────────

# Module-level mapping from hwnd → HarnessState so the WndProc can update state.
_hwnd_to_state: dict[int, HarnessState] = {}


@WNDPROCTYPE
def _wnd_proc(
    hwnd: int, msg: int, wparam: int, lparam: int
) -> int:  # pragma: no cover
    state = _hwnd_to_state.get(hwnd)

    if msg == WM_COMMAND:
        ctrl_id = wparam & 0xFFFF
        notification = (wparam >> 16) & 0xFFFF

        if notification == BN_CLICKED and ctrl_id in BUTTON_ID_TO_LABEL:
            label = BUTTON_ID_TO_LABEL[ctrl_id]
            if state:
                state.clicked.append(label)

        elif notification == EN_CHANGE and ctrl_id == _ID_BET_ENTRY and state:
            # Read current text from the Edit control.
            length = user32.GetWindowTextLengthW(lparam) + 1
            buf = ctypes.create_unicode_buffer(length)
            user32.GetWindowTextW(lparam, buf, length)
            state.last_bet_value = buf.value

    elif msg == WM_DESTROY:
        if state:
            state.done.set()
        _hwnd_to_state.pop(hwnd, None)
        user32.PostQuitMessage(0)

    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


# ── Harness runner ─────────────────────────────────────────────────────────────


def _create_control(
    cls: str,
    label: str,
    style: int,
    x: int,
    y: int,
    w: int,
    h: int,
    parent: int,
    ctrl_id: int,
    hInstance: int,
) -> int:
    """Helper to create a child Win32 control and return its hwnd."""
    hwnd = user32.CreateWindowExW(
        0,
        cls,
        label,
        style,
        x,
        y,
        w,
        h,
        parent,
        ctypes.cast(ctypes.c_void_p(ctrl_id), ctypes.wintypes.HMENU),
        hInstance,
        None,
    )
    return hwnd


def run_harness(state: HarnessState, auto_close_after: float = 10.0) -> None:
    """Create the harness window and run the Win32 message loop (blocking).

    This function is intended to run in a dedicated thread.  It signals
    ``state.ready`` once the window is visible, and ``state.done`` when the
    window is destroyed.

    Args:
        state:             HarnessState that records interactions.
        auto_close_after:  Seconds before the window closes automatically.
    """
    hInstance = kernel32.GetModuleHandleW(None)

    # Register window class (ignore error if already registered from a
    # previous test run in the same process).
    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.lpfnWndProc = _wnd_proc
    wc.hInstance = hInstance
    wc.lpszClassName = _HARNESS_CLASS
    # COLOR_BTNFACE + 1 = 16 gives the standard dialog background colour.
    wc.hbrBackground = ctypes.cast(ctypes.c_void_p(16), ctypes.wintypes.HANDLE)
    user32.RegisterClassExW(ctypes.byref(wc))

    # Main window
    hwnd = user32.CreateWindowExW(
        0,
        _HARNESS_CLASS,
        HARNESS_WINDOW_TITLE,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        300, 300, 640, 110,
        None, None, hInstance, None,
    )

    _hwnd_to_state[hwnd] = state

    # Action buttons
    btn_style = WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_TABSTOP
    x = 10
    for ctrl_id, label in BUTTON_ID_TO_LABEL.items():
        _create_control("Button", label, btn_style, x, 15, 110, 32, hwnd, ctrl_id, hInstance)
        x += 118

    # Bet-size Edit control
    edit_style = WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT
    edit_hwnd = _create_control(
        "Edit", "0", edit_style, x, 15, 80, 32, hwnd, _ID_BET_ENTRY, hInstance
    )
    state._edit_hwnd = edit_hwnd

    user32.ShowWindow(hwnd, SW_SHOW)
    user32.UpdateWindow(hwnd)

    state.ready.set()

    # Message loop with auto-close timeout
    msg = MSG()
    deadline = time.monotonic() + auto_close_after

    while True:
        if time.monotonic() >= deadline:
            user32.DestroyWindow(hwnd)
            # Drain remaining messages after DestroyWindow
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == WM_QUIT:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            break

        if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            if msg.message == WM_QUIT:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        else:
            time.sleep(0.005)  # yield CPU while idle

    state.done.set()


def run_harness_in_thread(
    state: HarnessState | None = None,
    auto_close_after: float = 10.0,
) -> tuple[threading.Thread, HarnessState]:
    """Start the harness window in a background thread.

    Args:
        state:             HarnessState to use; a new one is created if None.
        auto_close_after:  Seconds before the window auto-closes.

    Returns:
        (thread, state) — call ``state.ready.wait()`` before driving the
        executor, and ``state.done.wait()`` to block until the window closes.
    """
    if state is None:
        state = HarnessState()
    thread = threading.Thread(
        target=run_harness,
        args=(state, auto_close_after),
        daemon=True,
        name="HarnessThread",
    )
    thread.start()
    return thread, state


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    s = HarnessState()
    t, _ = run_harness_in_thread(s, auto_close_after=30.0)
    s.ready.wait(timeout=5)
    print(f"Harness running: title={HARNESS_WINDOW_TITLE!r}")
    print("Buttons: Fold | Call 50 | Check | [bet box] | Raise To")
    print("Window will auto-close in 30 seconds, or close it manually.")
    s.done.wait()
    print(f"Clicks recorded : {s.clicked}")
    print(f"Last bet value  : {s.last_bet_value!r}")
    sys.exit(0)
