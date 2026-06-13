import argparse
import ctypes
import ctypes.wintypes


user32 = ctypes.windll.user32


def _get_window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buf = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buf, length)
    return buf.value


def _get_class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _enum_child_windows(parent_hwnd: int) -> list[int]:
    children: list[int] = []

    EnumChildProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    @EnumChildProc
    def callback(hwnd: int, _lParam: int) -> bool:
        children.append(hwnd)
        return True

    user32.EnumChildWindows(parent_hwnd, callback, 0)
    return children


def find_top_window(title_hints: list[str]) -> int | None:
    if not title_hints:
        return None

    found: list[int] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    @EnumWindowsProc
    def callback(hwnd: int, _lParam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_text(hwnd)
        for hint in title_hints:
            if hint.lower() in title.lower():
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else None


def walk_controls(hwnd: int, depth: int = 0) -> list[tuple[int, int, str, str]]:
    rows: list[tuple[int, int, str, str]] = []
    for child in _enum_child_windows(hwnd):
        cls = _get_class_name(child)
        text = _get_window_text(child).strip()
        rows.append((child, depth, cls, text))
        rows.extend(walk_controls(child, depth + 1))
    return rows


def print_controls(
    controls: list[tuple[int, int, str, str]],
    class_filter: list[str] | None = None,
    label_filter: str | None = None,
) -> None:
    for hwnd, depth, cls, text in controls:
        if class_filter and not any(filter_ in cls.lower() for filter_ in class_filter):
            continue
        if label_filter and label_filter.lower() not in text.lower():
            continue
        short_text = text[:10]
        indent = "  " * depth
        print(f"{indent}hwnd={hwnd} class={cls!r} text={short_text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump window controls for a poker client window."
    )
    parser.add_argument(
        "--title-hint",
        required=True,
        help="Substring to match in the top-level window title",
    )
    parser.add_argument(
        "--class-filter",
        help="Comma-separated control classes to show, e.g. Button,Edit,Static",
    )
    parser.add_argument(
        "--label-filter",
        help="Only print controls whose text contains this substring",
    )
    args = parser.parse_args()

    hints = [args.title_hint]
    hwnd = find_top_window(hints)
    if hwnd is None:
        print(f"No visible top-level window found matching hint: {args.title_hint}")
        return

    title = _get_window_text(hwnd)
    print(f"Found window hwnd={hwnd} title={title!r}\n")

    class_filter = None
    if args.class_filter:
        class_filter = [c.strip().lower() for c in args.class_filter.split(",")]

    controls = walk_controls(hwnd)
    print_controls(controls, class_filter=class_filter, label_filter=args.label_filter)


if __name__ == "__main__":
    main()