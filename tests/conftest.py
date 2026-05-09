from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Expose sub-package source roots so integration tests can import from both
# poker-vision-hand-state-parser and poker-vision-decision-engine without
# needing the services to be running.
for _subpackage in (
    "poker-vision-hand-state-parser",
    "poker-vision-decision-engine",
):
    _path = str(ROOT / _subpackage)
    if _path not in sys.path:
        sys.path.insert(0, _path)
