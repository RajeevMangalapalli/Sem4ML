"""Persistence helper for the SpaceX ML project.

When wired into a script this module:
  * tees ``stdout`` so every ``print`` is shown *and* captured;
  * patches ``plt.show`` so every open figure is saved to ``src/output/``
    as a PNG *before* being displayed on screen;
  * writes the captured prints to ``<stem>-prints.md``;
  * exposes :func:`save_plot_data` to dump the values behind each plot
    into a human-readable ``<stem>-plot-data.md`` file.

Usage at the top of a script::

    from pathlib import Path
    import persist
    STEM = Path(__file__).stem
    persist.begin(STEM)

    # ... existing script ...

    persist.save_plot_data(STEM, "...markdown...")
    persist.end()
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "src" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_orig_show = plt.show
_state: dict = {"stem": None, "log": None, "tee": None, "plot_idx": 0}


class _Tee:
    """Stream that writes to both the real console and an in-memory log."""

    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, data):
        self.console.write(data)
        self.log.write(data)

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self.console, name)


def begin(stem: str) -> None:
    """Start capturing prints and patch ``plt.show`` to save figures."""
    _state["stem"] = stem
    _state["log"] = io.StringIO()
    _state["tee"] = _Tee(sys.stdout, _state["log"])
    sys.stdout = _state["tee"]
    _state["plot_idx"] = 0

    def _show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            idx = _state["plot_idx"]
            _state["plot_idx"] += 1
            fname = stem if idx == 0 else f"{stem}_{idx}"
            fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
        _orig_show(*args, **kwargs)

    plt.show = _show


def end() -> None:
    """Stop capturing, restore ``plt.show`` and write the prints markdown."""
    tee = _state.get("tee")
    if tee is not None:
        sys.stdout = tee.console
    plt.show = _orig_show

    stem = _state["stem"]
    content = _state["log"].getvalue() if _state["log"] is not None else ""
    header = f"# Prints from `{stem}.py`\n\n"
    if content.strip():
        body = f"```\n{content}\n```\n"
    else:
        body = "_No output was printed by this script._\n"
    (OUTPUT_DIR / f"{stem}-prints.md").write_text(header + body)


def save_plot_data(stem: str, markdown: str) -> None:
    """Write the values behind the plots to ``<stem>-plot-data.md``."""
    (OUTPUT_DIR / f"{stem}-plot-data.md").write_text(markdown)


def md_table(headers, rows) -> str:
    """Render a markdown table from ``headers`` and a list of row tuples."""
    out = []
    out.append("| " + " | ".join(str(h) for h in headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"
