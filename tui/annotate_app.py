"""Textual TUI for the annotate stage (taxonomy §7.1 human review).

Wraps an ``AnnotationSession``: navigate clips, view in-terminal frame
thumbnails (24-bit half-block render) + metrics, set verdict / toggle canonical
flags / adjust dimension scores, edit reject reason & fix actions, and save.
A browser handoff writes an HTML contact sheet of all sampled frames.

Runs headlessly under ``App.run_test()`` for CI (see tests/test_annotate.py).
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, ListItem, ListView, Label, Static

from video_screener.taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS


def render_thumbnail(path: str, cols: int = 48) -> Text:
    """Render an image as 24-bit ANSI half-blocks (2x vertical resolution)."""
    try:
        from PIL import Image
    except Exception:  # pragma: no cover
        return Text("(PIL unavailable)")
    if not path or not Path(path).exists():
        return Text("(no frame)")
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return Text("(unreadable frame)")
    w, h = im.size
    rows = max(1, int(cols * (h / w) * 0.5))
    im = im.resize((cols, rows * 2))
    px = im.load()
    out = Text()
    for y in range(rows):
        for x in range(cols):
            r1, g1, b1 = px[x, 2 * y]
            r2, g2, b2 = px[x, 2 * y + 1]
            out.append("▀", style=f"rgb({r1},{g1},{b1}) on rgb({r2},{g2},{b2})")
        out.append("\n")
    return out


def build_contact_sheet(record: dict, frames: list[dict], dest: Path) -> Path:
    """Write a simple HTML contact sheet of a clip's sampled frames."""
    imgs = "".join(
        f'<figure><img src="{Path(f["path"]).resolve().as_uri()}">'
        f'<figcaption>t={f.get("time_sec")}s'
        f'{" (scene)" if f.get("is_scene_change") else ""}</figcaption></figure>'
        for f in frames
    )
    html = f"""<!doctype html><meta charset=utf-8>
<title>{record.get('asset_id')} frames</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee}}
figure{{display:inline-block;margin:6px;text-align:center}}
img{{width:200px;border:1px solid #333}}</style>
<h2>{record.get('asset_id')} — verdict {record.get('verdict')}</h2>
<div>{imgs}</div>"""
    dest.write_text(html)
    return dest


class AnnotateApp(App):
    CSS = """
    #left { width: 30; border: solid $accent; }
    #thumb { height: 18; }
    #info { height: 8; color: $text-muted; }
    #anno { border: solid $accent; }
    Input { dock: bottom; }
    """

    BINDINGS = [
        Binding("right,l", "next", "Next"),
        Binding("left,h", "prev", "Prev"),
        Binding("p", "verdict('PASS')", "PASS"),
        Binding("f", "verdict('FIX')", "FIX"),
        Binding("r", "verdict('REJECT')", "REJECT"),
        Binding("d", "cycle_dim", "Dim"),
        Binding("c", "score_up", "Score+"),
        Binding("z", "score_down", "Score-"),
        Binding("e", "edit_reason", "Reason"),
        Binding("g", "edit_fix", "Fix"),
        Binding("o", "open_browser", "Browser"),
        Binding("w,s", "save", "Save"),
        Binding("q", "quit_save", "Quit"),
    ]

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.active_dim = 0
        self.index_frames = self._load_frame_index()
        self.status = ""

    def _load_frame_index(self) -> dict[str, list[dict]]:
        from video_screener.utils.io import read_json

        idx_path = self.session.cfg.stage_dir("ingest") / "index.json"
        out: dict[str, list[dict]] = {}
        if idx_path.exists():
            for a in read_json(idx_path)["assets"]:
                out[a["asset_id"]] = a.get("sampled_frames", [])
        return out

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(id="left")
            with Vertical():
                yield Static(id="thumb")
                yield Static(id="info")
                yield Static(id="anno")
        yield Input(placeholder="edit field, Enter to commit", id="editor", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        lv = self.query_one("#left", ListView)
        for rec in self.session.records:
            lv.append(ListItem(Label(self._list_label(rec))))
        ed = self.query_one("#editor", Input)
        ed.display = False
        ed.disabled = True          # keep it out of the focus chain until editing
        lv.index = self.session.idx
        lv.focus()                  # so key bindings reach the app, not the Input
        self.refresh_all()

    # ---- rendering ------------------------------------------------------
    def _list_label(self, rec: dict) -> str:
        mark = {"PASS": "✓", "FIX": "~", "REJECT": "✗"}.get(rec.get("verdict"), "?")
        return f"{mark} {rec['asset_id']}"

    def refresh_all(self) -> None:
        rec = self.session.current()
        frames = self.index_frames.get(rec["asset_id"], [])
        # thumbnail: middle sampled frame
        thumb = self.query_one("#thumb", Static)
        if frames:
            mid = frames[len(frames) // 2]
            thumb.update(render_thumbnail(mid["path"]))
        else:
            thumb.update(Text("(no frames — decode failure)", style="red"))
        # info
        info = self.query_one("#info", Static)
        ftimes = ", ".join(str(f.get("time_sec")) for f in frames) or "none"
        ok, err = self.session.validate()
        valid = "[green]valid[/green]" if ok else f"[red]invalid:[/red] {err}"
        info.update(
            f"asset {self.session.idx + 1}/{len(self.session)}  "
            f"dur={rec.get('duration_sec')}s  frames@[{ftimes}]\n"
            f"schema: {valid}   {self.status}"
        )
        # annotation panel
        self.query_one("#anno", Static).update(self._anno_text(rec))
        # sync list highlight
        lv = self.query_one("#left", ListView)
        if lv.index != self.session.idx:
            lv.index = self.session.idx

    def _anno_text(self, rec: dict) -> str:
        scores = rec.get("scores", {})
        dim_lines = []
        for i, d in enumerate(DIMENSIONS):
            cur = "➤" if i == self.active_dim else " "
            val = scores.get(d, "·")
            dim_lines.append(f" {cur} {d}: {val}")
        flag_lines = []
        for i, fl in enumerate(HARD_FAIL_FLAGS, start=1):
            on = "●" if fl in rec.get("hard_fail_flags", []) else "○"
            flag_lines.append(f"  {i} {on} {fl}")
        return (
            f"[b]verdict[/b]: {rec.get('verdict')}   "
            f"needs_review: {rec.get('needs_human_review')}\n"
            f"[b]scores[/b] (d=cycle, ↑/↓ adjust):\n" + "\n".join(dim_lines) + "\n"
            f"[b]flags[/b] (press digit to toggle):\n" + "\n".join(flag_lines) + "\n"
            f"[b]reject_reason[/b]: {rec.get('reject_reason','')!r}\n"
            f"[b]fix_actions[/b]: {rec.get('fix_actions', [])}"
        )

    # ---- actions --------------------------------------------------------
    def action_next(self) -> None:
        self.session.go_next(); self.status = ""; self.refresh_all()

    def action_prev(self) -> None:
        self.session.go_prev(); self.status = ""; self.refresh_all()

    def action_verdict(self, verdict: str) -> None:
        self.session.set_verdict(verdict)
        if verdict == "REJECT" and not self.session.current().get("reject_reason"):
            self.session.set_reason("manual reject — needs reason")
        self._sync_list_label(); self.refresh_all()

    def action_cycle_dim(self) -> None:
        self.active_dim = (self.active_dim + 1) % len(DIMENSIONS)
        self.refresh_all()

    def _adjust_score(self, delta: int) -> None:
        d = DIMENSIONS[self.active_dim]
        cur = self.session.current().get("scores", {}).get(d)
        base = 2 if cur is None else cur
        self.session.set_score(d, max(0, min(4, base + delta)))
        self.refresh_all()

    def action_score_up(self) -> None:
        self._adjust_score(+1)

    def action_score_down(self) -> None:
        self._adjust_score(-1)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.session.go_to(event.list_view.index or 0)
        self.refresh_all()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        # keep session index in sync when the list highlight moves (up/down)
        if event.list_view.index is not None and event.list_view.index != self.session.idx:
            self.session.go_to(event.list_view.index)
            self.refresh_all()

    def on_key(self, event) -> None:
        # digit keys toggle flags 1..9
        if event.character and event.character in "123456789":
            i = int(event.character) - 1
            if i < len(HARD_FAIL_FLAGS):
                self.session.toggle_flag(HARD_FAIL_FLAGS[i])
                self._sync_list_label(); self.refresh_all()

    def _editor(self) -> Input:
        return self.query_one("#editor", Input)

    def action_edit_reason(self) -> None:
        ed = self._editor()
        ed.disabled = False
        ed.display = True
        ed.value = self.session.current().get("reject_reason", "")
        ed.placeholder = "reject_reason — Enter to commit"
        self._edit_target = "reason"
        ed.focus()

    def action_edit_fix(self) -> None:
        ed = self._editor()
        ed.disabled = False
        ed.display = True
        ed.value = ", ".join(self.session.current().get("fix_actions", []))
        ed.placeholder = "fix_actions (comma-separated) — Enter to commit"
        self._edit_target = "fix"
        ed.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        target = getattr(self, "_edit_target", None)
        if target == "reason":
            self.session.set_reason(event.value)
        elif target == "fix":
            acts = [a.strip() for a in event.value.split(",") if a.strip()]
            self.session.set_fix_actions(acts)
        ed = self._editor()
        ed.display = False
        ed.disabled = True
        self.query_one("#left", ListView).focus()
        self.refresh_all()

    def action_open_browser(self) -> None:
        rec = self.session.current()
        frames = self.index_frames.get(rec["asset_id"], [])
        dest = self.session.cfg.stage_dir("annotate") / f"{rec['asset_id']}_frames.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        build_contact_sheet(rec, frames, dest)
        try:
            webbrowser.open(dest.resolve().as_uri())
        except Exception:
            pass
        self.status = f"contact sheet: {dest}"
        self.refresh_all()

    def action_save(self) -> None:
        out, n_ok, errors = self.session.save(reviewer="human")
        self.status = f"saved {n_ok} to {out.name}" + (
            f"  [red]{len(errors)} invalid[/red]" if errors else ""
        )
        self.refresh_all()

    def action_quit_save(self) -> None:
        self.session.save(reviewer="human")
        self.exit()

    def _sync_list_label(self) -> None:
        lv = self.query_one("#left", ListView)
        item = lv.children[self.session.idx]
        label = item.query_one(Label)
        label.update(self._list_label(self.session.current()))
