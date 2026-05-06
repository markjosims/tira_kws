#!/usr/bin/env python3
"""TUI for merging misspelled word variants via Levenshtein edit distance.

Usage:
    python merge_words.py <input.txt> [output.tsv]

Keys (main list):
    ↑/↓        Navigate words
    PgUp/PgDn   Scroll fast
    Enter       Select word as merge anchor
    /           Search/filter words
    q           Quit

Keys (merge view):
    ↑/↓        Navigate candidate list (sorted by edit distance)
    Space       Toggle word selection
    a           Select all from top through cursor
    u           Unselect all
    e           Edit the canonical (output) spelling
    Enter       Confirm merge and write to TSV
    Esc         Cancel and go back
"""

import curses
import sys
import os
from Levenshtein import distance as levenshtein
from tira_kws.constants import CAPSTONE_KWS_WORDLIST


# ── Application ──────────────────────────────────────────────────────────────


class MergeApp:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.words: list[str] = []
        self.unmerged: set[str] = set()
        self.merge_count = 0
        self._load()

    def _load(self):
        with open(self.input_path, "r") as f:
            self.words = [line.strip() for line in f if line.strip()]
        self.unmerged = set(self.words)

        # Resume: honour already-written merges
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        for v in parts[1:]:
                            self.unmerged.discard(v)
                        self.merge_count += 1

    def save_merge(self, canonical: str, variants: list[str]):
        with open(self.output_path, "a") as f:
            f.write("\t".join([canonical] + variants) + "\n")
        for v in variants:
            self.unmerged.discard(v)
        self.merge_count += 1

    def sorted_unmerged(self) -> list[str]:
        return sorted(self.unmerged, key=str.lower)

    def scored_against(self, anchor: str) -> list[tuple[str, int]]:
        pairs = [(w, levenshtein(anchor, w)) for w in self.unmerged]
        pairs.sort(key=lambda p: (p[1], p[0].lower()))
        return pairs

    # ── curses entry ─────────────────────────────────────────────────────

    def run(self, stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)  # cursor row
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # selected
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # distance
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_GREEN)  # header bar
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # status bar
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # merge count
        curses.init_pair(7, curses.COLOR_CYAN, -1)  # info text

        while True:
            if self._main_view(stdscr) is None:
                break

    # ── Main list view ───────────────────────────────────────────────────

    def _main_view(self, stdscr) -> str | None:
        cursor = 0
        scroll = 0
        search = ""
        searching = False

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            words = self.sorted_unmerged()
            if search:
                words = [wd for wd in words if search.lower() in wd.lower()]

            # ── header bar ──
            hdr = f" WORD MERGER — {len(self.unmerged)} unmerged / {len(self.words)} total "
            self._bar(stdscr, 0, hdr, w, 4)

            # ── merge count ──
            row_off = 1
            if self.merge_count:
                info = f" {self.merge_count} merge(s) saved → {self.output_path}"
                stdscr.addnstr(row_off, 0, info, w - 1, curses.color_pair(6))
                row_off = 2

            # ── status bar ──
            if searching:
                status = f" /:search  ‹{search}›_  Enter:done  Esc:cancel "
            else:
                status = " ↑↓:navigate  Enter:select anchor  /:search  q:quit "
            self._bar(stdscr, h - 1, status, w, 5)

            # ── word list ──
            list_top = row_off + 1
            list_h = h - list_top - 1
            if list_h < 1:
                stdscr.refresh()
                stdscr.getch()
                continue

            cursor = max(0, min(cursor, len(words) - 1))
            scroll = self._fix_scroll(scroll, cursor, list_h)

            for i in range(list_h):
                idx = scroll + i
                if idx >= len(words):
                    break
                r = list_top + i
                word = words[idx]
                if idx == cursor:
                    self._bar(stdscr, r, f" ▸ {word}", w, 1)
                else:
                    stdscr.addnstr(r, 0, f"   {word}", w - 1)

            if not words:
                msg = "No matches." if search else "All words merged!"
                stdscr.addnstr(list_top + 1, 3, msg, w - 4, curses.color_pair(7))

            stdscr.refresh()
            key = stdscr.getch()

            # ── search mode input ──
            if searching:
                if key == 27:  # Esc
                    searching, search = False, ""
                    cursor = scroll = 0
                elif key in (10, 13):  # Enter
                    searching = False
                    cursor = scroll = 0
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    search = search[:-1]
                    cursor = scroll = 0
                elif 32 <= key <= 126:
                    search += chr(key)
                    cursor = scroll = 0
                continue

            # ── normal mode input ──
            if key == ord("q"):
                return None
            elif key == ord("/"):
                searching, search = True, ""
                cursor = scroll = 0
            elif key == curses.KEY_UP:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_DOWN:
                cursor = min(len(words) - 1, cursor + 1)
            elif key == curses.KEY_PPAGE:
                cursor = max(0, cursor - list_h)
            elif key == curses.KEY_NPAGE:
                cursor = min(len(words) - 1, cursor + list_h)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = max(0, len(words) - 1)
            elif key in (10, 13) and words:
                anchor = words[cursor]
                self._merge_view(stdscr, anchor)
                cursor = scroll = 0
                search = ""

    # ── Merge / distance view ────────────────────────────────────────────

    def _merge_view(self, stdscr, anchor: str):
        scored = self.scored_against(anchor)
        selected: set[str] = {anchor}
        canonical = anchor
        cursor = 0
        scroll = 0
        editing = False

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            # ── header ──
            hdr = f' MERGE → "{canonical}"   ({len(selected)} selected) '
            self._bar(stdscr, 0, hdr, w, 4)

            # ── status bar ──
            if editing:
                status = f" canonical: {canonical}_   Enter:done  Esc:revert "
            else:
                status = " Space:toggle  a:select-to-cursor  u:unselect-all  e:edit-name  Enter:confirm  Esc:back "
            self._bar(stdscr, h - 1, status, w, 5)

            # ── list ──
            list_top = 2
            list_h = h - list_top - 1
            if list_h < 1:
                stdscr.refresh()
                stdscr.getch()
                continue

            cursor = max(0, min(cursor, len(scored) - 1))
            scroll = self._fix_scroll(scroll, cursor, list_h)

            dist_col = max(4, w - 10)

            for i in range(list_h):
                idx = scroll + i
                if idx >= len(scored):
                    break
                r = list_top + i
                word, dist = scored[idx]
                mark = "■" if word in selected else "□"
                label = f" {mark} {word}"

                if idx == cursor:
                    self._bar(stdscr, r, label.ljust(dist_col), w, 1)
                elif word in selected:
                    stdscr.addnstr(r, 0, label, dist_col, curses.color_pair(2))
                else:
                    stdscr.addnstr(r, 0, label, dist_col)

                dtag = f"d={dist}"
                try:
                    stdscr.addnstr(
                        r, dist_col, dtag, w - dist_col - 1, curses.color_pair(3)
                    )
                except curses.error:
                    pass

            stdscr.refresh()
            key = stdscr.getch()

            # ── editing canonical ──
            if editing:
                if key == 27:
                    editing = False
                    canonical = anchor
                elif key in (10, 13):
                    editing = False
                    canonical = canonical.strip() or anchor
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    canonical = canonical[:-1]
                elif 32 <= key <= 126:
                    canonical += chr(key)
                continue

            # ── normal merge-view input ──
            if key == 27:  # Esc → back
                return
            elif key == ord(" "):
                word = scored[cursor][0]
                selected.symmetric_difference_update({word})
                cursor = min(len(scored) - 1, cursor + 1)
            elif key == ord("a"):
                for j in range(cursor + 1):
                    selected.add(scored[j][0])
            elif key == ord("u"):
                selected.clear()
            elif key == ord("e"):
                editing = True
                canonical = anchor
            elif key == curses.KEY_UP:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_DOWN:
                cursor = min(len(scored) - 1, cursor + 1)
            elif key == curses.KEY_PPAGE:
                cursor = max(0, cursor - list_h)
            elif key == curses.KEY_NPAGE:
                cursor = min(len(scored) - 1, cursor + list_h)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = max(0, len(scored) - 1)
            elif key in (10, 13):
                if selected:
                    self.save_merge(canonical, sorted(selected, key=str.lower))
                return

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _bar(stdscr, row: int, text: str, width: int, pair: int):
        try:
            stdscr.addnstr(
                row, 0, text.ljust(width), width - 1, curses.color_pair(pair)
            )
        except curses.error:
            pass

    @staticmethod
    def _fix_scroll(scroll: int, cursor: int, page: int) -> int:
        if cursor < scroll:
            return cursor
        if cursor >= scroll + page:
            return cursor - page + 1
        return scroll


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        input_path = str(CAPSTONE_KWS_WORDLIST)
        output_path = input_path.removesuffix(".txt") + "_merged.tsv"
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "merged.tsv"

    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' not found", file=sys.stderr)
        sys.exit(1)

    app = MergeApp(input_path, output_path)
    curses.wrapper(app.run)


if __name__ == "__main__":
    main()
