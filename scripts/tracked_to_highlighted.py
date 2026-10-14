"""Turn a tracked-changes docx into a clean docx with every insertion highlighted.

RSC's editorial office returned the R1 "Highlighted Manuscript" (Word Compare output from
build_marked_changes.py, real tracked changes) on 2026-10-14 and asked for the file "without track
changes, showing only the highlighted edits". This converts the file that was uploaded rather than
rebuilding it, so the highlighted copy says exactly what the tracked one said.

Every insertion, moved-to block and inserted table cell is given a yellow highlight, then every
revision is accepted: deleted and moved-from text disappears, inserted text stays highlighted, and
the result carries no revisions. A deletion with nothing inserted in its place leaves no mark; the
response letter is where removals are accounted for.

Check: with --clean, the result is compared against the clean file it must equal (Word Compare,
formatting ignored) and the build fails unless the compare finds no difference. Run from the repo
root (Windows, Microsoft Word installed):

    python scripts/tracked_to_highlighted.py TRACKED.docx OUT.docx [--clean CLEAN.docx] [--pdf]
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_marked_changes import (  # noqa: E402  (same Word lifecycle as the tracked build)
    MSO_AUTOMATION_SECURITY_FORCE_DISABLE, REVISION_TYPES, WD_ALERTS_NONE,
    WD_COMPARE_DESTINATION_NEW, WD_EXPORT_FORMAT_PDF, WD_FORMAT_DOCUMENT_DEFAULT,
    WD_GRANULARITY_WORD_LEVEL, log, open_own_process, reap)

WD_YELLOW = 7
# insert, moved to, cell insertion: the text a reader of the revision has not seen before.
HIGHLIGHTED_TYPES = {1, 15, 16}


def convert(word, tracked: Path, out: Path, clean: Path | None, pdf: Path | None) -> dict:
    doc = word.Documents.Open(str(tracked), False, True, False)   # confirm, read-only, recent
    try:
        doc.TrackRevisions = False
        before = Counter(REVISION_TYPES.get(r.Type, str(r.Type)) for r in doc.Revisions)
        spans = [(r.Range.Start, r.Range.End) for r in doc.Revisions if r.Type in HIGHLIGHTED_TYPES]
        n_before = doc.Revisions.Count
        for start, end in spans:
            doc.Range(start, end).HighlightColorIndex = WD_YELLOW
        if doc.Revisions.Count != n_before:
            raise RuntimeError(f"highlighting changed the revision count ({n_before} -> "
                               f"{doc.Revisions.Count}): track changes was not off")
        doc.Revisions.AcceptAll()
        if doc.Revisions.Count:
            raise RuntimeError(f"{doc.Revisions.Count} revisions survive AcceptAll")
        if doc.Comments.Count:
            raise RuntimeError(f"{doc.Comments.Count} comments in the document")
        out.parent.mkdir(parents=True, exist_ok=True)
        doc.SaveAs2(str(out), WD_FORMAT_DOCUMENT_DEFAULT)
        log(f"saved {out.name}")
        if pdf is not None:
            doc.ExportAsFixedFormat(str(pdf), WD_EXPORT_FORMAT_PDF)
            log(f"exported {pdf.name}")
        summary = {"revisions": n_before, "by_type": dict(before), "highlighted": len(spans)}
    finally:
        doc.Close(0)
    if clean is not None:
        summary["diff_vs_clean"] = diff_count(word, clean, out)
    return summary


def diff_count(word, clean: Path, highlighted: Path) -> list[str]:
    """Word Compare of the clean file against the result, formatting ignored: what differs."""
    a = word.Documents.Open(str(clean), False, True, False)
    b = word.Documents.Open(str(highlighted), False, True, False)
    # Original, Revised, Destination, Granularity, CompareFormatting, CaseChanges, Whitespace,
    # Tables, Headers, Footnotes, Textboxes, Fields, Comments, Moves, RevisedAuthor, IgnoreWarnings.
    cmp = word.CompareDocuments(a, b, WD_COMPARE_DESTINATION_NEW, WD_GRANULARITY_WORD_LEVEL,
                                False, True, True, True, True, True, True, True, True, True,
                                "check", True)
    a.Close(0)
    b.Close(0)
    try:
        return [f"{REVISION_TYPES.get(r.Type, r.Type)}: {r.Range.Text[:80]!r}"
                for r in cmp.Revisions]
    finally:
        cmp.Close(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracked", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clean", type=Path, help="clean file the result must equal in text")
    ap.add_argument("--pdf", action="store_true", help="also export OUT as .pdf")
    ap.add_argument("--word-timeout", type=float, default=900.0)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    tracked, out = args.tracked.resolve(), args.out.resolve()
    clean = args.clean.resolve() if args.clean else None
    pdf = out.with_suffix(".pdf") if args.pdf else None
    if out == tracked:
        print("REFUSED: OUT would overwrite the tracked file")
        return 2

    import pythoncom
    import win32api
    import win32com.client

    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    handle = job = watchdog = None
    try:
        word.Visible = False
        word.DisplayAlerts = WD_ALERTS_NONE
        word.AutomationSecurity = MSO_AUTOMATION_SECURITY_FORCE_DISABLE
        handle, pid, job = open_own_process(word)
        watchdog = threading.Timer(args.word_timeout, lambda: win32api.TerminateProcess(handle, 1))
        watchdog.daemon = True
        watchdog.start()
        s = convert(word, tracked, out, clean, pdf)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        try:
            word.Quit(0)
        except Exception:
            pass
        word = None
        pythoncom.CoUninitialize()
        if handle is not None:
            log(reap(handle, pid))
        if job is not None:
            job.Close()

    print(f"{tracked.name}: {s['revisions']} revisions ("
          + ", ".join(f"{k} {v}" for k, v in sorted(s["by_type"].items())) + ")")
    print(f"  highlighted {s['highlighted']} insert / moved-to / cell-insert revisions; all accepted")
    print(f"  -> {out}")
    if clean is not None:
        diff = s["diff_vs_clean"]
        print(f"  text vs {clean.name}: {len(diff)} difference(s)")
        for line in diff[:20]:
            print(f"    {line}")
        if diff:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
