"""Build the "changes clearly marked" manuscript and ESI with Word's own Compare.

RSC Advances wants the revised manuscript "with changes clearly marked". A hand-highlighted copy
drifts from the clean file and cannot be checked, and a diff of the markdown is unreadable to an
editor. So both versions are rendered by the SAME markdown -> docx pipeline -- the as-reviewed
snapshot (commit 76d3a84, what the three referees read) and the current markdown -- and Word's
Compare turns the pair into real tracked changes: word-level, formatting changes ignored, every
revision attributed to Frank Cai. Rendering both sides identically is what makes the markup mean
"the text changed" rather than "the styling changed".

The renderer is scripts/build_docx.py's pipeline (read_source, check_figures, run_pandoc,
format_docx) when that module is importable, so the marked file has the clean submission file's
page, fonts, line numbers and figure placement; otherwise a bare pandoc build. The line printed as
"renderer:" says which.

Figures on the as-reviewed side. The snapshot's image links were written relative to paper/ (it
was paper/manuscript.md at the reviewed commit), so from its new home they point at nothing, and
resolving them against the working tree would show the CURRENT figures as though the referees had
seen them. They are read from the reviewed commit with git first, then from beside the snapshot,
never from the working tree. An image that still cannot be found is SUBSTITUTED by a labelled
placeholder with its caption kept, so each figure is present on both sides and the compare stays
aligned. A missing image on the current side is an error: it would be missing from the submission.

HTML comments never reach either docx, and the build checks the result for 'PENDING' and '<!--'.
Unresolved <!-- PENDING ... --> markers in the current markdown refuse the build unless
--allow-pending (test builds only: the output then lacks whatever the markers hold a place for,
and is written as <stem>_DRAFT.docx so it cannot be taken for the file to upload). A '<!--' that
is never closed refuses the build outright.

The PDF shows all markup with deletions inline. Word always draws table-cell insertions as
margin balloons, so any table that gained cells reserves the balloon column on every page.

Run from the repo root (Windows, Microsoft Word installed):

    python scripts/build_marked_changes.py [--pdf] [--allow-pending] [--out-dir DIR]

writes paper/manuscript_marked_changes.docx and paper/supplementary_marked_changes.docx, and the
two .pdf files under --pdf.

Word is a private invisible instance tied to this process by a job object, so it dies with the
build even when the build is killed outright; exit code 2 means the build refused to start.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "paper" / "submissions" / "rsc-advances-2026-07"
REVIEWED_REV = "76d3a84"
# The as-reviewed snapshots were paper/*.md at REVIEWED_REV, so their links are relative to paper/.
REVIEWED_LINK_BASE = "paper"
AUTHOR = "Frank Cai"

DOCUMENTS = {
    # key: (current markdown, as-reviewed markdown, output stem)
    "manuscript": (REPO / "paper" / "manuscript.md",
                   SUBMISSION / "manuscript-as-reviewed.md",
                   "manuscript_marked_changes"),
    "esi": (REPO / "paper" / "supplementary.md",
            SUBMISSION / "supplementary-as-reviewed.md",
            "supplementary_marked_changes"),
}

COMMENT = re.compile(r"<!--(.*?)-->", re.S)
# A comment filling whole lines goes with its line break, so a marker between two lines of one
# paragraph (the abstract has one) does not split the paragraph. Same rule as build_docx.py.
WHOLE_LINE_COMMENT = re.compile(r"^[ \t]*<!--(?:(?!-->).)*-->[ \t]*(?:\r?\n|\Z)", re.S | re.M)
INLINE_COMMENT = re.compile(r"[ \t]*<!--.*?-->", re.S)
IMAGE_EXT = r"(?:png|jpe?g|gif|tiff?|bmp|svg|emf|wmf|pdf)"
# Only the link target is matched: captions carry brackets of their own ("[0.36, 0.70]"), and only
# figures link to image files in these documents.
IMAGE_TARGET = re.compile(r"\]\(\s*([^)\s]+\." + IMAGE_EXT + r")((?:\s+\"[^\"]*\")?)\s*\)", re.I)

# Word enums (late-bound COM has no constants).
WD_ALERTS_NONE = 0
WD_COMPARE_DESTINATION_NEW = 2
WD_GRANULARITY_WORD_LEVEL = 1
WD_FORMAT_DOCUMENT_DEFAULT = 16
WD_EXPORT_FORMAT_PDF = 17
WD_EXPORT_DOCUMENT_WITH_MARKUP = 7
WD_EXPORT_CREATE_HEADING_BOOKMARKS = 1
WD_REVISIONS_MARKUP_ALL = 2
WD_REVISIONS_VIEW_FINAL = 0
WD_INLINE_REVISIONS = 1
MSO_AUTOMATION_SECURITY_FORCE_DISABLE = 3
STILL_ACTIVE = 259
REVISION_TYPES = {
    1: "insert", 2: "delete", 3: "property", 4: "paragraph number", 5: "display field",
    6: "reconcile", 7: "conflict", 8: "style", 9: "replace", 10: "paragraph property",
    11: "table property", 12: "section property", 13: "style definition", 14: "moved from",
    15: "moved to", 16: "cell insertion", 17: "cell deletion", 18: "cell merge", 19: "cell split",
}

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:6.1f} s] {msg}", flush=True)


def pending_markers(path: Path) -> list[tuple[int, str]]:
    """(line, marker names) for every HTML comment in ``path`` that holds a PENDING placeholder."""
    text = path.read_text(encoding="utf-8")
    found = []
    for m in COMMENT.finditer(text):
        names = re.findall(r"PENDING[-\w]*", m.group(1))
        if names:
            found.append((text.count("\n", 0, m.start()) + 1, " / ".join(dict.fromkeys(names))))
    return found


def strip_comments(text: str) -> str:
    return INLINE_COMMENT.sub("", WHOLE_LINE_COMMENT.sub("", text))


def unclosed_comment(path: Path) -> int | None:
    """Line of a '<!--' that is never closed, else None.

    pending_markers cannot see such a marker, and pandoc prints it as literal text with '--'
    turned into an en dash, so the '<!--' leak check on the compared document cannot see it
    either: it has to be caught in the markdown.
    """
    text = strip_comments(path.read_text(encoding="utf-8"))
    i = text.find("<!--")
    return None if i < 0 else text.count("\n", 0, i) + 1


# --- image resolution -------------------------------------------------------------------------

ImageSource = Callable[[str], "bytes | None"]


def worktree_images(md_path: Path) -> ImageSource:
    """Current side: the link as written relative to the markdown, else relative to paper/.

    The second location is what lets a copy of paper/manuscript.md (a test build in a scratch
    directory) still find its figures.
    """
    def source(target: str) -> bytes | None:
        for base in (md_path.parent, REPO / "paper"):
            p = (base / target).resolve()
            if p.is_file():
                return p.read_bytes()
        return None
    return source


def reviewed_images(md_path: Path, rev: str) -> ImageSource:
    """As-reviewed side: the blob at the reviewed commit, else a file next to the snapshot.

    The commit comes first because a copy of the snapshot placed in paper/, or in a scratch tree
    laid out like the repo, resolves its links onto CURRENT figures, which would then pass for
    what the referees saw.
    """
    def source(target: str) -> bytes | None:
        repo_path = os.path.normpath(os.path.join(REVIEWED_LINK_BASE, target)).replace("\\", "/")
        if not repo_path.startswith("../"):
            r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{repo_path}"],
                               capture_output=True)
            if r.returncode == 0 and r.stdout:
                return r.stdout
        local = (md_path.parent / target).resolve()
        return local.read_bytes() if local.is_file() else None
    return source


def placeholder_png(path: Path, label: str) -> None:
    """A grey box naming the missing image, so the figure and its caption survive the compare."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (3000, 700), "#eeeeee")
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, 2995, 695], outline="#888888", width=10)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:                       # Pillow < 10.1: fixed bitmap font only
        font = ImageFont.load_default()
    draw.text((1500, 350), label, fill="#333333", font=font, anchor="mm")
    img.save(path, dpi=(600, 600))


def stage(md_path: Path, images: ImageSource, work: Path, *, placeholder_ok: bool
          ) -> tuple[Path, int, list[str]]:
    """Copy one side into ``work`` with comments stripped and each image beside it.

    The two sides spell their links identically but must resolve to different files, so every
    image is copied next to the staged markdown under a fresh name and the link rewritten to it.
    Returns (staged markdown, images, links that got a placeholder).
    """
    work.mkdir(parents=True, exist_ok=True)
    text = strip_comments(md_path.read_text(encoding="utf-8"))
    count, missing = 0, []

    def relink(m: re.Match) -> str:
        nonlocal count
        target, title = m.group(1), m.group(2)
        count += 1
        dest = work / f"img{count:02d}_{Path(target).name}"
        data = images(target)
        if data is None:
            if not placeholder_ok:
                raise FileNotFoundError(f"{md_path}: image {target} not found")
            missing.append(target)
            dest = dest.with_suffix(".png")
            placeholder_png(dest, f"As-reviewed figure image not recoverable: {Path(target).name}")
        else:
            dest.write_bytes(data)
        return f"]({dest.name}{title})"

    staged = work / md_path.name
    staged.write_text(IMAGE_TARGET.sub(relink, text), encoding="utf-8", newline="\n")
    return staged, count, missing


# --- markdown -> docx -------------------------------------------------------------------------

def pandoc_docx(md: Path, out: Path) -> None:
    """Fallback build: pandoc markdown (superscript citations, pipe tables, implicit figures)."""
    subprocess.run(["pandoc", md.name, "-f", "markdown", "-t", "docx",
                    "--resource-path", str(md.parent), "-o", str(out)],
                   check=True, cwd=md.parent)


def docx_renderer() -> tuple[Callable[[Path, Path], None], str]:
    """build_docx.py's pipeline when importable; a missing module falls back, a broken one raises.

    Only the module itself being absent counts as missing: an ImportError raised inside
    build_docx.py (a dependency it cannot load) is a broken module, and falling back then would
    silently give the marked file a different page, font and figure layout from the clean one.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import build_docx as bd
    except ModuleNotFoundError as e:
        if e.name != "build_docx":
            raise
        return pandoc_docx, "internal pandoc (scripts/build_docx.py not present)"
    finally:
        sys.path.pop(0)
    needed = ("read_source", "check_figures", "run_pandoc", "format_docx")
    lacking = [n for n in needed if not callable(getattr(bd, n, None))]
    if lacking:
        return pandoc_docx, f"internal pandoc (scripts/build_docx.py lacks {', '.join(lacking)})"

    def render(md: Path, out: Path) -> None:
        src = bd.read_source(md)
        # Lenient: figure quality is the clean build's gate, not this one's.
        bd.check_figures(src, md.parent, True)
        for warning in bd.run_pandoc(src, md.parent, out):
            print(f"    pandoc: {warning}")
        bd.format_docx(out, src.title)

    return render, "scripts/build_docx.py (read_source, check_figures, run_pandoc, format_docx)"


# --- Word -------------------------------------------------------------------------------------

def open_own_process(word):
    """(handle, PID, job) for THIS Word instance's process, read off a scratch document's window.

    Word exposes no PID, and diffing the WINWORD.EXE list can pick up a Word another session
    starts at the same moment. The handle is opened immediately so the PID cannot be recycled.

    The finally clauses that quit Word never run if this Python is killed outright (taskkill, a
    closed console, an agent's turn ending), and the invisible Word then runs until logoff. So
    Word is also put in a job object that kills it when the job's last handle closes, which
    Windows does when this process ends, however it ends. If Windows refuses, the build goes on
    without that guarantee and says so.
    """
    import pywintypes
    import win32api
    import win32con
    import win32job
    import win32process

    scratch = word.Documents.Add()
    try:
        pid = win32process.GetWindowThreadProcessId(scratch.ActiveWindow.Hwnd)[1]
    finally:
        scratch.Close(0)
    access = (win32con.SYNCHRONIZE | win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA
              | win32con.PROCESS_QUERY_LIMITED_INFORMATION)
    handle = win32api.OpenProcess(access, False, pid)
    job = None
    try:
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        win32job.AssignProcessToJobObject(job, handle)
    except pywintypes.error as e:
        if job is not None:
            job.Close()
        job = None
        log(f"WARNING: Word pid {pid} could not be tied to this process ({e.strerror}); if this "
            "build is killed, end that WINWORD.EXE by hand")
    return handle, pid, job


def reap(handle, pid: int, timeout: float = 60.0) -> str:
    """Wait for Word to quit; terminate it only if it has not quit within ``timeout``.

    "Quit" means the process has an exit code. On this machine a quit WINWORD.EXE stays listed
    for minutes with no handles and one thread (teardown, not Word), so waiting for the process
    object to be signalled would hang or kill something already finished.
    """
    import win32api
    import win32process

    try:
        deadline = time.monotonic() + timeout
        while (code := win32process.GetExitCodeProcess(handle)) == STILL_ACTIVE:
            if time.monotonic() > deadline:
                win32api.TerminateProcess(handle, 1)
                return f"Word pid {pid} had not quit {timeout:.0f} s after Quit: TERMINATED"
            time.sleep(0.25)
        return f"Word pid {pid} quit (exit code {code})"
    finally:
        win32api.CloseHandle(handle)


def word_compare(original: Path, revised: Path, out_docx: Path, out_pdf: Path | None,
                 timeout: float) -> dict:
    """Word Compare -> tracked-changes docx (and a PDF with the markup shown).

    A private invisible Word (DispatchEx), so no document Frank has open is touched and the export
    cannot block on his session. It is always quit, then reaped by its own PID; a watchdog kills
    it if the whole compare outlives ``timeout`` (a modal dialog in an invisible Word would
    otherwise hang the build forever). A killed Word surfaces as whatever COM error the next call
    happens to hit, so a watchdog kill is reported as a TimeoutError instead. The two view
    settings the PDF needs (all markup, deletions inline) are user-level Word preferences, so
    they are put back before quitting.
    """
    import pythoncom
    import win32api
    import win32com.client

    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    handle = job = watchdog = None
    fired = threading.Event()

    def kill() -> None:
        fired.set()
        win32api.TerminateProcess(handle, 1)

    try:
        word.Visible = False
        word.DisplayAlerts = WD_ALERTS_NONE
        word.AutomationSecurity = MSO_AUTOMATION_SECURITY_FORCE_DISABLE
        handle, pid, job = open_own_process(word)
        watchdog = threading.Timer(timeout, kill)
        watchdog.daemon = True
        watchdog.start()
        return _compare(word, original, revised, out_docx, out_pdf)
    except Exception as e:
        if fired.is_set():
            raise TimeoutError(f"Word (pid {pid}) was still working {timeout:.0f} s into the "
                               "compare and was killed (--word-timeout)") from e
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
        try:
            word.Quit(0)
        except Exception:                   # already gone (watchdog) or disconnected
            pass
        word = None
        pythoncom.CoUninitialize()
        if handle is not None:
            log(reap(handle, pid))
        if job is not None:
            job.Close()


def _compare(word, original: Path, revised: Path, out_docx: Path, out_pdf: Path | None) -> dict:
    """The document work, in its own frame so every document reference is gone before Quit."""
    orig = word.Documents.Open(str(original), False, True, False)   # confirm, read-only, recent
    rev = word.Documents.Open(str(revised), False, True, False)
    log("comparing")
    # Positional: Original, Revised, Destination, Granularity, CompareFormatting, CaseChanges,
    # Whitespace, Tables, Headers, Footnotes, Textboxes, Fields, Comments, Moves, RevisedAuthor,
    # IgnoreAllComparisonWarnings.
    result = word.CompareDocuments(orig, rev, WD_COMPARE_DESTINATION_NEW,
                                   WD_GRANULARITY_WORD_LEVEL, False,
                                   True, True, True, True, True, True, True, True, True,
                                   AUTHOR, True)
    orig.Close(0)
    rev.Close(0)
    try:
        result.TrackRevisions = False
        text = result.Content.Text
        # pandoc's smart punctuation prints a stray '<!--' as '<!' + en dash.
        leaked = [t for t in ("PENDING", "<!--", "<!–") if t in text]
        if leaked:
            raise RuntimeError(f"{' and '.join(map(repr, leaked))} in the compared document")
        summary = revision_summary(result)
        out_docx.parent.mkdir(parents=True, exist_ok=True)
        result.SaveAs2(str(out_docx), WD_FORMAT_DOCUMENT_DEFAULT)
        log(f"saved {out_docx.name}")
        if out_pdf is not None:
            view = result.ActiveWindow.View
            saved = (view.RevisionsFilter.Markup, view.RevisionsFilter.View, view.MarkupMode)
            try:
                view.ShowRevisionsAndComments = True
                view.RevisionsFilter.Markup = WD_REVISIONS_MARKUP_ALL
                view.RevisionsFilter.View = WD_REVISIONS_VIEW_FINAL
                view.MarkupMode = WD_INLINE_REVISIONS
                # OutputFileName, ExportFormat, OpenAfterExport, OptimizeFor (print), Range (all),
                # From, To, Item, IncludeDocProps, KeepIRM, CreateBookmarks, DocStructureTags,
                # BitmapMissingFonts, UseISO19005_1.
                result.ExportAsFixedFormat(str(out_pdf), WD_EXPORT_FORMAT_PDF, False, 0, 0, 1, 1,
                                           WD_EXPORT_DOCUMENT_WITH_MARKUP, True, True,
                                           WD_EXPORT_CREATE_HEADING_BOOKMARKS, True, True, False)
            finally:
                view.RevisionsFilter.Markup, view.RevisionsFilter.View, view.MarkupMode = saved
            log(f"exported {out_pdf.name}")
        return summary
    finally:
        result.Close(0)


def revision_summary(doc) -> dict:
    """Word's own revision count, split by type, with the authors it attributes them to."""
    by_type, authors = Counter(), Counter()
    for r in doc.Revisions:
        by_type[REVISION_TYPES.get(r.Type, str(r.Type))] += 1
        authors[r.Author] += 1
    return {"total": doc.Revisions.Count, "by_type": dict(by_type), "authors": dict(authors)}


def pdf_pages(pdf: Path) -> int | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return len(PdfReader(str(pdf)).pages)


# --- driver -----------------------------------------------------------------------------------

def build_one(key: str, current: Path, reviewed: Path, stem: str, args, render,
              work: Path) -> dict:
    cur_md, n_cur, _ = stage(current, worktree_images(current), work / key / "current",
                             placeholder_ok=False)
    old_md, n_old, holders = stage(reviewed, reviewed_images(reviewed, args.reviewed_rev),
                                   work / key / "reviewed", placeholder_ok=True)
    old_docx = work / key / f"{stem}_as_reviewed.docx"
    new_docx = work / key / f"{stem}_current.docx"
    log(f"{key}: rendering as-reviewed and current")
    render(old_md, old_docx)
    render(cur_md, new_docx)
    out_docx = args.out_dir / f"{stem}.docx"
    out_pdf = args.out_dir / f"{stem}.pdf" if args.pdf else None
    summary = word_compare(old_docx, new_docx, out_docx, out_pdf, args.word_timeout)
    summary.update(docx=out_docx, pdf=out_pdf, pages=out_pdf and pdf_pages(out_pdf),
                   images=(n_cur, n_old), placeholders=holders)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdf", action="store_true", help="also export PDFs with the markup shown")
    ap.add_argument("--allow-pending", action="store_true",
                    help="build even though PENDING markers remain (test builds only)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "paper")
    ap.add_argument("--only", choices=sorted(DOCUMENTS), action="append",
                    help="build just this document (repeatable)")
    ap.add_argument("--manuscript", type=Path, help="current manuscript markdown")
    ap.add_argument("--esi", type=Path, help="current ESI markdown")
    ap.add_argument("--reviewed-manuscript", type=Path, help="as-reviewed manuscript markdown")
    ap.add_argument("--reviewed-esi", type=Path, help="as-reviewed ESI markdown")
    ap.add_argument("--reviewed-rev", default=REVIEWED_REV,
                    help="git revision the as-reviewed figures are read from")
    ap.add_argument("--keep-work", type=Path,
                    help="keep the staged markdown and the two plain docx per document here")
    ap.add_argument("--word-timeout", type=float, default=900.0,
                    help="seconds before a hung Word is killed (per document)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    args.out_dir = args.out_dir.resolve()

    jobs = {}
    for key, (cur, old, stem) in DOCUMENTS.items():
        if args.only and key not in args.only:
            continue
        cur = getattr(args, key) or cur
        old = getattr(args, f"reviewed_{key}") or old
        jobs[key] = (cur.resolve(), old.resolve(), stem)

    unclosed = [(p.name, line) for cur, old, _ in jobs.values() for p in (cur, old)
                if (line := unclosed_comment(p)) is not None]
    if unclosed:
        for name, line in unclosed:
            print(f"  {name}:{line}  '<!--' is never closed")
        print("REFUSED: an HTML comment is never closed; pandoc would print it, and everything "
              "after it, as text.")
        return 2

    pending = [(key, cur.name, line, names) for key, (cur, _, _) in jobs.items()
               for line, names in pending_markers(cur)]
    if pending:
        for _, name, line, names in pending:
            print(f"  {name}:{line}  {names}")
        if not args.allow_pending:
            print(f"REFUSED: {len(pending)} PENDING marker(s) remain in the current markdown. "
                  "Resolve them, or pass --allow-pending for a test build.")
            return 2
        # A draft must never sit in paper/ under the name of the file that gets uploaded.
        drafts = {key for key, *_ in pending}
        jobs = {key: (cur, old, f"{stem}_DRAFT" if key in drafts else stem)
                for key, (cur, old, stem) in jobs.items()}
        print(f"WARNING: DRAFT build, {len(pending)} PENDING marker(s) stripped "
              "(--allow-pending). The files are named *_DRAFT; do not submit them.")

    render, how = docx_renderer()
    print(f"renderer: {how}")
    # Word has let go of every file by the time a build returns or raises, except when it could
    # not be identified; a cleanup failure then must not replace the real error.
    with tempfile.TemporaryDirectory(prefix="marked_changes_", ignore_cleanup_errors=True) as tmp:
        work = args.keep_work.resolve() if args.keep_work else Path(tmp)
        for key, (cur, old, stem) in jobs.items():
            s = build_one(key, cur, old, stem, args, render, work)
            print(f"{key}: {s['docx']}")
            if s["pdf"]:
                print(f"  {s['pdf']} ({s['pages']} pages)")
            print(f"  revisions: {s['total']} ("
                  + ", ".join(f"{k} {v}" for k, v in sorted(s["by_type"].items())) + ")")
            print(f"  authors: {s['authors']}")
            holders = s["placeholders"]
            print(f"  images: current {s['images'][0]}, as-reviewed {s['images'][1]}"
                  + (f" ({len(holders)} placeholder: {', '.join(holders)})" if holders else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
