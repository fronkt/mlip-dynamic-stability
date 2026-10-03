"""Write and verify results/MANIFEST.sha256, the checksum list of the deposited results.

One line per file that git tracks under results/ (the manifest itself excluded), sorted by path:

    <sha256 hex>  <size in bytes>  <path relative to the repository root>

so that a reader of the Zenodo archive or a fresh clone can confirm that the ledger, the caches, the
revision tables and the figures are the bytes the paper was built from. ``sha256sum``-style check:

    awk '{print $1 "  " $3}' results/MANIFEST.sha256 | grep -v '^#' | sha256sum -c

Line endings. On Windows with ``core.autocrlf=true`` a text file that git stores with LF is checked
out with CRLF, and the same file on Linux has a different sha256. The manifest records the content
git stores: for a file whose index eol is ``lf`` and whose working-tree eol is ``crlf`` (as reported
by ``git ls-files --eol``) the CRLFs are turned back into LFs before hashing. ``--check`` accepts
either form, so it passes on a Windows checkout and a Linux one alike. Binary files and files that
git stores with CRLF are hashed as they are.

File list. By default the files are those in git's index (``git ls-files results``), which is what
a CI checkout contains; ``results/cache/`` is gitignored but force-added, so it is covered. Files
you have not yet ``git add``-ed are NOT listed (``--include-untracked`` lists them, honouring
.gitignore): run this script again after staging new results, or ``--check --strict`` (what CI runs)
will report them as present in git but missing from the manifest. Outside a git repository (a bare unzip of the
archive) generation walks results/ instead and ``--check`` skips the "unlisted file" test.

Usage (from anywhere; paths are resolved from this file):

    python scripts/make_manifest.py                       # write results/MANIFEST.sha256
    python scripts/make_manifest.py --check               # exit 1 on any missing or changed file; warns on unlisted ones
    python scripts/make_manifest.py --check --strict      # also exit 1 on a tracked file the manifest does not list (CI)
    python scripts/make_manifest.py --include-untracked   # also list new, un-added, non-ignored files

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
MANIFEST = RESULTS / "MANIFEST.sha256"
MANIFEST_REL = MANIFEST.relative_to(REPO).as_posix()

HEADER = [
    "# sha256  size_bytes  path   (every git-tracked file under results/, sorted; this file excluded)",
    "# Text files are hashed as git stores them (LF); see scripts/make_manifest.py.",
    "# Regenerate: python scripts/make_manifest.py     Verify: python scripts/make_manifest.py --check",
]


def _git(*args: str) -> bytes | None:
    """Raw stdout of ``git <args>`` run in the repository, or None if git or the repository is absent."""
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return r.stdout if r.returncode == 0 else None


def list_files(include_untracked: bool = False) -> tuple[list[str], set[str], bool]:
    """(sorted repo-relative paths under results/, those to hash with CRLF->LF, git_available)."""
    args = ["ls-files", "-z", "--eol", "--cached"] + (["--others", "--exclude-standard"]
                                                      if include_untracked else []) + ["--", "results"]
    out = _git(*args)
    if out is None:
        paths = sorted(p.relative_to(REPO).as_posix() for p in RESULTS.rglob("*") if p.is_file())
        return [p for p in paths if p != MANIFEST_REL], set(), False
    paths, normalise = set(), set()
    for rec in out.decode("utf-8").split("\0"):
        if not rec:
            continue
        meta, _, path = rec.partition("\t")
        fields = meta.split()
        paths.add(path)
        if "i/lf" in fields and "w/crlf" in fields:
            normalise.add(path)
    paths.discard(MANIFEST_REL)
    return sorted(paths), normalise, True


def digest(path: str, normalise: bool) -> tuple[str, int]:
    data = (REPO / path).read_bytes()
    if normalise:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest(), len(data)


def write_manifest(include_untracked: bool) -> int:
    paths, normalise, have_git = list_files(include_untracked)
    if not have_git:
        print("warning: git unavailable; listing every file under results/", file=sys.stderr)
    lines, skipped = [], []
    for p in paths:
        if not (REPO / p).is_file():
            skipped.append(p)           # tracked but deleted from the working tree
            continue
        h, n = digest(p, p in normalise)
        lines.append(f"{h}  {n}  {p}")
    MANIFEST.write_text("\n".join(HEADER + lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_REL}: {len(lines)} files, {sum(int(x.split()[1]) for x in lines):,} bytes")
    for p in skipped:
        print(f"warning: tracked but missing from the working tree, not listed: {p}", file=sys.stderr)
    return 0


def read_manifest() -> dict[str, tuple[str, int]]:
    entries: dict[str, tuple[str, int]] = {}
    for ln, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or len(parts[0]) != 64 or not parts[1].isdigit():
            raise SystemExit(f"{MANIFEST_REL}:{ln}: malformed line: {line!r}")
        entries[parts[2]] = (parts[0], int(parts[1]))
    return entries


def check_manifest(strict: bool = False) -> int:
    if not MANIFEST.exists():
        print(f"FAIL  {MANIFEST_REL} does not exist (run scripts/make_manifest.py)")
        return 1
    entries = read_manifest()
    bad: list[str] = []
    for p, (h, n) in sorted(entries.items()):
        f = REPO / p
        if not f.is_file():
            bad.append(f"missing    {p}")
            continue
        data = f.read_bytes()
        ok = len(data) == n and hashlib.sha256(data).hexdigest() == h
        if not ok and b"\r\n" in data:   # a Windows checkout of a file git stores with LF
            lf = data.replace(b"\r\n", b"\n")
            ok = len(lf) == n and hashlib.sha256(lf).hexdigest() == h
        if not ok:
            bad.append(f"changed    {p}")
    paths, _, have_git = list_files()
    unlisted = sorted(set(paths) - set(entries)) if have_git else []
    note = [f"unlisted   {p}  (tracked by git, not in the manifest: rerun scripts/make_manifest.py)"
            for p in unlisted]
    if strict:
        bad += note
    elif note:
        print(f"WARN  {len(note)} tracked file(s) under results/ are not in the manifest "
              f"(rerun scripts/make_manifest.py; --strict makes this a failure)")
        for n in note[:10]:
            print("  " + n)
        if len(note) > 10:
            print(f"  ... and {len(note) - 10} more")
    if bad:
        print(f"FAIL  {MANIFEST_REL}: {len(bad)} problem(s)")
        for b in bad[:50]:
            print("  " + b)
        if len(bad) > 50:
            print(f"  ... and {len(bad) - 50} more")
        return 1
    print(f"PASS  {MANIFEST_REL}: {len(entries)} files match")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the files against the manifest instead of writing it")
    ap.add_argument("--strict", action="store_true",
                    help="with --check: a tracked file that the manifest does not list is a failure, not a warning")
    ap.add_argument("--include-untracked", action="store_true",
                    help="also list files not yet git-added (honours .gitignore); ignored with --check")
    args = ap.parse_args()
    return check_manifest(args.strict) if args.check else write_manifest(args.include_untracked)


if __name__ == "__main__":
    raise SystemExit(main())
