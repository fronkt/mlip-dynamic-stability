"""Renumber the manuscript's references into order of first citation and propagate the numbers.

RSC numbers references by first citation. The revision added references wherever the argument
needed them, so the list now runs in order of addition and the introduction cites 1, 3, 2, 19,
20, 27, ... Renumbering the list by hand across the manuscript, the ESI (which cites main-text
references by number, 'refs 21, 22') and the response letter is the kind of edit that goes quietly
wrong, so it is done mechanically here, and --check proves the result instead of trusting it.

A citation is a pandoc superscript whose content is a list of reference numbers (^5^, ^5,6^,
^27–29^, ^5-7^, ^15a^), or a prose mention ('ref 22', 'refs 21, 22', 'ref. 5', 'refs 19 and 20',
'refs 27-37', 'Reference 20'). Math, code spans and code blocks are never touched, and a
superscript directly after 10 (10^4^) is a power, not a citation. A mark the script cannot
classify (0.78^22^, cm^2^, ^13^C, ^9-7^, ^15a,b^, 'refs 29-27') would either stay on the old paper
or turn an exponent into a citation, so any such mark stops the write until the source is fixed or
--force confirms every listed mark was read. First use is the reading order of everything above
'## References', figure captions and table cells included. HTML
comments are rewritten, so a PENDING note keeps pointing at the same paper, but they never reach
the submitted file and so do not count toward first use. References never cited go last, in their
original relative order, with a warning. A compound entry ('5. (a) ...; (b) ...') is one number
and moves as one.

Superscripts are re-emitted in canonical form: ascending, runs of three or more as a–b with an en
dash, pairs as a,b. Prose mentions keep their word and dash style and are left alone when their
numbers do not change. A prose range whose members no longer sit together is re-compressed ('refs
27-37' can become 'refs 20, 25-31, 33-35'), which is correct but worth a read, so every prose
rewrite is printed.

Nothing is written unless every file parses and every citation maps, and a file that changed on
disk while the script ran (another session editing it) aborts the write. The files are replaced
all or none: a manuscript renumbered without its ESI could never be repaired by a rerun, which
would find the manuscript already in order and map every number to itself. The same holds for a
file left out of the run, so leaving out the letter is reported.

    python scripts/renumber_refs.py --dry-run          # mapping and diff summary; writes nothing
    python scripts/renumber_refs.py                    # manuscript + ESI
    python scripts/renumber_refs.py --letter           # ... + paper/response/response_to_referees.md
    python scripts/renumber_refs.py --check --letter   # exit 1 unless the numbering is RSC-clean

--manuscript, --esi and --letter take a path, to run on copies. A copied manuscript needs an
explicit --esi (and a path after --letter), so a test run cannot rewrite the real ESI or letter
with the copy's numbering.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "paper" / "manuscript.md"
ESI = REPO / "paper" / "supplementary.md"
LETTER = REPO / "paper" / "response" / "response_to_referees.md"

EN_DASH = "–"
DASHES = "-–—"
# Escaped, because ',-–' inside a character class is a range that takes in every letter and digit.
_SEP = f"[,{re.escape(DASHES)}]"
# A reference number, optionally one labelled part of a compound entry (15a, 15(a)). At most three
# digits and not followed by a word character (which includes superscript digits) or a decimal, so
# years, 10⁴ and 2.5 are never read as references.
ITEM = r"\d{1,3}(?:\([a-z]\)|[a-z])?(?!\w|\.\d)"
ITEM_RE = re.compile(ITEM)
_WS = r"[ \t]*\n?[ \t]*"
SUPER = re.compile(r"(?<!\\)\^(?P<body>[^\^\s]+)\^")
SUPER_BODY = re.compile(rf"{ITEM}(?:{_SEP}{ITEM})*")
# Digits, parts and separators only, starting and ending on a number or part: what a citation list
# looks like, typo or not. Charges (2+, 2-) and signed exponents (-1) do not.
CITE_SHAPE = re.compile(rf"\d[\d,;a-z(){re.escape(DASHES)}]*[\da-z)]|\d")
PROSE = re.compile(
    rf"\b(?P<word>[Rr]ef(?:erence)?s?\.?)(?P<gap>[ \t\xa0]+\n?[ \t]*|\n[ \t]*)"
    rf"(?P<list>{ITEM}(?:(?:{_WS}{_SEP}{_WS}|(?:{_WS},)?{_WS}\band\b{_WS}){ITEM})*)"
)
# Leftmost-first alternation, so a '$' or backtick inside a comment cannot open a math or code span.
SEGMENT = re.compile(
    r"(?P<comment><!--.*?-->)"
    r"|^[ \t]*(?P<fence>`{3,}|~{3,}).*?^[ \t]*(?P=fence)[ \t]*$"
    r"|\$\$.+?\$\$"
    r"|(?P<tick>`+)(?:(?!\n[ \t]*\n).)+?(?P=tick)"
    r"|(?<![\\$])\$(?=[^\s$])[^$\n]*?(?<=[^\s\\])\$(?!\d)",
    re.S | re.M,
)
REF_HEADING = re.compile(r"^## References[ \t]*$", re.M)
NEXT_HEADING = re.compile(r"^#{1,2}[ \t]", re.M)
ENTRY = re.compile(r"(\d+)\.([ \t]+)(.*)")


@dataclass
class Cite:
    start: int
    end: int
    line: int
    old: str
    items: list[tuple[int, str]]  # (number, compound part) in written order, ranges expanded
    in_comment: bool
    prose: bool
    word: str = ""
    gap: str = ""
    dash: str = EN_DASH
    conj: str = ""


@dataclass
class Entry:
    number: int
    gap: str
    lines: list[str]  # the first line after 'N.<gap>', then continuation lines verbatim

    def render(self, number: int) -> str:
        return "\n".join([f"{number}.{self.gap}{self.lines[0]}", *self.lines[1:]])


@dataclass
class Manuscript:
    body: str
    heading: str
    lead: list[str]
    entries: list[Entry]
    sep: str
    trail: list[str]
    tail: str
    tail_line: int

    def assemble(self, body: str, ordered: list[Entry], tail: str) -> str:
        entries = self.sep.join(e.render(k) for k, e in enumerate(ordered, 1))
        region = "\n".join(self.lead + entries.split("\n") + self.trail)
        return body + self.heading + region + tail


def read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write_all(texts: dict[Path, str], originals: dict[Path, str]) -> str | None:
    """Replace every file or none; returns what went wrong instead of raising.

    Each file is staged as a temp file first, so a full disk or a locked directory fails before
    anything is replaced. A replace can still fail on Windows (the file open in an editor), and
    then the files already swapped are put back.
    """
    staged: dict[Path, Path] = {}
    swapped: list[Path] = []
    try:
        for path, text in texts.items():
            staged[path] = path.with_name(path.name + ".renumber.tmp")
            with open(staged[path], "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        for path, tmp in staged.items():
            os.replace(tmp, path)
            swapped.append(path)
        return None
    except OSError as exc:
        lost = []
        for path in swapped:
            try:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(originals[path])
            except OSError:
                lost.append(path)
        if lost:
            return (f"write failed ({exc}) and these could not be put back, so they hold the new "
                    f"numbering while the rest hold the old: {', '.join(map(str, lost))}")
        return f"write failed ({exc}); every file is as it was"
    finally:
        for tmp in staged.values():
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def segments(text: str) -> list[tuple[str, int, int]]:
    """Split text into ('text' | 'comment' | 'skip', start, end) runs that cover all of it."""
    runs, pos = [], 0
    for m in SEGMENT.finditer(text):
        if m.start() > pos:
            runs.append(("text", pos, m.start()))
        runs.append(("comment" if m.group("comment") else "skip", m.start(), m.end()))
        pos = m.end()
    if pos < len(text):
        runs.append(("text", pos, len(text)))
    return runs


def parse_items(listtext: str) -> tuple[list[tuple[int, str]], str, str] | None:
    """Expand '27–29', '5,6', '19 and 20' or '15(a)' into (number, part) pairs in written order.

    Returns None for a list that is not a clean citation (a reversed range, or a range with a
    lettered end), which the caller reports instead of guessing at.
    """
    found = list(ITEM_RE.finditer(listtext))
    items: list[tuple[int, str]] = []
    dash = conj = ""
    for i, m in enumerate(found):
        digits = re.match(r"\d+", m.group()).group()
        num, part = int(digits), m.group()[len(digits):]
        sep = listtext[found[i - 1].end():m.start()] if i else ""
        hit = next((ch for ch in sep if ch in DASHES), "")
        if hit:
            lo, lo_part = items[-1]
            if lo_part or part or num <= lo:
                return None
            dash = dash or hit
            items.extend((k, "") for k in range(lo + 1, num + 1))
            continue
        if "and" in sep:
            conj = ", and " if "," in sep else " and "
        items.append((num, part))
    return items, dash, conj


def find_cites(text: str, label: str, warnings: list[str], doubts: list[str],
               first_line: int = 1) -> list[Cite]:
    """Every citation in text outside math and code, in reading order.

    Marks that may or may not be citations go to doubts, which block the write: a guess either
    way leaves a citation on the wrong paper or turns an exponent into a citation. A doubtful mark
    that parses is still returned, so --force treats it as a citation.
    """
    cites: list[Cite] = []

    def line_at(pos: int) -> int:
        return first_line + text.count("\n", 0, pos)

    for kind, lo, hi in segments(text):
        if kind == "skip":
            continue
        chunk, in_comment = text[lo:hi], kind == "comment"
        for m in SUPER.finditer(chunk):
            start, end, body = lo + m.start(), lo + m.end(), m.group("body")
            before, where = text[max(0, start - 3):start], f"{label} L{line_at(start)}"
            shown = flat(text[max(0, start - 12):end + 3])
            shaped = CITE_SHAPE.fullmatch(body) is not None
            if re.search(r"\d\Z", before):
                if shaped and not re.search(r"(?<!\d)10\Z", before):
                    doubts.append(f"{where}: {m.group()} follows a number ('{shown}'); pandoc "
                                  "prints it as a superscript, so is it a citation or a power?")
                continue
            parsed = parse_items(body) if SUPER_BODY.fullmatch(body) else None
            if parsed is None:
                if shaped:
                    doubts.append(f"{where}: superscript {m.group()} looks like a citation but "
                                  "cannot be read (reversed range, lettered range, bad separator)")
                elif any(ch.isdigit() for ch in body):
                    warnings.append(f"{where}: superscript {m.group()} not read as a citation")
                continue
            if re.search(r"(?<!\w)[^\W\d_]{1,2}\Z", before):
                doubts.append(f"{where}: {m.group()} follows a one- or two-letter word "
                              f"('{shown}'); a citation or an exponent?")
            elif text[end:end + 1].isalnum():
                doubts.append(f"{where}: {m.group()} runs into the next word ('{shown}'); "
                              "a citation or an isotope label?")
            cites.append(Cite(start, end, line_at(start), m.group(), parsed[0],
                              in_comment, prose=False))
        for m in PROSE.finditer(chunk):
            start = lo + m.start()
            parsed = parse_items(m.group("list"))
            if parsed is None:
                doubts.append(f"{label} L{line_at(start)}: {flat(m.group())!r} looks like a "
                              "citation but cannot be read (reversed or lettered range)")
                continue
            items, dash, conj = parsed
            cites.append(Cite(start, lo + m.end(), line_at(start), m.group(), items, in_comment,
                              prose=True, word=m.group("word"), gap=m.group("gap"),
                              dash=dash or EN_DASH, conj=conj))
    cites.sort(key=lambda c: c.start)
    return cites


def parse_manuscript(text: str, path: Path) -> tuple[Manuscript, list[int]]:
    """Split the manuscript into body, '## References' heading, numbered entries and tail.

    Anything in the list that is neither an entry nor a continuation of one (a comment, a table,
    a stray paragraph) stops the run: moving entries around it would silently re-home it. Also
    returns the blank-line count between consecutive entries, so uneven spacing can be reported.
    """
    h = REF_HEADING.search(text)
    if h is None:
        raise SystemExit(f"{path}: no '## References' heading")
    nxt = NEXT_HEADING.search(text, h.end())
    list_end = nxt.start() if nxt else len(text)
    lines = text[h.end():list_end].split("\n")
    head_line = text.count("\n", 0, h.end()) + 1
    first = next((i for i, ln in enumerate(lines) if ENTRY.fullmatch(ln)), None)
    if first is None:
        raise SystemExit(f"{path}: no '1. ...' entries under '## References'")
    if any(ln.strip() for ln in lines[:first]):
        raise SystemExit(f"{path}: text between '## References' and the first entry")
    last = max(i for i, ln in enumerate(lines) if ln.strip())

    entries: list[Entry] = []
    gaps: list[int] = []
    blanks = 0
    for i in range(first, last + 1):
        ln = lines[i]
        m = ENTRY.fullmatch(ln)
        if m:
            if entries:
                gaps.append(blanks)
            entries.append(Entry(int(m[1]), m[2], [m[3]]))
            blanks = 0
        elif not ln.strip():
            blanks += 1
        elif not blanks and not ln.lstrip().startswith(("<!--", "#", "|", "![")):
            entries[-1].lines.append(ln)
        else:
            raise SystemExit(f"{path} L{head_line + i}: neither a reference entry nor a "
                             f"continuation of one: {ln[:70]!r}")

    numbers = [e.number for e in entries]
    if sorted(numbers) != list(range(1, len(numbers) + 1)):
        dup = sorted({n for n in numbers if numbers.count(n) > 1})
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        raise SystemExit(f"{path}: reference labels are not 1..N (duplicates {dup}, "
                         f"missing {missing})")
    gap = max(set(gaps), key=gaps.count) if gaps else 0
    return Manuscript(body=text[:h.start()], heading=text[h.start():h.end()],
                      lead=lines[:first], entries=entries, sep="\n" * (gap + 1),
                      trail=lines[last + 1:], tail=text[list_end:],
                      tail_line=text.count("\n", 0, list_end) + 1), gaps


def compress(items: list[tuple[int, str]], dash: str) -> list[str]:
    """Sorted, de-duplicated pieces: runs of three or more plain numbers become 'a–b'."""
    pieces: list[str] = []
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 3:
            pieces.append(f"{run[0]}{dash}{run[-1]}")
        else:
            pieces.extend(str(n) for n in run)
        run.clear()

    for n, part in sorted(set(items)):
        if part:
            flush()
            pieces.append(f"{n}{part}")
        elif run and n == run[-1] + 1:
            run.append(n)
        else:
            flush()
            run.append(n)
    flush()
    return pieces


def render(c: Cite, mapping: dict[int, int]) -> str:
    new = [(mapping[n], part) for n, part in c.items]
    if not c.prose:
        return "^" + ",".join(compress(new, EN_DASH)) + "^"
    # Same papers (a permutation inside 'refs 1, 2 and 3' included): the author's wording stands.
    if set(new) == set(c.items):
        return c.old
    pieces = compress(new, c.dash)
    if len(pieces) == 1:
        listtext = pieces[0]
    elif c.conj:
        joiner = " and " if len(pieces) == 2 else c.conj
        listtext = ", ".join(pieces[:-1]) + joiner + pieces[-1]
    else:
        listtext = ", ".join(pieces)
    return c.word + c.gap + listtext


def rewrite(text: str, cites: list[Cite], mapping: dict[int, int]) -> tuple[str, list[tuple]]:
    out, changes, last = [], [], 0
    for c in cites:
        new = render(c, mapping)
        if new != c.old:
            out += [text[last:c.start], new]
            last = c.end
            changes.append((c.line, c.old, new, c.prose, c.in_comment))
    out.append(text[last:])
    return "".join(out), changes


def first_use(cites: list[Cite]) -> tuple[list[int], dict[int, int]]:
    order: list[int] = []
    where: dict[int, int] = {}
    for c in cites:
        if c.in_comment:
            continue
        for n, _ in c.items:
            if n not in where:
                where[n] = c.line
                order.append(n)
    return order, where


def build_mapping(order: list[int], labels: list[int]) -> tuple[dict[int, int], list[int]]:
    mapping: dict[int, int] = {}
    for n in order:
        mapping.setdefault(n, len(mapping) + 1)
    uncited = [n for n in labels if n not in mapping]
    for n in uncited:
        mapping[n] = len(mapping) + 1
    return mapping, uncited


def flat(s: str) -> str:
    return " ".join(s.split())


def run_check(labels: list[int], order: list[int], where: dict[int, int],
              out_of_range: list[str], doubts: list[str],
              cites_by_file: dict[str, list[Cite]]) -> int:
    n_refs = len(labels)
    failures = 0

    def verdict(tag: str, problems: list[str]) -> None:
        nonlocal failures
        print(f"  {tag:<44} {'ok' if not problems else 'FAIL'}")
        for p in problems[:8]:
            print(f"      {p}")
        if len(problems) > 8:
            print(f"      ... {len(problems) - 8} more")
        failures += bool(problems)

    uncited = [n for n in labels if n not in where]
    verdict("(a) every reference 1..N is cited",
            [f"reference {n} is never cited in the body" for n in uncited])
    verdict("(b) citations in order of first use",
            [f"L{where[n]}: reference {n} is first cited in position {k}"
             for k, n in enumerate(order, 1) if n != k])
    expected = order + uncited
    verdict("(c) list sorted by first use",
            [f"list position {k} holds reference {got}; the reference first cited "
             f"at position {k} is {want}"
             for k, (got, want) in enumerate(zip(labels, expected), 1) if got != want])
    verdict(f"(d) no citation outside 1..{n_refs}", out_of_range)
    # A mark that cannot be read might be a citation (a)-(d) never saw.
    verdict("(e) no ambiguous or unreadable citation mark", doubts)

    identity = {n: n for n in labels}
    loose = [f"{name} L{c.line}: {c.old} is not in canonical form ({render(c, identity)})"
             for name, cites in cites_by_file.items() for c in cites
             if not c.prose and all(n in identity for n, _ in c.items)
             and render(c, identity) != c.old]
    if loose:
        print("  note: superscripts not in canonical form (a rewrite would normalise them)")
        for s in loose:
            print(f"      {s}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    ap.add_argument("--esi", type=Path, default=None,
                    help="default paper/supplementary.md; required with a copied --manuscript")
    # No type=Path here: argparse would convert the bare-flag sentinel too.
    ap.add_argument("--letter", nargs="?", const="", default=None,
                    help="also rewrite the response letter (default path if no value given)")
    ap.add_argument("--dry-run", action="store_true", help="print the mapping, write nothing")
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on a failure")
    ap.add_argument("--force", action="store_true",
                    help="write although ambiguous marks were listed (each one read and confirmed)")
    ap.add_argument("--verbose", action="store_true", help="list every superscript rewrite")
    args = ap.parse_args(argv)
    # The report quotes reference text (₃, á, –); a cp1252 Windows console would otherwise crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    copy = args.manuscript.resolve() != MANUSCRIPT.resolve()
    if copy and (args.esi is None or args.letter == ""):
        ap.error("--manuscript is not paper/manuscript.md, so give --esi (and --letter) a path; "
                 "the defaults are the real files")
    args.esi = args.esi or ESI
    args.letter = None if args.letter is None else Path(args.letter) if args.letter else LETTER

    warnings: list[str] = []
    doubts: list[str] = []
    raw = {"manuscript": read(args.manuscript)}
    paths = {"manuscript": args.manuscript, "ESI": args.esi}
    if args.letter:
        paths["letter"] = args.letter
    for name in ("ESI", "letter"):
        if name in paths:
            raw[name] = read(paths[name])
    crlf = {name: "\r\n" in text for name, text in raw.items()}
    text = {name: t.replace("\r\n", "\n") for name, t in raw.items()}
    for name in raw:
        if crlf[name] and text[name].count("\n") != raw[name].count("\r\n"):
            warnings.append(f"{name}: mixed line endings; a rewrite makes every one CRLF")

    ms, gaps = parse_manuscript(text["manuscript"], args.manuscript)
    if len(set(gaps)) > 1:
        warnings.append("manuscript: reference entries are separated unevenly; the rewrite uses "
                        f"{len(ms.sep) - 1} blank line(s) throughout")
    labels = [e.number for e in ms.entries]
    n_refs = len(labels)
    body_cites = find_cites(ms.body, "manuscript", warnings, doubts)
    tail_cites = find_cites(ms.tail, "manuscript", warnings, doubts, ms.tail_line)
    if tail_cites:
        warnings.append(f"manuscript: {len(tail_cites)} citation(s) after the reference list are "
                        "rewritten but do not count toward first use")
    cites = {"manuscript": body_cites + tail_cites}
    for name in ("ESI", "letter"):
        if name in text:
            cites[name] = find_cites(text[name], name, warnings, doubts)

    out_of_range = [f"{name} L{c.line}: {flat(c.old)} cites {n}"
                    for name, cs in cites.items() for c in cs for n, _ in c.items
                    if not 1 <= n <= n_refs]
    order, where = first_use(body_cites)
    order = [n for n in order if 1 <= n <= n_refs]

    for w in warnings:
        print(f"warning: {w}")
    if args.check:
        print(f"check {args.manuscript} ({n_refs} references"
              + "".join(f"; {k} {paths[k]}" for k in paths if k != "manuscript") + ")")
        return run_check(labels, order, where, out_of_range, doubts, cites)
    if out_of_range:
        print("citations past the end of the list; nothing written:")
        for s in out_of_range:
            print(f"  {s}")
        return 1

    mapping, uncited = build_mapping(order, labels)
    for n in uncited:
        print(f"warning: reference {n} is never cited in the body; it goes last, as "
              f"{mapping[n]}")
    ordered = sorted(ms.entries, key=lambda e: mapping[e.number])
    new_body, ch_body = rewrite(ms.body, body_cites, mapping)
    new_tail, ch_tail = rewrite(ms.tail, tail_cites, mapping)
    new = {"manuscript": ms.assemble(new_body, ordered, new_tail)}
    changes = {"manuscript": ch_body + ch_tail}
    for name in ("ESI", "letter"):
        if name in text:
            new[name], changes[name] = rewrite(text[name], cites[name], mapping)

    in_body = [c for c in body_cites if not c.in_comment]
    print(f"{args.manuscript}: {n_refs} references; {len(in_body)} citations in the body, "
          f"{len(body_cites) - len(in_body)} in comments, {len(tail_cites)} after the list")
    print("new <- old, in order of first use ('=' unchanged):")
    for k, e in enumerate(ordered, 1):
        flag = "=" if k == e.number else " "
        note = "  [never cited]" if e.number in uncited else ""
        print(f"  {k:>3} <- {e.number:<3}{flag} {flat(e.lines[0])[:70]}{note}")
    moved = sum(k != e.number for k, e in enumerate(ordered, 1))
    for name in new:
        n_sup = sum(not c[3] for c in changes[name])
        n_prose = sum(c[3] for c in changes[name])
        extra = f"; list: {moved} of {n_refs} entries renumbered" if name == "manuscript" else ""
        print(f"{name}: {n_sup} of {sum(not c.prose for c in cites[name])} superscripts and "
              f"{n_prose} of {sum(c.prose for c in cites[name])} prose mentions rewritten{extra}")
        for line, old, rep, prose, in_comment in changes[name]:
            if prose or args.verbose:
                where_ = " (in a comment)" if in_comment else ""
                print(f"    L{line}{where_}: {flat(old)}  ->  {flat(rep)}")
    print("diff summary:")
    for name in new:
        diff = list(difflib.unified_diff(text[name].split("\n"), new[name].split("\n"), n=0,
                                         lineterm=""))
        plus = sum(d.startswith("+") and not d.startswith("+++") for d in diff)
        minus = sum(d.startswith("-") and not d.startswith("---") for d in diff)
        print(f"  {paths[name]}: +{plus} -{minus} lines"
              + ("" if new[name] != text[name] else " (unchanged)"))

    if "letter" not in paths and moved:
        print(f"warning: {LETTER.name} is not part of this run (no --letter); once this writes, its "
              "reference numbers are stale and a rerun cannot fix them (the mapping is gone)")
    if doubts:
        print(f"ambiguous citation marks{' (--force: written as listed)' if args.force else ''}:")
        for s in doubts:
            print(f"  {s}")
    if args.dry_run:
        print("dry run: nothing written")
        return 1 if doubts and not args.force else 0
    if doubts and not args.force:
        print("nothing written: fix each mark in the source (a power belongs in math or as 10⁴), "
              "or rerun with --force once each is confirmed")
        return 1
    for name in new:
        if read(paths[name]) != raw[name]:
            print(f"{paths[name]} changed on disk while this ran; nothing written")
            return 1
    todo = {paths[name]: new[name].replace("\n", "\r\n") if crlf[name] else new[name]
            for name in new if new[name] != text[name]}
    failed = write_all(todo, {paths[name]: raw[name] for name in new})
    if failed:
        print(failed)
        return 1
    for path in todo:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
