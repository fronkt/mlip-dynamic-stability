"""Mechanically diff the headline numbers between the as-reviewed manuscript and the current one.

The RSC Advances referees (reports returned 2026-09-11) read commit 76d3a84, submitted
2026-07-13. Between submission and review the August audit re-measured everything in pinned
environments, so several numbers the referees quote no longer exist. The response letter has to
disclose that, and the disclosure has to be *generated* rather than recalled -- a hand-copied
table of "what changed" is exactly the artefact that should not be trusted.

This script extracts the per-model results tables from both versions and compares them cell by
cell, then does the same for a curated list of labelled scalar claims that live in prose. It
writes paper/response/number_changes.md.

Reads only. Run from the repo root:

    python scripts/reviewed_version_delta.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVIEWED_REV = "76d3a84"
CURRENT = REPO / "paper" / "manuscript.md"
OUT = REPO / "paper" / "response" / "number_changes.md"

# Prose claims worth tracking, as (label, reviewed_pattern, current_pattern, note).
#
# The two versions are given SEPARATE patterns on purpose. Several of these sentences were not
# merely re-numbered but rewritten, so a single regex that matched both would either miss one
# version or, worse, silently match the wrong quantity. Where a claim genuinely does not exist in
# a version, the pattern is None and the table says "absent" rather than inventing a value.
SCALAR_CLAIMS: list[tuple[str, str | None, str | None, str]] = [
    ("FE-perovskite screen recall",
     r"(\d+ of 30 model units \(recall 0?\.\d+\))",
     r"(\d+ of 30 model units \(recall 0?\.\d+\))",
     "the headline screen number"),
    ("SSCHA FE-perovskite recall",
     r"(\d+ of 30 units \(recall 0?\.\d+\))",
     r"only (\d+ of the \d+ units that return a physical number)[\s\S]{0,40}?\(recall (0?\.\d+)\)",
     "denominator changed: deepest-well units now fail loudly instead of returning junk"),
    ("bcc softmode-vs-SSCHA rank correlation, all models",
     r". = (0?\.\d+) with 0?\.\d+ sign agreement",
     r"not rank-correlated \(Spearman . = (0?\.\d+)\)",
     "**COLLAPSED.** This was the reviewed version's cross-validation statistic"),
    ("bcc softmode-vs-SSCHA rank correlation, excl ORB-v2",
     r"rank correlation softens to (0?\.\d+)",
     None,
     "not reported in the current version; ex-ORB rho is now ~= -0.003 (see Phase C)"),
    ("bcc sign agreement, all models",
     r". = 0?\.\d+ with (0?\.\d+) sign agreement",
     r"sign agreement (0?\.\d+);",
     "the statistic that replaced rho as the cross-validation measure"),
    ("bcc sign agreement, excl ORB-v2",
     r"sign agreement to (0?\.\d+), while",
     r"(0?\.\d+), excluding ORB-v2",
     ""),
    ("H2 association phi",
     r"gives\s*\n?. = (0?\.\d+)",
     r"φ = (−?-?0?\.\d+) at 300 K",
     "now temperature-resolved instead of a single pooled value"),
    ("H2 McNemar exact p",
     r"McNemar exact p = (0?\.\d+)",
     r"McNemar exact p = (0?\.\d+)\.",
     "R3 criticises the reviewed p = 0.34; the re-measurement moved it"),
    ("H2 discordant cells",
     r"(Seven) of the 71 harmonically-correct",
     r"(\d+)\s*\n?harmonically-correct units are mis-called",
     ""),
    ("Guardrail: consensus error rate",
     r"consensus is wrong on (\d+\.\d+)%",
     r"consensus is wrong on (\d+\.\d+)%",
     ""),
    ("Guardrail: split-vote error rate",
     r"the consensus error rate is (0?\.\d+)",
     r"the consensus error rate is (0?\.\d+)",
     ""),
    ("Guardrail: unanimous error rate",
     r"against (0?\.\d+) on the \d+ unanimous",
     r"against (0?\.\d+) on the \d+ unanimous",
     ""),
    ("Guardrail: vote-split AUC",
     r"reaches AUC (0?\.\d+)",
     r"reaches AUC (0?\.\d+)",
     ""),
    ("Guardrail: frequency-spread AUC",
     r"\(AUC (0?\.\d+), no better than chance",
     r"\(AUC (0?\.\d+), below chance",
     ""),
    ("SSCHA stochastic reproducibility (THz)",
     r"\+(\d+\.\d+) . 0\.001 THz",
     r"\+(\d+\.\d+) . 0\.001 THz",
     "unchanged - the one number the audit did not move"),
]

TABLE_ROW = re.compile(r"^\|\s*([A-Za-z][\w\-. ]*?)\s*\|(.+)\|\s*$")


def read_reviewed() -> str:
    out = subprocess.run(
        ["git", "show", f"{REVIEWED_REV}:paper/manuscript.md"],
        cwd=REPO, capture_output=True, text=True, check=True,
        encoding="utf-8",  # Windows defaults to cp1252 and the manuscript is full of UTF-8
    )
    return out.stdout


def section(text: str, start: str, end: str) -> str:
    """Return the slice of *text* between the headings *start* and *end*."""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i : j if j > 0 else len(text)]


def parse_model_table(block: str) -> dict[str, list[str]]:
    """Pull a per-model markdown table out of *block* as {model: [cells]}."""
    rows: dict[str, list[str]] = {}
    for line in block.splitlines():
        m = TABLE_ROW.match(line.strip())
        if not m:
            continue
        key, rest = m.group(1).strip(), m.group(2)
        if key.lower() in {"model", "#"} or set(key) <= set("-: "):
            continue
        cells = [c.strip() for c in rest.split("|")]
        # a model row is all-numeric after the name
        if cells and all(re.fullmatch(r"\(?\d*\.?\d+\)?", c) for c in cells if c):
            rows[key] = cells
    return rows


def diff_tables(name: str, old: dict[str, list[str]], new: dict[str, list[str]],
                headers: list[str]) -> list[str]:
    lines = [f"### {name}", ""]
    lines.append("| Model | " + " | ".join(f"{h} (reviewed → now)" for h in headers) + " |")
    lines.append("|---" * (len(headers) + 1) + "|")
    for model in sorted(set(old) | set(new)):
        o = old.get(model)
        n = new.get(model)
        cells = []
        for k in range(len(headers)):
            ov = o[k] if o and k < len(o) else "--"
            nv = n[k] if n and k < len(n) else "--"
            cells.append(f"{ov} → {nv}" + ("" if ov == nv else "  **changed**"))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def first_group(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g
    return None


def main() -> None:
    reviewed = read_reviewed()
    current = CURRENT.read_text(encoding="utf-8")

    out: list[str] = [
        "# What changed between the reviewed version and this revision",
        "",
        f"Generated by `scripts/reviewed_version_delta.py`. Reviewed version = commit "
        f"`{REVIEWED_REV}` (2026-07-08), submitted 2026-07-13 as DD-ART-07-2026-000468 and "
        "transferred to RSC Advances as RA-ART-07-2026-006452.",
        "",
        "Every difference below is a consequence of the August 2026 audit and re-measurement "
        "(`tasks/audit-2026-08-16.md`), not of the referee reports, which arrived on 2026-09-11.",
        "",
    ]

    h_old = parse_model_table(section(reviewed, "### 3.1", "### 3.2"))
    h_new = parse_model_table(section(current, "### 3.1", "### 3.2"))
    out += diff_tables(
        "Harmonic baseline (§3.1)", h_old, h_new,
        ["Accuracy", "False-stable rate", "False-unstable rate"],
    )

    f_old = parse_model_table(section(reviewed, "### 3.2", "### 3.3"))
    f_new = parse_model_table(section(current, "### 3.2", "### 3.3"))
    out += diff_tables(
        "Finite-temperature screen (§3.2)", f_old, f_new,
        ["Finite-T false-stable rate", "Finite-T accuracy", "(Harmonic accuracy)"],
    )

    out += [
        "### Scalar claims in prose",
        "",
        "| Claim | Reviewed | Now | | Note |",
        "|---|---|---|---|---|",
    ]
    misses: list[str] = []
    for label, pat_old, pat_new, note in SCALAR_CLAIMS:
        o = first_group(pat_old, reviewed) if pat_old else None
        n = first_group(pat_new, current) if pat_new else None
        if pat_old and o is None:
            misses.append(f"{label} (reviewed)")
        if pat_new and n is None:
            misses.append(f"{label} (current)")
        o_s, n_s = (o or "*not found*"), (n or "*not found*")
        flag = "" if o_s == n_s else "**changed**"
        out.append(f"| {label} | {o_s} | {n_s} | {flag} | {note} |")
    out.append("")
    if misses:
        # Loud, not silent: an unmatched pattern means the disclosure table has a hole, and a
        # hole in a disclosure is the failure mode this whole script exists to prevent.
        out += [
            "> **Extraction incomplete.** These patterns did not match, so the rows above are "
            "not a complete record and must be checked by hand before the letter goes out: "
            + "; ".join(misses),
            "",
        ]
        print("WARNING: unmatched patterns -> " + "; ".join(misses))

    out += [
        "### The consequence that matters",
        "",
        "In the reviewed version the harmonic leaders (MatterSim, SevenNet-0, both 1.000) were "
        "*not* the finite-temperature leaders (CHGNet and MACE-MP-0, both 0.933). That clean "
        "inversion is the result Referee 3's summary describes. **It does not survive the "
        "re-measurement.** SevenNet-0 now leads both layers at 0.900, and the paper says so "
        "explicitly in §3.2.",
        "",
        "Note the direction of travel: the screen's headline recall fell from 0.77 to 0.53 and "
        "the contrast with SSCHA narrowed. The correction cost the paper its cleanest claim.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(out)} lines)")


if __name__ == "__main__":
    main()
