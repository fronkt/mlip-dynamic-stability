"""Fill the official RSC FIRST 2026 Travel Support application form from travel-support.md.

The form is a Word document of seven tables with RSC/XMU letterhead art. We fill it by
editing word/document.xml in place -- inserting runs into the empty value cells and
ticking the factual checkboxes -- so every other part of the file stays byte-identical.

travel-support.md is the single source of truth: edit the Markdown, re-run this.

DELIBERATELY LEFT BLANK for Frank (see the checklist in travel-support.md):
  * the two Section 4 declaration checkboxes -- personal attestations
  * Applicant Signature
  * Date

Usage:
    python build_travel_support_docx.py [--form PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FORM = HERE / "rsc_first_2026_travel_support_form.docx"
DEFAULT_OUT = HERE / "travel-support-rsc-first-2026.docx"
SOURCE_MD = HERE / "travel-support.md"

# Only w:t, never w:tc / w:tcPr / w:tbl -- the trailing char class must not swallow them.
W_T = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
CELL = re.compile(r"<w:tc>.*?</w:tc>", re.S)
PARA = re.compile(r"<w:p\b(?:(?!</w:p>).)*</w:p>", re.S)


def norm(text: str) -> str:
    """Collapse whitespace so run-splitting and template padding don't defeat matching."""
    return " ".join(text.split())


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell_text(cell: str) -> str:
    return "".join(W_T.findall(cell)).strip()


def make_run(text: str) -> str:
    return (
        '<w:r><w:rPr><w:lang w:val="en-GB"/></w:rPr>'
        f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
    )


def fill_cell(cell: str, paragraphs: list[str]) -> str:
    """Insert text into an empty table cell, reusing its existing paragraph as the first."""
    m = re.search(r"<w:p\b(?:(?!</w:p>).)*</w:p>", cell, re.S)
    if not m:
        sys.exit("ERROR: cell has no paragraph to fill")
    proto = m.group(0)

    def with_text(p: str, text: str) -> str:
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", p, re.S)
        open_tag = re.match(r"<w:p\b[^>]*>", p).group(0)
        return open_tag + (ppr.group(0) if ppr else "") + make_run(text) + "</w:p>"

    built = "".join(with_text(proto, t) for t in paragraphs)
    return cell[: m.start()] + built + cell[m.end():]


# ---------------------------------------------------------------- parse source


def parse_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    def section(name: str) -> str:
        m = re.search(rf"^##\s+{name}[^\n]*\n(.*?)(?=^##\s+|\Z)", text, re.S | re.M)
        if not m:
            sys.exit(f"ERROR: section '## {name}' not found in {path.name}")
        return m.group(1).strip()

    fields = {}
    for line in section("Form fields").splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and "---" not in m.group(1) and m.group(1).strip() != "Field":
            fields[m.group(1).strip()] = m.group(2).strip()

    required = [
        "Full Name", "Email", "Affiliation / Institution", "Department / School",
        "Country / Region", "Career Stage", "Abstract Submission Status", "Abstract Title",
    ]
    missing = [k for k in required if k not in fields]
    if missing:
        sys.exit(f"ERROR: missing form fields: {missing}")

    def flow(name: str) -> list[str]:
        body = re.sub(r"^-{3,}\s*$", "", section(name), flags=re.M)  # drop MD rules
        blocks = re.split(r"\n\s*\n", body)
        return [" ".join(b.split()) for b in blocks if b.strip()]

    return dict(fields=fields, objectives=flow(r"Section 3a[^\n]*"), need=flow(r"Section 3b[^\n]*"))


# ---------------------------------------------------------------- build


def build(form: Path, out: Path, src: dict) -> dict:
    with zipfile.ZipFile(form) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    doc = blobs["word/document.xml"].decode("utf-8")
    f = src["fields"]
    report = {}

    # --- 1. tick the factual checkboxes (declarations are left for Frank) ---
    # The template splits "[   ] Student" across several runs, so match on the
    # paragraph's collapsed text and rebuild it as one run.
    ticked = []
    for label in (f["Career Stage"], f["Abstract Submission Status"]):
        want = norm(f"[ ] {label}")
        for m in PARA.finditer(doc):
            p = m.group(0)
            if norm("".join(W_T.findall(p))) != want:
                continue
            ppr = re.search(r"<w:pPr>.*?</w:pPr>", p, re.S)
            open_tag = re.match(r"<w:p\b[^>]*>", p).group(0)
            new_p = (
                open_tag
                + (ppr.group(0) if ppr else "")
                + make_run(f"[ X ] {label}")
                + "</w:p>"
            )
            doc = doc[: m.start()] + new_p + doc[m.end():]
            ticked.append(label)
            break
        else:
            sys.exit(f"ERROR: checkbox paragraph not found in form: {want!r}")
    report["ticked"] = ticked

    # --- 2. fill the label->value cells ---
    labels = {
        "Full Name": f["Full Name"],
        "Email": f["Email"],
        "Affiliation / Institution": f["Affiliation / Institution"],
        "Department / School": f["Department / School"],
        "Country / Region": f["Country / Region"],
        "Abstract Title:": f["Abstract Title"],
    }
    filled = []
    for label, value in labels.items():
        cells = list(CELL.finditer(doc))
        idx = next((i for i, c in enumerate(cells) if cell_text(c.group(0)) == label), None)
        if idx is None or idx + 1 >= len(cells):
            sys.exit(f"ERROR: no value cell found next to label {label!r}")
        target = cells[idx + 1]
        if cell_text(target.group(0)):
            sys.exit(f"ERROR: cell after {label!r} is not empty: {cell_text(target.group(0))!r}")
        doc = doc[: target.start()] + fill_cell(target.group(0), [value]) + doc[target.end():]
        filled.append(label)
    report["filled"] = filled

    # --- 3. the two free-text statement cells ---
    # Anchor each to its own prompt paragraph and take the first cell of the NEXT table.
    # (Picking "the first empty cells" is wrong: a vMerge continuation cell in the Career
    # Stage table and the spare cell beside "Not submitting an abstract" both come first.)
    prompts = [
        ("Please outline your primary objectives", src["objectives"]),
        ("Please provide a brief statement", src["need"]),
    ]
    # Fill bottom-up so earlier offsets stay valid.
    for prompt, paras in reversed(prompts):
        anchor = next(
            (m for m in PARA.finditer(doc) if prompt in norm("".join(W_T.findall(m.group(0))))),
            None,
        )
        if anchor is None:
            sys.exit(f"ERROR: prompt paragraph not found: {prompt!r}")
        tbl = re.compile(r"<w:tbl>.*?</w:tbl>", re.S).search(doc, anchor.end())
        if tbl is None:
            sys.exit(f"ERROR: no table follows the prompt {prompt!r}")
        cell = CELL.search(tbl.group(0))
        if cell is None or cell_text(cell.group(0)):
            sys.exit(f"ERROR: statement cell after {prompt!r} missing or already filled")
        start = tbl.start() + cell.start()
        end = tbl.start() + cell.end()
        doc = doc[:start] + fill_cell(cell.group(0), paras) + doc[end:]
    report["statements"] = {
        "objectives_paras": len(src["objectives"]),
        "objectives_words": sum(len(p.split()) for p in src["objectives"]),
        "need_paras": len(src["need"]),
        "need_words": sum(len(p.split()) for p in src["need"]),
    }
    report["left_blank"] = sum(1 for c in CELL.finditer(doc) if not cell_text(c.group(0)))

    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, doc.encode("utf-8") if n == "word/document.xml" else blobs[n])
    shutil.move(str(tmp), str(out))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", type=Path, default=DEFAULT_FORM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.form.exists():
        sys.exit(
            f"ERROR: form not found at {args.form}\nDownload it from the Registration page:\n"
            "  https://files.sciconf.cn/public/2117/54869/202608/20260803155935_55653.docx"
        )

    src = parse_markdown(SOURCE_MD)
    r = build(args.form, args.out, src)

    print("ticked        :", ", ".join(r["ticked"]))
    print("fields filled :", ", ".join(r["filled"]))
    s = r["statements"]
    print(f"objectives    : {s['objectives_words']} words in {s['objectives_paras']} para(s)")
    print(f"need          : {s['need_words']} words in {s['need_paras']} para(s)")
    print(f"left blank    : {r['left_blank']} cells (declarations / signature / date -- by design)")
    print("wrote         :", args.out)


if __name__ == "__main__":
    main()
