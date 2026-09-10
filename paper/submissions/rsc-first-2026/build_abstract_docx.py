"""Fill the official RSC FIRST 2026 abstract template from abstract.md.

The template (downloaded 2026-09-10 from the Abstract Submission page via the redirect
/en/web/jump/36624?mid=3190186&nid=148374) carries RSC + Xiamen letterhead art in its
headers, a Title style, and a numbered reference list. Regenerating the document with
pandoc would throw all of that away, so instead we surgically replace the placeholder
paragraphs in word/document.xml and leave every other part byte-identical.

abstract.md is the single source of truth: edit the Markdown, re-run this, never edit
the DOCX by hand.

Usage:
    python build_abstract_docx.py [--template PATH] [--out PATH]

Template requirements enforced here:
  * Times New Roman throughout (inherited from the template's run properties)
  * abstract body <= 300 words  (hard-fails if exceeded)
  * Keywords line present
  * References in RSC style, as the template's numbered list
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "rsc_first_2026_abstract_template.docx"
DEFAULT_OUT = HERE / "abstract-rsc-first-2026.docx"
SOURCE_MD = HERE / "abstract.md"

WORD_LIMIT = 300

# Run properties copied from the template so the output inherits its exact typography.
FONTS = (
    '<w:rFonts w:ascii="Times New Roman" w:eastAsia="SimSun" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
)


def rpr(bold=False, italic=False, underline=False, superscript=False, size=21):
    """Build a <w:rPr> matching the template's conventions."""
    parts = [FONTS]
    if bold:
        parts.append("<w:b/><w:bCs/>")
    if italic:
        parts.append("<w:i/><w:iCs/>")
    parts.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if underline:
        parts.append('<w:u w:val="single"/>')
    if superscript:
        parts.append('<w:vertAlign w:val="superscript"/>')
    parts.append('<w:lang w:val="en-GB"/>')
    return "<w:rPr>" + "".join(parts) + "</w:rPr>"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def run(text: str, **kw) -> str:
    # xml:space="preserve" so leading/trailing spaces between runs survive.
    return (
        "<w:r>"
        + rpr(**kw)
        + f'<w:t xml:space="preserve">{esc(text)}</w:t>'
        + "</w:r>"
    )


def italic_runs(text: str, size=21):
    """Split RSC-style Markdown (*italic*, **bold**) into runs."""
    out = []
    for chunk in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            out.append(run(chunk[2:-2], bold=True, size=size))
        elif chunk.startswith("*") and chunk.endswith("*"):
            out.append(run(chunk[1:-1], italic=True, size=size))
        else:
            out.append(run(chunk, size=size))
    return "".join(out)


def para(ppr: str, inner: str) -> str:
    return f"<w:p>{ppr}{inner}</w:p>"


# ---------------------------------------------------------------- parse source


def parse_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    def section(name: str) -> str:
        m = re.search(
            rf"^##\s+{name}[^\n]*\n(.*?)(?=^##\s+|\Z)", text, re.S | re.M
        )
        if not m:
            sys.exit(f"ERROR: section '## {name}' not found in {path.name}")
        return m.group(1).strip()

    title = " ".join(section("Title").split())

    author_block = [ln.strip() for ln in section("Author").splitlines() if ln.strip()]
    if len(author_block) < 4:
        sys.exit("ERROR: '## Author' needs name / affiliation / email / ORCID lines")
    name, affiliation, email = author_block[0], author_block[1], author_block[2]

    body = section(r"Abstract[^\n]*")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]

    keywords = " ".join(section("Keywords").split())

    refs = []
    for line in section("References").splitlines():
        m = re.match(r"^\s*\d+\.\s+(.*\S)\s*$", line)
        if m:
            refs.append(m.group(1))
    if not refs:
        sys.exit("ERROR: no numbered references parsed from '## References'")

    return dict(
        title=title,
        name=name,
        affiliation=affiliation,
        email=email,
        paragraphs=paragraphs,
        keywords=keywords,
        refs=refs,
    )


# ---------------------------------------------------------------- build docx


def build(template: Path, out: Path, src: dict) -> int:
    with zipfile.ZipFile(template) as z:
        doc = z.read("word/document.xml").decode("utf-8")
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    body_start = doc.index("<w:body>")
    head, body = doc[:body_start], doc[body_start:]
    paras = re.findall(r"<w:p\b.*?</w:p>", body, re.S)
    if len(paras) != 10:
        sys.exit(f"ERROR: template shape changed ({len(paras)} paragraphs, expected 10)")

    def ppr_of(idx: int) -> str:
        m = re.search(r"<w:pPr>.*?</w:pPr>", paras[idx], re.S)
        return m.group(0) if m else ""

    # 0 Title
    new = [para(ppr_of(0), run(src["title"], size=None or 21))]
    # Title style controls size; drop our explicit sz so the style wins.
    new[0] = para(
        ppr_of(0),
        f'<w:r><w:rPr>{FONTS}</w:rPr><w:t xml:space="preserve">'
        f'{esc(src["title"])}</w:t></w:r>',
    )

    # 1 Author line: presenting author underlined, superscript affiliation marker.
    new.append(
        para(
            ppr_of(1),
            run(src["name"], bold=True, underline=True)
            + run("1,*", bold=True, superscript=True),
        )
    )

    # 2 Affiliation (single affiliation -> template paragraph 3 is dropped).
    new.append(
        para(
            ppr_of(2),
            run("1", superscript=True) + run(" " + src["affiliation"]),
        )
    )

    # 4 Corresponding email
    new.append(para(ppr_of(4), run(f'* Email: {src["email"]}')))

    # 5 "Abstract:" heading, kept verbatim from the template
    new.append(paras[5])

    # 6 Abstract body, one template-styled paragraph per source paragraph
    for p in src["paragraphs"]:
        new.append(para(ppr_of(6), italic_runs(p)))

    # 7 Keywords
    new.append(
        para(
            ppr_of(7),
            run("Keywords: ", bold=True, size=22) + run(src["keywords"]),
        )
    )

    # 8 "References:" heading, verbatim
    new.append(paras[8])

    # 9 References as the template's numbered list
    for ref in src["refs"]:
        new.append(para(ppr_of(9), italic_runs(ref)))

    tail_start = body.index(paras[-1]) + len(paras[-1])
    new_body = "<w:body>" + "".join(new) + body[tail_start:]
    new_doc = head + new_body

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, new_doc.encode("utf-8") if n == "word/document.xml" else blobs[n])
    shutil.move(str(tmp), str(out))
    return len(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.template.exists():
        sys.exit(
            f"ERROR: template not found at {args.template}\n"
            "Download it from the Abstract Submission page:\n"
            "  https://www.rscfirst.org.cn/en/web/jump/36624?mid=3190186&nid=148374"
        )

    src = parse_markdown(SOURCE_MD)

    words = sum(len(p.split()) for p in src["paragraphs"])
    if words > WORD_LIMIT:
        sys.exit(f"ERROR: abstract is {words} words, template limit is {WORD_LIMIT}")

    n = build(args.template, args.out, src)
    print(f"abstract body : {words} words (limit {WORD_LIMIT}, headroom {WORD_LIMIT - words})")
    print(f"references    : {len(src['refs'])}")
    print(f"paragraphs out: {n}")
    print(f"wrote         : {args.out}")


if __name__ == "__main__":
    main()
