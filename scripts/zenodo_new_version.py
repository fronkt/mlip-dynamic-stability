"""Mint a new Zenodo version of record 20805824 with a fresh repo archive.

The live v1.0.0 record archives the pre-audit code and ledger: the single-mode screen, both
inverted acoustic masks, and the stale softmode grid. The manuscript now describes the
multi-mode v4 screen and the re-measured grids, so the record must advance with it.

Reads the token from ~/.config/zenodo/token (never printed). Metadata is taken from the
repo's .zenodo.json plus a version tag. Dry-run by default.

Usage:
    python scripts/zenodo_new_version.py                 # dry run: show planned actions
    python scripts/zenodo_new_version.py --apply v2.0.0  # create, upload, publish
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import urllib.request

DEP_ID = "20805824"
BASE = "https://zenodo.org/api/deposit/depositions"
ROOT = pathlib.Path(__file__).resolve().parent.parent

token = pathlib.Path(os.path.expanduser("~/.config/zenodo/token")).read_text().strip()


def call(method, url, payload=None, data=None, ctype=None):
    body = json.dumps(payload).encode() if payload is not None else data
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    elif ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req) as r:
            b = r.read()
            return json.loads(b) if b else {}
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on {method} {url}\n  {e.read().decode()[:500]}")
        raise


def main() -> int:
    apply = "--apply" in sys.argv
    version = sys.argv[sys.argv.index("--apply") + 1] if apply else "v2.0.0"

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("working tree is dirty - commit first; the archive must equal a commit")
        return 1

    zip_path = ROOT / f"mlip-dynamic-stability-{version}-{head}.zip"
    print(f"archive : {zip_path.name}  (git archive @ {head})")
    print(f"metadata: .zenodo.json + version={version}")
    if not apply:
        print("\nDRY RUN -- rerun with --apply <version> to mint.")
        return 0

    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip",
                    "-o", str(zip_path), "HEAD"], check=True)
    print(f"built {zip_path.name} ({zip_path.stat().st_size/1024:.0f} KB)")

    print("creating new version draft ...")
    nv = call("POST", f"{BASE}/{DEP_ID}/actions/newversion")
    draft_url = nv["links"]["latest_draft"]
    draft = call("GET", draft_url)
    did = draft["id"]
    print(f"draft deposition: {did}")

    # replace the old archive file with the new one
    for f in call("GET", f"{BASE}/{did}/files"):
        call("DELETE", f"{BASE}/{did}/files/{f['id']}")
        print(f"removed inherited file {f['filename']}")
    bucket = draft["links"]["bucket"]
    call("PUT", f"{bucket}/{zip_path.name}", data=zip_path.read_bytes(),
         ctype="application/octet-stream")
    print("uploaded new archive")

    meta = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    meta["version"] = version
    meta["publication_date"] = __import__("datetime").date.today().isoformat()
    call("PUT", f"{BASE}/{did}", {"metadata": meta})
    print("metadata written")

    pub = call("POST", f"{BASE}/{did}/actions/publish")
    print("\nPUBLISHED")
    print("  new version DOI :", pub.get("doi"))
    print("  concept DOI     :", pub.get("conceptdoi"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
