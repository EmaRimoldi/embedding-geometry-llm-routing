#!/usr/bin/env python
"""Lint OL/main.tex: banned lexicon, remaining \\missing{}, undefined references/citations and
multiply-defined labels from the .log, page count of the main text (via the final "Output
written on ... (N pages" line), and where the main text ends: main.aux is parsed for
\\newlabel{sec:limitations} (falling back to the \\appendix anchor label sec:appendix-start)
and the page recorded there is printed as "main text ends on page N". Exit 1 on any hard
failure (undefined refs, multiply-defined labels, banned words); the page budget is a warning
unless --enforce-pages is passed, in which case N > MAX_MAIN_PAGES also exits 1. The .log/.aux
may be absent (e.g. before a first compile); those checks are then skipped and reported as
"no log"/"no aux" rather than erroring (ruling 2)."""
from __future__ import annotations
import re, sys
from pathlib import Path

MAX_MAIN_PAGES = 9
MAIN_END_LABELS = ("sec:limitations", "sec:appendix-start")
BANNED = ["delve", "crucial", "pivotal", "intricate", "nuanced", "landscape", "realm", "underscore", "showcase", "notably",
          "it is worth noting", "plays a key role", "a testament to", "paving the way", "in summary", "overall, these results"]


def main_end_page(aux: Path) -> tuple[str | None, int | None]:
    """Page of the first present label in MAIN_END_LABELS, from ``\\newlabel{k}{{..}{page}..}``."""
    if not aux.exists():
        return None, None
    A = aux.read_text(errors="ignore")
    for key in MAIN_END_LABELS:
        m = re.search(r"\\newlabel\{" + re.escape(key) + r"\}\{\{[^{}]*\}\{(\d+)\}", A)
        if m:
            return key, int(m.group(1))
    return None, None


def main() -> None:
    argv = [x for x in sys.argv[1:] if not x.startswith("--")]
    enforce_pages = "--enforce-pages" in sys.argv[1:]
    tex = Path(argv[0]); src = tex.read_text(); body = re.sub(r"(?<!\\)%.*", "", src)
    main_text = body.split("\\appendix")[0]
    bad = [(w, len(re.findall(rf"\b{re.escape(w)}\b", main_text, flags=re.I))) for w in BANNED]
    bad = [(w, n) for w, n in bad if n]
    lev = len(re.findall(r"\bleverag(e|es|ed|ing)\b", main_text, flags=re.I)); missing = len(re.findall(r"\\missing\{", body))

    log = tex.with_suffix(".log")
    undef: list = []; multiply: list = []; pages: list = []
    if log.exists():
        L = log.read_text(errors="ignore")
        undef = re.findall(r"(Reference|Citation) `([^']+)' .*undefined", L)
        multiply = re.findall(r"Label `([^']+)' multiply defined", L)
        pages = re.findall(r"Output written on .* \((\d+) pages", L)
        pages_str = pages[-1] if pages else "?"
    else:
        pages_str = "no log"

    aux = tex.with_suffix(".aux")
    end_label, end_page = main_end_page(aux)

    print("banned words:", bad or "none", "| 'leverage' uses:", lev, "| \\missing{}:", missing,
          "| undefined refs:", (len(undef) if log.exists() else "no log"),
          "| multiply defined labels:", (len(multiply) if log.exists() else "no log"),
          "| total pages:", pages_str)
    if not log.exists():
        print("  no log: .log file not found at", log, "- skipping log-derived checks")
    if end_page is None:
        print("  no aux: no", "/".join(MAIN_END_LABELS), "in", aux, "- skipping the main-text page check")
    else:
        print(f"main text ends on page {end_page} (label {end_label}; budget {MAX_MAIN_PAGES})")
    for w in undef[:20]: print("  undefined:", w)
    for w in multiply[:20]: print("  multiply defined:", w)
    over = end_page is not None and end_page > MAX_MAIN_PAGES
    if over:
        print(f"{'ERROR' if enforce_pages else 'WARNING'}: main text ends on page {end_page} > {MAX_MAIN_PAGES}"
              + ("" if enforce_pages else " (pass --enforce-pages to make this a hard failure)"))
    sys.exit(1 if (bad or undef or multiply or (over and enforce_pages)) else 0)


if __name__ == "__main__":
    main()
