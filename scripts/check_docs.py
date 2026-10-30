#!/usr/bin/env python3
"""Check portable documentation links and literal versioned command paths offline."""
import argparse
from collections import Counter, deque
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from verify_release import ROOT, public_files


class HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.anchors = [], set()

    def handle_starttag(self, tag, attributes):
        for key, value in attributes:
            if value and (key == 'id' or tag == 'a' and key == 'name'):
                self.anchors.add(value)
            if value and key in {'href', 'src', 'poster'}:
                self.links.append((value, self.getpos()[0]))

    handle_startendtag = handle_starttag


def without_fences(text):
    result, fence = [], None
    for line in text.splitlines(keepends=True):
        match = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            result.append('\n')
        elif match:
            fence = match[1]
            result.append('\n')
        else:
            result.append(line)
    return ''.join(result)


def markdown_anchors(text):
    result, counts = set(), Counter()
    lines = without_fences(text).splitlines()
    for index, line in enumerate(lines):
        match = re.match(r'^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$', line)
        title = match[1] if match else None
        if title is None and index and re.fullmatch(r' {0,3}(?:=+|-+)\s*', line):
            title = lines[index - 1].strip()
        if not title:
            continue
        title = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', title)
        title = html.unescape(re.sub(r'<[^>]+>', '', title)).lower()
        slug = re.sub(r'[^\w\s-]', '', title)
        slug = re.sub(r'\s', '-', slug)
        anchor = slug if counts[slug] == 0 else slug + '-' + str(counts[slug])
        counts[slug] += 1
        result.add(anchor)
    return result


LINK = re.compile(r'\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+["\'][^\n]*?["\'])?\s*\)')
REFERENCE = re.compile(r'^ {0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)', re.M)
PATH = re.compile(r'(?<![\w.-])(?:src|scripts|jobs|configs|docs|reproducibility|experiments|environment|manifests|release_provenance)/[\w./*?\[\]{}<>$-]+')


def check(root=ROOT):
    root = Path(root).resolve()
    files = {p.relative_to(root) for p in public_files(root)}
    directories = {parent for path in files for parent in path.parents}
    documents = sorted(p for p in files if p.suffix.lower() in {'.md', '.html'}
                       and (len(p.parts) == 1 or p.parts[0] == 'docs' or p == Path('reproducibility/README.md')))
    texts = {p: (root / p).read_text() for p in documents}
    anchors, links, graph = {}, {}, {}
    errors, local_links, source_paths = [], 0, 0
    for path, text in texts.items():
        visible = without_fences(text) if path.suffix == '.md' else text
        parser = HTMLReferences()
        parser.feed(visible)
        anchors[path] = parser.anchors | (markdown_anchors(text) if path.suffix == '.md' else set())
        links[path] = list(parser.links)
        if path.suffix == '.md':
            for pattern in [LINK, REFERENCE]:
                links[path].extend((m[1].strip('<>'), visible.count('\n', 0, m.start()) + 1)
                                   for m in pattern.finditer(visible))
        graph[path] = set()

    def problem(path, line, value, reason):
        errors.append(dict(file=str(path), line=line, target=value, problem=reason))

    for path, targets in links.items():
        for value, line in targets:
            value = html.unescape(value)
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc:
                continue
            local_links += 1
            if parsed.path.startswith('/'):
                problem(path, line, value, 'Use a relative repository link')
                continue
            target = (root / path.parent / unquote(parsed.path)).resolve() if parsed.path else root / path
            if not target.is_relative_to(root):
                problem(path, line, value, 'Link escapes the repository')
                continue
            relative = target.relative_to(root)
            if relative not in files and relative not in directories:
                problem(path, line, value, 'Missing exported file or directory')
                continue
            if relative in directories and relative / 'README.md' in files:
                relative /= 'README.md'
            if relative in graph:
                graph[path].add(relative)
            if parsed.fragment and relative in anchors and unquote(parsed.fragment) not in anchors[relative]:
                problem(path, line, value, 'Missing heading or HTML anchor')

    # Literal paths use repository-root semantics throughout the manual.
    # Generated data/results and variable placeholders are deliberately separate.
    command_sources = {p: text for p, text in texts.items() if p.suffix == '.md'}
    command_sources.update({p: (root / p).read_text() for p in files if p.parts[0] == 'jobs'
                            and p.suffix in {'.sbatch', '.sh'}})
    command_sources.update({p: (root / p).read_text() for p in files
                            if p == Path('Makefile') or p.parts[:2] == ('.github', 'workflows')})
    for path, text in command_sources.items():
        text = LINK.sub('', text)
        for match in PATH.finditer(text):
            value = match[0].rstrip('.,:;')
            if '...' in value or any(token in value for token in '<>{}$'):
                continue
            source_paths += 1
            candidate = Path(value)
            exists = candidate in files or candidate in directories
            if any(token in value for token in '*?['):
                exists = any(p.match(str(candidate).rstrip('/')) for p in files | directories)
            if not exists:
                problem(path, text.count('\n', 0, match.start()) + 1, value,
                        'Missing versioned path in a command or inline reference')

    reachable, pending = set(), deque([Path('README.md')])
    while pending:
        path = pending.popleft()
        if path in reachable:
            continue
        reachable.add(path)
        pending.extend(graph.get(path, ()))
    orphaned = sorted(str(path) for path in documents if path.suffix == '.md' and path not in reachable)
    for path in orphaned:
        problem(Path(path), 1, path, 'Manual is not reachable from the repository README')
    return dict(passed=not errors, documents=len(documents), local_links=local_links,
                versioned_paths=source_paths, unreachable_documents=orphaned, errors=errors,
                scope='Offline exported-file links, Markdown/HTML anchors, manual reachability and '
                      'literal versioned paths in manuals, job launchers, Makefile and CI. External URLs and generated '
                      'runtime inputs/outputs are not executed or fetched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = check(args.root)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
