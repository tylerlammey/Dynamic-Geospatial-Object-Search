"""Check tracked Markdown local links, repository command paths, and changed whitespace.

External links and generated runtime paths are not fetched or executed. Commands
are checked for repository entry-point existence, not run. Use --files to include
new documents before staging; unrelated untracked documents are ignored.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+[^)]*)?\)")
REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]*>|\S+)", re.MULTILINE)
COMMAND = re.compile(r"\.\\((?:scripts|frontend)\\[^\s`'\";|]+\.(?:ps1|py|mjs))")
MODULE = re.compile(r"\s-m\s+(geospatial_search\.[a-zA-Z_][a-zA-Z_0-9]*)")


def git(*arguments: str, root: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
    )


def tracked_markdown(root: Path = ROOT) -> list[Path]:
    result = git("ls-files", "-z", root=root)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Cannot enumerate tracked files")
    return [root / path for path in result.stdout.split("\0") if path.lower().endswith(".md")]


def check_document(path: Path, root: Path = ROOT) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    errors = []
    prose = []
    commands = []
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence is None:
            prose.append((number, line))
        else:
            commands.append((number, line))
    for number, line in prose:
        # Inline code is literal; it may intentionally contain sample Markdown.
        line = re.sub(r"`+[^`]*`+", "", line)
        targets = [match.group(1) for match in LINK.finditer(line)]
        targets.extend(match.group(1) for match in REFERENCE.finditer(line))
        for target in targets:
            target = target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = path.parent / unquote(parsed.path).replace("\\", "/")
            if not destination.exists():
                errors.append(f"{path.relative_to(root)}:{number}: missing link target {target}")
    for number, line in commands:
        for match in COMMAND.finditer(line):
            target = root / match.group(1).replace("\\", "/")
            # Packaged executables and runtime artifacts are outside this pattern.
            if not target.is_file():
                errors.append(
                    f"{path.relative_to(root)}:{number}: missing command {match.group(0)}"
                )
        for match in MODULE.finditer(line):
            target = root / "src" / (match.group(1).replace(".", "/") + ".py")
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}:{number}: missing module {match.group(1)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", default=[], help="Additional repo-relative Markdown")
    args = parser.parse_args()
    paths = set(tracked_markdown())
    for name in args.files:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() != ".md":
            parser.error("--files must name Markdown files inside the repository")
        paths.add(path)
    errors = []
    for path in sorted(paths):
        if not path.is_file():
            # A tracked deletion is legitimate; remaining links still get checked.
            continue
        errors.extend(check_document(path))
    for arguments in (("diff", "--check"), ("diff", "--cached", "--check")):
        result = git(*arguments)
        if result.returncode:
            errors.append(result.stdout.strip() or result.stderr.strip())
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked {len(paths)} tracked/selected Markdown files and changed whitespace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
