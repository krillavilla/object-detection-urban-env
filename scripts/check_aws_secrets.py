#!/usr/bin/env python3
"""Reject likely AWS credentials without printing matching secret values."""

import argparse
import re
import subprocess
import sys


PATTERNS = (
    re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    # Build field names in pieces so the scanner does not flag its own source.
    re.compile(b"aws_" + b"secret_access_key", re.IGNORECASE),
    re.compile(b"aws_" + b"session_token", re.IGNORECASE),
)


def git(*args):
    return subprocess.check_output(["git", *args])


def scan(data):
    return any(pattern.search(data) for pattern in PATTERNS)


def staged_blobs():
    # Scan index contents, not working files: partially staged secrets must fail.
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    for entry in entries:
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if mode == b"160000":  # A submodule entry is not a blob.
            continue
        yield oid.decode(), path.decode(errors="replace")


def history_blobs():
    for entry in git("rev-list", "--objects", "--all").splitlines():
        oid, _, path = entry.partition(b" ")
        oid = oid.decode()
        if git("cat-file", "-t", oid).strip() == b"blob":
            yield oid, path.decode(errors="replace") or oid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", action="store_true", help="Scan all reachable Git history."
    )
    args = parser.parse_args()
    findings = []
    seen = set()
    blobs = history_blobs() if args.history else staged_blobs()
    for oid, path in blobs:
        if oid in seen:
            continue
        seen.add(oid)
        if scan(git("cat-file", "blob", oid)):
            findings.append(path)
    if findings:
        print("STOP: Possible AWS credentials detected; values withheld.", file=sys.stderr)
        for path in sorted(set(findings)):
            print(f"  {path}", file=sys.stderr)
        print("Do not commit or push. Remove credentials first.", file=sys.stderr)
        return 1
    print("AWS credential scan passed (pattern-based, not a security guarantee).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
