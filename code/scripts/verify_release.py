#!/usr/bin/env python3
"""Read-only package inventory check; --full also hashes every file."""
import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(
        p in ('.', '..') or ':' in p or '\\' in p for p in parts
    ):
        raise ValueError('Invalid relative inventory path: ' + relative)
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Inventory path escapes package: ' + relative)
    for node in (path, *path.parents):
        if node == root.parent:
            break
        attrs = node.stat(follow_symlinks=False)
        if stat.S_ISLNK(attrs.st_mode) or getattr(attrs, 'st_file_attributes', 0) & 1024:
            raise ValueError('Linked/reparse path is not supported: ' + relative)
    return path


def check(root, full):
    summary = {}
    for subject in ('code', 'data'):
        directory = root / subject
        index = json.loads((directory / 'SHA256SUMS.json').read_text(encoding='utf-8'))
        assert index['excludes_own_index'] is True
        rows = index['rows']
        expected = {row['relative'] for row in rows}
        assert len(expected) == len(rows), 'Duplicate index paths'
        actual = set()
        for path in directory.rglob('*'):
            if path.name in ('.aws', '.codex', '.agents', '.ssh', '.git'):
                raise ValueError('Unexpected private directory in release')
            attrs = path.stat(follow_symlinks=False)
            if stat.S_ISLNK(attrs.st_mode) or getattr(attrs, 'st_file_attributes', 0) & 1024:
                raise ValueError('Linked/reparse path in release')
            if path.is_file() and path != directory / 'SHA256SUMS.json':
                actual.add(path.relative_to(directory).as_posix())
        assert actual == expected, 'Missing or unindexed ' + subject + ' files'
        for row in rows:
            path = safe_path(directory, row['relative'])
            assert path.stat().st_size == row['bytes'], row['relative']
            if full:
                assert identity(path) == row['sha256'], row['relative']
        summary[subject] = {'indexed_files': len(rows), 'indexed_bytes': sum(r['bytes'] for r in rows)}
    return {'status': 'PASS', 'sha256_of_every_file_checked': full, 'subjects': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--full', action='store_true', help='Read and hash every indexed byte, including the validation cache')
    args = parser.parse_args()
    print(json.dumps(check(args.root.resolve(), args.full), indent=2))


if __name__ == '__main__':
    main()
