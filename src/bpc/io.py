import errno
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


def merge_config(base, updates):
    """Merge nested objects and replace other values without sharing mutable state."""
    merged = deepcopy(base)
    for key, value in updates.items():
        merged[key] = (merge_config(merged[key], value)
                       if isinstance(value, dict) and isinstance(merged.get(key), dict)
                       else deepcopy(value))
    return merged


def read_config(path, _parents=()):
    """Resolve a single relative JSON base; merge objects and replace lists.

    Returns a self-contained configuration for recording in the run directory.
    Existing standalone configurations keep their original values and fit IDs.
    """
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f'cyclic configuration inheritance: {path}')
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f'configuration must be a JSON object: {path}')
    if 'extends' not in config:
        return config
    parent = config.pop('extends')
    if not isinstance(parent, str) or not parent:
        raise ValueError('extends must name one configuration file')
    base = read_config(path.parent / parent, (*_parents, path))
    return merge_config(base, config)


@contextmanager
def atomic_path(path):
    """Yield a sibling temporary path, then make it the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value, *, allow_nan=True):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=allow_nan) + "\n")


def write_text(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(value)


def stage_symlink(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source)
    os.replace(temporary, destination)


def stage_hardlink(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and os.path.samefile(source, destination):
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
