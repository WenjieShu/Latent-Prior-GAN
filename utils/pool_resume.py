"""Guard streamed token pools against incompatible or nondeterministic resumes."""

import json
from pathlib import Path


def validate_shard_resume(metadata_path, signature, shard_path, progress_path, count, deterministic, restart=False):
    metadata_path = Path(metadata_path)
    shard_path, progress_path = Path(shard_path), Path(progress_path)
    if restart:
        for path in (metadata_path, shard_path, progress_path):
            path.unlink(missing_ok=True)
    exists = any(path.exists() for path in (metadata_path, shard_path, progress_path))
    if exists:
        if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != signature:
            raise ValueError("Pool resume configuration differs or metadata is absent; use --restart")
        if shard_path.exists() != progress_path.exists():
            raise ValueError("Incomplete shard/progress pair; use --restart")
    done = int(progress_path.read_text()) if progress_path.exists() else 0
    if not 0 <= done <= count:
        raise ValueError("Invalid pool progress; use --restart")
    if done and not deterministic:
        raise ValueError("Nondeterministic pools cannot resume without RNG state; use --restart or --deterministic")
    metadata_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
    return done
