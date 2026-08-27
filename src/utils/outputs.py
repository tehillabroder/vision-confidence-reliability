"""Protect saved project outputs from accidental replacement."""

from collections.abc import Iterable
from pathlib import Path
def check_output_paths(paths: Iterable[Path], overwrite: bool = False) -> None:
    if overwrite:
        return

    existing_paths = [Path(path) for path in paths if Path(path).exists()]
    if not existing_paths:
        return

    existing = ", ".join(str(path) for path in existing_paths)
    raise FileExistsError(f"Refusing to overwrite existing output: {existing}. Use --overwrite to replace existing evidence.")