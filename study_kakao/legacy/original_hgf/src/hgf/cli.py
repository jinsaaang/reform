"""Console entry points for the public HGF package."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .package import PACKAGE_ROOT


@contextmanager
def package_working_directory() -> Iterator[None]:
    """Resolve all frozen relative paths from the public repository root."""
    previous = Path.cwd()
    os.chdir(PACKAGE_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def replay_main() -> None:
    """Run HGF with the fixed exemplar artifacts."""
    from .runner import main

    with package_working_directory():
        main()


def main_table_main() -> None:
    """Run the six baselines and HGF under the shared paper protocol."""
    from .baselines import main

    with package_working_directory():
        main()
