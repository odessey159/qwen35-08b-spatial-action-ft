from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from ai2thor.build import Build
from ai2thor.platform import CloudRendering, Linux64


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a verified AI2-THOR build archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--releases-dir", type=Path, required=True)
    parser.add_argument("--commit-id", required=True)
    parser.add_argument(
        "--platform",
        choices=("CloudRendering", "Linux64"),
        default="CloudRendering",
    )
    args = parser.parse_args()

    platform = {"CloudRendering": CloudRendering, "Linux64": Linux64}[args.platform]

    archive = args.archive.resolve()
    releases_dir = args.releases_dir.resolve()
    releases_dir.mkdir(parents=True, exist_ok=True)
    build = Build(
        platform=platform,
        commit_id=args.commit_id,
        include_private_scenes=False,
        releases_dir=str(releases_dir),
    )
    destination = Path(build.base_dir)
    if destination.is_dir():
        print(f"already_installed={destination}")
        return

    temporary = Path(build.tmp_dir) / f"{build.name}.installing"
    if temporary.exists():
        raise FileExistsError(f"stale installation directory exists: {temporary}")
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive) as package:
        package.extractall(temporary)

    executable = Path(build.platform.executable_path(str(temporary), build.name))
    if not executable.is_file():
        raise FileNotFoundError(f"AI2-THOR executable missing after extraction: {executable}")
    os.chmod(executable, 0o755)
    os.replace(temporary, destination)
    print(f"installed={destination}")


if __name__ == "__main__":
    main()
