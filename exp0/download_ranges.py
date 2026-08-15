from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable parallel HTTP range downloader")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--sha256",
        help="Expected SHA-256. May be omitted for a slice later covered by a whole-file hash.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting byte offset within the remote object",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--parts",
        type=int,
        default=None,
        help="Number of stable byte ranges; defaults to --workers",
    )
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--read-timeout", type=int, default=180)
    return parser.parse_args()


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
    read_timeout: int,
    remote_offset: int,
) -> tuple[int, int]:
    expected_size = end - start + 1
    if part_path.is_file() and part_path.stat().st_size == expected_size:
        return start, expected_size

    temporary_path = part_path.with_suffix(part_path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            downloaded_size = temporary_path.stat().st_size if temporary_path.exists() else 0
            if downloaded_size > expected_size:
                temporary_path.unlink()
                downloaded_size = 0
            if downloaded_size == expected_size:
                os.replace(temporary_path, part_path)
                return start, expected_size
            request_start = start + downloaded_size
            remote_start = remote_offset + request_start
            remote_end = remote_offset + end
            with requests.get(
                url,
                headers={"Range": f"bytes={remote_start}-{remote_end}"},
                stream=True,
                timeout=(30, read_timeout),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {remote_start}-{remote_end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")
                with temporary_path.open("ab" if downloaded_size else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual_size = temporary_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"part size mismatch: expected {expected_size}, got {actual_size}"
                )
            os.replace(temporary_path, part_path)
            return start, expected_size
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.with_name(output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    part_count = max(1, int(args.parts)) if args.parts is not None else workers
    chunk_size = (args.size + part_count - 1) // part_count
    ranges: list[tuple[int, int, Path]] = []
    for index in range(part_count):
        start = index * chunk_size
        if start >= args.size:
            break
        end = min(args.size - 1, start + chunk_size - 1)
        ranges.append((start, end, parts_dir / f"part-{index:04d}"))

    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_part,
                args.url,
                part_path,
                start,
                end,
                int(args.retries),
                int(args.read_timeout),
                int(args.offset),
            ): (start, end)
            for start, end, part_path in ranges
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            _, downloaded = future.result()
            completed_bytes += downloaded
            print(
                f"parts={completed_count}/{len(ranges)} "
                f"bytes={completed_bytes}/{args.size}",
                flush=True,
            )

    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("wb") as destination:
        for _, _, part_path in ranges:
            with part_path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(chunk)
    if temporary_output.stat().st_size != args.size:
        raise RuntimeError("combined file size mismatch")
    actual_sha256 = sha256_file(temporary_output)
    if args.sha256 and actual_sha256.lower() != args.sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch: expected {args.sha256}, got {actual_sha256}"
        )
    os.replace(temporary_output, output)
    print(f"downloaded={output} sha256={actual_sha256}")


if __name__ == "__main__":
    main()
