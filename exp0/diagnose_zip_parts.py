from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path


def data_range(archive: Path, info: zipfile.ZipInfo) -> tuple[int, int]:
    with archive.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"invalid local header for {info.filename}")
    filename_length, extra_length = struct.unpack_from("<HH", header, 26)
    start = info.header_offset + 30 + filename_length + extra_length
    return start, start + max(0, info.compress_size - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map corrupt ZIP entries to range parts")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--part-size", type=int, required=True)
    args = parser.parse_args()

    corrupt_parts: set[int] = set()
    corrupt_entries = 0
    with zipfile.ZipFile(args.archive) as package:
        print(f"entries={len(package.infolist())}", flush=True)
        for info in package.infolist():
            source = None
            try:
                with package.open(info) as source:
                    while source.read(8 * 1024 * 1024):
                        pass
            except Exception as error:
                corrupt_entries += 1
                start, end = data_range(args.archive, info)
                first = start // args.part_size
                last = end // args.part_size
                parts = list(range(first, last + 1))
                corrupt_parts.update(parts)
                shared_file = getattr(source, "_fileobj", None)
                stream_position = getattr(shared_file, "_pos", None)
                stream_part = (
                    stream_position // args.part_size
                    if isinstance(stream_position, int)
                    else None
                )
                print(
                    f"corrupt={info.filename!r} error={type(error).__name__} "
                    f"message={str(error)!r} bytes={start}-{end} parts={parts} "
                    f"stream_position={stream_position} stream_part={stream_part}",
                    flush=True,
                )
    print(f"corrupt_entries={corrupt_entries}", flush=True)
    print("candidate_parts=" + ",".join(str(part) for part in sorted(corrupt_parts)))


if __name__ == "__main__":
    main()
