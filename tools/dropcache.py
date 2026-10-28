"""Drop the page cache of named files. posix_fadvise(DONTNEED) on own files: advisory, no root.

Before a server reboot in the same allocation, drop only the two Engram shards (README, "Hardware and software"): their page cache
plus the Engram shared memory otherwise pushed our 700 GiB job past its memory limit, and TP ranks 1-3 loaded in
~1,000 s instead of ~490 s. The other weight shards stay cached.

    python tools/dropcache.py $DSV41_MODEL_DIR/model-00047-of-00048.safetensors $DSV41_MODEL_DIR/model-00048-of-00048.safetensors
"""

import os
import sys


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: dropcache.py FILE [FILE ...]", file=sys.stderr)
        return 2
    total = 0
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        total += size
        print(f"fadvised DONTNEED {p} ({size / 1e9:.1f} GB)")
    print(f"{len(paths)} files, {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
