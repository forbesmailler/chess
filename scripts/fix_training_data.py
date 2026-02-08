"""Fix training data where search eval was stored as white's perspective instead of STM.

For every position where black is to move (byte 32 == 1), negates the
search eval float at bytes 35-38 to convert from white's perspective
to side-to-move perspective.
"""

import struct
import sys
from pathlib import Path

POSITION_SIZE = 42
SIDE_TO_MOVE_OFFSET = 32
EVAL_OFFSET = 35


def fix(path: str):
    data = bytearray(Path(path).read_bytes())
    n = len(data) // POSITION_SIZE
    fixed = 0

    for i in range(n):
        base = i * POSITION_SIZE
        stm = data[base + SIDE_TO_MOVE_OFFSET]
        if stm == 1:  # black to move
            offset = base + EVAL_OFFSET
            (val,) = struct.unpack_from("<f", data, offset)
            struct.pack_into("<f", data, offset, -val)
            fixed += 1

    Path(path).write_bytes(data)
    print(f"Fixed {fixed}/{n} positions ({fixed * 100 / n:.1f}% were black to move)")


if __name__ == "__main__":
    fix(sys.argv[1] if len(sys.argv) > 1 else "training_data.bin")
