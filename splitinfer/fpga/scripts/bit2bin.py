#!/usr/bin/env python3
"""bit2bin.py — strip Xilinx .bit metadata header and emit raw .bin payload.

Why this exists:
    Vivado 2025.2 added a `SW_CRC=...` field to the .bit metadata header
    that older versions of openFPGALoader (built before TRT 10.x equivalents
    in the FPGA flashing world) cannot parse — it errors with
    "Unknown key SW_CRC / BitParser: parseHeader failed".

    A Xilinx .bit file is just a small ASCII tagged-header followed by the
    raw FPGA configuration bitstream.  The header tags are:

        13-byte fixed magic prefix (0x00 0x09 0x0f 0xf0 0x0f 0xf0 0x0f 0xf0
                                    0x0f 0xf0 0x00 0x00 0x01)
        'a' tag : 2-byte BE length + ASCII design name
        'b' tag : 2-byte BE length + ASCII part name (e.g. "7a100tcsg324")
        'c' tag : 2-byte BE length + ASCII build date
        'd' tag : 2-byte BE length + ASCII build time
        'e' tag : 4-byte BE length + raw bitstream payload (the rest of the file)

    A .bin file is just the bytes after the 'e' tag's length field — no
    metadata at all.  openFPGALoader handles .bin files via `--file-type bin`
    and skips header parsing entirely, sidestepping the SW_CRC issue.

Usage:
    python3 bit2bin.py <input.bit> [<output.bin>]

If output is omitted, writes alongside input with .bin extension.
Returns exit code 0 on success, 1 on parse failure.
"""

import struct
import sys
import os


def bit2bin(in_path: str, out_path: str) -> int:
    with open(in_path, 'rb') as f:
        data = f.read()

    pos = 13  # skip the 13-byte fixed magic prefix
    while pos < len(data):
        tag = data[pos]
        pos += 1
        if tag == ord('e'):
            # 'e' tag: 4-byte big-endian length, then raw bitstream
            length = struct.unpack('>I', data[pos:pos + 4])[0]
            pos += 4
            bitstream = data[pos:pos + length]
            print(f"  Bitstream payload: {length} bytes (offset {pos})")
            with open(out_path, 'wb') as out:
                out.write(bitstream)
            print(f"  Wrote {out_path}")
            return 0
        else:
            # Other tags: 2-byte BE length + ASCII string
            slen = struct.unpack('>H', data[pos:pos + 2])[0]
            pos += 2
            val = data[pos:pos + slen].decode('ascii', errors='replace').rstrip('\x00')
            print(f"  Tag '{chr(tag)}': {val}")
            pos += slen

    print(f"  ERROR: no 'e' tag found in {in_path}", file=sys.stderr)
    return 1


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(__doc__, file=sys.stderr)
        return 2

    in_path = sys.argv[1]
    if not os.path.isfile(in_path):
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = sys.argv[2] if len(sys.argv) == 3 else os.path.splitext(in_path)[0] + ".bin"
    return bit2bin(in_path, out_path)


if __name__ == "__main__":
    sys.exit(main())
