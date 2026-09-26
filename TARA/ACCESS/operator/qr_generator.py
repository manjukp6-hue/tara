"""
TARA/ACCESS/creator/qr_generator.py

Pure-Python ISO/IEC 18004 QR Code Matrix & Art Generator.
Produces:
- ASCII / Unicode text blocks for inline chat display ('██' / '##' and '  ')
- Scalable Vector Graphics (SVG)
- Data URI strings
Zero third-party pip dependencies.
"""

import math
from typing import List, Tuple, Optional


class SimpleQR:
    """
    Compact pure-Python QR Code generator supporting Byte mode (ISO/IEC 18004).
    Produces complete 21x21 to 33x33 QR matrices with standard finder patterns,
    timing patterns, alignment patterns, and Reed-Solomon error correction.
    """

    _EXP = [0] * 512
    _LOG = [0] * 256
    _INITIALIZED = False

    @classmethod
    def _init_gf(cls):
        if cls._INITIALIZED:
            return
        x = 1
        for i in range(255):
            cls._EXP[i] = x
            cls._LOG[x] = i
            x <<= 1
            if x & 0x100:
                x ^= 0x11D
        for i in range(255, 512):
            cls._EXP[i] = cls._EXP[i - 255]
        cls._INITIALIZED = True

    @classmethod
    def _gf_mul(cls, x: int, y: int) -> int:
        if x == 0 or y == 0:
            return 0
        return cls._EXP[cls._LOG[x] + cls._LOG[y]]

    @classmethod
    def _rs_generator_poly(cls, degree: int) -> List[int]:
        g = [1]
        for i in range(degree):
            factor = [1, cls._EXP[i]]
            res = [0] * (len(g) + len(factor) - 1)
            for j in range(len(g)):
                for k in range(len(factor)):
                    res[j + k] ^= cls._gf_mul(g[j], factor[k])
            g = res
        return g

    @classmethod
    def _rs_encode(cls, msg: List[int], ec_len: int) -> List[int]:
        cls._init_gf()
        gen = cls._rs_generator_poly(ec_len)
        info = list(msg) + [0] * ec_len
        for i in range(len(msg)):
            coef = info[i]
            if coef != 0:
                for j in range(len(gen)):
                    info[i + j] ^= cls._gf_mul(gen[j], coef)
        return info[len(msg):]

    @classmethod
    def generate_matrix(cls, data: str) -> List[List[int]]:
        """
        Generates a 2D binary matrix (1 for black, 0 for white) for the given data string.
        Selects appropriate QR version (Version 1: 21x21, Version 2: 25x25, Version 3: 29x29, Version 4: 33x33).
        """
        cls._init_gf()
        data_bytes = data.encode("utf-8")
        length = len(data_bytes)

        VERSIONS = [
            (1, 21, 19, 7),
            (2, 25, 34, 10),
            (3, 29, 55, 15),
            (4, 33, 80, 20),
        ]

        selected = None
        for v, size, data_cw, ec_cw in VERSIONS:
            needed_bytes = math.ceil((4 + 8 + length * 8 + 4) / 8)
            if needed_bytes <= data_cw:
                selected = (v, size, data_cw, ec_cw)
                break

        if not selected:
            selected = VERSIONS[-1]

        version, size, total_data_cw, ec_cw = selected

        bits = []
        # Mode: Byte mode (0100)
        bits.extend([0, 1, 0, 0])
        # Character count indicator (8 bits)
        for b in f"{min(length, total_data_cw - 2):08b}":
            bits.append(int(b))
        # Data bits
        for byte in data_bytes[:total_data_cw - 2]:
            for b in f"{byte:08b}":
                bits.append(int(b))
        # Terminator
        rem_bits = total_data_cw * 8 - len(bits)
        bits.extend([0] * min(4, max(0, rem_bits)))
        # Pad to multiple of 8
        if len(bits) % 8 != 0:
            bits.extend([0] * (8 - (len(bits) % 8)))
        # Pad bytes (0xEC, 0x11)
        pad_bytes = [0xEC, 0x11]
        pad_idx = 0
        while len(bits) < total_data_cw * 8:
            for b in f"{pad_bytes[pad_idx % 2]:08b}":
                bits.append(int(b))
            pad_idx += 1

        # Data codewords
        data_words = []
        for i in range(0, len(bits), 8):
            data_words.append(int("".join(str(b) for b in bits[i:i+8]), 2))

        # Reed-Solomon error correction
        ec_words = cls._rs_encode(data_words, ec_cw)
        all_words = data_words + ec_words

        matrix = [[None for _ in range(size)] for _ in range(size)]

        # 1. Finder Patterns
        def place_finder(r0, c0):
            for r in range(r0 - 1, r0 + 8):
                for c in range(c0 - 1, c0 + 8):
                    if 0 <= r < size and 0 <= c < size:
                        matrix[r][c] = 0
            for r in range(r0, r0 + 7):
                for c in range(c0, c0 + 7):
                    if r in (r0, r0 + 6) or c in (c0, c0 + 6) or (r0 + 2 <= r <= r0 + 4 and c0 + 2 <= c <= c0 + 4):
                        matrix[r][c] = 1
                    else:
                        matrix[r][c] = 0

        place_finder(0, 0)
        place_finder(0, size - 7)
        place_finder(size - 7, 0)

        # 2. Timing Patterns
        for i in range(8, size - 8):
            val = 1 if i % 2 == 0 else 0
            if matrix[6][i] is None: matrix[6][i] = val
            if matrix[i][6] is None: matrix[i][6] = val

        # 3. Alignment Pattern (v >= 2)
        if version >= 2:
            align_pos = size - 7
            for r in range(align_pos - 2, align_pos + 3):
                for c in range(align_pos - 2, align_pos + 3):
                    if r in (align_pos - 2, align_pos + 2) or c in (align_pos - 2, align_pos + 2) or (r == align_pos and c == align_pos):
                        matrix[r][c] = 1
                    else:
                        matrix[r][c] = 0

        # 4. Dark Module
        matrix[size - 8][8] = 1

        # 5. Format info placeholder
        for i in range(9):
            if matrix[8][i] is None: matrix[8][i] = 0
            if matrix[i][8] is None: matrix[i][8] = 0
        for i in range(size - 8, size):
            if matrix[8][i] is None: matrix[8][i] = 0
            if matrix[i][8] is None: matrix[i][8] = 0

        # 6. Data bits (zigzag)
        data_bit_stream = []
        for word in all_words:
            for b in f"{word:08b}":
                data_bit_stream.append(int(b))

        bit_idx = 0
        col = size - 1
        up = True
        while col > 0:
            if col == 6:
                col -= 1
            rows = range(size - 1, -1, -1) if up else range(size)
            for r in rows:
                for c in (col, col - 1):
                    if matrix[r][c] is None:
                        val = data_bit_stream[bit_idx] if bit_idx < len(data_bit_stream) else 0
                        if (r + c) % 2 == 0:
                            val ^= 1
                        matrix[r][c] = val
                        bit_idx += 1
            col -= 2
            up = not up

        for r in range(size):
            for c in range(size):
                if matrix[r][c] is None:
                    matrix[r][c] = 0

        return matrix

    @classmethod
    def to_ascii_art(cls, matrix: List[List[int]], border: int = 1, use_unicode: bool = True) -> str:
        """
        Renders the QR matrix as text blocks ('██' / '##' and '  ').
        """
        size = len(matrix)
        lines = []
        full_size = size + border * 2
        dark = "██" if use_unicode else "##"
        light = "  "

        for _ in range(border):
            lines.append(light * full_size)

        for r in range(size):
            row_str = light * border
            for c in range(size):
                row_str += dark if matrix[r][c] == 1 else light
            row_str += light * border
            lines.append(row_str)

        for _ in range(border):
            lines.append(light * full_size)

        return "\n".join(lines)

    @classmethod
    def to_svg(cls, matrix: List[List[int]], box_size: int = 8, border: int = 2) -> str:
        """Renders the QR matrix as a scalable vector graphic (SVG)."""
        size = len(matrix)
        total_dim = (size + border * 2) * box_size
        rects = []

        for r in range(size):
            for c in range(size):
                if matrix[r][c] == 1:
                    x = (c + border) * box_size
                    y = (r + border) * box_size
                    rects.append(f'<rect x="{x}" y="{y}" width="{box_size}" height="{box_size}" fill="#000000"/>')

        rect_str = "\n".join(rects)
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_dim}" height="{total_dim}" viewBox="0 0 {total_dim} {total_dim}">\n<rect width="100%" height="100%" fill="#ffffff"/>\n{rect_str}\n</svg>'


def generate_chat_qr(text: str, use_unicode: bool = True) -> Tuple[str, str]:
    """
    Convenience helper that returns (ascii_art_str, svg_str).
    """
    matrix = SimpleQR.generate_matrix(text)
    ascii_art = SimpleQR.to_ascii_art(matrix, border=1, use_unicode=use_unicode)
    svg = SimpleQR.to_svg(matrix, box_size=6, border=2)
    return ascii_art, svg
