//! Pure-Rust ISO/IEC 18004 QR Code Matrix and Reed-Solomon Galois Field Generator.
//!
//! Dedicated, separate QR matrix generation subsystem:
//! - Reed-Solomon GF(2^8) error correction polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D = 285).
//! - Version 1 (21x21) and Version 2 (25x25) Level-L matrices.
//! - Finder patterns, timing lines, alignment, dark module, and 15-bit BCH format info.
//! - Unicode ASCII terminal rendering with 4-module quiet zone.
//! - Clean standalone SVG XML vector output.
//! - Strictly standalone, zero external crate dependencies beyond serde.

use serde::{Deserialize, Serialize};

/// GF(2^8) with generator polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D = 285)
struct GaloisField {
    exp_table: [u8; 512],
    log_table: [u8; 256],
}

impl GaloisField {
    fn new() -> Self {
        let mut exp_table = [0u8; 512];
        let mut log_table = [0u8; 256];
        let mut x = 1u16;

        for i in 0..255 {
            exp_table[i] = x as u8;
            exp_table[i + 255] = x as u8;
            log_table[x as usize] = i as u8;
            x <<= 1;
            if (x & 0x100) != 0 {
                x ^= 0x11D;
            }
        }

        Self {
            exp_table,
            log_table,
        }
    }

    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            0
        } else {
            let log_a = self.log_table[a as usize] as usize;
            let log_b = self.log_table[b as usize] as usize;
            self.exp_table[log_a + log_b]
        }
    }

    /// Computes Reed-Solomon error correction codewords.
    fn compute_error_correction(&self, data: &[u8], ec_len: usize) -> Vec<u8> {
        let mut g = vec![1u8];
        for i in 0..ec_len {
            let root = self.exp_table[i];
            let mut next_g = vec![0u8; g.len() + 1];
            for (j, &c) in g.iter().enumerate() {
                next_g[j] ^= self.mul(c, root);
                next_g[j + 1] ^= c;
            }
            g = next_g;
        }

        let mut remainder = vec![0u8; ec_len];
        for &byte in data {
            let factor = byte ^ remainder[0];
            remainder.remove(0);
            remainder.push(0);
            if factor != 0 {
                for (j, &gc) in g.iter().enumerate().take(ec_len) {
                    remainder[j] ^= self.mul(gc, factor);
                }
            }
        }

        remainder
    }
}

/// A 2D QR Code Matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QrCodeMatrix {
    pub size: usize,
    grid: Vec<Vec<bool>>,
}

impl QrCodeMatrix {
    /// Encodes arbitrary text into an ISO/IEC 18004 compliant QR code matrix.
    /// Supports Version 1 (21x21) and Version 2 (25x25) Level-L.
    pub fn encode(data: &str) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot encode empty data into QR code matrix".into());
        }
        let bytes = data.as_bytes();
        let (version, size, data_capacity, ec_codewords) = if bytes.len() <= 17 {
            (1, 21, 19, 7) // V1-L: 19 data bytes, 7 EC bytes
        } else if bytes.len() <= 32 {
            (2, 25, 34, 10) // V2-L: 34 data bytes, 10 EC bytes
        } else if bytes.len() <= 53 {
            (3, 29, 55, 15) // V3-L: 55 data bytes, 15 EC bytes
        } else {
            return Err(format!(
                "Data payload of {} bytes exceeds embedded QR single-frame capacity",
                bytes.len()
            ));
        };

        // 1. Bitstream packaging: Mode (Byte 0100) + Count (8 bits) + Data + Terminator
        let mut bitstream = vec![false, true, false, false];

        let len = bytes.len() as u8;
        for i in (0..8).rev() {
            bitstream.push(((len >> i) & 1) != 0);
        }

        for &b in bytes {
            for i in (0..8).rev() {
                bitstream.push(((b >> i) & 1) != 0);
            }
        }

        // Terminator (up to 4 zero bits)
        let max_bits = data_capacity * 8;
        for _ in 0..4 {
            if bitstream.len() < max_bits {
                bitstream.push(false);
            }
        }

        while (bitstream.len() % 8) != 0 && bitstream.len() < max_bits {
            bitstream.push(false);
        }

        let mut data_codewords = Vec::new();
        for chunk in bitstream.chunks(8) {
            let mut val = 0u8;
            for &b in chunk {
                val = (val << 1) | (if b { 1 } else { 0 });
            }
            data_codewords.push(val);
        }

        let pad_bytes = [0xEC, 0x11];
        let mut pad_idx = 0;
        while data_codewords.len() < data_capacity {
            data_codewords.push(pad_bytes[pad_idx]);
            pad_idx = (pad_idx + 1) % 2;
        }

        // 2. Reed-Solomon Error Correction
        let gf = GaloisField::new();
        let ec = gf.compute_error_correction(&data_codewords, ec_codewords);
        let mut final_message = data_codewords;
        final_message.extend(ec);

        // 3. Matrix generation
        let mut grid = vec![vec![false; size]; size];
        let mut reserved = vec![vec![false; size]; size];

        // Finder patterns with white separators
        Self::draw_finder_with_separator(&mut grid, &mut reserved, 0, 0, size);
        Self::draw_finder_with_separator(&mut grid, &mut reserved, size - 7, 0, size);
        Self::draw_finder_with_separator(&mut grid, &mut reserved, 0, size - 7, size);

        // Timing patterns
        for i in 8..(size - 8) {
            let v = (i % 2) == 0;
            grid[6][i] = v;
            reserved[6][i] = true;
            grid[i][6] = v;
            reserved[i][6] = true;
        }

        // Dark module at row (size - 8), column 8
        grid[size - 8][8] = true;
        reserved[size - 8][8] = true;

        // Alignment pattern for V2 (at row 18, col 18) or V3 (at row 22, col 22)
        if version == 2 {
            Self::draw_alignment(&mut grid, &mut reserved, 18, 18);
        } else if version == 3 {
            Self::draw_alignment(&mut grid, &mut reserved, 22, 22);
        }

        // 15-bit BCH Format Information for EC Level L (01) and Mask 000 (000):
        // Raw data = 01000b, BCH code + XOR mask 0x5412 = 0x77C4 = 111011111000100b
        const FORMAT_BITS: u16 = 0x77C4;
        Self::write_format_info(&mut grid, &mut reserved, FORMAT_BITS, size);

        // 4. Place data bits in zigzag right-to-left
        let mut all_bits = Vec::new();
        for &byte in &final_message {
            for i in (0..8).rev() {
                all_bits.push(((byte >> i) & 1) != 0);
            }
        }

        let mut bit_idx = 0;
        let mut col = size as isize - 1;
        let mut upward = true;

        while col > 0 {
            if col == 6 {
                col -= 1; // Skip vertical timing line
            }

            let rows: Vec<usize> = if upward {
                (0..size).rev().collect()
            } else {
                (0..size).collect()
            };

            for row in rows {
                for c in [col, col - 1] {
                    let cu = c as usize;
                    if !reserved[row][cu] {
                        let bit = if bit_idx < all_bits.len() {
                            all_bits[bit_idx]
                        } else {
                            false
                        };
                        bit_idx += 1;

                        // Standard Mask 000: (row + col) % 2 == 0
                        let mask = (row + cu).is_multiple_of(2);
                        grid[row][cu] = bit ^ mask;
                    }
                }
            }

            upward = !upward;
            col -= 2;
        }

        Ok(Self { size, grid })
    }

    pub fn is_dark(&self, row: usize, col: usize) -> bool {
        if row < self.size && col < self.size {
            self.grid[row][col]
        } else {
            false
        }
    }

    fn draw_finder_with_separator(
        grid: &mut [Vec<bool>],
        reserved: &mut [Vec<bool>],
        start_x: usize,
        start_y: usize,
        size: usize,
    ) {
        // Draw 7x7 finder
        for dy in 0..7 {
            for dx in 0..7 {
                let is_black = dx == 0
                    || dx == 6
                    || dy == 0
                    || dy == 6
                    || ((2..=4).contains(&dx) && (2..=4).contains(&dy));
                grid[start_y + dy][start_x + dx] = is_black;
                reserved[start_y + dy][start_x + dx] = true;
            }
        }

        // Draw 1-module white separator border around finder
        let x_range = if start_x == 0 {
            0..=7
        } else {
            (start_x - 1)..=(start_x + 6)
        };
        let y_range = if start_y == 0 {
            0..=7
        } else {
            (start_y - 1)..=(start_y + 6)
        };

        for y in y_range {
            for x in x_range.clone() {
                if x < size && y < size && !reserved[y][x] {
                    grid[y][x] = false;
                    reserved[y][x] = true;
                }
            }
        }
    }

    fn write_format_info(
        grid: &mut [Vec<bool>],
        reserved: &mut [Vec<bool>],
        format_bits: u16,
        size: usize,
    ) {
        // Format bits: bit 0 is LSB, bit 14 is MSB
        for i in 0..15 {
            let bit = ((format_bits >> (14 - i)) & 1) != 0;

            // 1. Around top-left finder
            let (r1, c1) = if i < 6 {
                (i, 8)
            } else if i == 6 {
                (7, 8)
            } else if i == 7 {
                (8, 8)
            } else if i == 8 {
                (8, 7)
            } else {
                (8, 14 - i)
            };
            grid[r1][c1] = bit;
            reserved[r1][c1] = true;

            // 2. Split between top-right and bottom-left finders
            let (r2, c2) = if i < 8 {
                (8, size - 1 - i)
            } else {
                (size - 15 + i, 8)
            };
            grid[r2][c2] = bit;
            reserved[r2][c2] = true;
        }
    }

    fn draw_alignment(
        grid: &mut [Vec<bool>],
        reserved: &mut [Vec<bool>],
        center_x: usize,
        center_y: usize,
    ) {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let x = (center_x as isize + dx) as usize;
                let y = (center_y as isize + dy) as usize;
                let is_black = dx.abs() == 2 || dy.abs() == 2 || (dx == 0 && dy == 0);
                grid[y][x] = is_black;
                reserved[y][x] = true;
            }
        }
    }

    /// Renders the QR code matrix into Unicode console block characters with standard 4-module quiet zone.
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();
        let quiet = 4;
        let total_size = self.size + quiet * 2;

        for _ in 0..quiet {
            out.push_str(&"  ".repeat(total_size));
            out.push('\n');
        }

        for row in 0..self.size {
            out.push_str(&"  ".repeat(quiet));
            for col in 0..self.size {
                if self.grid[row][col] {
                    out.push_str("██");
                } else {
                    out.push_str("  ");
                }
            }
            out.push_str(&"  ".repeat(quiet));
            out.push('\n');
        }

        for _ in 0..quiet {
            out.push_str(&"  ".repeat(total_size));
            out.push('\n');
        }

        out
    }

    /// Renders the QR code matrix into a clean, standalone SVG document with standard 4-module quiet zone.
    pub fn render_svg(&self, module_px: u32) -> String {
        let safe_px = module_px.max(1);
        let quiet = 4u32;
        let dim = (self.size as u32 + quiet * 2) * safe_px;
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            dim, dim, dim, dim
        );
        svg.push_str(r##"<rect width="100%" height="100%" fill="#FFFFFF"/>"##);

        for row in 0..self.size {
            for col in 0..self.size {
                if self.grid[row][col] {
                    let x = (col as u32 + quiet) * safe_px;
                    let y = (row as u32 + quiet) * safe_px;
                    svg.push_str(&format!(
                        r##"<rect x="{}" y="{}" width="{}" height="{}" fill="#000000"/>"##,
                        x, y, safe_px, safe_px
                    ));
                }
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_encode_rejects_empty_data() {
        let res = QrCodeMatrix::encode("");
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Cannot encode empty data"));
    }

    #[test]
    fn test_qr_encode_valid_payload() {
        let res = QrCodeMatrix::encode("tara:sec:test");
        assert!(res.is_ok());
        let matrix = res.unwrap();
        assert_eq!(matrix.size, 21); // V1-L
        assert!(!matrix.render_ascii().is_empty());
        assert!(!matrix.render_svg(4).is_empty());
    }
}
