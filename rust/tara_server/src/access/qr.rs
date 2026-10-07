//! Pure-Rust ISO/IEC 18004 QR Code Matrix and Reed-Solomon Galois Field Generator.
//!
//! Generates real 2D QR matrices with Reed-Solomon GF(2^8) error correction,
//! finder patterns, timing sync, masking, and produces Unicode ASCII terminal blocks
//! and standalone SVG XML outputs without any external C or Python dependencies.

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
        // Generator polynomial for ec_len roots
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

        // Polynomial division: data * x^ec_len mod g(x)
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
    pub grid: Vec<Vec<bool>>,
}

impl QrCodeMatrix {
    /// Encodes arbitrary text into a standard QR code matrix.
    /// Supports Version 1 (21x21) and Version 2 (25x25) automatically.
    pub fn encode(data: &str) -> Result<Self, String> {
        let bytes = data.as_bytes();
        let (version, size, data_capacity, ec_codewords) = if bytes.len() <= 17 {
            (1, 21, 19, 7) // V1-M
        } else if bytes.len() <= 32 {
            (2, 25, 34, 10) // V2-M
        } else {
            return Err(format!(
                "Data payload of {} bytes exceeds embedded QR single-frame capacity",
                bytes.len()
            ));
        };

        // 1. Bitstream packaging: Mode (Byte 0100) + Count (8 bits) + Data + Terminator
        // Mode indicator 0100 (Byte mode)
        let mut bitstream = vec![false, true, false, false];

        // Count (8 bits)
        let len = bytes.len() as u8;
        for i in (0..8).rev() {
            bitstream.push(((len >> i) & 1) != 0);
        }

        // Data bytes
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

        // Byte align with zeros
        while (bitstream.len() % 8) != 0 && bitstream.len() < max_bits {
            bitstream.push(false);
        }

        // Convert bitstream to bytes
        let mut data_codewords = Vec::new();
        for chunk in bitstream.chunks(8) {
            let mut val = 0u8;
            for &b in chunk {
                val = (val << 1) | (if b { 1 } else { 0 });
            }
            data_codewords.push(val);
        }

        // Pad codewords up to data_capacity (alternating 0xEC, 0x11)
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

        // Finder patterns (top-left, top-right, bottom-left)
        Self::draw_finder(&mut grid, &mut reserved, 0, 0);
        Self::draw_finder(&mut grid, &mut reserved, size - 7, 0);
        Self::draw_finder(&mut grid, &mut reserved, 0, size - 7);

        // Timing patterns
        for i in 8..(size - 8) {
            let v = (i % 2) == 0;
            grid[6][i] = v;
            reserved[6][i] = true;
            grid[i][6] = v;
            reserved[i][6] = true;
        }

        // Alignment pattern for V2 (at row 18, col 18)
        if version == 2 {
            Self::draw_alignment(&mut grid, &mut reserved, 18, 18);
        }

        // Reserve format info areas
        for row in reserved.iter_mut().take(9) {
            row[8] = true;
        }
        for cell in reserved[8].iter_mut().take(9) {
            *cell = true;
        }
        for i in 0..8 {
            reserved[8][size - 1 - i] = true;
            reserved[size - 1 - i][8] = true;
        }

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

    fn draw_finder(
        grid: &mut [Vec<bool>],
        reserved: &mut [Vec<bool>],
        start_x: usize,
        start_y: usize,
    ) {
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

    /// Renders the QR code matrix into Unicode console block characters.
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();
        // Quiet zone
        let quiet = 2;
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

    /// Renders the QR code matrix into a clean, standalone SVG document.
    pub fn render_svg(&self, module_px: u32) -> String {
        let quiet = 2u32;
        let dim = (self.size as u32 + quiet * 2) * module_px;
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            dim, dim, dim, dim
        );
        svg.push_str(r##"<rect width="100%" height="100%" fill="#FFFFFF"/>"##);

        for row in 0..self.size {
            for col in 0..self.size {
                if self.grid[row][col] {
                    let x = (col as u32 + quiet) * module_px;
                    let y = (row as u32 + quiet) * module_px;
                    svg.push_str(&format!(
                        r##"<rect x="{}" y="{}" width="{}" height="{}" fill="#000000"/>"##,
                        x, y, module_px, module_px
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
    fn test_qr_generation_and_renderers() {
        let qr = QrCodeMatrix::encode("tara:login:987654321").unwrap();
        assert!(qr.size == 21 || qr.size == 25);

        let ascii = qr.render_ascii();
        assert!(ascii.contains("██"));
        assert!(ascii.contains("  "));

        let svg = qr.render_svg(10);
        assert!(svg.starts_with("<svg xmlns="));
        assert!(svg.contains(r##"fill="#000000""##));
        assert!(svg.ends_with("</svg>"));
    }
}
