//! Strict native Rust RFC 8949 CBOR Decoder and Encoder for WebAuthn Authenticator Data.
//!
//! Provides zero-dependency, fail-closed parsing of CBOR maps, arrays, strings, and integers
//! adhering to RFC 8949 and W3C WebAuthn Level 2 / Level 3 specifications.
//! Enforces:
//! - Strict bounds checking (no buffer overflows).
//! - Duplicate map key rejection (fail closed against key confusion attacks).
//! - Maximum recursion depth ceiling (protection against stack exhaustion).
//! - Exact type enforcement.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CborValue {
    Unsigned(u64),
    Negative(u64),
    Bytes(Vec<u8>),
    Text(String),
    Array(Vec<CborValue>),
    Map(Vec<(CborValue, CborValue)>),
    Bool(bool),
    Null,
}

impl CborValue {
    pub fn as_unsigned(&self) -> Option<u64> {
        match self {
            CborValue::Unsigned(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            CborValue::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[CborValue]> {
        match self {
            CborValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&[(CborValue, CborValue)]> {
        match self {
            CborValue::Map(m) => Some(m.as_slice()),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            CborValue::Bytes(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            CborValue::Unsigned(n) => (*n).try_into().ok(),
            CborValue::Negative(n) => {
                let n_i64: Result<i64, _> = (*n).try_into();
                n_i64.ok().map(|v| -1 - v)
            }
            _ => None,
        }
    }

    pub fn get_map_text_key(&self, key: &str) -> Option<&CborValue> {
        if let CborValue::Map(entries) = self {
            for (k, v) in entries {
                if let CborValue::Text(t) = k {
                    if t == key {
                        return Some(v);
                    }
                }
            }
        }
        None
    }
}

pub struct CborDecoder<'a> {
    buf: &'a [u8],
    offset: usize,
    depth: usize,
}

const MAX_CBOR_DEPTH: usize = 16;
const MAX_CBOR_COLLECTION_LEN: usize = 256;

impl<'a> CborDecoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            offset: 0,
            depth: 0,
        }
    }

    pub fn decode(&mut self) -> Result<CborValue, String> {
        let val = self.decode_value()?;
        Ok(val)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.offset >= self.buf.len() {
            return Err("Unexpected end of CBOR buffer".to_string());
        }
        let b = self.buf[self.offset];
        self.offset += 1;
        Ok(b)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.offset + len > self.buf.len() {
            return Err(format!(
                "CBOR buffer truncated: requested {} bytes at offset {}, available {}",
                len,
                self.offset,
                self.buf.len() - self.offset
            ));
        }
        let slice = &self.buf[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    fn read_arg(&mut self, ai: u8) -> Result<u64, String> {
        match ai {
            0..=23 => Ok(ai as u64),
            24 => {
                let b = self.read_u8()?;
                Ok(b as u64)
            }
            25 => {
                let slice = self.read_exact(2)?;
                let bytes: [u8; 2] = slice.try_into().unwrap();
                Ok(u16::from_be_bytes(bytes) as u64)
            }
            26 => {
                let slice = self.read_exact(4)?;
                let bytes: [u8; 4] = slice.try_into().unwrap();
                Ok(u32::from_be_bytes(bytes) as u64)
            }
            27 => {
                let slice = self.read_exact(8)?;
                let bytes: [u8; 8] = slice.try_into().unwrap();
                Ok(u64::from_be_bytes(bytes))
            }
            _ => Err(format!("Unsupported CBOR additional information: {}", ai)),
        }
    }

    fn decode_value(&mut self) -> Result<CborValue, String> {
        if self.depth >= MAX_CBOR_DEPTH {
            return Err("CBOR recursion depth ceiling exceeded".to_string());
        }

        let header = self.read_u8()?;
        let major = header >> 5;
        let ai = header & 0x1f;

        match major {
            0 => {
                let val = self.read_arg(ai)?;
                Ok(CborValue::Unsigned(val))
            }
            1 => {
                let val = self.read_arg(ai)?;
                Ok(CborValue::Negative(val))
            }
            2 => {
                let len = self.read_arg(ai)? as usize;
                if len > self.buf.len() {
                    return Err(format!("CBOR byte string length {} exceeds buffer", len));
                }
                let bytes = self.read_exact(len)?;
                Ok(CborValue::Bytes(bytes.to_vec()))
            }
            3 => {
                let len = self.read_arg(ai)? as usize;
                if len > self.buf.len() {
                    return Err(format!("CBOR text string length {} exceeds buffer", len));
                }
                let bytes = self.read_exact(len)?;
                let s = String::from_utf8(bytes.to_vec())
                    .map_err(|e| format!("Invalid UTF-8 in CBOR text: {}", e))?;
                Ok(CborValue::Text(s))
            }
            4 => {
                let len = self.read_arg(ai)? as usize;
                if len > MAX_CBOR_COLLECTION_LEN {
                    return Err(format!("CBOR array length {} exceeds limit", len));
                }
                self.depth += 1;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.decode_value()?);
                }
                self.depth -= 1;
                Ok(CborValue::Array(items))
            }
            5 => {
                let len = self.read_arg(ai)? as usize;
                if len > MAX_CBOR_COLLECTION_LEN {
                    return Err(format!("CBOR map length {} exceeds limit", len));
                }
                self.depth += 1;
                let mut entries = Vec::with_capacity(len);
                let mut seen_keys = HashSet::new();

                for _ in 0..len {
                    let key = self.decode_value()?;
                    let key_str = match &key {
                        CborValue::Text(t) => format!("t:{}", t),
                        CborValue::Unsigned(u) => format!("u:{}", u),
                        CborValue::Negative(n) => format!("n:{}", n),
                        CborValue::Bytes(b) => format!("b:{}", hex::encode(b)),
                        other => format!("{:?}", other),
                    };

                    if !seen_keys.insert(key_str) {
                        return Err("Duplicate key detected in CBOR map (fail closed)".to_string());
                    }

                    let val = self.decode_value()?;
                    entries.push((key, val));
                }
                self.depth -= 1;
                Ok(CborValue::Map(entries))
            }
            7 => match ai {
                20 => Ok(CborValue::Bool(false)),
                21 => Ok(CborValue::Bool(true)),
                22 => Ok(CborValue::Null),
                _ => Err(format!("Unsupported CBOR simple/float value: {}", ai)),
            },
            _ => Err(format!("Unsupported CBOR major type: {}", major)),
        }
    }
}

pub struct CborEncoder;

impl CborEncoder {
    pub fn encode(val: &CborValue) -> Vec<u8> {
        let mut out = Vec::new();
        Self::encode_val(val, &mut out);
        out
    }

    fn write_header(major: u8, arg: u64, out: &mut Vec<u8>) {
        let major_bits = major << 5;
        if arg <= 23 {
            out.push(major_bits | (arg as u8));
        } else if arg <= 0xff {
            out.push(major_bits | 24);
            out.push(arg as u8);
        } else if arg <= 0xffff {
            out.push(major_bits | 25);
            out.extend_from_slice(&(arg as u16).to_be_bytes());
        } else if arg <= 0xffff_ffff {
            out.push(major_bits | 26);
            out.extend_from_slice(&(arg as u32).to_be_bytes());
        } else {
            out.push(major_bits | 27);
            out.extend_from_slice(&arg.to_be_bytes());
        }
    }

    fn encode_val(val: &CborValue, out: &mut Vec<u8>) {
        match val {
            CborValue::Unsigned(n) => {
                Self::write_header(0, *n, out);
            }
            CborValue::Negative(n) => {
                Self::write_header(1, *n, out);
            }
            CborValue::Bytes(b) => {
                Self::write_header(2, b.len() as u64, out);
                out.extend_from_slice(b);
            }
            CborValue::Text(s) => {
                Self::write_header(3, s.len() as u64, out);
                out.extend_from_slice(s.as_bytes());
            }
            CborValue::Array(arr) => {
                Self::write_header(4, arr.len() as u64, out);
                for item in arr {
                    Self::encode_val(item, out);
                }
            }
            CborValue::Map(entries) => {
                Self::write_header(5, entries.len() as u64, out);
                for (k, v) in entries {
                    Self::encode_val(k, out);
                    Self::encode_val(v, out);
                }
            }
            CborValue::Bool(false) => {
                out.push((7 << 5) | 20);
            }
            CborValue::Bool(true) => {
                out.push((7 << 5) | 21);
            }
            CborValue::Null => {
                out.push((7 << 5) | 22);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbor_roundtrip_primitives_and_maps() {
        let val = CborValue::Map(vec![
            (CborValue::Text("key1".to_string()), CborValue::Unsigned(42)),
            (
                CborValue::Text("key2".to_string()),
                CborValue::Text("hello".to_string()),
            ),
            (
                CborValue::Text("key3".to_string()),
                CborValue::Array(vec![CborValue::Bool(true), CborValue::Null]),
            ),
        ]);
        let bytes = CborEncoder::encode(&val);
        let mut decoder = CborDecoder::new(&bytes);
        let decoded = decoder.decode().expect("clean decode");

        match decoded {
            CborValue::Map(entries) => {
                assert_eq!(entries.len(), 3);
                assert_eq!(entries[0].0, CborValue::Text("key1".to_string()));
                assert_eq!(entries[0].1, CborValue::Unsigned(42));
                assert_eq!(entries[1].0, CborValue::Text("key2".to_string()));
                assert_eq!(entries[1].1, CborValue::Text("hello".to_string()));
            }
            other => panic!("expected Map, got {:?}", other),
        }
    }

    #[test]
    fn test_cbor_rejects_duplicate_keys() {
        let raw = vec![
            0xa2, // map of 2 items
            0x63, b'k', b'e', b'y', 0x01, // "key" => 1
            0x63, b'k', b'e', b'y', 0x02, // "key" => 2 (duplicate!)
        ];
        let mut decoder = CborDecoder::new(&raw);
        let res = decoder.decode();
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Duplicate key detected"));
    }

    #[test]
    fn test_cbor_rejects_excessive_depth() {
        let mut raw = vec![0x81; 20]; // array of 1 item nested 20 times
        raw.push(0x00); // 0
        let mut decoder = CborDecoder::new(&raw);
        let res = decoder.decode();
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("depth ceiling exceeded"));
    }

    #[test]
    fn test_cbor_type_prefixed_keys_no_collision() {
        let val = CborValue::Map(vec![
            (CborValue::Text("u:1".to_string()), CborValue::Unsigned(100)),
            (CborValue::Unsigned(1), CborValue::Unsigned(200)),
        ]);
        let bytes = CborEncoder::encode(&val);
        let mut decoder = CborDecoder::new(&bytes);
        let decoded = decoder.decode();
        assert!(decoded.is_ok(), "Text('u:1') and Unsigned(1) must not collide: {:?}", decoded.err());
    }
}
