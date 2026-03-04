//! A typed, owned GPU-ready column buffer.

use anyhow::{bail, Result};
use arrow_schema::DataType;

/// Supported GPU-native data types (we widen everything to 64-bit on the GPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDataType {
    Int64,
    Float64,
    /// 8-byte epoch-ms timestamp stored as Int64.
    TimestampMs,
    /// Fixed-width UTF-8 stored as Int64 (dictionary-encoded or hash).
    DictEncodedString,
}

impl GpuDataType {
    /// Byte width of one element.
    pub fn byte_width(self) -> usize {
        8 // all types are 64-bit on the GPU
    }

    /// Try to map an Arrow `DataType` to a `GpuDataType`.
    pub fn from_arrow(dt: &DataType) -> Result<Self> {
        match dt {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => Ok(Self::Int64),

            DataType::Float32 | DataType::Float64 => Ok(Self::Float64),

            DataType::Timestamp(_, _) | DataType::Date32 | DataType::Date64 => {
                Ok(Self::TimestampMs)
            }

            DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(_, _) => {
                Ok(Self::DictEncodedString)
            }

            other => bail!("Unsupported Arrow DataType for GPU: {:?}", other),
        }
    }
}

/// An owned buffer of GPU-ready column data.
#[derive(Debug, Clone)]
pub struct ColumnBuffer {
    pub name: String,
    pub dtype: GpuDataType,
    /// Raw bytes in GPU layout (little-endian, packed, no nullmap).
    pub data: Vec<u8>,
    /// Number of rows represented.
    pub n_rows: usize,
    /// Bitmask: 1 = valid, 0 = null (one byte per row for simplicity).
    pub validity: Option<Vec<u8>>,
}

impl ColumnBuffer {
    pub fn new_empty(name: impl Into<String>, dtype: GpuDataType) -> Self {
        Self { name: name.into(), dtype, data: Vec::new(), n_rows: 0, validity: None }
    }

    pub fn from_bytes(
        name: impl Into<String>,
        dtype: GpuDataType,
        data: Vec<u8>,
        n_rows: usize,
        validity: Option<Vec<u8>>,
    ) -> Result<Self> {
        let expected = n_rows * dtype.byte_width();
        if data.len() != expected {
            bail!(
                "ColumnBuffer size mismatch: expected {} bytes for {} rows, got {}",
                expected, n_rows, data.len()
            );
        }
        Ok(Self { name: name.into(), dtype, data, n_rows, validity })
    }

    pub fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
    pub fn byte_len(&self) -> usize { self.data.len() }
}
