//! Conversion between Arrow `RecordBatch` and `Vec<ColumnBuffer>`.

use anyhow::{bail, Context, Result};
use arrow_array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::column_buffer::{ColumnBuffer, GpuDataType};

pub fn record_batch_to_gpu_buffers(batch: &RecordBatch) -> Result<Vec<ColumnBuffer>> {
    let n_rows = batch.num_rows();
    let mut buffers = Vec::with_capacity(batch.num_columns());
    for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
        let buf = column_to_buffer(field.name(), col.as_ref(), n_rows)
            .with_context(|| format!("Converting column '{}'", field.name()))?;
        buffers.push(buf);
    }
    Ok(buffers)
}

fn column_to_buffer(name: &str, array: &dyn Array, n_rows: usize) -> Result<ColumnBuffer> {
    let validity: Option<Vec<u8>> = if array.null_count() > 0 {
        Some((0..n_rows).map(|i| if array.is_valid(i) { 1u8 } else { 0u8 }).collect())
    } else {
        None
    };
    match array.data_type() {
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let data = int_to_i64_bytes(a.iter());
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let data = float_to_f64_bytes(a.iter());
            ColumnBuffer::from_bytes(name, GpuDataType::Float64, data, n_rows, validity)
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            let data = string_to_hash_bytes(a.iter());
            ColumnBuffer::from_bytes(name, GpuDataType::DictEncodedString, data, n_rows, validity)
        }
        other => bail!("Unsupported column type for GPU conversion: {:?}", other),
    }
}

fn int_to_i64_bytes(iter: impl Iterator<Item = Option<i64>>) -> Vec<u8> {
    iter.flat_map(|v| v.unwrap_or(0i64).to_le_bytes()).collect()
}

fn float_to_f64_bytes(iter: impl Iterator<Item = Option<f64>>) -> Vec<u8> {
    iter.flat_map(|v| v.unwrap_or(0.0f64).to_le_bytes()).collect()
}

fn string_to_hash_bytes(iter: impl Iterator<Item = Option<&'static str>>) -> Vec<u8> {
    iter.flat_map(|v| {
        let hash: i64 = match v { Some(s) => fnv1a_hash(s) as i64, None => 0i64 };
        hash.to_le_bytes()
    }).collect()
}

fn fnv1a_hash(s: &str) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

pub fn gpu_buffers_to_record_batch(buffers: &[ColumnBuffer], schema: Arc<Schema>) -> Result<RecordBatch> {
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(buffers.len());
    for (buf, field) in buffers.iter().zip(schema.fields()) {
        columns.push(buffer_to_array(buf, field)?);
    }
    RecordBatch::try_new(schema, columns).context("Building RecordBatch from GPU buffers")
}

fn buffer_to_array(buf: &ColumnBuffer, field: &Field) -> Result<Arc<dyn Array>> {
    let n = buf.n_rows;
    match field.data_type() {
        DataType::Int64 => {
            let mut b = arrow_array::builder::Int64Builder::with_capacity(n);
            for i in 0..n {
                let val = i64::from_le_bytes(buf.data[i*8..(i+1)*8].try_into().unwrap());
                let is_valid = buf.validity.as_ref().map(|v| v[i] == 1).unwrap_or(true);
                if is_valid { b.append_value(val); } else { b.append_null(); }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Float64 => {
            let mut b = arrow_array::builder::Float64Builder::with_capacity(n);
            for i in 0..n {
                let val = f64::from_le_bytes(buf.data[i*8..(i+1)*8].try_into().unwrap());
                let is_valid = buf.validity.as_ref().map(|v| v[i] == 1).unwrap_or(true);
                if is_valid { b.append_value(val); } else { b.append_null(); }
            }
            Ok(Arc::new(b.finish()))
        }
        other => bail!("gpu_buffers_to_record_batch: unsupported target type {:?}", other),
    }
}
