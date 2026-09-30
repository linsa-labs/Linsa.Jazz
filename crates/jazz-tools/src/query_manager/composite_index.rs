//! Composite two-column indexes.
//!
//! An entry orders rows by the first column's value, then by the second's.
//! One prefix range of the index therefore holds one first-column value,
//! walked in second-column order. A page of `where first = x order by second`
//! is the head of that range and never needs the rest of it.
//!
//! An app declares them next to its tables (`index_declarations`); a store
//! maintains the ones in its record (`declared_index`). Entries are written by
//! the index mutation builders in `indices.rs`, alongside the single-column
//! ones. They are read by the ordered-window mode of `IndexScanNode`.
//!
//! An entry's value is `Value::Bytea(encode(first) ++ encode(second))`. It
//! therefore rides the ordinary index key path (`index_entry_key`), and its
//! value segment is `"09" ++ hex(encode(first)) ++ hex(encode(second))`.
//!
//! Both components must have fixed-width encodings. That is what keeps the
//! concatenation in the order of the pair, and what lets a reader cut the
//! second component out of a key by position.

use crate::query_manager::types::Value;

/// The order-preserving encoding of `value` when its width does not depend on
/// the value. Doubles are left out: `0.0` and `-0.0` compare equal but encode
/// differently, which a byte-ordered pair cannot express.
pub(crate) fn fixed_width_encoding(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Integer(_)
        | Value::BigInt(_)
        | Value::Timestamp(_)
        | Value::Uuid(_)
        | Value::Boolean(_) => Some(crate::storage::encode_value(value)),
        _ => None,
    }
}

/// The entry value for a row whose composite columns hold `first` and
/// `second`, or `None` when either is null or not fixed-width. A row with no
/// entry is simply absent from the index, as nulls are from single-column ones.
pub(crate) fn composite_value(first: &Value, second: &Value) -> Option<Value> {
    let mut bytes = fixed_width_encoding(first)?;
    bytes.extend(fixed_width_encoding(second)?);
    Some(Value::Bytea(bytes))
}

/// Lowercase hex, the alphabet of index value segments.
pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

/// The row id at the end of an index entry key (`{segment}:{uuid_hex}`).
pub(crate) fn entry_row_id(key: &str) -> Option<crate::object::ObjectId> {
    let (_, uuid_hex) = key.rsplit_once(':')?;
    if uuid_hex.len() != 32 {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (index, chunk) in uuid_hex.as_bytes().chunks_exact(2).enumerate() {
        let high = (chunk[0] as char).to_digit(16)?;
        let low = (chunk[1] as char).to_digit(16)?;
        bytes[index] = ((high << 4) | low) as u8;
    }
    Some(crate::object::ObjectId::from_uuid(uuid::Uuid::from_bytes(
        bytes,
    )))
}
