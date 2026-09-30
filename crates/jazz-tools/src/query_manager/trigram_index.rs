//! Trigram indexes: case-insensitive substring search inside one scope value.
//!
//! A search `where scope = x and text contains "needle"` otherwise loads every row
//! of the scope to test its text. Here every row files one entry per trigram of its
//! case-folded text, under its scope value, so the rows that can hold the needle are
//! the intersection of the needle's trigram posting lists, read as keys without
//! loading a row. Only those candidates are loaded and tested.
//!
//! An app declares them next to its tables (`index_declarations`); a store maintains
//! the ones in its record (`declared_index`).
//!
//! An entry's value is `Value::Bytea(encode(scope) ++ utf8(trigram))`, so it rides
//! the ordinary index key path and its value segment is
//! `"09" ++ hex(encode(scope)) ++ hex(utf8(trigram))`. The scope must have a
//! fixed-width encoding; the trigram is the last component, so its width does not
//! matter for an exact lookup.
//!
//! Folding (`fold`) is applied to the text and to the needle alike, which is also
//! what `contains` on text compares (`graph_nodes/filter.rs`): a folded needle is a
//! substring of the folded text only if all its trigrams are.

use std::collections::BTreeSet;

use crate::query_manager::composite_index::{fixed_width_encoding, hex};
use crate::query_manager::types::Value;

/// The case folding both sides of a text `contains` go through.
///
/// Char by char, so folding commutes with concatenation: a needle that is a substring
/// of the text folds to a substring of the folded text. `str::to_lowercase` does not —
/// it lowers a capital sigma by its context, to `ς` at the end of a word and to `σ`
/// elsewhere, so "Σ" missed the "Σ" that ends "ΟΣ". The final `ς` folds to `σ`, so a
/// typed final sigma meets a capital one.
pub(crate) fn fold(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .map(|folded| if folded == 'ς' { 'σ' } else { folded })
        .collect()
}

/// The distinct trigrams (three consecutive chars) of already-folded text.
pub(crate) fn trigrams(folded: &str) -> BTreeSet<String> {
    let chars: Vec<char> = folded.chars().collect();
    chars
        .windows(3)
        .map(|window| window.iter().collect::<String>())
        .collect()
}

/// The entry value for one trigram of a row in `scope`, or `None` when the scope
/// is null or not fixed-width.
pub(crate) fn entry_value(scope: &Value, trigram: &str) -> Option<Value> {
    let mut bytes = fixed_width_encoding(scope)?;
    bytes.extend_from_slice(trigram.as_bytes());
    Some(Value::Bytea(bytes))
}

/// The index value segment of `entry_value(scope, trigram)`.
pub(crate) fn entry_segment(scope: &Value, trigram: &str) -> Option<String> {
    Some(hex(&crate::storage::encode_value(&entry_value(
        scope, trigram,
    )?)))
}
