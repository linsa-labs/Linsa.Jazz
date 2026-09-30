//! The set of declared indexes (`composite_index`, `trigram_index`) an app asks for,
//! and a store maintains.
//!
//! An app declares them next to its tables; they travel in the permissions bundle, not
//! in the schema, because they are per table NAME across every schema generation — as
//! their entries and completeness marks are — and must not change the schema hash.
//!
//! The value is canonical: tables and indexes sorted, duplicates dropped, so two
//! declarations that differ only in order encode to the same bytes and compare equal.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::query_manager::types::{ColumnType, Schema, TableName};

/// A composite `(first, second)` index: rows ordered by `second` within one `first`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompositeIndex {
    /// The raw index column name, `"{first}+{second}"`.
    pub name: String,
    pub first: String,
    pub second: String,
}

/// A trigram index over `text`, scoped by `scope`: case-insensitive substring search
/// inside one scope value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrigramIndex {
    /// The raw index column name, `"{scope}>{text}"`.
    pub name: String,
    pub scope: String,
    pub text: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct TableIndexes {
    composites: Vec<CompositeIndex>,
    trigrams: Vec<TrigramIndex>,
}

/// The declared indexes of an app, by table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexDeclarations {
    tables: BTreeMap<String, TableIndexes>,
}

/// Why a declaration was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclarationError(pub String);

impl std::fmt::Display for DeclarationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeclarationError {}

/// The declarations of one table on the wire: `[first, second]` and `[scope, text]`
/// pairs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableIndexDeclarationsWire {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub composite: Vec<[String; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trigram: Vec<[String; 2]>,
}

/// The declarations on the wire (publish request, head response, TS apps).
pub type IndexDeclarationsWire = BTreeMap<String, TableIndexDeclarationsWire>;

/// Names joined into raw index table names (`idx:{table}:{index}:{branch}`) and into
/// index names must not hold the separators, or two declarations — or a declaration
/// and a column's own index — would share a raw table.
fn check_name(kind: &str, name: &str) -> Result<(), DeclarationError> {
    if name.is_empty() || name.contains([':', '+', '>']) {
        return Err(DeclarationError(format!(
            "{kind} name {name:?} cannot carry a declared index: it is empty or holds ':', '+' or '>'"
        )));
    }
    Ok(())
}

impl IndexDeclarations {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// Declare a composite index; names are checked, duplicates collapse.
    pub fn with_composite(
        mut self,
        table: &str,
        first: &str,
        second: &str,
    ) -> Result<Self, DeclarationError> {
        check_name("table", table)?;
        check_name("column", first)?;
        check_name("column", second)?;
        let indexes = self.tables.entry(table.to_string()).or_default();
        let index = CompositeIndex {
            name: format!("{first}+{second}"),
            first: first.to_string(),
            second: second.to_string(),
        };
        if !indexes.composites.contains(&index) {
            indexes.composites.push(index);
            indexes.composites.sort();
        }
        Ok(self)
    }

    /// Declare a trigram index; names are checked, duplicates collapse.
    pub fn with_trigram(
        mut self,
        table: &str,
        scope: &str,
        text: &str,
    ) -> Result<Self, DeclarationError> {
        check_name("table", table)?;
        check_name("column", scope)?;
        check_name("column", text)?;
        let indexes = self.tables.entry(table.to_string()).or_default();
        let index = TrigramIndex {
            name: format!("{scope}>{text}"),
            scope: scope.to_string(),
            text: text.to_string(),
        };
        if !indexes.trigrams.contains(&index) {
            indexes.trigrams.push(index);
            indexes.trigrams.sort();
        }
        Ok(self)
    }

    /// These declarations without the `(table, index name)`s in `removed`.
    pub fn without(&self, removed: &BTreeSet<(String, String)>) -> Self {
        if removed.is_empty() {
            return self.clone();
        }
        let mut tables = BTreeMap::new();
        for (table, indexes) in &self.tables {
            let gone = |name: &str| removed.contains(&(table.clone(), name.to_string()));
            let kept = TableIndexes {
                composites: indexes
                    .composites
                    .iter()
                    .filter(|index| !gone(&index.name))
                    .cloned()
                    .collect(),
                trigrams: indexes
                    .trigrams
                    .iter()
                    .filter(|index| !gone(&index.name))
                    .cloned()
                    .collect(),
            };
            if !kept.composites.is_empty() || !kept.trigrams.is_empty() {
                tables.insert(table.clone(), kept);
            }
        }
        Self { tables }
    }

    pub fn composites(&self, table: &str) -> &[CompositeIndex] {
        self.tables
            .get(table)
            .map(|indexes| indexes.composites.as_slice())
            .unwrap_or(&[])
    }

    pub fn trigrams(&self, table: &str) -> &[TrigramIndex] {
        self.tables
            .get(table)
            .map(|indexes| indexes.trigrams.as_slice())
            .unwrap_or(&[])
    }

    /// The tables that carry a declared index.
    pub fn tables(&self) -> impl Iterator<Item = &str> {
        self.tables.keys().map(String::as_str)
    }

    /// Every declared `(table, index name)`.
    pub fn index_names(&self) -> BTreeSet<(String, String)> {
        self.tables
            .iter()
            .flat_map(|(table, indexes)| {
                indexes
                    .composites
                    .iter()
                    .map(|index| index.name.clone())
                    .chain(indexes.trigrams.iter().map(|index| index.name.clone()))
                    .map(move |name| (table.clone(), name))
            })
            .collect()
    }

    pub fn from_wire(wire: &IndexDeclarationsWire) -> Result<Self, DeclarationError> {
        let mut declarations = Self::empty();
        for (table, indexes) in wire {
            for [first, second] in &indexes.composite {
                declarations = declarations.with_composite(table, first, second)?;
            }
            for [scope, text] in &indexes.trigram {
                declarations = declarations.with_trigram(table, scope, text)?;
            }
        }
        Ok(declarations)
    }

    pub fn to_wire(&self) -> IndexDeclarationsWire {
        self.tables
            .iter()
            .map(|(table, indexes)| {
                (
                    table.clone(),
                    TableIndexDeclarationsWire {
                        composite: indexes
                            .composites
                            .iter()
                            .map(|index| [index.first.clone(), index.second.clone()])
                            .collect(),
                        trigram: indexes
                            .trigrams
                            .iter()
                            .map(|index| [index.scope.clone(), index.text.clone()])
                            .collect(),
                    },
                )
            })
            .collect()
    }

    /// Check the declarations against the schema they are published with.
    ///
    /// - every column exists;
    /// - a composite's columns and a trigram's scope have a fixed-width encoding
    ///   (`composite_index::fixed_width_encoding`), and a trigram's text is text;
    /// - a composite's first column and a trigram's scope carry their own single-column
    ///   index: until an index is filled, a scan reads that column's index instead, and
    ///   one the schema leaves out (`indexOnly`) would return nothing;
    /// - no declared index name equals a column name, whose own index would share the
    ///   raw table, nor prefixes one as `{name}:…`, whose raw tables would sit under the
    ///   declared index's prefix and be cleared with it.
    pub fn validate_against(&self, schema: &Schema) -> Result<(), DeclarationError> {
        for (table, indexes) in &self.tables {
            let Some(table_schema) = schema.get(&TableName::new(table.as_str())) else {
                return Err(DeclarationError(format!(
                    "declared index on unknown table {table:?}"
                )));
            };
            // Raw index tables are `idx:{table}:{column}:{branch}`: a table named
            // `{table}:…` would share the declared index's prefix, and clearing or filling
            // it would reach into that table's indexes.
            if let Some(other) = schema
                .keys()
                .find(|other| other.as_str().starts_with(&format!("{table}:")))
            {
                return Err(DeclarationError(format!(
                    "declared index on {table:?}: table {:?} shares its raw index prefix",
                    other.as_str()
                )));
            }
            let column_type = |column: &str| -> Result<&ColumnType, DeclarationError> {
                table_schema
                    .columns
                    .column(column)
                    .map(|descriptor| &descriptor.column_type)
                    .ok_or_else(|| {
                        DeclarationError(format!(
                            "declared index on {table:?} names unknown column {column:?}"
                        ))
                    })
            };
            let fixed_width = |column: &str| -> Result<(), DeclarationError> {
                match column_type(column)? {
                    ColumnType::Integer
                    | ColumnType::BigInt
                    | ColumnType::Timestamp
                    | ColumnType::Uuid
                    | ColumnType::Boolean => Ok(()),
                    other => Err(DeclarationError(format!(
                        "declared index on {table:?}: column {column:?} is {other:?}, which has no fixed-width encoding"
                    ))),
                }
            };
            let indexed = |column: &str| -> Result<(), DeclarationError> {
                if table_schema.is_indexed_column(column) {
                    Ok(())
                } else {
                    Err(DeclarationError(format!(
                        "declared index on {table:?}: column {column:?} carries no index of its own"
                    )))
                }
            };
            let not_a_column = |name: &str| -> Result<(), DeclarationError> {
                if table_schema.columns.column(name).is_some() {
                    return Err(DeclarationError(format!(
                        "declared index {name:?} on {table:?} is also a column name"
                    )));
                }
                let prefix = format!("{name}:");
                if let Some(column) = table_schema
                    .columns
                    .columns
                    .iter()
                    .find(|column| column.name.as_str().starts_with(&prefix))
                {
                    return Err(DeclarationError(format!(
                        "declared index {name:?} on {table:?}: column {:?} shares its raw index prefix",
                        column.name.as_str()
                    )));
                }
                Ok(())
            };
            for index in &indexes.composites {
                fixed_width(&index.first)?;
                fixed_width(&index.second)?;
                indexed(&index.first)?;
                not_a_column(&index.name)?;
            }
            for index in &indexes.trigrams {
                fixed_width(&index.scope)?;
                indexed(&index.scope)?;
                if !matches!(column_type(&index.text)?, ColumnType::Text) {
                    return Err(DeclarationError(format!(
                        "trigram index on {table:?}: column {:?} is not text",
                        index.text
                    )));
                }
                not_a_column(&index.name)?;
            }
        }
        Ok(())
    }

    const ENCODING_VERSION: u8 = 1;

    /// The canonical bytes: version, then tables in order, each with its composites and
    /// trigrams in order, every name length-prefixed.
    pub fn encode(&self) -> Vec<u8> {
        fn put_str(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut bytes = vec![Self::ENCODING_VERSION];
        bytes.extend_from_slice(&(self.tables.len() as u32).to_le_bytes());
        for (table, indexes) in &self.tables {
            put_str(&mut bytes, table);
            bytes.extend_from_slice(&(indexes.composites.len() as u32).to_le_bytes());
            for index in &indexes.composites {
                put_str(&mut bytes, &index.first);
                put_str(&mut bytes, &index.second);
            }
            bytes.extend_from_slice(&(indexes.trigrams.len() as u32).to_le_bytes());
            for index in &indexes.trigrams {
                put_str(&mut bytes, &index.scope);
                put_str(&mut bytes, &index.text);
            }
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DeclarationError> {
        struct Reader<'a> {
            bytes: &'a [u8],
            at: usize,
        }
        impl Reader<'_> {
            fn take(&mut self, len: usize) -> Result<&[u8], DeclarationError> {
                let end = self
                    .at
                    .checked_add(len)
                    .filter(|end| *end <= self.bytes.len())
                    .ok_or_else(|| DeclarationError("truncated index declarations".into()))?;
                let slice = &self.bytes[self.at..end];
                self.at = end;
                Ok(slice)
            }
            fn count(&mut self) -> Result<usize, DeclarationError> {
                let raw: [u8; 4] = self.take(4)?.try_into().expect("four bytes");
                Ok(u32::from_le_bytes(raw) as usize)
            }
            fn string(&mut self) -> Result<String, DeclarationError> {
                let len = self.count()?;
                String::from_utf8(self.take(len)?.to_vec())
                    .map_err(|_| DeclarationError("index declaration name is not utf-8".into()))
            }
        }
        let mut reader = Reader { bytes, at: 0 };
        let version = reader.take(1)?[0];
        if version != Self::ENCODING_VERSION {
            return Err(DeclarationError(format!(
                "unsupported index declarations version {version}"
            )));
        }
        let mut declarations = Self::empty();
        for _ in 0..reader.count()? {
            let table = reader.string()?;
            for _ in 0..reader.count()? {
                let (first, second) = (reader.string()?, reader.string()?);
                declarations = declarations.with_composite(&table, &first, &second)?;
            }
            for _ in 0..reader.count()? {
                let (scope, text) = (reader.string()?, reader.string()?);
                declarations = declarations.with_trigram(&table, &scope, &text)?;
            }
        }
        if reader.at != bytes.len() {
            return Err(DeclarationError(
                "trailing bytes after index declarations".into(),
            ));
        }
        Ok(declarations)
    }
}

/// Where a declared index stands in the store that maintains it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexPhase {
    /// Deleting whatever an earlier incarnation, or an engine that did not maintain it,
    /// left under its prefix; `after` is the last index entry deleted. Writes file no
    /// entries meanwhile, so nothing new lands under the prefix and the clear ends.
    Clearing { after: Option<String> },
    /// Filing the entries of the rows that existed when the clear ended, walking the
    /// `_id` index up to `until`, its last entry then; `after` is the last `_id` entry
    /// filed. Writes file their rows' entries from the start of this phase, so a row
    /// past `until` — ids grow with time — is already filed, and the walk ends.
    Filling {
        after: Option<String>,
        until: String,
    },
    /// Every live row has its entries: scans may read it.
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexState {
    /// Bumped on every add, so work planned for an earlier incarnation of the same
    /// index — by this runtime or another one over the store — is recognised and
    /// dropped.
    pub incarnation: u64,
    pub phase: IndexPhase,
}

/// The declared indexes a store maintains, persisted in the store itself.
///
/// Writers maintain the entries of every index in `declarations` once its clear is done
/// (`written`); the phases track the work that makes the index whole. The app's
/// declarations (the permissions head) only PROPOSE a set; `propose` turns a proposal
/// into the next record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaintainedIndexes {
    pub declarations: IndexDeclarations,
    /// One state per declared `(table, index name)`.
    pub states: BTreeMap<(String, String), IndexState>,
    /// Indexes no longer maintained whose entries are still being deleted, with the
    /// last entry deleted. Nothing reads them; a later add clears its own prefix first.
    pub retired: BTreeMap<(String, String), Option<String>>,
    pub next_incarnation: u64,
}

impl MaintainedIndexes {
    const ENCODING_VERSION: u8 = 1;

    /// The state of `index` on `table`, if it is maintained.
    pub fn state(&self, table: &str, index: &str) -> Option<&IndexState> {
        self.states.get(&(table.to_string(), index.to_string()))
    }

    /// The incarnation `index` on `table` is complete in, if it is.
    pub fn complete_incarnation(&self, table: &str, index: &str) -> Option<u64> {
        self.state(table, index)
            .filter(|state| state.phase == IndexPhase::Complete)
            .map(|state| state.incarnation)
    }

    /// The declarations writes file entries for: every declared index but those still
    /// clearing their prefix.
    pub fn written(&self) -> IndexDeclarations {
        let clearing: BTreeSet<(String, String)> = self
            .states
            .iter()
            .filter(|(_, state)| matches!(state.phase, IndexPhase::Clearing { .. }))
            .map(|(key, _)| key.clone())
            .collect();
        self.declarations.without(&clearing)
    }

    /// Whether any index still has clearing or filling to do.
    pub fn has_work(&self) -> bool {
        !self.retired.is_empty()
            || self
                .states
                .values()
                .any(|state| state.phase != IndexPhase::Complete)
    }

    /// The record after `next` replaces the maintained declarations, or `None` when
    /// they are equal. Removed indexes are retired; added ones — re-added ones included —
    /// start a new incarnation in `Clearing`, and leave `retired`, since their own clear
    /// covers what is left under their prefix.
    pub fn propose(&self, next: &IndexDeclarations) -> Option<Self> {
        if &self.declarations == next {
            return None;
        }
        let mut record = self.clone();
        let wanted = next.index_names();
        for key in self.declarations.index_names() {
            if !wanted.contains(&key) {
                record.states.remove(&key);
                record.retired.insert(key, None);
            }
        }
        for key in wanted {
            if record.states.contains_key(&key) {
                continue;
            }
            record.retired.remove(&key);
            record.states.insert(
                key,
                IndexState {
                    incarnation: record.next_incarnation,
                    phase: IndexPhase::Clearing { after: None },
                },
            );
            record.next_incarnation += 1;
        }
        record.declarations = next.clone();
        Some(record)
    }

    pub fn encode(&self) -> Vec<u8> {
        fn put_str(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn put_cursor(bytes: &mut Vec<u8>, cursor: &Option<String>) {
            match cursor {
                None => bytes.push(0),
                Some(after) => {
                    bytes.push(1);
                    put_str(bytes, after);
                }
            }
        }
        let mut bytes = vec![Self::ENCODING_VERSION];
        bytes.extend_from_slice(&self.next_incarnation.to_le_bytes());
        let declarations = self.declarations.encode();
        bytes.extend_from_slice(&(declarations.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&declarations);
        bytes.extend_from_slice(&(self.states.len() as u32).to_le_bytes());
        for ((table, index), state) in &self.states {
            put_str(&mut bytes, table);
            put_str(&mut bytes, index);
            bytes.extend_from_slice(&state.incarnation.to_le_bytes());
            match &state.phase {
                IndexPhase::Clearing { after } => {
                    bytes.push(0);
                    put_cursor(&mut bytes, after);
                }
                IndexPhase::Filling { after, until } => {
                    bytes.push(1);
                    put_cursor(&mut bytes, after);
                    put_str(&mut bytes, until);
                }
                IndexPhase::Complete => bytes.push(2),
            }
        }
        bytes.extend_from_slice(&(self.retired.len() as u32).to_le_bytes());
        for ((table, index), after) in &self.retired {
            put_str(&mut bytes, table);
            put_str(&mut bytes, index);
            put_cursor(&mut bytes, after);
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DeclarationError> {
        struct Reader<'a> {
            bytes: &'a [u8],
            at: usize,
        }
        impl Reader<'_> {
            fn take(&mut self, len: usize) -> Result<&[u8], DeclarationError> {
                let end = self
                    .at
                    .checked_add(len)
                    .filter(|end| *end <= self.bytes.len())
                    .ok_or_else(|| DeclarationError("truncated maintained-index record".into()))?;
                let slice = &self.bytes[self.at..end];
                self.at = end;
                Ok(slice)
            }
            fn u8(&mut self) -> Result<u8, DeclarationError> {
                Ok(self.take(1)?[0])
            }
            fn u32(&mut self) -> Result<usize, DeclarationError> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")) as usize)
            }
            fn u64(&mut self) -> Result<u64, DeclarationError> {
                Ok(u64::from_le_bytes(
                    self.take(8)?.try_into().expect("eight bytes"),
                ))
            }
            fn string(&mut self) -> Result<String, DeclarationError> {
                let len = self.u32()?;
                String::from_utf8(self.take(len)?.to_vec())
                    .map_err(|_| DeclarationError("maintained-index name is not utf-8".into()))
            }
            fn cursor(&mut self) -> Result<Option<String>, DeclarationError> {
                match self.u8()? {
                    0 => Ok(None),
                    1 => Ok(Some(self.string()?)),
                    _ => Err(DeclarationError(
                        "unknown cursor marker in maintained-index record".into(),
                    )),
                }
            }
        }
        let mut reader = Reader { bytes, at: 0 };
        if reader.u8()? != Self::ENCODING_VERSION {
            return Err(DeclarationError(
                "unsupported maintained-index record version".into(),
            ));
        }
        let next_incarnation = reader.u64()?;
        let declarations_len = reader.u32()?;
        let declarations = IndexDeclarations::decode(reader.take(declarations_len)?)?;
        let mut states = BTreeMap::new();
        for _ in 0..reader.u32()? {
            let key = (reader.string()?, reader.string()?);
            let incarnation = reader.u64()?;
            let phase = match reader.u8()? {
                0 => IndexPhase::Clearing {
                    after: reader.cursor()?,
                },
                1 => IndexPhase::Filling {
                    after: reader.cursor()?,
                    until: reader.string()?,
                },
                2 => IndexPhase::Complete,
                // Read as complete, a phase this engine does not know would be trusted.
                _ => {
                    return Err(DeclarationError(
                        "unknown phase in maintained-index record".into(),
                    ));
                }
            };
            states.insert(key, IndexState { incarnation, phase });
        }
        let mut retired = BTreeMap::new();
        for _ in 0..reader.u32()? {
            let key = (reader.string()?, reader.string()?);
            retired.insert(key, reader.cursor()?);
        }
        if reader.at != bytes.len() {
            return Err(DeclarationError(
                "trailing bytes after maintained-index record".into(),
            ));
        }
        Ok(Self {
            declarations,
            states,
            retired,
            next_incarnation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::types::{ColumnDescriptor, RowDescriptor, TableSchema};

    fn messages_schema() -> Schema {
        let mut schema = Schema::new();
        schema.insert(
            TableName::new("messages"),
            TableSchema::new(RowDescriptor::new(vec![
                ColumnDescriptor::new("chatId", ColumnType::Uuid),
                ColumnDescriptor::new("createdAtMs", ColumnType::BigInt),
                ColumnDescriptor::new("text", ColumnType::Text),
                ColumnDescriptor::new("score", ColumnType::Double),
            ])),
        );
        schema
    }

    #[test]
    fn declarations_are_canonical_and_round_trip() {
        let one = IndexDeclarations::empty()
            .with_trigram("messages", "chatId", "text")
            .unwrap()
            .with_composite("messages", "chatId", "createdAtMs")
            .unwrap()
            .with_composite("messages", "chatId", "createdAtMs")
            .unwrap();
        let other = IndexDeclarations::from_wire(&one.to_wire()).unwrap();
        assert_eq!(one, other);
        assert_eq!(one.encode(), other.encode());
        assert_eq!(IndexDeclarations::decode(&one.encode()).unwrap(), one);
        assert_eq!(one.composites("messages").len(), 1);
        assert_eq!(
            one.index_names(),
            BTreeSet::from([
                ("messages".to_string(), "chatId+createdAtMs".to_string()),
                ("messages".to_string(), "chatId>text".to_string()),
            ])
        );
        assert!(IndexDeclarations::decode(&[one.encode(), vec![0]].concat()).is_err());
    }

    #[test]
    fn separators_in_names_are_refused() {
        for (table, first) in [
            ("m:x", "chatId"),
            ("messages", "chat+id"),
            ("messages", "a>b"),
        ] {
            assert!(
                IndexDeclarations::empty()
                    .with_composite(table, first, "createdAtMs")
                    .is_err(),
                "{table}.{first}"
            );
        }
    }

    #[test]
    fn declarations_are_checked_against_the_schema() {
        let schema = messages_schema();
        let valid = IndexDeclarations::empty()
            .with_composite("messages", "chatId", "createdAtMs")
            .unwrap()
            .with_trigram("messages", "chatId", "text")
            .unwrap();
        valid.validate_against(&schema).unwrap();

        let refused = [
            IndexDeclarations::empty().with_composite("nope", "chatId", "createdAtMs"),
            IndexDeclarations::empty().with_composite("messages", "chatId", "missing"),
            IndexDeclarations::empty().with_composite("messages", "chatId", "score"),
            IndexDeclarations::empty().with_composite("messages", "text", "createdAtMs"),
            IndexDeclarations::empty().with_trigram("messages", "chatId", "createdAtMs"),
        ];
        for declarations in refused {
            assert!(declarations.unwrap().validate_against(&schema).is_err());
        }

        let mut unindexed = schema.clone();
        unindexed
            .get_mut(&TableName::new("messages"))
            .unwrap()
            .indexed_columns = Some(vec!["createdAtMs".into()]);
        assert!(valid.validate_against(&unindexed).is_err());

        // `idx:messages:chatId+createdAtMs:x:{branch}` sits under the composite's prefix.
        let mut shadowed = messages_schema();
        let messages = shadowed.get_mut(&TableName::new("messages")).unwrap();
        let mut columns = messages.columns.columns.to_vec();
        columns.push(ColumnDescriptor::new(
            "chatId+createdAtMs:x",
            ColumnType::Integer,
        ));
        *messages = TableSchema::new(RowDescriptor::new(columns));
        assert!(valid.validate_against(&shadowed).is_err());
    }

    #[test]
    fn a_proposal_retires_removed_indexes_and_restarts_re_added_ones() {
        let window = IndexDeclarations::empty()
            .with_composite("messages", "chatId", "createdAtMs")
            .unwrap();
        let both = window
            .clone()
            .with_trigram("messages", "chatId", "text")
            .unwrap();
        let key = |name: &str| ("messages".to_string(), name.to_string());

        let first = MaintainedIndexes::default().propose(&both).unwrap();
        assert_eq!(first.states.len(), 2);
        assert!(
            first
                .states
                .values()
                .all(|state| state.phase == IndexPhase::Clearing { after: None })
        );
        assert!(
            first.propose(&both).is_none(),
            "an equal proposal changes nothing"
        );

        let mut done = first.clone();
        for state in done.states.values_mut() {
            state.phase = IndexPhase::Complete;
        }
        let dropped = done.propose(&window).unwrap();
        assert_eq!(dropped.retired.get(&key("chatId>text")), Some(&None));
        assert_eq!(
            dropped.state("messages", "chatId+createdAtMs"),
            done.state("messages", "chatId+createdAtMs"),
            "a kept index keeps its phase and incarnation"
        );

        let readded = dropped.propose(&both).unwrap();
        let state = readded.state("messages", "chatId>text").unwrap();
        assert_eq!(state.phase, IndexPhase::Clearing { after: None });
        assert!(
            state.incarnation > first.state("messages", "chatId>text").unwrap().incarnation,
            "a re-add is a new incarnation"
        );
        assert!(readded.retired.is_empty(), "a re-add clears its own prefix");

        let mut cursors = readded.clone();
        cursors
            .states
            .get_mut(&key("chatId+createdAtMs"))
            .unwrap()
            .phase = IndexPhase::Filling {
            after: Some("main:00:ab".into()),
            until: "main:00:ff".into(),
        };
        cursors
            .retired
            .insert(key("gone+x"), Some("main:ff:cd".into()));
        assert_eq!(
            MaintainedIndexes::decode(&cursors.encode()).unwrap(),
            cursors
        );
        assert!(MaintainedIndexes::decode(&[cursors.encode(), vec![0]].concat()).is_err());
    }

    /// A tag the record's encoding does not define fails the decode. Read as the last
    /// phase, an unknown one would be trusted as complete; read as a cursor, an unknown
    /// marker would resume a walk from a key nobody wrote.
    #[test]
    fn a_record_with_an_unknown_tag_does_not_decode() {
        let one = |phase: IndexPhase| MaintainedIndexes {
            states: BTreeMap::from([(
                ("messages".to_string(), "chatId+createdAtMs".to_string()),
                IndexState {
                    incarnation: 1,
                    phase,
                },
            )]),
            ..MaintainedIndexes::default()
        };

        // The phase is the state's last byte, before the retired count (four bytes).
        let complete = one(IndexPhase::Complete).encode();
        let phase_at = complete.len() - 5;
        assert_eq!(complete[phase_at], 2, "the phase byte of a complete index");
        let mut unknown_phase = complete.clone();
        unknown_phase[phase_at] = 3;
        assert!(
            MaintainedIndexes::decode(&unknown_phase).is_err(),
            "an unknown phase decoded as {:?}",
            MaintainedIndexes::decode(&unknown_phase)
        );

        // A clearing phase ends with its cursor's marker: none (0) or one (1) key.
        let clearing = one(IndexPhase::Clearing { after: None }).encode();
        let marker_at = clearing.len() - 5;
        assert_eq!(clearing[marker_at], 0, "the cursor marker of a fresh clear");
        let mut unknown_marker = clearing[..marker_at].to_vec();
        unknown_marker.push(2);
        unknown_marker.extend_from_slice(&0u32.to_le_bytes());
        unknown_marker.extend_from_slice(&clearing[marker_at + 1..]);
        assert!(
            MaintainedIndexes::decode(&unknown_marker).is_err(),
            "an unknown cursor marker decoded as {:?}",
            MaintainedIndexes::decode(&unknown_marker)
        );
    }
}
