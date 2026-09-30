use std::ops::Deref;
use std::sync::Arc;

use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_bytes::ByteBuf;

use crate::metadata::RowProvenance;
use crate::object::ObjectId;
use crate::row_histories::BatchId;
use std::collections::HashMap;

/// Shared row payload.
///
/// `Arc<Box<[u8]>>`, NOT `Arc<[u8]>`, and the extra indirection is the entire point.
/// `Arc<[u8]>` stores the strong count, the weak count and the bytes in ONE allocation, so
/// a surviving `Weak` keeps the whole payload resident long after the last strong
/// reference is gone — the value is dropped, the memory is not. `row_bytes_dedup` holds
/// exactly such a `Weak` per (row, batch) and only sweeps them every few thousand
/// operations, which on blob rows meant megabytes per served part sitting in RSS with
/// nothing pointing at them. Measured on a copy of the production store: serving 5.2 MiB
/// of `file_parts` left 5.16 MiB allocated-but-dead; with the sweep forced on every
/// operation the same run left 0.01 MiB.
///
/// Boxing the payload separates the two lifetimes: a stale `Weak` now pins the Arc header
/// (32 bytes — two counters plus the `Box` fat pointer) and the bytes are freed with the
/// last strong reference. Deduplication is unchanged — subscribers still share one payload
/// allocation. Nor is it a cost on the load path: an `Arc<[u8]>` can only be built through
/// `impl From<Box<T>> for Arc<T>`, which allocates a fresh block and memcpys the whole
/// payload into it on every row load; `Arc::new(Box<[u8]>)` allocates the header and moves
/// a pointer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RowBytes(Arc<Box<[u8]>>);

impl RowBytes {
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.as_ref().to_vec()
    }

    /// The shared allocation behind these bytes (for cross-subscription
    /// deduplication — see `query_manager::row_bytes_dedup`).
    pub(crate) fn as_arc(&self) -> &Arc<Box<[u8]>> {
        &self.0
    }

    pub(crate) fn from_arc(arc: Arc<Box<[u8]>>) -> Self {
        Self(arc)
    }
}

impl From<Vec<u8>> for RowBytes {
    fn from(value: Vec<u8>) -> Self {
        Self(Arc::new(value.into_boxed_slice()))
    }
}

impl From<&[u8]> for RowBytes {
    fn from(value: &[u8]) -> Self {
        Self(Arc::new(value.to_vec().into_boxed_slice()))
    }
}

impl Deref for RowBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().as_ref()
    }
}

impl AsRef<[u8]> for RowBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref().as_ref()
    }
}

impl PartialEq<Vec<u8>> for RowBytes {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.0.as_ref().as_ref() == other.as_slice()
    }
}

impl PartialEq<RowBytes> for Vec<u8> {
    fn eq(&self, other: &RowBytes) -> bool {
        self.as_slice() == other.0.as_ref().as_ref()
    }
}

impl Serialize for RowBytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0.as_ref())
    }
}

impl<'de> Deserialize<'de> for RowBytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from(ByteBuf::deserialize(deserializer)?.into_vec()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SharedString(Arc<str>);

impl SharedString {
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl From<String> for SharedString {
    fn from(value: String) -> Self {
        Self(Arc::from(value.into_boxed_str()))
    }
}

impl From<&str> for SharedString {
    fn from(value: &str) -> Self {
        Self(Arc::from(value))
    }
}

impl From<SharedString> for String {
    fn from(value: SharedString) -> Self {
        value.0.as_ref().to_owned()
    }
}

impl From<&SharedString> for String {
    fn from(value: &SharedString) -> Self {
        value.0.as_ref().to_owned()
    }
}

impl Deref for SharedString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl AsRef<str> for SharedString {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl std::borrow::Borrow<str> for SharedString {
    fn borrow(&self) -> &str {
        self.0.as_ref()
    }
}

impl PartialEq<&str> for SharedString {
    fn eq(&self, other: &&str) -> bool {
        self.0.as_ref() == *other
    }
}

impl PartialEq<SharedString> for &str {
    fn eq(&self, other: &SharedString) -> bool {
        *self == other.0.as_ref()
    }
}

impl PartialEq<String> for SharedString {
    fn eq(&self, other: &String) -> bool {
        self.0.as_ref() == other.as_str()
    }
}

impl PartialEq<SharedString> for String {
    fn eq(&self, other: &SharedString) -> bool {
        self.as_str() == other.0.as_ref()
    }
}

impl std::fmt::Display for SharedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_ref())
    }
}

impl Serialize for SharedString {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.as_ref())
    }
}

impl<'de> Deserialize<'de> for SharedString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SharedStringVisitor;

        impl Visitor<'_> for SharedStringVisitor {
            type Value = SharedString;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string")
            }

            fn visit_borrowed_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(SharedString::from(value))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(SharedString::from(value))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(SharedString::from(value))
            }
        }

        deserializer.deserialize_str(SharedStringVisitor)
    }
}

/// A row with its object ID, binary data, and batch identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub id: ObjectId,
    /// Binary encoded row data.
    pub data: RowBytes,
    pub batch_id: BatchId,
    pub provenance: RowProvenance,
}

impl Row {
    pub fn new(
        id: ObjectId,
        data: impl Into<RowBytes>,
        batch_id: BatchId,
        provenance: RowProvenance,
    ) -> Self {
        Self {
            id,
            data: data.into(),
            batch_id,
            provenance,
        }
    }
}

/// Delta for row-level changes (after materialization).
/// Contains full row data for processing by filter/sort/output nodes.
#[derive(Debug, Clone, Default)]
pub struct RowDelta {
    pub added: Vec<Row>,
    pub removed: Vec<Row>,
    /// Rows that stayed in-window but changed position.
    /// Semantics: detach these IDs from current order, then append in listed order.
    pub moved: Vec<ObjectId>,
    /// Updated rows as (old, new) pairs.
    pub updated: Vec<(Row, Row)>,
}

impl RowDelta {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.moved.is_empty()
            && self.updated.is_empty()
    }

    /// The delta of `self` followed by `next`: from the state before `self` to the
    /// state after `next`.
    pub fn compose(self, next: RowDelta) -> RowDelta {
        if self.is_empty() {
            return next;
        }
        if next.is_empty() {
            return self;
        }
        // Per row: its state before `self` and after `next` (`None` = absent), in
        // first-touched order.
        let mut order: Vec<ObjectId> = Vec::new();
        let mut states: HashMap<ObjectId, (Option<Row>, Option<Row>)> = HashMap::new();
        let mut touch =
            |id: ObjectId, before: Option<Row>, after: Option<Row>| match states.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().1 = after;
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    order.push(id);
                    entry.insert((before, after));
                }
            };
        for row in self.added {
            touch(row.id, None, Some(row));
        }
        for row in self.removed {
            touch(row.id, Some(row), None);
        }
        for (old, new) in self.updated {
            touch(new.id, Some(old), Some(new));
        }
        for row in next.added {
            touch(row.id, None, Some(row));
        }
        for row in next.removed {
            touch(row.id, Some(row), None);
        }
        for (old, new) in next.updated {
            touch(new.id, Some(old), Some(new));
        }

        let mut composed = RowDelta::new();
        for id in &order {
            match states.remove(id) {
                Some((None, Some(after))) => composed.added.push(after),
                Some((Some(before), None)) => composed.removed.push(before),
                Some((Some(before), Some(after))) => composed.updated.push((before, after)),
                _ => {}
            }
        }
        let touched: std::collections::HashSet<ObjectId> = order.into_iter().collect();
        let mut moved_seen = std::collections::HashSet::new();
        for id in self.moved.into_iter().chain(next.moved) {
            if !touched.contains(&id) && moved_seen.insert(id) {
                composed.moved.push(id);
            }
        }
        composed
    }
}

#[derive(Debug, Clone)]
pub struct OrderedAdded {
    pub id: ObjectId,
    pub index: usize,
    pub row: Row,
}

#[derive(Debug, Clone)]
pub struct OrderedRemoved {
    pub id: ObjectId,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct OrderedUpdated {
    pub id: ObjectId,
    pub old_index: usize,
    pub new_index: usize,
    pub row: Option<Row>,
}

#[derive(Debug, Clone, Default)]
pub struct OrderedRowDelta {
    pub added: Vec<OrderedAdded>,
    pub removed: Vec<OrderedRemoved>,
    pub updated: Vec<OrderedUpdated>,
    pub pending: bool,
}

impl OrderedRowDelta {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.updated.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct OrderedDeltaResult {
    pub delta: OrderedRowDelta,
    pub ordered_ids_after: Vec<ObjectId>,
}

/// Build an ordered, wire-ready delta using an explicit post-order.
///
/// This variant avoids reconstructing order from delta semantics and should be used
/// when the caller already has the exact post-settle ordered IDs.
pub fn build_ordered_delta_with_post_ids(
    ordered_ids_before: &[ObjectId],
    ordered_ids_after: &[ObjectId],
    delta: &RowDelta,
    pending: bool,
) -> OrderedDeltaResult {
    let pre_index_by_id: HashMap<_, _> = ordered_ids_before
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();
    let post_index_by_id: HashMap<_, _> = ordered_ids_after
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();

    let added = delta
        .added
        .iter()
        .map(|row| OrderedAdded {
            id: row.id,
            index: post_index_by_id.get(&row.id).copied().unwrap_or(0),
            row: row.clone(),
        })
        .collect();

    let removed = delta
        .removed
        .iter()
        .map(|row| OrderedRemoved {
            id: row.id,
            index: pre_index_by_id.get(&row.id).copied().unwrap_or(0),
        })
        .collect();

    let mut updated = delta
        .moved
        .iter()
        .map(|id| OrderedUpdated {
            id: *id,
            old_index: pre_index_by_id.get(id).copied().unwrap_or(0),
            new_index: post_index_by_id.get(id).copied().unwrap_or(0),
            row: None,
        })
        .collect::<Vec<_>>();

    for (old, new) in &delta.updated {
        let old_index = pre_index_by_id.get(&old.id).copied().unwrap_or(0);
        let new_index = post_index_by_id.get(&new.id).copied().unwrap_or(0);
        let row_changed = old.data != new.data || old.batch_id != new.batch_id;

        if row_changed {
            updated.push(OrderedUpdated {
                id: new.id,
                old_index,
                new_index,
                row: Some(new.clone()),
            });
        } else if old_index != new_index {
            updated.push(OrderedUpdated {
                id: new.id,
                old_index,
                new_index,
                row: None,
            });
        }
    }

    OrderedDeltaResult {
        delta: OrderedRowDelta {
            added,
            removed,
            updated,
            pending,
        },
        ordered_ids_after: ordered_ids_after.to_vec(),
    }
}
