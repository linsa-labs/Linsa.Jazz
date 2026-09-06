//! Synchronous Storage trait and implementations.
//!
//! This is the foundation of the sync storage architecture. All storage
//! and index operations are synchronous - they return immediately with results.
//!
//! # Design: Single-threaded
//!
//! No `Send + Sync` bounds on Storage. Each thread (main, worker) has its own
//! Storage instance. Cross-thread communication uses the sync protocol over
//! postMessage, not shared mutable state.

#[cfg(test)]
pub mod conformance;
#[cfg(test)]
pub mod conformance_differential;
mod key_codec;
mod memory;
mod opfs_btree;
mod storage_core;
mod storage_trait;
pub use memory::MemoryStorage;
pub use opfs_btree::OpfsBTreeStorage;
pub use storage_trait::{PassOutcome, Storage};
#[cfg(all(feature = "rocksdb", not(target_arch = "wasm32")))]
mod rocksdb;
#[cfg(all(feature = "rocksdb", not(target_arch = "wasm32")))]
pub use rocksdb::RocksDBStorage;
#[cfg(all(feature = "sqlite", not(target_arch = "wasm32")))]
mod sqlite;
#[cfg(all(feature = "sqlite", not(target_arch = "wasm32")))]
pub use sqlite::SqliteStorage;
pub mod graft;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use smolset::SmolSet;

use crate::batch_fate::{
    BatchFate, CapturedFrontierMember, LocalBatchMember, LocalBatchRecord, SealedBatchSubmission,
};
use crate::catalogue::CatalogueEntry;
use crate::digest::Digest32;
use crate::metadata::MetadataKey;
use crate::object::{BranchName, ObjectId};
use crate::query_manager::types::{
    ColumnDescriptor, ColumnName, ColumnType, ComposedBranchName, RowDescriptor, SchemaHash,
    SharedString, Value,
};
use crate::row_format::{decode_row, encode_row};
use crate::row_histories::{
    BatchId, FlatRowCodecs, HistoryScan, QueryRowBatch, RowState, StoredRowBatch, VisibleRowEntry,
    decode_flat_history_row_with_codecs, decode_flat_visible_row_entry_with_codecs,
    flat_row_codecs,
};
use crate::sync_manager::DurabilityTier;

// ============================================================================
// Storage Types
// ============================================================================

type EncodedTableRowHistories = BTreeMap<ObjectId, BTreeMap<(SharedString, BatchId), Vec<u8>>>;

pub(super) fn batch_fate_confirmed_tier_for_row(
    settlement: &BatchFate,
    _row: &StoredRowBatch,
) -> Option<DurabilityTier> {
    settlement.confirmed_tier()
}

pub(super) fn row_confirmed_tier_with_batch_fate<H: Storage + ?Sized>(
    storage: &H,
    row: &StoredRowBatch,
) -> Result<Option<DurabilityTier>, StorageError> {
    Ok(match storage.load_authoritative_batch_fate(row.batch_id)? {
        Some(settlement) => batch_fate_confirmed_tier_for_row(&settlement, row),
        None => row.confirmed_tier,
    })
}

pub(super) fn apply_batch_fate_tiers_to_rows<H: Storage + ?Sized>(
    storage: &H,
    rows: &mut [StoredRowBatch],
) -> Result<(), StorageError> {
    let mut settlement_cache = HashMap::<BatchId, Option<BatchFate>>::new();
    for row in rows {
        let settlement = if let Some(settlement) = settlement_cache.get(&row.batch_id) {
            settlement
        } else {
            let settlement = storage.load_authoritative_batch_fate(row.batch_id)?;
            settlement_cache.insert(row.batch_id, settlement);
            settlement_cache
                .get(&row.batch_id)
                .expect("settlement cache should contain inserted batch")
        };

        row.confirmed_tier = match settlement {
            Some(settlement) => batch_fate_confirmed_tier_for_row(settlement, row),
            None => row.confirmed_tier,
        };
    }
    Ok(())
}

/// Errors from storage operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageError {
    NotFound,
    IoError(String),
    IndexKeyTooLarge {
        table: String,
        column: String,
        branch: String,
        key_bytes: usize,
        max_key_bytes: usize,
    },
    SecurityError(String),
    /// v18 item 4: the store's explicit transaction was ended behind its back (SQLite's own
    /// full rollback on NOMEM/IOERR/INTERRUPT/FULL) with landed writes in it. Strict: the
    /// store reports it on every transaction boundary until it is reopened.
    LostWrites {
        detail: String,
    },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::NotFound => write!(f, "not found"),
            StorageError::IoError(message) => write!(f, "{message}"),
            StorageError::IndexKeyTooLarge {
                table,
                column,
                branch,
                key_bytes,
                max_key_bytes,
            } => write!(
                f,
                "indexed value too large for {table}.{column} on branch {branch}: index key would be {key_bytes} bytes (max {max_key_bytes})"
            ),
            StorageError::SecurityError(message) => write!(f, "security error: {message}"),
            StorageError::LostWrites { detail } => write!(f, "lost writes: {detail}"),
        }
    }
}

impl std::error::Error for StorageError {}

pub(crate) fn validate_index_value_size(
    table: &str,
    column: &str,
    branch: &str,
    value: &Value,
) -> Result<(), StorageError> {
    key_codec::validate_index_entry_size(table, column, branch, value)
}

pub type RowLocatorRows = Vec<(ObjectId, RowLocator)>;
pub type RawTableRows = Vec<(String, Vec<u8>)>;
pub type RawTableKeys = Vec<String>;

const ROW_LOCATOR_TABLE: &str = "__row_locator";
const VISIBLE_ROW_TABLE_LOCATOR_TABLE: &str = "__visible_row_table_locator";
const HISTORY_ROW_BATCH_TABLE_LOCATOR_TABLE: &str = "__history_row_batch_table_locator";
const LOCAL_BATCH_RECORD_TABLE: &str = "__local_batch_record";
const LOCAL_BATCH_ROW_INDEX_TABLE: &str = "__local_batch_row_index";
const AUTHORITATIVE_BATCH_SETTLEMENT_TABLE: &str = "__authoritative_batch_settlement";
const ACKNOWLEDGED_REJECTED_BATCH_TABLE: &str = "__acknowledged_rejected_batch";
const SEALED_BATCH_SUBMISSION_TABLE: &str = "__sealed_batch_submission";
const RAW_TABLE_HEADER_TABLE: &str = "__raw_table_header";
const BRANCH_ORD_BY_NAME_TABLE: &str = "__branch_ord_by_name";
const BRANCH_NAME_BY_ORD_TABLE: &str = "__branch_name_by_ord";
const BRANCH_ORD_META_TABLE: &str = "__branch_ord_meta";
/// Which generation-set each table was last successfully swept for. Turns the
/// defect-27 repair from a per-boot tax into a once-per-deployment one.
const VISIBLE_FAMILY_SWEEP_TABLE: &str = "__visible_family_sweep";
const BRANCH_ORD_NEXT_ORD_KEY: &str = "next_ord";
pub(crate) const STORE_MANIFEST_KEY: &str = "__jazz_store_manifest";
const STORE_MANIFEST_MAGIC: &[u8; 10] = b"JAZZSTORE1";
const STORE_FORMAT_V3: i32 = 3;
const ROW_STORAGE_FORMAT_V3: i32 = 3;
const ROW_LOCATOR_STORAGE_FORMAT_V1: i32 = 1;
const EXACT_ROW_TABLE_LOCATOR_STORAGE_FORMAT_V1: i32 = 1;
const CATALOGUE_STORAGE_FORMAT_V1: i32 = 1;
const BRANCH_ORD_BY_NAME_FORMAT_V1: i32 = 1;
const BRANCH_NAME_BY_ORD_FORMAT_V1: i32 = 1;
const BRANCH_ORD_META_FORMAT_V1: i32 = 1;
const VISIBLE_FAMILY_SWEEP_FORMAT_V1: i32 = 1;
const SEALED_BATCH_SUBMISSION_FORMAT_V2: i32 = 2;
const AUTHORITATIVE_BATCH_SETTLEMENT_FORMAT_V2: i32 = 2;
const ACKNOWLEDGED_REJECTED_BATCH_FORMAT_V1: i32 = 1;
const LOCAL_BATCH_RECORD_FORMAT_V3: i32 = 3;
const LOCAL_BATCH_ROW_INDEX_FORMAT_V1: i32 = 1;

pub type BranchOrd = i32;

const STORAGE_KIND_ROW_LOCATOR: &str = "row_locator";
const STORAGE_KIND_VISIBLE_ROW_TABLE_LOCATOR: &str = "visible_row_table_locator";
const STORAGE_KIND_HISTORY_ROW_BATCH_TABLE_LOCATOR: &str = "history_row_batch_table_locator";
const STORAGE_KIND_BRANCH_ORD_BY_NAME: &str = "branch_ord_by_name";
const STORAGE_KIND_BRANCH_NAME_BY_ORD: &str = "branch_name_by_ord";
const STORAGE_KIND_BRANCH_ORD_META: &str = "branch_ord_meta";
const STORAGE_KIND_VISIBLE_FAMILY_SWEEP: &str = "visible_family_sweep";
const STORAGE_KIND_LOCAL_BATCH_RECORD: &str = "local_batch_record";
const STORAGE_KIND_LOCAL_BATCH_ROW_INDEX: &str = "local_batch_row_index";
const STORAGE_KIND_AUTHORITATIVE_BATCH_SETTLEMENT: &str = "authoritative_batch_settlement";
const STORAGE_KIND_ACKNOWLEDGED_REJECTED_BATCH: &str = "acknowledged_rejected_batch";
const STORAGE_KIND_SEALED_BATCH_SUBMISSION: &str = "sealed_batch_submission";
const STORAGE_KIND_CATALOGUE: &str = "catalogue";
#[cfg(feature = "sqlite")]
pub(crate) const SQLITE_STORE_KIND: &str = "sqlite";
#[cfg(feature = "rocksdb")]
pub(crate) const ROCKSDB_STORE_KIND: &str = "rocksdb";
pub(crate) const OPFS_BTREE_STORE_KIND: &str = "opfs_btree";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreManifest {
    pub store_kind: String,
    pub store_format_version: i32,
}

pub(crate) fn expected_store_manifest(store_kind: &str) -> StoreManifest {
    StoreManifest {
        store_kind: store_kind.to_string(),
        store_format_version: STORE_FORMAT_V3,
    }
}

pub(crate) fn encode_store_manifest(manifest: &StoreManifest) -> Result<Vec<u8>, StorageError> {
    let kind_bytes = manifest.store_kind.as_bytes();
    if kind_bytes.len() > u8::MAX as usize {
        return Err(StorageError::IoError(format!(
            "store manifest kind too long: {} bytes",
            kind_bytes.len()
        )));
    }
    let mut bytes = Vec::with_capacity(
        STORE_MANIFEST_MAGIC.len() + std::mem::size_of::<i32>() + 1 + kind_bytes.len(),
    );
    bytes.extend_from_slice(STORE_MANIFEST_MAGIC);
    bytes.extend_from_slice(&manifest.store_format_version.to_le_bytes());
    bytes.push(kind_bytes.len() as u8);
    bytes.extend_from_slice(kind_bytes);
    Ok(bytes)
}

pub(crate) fn decode_store_manifest(bytes: &[u8]) -> Result<StoreManifest, StorageError> {
    let min_len = STORE_MANIFEST_MAGIC.len() + std::mem::size_of::<i32>() + 1;
    if bytes.len() < min_len {
        return Err(StorageError::IoError(
            "store manifest too short".to_string(),
        ));
    }
    if &bytes[..STORE_MANIFEST_MAGIC.len()] != STORE_MANIFEST_MAGIC {
        return Err(StorageError::IoError(
            "store manifest magic mismatch".to_string(),
        ));
    }
    let mut version_bytes = [0u8; 4];
    version_bytes
        .copy_from_slice(&bytes[STORE_MANIFEST_MAGIC.len()..STORE_MANIFEST_MAGIC.len() + 4]);
    let store_format_version = i32::from_le_bytes(version_bytes);
    let kind_len = bytes[STORE_MANIFEST_MAGIC.len() + 4] as usize;
    let kind_start = STORE_MANIFEST_MAGIC.len() + 5;
    let kind_end = kind_start + kind_len;
    if bytes.len() != kind_end {
        return Err(StorageError::IoError(
            "store manifest trailing bytes mismatch".to_string(),
        ));
    }
    let store_kind = String::from_utf8(bytes[kind_start..kind_end].to_vec())
        .map_err(|err| StorageError::IoError(format!("invalid store manifest kind utf8: {err}")))?;
    Ok(StoreManifest {
        store_kind,
        store_format_version,
    })
}

pub(crate) fn validate_store_manifest(
    actual: &StoreManifest,
    expected: &StoreManifest,
) -> Result<(), StorageError> {
    if actual.store_kind != expected.store_kind {
        return Err(StorageError::IoError(format!(
            "store manifest kind mismatch: expected {}, got {}",
            expected.store_kind, actual.store_kind
        )));
    }
    if actual.store_format_version != expected.store_format_version {
        return Err(StorageError::IoError(format!(
            "store manifest version mismatch for {}: expected {}, got {}",
            expected.store_kind, expected.store_format_version, actual.store_format_version
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawTableHeader {
    pub storage_kind: SharedString,
    pub storage_format_version: i32,
    pub logical_table_name: Option<SharedString>,
    pub schema_hash: Option<SchemaHash>,
    pub row_descriptor_bytes: Option<Vec<u8>>,
}

impl RawTableHeader {
    pub fn new(storage_kind: impl Into<String>, storage_format_version: i32) -> Self {
        Self {
            storage_kind: storage_kind.into().into(),
            storage_format_version,
            logical_table_name: None,
            schema_hash: None,
            row_descriptor_bytes: None,
        }
    }

    pub fn row_raw_table(
        kind: RowRawTableKind,
        table_name: impl Into<String>,
        schema_hash: SchemaHash,
        user_descriptor: &RowDescriptor,
    ) -> Self {
        let table_name: SharedString = table_name.into().into();
        Self {
            storage_kind: kind.storage_kind().into(),
            storage_format_version: ROW_STORAGE_FORMAT_V3,
            logical_table_name: Some(table_name),
            schema_hash: Some(schema_hash),
            row_descriptor_bytes: Some(
                crate::schema_manager::encoding::encode_row_descriptor_bytes(user_descriptor),
            ),
        }
    }

    pub fn system(storage_kind: impl Into<String>, storage_format_version: i32) -> Self {
        Self::new(storage_kind, storage_format_version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowLocator {
    pub table: SharedString,
    pub origin_schema_hash: Option<SchemaHash>,
}

/// Resolve the table that owns a row's history, falling back to the caller's
/// table when the row has not been located yet.
pub(crate) fn history_table_for_row<H: Storage + ?Sized>(
    storage: &H,
    row_id: ObjectId,
    fallback_table: &str,
) -> String {
    storage
        .load_row_locator(row_id)
        .ok()
        .flatten()
        .map(|locator| locator.table.to_string())
        .unwrap_or_else(|| fallback_table.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactRowTableLocator {
    pub row_raw_table: SharedString,
    pub table_name: SharedString,
    pub schema_hash: SchemaHash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RowRawTableKind {
    Visible,
    History,
}

impl RowRawTableKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Visible => "visible",
            Self::History => "history",
        }
    }

    fn storage_kind(self) -> &'static str {
        match self {
            Self::Visible => "visible_rows",
            Self::History => "row_history",
        }
    }

    fn from_str(raw: &str) -> Result<Self, StorageError> {
        match raw {
            "visible" => Ok(Self::Visible),
            "history" => Ok(Self::History),
            other => Err(StorageError::IoError(format!(
                "unknown row raw table kind '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RowRawTableId {
    pub kind: RowRawTableKind,
    pub table_name: SharedString,
    pub schema_hash: SchemaHash,
    pub raw_table_name: SharedString,
}

impl RowRawTableId {
    pub fn new(
        kind: RowRawTableKind,
        table_name: impl Into<String>,
        schema_hash: SchemaHash,
    ) -> Self {
        let table_name: SharedString = table_name.into().into();
        Self {
            kind,
            raw_table_name: format!("rowtable:{}:{}:{}", kind.as_str(), table_name, schema_hash)
                .into(),
            table_name,
            schema_hash,
        }
    }

    pub fn raw_table_name(&self) -> &str {
        self.raw_table_name.as_str()
    }

    fn parse_raw_table_name(raw: &str) -> Result<Self, StorageError> {
        let Some(rest) = raw.strip_prefix("rowtable:") else {
            return Err(StorageError::IoError(format!(
                "invalid row raw table id '{raw}'"
            )));
        };
        let mut parts = rest.splitn(3, ':');
        let kind =
            RowRawTableKind::from_str(parts.next().ok_or_else(|| {
                StorageError::IoError(format!("invalid row raw table id '{raw}'"))
            })?)?;
        let table_name = parts
            .next()
            .ok_or_else(|| StorageError::IoError(format!("invalid row raw table id '{raw}'")))?;
        let schema_hash = parts
            .next()
            .and_then(SchemaHash::from_hex)
            .ok_or_else(|| StorageError::IoError(format!("invalid row raw table id '{raw}'")))?;
        Ok(Self {
            kind,
            table_name: table_name.into(),
            schema_hash,
            raw_table_name: raw.into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexMutation<'a> {
    Insert {
        table: &'a str,
        column: &'a str,
        branch: &'a str,
        value: Value,
        row_id: ObjectId,
    },
    Remove {
        table: &'a str,
        column: &'a str,
        branch: &'a str,
        value: Value,
        row_id: ObjectId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawTableMutation<'a> {
    Put {
        table: &'a str,
        key: &'a str,
        value: &'a [u8],
    },
    Delete {
        table: &'a str,
        key: &'a str,
    },
}

pub struct HistoryRowBytes<'a> {
    pub row_raw_table: &'a str,
    pub branch: &'a str,
    pub row_id: ObjectId,
    pub batch_id: BatchId,
    pub bytes: &'a [u8],
}

#[doc(hidden)]
pub struct OwnedHistoryRowBytes {
    pub row_raw_table_id: RowRawTableId,
    pub row_raw_table: String,
    pub user_descriptor: Arc<RowDescriptor>,
    pub branch: String,
    pub row_id: ObjectId,
    pub batch_id: BatchId,
    pub needs_exact_locator: bool,
    pub bytes: Vec<u8>,
}

pub struct VisibleRowBytes<'a> {
    pub row_raw_table: &'a str,
    pub branch: &'a str,
    pub row_id: ObjectId,
    pub bytes: &'a [u8],
}

#[doc(hidden)]
pub struct OwnedVisibleRowBytes {
    pub row_raw_table_id: RowRawTableId,
    pub row_raw_table: String,
    pub user_descriptor: Arc<RowDescriptor>,
    pub branch: String,
    pub row_id: ObjectId,
    pub needs_exact_locator: bool,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ResolvedRowTable {
    row_raw_table: String,
    user_descriptor: Arc<RowDescriptor>,
    row_codecs: Arc<FlatRowCodecs>,
}

#[derive(Clone)]
pub(crate) struct PreparedRowTableContext {
    pub history_row_raw_table_id: RowRawTableId,
    pub visible_row_raw_table_id: RowRawTableId,
    pub user_descriptor: Arc<RowDescriptor>,
}

#[derive(Clone)]
pub(crate) struct PreparedRowWriteContext {
    pub table_context: Arc<PreparedRowTableContext>,
    pub needs_exact_locator: bool,
}

impl PreparedRowWriteContext {
    pub(crate) fn history_row_raw_table_id(&self) -> &RowRawTableId {
        &self.table_context.history_row_raw_table_id
    }

    pub(crate) fn visible_row_raw_table_id(&self) -> &RowRawTableId {
        &self.table_context.visible_row_raw_table_id
    }

    pub(crate) fn user_descriptor(&self) -> &Arc<RowDescriptor> {
        &self.table_context.user_descriptor
    }
}

type CatalogueUserDescriptorCache = HashMap<(usize, String, SchemaHash), Arc<RowDescriptor>>;
type BranchSchemaHashCache = HashMap<(usize, String), Vec<SchemaHash>>;
type TableCatalogueDescriptorCache =
    HashMap<(usize, String), Vec<(SchemaHash, crate::query_manager::types::RowDescriptor)>>;

fn row_raw_table_descriptor_cache() -> &'static Mutex<HashMap<String, Arc<RowDescriptor>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<RowDescriptor>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn next_storage_cache_namespace() -> usize {
    static NEXT_STORAGE_CACHE_NAMESPACE: AtomicUsize = AtomicUsize::new(1);
    NEXT_STORAGE_CACHE_NAMESPACE.fetch_add(1, Ordering::Relaxed)
}

fn raw_table_header_cache() -> &'static Mutex<HashMap<(usize, String), RawTableHeader>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, String), RawTableHeader>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_raw_table_header_with_storage<H: Storage + ?Sized>(
    storage: &H,
    raw_table: &str,
) -> Option<RawTableHeader> {
    raw_table_header_cache()
        .lock()
        .expect("raw table header cache poisoned")
        .get(&(storage.storage_cache_namespace(), raw_table.to_string()))
        .cloned()
}

fn cache_raw_table_header_with_storage<H: Storage + ?Sized>(
    storage: &H,
    raw_table: &str,
    header: RawTableHeader,
) {
    raw_table_header_cache()
        .lock()
        .expect("raw table header cache poisoned")
        .insert(
            (storage.storage_cache_namespace(), raw_table.to_string()),
            header,
        );
}

fn validated_raw_table_cache() -> &'static Mutex<HashSet<(usize, String)>> {
    static CACHE: OnceLock<Mutex<HashSet<(usize, String)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn raw_table_validated_with_storage<H: Storage + ?Sized>(storage: &H, raw_table: &str) -> bool {
    validated_raw_table_cache()
        .lock()
        .expect("validated raw table cache poisoned")
        .contains(&(storage.storage_cache_namespace(), raw_table.to_string()))
}

fn cache_validated_raw_table_with_storage<H: Storage + ?Sized>(storage: &H, raw_table: &str) {
    validated_raw_table_cache()
        .lock()
        .expect("validated raw table cache poisoned")
        .insert((storage.storage_cache_namespace(), raw_table.to_string()));
}

fn invalidate_validated_raw_table_with_storage<H: Storage + ?Sized>(storage: &H, raw_table: &str) {
    validated_raw_table_cache()
        .lock()
        .expect("validated raw table cache poisoned")
        .remove(&(storage.storage_cache_namespace(), raw_table.to_string()));
}

fn cached_row_descriptor(raw_table: &str) -> Option<Arc<RowDescriptor>> {
    row_raw_table_descriptor_cache()
        .lock()
        .expect("row raw table descriptor cache poisoned")
        .get(raw_table)
        .cloned()
}

fn cache_row_descriptor(raw_table: &str, descriptor: Arc<RowDescriptor>) {
    row_raw_table_descriptor_cache()
        .lock()
        .expect("row raw table descriptor cache poisoned")
        .insert(raw_table.to_string(), descriptor);
}

fn catalogue_user_descriptor_cache() -> &'static Mutex<CatalogueUserDescriptorCache> {
    static CACHE: OnceLock<Mutex<CatalogueUserDescriptorCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn branch_schema_hash_cache() -> &'static Mutex<BranchSchemaHashCache> {
    static CACHE: OnceLock<Mutex<BranchSchemaHashCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn table_catalogue_descriptor_cache() -> &'static Mutex<TableCatalogueDescriptorCache> {
    static CACHE: OnceLock<Mutex<TableCatalogueDescriptorCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn invalidate_catalogue_lookup_caches_with_storage<H: Storage + ?Sized>(storage: &H) {
    let namespace = storage.storage_cache_namespace();
    catalogue_user_descriptor_cache()
        .lock()
        .expect("catalogue user descriptor cache poisoned")
        .retain(|(cached_namespace, _, _), _| *cached_namespace != namespace);
    branch_schema_hash_cache()
        .lock()
        .expect("branch schema hash cache poisoned")
        .retain(|(cached_namespace, _), _| *cached_namespace != namespace);
    table_catalogue_descriptor_cache()
        .lock()
        .expect("table catalogue descriptor cache poisoned")
        .retain(|(cached_namespace, _), _| *cached_namespace != namespace);
}

fn cached_catalogue_user_descriptor_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table_name: &str,
    schema_hash: SchemaHash,
) -> Option<Arc<RowDescriptor>> {
    catalogue_user_descriptor_cache()
        .lock()
        .expect("catalogue user descriptor cache poisoned")
        .get(&(
            storage.storage_cache_namespace(),
            table_name.to_string(),
            schema_hash,
        ))
        .cloned()
}

fn cache_catalogue_user_descriptor_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table_name: &str,
    schema_hash: SchemaHash,
    descriptor: Arc<RowDescriptor>,
) {
    catalogue_user_descriptor_cache()
        .lock()
        .expect("catalogue user descriptor cache poisoned")
        .insert(
            (
                storage.storage_cache_namespace(),
                table_name.to_string(),
                schema_hash,
            ),
            descriptor,
        );
}

fn metadata_raw_key(id: ObjectId) -> String {
    hex::encode(id.uuid().as_bytes())
}

fn decode_metadata_raw_key(key: &str) -> Result<ObjectId, StorageError> {
    let bytes = hex::decode(key)
        .map_err(|err| StorageError::IoError(format!("invalid metadata key '{key}': {err}")))?;
    let uuid = uuid::Uuid::from_slice(&bytes)
        .map_err(|err| StorageError::IoError(format!("invalid metadata uuid '{key}': {err}")))?;
    Ok(ObjectId::from_uuid(uuid))
}

pub(crate) fn row_locator_from_metadata(metadata: &HashMap<String, String>) -> Option<RowLocator> {
    Some(RowLocator {
        table: metadata.get(MetadataKey::Table.as_str())?.clone().into(),
        origin_schema_hash: metadata
            .get(MetadataKey::OriginSchemaHash.as_str())
            .and_then(|raw_hash| SchemaHash::from_hex(raw_hash)),
    })
}

pub(crate) fn metadata_from_row_locator(locator: &RowLocator) -> HashMap<String, String> {
    let mut metadata = HashMap::from([(MetadataKey::Table.to_string(), locator.table.to_string())]);
    if let Some(origin_schema_hash) = locator.origin_schema_hash {
        metadata.insert(
            MetadataKey::OriginSchemaHash.to_string(),
            origin_schema_hash.to_string(),
        );
    }
    metadata
}

fn encode_row_locator(locator: &RowLocator) -> Result<Vec<u8>, StorageError> {
    postcard::to_allocvec(locator)
        .map_err(|err| StorageError::IoError(format!("serialize row locator: {err}")))
}

fn decode_row_locator(bytes: &[u8]) -> Result<RowLocator, StorageError> {
    postcard::from_bytes(bytes)
        .map_err(|err| StorageError::IoError(format!("deserialize row locator: {err}")))
}

fn exact_row_table_locator_storage_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("row_raw_table", ColumnType::Text),
        ColumnDescriptor::new("table_name", ColumnType::Text),
        ColumnDescriptor::new("schema_hash", ColumnType::Bytea),
    ])
}

fn encode_exact_row_table_locator(locator: &ExactRowTableLocator) -> Result<Vec<u8>, StorageError> {
    encode_row(
        &exact_row_table_locator_storage_descriptor(),
        &[
            Value::Text(locator.row_raw_table.to_string()),
            Value::Text(locator.table_name.to_string()),
            Value::Bytea(locator.schema_hash.as_bytes().to_vec()),
        ],
    )
    .map_err(|err| StorageError::IoError(format!("encode exact row table locator: {err}")))
}

fn decode_exact_row_table_locator(bytes: &[u8]) -> Result<ExactRowTableLocator, StorageError> {
    let values = decode_row(&exact_row_table_locator_storage_descriptor(), bytes)
        .map_err(|err| StorageError::IoError(format!("decode exact row table locator: {err}")))?;
    let [row_raw_table, table_name, schema_hash] = values.as_slice() else {
        return Err(StorageError::IoError(
            "malformed exact row table locator".to_string(),
        ));
    };
    let row_raw_table = match row_raw_table {
        Value::Text(raw) => SharedString::from(raw.clone()),
        other => {
            return Err(StorageError::IoError(format!(
                "exact row table locator row_raw_table was {other:?}"
            )));
        }
    };
    let table_name = match table_name {
        Value::Text(raw) => SharedString::from(raw.clone()),
        other => {
            return Err(StorageError::IoError(format!(
                "exact row table locator table_name was {other:?}"
            )));
        }
    };
    let schema_hash = match schema_hash {
        Value::Bytea(bytes) => SchemaHash::from_bytes(bytes.clone().try_into().map_err(|_| {
            StorageError::IoError(
                "exact row table locator schema_hash must be 32 bytes".to_string(),
            )
        })?),
        other => {
            return Err(StorageError::IoError(format!(
                "exact row table locator schema_hash was {other:?}"
            )));
        }
    };
    Ok(ExactRowTableLocator {
        row_raw_table,
        table_name,
        schema_hash,
    })
}

fn visible_row_table_locator_key(branch: &str, row_id: ObjectId) -> String {
    key_codec::visible_row_raw_table_key(branch, row_id)
}

fn history_row_batch_table_locator_key(
    row_id: ObjectId,
    branch: &str,
    batch_id: BatchId,
) -> String {
    key_codec::history_row_raw_table_key(row_id, branch, batch_id)
}

fn local_batch_record_key(batch_id: BatchId) -> String {
    const PREFIX: &str = "batch:";
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

    let mut key = String::with_capacity(PREFIX.len() + batch_id.as_bytes().len() * 2);
    key.push_str(PREFIX);
    for &byte in batch_id.as_bytes() {
        key.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        key.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    key
}

fn decode_local_batch_record_key(key: &str) -> Result<BatchId, StorageError> {
    let Some(hex_id) = key.strip_prefix("batch:") else {
        return Err(StorageError::IoError(format!(
            "invalid local batch record key '{key}'"
        )));
    };
    let bytes = hex::decode(hex_id).map_err(|err| {
        StorageError::IoError(format!("invalid local batch record key '{key}': {err}"))
    })?;
    let bytes: [u8; 16] = bytes.try_into().map_err(|_| {
        StorageError::IoError(format!(
            "invalid local batch record batch id '{key}': expected 16 bytes, got {}",
            hex_id.len() / 2
        ))
    })?;
    Ok(BatchId(bytes))
}

fn branch_ord_by_name_storage_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![ColumnDescriptor::new(
        "branch_ord",
        ColumnType::Integer,
    )])
}

fn branch_name_by_ord_storage_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![ColumnDescriptor::new("branch_name", ColumnType::Text)])
}

fn branch_ord_meta_storage_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![ColumnDescriptor::new("next_ord", ColumnType::Integer)])
}

fn branch_ord_by_name_key(branch_name: BranchName) -> String {
    branch_name.as_str().to_string()
}

fn branch_name_by_ord_key(branch_ord: BranchOrd) -> Result<String, StorageError> {
    if branch_ord < 1 {
        return Err(StorageError::IoError(format!(
            "branch ord must be >= 1, got {branch_ord}"
        )));
    }
    Ok(format!("{branch_ord:010}"))
}

fn encode_branch_ord_value(branch_ord: BranchOrd) -> Result<Vec<u8>, StorageError> {
    if branch_ord < 1 {
        return Err(StorageError::IoError(format!(
            "branch ord must be >= 1, got {branch_ord}"
        )));
    }
    encode_row(
        &branch_ord_by_name_storage_descriptor(),
        &[Value::Integer(branch_ord)],
    )
    .map_err(|err| StorageError::IoError(format!("encode branch ord value: {err}")))
}

fn decode_branch_ord_value(bytes: &[u8]) -> Result<BranchOrd, StorageError> {
    let values = decode_row(&branch_ord_by_name_storage_descriptor(), bytes)
        .map_err(|err| StorageError::IoError(format!("decode branch ord value: {err}")))?;
    let [branch_ord] = values.as_slice() else {
        return Err(StorageError::IoError(
            "unexpected branch ord row shape".to_string(),
        ));
    };
    match branch_ord {
        Value::Integer(branch_ord) if *branch_ord >= 1 => Ok(*branch_ord),
        Value::Integer(branch_ord) => Err(StorageError::IoError(format!(
            "branch ord must be >= 1, got {branch_ord}"
        ))),
        other => Err(StorageError::IoError(format!(
            "branch ord row must contain Integer, got {other:?}"
        ))),
    }
}

fn encode_branch_name_value(branch_name: BranchName) -> Result<Vec<u8>, StorageError> {
    encode_row(
        &branch_name_by_ord_storage_descriptor(),
        &[Value::Text(branch_name.as_str().to_string())],
    )
    .map_err(|err| StorageError::IoError(format!("encode branch name value: {err}")))
}

fn decode_branch_name_value(bytes: &[u8]) -> Result<BranchName, StorageError> {
    let values = decode_row(&branch_name_by_ord_storage_descriptor(), bytes)
        .map_err(|err| StorageError::IoError(format!("decode branch name value: {err}")))?;
    let [branch_name] = values.as_slice() else {
        return Err(StorageError::IoError(
            "unexpected branch name row shape".to_string(),
        ));
    };
    match branch_name {
        Value::Text(branch_name) => Ok(BranchName::new(branch_name.clone())),
        other => Err(StorageError::IoError(format!(
            "branch name row must contain Text, got {other:?}"
        ))),
    }
}

fn encode_branch_ord_meta(next_ord: BranchOrd) -> Result<Vec<u8>, StorageError> {
    if next_ord < 1 {
        return Err(StorageError::IoError(format!(
            "next branch ord must be >= 1, got {next_ord}"
        )));
    }
    encode_row(
        &branch_ord_meta_storage_descriptor(),
        &[Value::Integer(next_ord)],
    )
    .map_err(|err| StorageError::IoError(format!("encode branch ord meta: {err}")))
}

fn decode_branch_ord_meta(bytes: &[u8]) -> Result<BranchOrd, StorageError> {
    let values = decode_row(&branch_ord_meta_storage_descriptor(), bytes)
        .map_err(|err| StorageError::IoError(format!("decode branch ord meta: {err}")))?;
    let [next_ord] = values.as_slice() else {
        return Err(StorageError::IoError(
            "unexpected branch ord meta row shape".to_string(),
        ));
    };
    match next_ord {
        Value::Integer(next_ord) if *next_ord >= 1 => Ok(*next_ord),
        Value::Integer(next_ord) => Err(StorageError::IoError(format!(
            "next branch ord must be >= 1, got {next_ord}"
        ))),
        other => Err(StorageError::IoError(format!(
            "branch ord meta row must contain Integer, got {other:?}"
        ))),
    }
}

/// The store's ONE branch, when it has exactly one.
///
/// Two point lookups against the branch registry — cheap enough to ask on a
/// write path, unlike enumerating a row's history to find out which branches
/// it spans. Ords are allocated from 1, so a next-ord of 2 means a single
/// branch was ever registered.
pub(crate) fn sole_branch_name<H: Storage + ?Sized>(
    storage: &H,
) -> Result<Option<BranchName>, StorageError> {
    if load_next_branch_ord(storage)? != 2 {
        return Ok(None);
    }
    storage.load_branch_name_by_ord(1)
}

/// Does this row have history on a branch other than `incoming_branch`?
///
/// A thin pass-through to `Storage::row_has_history_outside_branch`, kept as a
/// named function because the caller reads better for it and because an earlier
/// version answered this from the branch-ord registry — which is written by seal
/// persistence, not by history application, so it missed branches that only ever
/// had rows written on them.
pub(crate) fn row_has_history_on_another_branch<H: Storage + ?Sized>(
    storage: &H,
    history_table: &str,
    row_id: ObjectId,
    incoming_branch: &str,
) -> Result<bool, StorageError> {
    storage.row_has_history_outside_branch(history_table, row_id, incoming_branch)
}

fn load_next_branch_ord<H: Storage + ?Sized>(storage: &H) -> Result<BranchOrd, StorageError> {
    match storage.raw_table_get(BRANCH_ORD_META_TABLE, BRANCH_ORD_NEXT_ORD_KEY)? {
        Some(bytes) => {
            ensure_system_raw_table_header_validated_once(
                storage,
                BRANCH_ORD_META_TABLE,
                STORAGE_KIND_BRANCH_ORD_META,
                BRANCH_ORD_META_FORMAT_V1,
            )?;
            decode_branch_ord_meta(&bytes)
        }
        None => Ok(1),
    }
}

fn encode_raw_table_header(header: &RawTableHeader) -> Result<Vec<u8>, StorageError> {
    encode_row(
        &raw_table_header_storage_descriptor(),
        &[
            Value::Text(header.storage_kind.to_string()),
            Value::Integer(header.storage_format_version),
            header
                .logical_table_name
                .as_ref()
                .map(|name| Value::Text(name.to_string()))
                .unwrap_or(Value::Null),
            header
                .schema_hash
                .map(|schema_hash| Value::Bytea(schema_hash.as_bytes().to_vec()))
                .unwrap_or(Value::Null),
            header
                .row_descriptor_bytes
                .as_ref()
                .map(|bytes| Value::Bytea(bytes.clone()))
                .unwrap_or(Value::Null),
        ],
    )
    .map_err(|err| StorageError::IoError(format!("encode raw table header: {err}")))
}

fn decode_raw_table_header(bytes: &[u8]) -> Result<RawTableHeader, StorageError> {
    let values = decode_row(&raw_table_header_storage_descriptor(), bytes)
        .map_err(|err| StorageError::IoError(format!("decode raw table header: {err}")))?;
    let [
        storage_kind,
        storage_format_version,
        logical_table_name,
        schema_hash,
        row_descriptor_bytes,
    ] = values.as_slice()
    else {
        return Err(StorageError::IoError(
            "unexpected raw table header shape".to_string(),
        ));
    };

    let storage_kind = match storage_kind {
        Value::Text(value) => SharedString::from(value.clone()),
        other => {
            return Err(StorageError::IoError(format!(
                "raw table header storage_kind was {other:?}"
            )));
        }
    };
    let storage_format_version = match storage_format_version {
        Value::Integer(value) => *value,
        other => {
            return Err(StorageError::IoError(format!(
                "raw table header storage_format_version was {other:?}"
            )));
        }
    };
    let logical_table_name = match logical_table_name {
        Value::Null => None,
        Value::Text(value) => Some(SharedString::from(value.clone())),
        other => {
            return Err(StorageError::IoError(format!(
                "raw table header logical_table_name was {other:?}"
            )));
        }
    };
    let schema_hash = match schema_hash {
        Value::Null => None,
        Value::Bytea(bytes) => Some(SchemaHash::from_bytes(bytes.clone().try_into().map_err(
            |_| StorageError::IoError("raw table header schema_hash must be 32 bytes".to_string()),
        )?)),
        other => {
            return Err(StorageError::IoError(format!(
                "raw table header schema_hash was {other:?}"
            )));
        }
    };
    let row_descriptor_bytes = match row_descriptor_bytes {
        Value::Null => None,
        Value::Bytea(bytes) => Some(bytes.clone()),
        other => {
            return Err(StorageError::IoError(format!(
                "raw table header row_descriptor_bytes was {other:?}"
            )));
        }
    };

    Ok(RawTableHeader {
        storage_kind,
        storage_format_version,
        logical_table_name,
        schema_hash,
        row_descriptor_bytes,
    })
}

fn raw_table_header_storage_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("storage_kind", ColumnType::Text),
        ColumnDescriptor::new("storage_format_version", ColumnType::Integer),
        ColumnDescriptor::new("logical_table_name", ColumnType::Text).nullable(),
        ColumnDescriptor::new("schema_hash", ColumnType::Bytea).nullable(),
        ColumnDescriptor::new("row_descriptor_bytes", ColumnType::Bytea).nullable(),
    ])
}

fn supported_storage_format_version(storage_kind: &str) -> Result<i32, StorageError> {
    match storage_kind {
        STORAGE_KIND_ROW_LOCATOR => Ok(ROW_LOCATOR_STORAGE_FORMAT_V1),
        STORAGE_KIND_VISIBLE_ROW_TABLE_LOCATOR => Ok(EXACT_ROW_TABLE_LOCATOR_STORAGE_FORMAT_V1),
        STORAGE_KIND_HISTORY_ROW_BATCH_TABLE_LOCATOR => {
            Ok(EXACT_ROW_TABLE_LOCATOR_STORAGE_FORMAT_V1)
        }
        STORAGE_KIND_BRANCH_ORD_BY_NAME => Ok(BRANCH_ORD_BY_NAME_FORMAT_V1),
        STORAGE_KIND_BRANCH_NAME_BY_ORD => Ok(BRANCH_NAME_BY_ORD_FORMAT_V1),
        STORAGE_KIND_BRANCH_ORD_META => Ok(BRANCH_ORD_META_FORMAT_V1),
        STORAGE_KIND_VISIBLE_FAMILY_SWEEP => Ok(VISIBLE_FAMILY_SWEEP_FORMAT_V1),
        STORAGE_KIND_LOCAL_BATCH_RECORD => Ok(LOCAL_BATCH_RECORD_FORMAT_V3),
        STORAGE_KIND_LOCAL_BATCH_ROW_INDEX => Ok(LOCAL_BATCH_ROW_INDEX_FORMAT_V1),
        STORAGE_KIND_SEALED_BATCH_SUBMISSION => Ok(SEALED_BATCH_SUBMISSION_FORMAT_V2),
        STORAGE_KIND_AUTHORITATIVE_BATCH_SETTLEMENT => Ok(AUTHORITATIVE_BATCH_SETTLEMENT_FORMAT_V2),
        STORAGE_KIND_ACKNOWLEDGED_REJECTED_BATCH => Ok(ACKNOWLEDGED_REJECTED_BATCH_FORMAT_V1),
        STORAGE_KIND_CATALOGUE => Ok(CATALOGUE_STORAGE_FORMAT_V1),
        "visible_rows" | "row_history" => Ok(ROW_STORAGE_FORMAT_V3),
        other => Err(StorageError::IoError(format!(
            "unknown raw table header storage_kind '{other}'"
        ))),
    }
}

fn validate_raw_table_header_storage_format(
    raw_table: &str,
    header: &RawTableHeader,
) -> Result<(), StorageError> {
    let expected_version = supported_storage_format_version(header.storage_kind.as_str())?;
    if header.storage_format_version != expected_version {
        return Err(StorageError::IoError(format!(
            "raw table header storage_format_version mismatch for {raw_table}: expected {expected_version}, got {}",
            header.storage_format_version
        )));
    }
    Ok(())
}

fn validate_system_raw_table_header(
    raw_table: &str,
    header: RawTableHeader,
    expected_storage_kind: &str,
    expected_storage_format_version: i32,
) -> Result<RawTableHeader, StorageError> {
    validate_raw_table_header_storage_format(raw_table, &header)?;
    if header.storage_kind.as_str() != expected_storage_kind {
        return Err(StorageError::IoError(format!(
            "raw table header storage_kind mismatch for {raw_table}: expected {expected_storage_kind}, got {}",
            header.storage_kind
        )));
    }
    if header.storage_format_version != expected_storage_format_version {
        return Err(StorageError::IoError(format!(
            "raw table header storage_format_version mismatch for {raw_table}: expected {expected_storage_format_version}, got {}",
            header.storage_format_version
        )));
    }
    Ok(header)
}

fn ensure_system_raw_table_header_validated_once<H: Storage + ?Sized>(
    storage: &H,
    raw_table: &str,
    expected_storage_kind: &str,
    expected_storage_format_version: i32,
) -> Result<(), StorageError> {
    if raw_table_validated_with_storage(storage, raw_table) {
        return Ok(());
    }
    let header = storage.load_raw_table_header(raw_table)?.ok_or_else(|| {
        StorageError::IoError(format!("missing raw table header for {raw_table}"))
    })?;
    validate_system_raw_table_header(
        raw_table,
        header,
        expected_storage_kind,
        expected_storage_format_version,
    )?;
    cache_validated_raw_table_with_storage(storage, raw_table);
    Ok(())
}

fn ensure_raw_table_header<H: Storage + ?Sized>(
    storage: &mut H,
    raw_table: &str,
    expected_header: &RawTableHeader,
) -> Result<(), StorageError> {
    match storage.load_raw_table_header(raw_table)? {
        Some(existing) => {
            if existing == *expected_header {
                return Ok(());
            }
            if existing.storage_kind == expected_header.storage_kind
                && existing.storage_format_version == expected_header.storage_format_version
                && existing.logical_table_name == expected_header.logical_table_name
                && existing.schema_hash == expected_header.schema_hash
            {
                return storage.upsert_raw_table_header(raw_table, expected_header);
            }
            Err(StorageError::IoError(format!(
                "raw table header mismatch for {raw_table}"
            )))
        }
        None => {
            if !storage
                .raw_table_scan_prefix_keys(raw_table, "")?
                .is_empty()
            {
                return Err(StorageError::IoError(format!(
                    "missing raw table header for non-empty table {raw_table}"
                )));
            }
            storage.upsert_raw_table_header(raw_table, expected_header)
        }
    }
}

fn history_row_raw_table_id(table: &str, schema_hash: SchemaHash) -> RowRawTableId {
    RowRawTableId::new(RowRawTableKind::History, table, schema_hash)
}

fn visible_row_raw_table_id(table: &str, schema_hash: SchemaHash) -> RowRawTableId {
    RowRawTableId::new(RowRawTableKind::Visible, table, schema_hash)
}

fn load_user_descriptor_for_schema_hash<H: Storage + ?Sized>(
    storage: &H,
    table_name: &str,
    schema_hash: SchemaHash,
) -> Result<Arc<RowDescriptor>, StorageError> {
    load_history_user_descriptor_for_schema_hash(storage, table_name, schema_hash)?.ok_or_else(
        || {
            StorageError::IoError(format!(
                "missing catalogue descriptor for table {table_name} at schema {schema_hash}"
            ))
        },
    )
}

fn prepared_row_write_context_for_descriptor(
    table_name: &str,
    schema_hash: SchemaHash,
    user_descriptor: Arc<RowDescriptor>,
    needs_exact_locator: bool,
) -> Result<PreparedRowWriteContext, StorageError> {
    let table_context =
        prepared_row_table_context_for_descriptor(table_name, schema_hash, user_descriptor)?;
    Ok(prepared_row_write_context_from_table_context(
        table_context,
        needs_exact_locator,
    ))
}

pub(crate) fn prepared_row_table_context_for_descriptor(
    table_name: &str,
    schema_hash: SchemaHash,
    user_descriptor: Arc<RowDescriptor>,
) -> Result<Arc<PreparedRowTableContext>, StorageError> {
    Ok(Arc::new(PreparedRowTableContext {
        history_row_raw_table_id: history_row_raw_table_id(table_name, schema_hash),
        visible_row_raw_table_id: visible_row_raw_table_id(table_name, schema_hash),
        user_descriptor,
    }))
}

pub(crate) fn prepared_row_write_context_from_table_context(
    table_context: Arc<PreparedRowTableContext>,
    needs_exact_locator: bool,
) -> PreparedRowWriteContext {
    PreparedRowWriteContext {
        table_context,
        needs_exact_locator,
    }
}

/// The caller has already written `__row_locator` itself, so it — and only it —
/// knows whether that write MOVED the row between schema generations. It must
/// say so: `needs_exact_locator` is what makes the apply record an exact visible
/// locator and realign the row's family, and hardcoding it to `false` here is
/// how the local write path forked heads across generations (defect 27).
pub(crate) fn prepared_row_write_context_for_known_exact_locator(
    table_name: &str,
    schema_hash: SchemaHash,
    user_descriptor: Arc<RowDescriptor>,
) -> Result<PreparedRowWriteContext, StorageError> {
    // `false` = "this write did not land in a family other than the one the row
    // was already in". On this path the caller stamps `__row_locator` at the
    // branch's schema hash immediately before, so relative to the pointer it just
    // wrote the statement holds.
    //
    // A round-2 review argued this hardcoding hides a local-write twin of defect
    // 27. It does not, and the reason is worth recording: a generation change on
    // this path also moves the write to a NEW BRANCH, and `(row, new-branch)` has
    // no prior head anywhere to fork from. Computing the flag honestly here was
    // tried and MEASURED as dead weight — with it forced back to `false` the
    // entire suite is byte-identical, including the oracle, which drives this
    // writer on 30% of its steps. It was removed rather than shipped
    // unfalsifiable. The invariant it was meant to protect is pinned directly
    // instead, by `runtime_core::tests::cross_generation_local_write`.
    prepared_row_write_context_for_descriptor(table_name, schema_hash, user_descriptor, false)
}

pub(crate) fn prepared_row_write_context_for_schema_hash_and_descriptor<H: Storage + ?Sized>(
    storage: &H,
    table_name: &str,
    schema_hash: SchemaHash,
    row_id: ObjectId,
    user_descriptor: Arc<RowDescriptor>,
) -> Result<PreparedRowWriteContext, StorageError> {
    let needs_exact_locator = storage
        .load_row_locator(row_id)?
        .and_then(|locator| locator.origin_schema_hash)
        != Some(schema_hash);
    prepared_row_write_context_for_descriptor(
        table_name,
        schema_hash,
        user_descriptor,
        needs_exact_locator,
    )
}

pub(crate) fn prepared_row_table_context_for_schema_hash<H: Storage + ?Sized>(
    storage: &H,
    table_name: &str,
    schema_hash: SchemaHash,
) -> Result<Arc<PreparedRowTableContext>, StorageError> {
    let user_descriptor = load_user_descriptor_for_schema_hash(storage, table_name, schema_hash)?;
    prepared_row_table_context_for_descriptor(table_name, schema_hash, user_descriptor)
}

fn prepared_row_write_context_for_schema_hash<H: Storage + ?Sized>(
    storage: &H,
    table_name: &str,
    schema_hash: SchemaHash,
    row_id: ObjectId,
) -> Result<PreparedRowWriteContext, StorageError> {
    let user_descriptor = load_user_descriptor_for_schema_hash(storage, table_name, schema_hash)?;
    prepared_row_write_context_for_schema_hash_and_descriptor(
        storage,
        table_name,
        schema_hash,
        row_id,
        user_descriptor,
    )
}

fn load_user_descriptor_from_raw_table_header(
    header: &RawTableHeader,
) -> Result<Option<RowDescriptor>, StorageError> {
    let Some(raw) = header.row_descriptor_bytes.as_ref() else {
        return Ok(None);
    };
    crate::schema_manager::encoding::decode_row_descriptor_bytes(raw)
        .map(Some)
        .map_err(|err| {
            StorageError::IoError(format!(
                "decode row descriptor from raw table header: {err}"
            ))
        })
}

fn resolved_user_descriptor_for_raw_table<H: Storage + ?Sized>(
    storage: &H,
    raw_table_name: &str,
    table_name: &str,
    schema_hash: SchemaHash,
    header: &RawTableHeader,
) -> Result<Arc<RowDescriptor>, StorageError> {
    if let Some(descriptor) = cached_row_descriptor(raw_table_name) {
        return Ok(descriptor);
    }

    let descriptor = match load_user_descriptor_from_raw_table_header(header)? {
        Some(descriptor) => Arc::new(descriptor),
        None => load_user_descriptor_for_schema_hash(storage, table_name, schema_hash)?,
    };
    cache_row_descriptor(raw_table_name, descriptor.clone());
    Ok(descriptor)
}

fn row_raw_table_header(id: &RowRawTableId, user_descriptor: &RowDescriptor) -> RawTableHeader {
    RawTableHeader::row_raw_table(
        id.kind,
        id.table_name.to_string(),
        id.schema_hash,
        user_descriptor,
    )
}

fn row_raw_table_header_prefix(kind: RowRawTableKind, table: &str) -> String {
    format!("rowtable:{}:{}:", kind.as_str(), table)
}

/// The header-table prefix for a KIND, across every logical table — what the
/// startup sweep enumerates tables from.
fn row_raw_table_header_kind_prefix(kind: RowRawTableKind) -> String {
    format!("rowtable:{}:", kind.as_str())
}

/// Move a `(branch, row)` visible head out of the schema-generation family it
/// has left, without touching either locator.
///
/// A cross-generation write lands its bytes in the family the write resolved to
/// and stamps the authoritative `__visible_row_table_locator` there — but the
/// previous family's entry survives. One `(row, branch)` then has TWO visible
/// heads: the read ladder has to guess between them, and
/// `scan_visible_row_bytes_with_storage` (which iterates every family) returns
/// the row twice (defect 27, production 2026-08-16 — presence heartbeats landed
/// in the new family every 10s while every reader served the old family's copy,
/// frozen at the last server restart).
///
/// Deliberately NOT `delete_visible_region_row`: that one also clears the exact
/// visible locator, which is the very pointer the write just stamped at the
/// live family.
///
/// Returns whether a stale head was actually removed.
pub(crate) fn drop_stale_visible_row_family_entry<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
    stale_schema_hash: SchemaHash,
    live_schema_hash: Option<SchemaHash>,
) -> Result<bool, StorageError> {
    let stale_raw_table = visible_row_raw_table_id(table, stale_schema_hash);
    let key = key_codec::visible_row_raw_table_key(branch, row_id);
    let Some(stale_bytes) = storage.raw_table_get(stale_raw_table.raw_table_name(), &key)? else {
        return Ok(false);
    };

    // The index is NOT family-scoped, so the head being dropped has live index
    // entries pointing at it. Retire them before the bytes go, while they can
    // still be decoded. See `retire_index_entries_for_dropped_visible_head` for
    // what trusts the index without re-reading the row.
    // Descriptor sourced the way the READ path sources it, and degrading rather
    // than failing: a fossil generation whose catalogue entry is gone still reads,
    // so it must still be writable and deletable. Losing the retirement leaves
    // stale index entries — bad — but wedging every write to the row is worse, and
    // the sweep is what such a store is waiting for.
    match stored_bytes_descriptor_for_family(storage, table, stale_schema_hash) {
        Some(stale_descriptor) => {
            let surviving = match live_schema_hash {
                Some(live_schema_hash) => {
                    let live_raw_table = visible_row_raw_table_id(table, live_schema_hash);
                    match storage.raw_table_get(live_raw_table.raw_table_name(), &key)? {
                        Some(bytes) => {
                            stored_bytes_descriptor_for_family(storage, table, live_schema_hash)
                                .map(|descriptor| (descriptor, bytes))
                        }
                        None => None,
                    }
                }
                None => None,
            };
            retire_index_entries_for_dropped_visible_head(
                storage,
                table,
                branch,
                row_id,
                (stale_descriptor.as_ref(), &stale_bytes),
                surviving
                    .as_ref()
                    .map(|(descriptor, bytes)| (descriptor.as_ref(), bytes.as_slice())),
            )?;
        }
        None => tracing::warn!(
            table,
            branch,
            %row_id,
            stale_schema = %stale_schema_hash.short(),
            "dropping a visible head whose generation has no resolvable descriptor; \
             its index entries cannot be retired and may outlive it"
        ),
    }

    storage.raw_table_delete(stale_raw_table.raw_table_name(), &key)?;
    Ok(true)
}

/// The exact visible-row locator naming one schema generation's family.
pub(crate) fn visible_row_table_locator_for(
    table: &str,
    schema_hash: SchemaHash,
) -> ExactRowTableLocator {
    let row_raw_table_id = visible_row_raw_table_id(table, schema_hash);
    ExactRowTableLocator {
        row_raw_table: SharedString::from(row_raw_table_id.raw_table_name().to_string()),
        table_name: row_raw_table_id.table_name.clone(),
        schema_hash,
    }
}

/// Which schema-generation families PHYSICALLY hold `(branch, row)`'s visible
/// head right now.
///
/// Measured, never inferred from a locator. Both locators are exactly the things
/// that go stale in this defect family, so any repair that consults one to
/// decide what to repair can only launder the corruption. Bounded by the handful
/// of generations a store ever holds.
pub(crate) fn visible_row_families_holding<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
) -> Result<Vec<SchemaHash>, StorageError> {
    let key = key_codec::visible_row_raw_table_key(branch, row_id);
    let mut holders = Vec::new();
    for row_raw_table_id in row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)? {
        if storage
            .raw_table_get(row_raw_table_id.raw_table_name(), &key)?
            .is_some()
        {
            holders.push(row_raw_table_id.schema_hash);
        }
    }
    Ok(holders)
}

/// Every raw table that PHYSICALLY holds `(branch, row)`'s visible head, as raw
/// table names ready to delete from.
///
/// This is what a delete must use. Resolving a single locator and deleting there
/// removes one head out of however many exist, and — because the delete also
/// clears the authoritative locator — hands the read ladder straight to the
/// derived `__row_locator`, which still names a surviving fossil family. The
/// deleted row is then served again, permanently: a row with one head is not
/// "split" any more, so no repair pass ever revisits it.
pub(crate) fn visible_row_raw_tables_holding<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
) -> Result<Vec<String>, StorageError> {
    Ok(
        visible_row_families_holding(storage, table, branch, row_id)?
            .into_iter()
            .map(|schema_hash| {
                visible_row_raw_table_id(table, schema_hash)
                    .raw_table_name()
                    .to_string()
            })
            .collect(),
    )
}

/// The index entries a row's own column values justify, as `(column, value)`.
///
/// Mirrors `QueryManager::index_mutations_for_*` on the three rules that decide
/// what an indexed row owns (`query_manager/indices.rs`): `Bytea` is never
/// indexed, `Value::Null` is never indexed, and a `references` column of type
/// `Array { element: Uuid }` owns one entry per element IN ADDITION to the
/// whole-array entry.
///
/// NOT pinned against `QueryManager`'s own implementation: the equivalence is
/// asserted against a hand-written model in
/// `a_moved_head_retires_its_own_index_entries_and_only_its_own`, which covers
/// each rule but would not catch this copy DRIFTING from
/// `query_manager/indices.rs` if the rules there change. Such drift would put a
/// REBAC grant on the wrong side, so a real pin is worth writing; the two were
/// hand-diffed and agree as of 2026-08-16.
///
/// Deliberately ignores the catalogue's `indexed_columns`: an entry for a column
/// that was never indexed simply does not exist, so removing it is a no-op, and
/// not consulting the list makes this correct even when `indexed_columns` has
/// changed since the row was written — which, on a repair path for rows written
/// under a previous schema generation, is the normal case.
fn column_index_entries_for_row_bytes(
    descriptor: &RowDescriptor,
    data: &[u8],
) -> Vec<(ColumnName, Value)> {
    let mut entries = Vec::new();
    for (column_index, column) in descriptor.columns.iter().enumerate() {
        if matches!(column.column_type, ColumnType::Bytea) {
            continue;
        }
        let Ok(value) = crate::row_format::decode_column(descriptor, data, column_index) else {
            continue;
        };
        if value == Value::Null {
            continue;
        }
        let is_uuid_array_reference = column.references.is_some()
            && matches!(
                &column.column_type,
                ColumnType::Array { element } if matches!(element.as_ref(), ColumnType::Uuid)
            );
        if is_uuid_array_reference && let Value::Array(elements) = &value {
            for element in elements {
                if matches!(element, Value::Uuid(_)) {
                    entries.push((column.name, element.clone()));
                }
            }
        }
        entries.push((column.name, value));
    }
    entries
}

/// Remove one index entry, covering the signed-zero split that `index_remove`
/// does not: `Value::Double(0.0)` and `Value::Double(-0.0)` encode to DIFFERENT
/// key segments, and the lookup path probes both while the removal path probes
/// one. Dropping only one leaves a live entry pointing at bytes we just deleted.
fn remove_index_entry_including_signed_zero<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    column: &str,
    branch: &str,
    value: &Value,
    row_id: ObjectId,
) -> Result<(), StorageError> {
    storage.index_remove(table, column, branch, value, row_id)?;
    if let Value::Double(double) = value
        && *double == 0.0
    {
        let mirrored = Value::Double(if double.is_sign_negative() { 0.0 } else { -0.0 });
        storage.index_remove(table, column, branch, &mirrored, row_id)?;
    }
    Ok(())
}

/// The descriptor to decode a family's stored bytes with, sourced the way the
/// READ path sources it.
///
/// `load_user_descriptor_for_schema_hash` consults the CATALOGUE and errors when
/// the entry is absent. The read ladder does not: `resolved_row_table_from_id`
/// prefers the descriptor embedded in the raw table HEADER and only falls back to
/// the catalogue. A store whose fossil generation's catalogue entry is gone
/// therefore still reads fine, and it must still be writable and deletable — the
/// repair paths below exist to clean such stores up, so erroring there turns a
/// silent fork into a wedged write, which is a worse trade.
///
/// Returns `None` when neither source has it: the caller then skips index
/// retirement for that head with a warn rather than failing the write. Using the
/// same source as the scan also removes a real asymmetry — these bytes were
/// previously decoded with a possibly different descriptor than the one the scan
/// hands the same family.
fn stored_bytes_descriptor_for_family<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    schema_hash: SchemaHash,
) -> Option<Arc<RowDescriptor>> {
    // Header first (this is what `resolved_row_table_from_id` prefers, and it
    // carries its own catalogue fallback), then the catalogue on its own for a
    // family whose header was never registered. Only when NEITHER source has it
    // do we give up — and then by skipping the retirement, never by failing the
    // write.
    let row_raw_table_id = visible_row_raw_table_id(table, schema_hash);
    match resolved_row_table_from_id(storage, row_raw_table_id) {
        Ok(Some(resolved)) => return Some(resolved.user_descriptor),
        Ok(None) => {}
        Err(error) => tracing::warn!(
            table,
            schema = %schema_hash.short(),
            %error,
            "raw-table header descriptor unavailable for a schema-generation family; \
             falling back to the catalogue"
        ),
    }
    match load_history_user_descriptor_for_schema_hash(storage, table, schema_hash) {
        Ok(descriptor) => descriptor,
        Err(error) => {
            tracing::warn!(
                table,
                schema = %schema_hash.short(),
                %error,
                "could not resolve a descriptor for a schema-generation family; \
                 skipping its index retirement"
            );
            None
        }
    }
}

/// Retire the index entries that ONLY a dropped visible head justified.
///
/// Index raw tables are `idx:<table>:<column>:<branch>` — they carry NO schema
/// hash (`key_codec::index_raw_table`), so both generations' heads write into the
/// same index. Dropping one family's head therefore leaves that head's values
/// indexed with nothing behind them, and three consumers trust the index without
/// re-reading the row:
///
///   * a fully-covered indexed predicate drops its residual filter entirely
///     (`graph/compile.rs`, `build_remaining_predicate_from_disjuncts` returns
///     `Predicate::True`), so a stale entry is a phantom QUERY RESULT;
///   * `row_is_indexed_on_branch` / `row_is_deleted_on_branch` are pure index
///     reads (`query_manager/writes.rs`);
///   * REBAC edge traversal takes `index_lookup` directly when the referencing
///     column is an indexed scalar `Uuid`, and only the NON-indexed fallback
///     re-verifies with `declared_edge_references_target` — so a stale entry
///     there GRANTS ACCESS through an edge that no longer exists.
///
/// `surviving` is the head that remains, if any. Entries both heads justify are
/// kept; only the difference is removed. When nothing survives, the implicit
/// `_id` / `_id_deleted` entries go too — but while a head remains they must
/// stay, or the surviving row becomes invisible to every `_id` scan.
fn retire_index_entries_for_dropped_visible_head<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
    dropped: (&RowDescriptor, &[u8]),
    surviving: Option<(&RowDescriptor, &[u8])>,
) -> Result<usize, StorageError> {
    let dropped_entries = column_index_entries_for_row_bytes(dropped.0, dropped.1);
    let surviving_entries = surviving
        .map(|(descriptor, bytes)| column_index_entries_for_row_bytes(descriptor, bytes))
        .unwrap_or_default();

    let mut retired = 0usize;
    for (column, value) in &dropped_entries {
        if surviving_entries
            .iter()
            .any(|(kept_column, kept_value)| kept_column == column && kept_value == value)
        {
            continue;
        }
        remove_index_entry_including_signed_zero(
            storage,
            table,
            column.as_str(),
            branch,
            value,
            row_id,
        )?;
        retired += 1;
    }

    if surviving.is_none() {
        for implicit in ["_id", "_id_deleted"] {
            storage.index_remove(table, implicit, branch, &Value::Uuid(row_id), row_id)?;
            retired += 1;
        }
    }
    Ok(retired)
}

/// Retire the index entries owned by the EXTRA heads of a row that is about to be
/// deleted while it is still split.
///
/// A caller deleting a row builds its index removals from the one version the
/// point read served (`index_mutations_for_hard_delete_on_branch` and friends
/// take a single `old_data`). On a split row the other family's head holds
/// DIFFERENT values, and those entries survive the delete pointing at nothing.
///
/// Only the column entries: `_id` / `_id_deleted` are the caller's business —
/// a soft delete deliberately inserts `_id_deleted` — and touching them here
/// would undo that.
///
/// Costs nothing on a healthy store: a row held by one family cannot have extra
/// heads, and returns immediately.
pub(crate) fn retire_index_entries_for_extra_visible_heads<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
) -> Result<(), StorageError> {
    let holders = visible_row_families_holding(storage, table, branch, row_id)?;
    if holders.len() < 2 {
        return Ok(());
    }
    let key = key_codec::visible_row_raw_table_key(branch, row_id);
    for schema_hash in holders {
        let raw_table = visible_row_raw_table_id(table, schema_hash);
        let Some(bytes) = storage.raw_table_get(raw_table.raw_table_name(), &key)? else {
            continue;
        };
        let Some(descriptor) = stored_bytes_descriptor_for_family(storage, table, schema_hash)
        else {
            continue;
        };
        for (column, value) in column_index_entries_for_row_bytes(descriptor.as_ref(), &bytes) {
            remove_index_entry_including_signed_zero(
                storage,
                table,
                column.as_str(),
                branch,
                &value,
                row_id,
            )?;
        }
    }
    tracing::warn!(
        table,
        branch,
        %row_id,
        "deleted a row that was still split across schema generations (defect 27); \
         retired every family's index entries"
    );
    Ok(())
}

/// After ANY visible write, make the family it landed in the only one holding
/// that `(branch, row)`.
///
/// The main write path is not the only writer that puts visible bytes into a
/// caller-chosen family. Four others do, and each was measured (round-2 review)
/// to be able to fork a head:
///
///   * `patch_exact_row_batch_for_schema_hash` (`storage_trait.rs`, from
///     `runtime_core/ticks.rs`) writes into a schema hash taken from the rejected
///     batch's HISTORY locator — "where this batch's history lives" is not "where
///     this row's visible head lives";
///   * `patch_row_region_rows_by_batch_with_storage` resolves the family from
///     `__row_locator`, which is keyed by row id ALONE while heads are keyed by
///     `(branch, row)`, so a row live on two branches writes one branch's head
///     into the other branch's family;
///   * `restore_local_rejected_delete_row` and
///     `restore_permission_rejected_delete_row` both rebuild from
///     `scan_history_row_batches`, which carries NO branch filter, and write the
///     winner through `__row_locator`'s family.
///
/// All four are add-only — visible keys carry no batch id, so a write into a
/// second family ADDS a head — and none of them realigns. Rather than patch four
/// call sites and hope there is no fifth, the invariant is enforced where the
/// bytes actually land.
///
/// Free on a store that has never deployed a second schema: one header prefix
/// scan for the whole batch, then nothing.
pub(crate) fn enforce_single_visible_family_after_write<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    written: &[OwnedVisibleRowBytes],
) -> Result<(), StorageError> {
    if written.is_empty() {
        return Ok(());
    }
    // The head MOVE is only possible where more than one family exists, so it is
    // gated on that — one header prefix scan for the whole batch. The POINTER
    // alignment is not: defect 20's guarantee is that after a successful apply
    // `__row_locator` names the family the write resolved to, however many
    // families the store has. `needs_exact_locator` is precisely "this write
    // resolved somewhere other than the locator names", so it gates the pointer
    // work the same way it gated the old post-apply pass.
    let multi_generation =
        row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)?.len() > 1;

    for row in written {
        let live_schema_hash = row.row_raw_table_id.schema_hash;

        let mut moved_a_head = false;
        if multi_generation {
            for stale_schema_hash in
                visible_row_families_holding(storage, table, &row.branch, row.row_id)?
                    .into_iter()
                    .filter(|schema_hash| *schema_hash != live_schema_hash)
            {
                if drop_stale_visible_row_family_entry(
                    storage,
                    table,
                    &row.branch,
                    row.row_id,
                    stale_schema_hash,
                    Some(live_schema_hash),
                )? {
                    moved_a_head = true;
                    tracing::warn!(
                        table,
                        branch = %row.branch,
                        row_id = %row.row_id,
                        stale_schema = %stale_schema_hash.short(),
                        live_schema = %live_schema_hash.short(),
                        "a visible write landed in a different generation than the row's \
                         head; moved the head instead of forking it (defect 27)"
                    );
                }
            }
        }

        // Align when the write resolved somewhere other than the locator named,
        // and ALSO whenever a head actually moved: a write can resolve to exactly
        // the family `__row_locator` names (so `needs_exact_locator` is false)
        // and still have moved the head out of a DIFFERENT family, leaving the
        // authoritative locator — which the read ladder consults first — pointing
        // at the fossil it just deleted.
        if !row.needs_exact_locator && !moved_a_head {
            continue;
        }
        // Both pointers follow the head, or the read ladder's derived step and
        // `old_content_schema_hash` aim at a family that does not hold the row.
        let stored_locator = storage.load_row_locator(row.row_id)?;
        if stored_locator
            .as_ref()
            .and_then(|locator| locator.origin_schema_hash)
            != Some(live_schema_hash)
        {
            storage.put_row_locator(
                row.row_id,
                Some(&RowLocator {
                    table: stored_locator
                        .as_ref()
                        .map(|locator| locator.table.clone())
                        .unwrap_or_else(|| SharedString::from(table.to_string())),
                    origin_schema_hash: Some(live_schema_hash),
                }),
            )?;
        }
        // The read ladder consults this one FIRST and nothing else in the engine
        // ever corrects it, so a stale one would freeze reads on a fossil.
        let live = visible_row_table_locator_for(table, live_schema_hash);
        if storage
            .load_visible_row_table_locator(&row.branch, row.row_id)?
            .as_ref()
            != Some(&live)
        {
            storage.put_visible_row_table_locator(&row.branch, row.row_id, Some(&live))?;
        }
    }
    Ok(())
}

/// What a [`repair_split_visible_row_families`] pass touched. All zeroes means
/// the store was already healthy.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VisibleFamilySplitRepair {
    /// `(branch, row)` pairs that held a visible head in more than one
    /// schema-generation family.
    pub split_rows: usize,
    /// Stale heads deleted.
    pub dropped_heads: usize,
    /// Rows whose surviving head was rewritten from history.
    pub rebuilt_rows: usize,
    /// Split rows left alone because they had no history to rebuild from.
    pub unresolved_rows: usize,
    /// WHICH rows were repaired, as `(table, branch, row_id)`. Four integers
    /// tell nobody which row to go and check afterwards — and after a repair
    /// that also retired index entries, the identities are the only way to
    /// audit what the store looked like before.
    pub repaired: Vec<(String, String, ObjectId)>,
    /// Rows left in a split state, same shape. These are the ones still capable
    /// of serving a stale read.
    pub unresolved: Vec<(String, String, ObjectId)>,
    /// Tables whose repair FAILED, with the error. The sweep continues past a
    /// failing table rather than abandoning every table after it — a partial
    /// silent sweep leaves exactly the half-repaired state the defect needs.
    pub failed_tables: Vec<(String, String)>,
}

impl VisibleFamilySplitRepair {
    pub fn is_noop(&self) -> bool {
        *self == Self::default()
    }

    fn merge(&mut self, other: Self) {
        self.split_rows += other.split_rows;
        self.dropped_heads += other.dropped_heads;
        self.rebuilt_rows += other.rebuilt_rows;
        self.unresolved_rows += other.unresolved_rows;
        self.repaired.extend(other.repaired);
        self.unresolved.extend(other.unresolved);
        self.failed_tables.extend(other.failed_tables);
    }
}

/// Heal a store that already carries defect 27's damage: one `(row, branch)`
/// with a visible head in two schema-generation families, reads frozen on the
/// stale one.
///
/// The write path stops NEW splits (`row_histories::mutations`), but it cannot
/// undo the ones a running deployment already made — nothing on the write path
/// ever visits a row again unless a write arrives for it, and the stale head
/// outlives every restart. This is the sweep that does.
///
/// The winner is recomputed from `scan_history_row_batches`, which is
/// sibling-complete across families (`storage_trait.rs`,
/// `resolved_row_tables_for_table`), through the same
/// `VisibleRowEntry::rebuild_with_descriptor` the write path uses — defect 21's
/// rule, not a timestamp comparison between the two heads. Comparing the heads
/// is exactly the reasoning that made the stale one look defensible.
///
/// Idempotent, and bounded: on a store with fewer than two families for the
/// table it costs one header prefix scan and stops.
pub fn repair_split_visible_row_families<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
) -> Result<VisibleFamilySplitRepair, StorageError> {
    let mut report = VisibleFamilySplitRepair::default();
    let families = row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)?;
    if families.len() < 2 {
        return Ok(report);
    }
    // Already swept for exactly this generation-set: nothing new can have split
    // since, because a split needs a family that did not exist then. Two point
    // reads instead of a full key scan of every family.
    let generation_set = visible_family_generation_set(storage, table)?;
    if visible_family_sweep_is_current(storage, table, &generation_set)? {
        return Ok(report);
    }

    let mut heads: BTreeMap<(String, ObjectId), Vec<SchemaHash>> = BTreeMap::new();
    for family in &families {
        // Keys only: this runs at every startup once a store has more than one
        // generation, and the values are megabytes we would only drop.
        for key in storage.raw_table_scan_prefix_keys(family.raw_table_name(), "")? {
            let (branch, row_id) = key_codec::decode_visible_row_raw_table_key(&key)?;
            heads
                .entry((branch, row_id))
                .or_default()
                .push(family.schema_hash);
        }
    }

    for ((branch, row_id), family_hashes) in heads {
        if family_hashes.len() < 2 {
            continue;
        }
        report.split_rows += 1;

        let history_rows = storage
            .scan_history_row_batches(table, row_id)?
            .into_iter()
            .filter(|row| row.branch.as_str() == branch)
            .collect::<Vec<_>>();
        let Some(newest) = history_rows
            .iter()
            .max_by_key(|row| (row.updated_at, row.batch_id()))
            .cloned()
        else {
            // Two heads and no history to arbitrate between them. Dropping one
            // would be a coin flip, so say so and leave the row alone.
            tracing::warn!(
                table,
                branch,
                %row_id,
                families = family_hashes.len(),
                "split visible head has no history to rebuild from; left untouched"
            );
            report.unresolved_rows += 1;
            report
                .unresolved
                .push((table.to_string(), branch.clone(), row_id));
            continue;
        };

        // The generation a batch belongs to is read off the family its HISTORY
        // bytes physically live in — never off `__row_locator`, which is the
        // poisoned input this sweep exists to repair, and never off
        // `resolve_history_row_write_context`, whose first ladder step is that
        // same locator (`required_history_user_descriptor_and_schema_hash_for_row`).
        // History is safe to ask because its keys carry the batch id, so a batch
        // exists in exactly one family and a wrong guess misses instead of
        // answering wrongly.
        let family_of = |storage: &H, batch_id: BatchId| -> Result<_, StorageError> {
            load_history_row_batch_row_bytes_with_storage(storage, table, &branch, row_id, batch_id)
        };

        let Some(newest_bytes) = family_of(storage, newest.batch_id())? else {
            tracing::warn!(
                table,
                branch,
                %row_id,
                "split visible head's newest batch is not in any history family; left untouched"
            );
            report.unresolved_rows += 1;
            report
                .unresolved
                .push((table.to_string(), branch.clone(), row_id));
            continue;
        };
        let entry = crate::row_histories::VisibleRowEntry::rebuild_with_descriptor(
            newest_bytes.user_descriptor.as_ref(),
            &history_rows,
        )
        .map_err(|err| StorageError::IoError(format!("rebuild split visible head: {err}")))?;

        // The surviving head belongs in the WINNER's own generation — the winner
        // is not always the newest row, and not always the newest generation.
        let live_schema_hash = match entry.as_ref() {
            Some(entry) => match family_of(storage, entry.current_row.batch_id())? {
                Some(winner_bytes) => winner_bytes.row_raw_table_id.schema_hash,
                None => newest_bytes.row_raw_table_id.schema_hash,
            },
            None => newest_bytes.row_raw_table_id.schema_hash,
        };

        for stale_schema_hash in family_hashes
            .iter()
            .copied()
            .filter(|schema_hash| *schema_hash != live_schema_hash)
        {
            // Retires this head's index entries too — the index is not
            // family-scoped, so a dropped head leaves live entries behind that
            // queries, `_id` probes and REBAC edge traversal all trust.
            if drop_stale_visible_row_family_entry(
                storage,
                table,
                branch.as_str(),
                row_id,
                stale_schema_hash,
                Some(live_schema_hash),
            )? {
                report.dropped_heads += 1;
            }
        }

        // Stamp the derived locator BEFORE rewriting the head, so the write
        // below sees an already-aligned store and the two locators cannot
        // disagree afterwards.
        storage.put_row_locator(
            row_id,
            Some(&RowLocator {
                table: SharedString::from(table.to_string()),
                origin_schema_hash: Some(live_schema_hash),
            }),
        )?;

        let live = visible_row_raw_table_id(table, live_schema_hash);
        match entry {
            Some(entry) => {
                storage.upsert_visible_region_rows(table, std::slice::from_ref(&entry))?;
                storage.put_visible_row_table_locator(
                    &branch,
                    row_id,
                    Some(&ExactRowTableLocator {
                        row_raw_table: SharedString::from(live.raw_table_name().to_string()),
                        table_name: live.table_name.clone(),
                        schema_hash: live_schema_hash,
                    }),
                )?;
                report.rebuilt_rows += 1;
                // Recorded only now that the pass has actually arbitrated and
                // rewritten the head. Pushing on entry made every unresolved row
                // appear in BOTH lists, which is precisely the case an auditor
                // reads this report to find.
                report
                    .repaired
                    .push((table.to_string(), branch.clone(), row_id));
            }
            None => {
                // Nothing in the history is visible any more (every version
                // rejected or superseded): the row has no head at all, which is
                // the state `delete_visible_region_row` leaves behind. Nothing
                // survives, so the implicit `_id` / `_id_deleted` entries go too.
                if drop_stale_visible_row_family_entry(
                    storage,
                    table,
                    branch.as_str(),
                    row_id,
                    live_schema_hash,
                    None,
                )? {
                    report.dropped_heads += 1;
                }
                storage.put_visible_row_table_locator(&branch, row_id, None)?;
            }
        }
    }

    // Only claim cleanliness the pass actually established.
    if report.unresolved_rows == 0 {
        mark_visible_family_sweep_current(storage, table, &generation_set)?;
    }

    if !report.is_noop() {
        tracing::warn!(
            table,
            families = families.len(),
            split_rows = report.split_rows,
            dropped_heads = report.dropped_heads,
            rebuilt_rows = report.rebuilt_rows,
            unresolved_rows = report.unresolved_rows,
            repaired = ?report.repaired,
            unresolved = ?report.unresolved,
            "repaired visible heads split across schema generations (defect 27)"
        );
    }
    Ok(report)
}

/// The generation-set a table currently has, as the marker value: every visible
/// family's schema hash, sorted, joined. A new deployment registers a new family
/// and changes this string, which is what re-arms the sweep.
fn visible_family_generation_set<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
) -> Result<String, StorageError> {
    let mut hashes: Vec<String> =
        row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)?
            .into_iter()
            .map(|row_raw_table_id| row_raw_table_id.schema_hash.to_string())
            .collect();
    hashes.sort();
    Ok(hashes.join(","))
}

/// Has this exact generation-set already been swept clean?
///
/// Without this the sweep is a permanent boot tax: it scans the KEYS of every
/// family of every multi-generation table on EVERY start, which measures
/// (release, rocksdb, healthy two-generation store) 32 ms at 50k rows, 144 ms at
/// 200k and 807 ms at 1M — roughly 800 ns/row, plus a transient map linear in
/// row count, paid forever to find nothing. The damage it repairs is created by
/// a deployment, so a marker naming the generation-set it was last completed for
/// is the honest bound: it re-arms exactly when a new family appears.
fn visible_family_sweep_is_current<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    generation_set: &str,
) -> Result<bool, StorageError> {
    match storage.raw_table_get(VISIBLE_FAMILY_SWEEP_TABLE, table)? {
        Some(bytes) => {
            ensure_system_raw_table_header_validated_once(
                storage,
                VISIBLE_FAMILY_SWEEP_TABLE,
                STORAGE_KIND_VISIBLE_FAMILY_SWEEP,
                VISIBLE_FAMILY_SWEEP_FORMAT_V1,
            )?;
            Ok(std::str::from_utf8(&bytes)
                .map_err(|err| StorageError::IoError(format!("sweep marker utf8: {err}")))?
                == generation_set)
        }
        None => Ok(false),
    }
}

/// Record that `table` is swept clean for this generation-set.
///
/// Only called after a pass that completed WITHOUT leaving anything unresolved
/// and without failing: a pass that could not arbitrate some row must stay
/// re-armed, or the marker would promise a cleanliness the store does not have.
fn mark_visible_family_sweep_current<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    generation_set: &str,
) -> Result<(), StorageError> {
    ensure_raw_table_header(
        storage,
        VISIBLE_FAMILY_SWEEP_TABLE,
        &RawTableHeader::system(
            STORAGE_KIND_VISIBLE_FAMILY_SWEEP,
            VISIBLE_FAMILY_SWEEP_FORMAT_V1,
        ),
    )?;
    storage.raw_table_put(VISIBLE_FAMILY_SWEEP_TABLE, table, generation_set.as_bytes())
}

/// Every schema generation this store holds a VISIBLE region for, sorted.
///
/// The universe a runtime actually reads is derived from the schema manager's
/// `live_schemas` (`SchemaContext::all_branch_names`,
/// schema_manager/context.rs:212) — never from the store. When the store holds
/// a generation the manager never learned, that generation's rows are indexed,
/// persisted, and unreachable at every durability tier. This is the store half
/// of that comparison; `RuntimeCore::unknown_store_schema_generations` is the
/// difference.
///
/// One header-prefix scan, the same one the split-family sweep already opens
/// with — no per-row work.
pub fn visible_schema_generations<H: Storage + ?Sized>(
    storage: &H,
) -> Result<Vec<SchemaHash>, StorageError> {
    let mut generations = Vec::new();
    for (raw_table_name, _) in storage.raw_table_scan_prefix(
        RAW_TABLE_HEADER_TABLE,
        &row_raw_table_header_kind_prefix(RowRawTableKind::Visible),
    )? {
        let row_raw_table_id = RowRawTableId::parse_raw_table_name(&raw_table_name)?;
        if !generations.contains(&row_raw_table_id.schema_hash) {
            generations.push(row_raw_table_id.schema_hash);
        }
    }
    generations.sort_by_key(|schema_hash| schema_hash.0);
    Ok(generations)
}

/// [`repair_split_visible_row_families`] over every logical table the store
/// holds a visible region for. This is the startup entry point.
pub fn repair_all_split_visible_row_families<H: Storage + ?Sized>(
    storage: &mut H,
) -> Result<VisibleFamilySplitRepair, StorageError> {
    let mut tables = BTreeSet::new();
    for (raw_table_name, _) in storage.raw_table_scan_prefix(
        RAW_TABLE_HEADER_TABLE,
        &row_raw_table_header_kind_prefix(RowRawTableKind::Visible),
    )? {
        let row_raw_table_id = RowRawTableId::parse_raw_table_name(&raw_table_name)?;
        tables.insert(row_raw_table_id.table_name.to_string());
    }

    let mut report = VisibleFamilySplitRepair::default();
    for table in tables {
        // Isolated per table. A `?` here would abandon every LATER table on the
        // first failure, and the caller logs this at warn — so one bad table
        // would silently leave the rest of the store unswept, which is exactly
        // the half-repaired state a resurrecting delete needs.
        match repair_split_visible_row_families(storage, &table) {
            Ok(table_report) => report.merge(table_report),
            Err(error) => {
                tracing::error!(
                    table,
                    %error,
                    "visible-family split repair FAILED for this table; continuing with the rest"
                );
                report.failed_tables.push((table, error.to_string()));
            }
        }
    }
    Ok(report)
}

fn validate_row_raw_table_header(
    id: &RowRawTableId,
    header: RawTableHeader,
) -> Result<RawTableHeader, StorageError> {
    validate_row_raw_table_header_fields(
        id.kind,
        id.table_name.as_str(),
        id.schema_hash,
        id.raw_table_name(),
        header,
    )
}

fn validate_row_raw_table_header_fields(
    kind: RowRawTableKind,
    table_name: &str,
    schema_hash: SchemaHash,
    raw_table_name: &str,
    header: RawTableHeader,
) -> Result<RawTableHeader, StorageError> {
    validate_raw_table_header_storage_format(raw_table_name, &header)?;
    if header.storage_kind.as_str() != kind.storage_kind() {
        return Err(StorageError::IoError(format!(
            "raw table header storage_kind '{}' did not match expected row raw table kind '{}'",
            header.storage_kind,
            kind.storage_kind()
        )));
    }
    if header.logical_table_name.as_ref().map(|name| name.as_str()) != Some(table_name) {
        return Err(StorageError::IoError(format!(
            "row raw table header logical_table_name mismatch for {}",
            raw_table_name
        )));
    }
    if header.schema_hash != Some(schema_hash) {
        return Err(StorageError::IoError(format!(
            "row raw table header schema_hash mismatch for {}",
            raw_table_name
        )));
    }
    Ok(header)
}

fn ensure_row_raw_table_header_validated_once<H: Storage + ?Sized>(
    storage: &H,
    row_raw_table_id: &RowRawTableId,
) -> Result<(), StorageError> {
    if raw_table_validated_with_storage(storage, row_raw_table_id.raw_table_name()) {
        return Ok(());
    }
    let header = storage
        .load_raw_table_header(row_raw_table_id.raw_table_name())?
        .ok_or_else(|| {
            StorageError::IoError(format!(
                "missing raw table header for {}",
                row_raw_table_id.raw_table_name()
            ))
        })?;
    validate_row_raw_table_header(row_raw_table_id, header)?;
    cache_validated_raw_table_with_storage(storage, row_raw_table_id.raw_table_name());
    Ok(())
}

pub(crate) fn load_row_raw_table_header_with_storage<H: Storage + ?Sized>(
    storage: &H,
    id: &RowRawTableId,
) -> Result<Option<RawTableHeader>, StorageError> {
    let header = storage
        .load_raw_table_header(id.raw_table_name())?
        .map(|header| validate_row_raw_table_header(id, header))
        .transpose()?;
    if header.is_some() {
        cache_validated_raw_table_with_storage(storage, id.raw_table_name());
    }
    Ok(header)
}

pub(crate) fn scan_row_raw_table_headers_with_storage<H: Storage + ?Sized>(
    storage: &H,
) -> Result<Vec<(RowRawTableId, RawTableHeader)>, StorageError> {
    let mut rows = Vec::new();
    for (raw_table_name, header) in storage.scan_raw_table_headers()? {
        if !raw_table_name.starts_with("rowtable:") {
            continue;
        }
        let row_raw_table_id = RowRawTableId::parse_raw_table_name(&raw_table_name)?;
        cache_validated_raw_table_with_storage(storage, &raw_table_name);
        rows.push((
            row_raw_table_id.clone(),
            validate_row_raw_table_header(&row_raw_table_id, header)?,
        ));
    }
    rows.sort_by_key(|(row_raw_table_id, _)| row_raw_table_id.raw_table_name().to_string());
    Ok(rows)
}

fn row_raw_table_ids_for_table<H: Storage + ?Sized>(
    storage: &H,
    kind: RowRawTableKind,
    table: &str,
) -> Result<Vec<RowRawTableId>, StorageError> {
    let prefix = row_raw_table_header_prefix(kind, table);
    let mut ids = Vec::new();
    for (raw_table_name, bytes) in storage.raw_table_scan_prefix(RAW_TABLE_HEADER_TABLE, &prefix)? {
        let row_raw_table_id = RowRawTableId::parse_raw_table_name(&raw_table_name)?;
        let header = decode_raw_table_header(&bytes)?;
        let header = validate_row_raw_table_header(&row_raw_table_id, header)?;
        cache_raw_table_header_with_storage(storage, &raw_table_name, header);
        cache_validated_raw_table_with_storage(storage, &raw_table_name);
        ids.push(row_raw_table_id);
    }
    ids.sort_by_key(|row_raw_table_id| row_raw_table_id.raw_table_name().to_string());
    Ok(ids)
}

fn resolved_row_table_from_header<H: Storage + ?Sized>(
    storage: &H,
    row_raw_table_id: RowRawTableId,
    header: RawTableHeader,
) -> Result<ResolvedRowTable, StorageError> {
    let table_name = header
        .logical_table_name
        .as_ref()
        .expect("validated row raw table header must have logical_table_name")
        .as_str();
    let schema_hash = header
        .schema_hash
        .expect("validated row raw table header must have schema_hash");
    let user_descriptor = resolved_user_descriptor_for_raw_table(
        storage,
        row_raw_table_id.raw_table_name(),
        table_name,
        schema_hash,
        &header,
    )?;
    let row_codecs = flat_row_codecs(user_descriptor.as_ref());
    Ok(ResolvedRowTable {
        row_raw_table: row_raw_table_id.raw_table_name().to_string(),
        user_descriptor,
        row_codecs,
    })
}

fn resolved_row_table_from_id<H: Storage + ?Sized>(
    storage: &H,
    row_raw_table_id: RowRawTableId,
) -> Result<Option<ResolvedRowTable>, StorageError> {
    let Some(header) = load_row_raw_table_header_with_storage(storage, &row_raw_table_id)? else {
        return Ok(None);
    };
    resolved_row_table_from_header(storage, row_raw_table_id, header).map(Some)
}

fn resolved_row_table_from_locator<H: Storage + ?Sized>(
    storage: &H,
    locator: &ExactRowTableLocator,
) -> Result<Option<ResolvedRowTable>, StorageError> {
    let row_raw_table_id = RowRawTableId {
        kind: RowRawTableId::parse_raw_table_name(locator.row_raw_table.as_str())?.kind,
        table_name: locator.table_name.clone(),
        schema_hash: locator.schema_hash,
        raw_table_name: locator.row_raw_table.clone(),
    };
    ensure_row_raw_table_header_validated_once(storage, &row_raw_table_id)?;
    let user_descriptor =
        if let Some(descriptor) = cached_row_descriptor(locator.row_raw_table.as_str()) {
            descriptor
        } else {
            let descriptor = load_user_descriptor_for_schema_hash(
                storage,
                locator.table_name.as_str(),
                locator.schema_hash,
            )?;
            cache_row_descriptor(locator.row_raw_table.as_str(), descriptor.clone());
            descriptor
        };
    let row_codecs = flat_row_codecs(user_descriptor.as_ref());
    Ok(Some(ResolvedRowTable {
        row_raw_table: locator.row_raw_table.to_string(),
        user_descriptor,
        row_codecs,
    }))
}

fn resolved_row_tables_for_table<H: Storage + ?Sized>(
    storage: &H,
    kind: RowRawTableKind,
    table: &str,
) -> Result<Vec<ResolvedRowTable>, StorageError> {
    let prefix = row_raw_table_header_prefix(kind, table);
    let mut tables = Vec::new();
    for (raw_table_name, bytes) in storage.raw_table_scan_prefix(RAW_TABLE_HEADER_TABLE, &prefix)? {
        let row_raw_table_id = RowRawTableId::parse_raw_table_name(&raw_table_name)?;
        let schema_hash = row_raw_table_id.schema_hash;
        let header = decode_raw_table_header(&bytes)?;
        let header = validate_row_raw_table_header_fields(
            kind,
            table,
            schema_hash,
            &raw_table_name,
            header,
        )?;
        cache_raw_table_header_with_storage(storage, &raw_table_name, header.clone());
        cache_validated_raw_table_with_storage(storage, &raw_table_name);
        tables.push(resolved_row_table_from_header(
            storage,
            row_raw_table_id,
            header,
        )?);
    }
    tables.sort_by(|left, right| left.row_raw_table.cmp(&right.row_raw_table));
    Ok(tables)
}

fn common_case_exact_row_table_locator<H: Storage + ?Sized>(
    storage: &H,
    row_id: ObjectId,
) -> Result<Option<RowLocator>, StorageError> {
    let Some(locator) = storage.load_row_locator(row_id)? else {
        return Ok(None);
    };
    if locator.origin_schema_hash.is_none() {
        return Ok(None);
    }
    Ok(Some(locator))
}

fn common_case_exact_history_row_table_locator<H: Storage + ?Sized>(
    storage: &H,
    row_id: ObjectId,
) -> Result<Option<ExactRowTableLocator>, StorageError> {
    let Some(locator) = common_case_exact_row_table_locator(storage, row_id)? else {
        return Ok(None);
    };
    let Some(schema_hash) = locator.origin_schema_hash else {
        return Ok(None);
    };
    let row_raw_table_id = history_row_raw_table_id(locator.table.as_str(), schema_hash);
    Ok(Some(ExactRowTableLocator {
        row_raw_table: row_raw_table_id.raw_table_name.clone(),
        table_name: row_raw_table_id.table_name.clone(),
        schema_hash,
    }))
}

fn common_case_exact_visible_row_table_locator<H: Storage + ?Sized>(
    storage: &H,
    row_id: ObjectId,
) -> Result<Option<ExactRowTableLocator>, StorageError> {
    let Some(locator) = common_case_exact_row_table_locator(storage, row_id)? else {
        return Ok(None);
    };
    let Some(schema_hash) = locator.origin_schema_hash else {
        return Ok(None);
    };
    let row_raw_table_id = visible_row_raw_table_id(locator.table.as_str(), schema_hash);
    Ok(Some(ExactRowTableLocator {
        row_raw_table: row_raw_table_id.raw_table_name.clone(),
        table_name: row_raw_table_id.table_name.clone(),
        schema_hash,
    }))
}

fn sealed_batch_submission_storage_descriptor_with_branch_ords() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("batch_id", ColumnType::BatchId),
        ColumnDescriptor::new("mode", ColumnType::Text),
        ColumnDescriptor::new("target_branch_ord", ColumnType::Integer),
        ColumnDescriptor::new("batch_digest", ColumnType::Bytea),
        ColumnDescriptor::new(
            "members",
            ColumnType::Array {
                element: Box::new(ColumnType::Row {
                    columns: Box::new(RowDescriptor::new(vec![
                        ColumnDescriptor::new("object_id", ColumnType::Bytea),
                        ColumnDescriptor::new("row_digest", ColumnType::Bytea),
                    ])),
                }),
            },
        ),
        ColumnDescriptor::new(
            "captured_frontier",
            ColumnType::Array {
                element: Box::new(ColumnType::Row {
                    columns: Box::new(RowDescriptor::new(vec![
                        ColumnDescriptor::new("object_id", ColumnType::Bytea),
                        ColumnDescriptor::new("branch_ord", ColumnType::Integer),
                        ColumnDescriptor::new("batch_id", ColumnType::BatchId),
                    ])),
                }),
            },
        ),
    ])
}

fn local_batch_record_storage_descriptor_with_branch_ords() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("batch_id", ColumnType::BatchId),
        ColumnDescriptor::new("mode", ColumnType::Text),
        ColumnDescriptor::new("sealed", ColumnType::Boolean),
        ColumnDescriptor::new(
            "members",
            ColumnType::Array {
                element: Box::new(ColumnType::Row {
                    columns: Box::new(RowDescriptor::new(vec![
                        ColumnDescriptor::new("object_id", ColumnType::Bytea),
                        ColumnDescriptor::new("table_name", ColumnType::Text),
                        ColumnDescriptor::new("branch_ord", ColumnType::Integer),
                        ColumnDescriptor::new("schema_hash", ColumnType::Bytea),
                        ColumnDescriptor::new("row_digest", ColumnType::Bytea),
                    ])),
                }),
            },
        ),
    ])
}

fn local_batch_row_index_storage_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("batch_id", ColumnType::BatchId),
        ColumnDescriptor::new(
            "members",
            ColumnType::Array {
                element: Box::new(ColumnType::Row {
                    columns: Box::new(RowDescriptor::new(vec![
                        ColumnDescriptor::new("object_id", ColumnType::Bytea),
                        ColumnDescriptor::new("table_name", ColumnType::Text),
                        ColumnDescriptor::new("branch_name", ColumnType::Text),
                        ColumnDescriptor::new("schema_hash", ColumnType::Bytea),
                        ColumnDescriptor::new("row_digest", ColumnType::Bytea),
                    ])),
                }),
            },
        ),
    ])
}

fn encode_batch_mode(mode: crate::batch_fate::BatchMode) -> &'static str {
    match mode {
        crate::batch_fate::BatchMode::Direct => "direct",
        crate::batch_fate::BatchMode::Transactional => "transactional",
    }
}

fn decode_batch_mode(raw: &str) -> Result<crate::batch_fate::BatchMode, StorageError> {
    match raw {
        "direct" => Ok(crate::batch_fate::BatchMode::Direct),
        "transactional" => Ok(crate::batch_fate::BatchMode::Transactional),
        other => Err(StorageError::IoError(format!(
            "unknown batch mode '{other}'"
        ))),
    }
}

fn load_history_user_descriptor_for_schema_hash<H: Storage + ?Sized>(
    storage: &H,
    table_hint: &str,
    schema_hash: SchemaHash,
) -> Result<Option<Arc<RowDescriptor>>, StorageError> {
    if let Some(descriptor) =
        cached_catalogue_user_descriptor_with_storage(storage, table_hint, schema_hash)
    {
        return Ok(Some(descriptor));
    }

    let Some(entry) = storage.load_catalogue_entry(schema_hash.to_object_id())? else {
        return Ok(None);
    };
    let Some(descriptor) = crate::schema_manager::encoding::decode_table_descriptor_from_schema(
        &entry.content,
        table_hint,
    )
    .map_err(|err| StorageError::IoError(format!("decode schema for row history: {err}")))?
    else {
        return Ok(None);
    };
    let descriptor = Arc::new(descriptor);
    cache_catalogue_user_descriptor_with_storage(
        storage,
        table_hint,
        schema_hash,
        descriptor.clone(),
    );
    Ok(Some(descriptor))
}

fn catalogue_schema_hash(entry: &CatalogueEntry) -> Result<SchemaHash, StorageError> {
    if let Some(schema_hash) = entry
        .metadata
        .get(MetadataKey::SchemaHash.as_str())
        .and_then(|raw| SchemaHash::from_hex(raw))
    {
        return Ok(schema_hash);
    }

    let schema = crate::schema_manager::encoding::decode_schema(&entry.content)
        .map_err(|err| StorageError::IoError(format!("decode schema for row history: {err}")))?;
    Ok(SchemaHash::compute(&schema))
}

fn schema_hashes_matching_branch<H: Storage + ?Sized>(
    storage: &H,
    branch: &str,
) -> Result<Vec<SchemaHash>, StorageError> {
    let cache_key = (storage.storage_cache_namespace(), branch.to_string());
    if let Some(cached) = branch_schema_hash_cache()
        .lock()
        .expect("branch schema hash cache poisoned")
        .get(&cache_key)
        .cloned()
    {
        return Ok(cached);
    }

    let Some(composed) = ComposedBranchName::parse(&BranchName::new(branch)) else {
        let hashes = Vec::new();
        branch_schema_hash_cache()
            .lock()
            .expect("branch schema hash cache poisoned")
            .insert(cache_key, hashes.clone());
        return Ok(hashes);
    };

    let mut hashes = storage
        .scan_catalogue_entries()?
        .into_iter()
        .filter_map(|entry| catalogue_schema_hash(&entry).ok())
        .filter(|schema_hash| schema_hash.short() == composed.schema_hash.short())
        .collect::<Vec<_>>();
    hashes.sort_by_key(|schema_hash| schema_hash.to_string());
    hashes.dedup();
    branch_schema_hash_cache()
        .lock()
        .expect("branch schema hash cache poisoned")
        .insert(cache_key, hashes.clone());
    Ok(hashes)
}

fn catalogue_row_descriptors_for_table<H: Storage + ?Sized>(
    storage: &H,
    table_hint: &str,
) -> Result<Vec<(SchemaHash, crate::query_manager::types::RowDescriptor)>, StorageError> {
    let cache_key = (storage.storage_cache_namespace(), table_hint.to_string());
    if let Some(cached) = table_catalogue_descriptor_cache()
        .lock()
        .expect("table catalogue descriptor cache poisoned")
        .get(&cache_key)
        .cloned()
    {
        return Ok(cached);
    }

    let table_name = crate::query_manager::types::TableName::new(table_hint);
    let mut candidates = Vec::new();

    for entry in storage.scan_catalogue_entries()? {
        let schema =
            crate::schema_manager::encoding::decode_schema(&entry.content).map_err(|err| {
                StorageError::IoError(format!("decode schema for row history: {err}"))
            })?;
        let Some(table_schema) = schema.get(&table_name) else {
            continue;
        };
        candidates.push((SchemaHash::compute(&schema), table_schema.columns.clone()));
    }

    candidates.sort_by_key(|(schema_hash, _)| schema_hash.to_string());
    candidates.dedup_by(|(left_hash, _), (right_hash, _)| left_hash == right_hash);
    table_catalogue_descriptor_cache()
        .lock()
        .expect("table catalogue descriptor cache poisoned")
        .insert(cache_key, candidates.clone());
    Ok(candidates)
}

fn required_history_user_descriptor_and_schema_hash_for_row<H: Storage + ?Sized>(
    storage: &H,
    table_hint: &str,
    row: &StoredRowBatch,
) -> Result<(SchemaHash, Arc<RowDescriptor>), StorageError> {
    let row_data_matches = |descriptor: &crate::query_manager::types::RowDescriptor| {
        row.data.is_empty() || crate::row_format::decode_row(descriptor, &row.data).is_ok()
    };
    let try_schema_hash = |candidates: &mut Vec<(SchemaHash, Arc<RowDescriptor>)>,
                           table_name: &str,
                           schema_hash: SchemaHash|
     -> Result<(), StorageError> {
        if let Some(descriptor) =
            load_history_user_descriptor_for_schema_hash(storage, table_name, schema_hash)?
            && row_data_matches(descriptor.as_ref())
        {
            candidates.push((schema_hash, descriptor));
        }
        Ok(())
    };

    let row_locator = storage.load_row_locator(row.row_id)?;
    let mut locator_candidates = Vec::new();
    if let Some(row_locator) = row_locator.as_ref()
        && let Some(origin_schema_hash) = row_locator.origin_schema_hash
    {
        try_schema_hash(
            &mut locator_candidates,
            row_locator.table.as_str(),
            origin_schema_hash,
        )?;
        if row_locator.table.as_str() != table_hint {
            try_schema_hash(&mut locator_candidates, table_hint, origin_schema_hash)?;
        }
    }
    locator_candidates.sort_by_key(|(schema_hash, _)| schema_hash.to_string());
    locator_candidates.dedup_by(|(left_hash, _), (right_hash, _)| left_hash == right_hash);
    if let [(schema_hash, descriptor)] = locator_candidates.as_slice() {
        return Ok((*schema_hash, descriptor.clone()));
    }

    let mut branch_candidates = Vec::new();
    for schema_hash in schema_hashes_matching_branch(storage, row.branch.as_str())? {
        try_schema_hash(&mut branch_candidates, table_hint, schema_hash)?;
    }
    branch_candidates.sort_by_key(|(schema_hash, _)| schema_hash.to_string());
    branch_candidates.dedup_by(|(left_hash, _), (right_hash, _)| left_hash == right_hash);
    if let [(schema_hash, descriptor)] = branch_candidates.as_slice() {
        return Ok((*schema_hash, descriptor.clone()));
    }

    let mut table_candidates = catalogue_row_descriptors_for_table(storage, table_hint)?
        .into_iter()
        .filter(|(_, descriptor)| row_data_matches(descriptor))
        .map(|(schema_hash, descriptor)| (schema_hash, Arc::new(descriptor)))
        .collect::<Vec<_>>();
    table_candidates.sort_by_key(|(schema_hash, _)| schema_hash.to_string());
    table_candidates.dedup_by(|(left_hash, _), (right_hash, _)| left_hash == right_hash);
    if let [(schema_hash, descriptor)] = table_candidates.as_slice() {
        return Ok((*schema_hash, descriptor.clone()));
    }

    let row_locator_debug = row_locator.or(storage.load_row_locator(row.row_id)?);
    let candidate_hashes = branch_candidates
        .iter()
        .chain(locator_candidates.iter())
        .chain(table_candidates.iter())
        .map(|(schema_hash, _)| schema_hash.to_string())
        .collect::<BTreeSet<_>>();

    if candidate_hashes.is_empty() {
        return Err(StorageError::IoError(format!(
            "missing catalogue-backed row descriptor for history row {} in table {} on branch {} (row_locator={row_locator_debug:?})",
            row.row_id, table_hint, row.branch,
        )));
    }

    Err(StorageError::IoError(format!(
        "ambiguous exact schema hash for history row {} in table {} on branch {}: {:?}",
        row.row_id, table_hint, row.branch, candidate_hashes
    )))
}

pub(crate) fn resolve_history_row_write_context<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    row: &StoredRowBatch,
) -> Result<PreparedRowWriteContext, StorageError> {
    let (schema_hash, user_descriptor) =
        required_history_user_descriptor_and_schema_hash_for_row(storage, table, row)?;
    prepared_row_write_context_for_schema_hash_and_descriptor(
        storage,
        table,
        schema_hash,
        row.row_id,
        user_descriptor,
    )
}

pub(crate) fn encode_history_row_bytes_with_context(
    context: &PreparedRowWriteContext,
    row: &StoredRowBatch,
) -> Result<OwnedHistoryRowBytes, StorageError> {
    let bytes =
        crate::row_histories::encode_flat_history_row(context.user_descriptor().as_ref(), row)
            .map_err(|err| StorageError::IoError(format!("encode flat history row: {err}")))?;

    Ok(OwnedHistoryRowBytes {
        row_raw_table: context
            .history_row_raw_table_id()
            .raw_table_name()
            .to_string(),
        row_raw_table_id: context.history_row_raw_table_id().clone(),
        user_descriptor: context.user_descriptor().clone(),
        branch: row.branch.to_string(),
        row_id: row.row_id,
        batch_id: row.batch_id(),
        needs_exact_locator: context.needs_exact_locator,
        bytes,
    })
}

pub(crate) fn encode_visible_row_bytes_with_context(
    context: &PreparedRowWriteContext,
    entry: &VisibleRowEntry,
) -> Result<OwnedVisibleRowBytes, StorageError> {
    let bytes = crate::row_histories::encode_flat_visible_row_entry(
        context.user_descriptor().as_ref(),
        entry,
    )
    .map_err(|err| StorageError::IoError(format!("encode flat visible row: {err}")))?;

    Ok(OwnedVisibleRowBytes {
        row_raw_table: context
            .visible_row_raw_table_id()
            .raw_table_name()
            .to_string(),
        row_raw_table_id: context.visible_row_raw_table_id().clone(),
        user_descriptor: context.user_descriptor().clone(),
        branch: entry.current_row.branch.to_string(),
        row_id: entry.current_row.row_id,
        needs_exact_locator: context.needs_exact_locator,
        bytes,
    })
}

/// The write context for the family that PHYSICALLY holds an EXISTING batch's
/// history bytes, falling back to the normal resolution when none holds it.
///
/// Re-writing a batch has to put it back where it already is. Resolving the
/// family from `__row_locator` instead writes a SECOND copy into another
/// generation — and `scan_history_row_batches` is sibling-complete, so the row
/// then carries two versions of ONE batch with different states, and the visible
/// resolution picks whichever of them is visible. A batch patched to `Rejected`
/// goes on being served from its other copy. Defect 27's history-side twin,
/// found by the differential oracle at
/// `sync_manager::tests::cross_generation_oracle`.
///
/// Safe to ask: history keys carry the batch id, so a batch lives in exactly one
/// family and this point read either finds it or proves the batch is new.
pub(crate) fn existing_history_row_write_context<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
    batch_id: BatchId,
    row: &StoredRowBatch,
) -> Result<PreparedRowWriteContext, StorageError> {
    let Some(existing) =
        load_history_row_batch_row_bytes_with_storage(storage, table, branch, row_id, batch_id)?
    else {
        return resolve_history_row_write_context(storage, table, row);
    };
    // Only the FAMILY is taken from the existing bytes; the descriptor still
    // comes from the catalogue, exactly as `resolve_history_row_write_context`
    // would build it. Nothing about how a row encodes changes when the family
    // was never in doubt — this only stops a rewrite from landing in a family
    // the batch is not already in.
    let schema_hash = existing.row_raw_table_id.schema_hash;
    prepared_row_write_context_for_schema_hash_and_descriptor(
        storage,
        table,
        schema_hash,
        row_id,
        load_user_descriptor_for_schema_hash(storage, table, schema_hash)?,
    )
}

pub(crate) fn encode_history_row_bytes_for_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    rows: &[StoredRowBatch],
) -> Result<Vec<OwnedHistoryRowBytes>, StorageError> {
    rows.iter()
        .map(|row| {
            let context = resolve_history_row_write_context(storage, table, row)?;
            encode_history_row_bytes_with_context(&context, row)
        })
        .collect()
}

pub(crate) fn encode_visible_row_bytes_for_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    entries: &[VisibleRowEntry],
) -> Result<Vec<OwnedVisibleRowBytes>, StorageError> {
    entries
        .iter()
        .map(|entry| {
            let context = resolve_history_row_write_context(storage, table, &entry.current_row)?;
            encode_visible_row_bytes_with_context(&context, entry)
        })
        .collect()
}

fn decode_history_row_bytes_in_table(
    resolved: &ResolvedRowTable,
    row_id: ObjectId,
    branch: &str,
    batch_id: BatchId,
    bytes: &[u8],
) -> Result<StoredRowBatch, StorageError> {
    // Every history-decoding path funnels through here, so this is the one place
    // that can say what a settle actually spent on accumulated history.
    crate::query_manager::settle_cost::bump(&crate::query_manager::settle_cost::HISTORY_ENTRIES);
    crate::query_manager::settle_cost::add(
        &crate::query_manager::settle_cost::HISTORY_BYTES,
        bytes.len() as u64,
    );
    decode_flat_history_row_with_codecs(
        resolved.row_codecs.as_ref(),
        row_id,
        branch,
        batch_id,
        bytes,
    )
    .map_err(|err| StorageError::IoError(format!("decode flat history row: {err}")))
}

fn decode_visible_row_entry_bytes_in_table(
    resolved: &ResolvedRowTable,
    row_id: ObjectId,
    branch: &str,
    bytes: &[u8],
) -> Result<VisibleRowEntry, StorageError> {
    decode_flat_visible_row_entry_with_codecs(resolved.row_codecs.as_ref(), row_id, branch, bytes)
        .map_err(|err| StorageError::IoError(format!("decode flat visible row: {err}")))
}

pub(super) fn scan_history_row_bytes_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    scan: HistoryScan,
) -> Result<Vec<OwnedHistoryRowBytes>, StorageError> {
    let row_raw_table_ids = row_raw_table_ids_for_table(storage, RowRawTableKind::History, table)?;
    let prefix = match scan {
        HistoryScan::Branch | HistoryScan::AsOf { .. } => {
            key_codec::history_row_raw_table_prefix(None)
        }
        HistoryScan::Row { row_id } => key_codec::history_row_raw_table_prefix(Some(row_id)),
    };
    let mut rows = Vec::new();
    for row_raw_table_id in row_raw_table_ids {
        let resolved = resolved_row_table_from_id(storage, row_raw_table_id.clone())?
            .expect("row raw table id from header scan must resolve");
        let row_raw_table = resolved.row_raw_table.clone();
        for (key, bytes) in storage.raw_table_scan_prefix(&row_raw_table, &prefix)? {
            let (row_id, branch, batch_id) = key_codec::decode_history_row_raw_table_key(&key)?;
            rows.push(OwnedHistoryRowBytes {
                row_raw_table_id: row_raw_table_id.clone(),
                row_raw_table: row_raw_table.clone(),
                user_descriptor: resolved.user_descriptor.clone(),
                branch,
                row_id,
                batch_id,
                needs_exact_locator: true,
                bytes,
            });
        }
    }
    Ok(rows)
}

pub(super) fn scan_visible_row_bytes_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    branch: &str,
) -> Result<Vec<OwnedVisibleRowBytes>, StorageError> {
    let row_raw_table_ids = row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)?;
    let families_scanned = row_raw_table_ids.len();
    let prefix = key_codec::visible_row_raw_table_prefix(branch);
    let mut rows = Vec::new();
    for row_raw_table_id in row_raw_table_ids {
        let resolved = resolved_row_table_from_id(storage, row_raw_table_id.clone())?
            .expect("row raw table id from header scan must resolve");
        let row_raw_table = resolved.row_raw_table.clone();
        for (key, bytes) in storage.raw_table_scan_prefix(&row_raw_table, &prefix)? {
            let (decoded_branch, row_id) = key_codec::decode_visible_row_raw_table_key(&key)?;
            if decoded_branch != branch {
                return Err(StorageError::IoError(format!(
                    "visible row raw table key '{key}' decoded unexpected branch '{decoded_branch}'"
                )));
            }
            rows.push(OwnedVisibleRowBytes {
                row_raw_table_id: row_raw_table_id.clone(),
                row_raw_table: row_raw_table.clone(),
                user_descriptor: resolved.user_descriptor.clone(),
                branch: decoded_branch,
                row_id,
                needs_exact_locator: true,
                bytes,
            });
        }
    }

    // A split row physically exists in two families, so the loop above emits it
    // TWICE, with two different contents — a phantom duplicate carrying stale
    // bytes, in every scan-driven query. The point-read ladder does nothing for
    // this surface: it is reached only by `load_visible_region_row_bytes`.
    //
    // Collapse duplicates onto whatever the point read would have served, so the
    // two surfaces cannot disagree. Only paid when a duplicate really exists;
    // a store with one family per table cannot produce one, and short-circuits.
    if families_scanned > 1 {
        retain_point_read_winner_per_row(
            storage,
            table,
            branch,
            &mut rows,
            |row| row.row_id,
            |row| row.row_raw_table.as_str(),
        )?;
    }
    Ok(rows)
}

/// Collapse `(branch, row)` duplicates produced by a store that is still split,
/// keeping the version the point-read ladder serves.
///
/// Generic over the scan surface: `scan_visible_row_bytes_with_storage` and
/// `Storage::scan_visible_region` are two separate iterations over the same
/// families, and BOTH emit a split row twice. Sharing the collapse is what keeps
/// them from disagreeing with each other as well as with the point read.
pub(crate) fn retain_point_read_winner_per_row<H: Storage + ?Sized, T>(
    storage: &H,
    table: &str,
    branch: &str,
    rows: &mut Vec<T>,
    row_id_of: impl Fn(&T) -> ObjectId,
    raw_table_of: impl Fn(&T) -> &str,
) -> Result<(), StorageError> {
    // A two-generation store with no duplicates left — which is every store after
    // the sweep, production included — must not pay for an ordered set over every
    // scanned row on every scan. `HashSet`, and the second set stays unallocated
    // until a duplicate is actually seen.
    let mut seen: HashSet<ObjectId> = HashSet::with_capacity(rows.len());
    let mut duplicated: HashSet<ObjectId> = HashSet::new();
    for row in rows.iter() {
        if !seen.insert(row_id_of(row)) {
            duplicated.insert(row_id_of(row));
        }
    }
    if duplicated.is_empty() {
        return Ok(());
    }

    // For each duplicated row, ask the point read which family wins, and keep
    // only that one. `load_visible_region_row_bytes_with_storage` IS the ladder,
    // so the scan cannot drift from the point read by construction.
    let mut winning_raw_table: HashMap<ObjectId, Option<String>> = HashMap::new();
    for row_id in &duplicated {
        let winner = load_visible_region_row_bytes_with_storage(storage, table, branch, *row_id)?
            .map(|row| row.row_raw_table);
        tracing::warn!(
            table,
            branch,
            %row_id,
            winner = winner.as_deref().unwrap_or("<none>"),
            "visible scan saw one row in more than one schema-generation family \
             (defect 27); collapsing onto the point read's answer"
        );
        winning_raw_table.insert(*row_id, winner);
    }

    let mut kept: HashSet<ObjectId> = HashSet::new();
    rows.retain(|row| {
        let row_id = row_id_of(row);
        if !duplicated.contains(&row_id) {
            return true;
        }
        match winning_raw_table.get(&row_id) {
            // The ladder picked a family: keep exactly that copy.
            Some(Some(winner)) => raw_table_of(row) == winner.as_str(),
            // The ladder served nothing at all (no locator resolves, no probe
            // hits). Keep the first copy rather than dropping the row from the
            // scan entirely — a phantom duplicate is a bug, a vanished row is
            // worse.
            _ => kept.insert(row_id),
        }
    });
    Ok(())
}

pub(super) fn load_history_row_batch_row_bytes_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
    batch_id: BatchId,
) -> Result<Option<OwnedHistoryRowBytes>, StorageError> {
    let key = key_codec::history_row_raw_table_key(row_id, branch, batch_id);
    if let Some(locator) = common_case_exact_history_row_table_locator(storage, row_id)? {
        let row_raw_table = locator.row_raw_table.to_string();
        if let Some(bytes) = storage.raw_table_get(&row_raw_table, &key)? {
            let resolved = resolved_row_table_from_locator(storage, &locator)?
                .expect("common-case locator-resolved history row table must exist");
            return Ok(Some(OwnedHistoryRowBytes {
                row_raw_table_id: RowRawTableId {
                    kind: RowRawTableKind::History,
                    table_name: locator.table_name.clone(),
                    schema_hash: locator.schema_hash,
                    raw_table_name: locator.row_raw_table.clone(),
                },
                row_raw_table,
                user_descriptor: resolved.user_descriptor,
                branch: branch.to_string(),
                row_id,
                batch_id,
                needs_exact_locator: false,
                bytes,
            }));
        }
    }

    if let Some(locator) = storage.load_history_row_batch_table_locator(branch, row_id, batch_id)? {
        let resolved = resolved_row_table_from_locator(storage, &locator)?
            .expect("locator-resolved row table must exist");
        let row_raw_table = locator.row_raw_table.to_string();
        if let Some(bytes) = storage.raw_table_get(&row_raw_table, &key)? {
            return Ok(Some(OwnedHistoryRowBytes {
                row_raw_table_id: RowRawTableId {
                    kind: RowRawTableKind::History,
                    table_name: locator.table_name.clone(),
                    schema_hash: locator.schema_hash,
                    raw_table_name: locator.row_raw_table.clone(),
                },
                row_raw_table,
                user_descriptor: resolved.user_descriptor,
                branch: branch.to_string(),
                row_id,
                batch_id,
                needs_exact_locator: true,
                bytes,
            }));
        }
    }

    // Last resort — the history twin of the visible sibling probe below: a
    // poisoned store's history batches sit in a raw table the locators do not
    // name (defect 20), and this point read backs parent checks, tier
    // patches, replay dedup and the USING-policy old-content load — misses
    // here surface as "no old content" rejections.
    for row_raw_table_id in row_raw_table_ids_for_table(storage, RowRawTableKind::History, table)? {
        let Some(resolved) = resolved_row_table_from_id(storage, row_raw_table_id.clone())? else {
            continue;
        };
        let row_raw_table = row_raw_table_id.raw_table_name().to_string();
        if let Some(bytes) = storage.raw_table_get(&row_raw_table, &key)? {
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::LOCATOR_LADDER_RECOVERIES,
            );
            // v18 item 5: the history walk has its own counter — no D2 hook heals it, so a
            // count that never converges is the signal, and mixing it into the visible one
            // would hide exactly that.
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::HISTORY_LOCATOR_LADDER_RECOVERIES,
            );
            // `debug`, not `info`: this fires on every READ of a split row, and the
            // ladder never writes the exact locator back, so it repeats for the life of
            // the row. Production logged 8751 of these in eight minutes from one table —
            // formatting and I/O on the settle path, and enough noise to bury the lines
            // that matter. The count is on the settle line as `locator_ladder_recoveries`,
            // which is where an operator should read it; `LOCATOR_LADDER_RECOVERIES` above
            // is bumped either way, so nothing is lost by lowering this.
            tracing::debug!(
                table,
                branch,
                %row_id,
                raw_table = %row_raw_table,
                "history batch recovered from a sibling raw table the locator did not name"
            );
            return Ok(Some(OwnedHistoryRowBytes {
                row_raw_table_id: row_raw_table_id.clone(),
                row_raw_table,
                user_descriptor: resolved.user_descriptor,
                branch: branch.to_string(),
                row_id,
                batch_id,
                needs_exact_locator: true,
                bytes,
            }));
        }
    }

    Ok(None)
}

pub(super) fn load_visible_region_row_bytes_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
) -> Result<Option<OwnedVisibleRowBytes>, StorageError> {
    let key = key_codec::visible_row_raw_table_key(branch, row_id);

    // The AUTHORITATIVE per-(branch, row) locator first. Every write that puts a
    // visible head into a family other than the one `__row_locator` names stamps
    // this pointer at the family it wrote (`storage_trait.rs`,
    // `apply_encoded_row_mutation`), so it is the only locator kept current by
    // the write path itself.
    //
    // It used to come SECOND, behind the locator DERIVED from
    // `__row_locator.origin_schema_hash` below — which the inbound sync path
    // never rewrote, so after a schema deployment it named the generation the
    // row was born in forever. The derived step hit that fossil and returned,
    // and the authoritative step never ran (defect 27, production 2026-08-16:
    // `"row locator aligned to the schema hash the write resolved"` occurrences
    // = 0, defect-20 sibling-scan hits = 0 — step one always hit).
    //
    // Cost: this table is only written on a cross-generation write, so the
    // common case pays one extra point read that MISSES. Measured on sqlite
    // (release, 2000 rows, 20k reads, none of them carrying an exact locator —
    // `sync_manager::tests::cross_generation_visible_split::visible_read_ladder_cost`):
    // 393 ns/op against 4230 ns/op for the whole visible read, i.e. 9.3%. Paid on
    // every visible read; correctness first, and the alternative — leaving a
    // pointer nothing keeps current in front of one that is — is what this
    // defect is.
    if let Some(locator) = storage.load_visible_row_table_locator(branch, row_id)? {
        let resolved = resolved_row_table_from_locator(storage, &locator)?
            .expect("locator-resolved row table must exist");
        let row_raw_table = locator.row_raw_table.to_string();
        if let Some(bytes) = storage.raw_table_get(&row_raw_table, &key)? {
            return Ok(Some(OwnedVisibleRowBytes {
                row_raw_table_id: RowRawTableId {
                    kind: RowRawTableKind::Visible,
                    table_name: locator.table_name.clone(),
                    schema_hash: locator.schema_hash,
                    raw_table_name: locator.row_raw_table.clone(),
                },
                row_raw_table,
                user_descriptor: resolved.user_descriptor,
                branch: branch.to_string(),
                row_id,
                needs_exact_locator: true,
                bytes,
            }));
        }
    }

    // The common case: no exact locator was ever needed for this row, so the
    // family derived from `__row_locator` is the only one it has ever lived in.
    if let Some(locator) = common_case_exact_visible_row_table_locator(storage, row_id)? {
        let row_raw_table = locator.row_raw_table.to_string();
        if let Some(bytes) = storage.raw_table_get(&row_raw_table, &key)? {
            let resolved = resolved_row_table_from_locator(storage, &locator)?
                .expect("common-case locator-resolved visible row table must exist");
            return Ok(Some(OwnedVisibleRowBytes {
                row_raw_table_id: RowRawTableId {
                    kind: RowRawTableKind::Visible,
                    table_name: locator.table_name.clone(),
                    schema_hash: locator.schema_hash,
                    raw_table_name: locator.row_raw_table.clone(),
                },
                row_raw_table,
                user_descriptor: resolved.user_descriptor,
                branch: branch.to_string(),
                row_id,
                needs_exact_locator: false,
                bytes,
            }));
        }
    }

    // Last resort: the locators are missing or name a raw table that does not
    // hold the row — probe every registered visible raw table of this logical
    // table for `<branch>:<row>`. A row delivered before the catalogue knew
    // its origin schema was placed via the any-decoding-descriptor fallback
    // (the CURRENT schema's raw table) while the row locator kept the
    // server-stamped origin hash, so the locator-directed reads above miss it
    // forever (defect 20, production 2026-08-15: the account query settled
    // empty over a delivered, confirmed row — welcome, chats, kick). Bounded
    // by the handful of schema generations a store ever holds.
    for row_raw_table_id in row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)? {
        let Some(resolved) = resolved_row_table_from_id(storage, row_raw_table_id.clone())? else {
            continue;
        };
        let row_raw_table = row_raw_table_id.raw_table_name().to_string();
        if let Some(bytes) = storage.raw_table_get(&row_raw_table, &key)? {
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::LOCATOR_LADDER_RECOVERIES,
            );
            // `debug`, not `info`: this fires on every READ of a split row until the
            // recovery below persists the exact locator — after v18 item 5 that is once
            // per row, not once per read. Before the hook, production logged 8751 of these
            // in eight minutes from one table — formatting and I/O on the settle path, and
            // enough noise to bury the lines that matter. The count is on the settle line
            // as `locator_ladder_recoveries`, which is where an operator should read it;
            // `LOCATOR_LADDER_RECOVERIES` above is bumped either way. The history arm
            // keeps the old wording: nothing heals it, so it does repeat for the row's life.
            tracing::debug!(
                table,
                branch,
                %row_id,
                raw_table = %row_raw_table,
                "visible row recovered from a sibling raw table the locator did not name"
            );
            // v18 item 5 (D2): the walk found the family; persist the exact locator so the
            // next read of this row goes straight to it. The write path stamps one only on
            // a cross-generation WRITE, and a row delivered before the catalogue knew its
            // origin never gets that write — so without this the walk repeats for the life
            // of the row (1,200 in five minutes against one `users` row, 2026-08-18). The
            // store counts its own walks and its own persist failures; a failure never
            // fails the read — a `LostWrites` is the barrier's to report, anything else is
            // worth one warning here.
            storage.note_visible_locator_recovery();
            let recovered = ExactRowTableLocator {
                row_raw_table: row_raw_table.clone().into(),
                table_name: row_raw_table_id.table_name.clone(),
                schema_hash: row_raw_table_id.schema_hash,
            };
            if let Err(error) =
                storage.record_visible_row_table_locator_recovery(branch, row_id, &recovered)
                && !matches!(error, StorageError::LostWrites { .. })
            {
                tracing::warn!(
                    table,
                    branch,
                    %row_id,
                    %error,
                    "recovered visible locator could not be persisted"
                );
            }
            return Ok(Some(OwnedVisibleRowBytes {
                row_raw_table_id: row_raw_table_id.clone(),
                row_raw_table,
                user_descriptor: resolved.user_descriptor,
                branch: branch.to_string(),
                row_id,
                needs_exact_locator: true,
                bytes,
            }));
        }
    }

    Ok(None)
}

fn scan_history_row_batches_for_schema_hash<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    schema_hash: SchemaHash,
    row_id: ObjectId,
) -> Result<Vec<StoredRowBatch>, StorageError> {
    let row_raw_table_id = history_row_raw_table_id(table, schema_hash);
    let Some(resolved) = resolved_row_table_from_id(storage, row_raw_table_id.clone())? else {
        return Ok(Vec::new());
    };

    let prefix = key_codec::history_row_raw_table_prefix(Some(row_id));
    let mut rows = Vec::new();
    for (key, bytes) in storage.raw_table_scan_prefix(row_raw_table_id.raw_table_name(), &prefix)? {
        let (decoded_row_id, branch, batch_id) = key_codec::decode_history_row_raw_table_key(&key)?;
        rows.push(decode_history_row_bytes_in_table(
            &resolved,
            decoded_row_id,
            branch.as_str(),
            batch_id,
            &bytes,
        )?);
    }
    rows.sort_by_key(|row| (row.branch.clone(), row.updated_at, row.batch_id()));
    Ok(rows)
}

pub(super) fn scan_visible_region_row_batch_branches_with_storage<H: Storage + ?Sized>(
    storage: &H,
    table: &str,
    row_id: ObjectId,
) -> Result<Vec<String>, StorageError> {
    let row_raw_table_ids = row_raw_table_ids_for_table(storage, RowRawTableKind::History, table)?;
    let prefix = key_codec::history_row_raw_table_prefix(Some(row_id));
    let mut branches = Vec::new();
    for row_raw_table_id in row_raw_table_ids {
        for key in storage.raw_table_scan_prefix_keys(row_raw_table_id.raw_table_name(), &prefix)? {
            let (_decoded_row_id, branch, _batch_id) =
                key_codec::decode_history_row_raw_table_key(&key)?;
            branches.push(branch);
        }
    }
    branches.sort();
    branches.dedup();
    Ok(branches)
}

pub(crate) fn patch_row_region_rows_by_batch_with_storage<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    batch_id: crate::row_histories::BatchId,
    state: Option<RowState>,
    confirmed_tier: Option<DurabilityTier>,
) -> Result<(), StorageError> {
    let history_rows = {
        let resolved_tables =
            resolved_row_tables_for_table(storage, RowRawTableKind::History, table)?;
        let mut rows = Vec::new();
        for resolved in &resolved_tables {
            for (key, bytes) in storage.raw_table_scan_prefix(
                &resolved.row_raw_table,
                &key_codec::history_row_raw_table_prefix(None),
            )? {
                let (row_id, branch, batch_id) = key_codec::decode_history_row_raw_table_key(&key)?;
                rows.push(decode_history_row_bytes_in_table(
                    resolved,
                    row_id,
                    branch.as_str(),
                    batch_id,
                    &bytes,
                )?);
            }
        }
        rows
    };

    let mut patched_history = Vec::new();
    let mut history_by_visible_row = HashMap::<(String, ObjectId), Vec<StoredRowBatch>>::new();
    let mut affected_visible_rows = HashSet::<(String, ObjectId)>::new();

    for mut row in history_rows {
        if row.batch_id == batch_id {
            if let Some(state) = state {
                row.state = state;
            }
            row.confirmed_tier = match (row.confirmed_tier, confirmed_tier) {
                (Some(existing), Some(incoming)) => Some(existing.max(incoming)),
                (Some(existing), None) => Some(existing),
                (None, incoming) => incoming,
            };
            affected_visible_rows.insert((row.branch.to_string(), row.row_id));
            patched_history.push(row.clone());
        }

        history_by_visible_row
            .entry((row.branch.to_string(), row.row_id))
            .or_default()
            .push(row);
    }

    if !patched_history.is_empty() {
        storage.append_history_region_rows(table, &patched_history)?;
    }

    let mut rebuilt_visible_entries = Vec::new();
    let mut rows_without_visible_head = Vec::new();
    for (branch, row_id) in &affected_visible_rows {
        let Some(existing_entry) = storage.load_visible_region_entry(table, branch, *row_id)?
        else {
            continue;
        };

        let history_rows = history_by_visible_row
            .remove(&(branch.clone(), *row_id))
            .unwrap_or_default();
        let context = if let Some(current_row) = history_rows.first() {
            resolve_history_row_write_context(storage, table, current_row)?
        } else {
            continue;
        };
        if let Some(entry) = VisibleRowEntry::rebuild_with_descriptor(
            context.user_descriptor().as_ref(),
            &history_rows,
        )
        .map_err(|err| StorageError::IoError(format!("rebuild visible entry: {err}")))?
        {
            rebuilt_visible_entries.push(entry);
            continue;
        }

        let mut current = existing_entry.current_row.clone();
        if current.batch_id == batch_id {
            if let Some(state) = state {
                current.state = state;
            }
            current.confirmed_tier = match (current.confirmed_tier, confirmed_tier) {
                (Some(existing), Some(incoming)) => Some(existing.max(incoming)),
                (Some(existing), None) => Some(existing),
                (None, incoming) => incoming,
            };
        }

        if current.state.is_visible() {
            if let Some(entry) = VisibleRowEntry::rebuild_with_descriptor(
                context.user_descriptor().as_ref(),
                &history_rows,
            )
            .map_err(|err| StorageError::IoError(format!("rebuild visible entry: {err}")))?
            {
                rebuilt_visible_entries.push(entry);
            }
        } else {
            rows_without_visible_head.push((branch.clone(), *row_id));
        }
    }

    if !rebuilt_visible_entries.is_empty() {
        storage.upsert_visible_region_rows(table, &rebuilt_visible_entries)?;
    }
    for (branch, row_id) in rows_without_visible_head {
        storage.delete_visible_region_row(table, &branch, row_id)?;
    }

    Ok(())
}

/// Full-scan patch variant, deliberately WITHOUT the fast paths of its hot
/// sibling `row_histories::patch_row_batch_state`: its only production
/// reachability is `MemoryStorage::patch_exact_row_batch_for_schema_hash` ←
/// `runtime_core/ticks.rs` local-batch rejection cleanup, and `→ Rejected`
/// transitions are always-full-path by design even on the hot sibling
/// (removing a batch from the visible set can expose a hidden ancestor as
/// the new winner). If a visible-preserving caller ever appears here, port
/// the fast-path routing from `patch_row_batch_state`.
pub(crate) fn patch_exact_row_batch_with_storage<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    branch: &str,
    row_id: ObjectId,
    batch_id: crate::row_histories::BatchId,
    state: Option<RowState>,
    confirmed_tier: Option<DurabilityTier>,
) -> Result<bool, StorageError> {
    let Some(mut row) = storage.load_history_row_batch(table, branch, row_id, batch_id)? else {
        return Ok(false);
    };

    if let Some(state) = state {
        row.state = state;
    }
    row.confirmed_tier = match (row.confirmed_tier, confirmed_tier) {
        (Some(existing), Some(incoming)) => Some(existing.max(incoming)),
        (Some(existing), None) => Some(existing),
        (None, incoming) => incoming,
    };
    let history_rows = storage.scan_history_row_batches(table, row_id)?;
    let mut patched_history = history_rows.clone();
    if let Some(existing) = patched_history
        .iter_mut()
        .find(|candidate| candidate.branch == branch && candidate.batch_id() == batch_id)
    {
        *existing = row.clone();
    }
    let context = resolve_history_row_write_context(storage, table, &row)?;

    let visible_entries = VisibleRowEntry::rebuild_with_descriptor(
        context.user_descriptor().as_ref(),
        &patched_history,
    )
    .map_err(|err| StorageError::IoError(format!("rebuild visible entry: {err}")))?
    .into_iter()
    .collect::<Vec<_>>();

    storage.apply_row_mutation(table, std::slice::from_ref(&row), &visible_entries, &[])?;
    if visible_entries.is_empty() {
        storage.delete_visible_region_row(table, branch, row_id)?;
    }

    Ok(true)
}

fn branch_matches_transaction_family(
    branch_name: BranchName,
    target_branch_name: BranchName,
) -> bool {
    match (
        ComposedBranchName::parse(&branch_name),
        ComposedBranchName::parse(&target_branch_name),
    ) {
        (Some(branch), Some(target)) => {
            branch.matches_env_and_branch(&target.env, &target.user_branch)
        }
        _ => branch_name == target_branch_name,
    }
}

fn decode_storage_batch_id_value(value: &Value, context: &str) -> Result<BatchId, StorageError> {
    match value {
        Value::BatchId(bytes) => Ok(BatchId(*bytes)),
        Value::Bytea(bytes) => {
            let bytes: [u8; 16] = bytes.as_slice().try_into().map_err(|_| {
                StorageError::IoError(format!("{context}: expected 16 bytes, got {}", bytes.len()))
            })?;
            Ok(BatchId(bytes))
        }
        other => Err(StorageError::IoError(format!(
            "{context}: expected batch id bytes, got {other:?}"
        ))),
    }
}

fn encode_sealed_batch_submission_with_branch_ords<H: Storage + ?Sized>(
    storage: &mut H,
    submission: &SealedBatchSubmission,
) -> Result<Vec<u8>, StorageError> {
    let target_branch_ord = storage.resolve_or_alloc_branch_ord(submission.target_branch_name)?;
    let member_values = submission
        .members
        .iter()
        .map(|member| {
            Ok(Value::Row {
                id: None,
                values: vec![
                    Value::Bytea(member.object_id.uuid().as_bytes().to_vec()),
                    Value::Bytea(member.row_digest.0.to_vec()),
                ],
            })
        })
        .collect::<Result<Vec<_>, StorageError>>()?;
    let frontier_values = submission
        .captured_frontier
        .iter()
        .map(|member| {
            // Legacy compatibility payload: persisted for old sealed batch rows
            // but ignored by transaction validation. Drop this column in the
            // next compat-breaking storage format refactor.
            let branch_ord = storage.resolve_or_alloc_branch_ord(member.branch_name)?;
            Ok(Value::Row {
                id: None,
                values: vec![
                    Value::Bytea(member.object_id.uuid().as_bytes().to_vec()),
                    Value::Integer(branch_ord),
                    Value::BatchId(member.batch_id.0),
                ],
            })
        })
        .collect::<Result<Vec<_>, StorageError>>()?;
    encode_row(
        &sealed_batch_submission_storage_descriptor_with_branch_ords(),
        &[
            Value::BatchId(*submission.batch_id.as_bytes()),
            Value::Text(encode_batch_mode(submission.mode).to_string()),
            Value::Integer(target_branch_ord),
            Value::Bytea(submission.batch_digest.0.to_vec()),
            Value::Array(member_values),
            Value::Array(frontier_values),
        ],
    )
    .map_err(|err| StorageError::IoError(format!("encode sealed batch submission: {err}")))
}

fn decode_sealed_batch_submission_with_branch_ords<H: Storage + ?Sized>(
    storage: &H,
    bytes: &[u8],
) -> Result<SealedBatchSubmission, StorageError> {
    let values = decode_row(
        &sealed_batch_submission_storage_descriptor_with_branch_ords(),
        bytes,
    )
    .map_err(|err| StorageError::IoError(format!("decode sealed batch submission: {err}")))?;
    let [
        batch_id,
        mode,
        target_branch_ord,
        batch_digest,
        members,
        captured_frontier,
    ] = values.as_slice()
    else {
        return Err(StorageError::IoError(
            "unexpected sealed batch submission shape".to_string(),
        ));
    };

    let batch_id = decode_storage_batch_id_value(batch_id, "decode sealed batch id")?;
    let mode = match mode {
        Value::Text(raw) => decode_batch_mode(raw)?,
        other => {
            return Err(StorageError::IoError(format!(
                "expected sealed batch mode text, got {other:?}"
            )));
        }
    };
    let target_branch_ord = match target_branch_ord {
        Value::Integer(raw) => *raw,
        other => {
            return Err(StorageError::IoError(format!(
                "expected target branch ord integer, got {other:?}"
            )));
        }
    };
    let target_branch_name = storage
        .load_branch_name_by_ord(target_branch_ord)?
        .ok_or_else(|| {
            StorageError::IoError(format!(
                "missing branch name for target branch ord {target_branch_ord}"
            ))
        })?;
    let batch_digest = match batch_digest {
        Value::Bytea(bytes) => Digest32(bytes.as_slice().try_into().map_err(|_| {
            StorageError::IoError(format!(
                "expected sealed batch digest to be 32 bytes, got {}",
                bytes.len()
            ))
        })?),
        other => {
            return Err(StorageError::IoError(format!(
                "expected sealed batch digest bytes, got {other:?}"
            )));
        }
    };

    let members = match members {
        Value::Array(elements) => elements
            .iter()
            .map(|element| match element {
                Value::Row { values, .. } => {
                    let [object_id, row_digest] = values.as_slice() else {
                        return Err(StorageError::IoError(
                            "expected sealed batch member row to have two values".to_string(),
                        ));
                    };
                    let object_id = match object_id {
                        Value::Bytea(bytes) => uuid::Uuid::from_slice(bytes)
                            .map(ObjectId::from_uuid)
                            .map_err(|err| {
                                StorageError::IoError(format!(
                                    "decode sealed batch member object id uuid: {err}"
                                ))
                            })?,
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected sealed batch member object id bytes, got {other:?}"
                            )));
                        }
                    };
                    let row_digest = match row_digest {
                        Value::Bytea(bytes) => Digest32(bytes.as_slice().try_into().map_err(
                            |_| {
                                StorageError::IoError(format!(
                                    "expected sealed batch member row digest to be 32 bytes, got {}",
                                    bytes.len()
                                ))
                            },
                        )?),
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected sealed batch member row digest bytes, got {other:?}"
                            )));
                        }
                    };
                    Ok(crate::batch_fate::SealedBatchMember {
                        object_id,
                        row_digest,
                    })
                }
                other => Err(StorageError::IoError(format!(
                    "expected sealed batch member row, got {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>, StorageError>>()?,
        other => {
            return Err(StorageError::IoError(format!(
                "expected sealed batch members array, got {other:?}"
            )));
        }
    };

    // Decode the legacy compatibility payload so old sealed submissions keep
    // round-tripping. It has no transaction validation semantics anymore.
    let captured_frontier = match captured_frontier {
        Value::Array(elements) => elements
            .iter()
            .map(|element| match element {
                Value::Row { values, .. } => {
                    let [object_id, branch_ord, batch_id] = values.as_slice() else {
                        return Err(StorageError::IoError(
                            "expected captured frontier row to have three values".to_string(),
                        ));
                    };
                    let object_id = match object_id {
                        Value::Bytea(bytes) => uuid::Uuid::from_slice(bytes)
                            .map(ObjectId::from_uuid)
                            .map_err(|err| {
                                StorageError::IoError(format!(
                                    "decode captured frontier object id uuid: {err}"
                                ))
                            })?,
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected captured frontier object id bytes, got {other:?}"
                            )));
                        }
                    };
                    let branch_ord = match branch_ord {
                        Value::Integer(raw) => *raw,
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected captured frontier branch ord integer, got {other:?}"
                            )));
                        }
                    };
                    let branch_name =
                        storage
                            .load_branch_name_by_ord(branch_ord)?
                            .ok_or_else(|| {
                                StorageError::IoError(format!(
                                    "missing branch name for captured frontier ord {branch_ord}"
                                ))
                            })?;
                    let batch_id = decode_storage_batch_id_value(
                        batch_id,
                        "decode captured frontier batch id",
                    )?;
                    Ok(CapturedFrontierMember {
                        object_id,
                        branch_name,
                        batch_id,
                    })
                }
                other => Err(StorageError::IoError(format!(
                    "expected captured frontier row, got {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>, StorageError>>()?,
        other => {
            return Err(StorageError::IoError(format!(
                "expected captured frontier array, got {other:?}"
            )));
        }
    };

    let submission = SealedBatchSubmission::new(
        batch_id,
        mode,
        target_branch_name,
        members,
        captured_frontier,
    );
    if submission.batch_digest != batch_digest {
        return Err(StorageError::IoError(format!(
            "sealed batch digest mismatch: expected {batch_digest:?}, computed {:?}",
            submission.batch_digest
        )));
    }
    Ok(submission)
}

fn encode_local_batch_record_with_branch_ords<H: Storage + ?Sized>(
    storage: &mut H,
    record: &LocalBatchRecord,
) -> Result<Vec<u8>, StorageError> {
    encode_row(
        &local_batch_record_storage_descriptor_with_branch_ords(),
        &[
            Value::BatchId(*record.batch_id.as_bytes()),
            Value::Text(encode_batch_mode(record.mode).to_string()),
            Value::Boolean(record.sealed),
            Value::Array(
                record
                    .members
                    .iter()
                    .map(|member| {
                        let branch_ord = storage.resolve_or_alloc_branch_ord(member.branch_name)?;
                        Ok(Value::Row {
                            id: None,
                            values: vec![
                                Value::Bytea(member.object_id.uuid().as_bytes().to_vec()),
                                Value::Text(member.table_name.clone()),
                                Value::Integer(branch_ord),
                                Value::Bytea(member.schema_hash.as_bytes().to_vec()),
                                Value::Bytea(member.row_digest.0.to_vec()),
                            ],
                        })
                    })
                    .collect::<Result<Vec<_>, StorageError>>()?,
            ),
        ],
    )
    .map_err(|err| StorageError::IoError(format!("encode local batch record: {err}")))
}

fn compare_local_batch_member_identity(
    left: &LocalBatchMember,
    right: &LocalBatchMember,
) -> std::cmp::Ordering {
    left.object_id
        .uuid()
        .as_bytes()
        .cmp(right.object_id.uuid().as_bytes())
        .then_with(|| left.table_name.cmp(&right.table_name))
        .then_with(|| left.branch_name.as_str().cmp(right.branch_name.as_str()))
}

fn compare_local_batch_member_version(
    left: &LocalBatchMember,
    right: &LocalBatchMember,
) -> std::cmp::Ordering {
    left.schema_hash
        .as_bytes()
        .cmp(right.schema_hash.as_bytes())
        .then_with(|| left.row_digest.0.cmp(&right.row_digest.0))
}

pub(crate) fn upsert_local_batch_member(
    members: &mut Vec<LocalBatchMember>,
    member: LocalBatchMember,
) {
    match members.binary_search_by(|existing| {
        compare_local_batch_member_identity(existing, &member)
            .then_with(|| compare_local_batch_member_version(existing, &member))
    }) {
        Ok(index) => {
            members[index] = member;
        }
        Err(index) => {
            if index > 0
                && compare_local_batch_member_identity(&members[index - 1], &member)
                    == std::cmp::Ordering::Equal
            {
                members[index - 1] = member;
            } else if index < members.len()
                && compare_local_batch_member_identity(&members[index], &member)
                    == std::cmp::Ordering::Equal
            {
                members[index] = member;
            } else {
                members.insert(index, member);
            }
        }
    }
}

fn encode_local_batch_members(members: &[LocalBatchMember]) -> Value {
    Value::Array(
        members
            .iter()
            .map(|member| Value::Row {
                id: None,
                values: vec![
                    Value::Bytea(member.object_id.uuid().as_bytes().to_vec()),
                    Value::Text(member.table_name.clone()),
                    Value::Text(member.branch_name.to_string()),
                    Value::Bytea(member.schema_hash.as_bytes().to_vec()),
                    Value::Bytea(member.row_digest.0.to_vec()),
                ],
            })
            .collect(),
    )
}

fn encode_local_batch_row_index(
    batch_id: BatchId,
    members: &[LocalBatchMember],
) -> Result<Vec<u8>, StorageError> {
    encode_row(
        &local_batch_row_index_storage_descriptor(),
        &[
            Value::BatchId(*batch_id.as_bytes()),
            encode_local_batch_members(members),
        ],
    )
    .map_err(|err| StorageError::IoError(format!("encode local batch row index: {err}")))
}

fn decode_local_batch_members(members: &Value) -> Result<Vec<LocalBatchMember>, StorageError> {
    match members {
        Value::Array(values) => values
            .iter()
            .map(|value| match value {
                Value::Row { values, .. } => {
                    let [object_id, table_name, branch_name, schema_hash, row_digest] =
                        values.as_slice()
                    else {
                        return Err(StorageError::IoError(
                            "expected local batch member row to have five values".to_string(),
                        ));
                    };
                    let object_id = match object_id {
                        Value::Bytea(bytes) => {
                            let uuid = uuid::Uuid::from_slice(bytes).map_err(|err| {
                                StorageError::IoError(format!(
                                    "decode local batch member object id: expected uuid bytes: {err}"
                                ))
                            })?;
                            ObjectId::from_uuid(uuid)
                        }
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member object id bytes, got {other:?}"
                            )));
                        }
                    };
                    let table_name = match table_name {
                        Value::Text(raw) => raw.clone(),
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member table name text, got {other:?}"
                            )));
                        }
                    };
                    let branch_name = match branch_name {
                        Value::Text(raw) => BranchName::new(raw),
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member branch name text, got {other:?}"
                            )));
                        }
                    };
                    let schema_hash = match schema_hash {
                        Value::Bytea(bytes) => {
                            let bytes: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
                                StorageError::IoError(format!(
                                    "expected local batch member schema hash to be 32 bytes, got {}",
                                    bytes.len()
                                ))
                            })?;
                            SchemaHash::from_bytes(bytes)
                        }
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member schema hash bytes, got {other:?}"
                            )));
                        }
                    };
                    let row_digest = match row_digest {
                        Value::Bytea(bytes) => Digest32(bytes.as_slice().try_into().map_err(
                            |_| {
                                StorageError::IoError(format!(
                                    "expected local batch member row digest to be 32 bytes, got {}",
                                    bytes.len()
                                ))
                            },
                        )?),
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member row digest bytes, got {other:?}"
                            )));
                        }
                    };
                    Ok(LocalBatchMember {
                        object_id,
                        table_name,
                        branch_name,
                        schema_hash,
                        row_digest,
                    })
                }
                other => Err(StorageError::IoError(format!(
                    "expected local batch member row, got {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>, StorageError>>(),
        other => Err(StorageError::IoError(format!(
            "expected local batch members array, got {other:?}"
        ))),
    }
}

fn decode_local_batch_row_index(
    bytes: &[u8],
) -> Result<(BatchId, Vec<LocalBatchMember>), StorageError> {
    let values = decode_row(&local_batch_row_index_storage_descriptor(), bytes)
        .map_err(|err| StorageError::IoError(format!("decode local batch row index: {err}")))?;
    let [batch_id, members] = values.as_slice() else {
        return Err(StorageError::IoError(
            "unexpected local batch row index shape".to_string(),
        ));
    };

    let batch_id =
        decode_storage_batch_id_value(batch_id, "decode local batch row index batch id")?;
    Ok((batch_id, decode_local_batch_members(members)?))
}

fn decode_local_batch_record_with_branch_ords<H: Storage + ?Sized>(
    storage: &H,
    bytes: &[u8],
) -> Result<LocalBatchRecord, StorageError> {
    let values = decode_row(
        &local_batch_record_storage_descriptor_with_branch_ords(),
        bytes,
    )
    .map_err(|err| StorageError::IoError(format!("decode local batch record: {err}")))?;
    let [batch_id, mode, sealed, members] = values.as_slice() else {
        return Err(StorageError::IoError(
            "unexpected local batch record shape".to_string(),
        ));
    };

    let batch_id = decode_storage_batch_id_value(batch_id, "decode local batch record batch id")?;
    let mode = match mode {
        Value::Text(raw) => decode_batch_mode(raw)?,
        other => {
            return Err(StorageError::IoError(format!(
                "expected batch mode text, got {other:?}"
            )));
        }
    };
    let sealed = match sealed {
        Value::Boolean(value) => *value,
        other => {
            return Err(StorageError::IoError(format!(
                "expected sealed boolean, got {other:?}"
            )));
        }
    };
    let members = match members {
        Value::Array(values) => values
                .iter()
                .map(|value| match value {
                    Value::Row { values, .. } => {
                    let [object_id, table_name, branch_ord, schema_hash, row_digest] =
                        values.as_slice()
                    else {
                        return Err(StorageError::IoError(
                            "expected local batch member row to have five values".to_string(),
                        ));
                    };
                    let object_id = match object_id {
                        Value::Bytea(bytes) => {
                            let uuid = uuid::Uuid::from_slice(bytes).map_err(|err| {
                                StorageError::IoError(format!(
                                    "decode local batch member object id: expected uuid bytes: {err}"
                                ))
                            })?;
                            ObjectId::from_uuid(uuid)
                        }
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member object id bytes, got {other:?}"
                            )));
                        }
                    };
                    let table_name = match table_name {
                        Value::Text(raw) => raw.clone(),
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member table name text, got {other:?}"
                            )));
                        }
                    };
                    let branch_ord = match branch_ord {
                        Value::Integer(raw) => *raw,
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member branch ord integer, got {other:?}"
                            )));
                        }
                    };
                    let branch_name = storage
                        .load_branch_name_by_ord(branch_ord)?
                        .ok_or_else(|| {
                            StorageError::IoError(format!(
                                "missing branch name for local batch member branch ord {branch_ord}"
                            ))
                        })?;
                    let schema_hash = match schema_hash {
                        Value::Bytea(bytes) => {
                            let bytes: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
                                StorageError::IoError(format!(
                                    "expected local batch member schema hash to be 32 bytes, got {}",
                                    bytes.len()
                                ))
                            })?;
                            SchemaHash::from_bytes(bytes)
                        }
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member schema hash bytes, got {other:?}"
                            )));
                        }
                    };
                    let row_digest = match row_digest {
                        Value::Bytea(bytes) => Digest32(bytes.as_slice().try_into().map_err(
                            |_| {
                                StorageError::IoError(format!(
                                    "expected local batch member row digest to be 32 bytes, got {}",
                                    bytes.len()
                                ))
                            },
                        )?),
                        other => {
                            return Err(StorageError::IoError(format!(
                                "expected local batch member row digest bytes, got {other:?}"
                            )));
                        }
                    };
                    Ok(crate::batch_fate::LocalBatchMember {
                        object_id,
                        table_name,
                        branch_name,
                        schema_hash,
                        row_digest,
                    })
                }
                other => Err(StorageError::IoError(format!(
                    "expected local batch member row, got {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>, StorageError>>()?,
        other => {
            return Err(StorageError::IoError(format!(
                "expected local batch members array, got {other:?}"
            )));
        }
    };

    Ok(LocalBatchRecord {
        batch_id,
        mode,
        sealed,
        members,
        sealed_submission: storage.load_sealed_batch_submission(batch_id)?,
        latest_fate: storage.load_authoritative_batch_fate(batch_id)?,
    })
}
// ============================================================================
// Value Encoding for Index Keys
// ============================================================================
//
// Values must be encoded so lexicographic byte ordering equals semantic ordering.
// This enables range queries via BTreeMap::range().

/// Returns true if the value is Double(0.0) or Double(-0.0).
///
/// IEEE 754 defines -0.0 == 0.0, but they have distinct bit patterns and
/// therefore distinct index encodings. Query operations must check both.
pub(crate) fn is_double_zero(value: &Value) -> bool {
    matches!(value, Value::Double(f) if *f == 0.0)
}

/// Encode a Value into bytes that sort correctly for range queries.
pub(crate) fn encode_value(value: &Value) -> Vec<u8> {
    match value {
        Value::Null => vec![0x00], // Null sorts first

        Value::Boolean(b) => {
            // false (0x01) < true (0x02)
            vec![0x01, if *b { 0x02 } else { 0x01 }]
        }

        Value::Integer(n) => {
            // Flip sign bit so negative < positive, big-endian for correct ordering
            let mut bytes = vec![0x02];
            bytes.extend_from_slice(&((*n as i64) ^ i64::MIN).to_be_bytes());
            bytes
        }

        Value::BigInt(n) => {
            // Flip sign bit so negative < positive, big-endian for correct ordering
            let mut bytes = vec![0x03];
            bytes.extend_from_slice(&(*n ^ i64::MIN).to_be_bytes());
            bytes
        }

        Value::Double(f) => {
            let mut bytes = vec![0x09];
            let bits = f.to_bits();
            // Flip for lexicographic ordering: if sign bit set, flip all bits;
            // otherwise flip only the sign bit.
            let ordered = if bits & (1u64 << 63) != 0 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            bytes.extend_from_slice(&ordered.to_be_bytes());
            bytes
        }

        Value::Timestamp(ts) => {
            // Unsigned, big-endian (already sorts correctly)
            let mut bytes = vec![0x04];
            bytes.extend_from_slice(&ts.to_be_bytes());
            bytes
        }

        Value::Text(s) => {
            // UTF-8 bytes sort correctly for ASCII; good enough for now
            let mut bytes = vec![0x05];
            bytes.extend_from_slice(s.as_bytes());
            bytes
        }

        Value::Uuid(id) => {
            // UUID bytes compare lexicographically by raw value.
            let mut bytes = vec![0x06];
            bytes.extend_from_slice(id.uuid().as_bytes());
            bytes
        }

        Value::BatchId(batch_id) => {
            let mut bytes = vec![0x0A];
            bytes.extend_from_slice(batch_id);
            bytes
        }

        Value::Bytea(bytes_value) => {
            // Raw bytes for exact-match index semantics.
            let mut bytes = vec![0x09];
            bytes.extend_from_slice(bytes_value);
            bytes
        }

        Value::Array(_) => {
            // Arrays use serialized bytes for equality semantics.
            // The durable key codec hashes oversized segments if needed.
            let mut bytes = vec![0x07];
            let json = serde_json::to_string(value).unwrap_or_default();
            bytes.extend_from_slice(json.as_bytes());
            bytes
        }

        Value::Row { .. } => {
            // Rows use serialized bytes for equality semantics.
            // The durable key codec hashes oversized segments if needed.
            let mut bytes = vec![0x08];
            let json = serde_json::to_string(value).unwrap_or_default();
            bytes.extend_from_slice(json.as_bytes());
            bytes
        }
    }
}

// Throwaway diagnostic probe for a copied production/local RocksDB store.
// Run:
//   JAZZ_PROBE_PATH=/path/to/jazz.rocksdb.copy cargo test -p jazz-tools \
//     --features "rocksdb test-utils" --lib -- storage::store_probe --ignored --nocapture
#[cfg(all(test, feature = "rocksdb"))]
mod store_probe {
    use super::*;

    #[test]
    #[ignore]
    fn dump_branches_and_visible_row_counts() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        let next_ord = load_next_branch_ord(&storage).expect("branch ord meta readable");
        println!("next_branch_ord = {next_ord}");
        let mut branches = Vec::new();
        for ord in 1..next_ord {
            match storage.load_branch_name_by_ord(ord) {
                Ok(Some(name)) => {
                    println!("branch[{ord}] = {name}");
                    branches.push(name);
                }
                Ok(None) => println!("branch[{ord}] = <none>"),
                Err(e) => println!("branch[{ord}] = error: {e}"),
            }
        }
        let tables = [
            "users",
            "apple_identities",
            "user_emails",
            "unique_names",
            "auth_pending_state",
            "chats",
            "chat_members",
            "messages",
        ];
        for branch in &branches {
            for table in tables {
                match scan_visible_row_bytes_with_storage(&storage, table, branch.as_str()) {
                    Ok(rows) if !rows.is_empty() => {
                        println!("visible {table} @ {branch} = {}", rows.len());
                        if table == "chats" || table == "chat_members" {
                            for row in &rows {
                                println!("  {} {}", table, row.row_id);
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(e) => println!("visible {table} @ {branch} = error: {e}"),
                }
            }
        }
        println!("probe done");
    }

    /// End-to-end incident verification against a COPY of a real store:
    /// rehydrate a runtime from its catalogue and require the pre-migration
    /// `users` row to be visible — both to a bare subscription and to a
    /// session-scoped one (the app's shape).
    fn probe_incident_store<S: Storage>(mut storage: S) {
        use crate::query_manager::session::Session as PolicySession;
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let current_hash_hex =
            std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let user_row_id = crate::object::ObjectId::from_uuid(
            uuid::Uuid::parse_str(
                &std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID"),
            )
            .expect("valid row id"),
        );

        // Phase 1: extract the app's current schema from the catalogue. With
        // an empty current schema every catalogue schema is identity-activated
        // (no shared tables), so both land in live_schemas.
        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target_hash = crate::query_manager::types::SchemaHash::from_hex(&current_hash_hex)
            .expect("valid schema hash");
        let current_schema = extractor
            .context()
            .live_schemas
            .get(&target_hash)
            .or_else(|| extractor.context().pending_schemas.get(&target_hash))
            .cloned()
            .expect("the app's schema must be in the catalogue");

        // Phase 2: the app's world — current schema + full catalogue.
        let mut sm = SchemaManager::new(SyncManager::new(), current_schema, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let qm = sm.query_manager_mut();
        println!("branches queried: {:?}", qm.all_query_branches());

        let sub = qm
            .subscribe(qm.query("users").build())
            .expect("subscribe to users");
        qm.process(&mut storage);
        let results = qm.get_subscription_results(sub);
        println!("no-session users rows visible: {}", results.len());
        let ids: Vec<String> = results.iter().map(|(id, _)| id.to_string()).collect();
        assert!(
            ids.iter().any(|id| id == &user_row_id.to_string()),
            "no-session: the incident user's row is invisible (visible ids: {ids:?})"
        );

        // The app's shape: the owner's session.
        let session = PolicySession::new(&user_row_id.to_string());
        let query = qm.query("users").build();
        let sub2 = qm
            .subscribe_with_session(query, Some(session), None)
            .expect("session subscribe to users");
        qm.process(&mut storage);
        let with_session = qm.get_subscription_results(sub2);
        println!("with-session users rows visible: {}", with_session.len());
        let session_ids: Vec<String> = with_session.iter().map(|(id, _)| id.to_string()).collect();
        assert!(
            session_ids.iter().any(|id| id == &user_row_id.to_string()),
            "with-session: the incident user's own row is invisible (visible ids: {session_ids:?})"
        );
    }

    #[test]
    #[ignore]
    fn incident_store_serves_the_users_row_after_rehydrate() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        probe_incident_store(storage);
    }

    /// Does this store hold any terminal `Rejected` fate? A rejected ancestor
    /// silences a row's outbound sync forever (defect 28), so the question
    /// "is there one at all" decides whether that mechanism is live here.
    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_rejected_fates() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-fate-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the store for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");
        let fates = storage
            .scan_authoritative_batch_fates()
            .expect("scan authoritative batch fates");
        let mut missing = 0usize;
        let mut rejected = Vec::new();
        let mut durable = 0usize;
        let mut accepted = 0usize;
        for fate in &fates {
            match fate {
                crate::batch_fate::BatchFate::Missing { .. } => missing += 1,
                crate::batch_fate::BatchFate::Rejected {
                    batch_id,
                    code,
                    reason,
                } => rejected.push((*batch_id, code.clone(), reason.clone())),
                crate::batch_fate::BatchFate::DurableDirect { .. } => durable += 1,
                crate::batch_fate::BatchFate::AcceptedTransaction { .. } => accepted += 1,
            }
        }
        println!(
            "fates: total={} durable_direct={durable} accepted_transaction={accepted} missing={missing} rejected={}",
            fates.len(),
            rejected.len()
        );
        for (batch_id, code, reason) in rejected.iter().take(10) {
            println!("  REJECTED {batch_id:?} code={code} reason={reason}");
        }

        // Which tier confirmed them, newest first: a beat the client never
        // pushed can only be Local-confirmed.
        let mut tiers: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut with_time: Vec<(u64, String)> = Vec::new();
        for fate in &fates {
            let tier = fate
                .confirmed_tier()
                .map(|tier| format!("{tier:?}"))
                .unwrap_or_else(|| "none".to_string());
            *tiers.entry(tier.clone()).or_default() += 1;
            let batch_id = fate.batch_id();
            let bytes = batch_id.as_bytes();
            let ms = u64::from(bytes[0]) << 40
                | u64::from(bytes[1]) << 32
                | u64::from(bytes[2]) << 24
                | u64::from(bytes[3]) << 16
                | u64::from(bytes[4]) << 8
                | u64::from(bytes[5]);
            with_time.push((ms, tier));
        }
        println!("confirmed tiers: {tiers:?}");
        with_time.sort_by_key(|entry| entry.0);
        for (ms, tier) in with_time.iter().rev().take(6) {
            println!("  newest fate ms={ms} tier={tier}");
        }
    }

    /// Raw-table forensics: for every raw table holding the probed row, print
    /// the newest version stamp actually stored. Distinguishes "the write never
    /// arrived" from "it arrived but reads do not serve it".
    #[test]
    #[ignore]
    fn probe_raw_row_versions() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        let row_hex = std::env::var("JAZZ_PROBE_ROW_ID")
            .expect("set JAZZ_PROBE_ROW_ID")
            .replace('-', "");
        for (name, _header) in storage
            .scan_raw_table_headers()
            .expect("scan raw table headers")
        {
            if !name.contains("users") {
                continue;
            }
            let mut newest: u64 = 0;
            let mut count = 0usize;
            for (key, value) in storage
                .raw_table_scan_prefix(&name, "")
                .expect("scan raw table")
            {
                if !key.contains(&row_hex) {
                    continue;
                }
                count += 1;
                if value.len() >= 0x18
                    && let Ok(stamp_bytes) = <[u8; 8]>::try_from(&value[0x10..0x18])
                {
                    newest = newest.max(u64::from_le_bytes(stamp_bytes));
                }
            }
            if count > 0 {
                println!("{name}: entries_for_row={count} newest_stamp_us={newest}");
            }
        }
    }

    /// The rpc-server's membership check, replayed against a copy of its own
    /// replica: which predicate makes it come back empty. Env: JAZZ_PROBE_PATH,
    /// JAZZ_PROBE_APP_ID, JAZZ_PROBE_SCHEMA_HASH, JAZZ_PROBE_ROW_ID (userId),
    /// JAZZ_PROBE_CHAT_ID.
    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_rpc_membership_query() {
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-rpc-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica for the probe");
        // v18 item 8, diff r24 SF2: the sidecars MUST come too. This probe reads a copy of a
        // live store, and until item 8 the engine checkpointed on every barrier, which kept the
        // main file near-current and let this get away with copying it alone. With a 30 s
        // interval the main file lags by up to an interval, and by arbitrarily more whenever a
        // reader is pinning the WAL — so a probe that skips the sidecar reads stale rows and
        // reports them as the store's contents. Its siblings in this file already do this.
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let mut storage = SqliteStorage::open(&scratch).expect("open the scratch copy");

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let hash_hex = std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let user_id = std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID");
        let chat_id = std::env::var("JAZZ_PROBE_CHAT_ID").expect("set JAZZ_PROBE_CHAT_ID");

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target = crate::query_manager::types::SchemaHash::from_hex(&hash_hex)
            .expect("valid schema hash");
        let current = extractor
            .context()
            .live_schemas
            .get(&target)
            .or_else(|| extractor.context().pending_schemas.get(&target))
            .cloned()
            .expect("schema in catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let qm = sm.query_manager_mut();
        println!("branches: {:?}", qm.all_query_branches());

        let run = |qm: &mut crate::query_manager::manager::QueryManager,
                   storage: &mut SqliteStorage,
                   label: &str,
                   query: crate::query_manager::query::Query| {
            let sub = qm.subscribe(query).expect("subscribe");
            qm.process(storage);
            println!("{label}: {} rows", qm.get_subscription_results(sub).len());
        };

        run(
            qm,
            &mut storage,
            "all chat_members",
            qm.query("chat_members").build(),
        );
        run(
            qm,
            &mut storage,
            "userId only",
            qm.query("chat_members")
                .filter_eq("userId", Value::Text(user_id.clone()))
                .build(),
        );
        run(
            qm,
            &mut storage,
            "userId + chatId",
            qm.query("chat_members")
                .filter_eq("userId", Value::Text(user_id.clone()))
                .filter_eq("chatId", Value::Text(chat_id.clone()))
                .build(),
        );
        // As the BACKEND principal — the session the rpc-server's facade uses.
        for table in ["chat_members", "apple_identities", "unique_names", "users"] {
            let session = crate::query_manager::session::Session::new("jazz:system");
            let query = qm.query(table).build();
            let sub = qm
                .subscribe_with_session(query, Some(session), None)
                .expect("backend-session subscribe");
            qm.process(&mut storage);
            println!(
                "AS jazz:system, {table}: {} rows",
                qm.get_subscription_results(sub).len()
            );
            let sub_none = qm
                .subscribe(qm.query(table).build())
                .expect("no-session subscribe");
            qm.process(&mut storage);
            println!(
                "   no session,  {table}: {} rows",
                qm.get_subscription_results(sub_none).len()
            );
        }

        // The same check, but SESSION-SCOPED — the shape a policied runtime uses.
        {
            let session = crate::query_manager::session::Session::new(&user_id);
            let query = qm
                .query("chat_members")
                .filter_eq("userId", Value::Text(user_id.clone()))
                .filter_eq("chatId", Value::Text(chat_id.clone()))
                .filter_eq("isBanned", Value::Boolean(false))
                .build();
            let sub = qm
                .subscribe_with_session(query, Some(session), None)
                .expect("session subscribe");
            qm.process(&mut storage);
            println!(
                "WITH SESSION, the rpc's own check: {} rows",
                qm.get_subscription_results(sub).len()
            );
        }
        println!(
            "permissions head present in this replica: {:?}",
            sm.current_permissions().map(|p| p.head.version)
        );
        let qm = sm.query_manager_mut();

        run(
            qm,
            &mut storage,
            "userId + chatId + isBanned=false (the rpc's own check)",
            qm.query("chat_members")
                .filter_eq("userId", Value::Text(user_id))
                .filter_eq("chatId", Value::Text(chat_id))
                .filter_eq("isBanned", Value::Boolean(false))
                .build(),
        );
    }

    /// The same replayed membership check, but run at each DURABILITY TIER —
    /// the only thing the live rpc-server adds over `probe_rpc_membership_query`
    /// on the LOCAL path. Also dumps, per membership row, the stored
    /// `confirmed_tier` and the authoritative batch fate the tier gate consults,
    /// so "the tier gate hides it" can be told apart from "the row is absent".
    /// Env: JAZZ_PROBE_PATH, JAZZ_PROBE_APP_ID, JAZZ_PROBE_SCHEMA_HASH,
    /// JAZZ_PROBE_ROW_ID (userId), JAZZ_PROBE_CHAT_ID.
    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_rpc_membership_tiered() {
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-rpc-tier-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let mut storage = SqliteStorage::open(&scratch).expect("open the scratch copy");

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let hash_hex = std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let user_id = std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID");
        let chat_id = std::env::var("JAZZ_PROBE_CHAT_ID").expect("set JAZZ_PROBE_CHAT_ID");

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target = crate::query_manager::types::SchemaHash::from_hex(&hash_hex)
            .expect("valid schema hash");
        let current = extractor
            .context()
            .live_schemas
            .get(&target)
            .or_else(|| extractor.context().pending_schemas.get(&target))
            .cloned()
            .expect("schema in catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let qm = sm.query_manager_mut();
        let branches = qm.all_query_branches();
        println!("branches: {branches:?}");

        // Per-row tier forensics straight off the store.
        for branch in &branches {
            let rows = match storage.scan_visible_region("chat_members", branch) {
                Ok(rows) => rows,
                Err(error) => {
                    println!("scan {branch}: error {error}");
                    continue;
                }
            };
            println!("branch {branch}: {} visible chat_members rows", rows.len());
            for row in &rows {
                let fate = storage
                    .load_authoritative_batch_fate(row.batch_id)
                    .expect("load authoritative batch fate");
                let effective = row_confirmed_tier_with_batch_fate(&storage, row)
                    .expect("effective confirmed tier");
                println!(
                    "  row={} batch={:?} state={:?} stored_tier={:?} fate={:?} effective_tier={:?}",
                    row.row_id,
                    row.batch_id,
                    row.state,
                    row.confirmed_tier,
                    fate.as_ref().map(|fate| format!("{fate:?}")),
                    effective,
                );
                for tier in [
                    DurabilityTier::Local,
                    DurabilityTier::EdgeServer,
                    DurabilityTier::GlobalServer,
                ] {
                    let served = storage
                        .load_visible_query_row_for_tier("chat_members", branch, row.row_id, tier)
                        .expect("tiered point read");
                    println!(
                        "    point read tier={tier:?} -> served={}",
                        served.is_some()
                    );
                }
            }
        }

        // The rpc's own predicate, once per tier the facade can ask for.
        for tier in [
            None,
            Some(DurabilityTier::Local),
            Some(DurabilityTier::EdgeServer),
            Some(DurabilityTier::GlobalServer),
        ] {
            let session = crate::query_manager::session::Session::new(&user_id);
            let query = qm
                .query("chat_members")
                .filter_eq("userId", Value::Text(user_id.clone()))
                .filter_eq("chatId", Value::Text(chat_id.clone()))
                .filter_eq("isBanned", Value::Boolean(false))
                .build();
            let sub = qm
                .subscribe_with_session(query, Some(session), tier)
                .expect("session subscribe");
            qm.process(&mut storage);
            println!(
                "rpc check tier={tier:?} -> {} rows",
                qm.get_subscription_results(sub).len()
            );
        }

        // Which GENERATION actually holds the membership row: the same check
        // pinned to one branch at a time. A hit only on the old branch means a
        // reader restricted to the current generation answers "not a member".
        for branch in &branches {
            let query = qm
                .query("chat_members")
                .branch(branch.as_str())
                .filter_eq("userId", Value::Text(user_id.clone()))
                .filter_eq("chatId", Value::Text(chat_id.clone()))
                .filter_eq("isBanned", Value::Boolean(false))
                .build();
            let sub = qm.subscribe(query).expect("branch-pinned subscribe");
            qm.process(&mut storage);
            let results = qm.get_subscription_results(sub);
            println!(
                "rpc check pinned to branch {branch} -> {} rows {:?}",
                results.len(),
                results
                    .iter()
                    .map(|(id, _)| id.to_string())
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Store census: for every table named in JAZZ_PROBE_TABLES (comma
    /// separated), print the raw-table families that hold it, the visible row
    /// count per branch, and the count a plain no-session query serves. Run it
    /// against a copy of the rpc replica AND a copy of the sync server's store
    /// to see which side is missing what. Env: JAZZ_PROBE_PATH,
    /// JAZZ_PROBE_APP_ID, JAZZ_PROBE_SCHEMA_HASH, JAZZ_PROBE_TABLES.
    fn probe_table_census<S: Storage>(mut storage: S) {
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let hash_hex = std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let tables = std::env::var("JAZZ_PROBE_TABLES").expect("set JAZZ_PROBE_TABLES");

        for (name, _header) in storage
            .scan_raw_table_headers()
            .expect("scan raw table headers")
        {
            if name.contains("rowtable") {
                println!("raw family: {name}");
            }
        }

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target = crate::query_manager::types::SchemaHash::from_hex(&hash_hex)
            .expect("valid schema hash");
        let current = extractor
            .context()
            .live_schemas
            .get(&target)
            .or_else(|| extractor.context().pending_schemas.get(&target))
            .cloned()
            .expect("schema in catalogue");

        // JAZZ_PROBE_NO_REHYDRATE=1 reproduces the boot shape jazz-napi had
        // before crates/jazz-napi/src/lib.rs:624: `SchemaManager::new*` over the
        // DECLARED schema and no catalogue rehydrate, versus
        // jazz-rn/rust/src/lib.rs:734, jazz-wasm/src/runtime.rs:1364,
        // jazz-tools/src/client.rs:127 and server/builder.rs:236, which all do.
        let skip_rehydrate = std::env::var("JAZZ_PROBE_NO_REHYDRATE").is_ok();
        let mut sm = SchemaManager::new(SyncManager::new(), current, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        if skip_rehydrate {
            println!("phase-2: NO catalogue rehydrate (jazz-napi boot shape)");
        } else {
            crate::schema_manager::rehydrate_schema_manager_from_catalogue(
                &mut sm, &storage, app_id,
            )
            .expect("phase-2 rehydrate");
        }
        println!(
            "permissions head: {:?}",
            sm.current_permissions().map(|p| p.head.version)
        );
        let qm = sm.query_manager_mut();
        let branches = qm.all_query_branches();
        println!("branches: {branches:?}");

        for table in tables.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let mut per_branch = Vec::new();
            for branch in &branches {
                let count = storage
                    .scan_visible_region(table, branch)
                    .map(|rows| rows.len() as i64)
                    .unwrap_or(-1);
                per_branch.push(format!("{branch}={count}"));
            }
            let sub = qm
                .subscribe(qm.query(table).build())
                .expect("subscribe to table");
            qm.process(&mut storage);
            let served = qm.get_subscription_results(sub).len();
            println!(
                "table {table}: query_serves={served} visible_per_branch=[{}]",
                per_branch.join(" ")
            );
        }
    }

    #[test]
    #[ignore]
    fn probe_table_census_rocksdb() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        probe_table_census(storage);
    }

    /// What the per-tick sealed-batch recovery sweep actually reads.
    ///
    /// `recover_completed_sealed_batches_with_storage` runs first in EVERY
    /// `immediate_tick` (runtime_core/ticks.rs:661), unconditionally: it scans the
    /// whole retained sealed-submission table, decodes each row (resolving a branch
    /// name per member ord), then point-gets the authoritative fate of each batch and
    /// `continue`s on anything already settled. The scan is documented as small —
    /// "Submissions are deleted once the runtime no longer needs the original
    /// seal/member list" — so this probe measures whether that holds in a real store,
    /// and how much of the sweep is re-read every tick only to be discarded.
    /// Where does a presence heartbeat land, and where is it read from?
    ///
    /// `users` carries the presence stamp, and the row is split across schema generations:
    /// one `__row_locator` entry names ONE origin hash, so a row alive on N generations has
    /// N raw-table families and the ladder recovers the other N-1 on every read. If the
    /// writer's generation and the reader's differ, a heartbeat updates a copy nobody reads
    /// — presence freezes while a single-generation table like typing keeps working.
    ///
    /// Prints, per branch, how fresh the newest users row is.
    #[test]
    #[ignore]
    fn probe_users_presence_across_generations() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");

        let mut branches: Vec<String> = storage
            .scan_raw_table_headers()
            .expect("scan raw table headers")
            .into_iter()
            .filter_map(|(name, _)| {
                let rest = name.strip_prefix("rowtable:visible:users:")?;
                Some(format!("dev-{}-main", &rest[..12.min(rest.len())]))
            })
            .collect();
        branches.sort();
        branches.dedup();

        for branch in &branches {
            let rows = storage
                .scan_visible_region("users", branch)
                .unwrap_or_default();
            let mut per_row: Vec<(String, u64)> = rows
                .iter()
                .map(|row| {
                    (
                        row.row_id.to_string()[..8].to_string(),
                        row.row_provenance().updated_at,
                    )
                })
                .collect();
            per_row.sort_by_key(|(_, at)| std::cmp::Reverse(*at));
            let newest = per_row.first().map(|(_, at)| *at).unwrap_or(0);
            println!("branch {branch}: {} visible users rows", rows.len());
            println!("  newest updated_at = {newest}");
            for (row, at) in per_row.iter().take(4) {
                println!("    {row} {at}");
            }
        }
    }

    /// Does the server hold a FATE for a batch whose row it never applied?
    ///
    /// If it does, the deadlock is explained: the client was told the batch was durable, so
    /// it retired its own copy and can no longer answer the `Missing` request the server is
    /// now making. Nobody has the row, and every later write on that chain parents from a
    /// batch that will never arrive.
    ///
    /// `JAZZ_PROBE_BATCH_HEX` is the batch id; `JAZZ_PROBE_ROW_HEX` the row it belonged to.
    #[test]
    #[ignore]
    fn probe_orphaned_fate_without_row() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let batch_hex = std::env::var("JAZZ_PROBE_BATCH_HEX").expect("set JAZZ_PROBE_BATCH_HEX");
        let row_hex = std::env::var("JAZZ_PROBE_ROW_HEX").expect("set JAZZ_PROBE_ROW_HEX");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");

        let bytes = hex::decode(&batch_hex).expect("batch hex");
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes);
        let batch_id = crate::row_histories::BatchId(id);

        println!("batch {batch_hex}");
        println!(
            "  authoritative fate : {:?}",
            storage.load_authoritative_batch_fate(batch_id)
        );
        println!(
            "  sealed submission  : {}",
            storage
                .load_sealed_batch_submission(batch_id)
                .ok()
                .flatten()
                .is_some()
        );
        println!(
            "  local batch record : {}",
            storage
                .load_local_batch_record(batch_id)
                .ok()
                .flatten()
                .is_some()
        );

        // Is the row version itself anywhere in this row's history?
        let uuid = uuid::Uuid::parse_str(&row_hex).expect("row hex");
        let row_id = crate::object::ObjectId::from_uuid(uuid);
        for (name, _) in storage.scan_raw_table_headers().expect("headers") {
            if !name.contains("history") || !name.contains(":users:") {
                continue;
            }
            let keys = storage
                .raw_table_scan_prefix_keys(&name, &row_hex.replace('-', ""))
                .unwrap_or_default();
            let has_batch = keys.iter().any(|k| k.ends_with(&batch_hex));
            println!(
                "  {name}: {} versions of this row, holds the batch: {has_batch}",
                keys.len()
            );
        }
    }

    /// What one presence beat costs the CLIENT, on its own store.
    ///
    /// The device pays the same apply path as the server: if the row's stored frontier has
    /// more than one tip, the incoming single-parent write cannot cover it, the O(1) fast
    /// path declines, and every beat re-reads the row's whole history and rebuilds the
    /// visible entry. A phone profile showed a CPU spike on every beat, peaking at 107% of
    /// a core.
    ///
    /// Point `JAZZ_PROBE_PATH` at a copy of a device's sqlite store.
    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_client_row_frontier_and_depth() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage = SqliteStorage::open(&path).expect("open the device store");

        for (name, _) in storage.scan_raw_table_headers().expect("headers") {
            let Some(rest) = name.strip_prefix("rowtable:visible:") else {
                continue;
            };
            let Some((table, hash)) = rest.split_once(':') else {
                continue;
            };
            let branch = format!("dev-{}-main", &hash[..12.min(hash.len())]);
            let rows = storage
                .scan_visible_region(table, &branch)
                .unwrap_or_default();
            if rows.is_empty() {
                continue;
            }
            println!("--- {table} on {branch}: {} visible rows", rows.len());
            let mut worst: Vec<(usize, usize, String)> = Vec::new();
            for row in &rows {
                let Ok(Some(entry)) = storage.load_visible_region_entry(table, &branch, row.row_id)
                else {
                    continue;
                };
                let depth = storage
                    .scan_history_row_batches(table, row.row_id)
                    .map(|v| v.len())
                    .unwrap_or(0);
                worst.push((
                    entry.branch_frontier.len(),
                    depth,
                    row.row_id.to_string()[..8].to_string(),
                ));
            }
            worst.sort_by_key(|(tips, depth, _)| std::cmp::Reverse((*tips, *depth)));
            for (tips, depth, row) in worst.iter().take(5) {
                println!("    row {row}: frontier tips = {tips}, history depth = {depth}");
            }
        }
    }

    #[test]
    #[ignore]
    fn probe_recovery_sweep_rocksdb() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");

        let raw = storage
            .raw_table_scan_prefix(SEALED_BATCH_SUBMISSION_TABLE, "batch:")
            .expect("scan submission keys");
        println!("retained sealed submissions (raw keys): {}", raw.len());

        // How many row versions each table carries — the denominator for "cost per write".
        for (name, _header) in storage.scan_raw_table_headers().expect("headers") {
            if name.contains("history") && name.contains("chat_drafts") {
                let keys = storage
                    .raw_table_scan_prefix_keys(&name, "")
                    .unwrap_or_default();
                // key = <row_id_hex>:<branch>:<batch_id_hex>
                let mut rows_by_id: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                let mut branches: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                let mut batches: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();
                let mut oldest = i64::MAX;
                let mut newest = i64::MIN;
                for key in &keys {
                    let parts: Vec<&str> = key.rsplitn(2, ':').collect();
                    if parts.len() != 2 {
                        continue;
                    }
                    let batch_hex = parts[0];
                    let head = parts[1];
                    batches.insert(batch_hex.to_string());
                    if let Some((row_id, branch)) = head.split_once(':') {
                        *rows_by_id.entry(row_id.to_string()).or_default() += 1;
                        *branches.entry(branch.to_string()).or_default() += 1;
                    }
                    if let Ok(bytes) = hex::decode(batch_hex)
                        && bytes.len() == 16
                    {
                        let ms = ((bytes[0] as i64) << 40)
                            | ((bytes[1] as i64) << 32)
                            | ((bytes[2] as i64) << 24)
                            | ((bytes[3] as i64) << 16)
                            | ((bytes[4] as i64) << 8)
                            | (bytes[5] as i64);
                        oldest = oldest.min(ms);
                        newest = newest.max(ms);
                    }
                }
                println!("history rows in {name}: {}", keys.len());
                println!("  distinct batches (logical writes): {}", batches.len());
                println!("  distinct row ids: {}", rows_by_id.len());
                println!("  per branch: {branches:?}");
                println!("  batch time span: {oldest} .. {newest} unix_ms");
                // The last ten minutes only: that is the session just measured.
                let cutoff = newest - 600_000;
                let mut recent_by_row: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                let mut recent_by_second: std::collections::BTreeMap<i64, usize> =
                    std::collections::BTreeMap::new();
                for key in &keys {
                    let parts: Vec<&str> = key.rsplitn(2, ':').collect();
                    if parts.len() != 2 {
                        continue;
                    }
                    let Ok(bytes) = hex::decode(parts[0]) else {
                        continue;
                    };
                    if bytes.len() != 16 {
                        continue;
                    }
                    let ms = ((bytes[0] as i64) << 40)
                        | ((bytes[1] as i64) << 32)
                        | ((bytes[2] as i64) << 24)
                        | ((bytes[3] as i64) << 16)
                        | ((bytes[4] as i64) << 8)
                        | (bytes[5] as i64);
                    if ms < cutoff {
                        continue;
                    }
                    if let Some((row_id, _)) = parts[1].split_once(':') {
                        *recent_by_row.entry(row_id[..8].to_string()).or_default() += 1;
                    }
                    *recent_by_second.entry(ms / 1000).or_default() += 1;
                }
                let total_recent: usize = recent_by_row.values().sum();
                println!(
                    "  last 10 min: {total_recent} writes across {} rows",
                    recent_by_row.len()
                );
                println!("  per row: {recent_by_row:?}");
                let mut per_sec: Vec<usize> = recent_by_second.values().copied().collect();
                per_sec.sort_unstable();
                // Which apply shape is this row in? The serial fast path needs the incoming
                // parents to cover the STORED frontier exactly; a frontier that has grown
                // more than one tip can never be covered by a single-parent write, and every
                // write then pays a full history scan plus a rebuild.
                if let Some((row_hex, _)) = recent_by_row.iter().max_by_key(|(_, n)| **n) {
                    for key in &keys {
                        if !key.starts_with(row_hex.as_str()) {
                            continue;
                        }
                        let Some((full_row_hex, rest)) = key.split_once(':') else {
                            continue;
                        };
                        let Some((branch, _)) = rest.split_once(':') else {
                            continue;
                        };
                        if let Ok(uuid) = uuid::Uuid::parse_str(full_row_hex)
                            && let row_id = crate::object::ObjectId::from_uuid(uuid)
                            && let Ok(Some(entry)) =
                                storage.load_visible_region_entry("chat_drafts", branch, row_id)
                        {
                            println!(
                                "  busiest row {full_row_hex}: frontier tips = {}, winner pool = {}, merge artifacts present = {}",
                                entry.branch_frontier.len(),
                                entry.winner_batch_pool.len(),
                                entry.merge_artifacts.is_some()
                            );
                        }
                        break;
                    }
                }
                println!(
                    "  active seconds: {}, writes/s median {}, max {}",
                    per_sec.len(),
                    per_sec.get(per_sec.len() / 2).copied().unwrap_or(0),
                    per_sec.last().copied().unwrap_or(0)
                );
            }
        }

        // Warm the block cache before timing anything: whichever order ran first would
        // otherwise pay for the other one's cold reads and the comparison would be a
        // measurement of cache state, not of the change.
        let _ = storage
            .scan_sealed_batch_submissions()
            .expect("warm the cache");
        let _ = storage
            .scan_sealed_batch_submission_ids()
            .expect("warm the cache");

        let started = std::time::Instant::now();
        let submissions = storage
            .scan_sealed_batch_submissions()
            .expect("scan sealed batch submissions");
        let scan_cost = started.elapsed();

        let started = std::time::Instant::now();
        let mut already_settled = 0usize;
        let mut by_variant: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for submission in &submissions {
            match storage
                .load_authoritative_batch_fate(submission.batch_id)
                .expect("load fate")
            {
                Some(fate) => {
                    already_settled += 1;
                    let label = match &fate {
                        crate::batch_fate::BatchFate::Rejected { .. } => "Rejected".to_string(),
                        crate::batch_fate::BatchFate::Missing { .. } => "Missing".to_string(),
                        other => format!("{:?}", other.confirmed_tier()),
                    };
                    *by_variant.entry(label).or_default() += 1;
                }
                None => *by_variant.entry("no fate".to_string()).or_default() += 1,
            }
        }
        let fate_cost = started.elapsed();
        println!("fates of retained submissions: {by_variant:?}");

        println!(
            "decoded {} submissions in {:?}; fate lookups {:?}; already settled {} ({:.1}%)",
            submissions.len(),
            scan_cost,
            fate_cost,
            already_settled,
            if submissions.is_empty() {
                0.0
            } else {
                100.0 * already_settled as f64 / submissions.len() as f64
            }
        );
        // Cross-check the two fate stores. `LocalBatchRecord::apply_fate` (batch_fate.rs:384)
        // REFUSES to downgrade a confirmed fate to `Missing`; `merged_with` (batch_fate.rs:123,
        // the `_ => incoming.clone()` catch-all) accepts it, and that is the path
        // `upsert_authoritative_batch_fate` takes. So a record that still says
        // DurableDirect/AcceptedTransaction while the authoritative table says `Missing` is
        // direct evidence that the downgrade fired here, rather than the batch simply never
        // having landed.
        let mut record_fates: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for submission in &submissions {
            let label = match storage.load_local_batch_record(submission.batch_id) {
                Ok(Some(record)) => match record.latest_fate {
                    Some(crate::batch_fate::BatchFate::Missing { .. }) => "record: Missing",
                    Some(crate::batch_fate::BatchFate::Rejected { .. }) => "record: Rejected",
                    Some(crate::batch_fate::BatchFate::DurableDirect { .. }) => {
                        "record: DurableDirect (DOWNGRADED)"
                    }
                    Some(crate::batch_fate::BatchFate::AcceptedTransaction { .. }) => {
                        "record: AcceptedTransaction (DOWNGRADED)"
                    }
                    None => "record present, no fate",
                },
                Ok(None) => "no local batch record",
                Err(_) => "record read error",
            };
            *record_fates.entry(label.to_string()).or_default() += 1;
        }
        println!("local batch records for the same batches: {record_fates:?}");

        // BatchId is a UUIDv7: the first 48 bits are a millisecond unix timestamp, so
        // every retained submission can be dated exactly. That separates a historical
        // residue (all old, nothing accruing) from a live leak (arrivals every day).
        let mut days: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        let mut oldest_ms = i64::MAX;
        let mut newest_ms = i64::MIN;
        for submission in &submissions {
            let b = submission.batch_id.as_bytes();
            let ms = ((b[0] as i64) << 40)
                | ((b[1] as i64) << 32)
                | ((b[2] as i64) << 24)
                | ((b[3] as i64) << 16)
                | ((b[4] as i64) << 8)
                | (b[5] as i64);
            oldest_ms = oldest_ms.min(ms);
            newest_ms = newest_ms.max(ms);
            *days.entry(ms / 86_400_000).or_default() += 1;
        }
        println!("oldest submission unix_ms={oldest_ms}, newest unix_ms={newest_ms}");
        println!("per-day counts (unix_day -> retained): {days:?}");

        // The same work in the order the sweep now uses: ids only, then the fate, and the
        // row read only for whatever survives. On a store where nothing is drivable the
        // rows are never read at all, which is the whole point.
        let started = std::time::Instant::now();
        let ids = storage
            .scan_sealed_batch_submission_ids()
            .expect("scan sealed batch submission ids");
        let id_scan_cost = started.elapsed();
        let started = std::time::Instant::now();
        let mut drivable = 0usize;
        for batch_id in &ids {
            match storage
                .load_authoritative_batch_fate(*batch_id)
                .expect("load fate")
            {
                None => drivable += 1,
                Some(crate::batch_fate::BatchFate::DurableDirect { .. }) => drivable += 1,
                Some(_) => {}
            }
        }
        let new_fate_cost = started.elapsed();
        println!(
            "NEW order: {} ids in {:?}; fate lookups {:?}; rows worth reading: {}",
            ids.len(),
            id_scan_cost,
            new_fate_cost,
            drivable
        );
        println!("NEW sweep costs {:?}", id_scan_cost + new_fate_cost);

        let sweep = scan_cost + fate_cost;
        println!(
            "one sweep costs {:?}; at 100 ticks/s that is {:.1}% of a core",
            sweep,
            100.0 * sweep.as_secs_f64() * 100.0
        );
    }

    /// How much of the store is the SAME row id on two generations' branches,
    /// and how much of that disagrees.
    ///
    /// The honest size of the surprise when a runtime's universe widens from one
    /// branch to two. A blind server's `upsert` could not find a row on the only
    /// branch it could see, so it fell through to `insert` with the same object
    /// id on the CURRENT branch — leaving one row id with a visible head under
    /// both generations. Widening makes both visible; the union collapses them
    /// by id and the newest wins, so the row served is right. What this measures
    /// is how often the loser DISAGREES with the winner, and how often the older
    /// copy would win — the two numbers worth knowing before shipping.
    ///
    /// Env: JAZZ_PROBE_PATH, JAZZ_PROBE_APP_ID, JAZZ_PROBE_SCHEMA_HASH,
    /// JAZZ_PROBE_TABLES.
    fn probe_cross_branch_duplicates<S: Storage>(storage: S) {
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let hash_hex = std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let tables = std::env::var("JAZZ_PROBE_TABLES").expect("set JAZZ_PROBE_TABLES");

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target = crate::query_manager::types::SchemaHash::from_hex(&hash_hex)
            .expect("valid schema hash");
        let current = extractor
            .context()
            .live_schemas
            .get(&target)
            .or_else(|| extractor.context().pending_schemas.get(&target))
            .cloned()
            .expect("schema in catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let branches = sm.query_manager_mut().all_query_branches();
        println!("branches: {branches:?}");
        if branches.len() < 2 {
            println!("single-generation store — nothing to cross-check");
            return;
        }

        let mut total_shared = 0usize;
        let mut total_disagreeing = 0usize;
        let mut total_older_wins = 0usize;
        for table in tables.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            // Mirror the resolver exactly, or the answer is about a different
            // question than the one being asked. `load_best_visible_row_batch_from_storage_with_locator`
            // (query_manager/manager.rs:2804-2817) skips non-visible heads and
            // compares the TUPLE `(updated_at, batch_id)` — dropping either would
            // both under- and over-count which copy actually gets served.
            let per_branch: Vec<HashMap<ObjectId, (u64, BatchId, Vec<u8>)>> = branches
                .iter()
                .map(|branch| {
                    storage
                        .scan_visible_region(table, branch)
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|row| row.state.is_visible())
                        .map(|row| {
                            (
                                row.row_id,
                                (row.updated_at, row.batch_id, row.data.to_vec()),
                            )
                        })
                        .collect()
                })
                .collect();

            let (current_branch, older_branches) = per_branch.split_first().expect("two branches");
            let mut shared = 0usize;
            let mut disagreeing = 0usize;
            let mut older_wins = 0usize;
            let mut only_older = 0usize;
            for older in older_branches {
                for (row_id, (older_updated_at, older_batch_id, older_content)) in older {
                    let Some((current_updated_at, current_batch_id, current_content)) =
                        current_branch.get(row_id)
                    else {
                        only_older += 1;
                        continue;
                    };
                    shared += 1;
                    // NOTE: `data` is encoded under each generation's own row
                    // descriptor. For an identity crossing — every shared table
                    // byte-identical, which is what lets the older generation
                    // activate at all — the encodings are comparable. Under a
                    // lens crossing they are not, and this count would be noise.
                    if older_content != current_content {
                        disagreeing += 1;
                    }
                    if (*older_updated_at, *older_batch_id)
                        > (*current_updated_at, *current_batch_id)
                    {
                        older_wins += 1;
                    }
                }
            }
            total_shared += shared;
            total_disagreeing += disagreeing;
            total_older_wins += older_wins;
            println!(
                "table {table}: current_only={} older_only={only_older} shared={shared} \
                 disagreeing={disagreeing} older_would_win={older_wins}",
                current_branch.len() - shared
            );
        }
        println!(
            "TOTAL shared={total_shared} disagreeing={total_disagreeing} \
             older_would_win={total_older_wins}"
        );
    }

    /// Dump the DECODED rows of a table, per branch, straight out of the visible
    /// region — bypassing the query manager, and therefore its read policy.
    ///
    /// Needed because a table can be perfectly present in storage and still serve
    /// zero rows to a session-less query (`auth_pending_secrets` is policy-gated).
    /// Env: JAZZ_PROBE_PATH, JAZZ_PROBE_APP_ID, JAZZ_PROBE_SCHEMA_HASH,
    /// JAZZ_PROBE_TABLES.
    fn probe_dump_rows<S: Storage>(storage: S) {
        use crate::query_manager::types::{Schema, TableName};
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let hash_hex = std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let tables = std::env::var("JAZZ_PROBE_TABLES").expect("set JAZZ_PROBE_TABLES");

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target = crate::query_manager::types::SchemaHash::from_hex(&hash_hex)
            .expect("valid schema hash");
        let current = extractor
            .context()
            .live_schemas
            .get(&target)
            .or_else(|| extractor.context().pending_schemas.get(&target))
            .cloned()
            .expect("schema in catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let branches = sm.query_manager_mut().all_query_branches();

        for table in tables.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            println!("=== table {table} ===");
            for branch in &branches {
                let rows = storage
                    .scan_visible_region(table, branch)
                    .unwrap_or_default();
                if rows.is_empty() {
                    continue;
                }
                // Decode against the schema of the generation this branch names.
                let schema_hash = crate::query_manager::types::ComposedBranchName::parse(
                    &BranchName::new(branch.as_str()),
                )
                .map(|composed| composed.schema_hash);
                let schema = schema_hash.and_then(|short| {
                    if sm.context().current_hash.0[..6] == short.0[..6] {
                        return Some(sm.context().current_schema.clone());
                    }
                    sm.context()
                        .live_schemas
                        .iter()
                        .find(|(full, _)| full.0[..6] == short.0[..6])
                        .map(|(_, schema)| schema.clone())
                });
                for row in rows {
                    let decoded = schema
                        .as_ref()
                        .and_then(|schema| schema.get(&TableName::new(table)))
                        .and_then(|table_schema| {
                            decode_row(&table_schema.columns, &row.data.to_vec()).ok()
                        });
                    println!(
                        "  branch={branch} row_id={} state={:?} updated_at={}",
                        row.row_id, row.state, row.updated_at
                    );
                    match decoded {
                        Some(values) => {
                            for value in values {
                                println!("      {value:?}");
                            }
                        }
                        None => println!("      <could not decode against this generation>"),
                    }
                }
            }
        }
    }

    /// Dump the raw index tables whose name matches JAZZ_PROBE_INDEX (substring),
    /// with their entry keys. A backref lookup answers from
    /// `index:<table>:<column>:<branch>`; a row that is VisibleDirect in storage
    /// but missing here is findable by `_id` and invisible to the backref scan.
    fn probe_dump_index<S: Storage>(storage: S) {
        let needle = std::env::var("JAZZ_PROBE_INDEX").expect("set JAZZ_PROBE_INDEX");
        // Index raw tables carry no header, so they are invisible to a header
        // scan: an exact name (containing ':') is scanned directly.
        let names: Vec<String> = if needle.contains(':') {
            vec![needle.clone()]
        } else {
            storage
                .scan_raw_table_headers()
                .expect("scan raw table headers")
                .into_iter()
                .map(|(name, _)| name)
                .filter(|name| name.contains(&needle))
                .collect()
        };
        for name in names {
            let entries = storage.raw_table_scan_prefix(&name, "").unwrap_or_default();
            println!("index {name}: {} entries", entries.len());
            // Capped so a probe on a big table stays readable; raise it with
            // `JAZZ_PROBE_LIMIT` when the question is "what is the SHAPE of every key",
            // because a silent cap answers that question wrongly and convincingly.
            let limit = std::env::var("JAZZ_PROBE_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(40);
            for (key, _value) in entries.iter().take(limit) {
                println!("    {key}");
            }
        }
    }

    /// Full HISTORY of one row, per branch: every version, its state and its batch.
    /// A row whose head is VisibleDirect while its history carries a delete is the
    /// shape a rejected/partial delete leaves behind.
    /// Env: JAZZ_PROBE_PATH, JAZZ_PROBE_TABLE, JAZZ_PROBE_ROW_ID.
    fn probe_dump_history<S: Storage>(storage: S) {
        let table = std::env::var("JAZZ_PROBE_TABLE").expect("set JAZZ_PROBE_TABLE");
        let row_id = crate::object::ObjectId::from_uuid(
            std::env::var("JAZZ_PROBE_ROW_ID")
                .expect("set JAZZ_PROBE_ROW_ID")
                .parse::<uuid::Uuid>()
                .expect("valid row id"),
        );
        let rows = storage
            .scan_history_row_batches(&table, row_id)
            .expect("history scan");
        println!("history for {table}/{row_id}: {} versions", rows.len());
        for row in &rows {
            println!(
                "  branch={} state={:?} deleted={} kind={:?} updated_at={} batch={:?}",
                row.branch,
                row.state,
                row.is_deleted,
                row.delete_kind,
                row.updated_at,
                row.batch_id
            );
        }
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_dump_history_sqlite() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-hist-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                let to = scratch.with_file_name(format!(
                    "{}{sidecar}",
                    scratch.file_name().unwrap().to_string_lossy()
                ));
                let _ = std::fs::copy(&from, &to);
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the copied replica");
        probe_dump_history(storage);
    }

    #[test]
    #[ignore]
    fn probe_dump_history_rocksdb() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        probe_dump_history(storage);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_dump_index_sqlite() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-index-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                let to = scratch.with_file_name(format!(
                    "{}{sidecar}",
                    scratch.file_name().unwrap().to_string_lossy()
                ));
                let _ = std::fs::copy(&from, &to);
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the copied replica");
        probe_dump_index(storage);
    }

    #[test]
    #[ignore]
    fn probe_dump_index_rocksdb() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        probe_dump_index(storage);
    }

    #[test]
    #[ignore]
    fn probe_dump_rows_rocksdb() {
        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        probe_dump_rows(storage);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_dump_rows_sqlite() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-dump-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica for the probe");
        // v18 item 8, diff r24 SF2: the sidecars MUST come too. This probe reads a copy of a
        // live store, and until item 8 the engine checkpointed on every barrier, which kept the
        // main file near-current and let this get away with copying it alone. With a 30 s
        // interval the main file lags by up to an interval, and by arbitrarily more whenever a
        // reader is pinning the WAL — so a probe that skips the sidecar reads stale rows and
        // reports them as the store's contents. Its siblings in this file already do this.
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the copied replica");
        probe_dump_rows(storage);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_cross_branch_duplicates_sqlite() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch = std::env::temp_dir().join(format!(
            "jazz-duplicates-probe-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                let to = scratch.with_file_name(format!(
                    "{}{sidecar}",
                    scratch.file_name().unwrap().to_string_lossy()
                ));
                std::fs::copy(&from, &to).expect("copy sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the copied replica");
        probe_cross_branch_duplicates(storage);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn probe_table_census_sqlite() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-census-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the replica for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");
        probe_table_census(storage);
    }

    /// Presence forensics: print every `users` row's `onlineTimeUpdatedAtMs`
    /// as the store holds it. Run against a COPY of the sync server's store to
    /// tell "the server applied the heartbeat" from "the client wrote it
    /// locally and the server never took it" (defect 26).
    #[test]
    #[ignore]
    fn probe_presence_timestamps() {
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let path = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let mut storage =
            RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open copied rocksdb store");
        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let current_hash_hex =
            std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target_hash = crate::query_manager::types::SchemaHash::from_hex(&current_hash_hex)
            .expect("valid schema hash");
        let current_schema = extractor
            .context()
            .live_schemas
            .get(&target_hash)
            .or_else(|| extractor.context().pending_schemas.get(&target_hash))
            .cloned()
            .expect("the app's schema must be in the catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current_schema, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let qm = sm.query_manager_mut();
        let sub = qm.subscribe(qm.query("users").build()).expect("subscribe");
        qm.process(&mut storage);
        for (id, values) in qm.get_subscription_results(sub) {
            let stamps: Vec<String> = values
                .iter()
                .filter_map(|value| match value {
                    Value::Timestamp(ts) => Some(format!("{ts:?}")),
                    _ => None,
                })
                .collect();
            println!("users {id} timestamps={stamps:?}");
        }
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn incident_app_store_serves_the_users_row_after_rehydrate() {
        // Copy-then-open, always: SqliteStorage::open mutates unconditionally
        // (WAL pragma, CREATE TABLE, manifest insert) and has already
        // truncated two working store copies opened in place.
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch =
            std::env::temp_dir().join(format!("jazz-incident-probe-{}.sqlite", std::process::id()));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the store for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");
        probe_incident_store(storage);
    }

    /// The 2026-08-15 shrinking-list incident, third pass: run the app's chat
    /// LIST query (chat_members with a parent ref-include of chats) against a
    /// COPY of the sim app's store, rehydrated from its own catalogue. Prints
    /// every membership with how many chat rows its include resolved.
    fn probe_membership_include<S: Storage>(mut storage: S) {
        use crate::query_manager::session::Session as PolicySession;
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let current_hash_hex =
            std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");

        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target_hash = crate::query_manager::types::SchemaHash::from_hex(&current_hash_hex)
            .expect("valid schema hash");
        let current_schema = extractor
            .context()
            .live_schemas
            .get(&target_hash)
            .or_else(|| extractor.context().pending_schemas.get(&target_hash))
            .cloned()
            .expect("the app's schema must be in the catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current_schema, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");
        let qm = sm.query_manager_mut();
        println!("branches queried: {:?}", qm.all_query_branches());

        let sub = qm
            .subscribe(
                qm.query("chat_members")
                    .with_array("chat", |sub| {
                        sub.from("chats").correlate("id", "chat_members.chatId")
                    })
                    .build(),
            )
            .expect("subscribe members+include");
        qm.process(&mut storage);
        let results = qm.get_subscription_results(sub);
        println!("no-session memberships: {}", results.len());
        for (id, values) in &results {
            let includes: Vec<usize> = values
                .iter()
                .filter_map(|value| value.as_array().map(|rows| rows.len()))
                .collect();
            println!("  member {id} include_counts={includes:?}");
        }

        let session =
            PolicySession::new(&std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID"));
        let sub2 = qm
            .subscribe_with_session(
                qm.query("chat_members")
                    .with_array("chat", |sub| {
                        sub.from("chats").correlate("id", "chat_members.chatId")
                    })
                    .build(),
                Some(session.clone()),
                None,
            )
            .expect("session subscribe members+include");
        qm.process(&mut storage);
        let with_session = qm.get_subscription_results(sub2);
        println!("with-session memberships: {}", with_session.len());
        for (id, values) in &with_session {
            let includes: Vec<usize> = values
                .iter()
                .filter_map(|value| value.as_array().map(|rows| rows.len()))
                .collect();
            println!("  member {id} include_counts={includes:?}");
        }

        // The app's EXACT list shape: userId + isBanned filters on top of the
        // include, under the owner's session.
        let user_id_text = std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID");
        let sub3 = qm
            .subscribe_with_session(
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .with_array("chat", |sub| {
                        sub.from("chats").correlate("id", "chat_members.chatId")
                    })
                    .build(),
                Some(session),
                None,
            )
            .expect("session subscribe app-shaped list");
        qm.process(&mut storage);
        let app_shaped = qm.get_subscription_results(sub3);
        println!(
            "app-shaped (userId+isBanned) memberships: {}",
            app_shaped.len()
        );
        for (id, values) in &app_shaped {
            let includes: Vec<usize> = values
                .iter()
                .filter_map(|value| value.as_array().map(|rows| rows.len()))
                .collect();
            println!("  member {id} include_counts={includes:?}");
        }

        // The FULL app shape: nested includes (chat -> its member set -> each
        // member's user -> the user's handles; plus the joinable group call)
        // and the $createdAt magic-column ordering — chats.ts verbatim.
        let session2 = PolicySession::new(&user_id_text);
        let sub4 = qm
            .subscribe_with_session(
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .order_by_desc("$createdAt")
                    .with_array("chat", |sub| {
                        sub.from("chats")
                            .correlate("id", "chat_members.chatId")
                            .with_array("chat_membersViaChat", |sub| {
                                sub.from("chat_members")
                                    .correlate("chatId", "chats.id")
                                    .with_array("user", |sub| {
                                        sub.from("users")
                                            .correlate("id", "chat_members.userId")
                                            .with_array("unique_namesViaUser", |sub| {
                                                sub.from("unique_names")
                                                    .correlate("userId", "users.id")
                                            })
                                    })
                            })
                            .with_array("callsViaChat", |sub| {
                                sub.from("calls")
                                    .correlate("chatId", "chats.id")
                                    .filter_eq("callShape", Value::Text("group".into()))
                                    .filter_eq("isEnded", Value::Boolean(false))
                                    .limit(1)
                            })
                    })
                    .build(),
                Some(session2),
                None,
            )
            .expect("session subscribe full app list shape");
        qm.process(&mut storage);
        let full_shape = qm.get_subscription_results(sub4);
        println!("full-shape memberships: {}", full_shape.len());
        for (id, values) in &full_shape {
            let includes: Vec<usize> = values
                .iter()
                .filter_map(|value| value.as_array().map(|rows| rows.len()))
                .collect();
            println!("  member {id} include_counts={includes:?}");
        }

        // Bisect: which half of the full shape drops the old-branch rows?
        // (a) simple include + the $createdAt ordering
        let session3 = PolicySession::new(&user_id_text);
        let sub5 = qm
            .subscribe_with_session(
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .order_by_desc("$createdAt")
                    .with_array("chat", |sub| {
                        sub.from("chats").correlate("id", "chat_members.chatId")
                    })
                    .build(),
                Some(session3),
                None,
            )
            .expect("session subscribe orderBy bisect");
        qm.process(&mut storage);
        println!(
            "bisect orderBy+simple-include memberships: {}",
            qm.get_subscription_results(sub5).len()
        );

        // (b) nested includes, NO ordering
        let session4 = PolicySession::new(&user_id_text);
        let sub6 = qm
            .subscribe_with_session(
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .with_array("chat", |sub| {
                        sub.from("chats")
                            .correlate("id", "chat_members.chatId")
                            .with_array("chat_membersViaChat", |sub| {
                                sub.from("chat_members")
                                    .correlate("chatId", "chats.id")
                                    .with_array("user", |sub| {
                                        sub.from("users")
                                            .correlate("id", "chat_members.userId")
                                            .with_array("unique_namesViaUser", |sub| {
                                                sub.from("unique_names")
                                                    .correlate("userId", "users.id")
                                            })
                                    })
                            })
                            .with_array("callsViaChat", |sub| {
                                sub.from("calls")
                                    .correlate("chatId", "chats.id")
                                    .filter_eq("callShape", Value::Text("group".into()))
                                    .filter_eq("isEnded", Value::Boolean(false))
                                    .limit(1)
                            })
                    })
                    .build(),
                Some(session4),
                None,
            )
            .expect("session subscribe nested bisect");
        qm.process(&mut storage);
        println!(
            "bisect nested-include no-orderBy memberships: {}",
            qm.get_subscription_results(sub6).len()
        );

        // (c) chat + member set only; (d) chat + calls only;
        // (e) chat + member set + user (no handles).
        let variants: Vec<(&str, crate::query_manager::query::Query)> = vec![
            (
                "chat+membersViaChat",
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .with_array("chat", |sub| {
                        sub.from("chats")
                            .correlate("id", "chat_members.chatId")
                            .with_array("chat_membersViaChat", |sub| {
                                sub.from("chat_members").correlate("chatId", "chats.id")
                            })
                    })
                    .build(),
            ),
            (
                "chat+callsViaChat",
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .with_array("chat", |sub| {
                        sub.from("chats")
                            .correlate("id", "chat_members.chatId")
                            .with_array("callsViaChat", |sub| {
                                sub.from("calls")
                                    .correlate("chatId", "chats.id")
                                    .filter_eq("callShape", Value::Text("group".into()))
                                    .filter_eq("isEnded", Value::Boolean(false))
                                    .limit(1)
                            })
                    })
                    .build(),
            ),
            (
                "chat+members+user",
                qm.query("chat_members")
                    .filter_eq("userId", Value::Text(user_id_text.clone()))
                    .filter_eq("isBanned", Value::Boolean(false))
                    .with_array("chat", |sub| {
                        sub.from("chats")
                            .correlate("id", "chat_members.chatId")
                            .with_array("chat_membersViaChat", |sub| {
                                sub.from("chat_members")
                                    .correlate("chatId", "chats.id")
                                    .with_array("user", |sub| {
                                        sub.from("users").correlate("id", "chat_members.userId")
                                    })
                            })
                    })
                    .build(),
            ),
        ];
        for (label, query) in variants {
            let session_v = PolicySession::new(&user_id_text);
            let sub_v = qm
                .subscribe_with_session(query, Some(session_v), None)
                .expect("bisect variant subscribe");
            qm.process(&mut storage);
            let rows = qm.get_subscription_results(sub_v);
            let ids: Vec<String> = rows
                .iter()
                .map(|(id, _)| id.to_string()[..8].to_string())
                .collect();
            println!("bisect {label}: {} {ids:?}", rows.len());
        }
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn incident_app_store_serves_the_membership_include() {
        // Copy-then-open, same reason as the users probe above.
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch = std::env::temp_dir().join(format!(
            "jazz-membership-probe-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the store for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");
        probe_membership_include(storage);
    }

    /// Defect-25 write probe: the app's presence heartbeat —
    /// `db.update(users, <own id>, { onlineTimeUpdatedAtMs: now })` under the
    /// owner's session — against a COPY of the incident store, rehydrated
    /// from its own catalogue (permissions head included, so write
    /// authorization is enforced exactly as on the device). Prints the exact
    /// failure if the write does not apply, and where the row's visible
    /// versions sit per branch before and after.
    fn probe_incident_write<S: Storage>(mut storage: S) {
        use crate::query_manager::session::{Session as PolicySession, WriteContext};
        use crate::query_manager::types::Schema;
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let current_hash_hex =
            std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let user_row_id = crate::object::ObjectId::from_uuid(
            uuid::Uuid::parse_str(
                &std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID"),
            )
            .expect("valid row id"),
        );

        // Two-phase rehydrate, identical to the read probes.
        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target_hash = crate::query_manager::types::SchemaHash::from_hex(&current_hash_hex)
            .expect("valid schema hash");
        let current_schema = extractor
            .context()
            .live_schemas
            .get(&target_hash)
            .or_else(|| extractor.context().pending_schemas.get(&target_hash))
            .cloned()
            .expect("the app's schema must be in the catalogue");

        let mut sm = SchemaManager::new(SyncManager::new(), current_schema, app_id, "dev", "main")
            .expect("phase-2 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("phase-2 rehydrate");

        let branches = sm.query_manager().all_query_branches();
        println!("branches: {branches:?}");

        let dump_row = |storage: &S, label: &str| {
            for branch in &branches {
                let rows = scan_visible_row_bytes_with_storage(storage, "users", branch.as_str())
                    .unwrap_or_default();
                for row in rows.iter().filter(|row| row.row_id == user_row_id) {
                    let version = row
                        .bytes
                        .get(0x10..0x18)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()));
                    println!(
                        "{label}: users {} @ {branch} len={} version={version:?}",
                        row.row_id,
                        row.bytes.len()
                    );
                }
            }
        };
        dump_row(&storage, "before");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_millis() as u64;
        let session = PolicySession::new(user_row_id.to_string());
        let write_context = WriteContext::from_session(session);
        let result = sm.update(
            &mut storage,
            user_row_id,
            &[(
                "onlineTimeUpdatedAtMs".to_string(),
                Value::Timestamp(now_ms),
            )],
            Some(&write_context),
        );
        println!("owner heartbeat update result: {result:?}");
        dump_row(&storage, "after");

        // Negative control BEFORE the assert: the same session updating
        // ANOTHER user's row must stay denied (whereOld must still bite).
        let other_row_id = branches.iter().find_map(|branch| {
            scan_visible_row_bytes_with_storage(&storage, "users", branch.as_str())
                .unwrap_or_default()
                .into_iter()
                .map(|row| row.row_id)
                .find(|row_id| *row_id != user_row_id)
        });
        if let Some(other_row_id) = other_row_id {
            let foreign = sm.update(
                &mut storage,
                other_row_id,
                &[(
                    "onlineTimeUpdatedAtMs".to_string(),
                    Value::Timestamp(now_ms),
                )],
                Some(&write_context),
            );
            println!("foreign-row update result (must be denied): {foreign:?}");
            assert!(
                foreign.is_err(),
                "the session updated ANOTHER user's row — whereOld no longer bites"
            );
        } else {
            println!("no second users row found for the negative control");
        }

        assert!(
            result.is_ok(),
            "the owner's own heartbeat update must apply: {result:?}"
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn incident_app_store_applies_the_owner_heartbeat() {
        // Copy-then-open, same reason as the probes above.
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch = std::env::temp_dir().join(format!(
            "jazz-heartbeat-probe-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the store for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");
        probe_incident_write(storage);
    }

    /// Defect-25 write probe, DEVICE-SHAPED: the exact jazz-rn init order
    /// (SchemaManager::new with the bundle schema, catalogue rehydrate,
    /// RuntimeCore::new, persist_schema) followed by the app's heartbeat
    /// update through RuntimeCore::update — the device's actual write entry
    /// point.
    #[cfg(feature = "sqlite")]
    fn probe_incident_runtime_write(storage: SqliteStorage) {
        use crate::query_manager::session::{Session as PolicySession, WriteContext};
        use crate::query_manager::types::Schema;
        use crate::runtime_core::{NoopScheduler, RuntimeCore, VecSyncSender};
        use crate::schema_manager::{AppId, SchemaManager};
        use crate::sync_manager::SyncManager;

        let app_id =
            AppId::from_string(&std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"))
                .expect("valid app id");
        let current_hash_hex =
            std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH");
        let user_row_id = crate::object::ObjectId::from_uuid(
            uuid::Uuid::parse_str(
                &std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID"),
            )
            .expect("valid row id"),
        );

        // Extract the bundle-equivalent current schema from the catalogue.
        let mut extractor =
            SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                .expect("phase-1 schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(
            &mut extractor,
            &storage,
            app_id,
        )
        .expect("phase-1 rehydrate");
        let target_hash = crate::query_manager::types::SchemaHash::from_hex(&current_hash_hex)
            .expect("valid schema hash");
        let current_schema = extractor
            .context()
            .live_schemas
            .get(&target_hash)
            .or_else(|| extractor.context().pending_schemas.get(&target_hash))
            .cloned()
            .expect("the app's schema must be in the catalogue");

        // The jazz-rn constructor's exact order.
        let mut sm = SchemaManager::new(SyncManager::new(), current_schema, app_id, "dev", "main")
            .expect("device-shaped schema manager");
        crate::schema_manager::rehydrate_schema_manager_from_catalogue(&mut sm, &storage, app_id)
            .expect("device-shaped rehydrate");
        let mut core = RuntimeCore::new(sm, storage, NoopScheduler);
        core.set_sync_sender(Box::new(VecSyncSender::new()));
        core.persist_schema();

        let branches = core.schema_manager().query_manager().all_query_branches();
        println!("runtime branches: {branches:?}");
        let dump_row = |storage: &SqliteStorage, label: &str| {
            for branch in &branches {
                let rows = scan_visible_row_bytes_with_storage(storage, "users", branch.as_str())
                    .unwrap_or_default();
                for row in rows.iter().filter(|row| row.row_id == user_row_id) {
                    let version = row
                        .bytes
                        .get(0x10..0x18)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()));
                    println!(
                        "{label}: users {} @ {branch} len={} version={version:?}",
                        row.row_id,
                        row.bytes.len()
                    );
                }
            }
        };
        dump_row(core.storage(), "runtime before");

        // Optionally reproduce the app's live-session shape before the write:
        // session subscriptions (the users query and the full chat list
        // shape) plus a tick, the way the app reads before it heartbeats.
        if std::env::var("JAZZ_PROBE_SUBSCRIBE_FIRST").is_ok() {
            let session = PolicySession::new(user_row_id.to_string());
            let users_query = core
                .schema_manager_mut()
                .query_manager_mut()
                .query("users")
                .build();
            let sub = core.subscribe(users_query, |_| {}, Some(session.clone()));
            println!("runtime users subscription: {:?}", sub.map(|_| "ok"));

            let user_id_text = user_row_id.to_string();
            let list_query = core
                .schema_manager_mut()
                .query_manager_mut()
                .query("chat_members")
                .filter_eq("userId", Value::Text(user_id_text.clone()))
                .filter_eq("isBanned", Value::Boolean(false))
                .order_by_desc("$createdAt")
                .with_array("chat", |sub| {
                    sub.from("chats")
                        .correlate("id", "chat_members.chatId")
                        .with_array("chat_membersViaChat", |sub| {
                            sub.from("chat_members")
                                .correlate("chatId", "chats.id")
                                .with_array("user", |sub| {
                                    sub.from("users")
                                        .correlate("id", "chat_members.userId")
                                        .with_array("unique_namesViaUser", |sub| {
                                            sub.from("unique_names").correlate("userId", "users.id")
                                        })
                                })
                        })
                        .with_array("callsViaChat", |sub| {
                            sub.from("calls")
                                .correlate("chatId", "chats.id")
                                .filter_eq("callShape", Value::Text("group".into()))
                                .filter_eq("isEnded", Value::Boolean(false))
                                .limit(1)
                        })
                })
                .build();
            let sub2 = core.subscribe(list_query, |_| {}, Some(session));
            println!("runtime list subscription: {:?}", sub2.map(|_| "ok"));
            core.batched_tick();
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_millis() as u64;
        let session = PolicySession::new(user_row_id.to_string());
        let write_context = WriteContext::from_session(session);
        let result = core.update(
            user_row_id,
            vec![(
                "onlineTimeUpdatedAtMs".to_string(),
                Value::Timestamp(now_ms),
            )],
            Some(&write_context),
        );
        println!("runtime owner heartbeat update result: {result:?}");
        dump_row(core.storage(), "runtime after");

        // Negative control: the same session on another user's row stays denied.
        let other_row_id = branches.iter().find_map(|branch| {
            scan_visible_row_bytes_with_storage(core.storage(), "users", branch.as_str())
                .unwrap_or_default()
                .into_iter()
                .map(|row| row.row_id)
                .find(|row_id| *row_id != user_row_id)
        });
        if let Some(other_row_id) = other_row_id {
            let foreign = core.update(
                other_row_id,
                vec![(
                    "onlineTimeUpdatedAtMs".to_string(),
                    Value::Timestamp(now_ms),
                )],
                Some(&write_context),
            );
            println!("runtime foreign-row update result (must be denied): {foreign:?}");
            assert!(
                foreign.is_err(),
                "the session updated ANOTHER user's row through the runtime"
            );
        } else {
            println!("no second users row found for the negative control");
        }

        assert!(
            result.is_ok(),
            "the owner's heartbeat update must apply through the runtime: {result:?}"
        );
    }

    /// Defect-25 forensics: the incident store holds ~1377 users-row history
    /// batches on the new branch (the 10s heartbeats of the v16.13/v16.14
    /// sessions) that never became visible. Dump everything the publish path
    /// consults for a sample of them: row locator, batch table locators,
    /// per-generation loads, row state, local batch record, sealed
    /// submission, authoritative fate, and the family history scan.
    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn incident_app_store_stuck_heartbeat_forensics() {
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch = std::env::temp_dir().join(format!(
            "jazz-forensics-probe-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the store for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");

        let user_row_id = crate::object::ObjectId::from_uuid(
            uuid::Uuid::parse_str(
                &std::env::var("JAZZ_PROBE_ROW_ID").expect("set JAZZ_PROBE_ROW_ID"),
            )
            .expect("valid row id"),
        );
        let branch = "dev-b32dae47bbd9-main";

        // The current-generation users descriptor, for decoding batch data.
        let users_descriptor = {
            use crate::query_manager::types::Schema;
            use crate::schema_manager::{AppId, SchemaManager};
            use crate::sync_manager::SyncManager;
            let app_id = AppId::from_string(
                &std::env::var("JAZZ_PROBE_APP_ID").expect("set JAZZ_PROBE_APP_ID"),
            )
            .expect("valid app id");
            let mut extractor =
                SchemaManager::new(SyncManager::new(), Schema::new(), app_id, "dev", "main")
                    .expect("descriptor schema manager");
            crate::schema_manager::rehydrate_schema_manager_from_catalogue(
                &mut extractor,
                &storage,
                app_id,
            )
            .expect("descriptor rehydrate");
            let target_hash = crate::query_manager::types::SchemaHash::from_hex(
                &std::env::var("JAZZ_PROBE_SCHEMA_HASH").expect("set JAZZ_PROBE_SCHEMA_HASH"),
            )
            .expect("valid schema hash");
            extractor
                .context()
                .live_schemas
                .get(&target_hash)
                .or_else(|| extractor.context().pending_schemas.get(&target_hash))
                .expect("the app's schema must be in the catalogue")
                .get(&crate::query_manager::types::TableName::new("users"))
                .expect("users table")
                .columns
                .clone()
        };
        let describe_data = |data: &[u8]| -> String {
            match crate::row_format::decode_row(&users_descriptor, data) {
                Ok(values) => users_descriptor
                    .columns
                    .iter()
                    .zip(values.iter())
                    .filter(|(_, value)| !matches!(value, Value::Null))
                    .map(|(column, value)| format!("{}={value:?}", column.name.as_str()))
                    .collect::<Vec<_>>()
                    .join(" "),
                Err(err) => format!("DECODE ERR {err:?}"),
            }
        };

        println!("row locator: {:?}", storage.load_row_locator(user_row_id));

        match storage.scan_history_row_batches("users", user_row_id) {
            Ok(rows) => {
                println!("family history scan: Ok({} rows)", rows.len());
                let mut by_state: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                let mut by_author: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for row in &rows {
                    *by_state.entry(format!("{:?}", row.state)).or_default() += 1;
                    let provenance = row.row_provenance();
                    *by_author
                        .entry(format!(
                            "created_by={} updated_by={}",
                            provenance.created_by, provenance.updated_by
                        ))
                        .or_default() += 1;
                }
                println!("  states: {by_state:?}");
                println!("  authors: {by_author:?}");
                // The last five by (branch, updated_at): batch id time vs
                // provenance time vs decoded content.
                for row in rows.iter().rev().take(5) {
                    let batch_ms = u64::from_be_bytes([
                        0,
                        0,
                        row.batch_id().0[0],
                        row.batch_id().0[1],
                        row.batch_id().0[2],
                        row.batch_id().0[3],
                        row.batch_id().0[4],
                        row.batch_id().0[5],
                    ]);
                    let provenance = row.row_provenance();
                    println!(
                        "  tail: batch={} batch_ms={batch_ms} branch={} state={:?} \
                         created_at={} updated_at={} updated_by={} data_len={}\n    \
                         values: {}",
                        row.batch_id(),
                        row.branch,
                        row.state,
                        provenance.created_at,
                        provenance.updated_at,
                        provenance.updated_by,
                        row.data.len(),
                        describe_data(&row.data)
                    );
                }
            }
            Err(err) => println!("family history scan: ERR {err}"),
        }

        let batch_hex = [
            // first stuck heartbeat (12:40:17Z), one with a twin row in the
            // old generation raw table, the last stuck one (16:51:47Z), and
            // an Aug-13-era batch stored in the old generation raw table
            // under the new branch.
            "01a0047d910e773284d20dbc3a20553c",
            "01a0047e2ee778a081381e5dce3821b5",
            "01a0056390197ef094f40cf5300d81ff",
            "019ffb87202a7ed2b73d3f994a928da6",
        ];
        let hashes = [
            (
                "new",
                SchemaHash::from_hex(
                    "b32dae47bbd935a12360128d126c0208654da71704dd24e57ffb6d8c5117549b",
                )
                .unwrap(),
            ),
            (
                "old",
                SchemaHash::from_hex(
                    "53710882d8e01a0184604d82934f3ec8bb8359afec4aa326dc10ee488fc6cb59",
                )
                .unwrap(),
            ),
        ];
        for hex in batch_hex {
            let mut bytes = [0u8; 16];
            for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
                bytes[i] = u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
            }
            let batch_id = crate::row_histories::BatchId(bytes);
            println!("batch {hex}:");
            println!(
                "  batch table locator: {:?}",
                storage.load_history_row_batch_table_locator(branch, user_row_id, batch_id)
            );
            for (label, hash) in &hashes {
                match storage.load_history_row_batch_for_schema_hash(
                    "users",
                    *hash,
                    branch,
                    user_row_id,
                    batch_id,
                ) {
                    Ok(Some(row)) => println!(
                        "  {label}-gen row: state={:?} updated_at={} parents={} data_len={}",
                        row.state,
                        row.updated_at,
                        row.parents.len(),
                        row.data.len()
                    ),
                    Ok(None) => println!("  {label}-gen row: none"),
                    Err(err) => println!("  {label}-gen row: ERR {err}"),
                }
            }
            println!(
                "  local batch record: {:?}",
                storage.load_local_batch_record(batch_id).map(|record| {
                    record.map(|record| {
                        format!(
                            "mode={:?} sealed={} members={} fate={:?}",
                            record.mode,
                            record.sealed,
                            record.members.len(),
                            record.latest_fate
                        )
                    })
                })
            );
            println!(
                "  local batch row index: {:?}",
                storage.load_local_batch_row_index(batch_id).map(|index| {
                    index.map(|members| {
                        members
                            .iter()
                            .map(|member| {
                                format!(
                                    "{}@{} hash={}",
                                    member.table_name,
                                    member.branch_name,
                                    &member.schema_hash.to_string()[..12]
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
            );
            println!(
                "  sealed submission: {:?}",
                storage
                    .load_sealed_batch_submission(batch_id)
                    .map(|submission| submission.map(|submission| format!(
                        "mode={:?} target={} members={}",
                        submission.mode,
                        submission.target_branch_name,
                        submission.members.len()
                    )))
            );
            println!(
                "  authoritative fate: {:?}",
                storage.load_authoritative_batch_fate(batch_id)
            );
        }
    }

    #[cfg(feature = "sqlite")]
    #[test]
    #[ignore]
    fn incident_app_store_applies_the_owner_heartbeat_through_the_runtime() {
        // Copy-then-open, same reason as the probes above.
        let source = std::env::var("JAZZ_PROBE_PATH").expect("set JAZZ_PROBE_PATH");
        let scratch = std::env::temp_dir().join(format!(
            "jazz-runtime-heartbeat-probe-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&scratch);
        std::fs::copy(&source, &scratch).expect("copy the store for the probe");
        for sidecar in ["-wal", "-shm"] {
            let from = format!("{source}{sidecar}");
            if std::path::Path::new(&from).exists() {
                std::fs::copy(
                    &from,
                    scratch.with_file_name(format!(
                        "{}{sidecar}",
                        scratch.file_name().unwrap().to_string_lossy()
                    )),
                )
                .expect("copy the store sidecar");
            }
        }
        let storage = SqliteStorage::open(&scratch).expect("open the scratch copy");
        probe_incident_runtime_write(storage);
    }
}

// Deterministic CI witness for the defect-20 read fallbacks: a physical
// sqlite store whose row locator lies about the raw table holding the row.
// The env-driven incident probes above validate against real store copies but
// cannot run in CI; this is the hermetic red→green twin.
#[cfg(all(test, feature = "sqlite"))]
mod split_locator_tests {
    use super::*;
    use crate::object::BranchName;
    use crate::query_manager::types::{
        ColumnDescriptor, ColumnType, ComposedBranchName, RowDescriptor, Schema, TableName,
    };
    use crate::row_histories::{RowState, StoredRowBatch, apply_row_batch};

    fn users_schema() -> Schema {
        let mut schema = Schema::new();
        schema.insert(
            TableName::new("users"),
            RowDescriptor::new(vec![ColumnDescriptor::new("name", ColumnType::Text)]).into(),
        );
        schema
    }

    #[test]
    fn a_visible_row_behind_a_lying_locator_is_recovered_and_deletable() {
        let path = std::env::temp_dir().join(format!(
            "jazz-split-locator-test-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut storage = SqliteStorage::open(&path).expect("temp sqlite store opens");
        // v18 item 5: the fixture (schema, row, honest read, poison) is
        // `crate::test_support::poisoned_split_row`, shared with the RocksDB twin G-D1r so
        // both stores are gated on ONE shape. Unchanged in behaviour, including the
        // positive control, which now runs inside the helper.
        let (branch, row_id, schema_hash) = crate::test_support::poisoned_split_row(&mut storage);
        // The read fallback must recover the row from the sibling raw table.
        let recovered =
            load_visible_region_row_bytes_with_storage(&storage, "users", branch.as_str(), row_id)
                .expect("read succeeds");
        assert!(
            recovered.is_some(),
            "a visible row behind a lying locator was not recovered — the defect-20 \
             read fallback regressed"
        );

        // And the delete must reach the raw table that physically holds the
        // row, or the recovered row becomes an undeletable ghost. Defect 27
        // replaced the single-locator resolver with a measurement: a delete goes
        // to every family that actually holds the row, so a lying locator can
        // neither misdirect it nor hide a second head from it.
        let holders = visible_row_raw_tables_holding(&storage, "users", branch.as_str(), row_id)
            .expect("holder measurement succeeds");
        assert_eq!(
            holders,
            vec![
                visible_row_raw_table_id("users", schema_hash)
                    .raw_table_name()
                    .to_string()
            ],
            "the delete must target the raw table holding the bytes, not the locator's lie"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// The locator must not move on a FAILED apply. The aligned locator rides
    /// the request, but persists only after `apply_row_batch_with_context`
    /// succeeds — a batch that fails validation (a routine ParentNotFound on
    /// out-of-order delivery) must leave the stored locator untouched, or
    /// batches stored without exact locators under the old hash become
    /// unreachable (parent checks, tier patches, replay dedup, the
    /// USING-policy old-content load).
    #[test]
    fn a_failed_apply_leaves_the_row_locator_untouched() {
        let path = std::env::temp_dir().join(format!(
            "jazz-locator-flip-test-{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut storage = SqliteStorage::open(&path).expect("temp sqlite store opens");

        let schema = users_schema();
        let schema_hash = crate::test_support::persist_test_schema(&mut storage, &schema);
        let branch = ComposedBranchName::new("dev", schema_hash, "main").to_branch_name();

        // The stored locator names a hash the resolve ladder will NOT pick —
        // alignment would trigger if the apply succeeded.
        let stamped_hash = SchemaHash::from_bytes([7u8; 32]);
        let row_id = ObjectId::new();
        storage
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".to_string().into(),
                    origin_schema_hash: Some(stamped_hash),
                }),
            )
            .expect("locator persists");

        let descriptor = schema
            .get(&TableName::new("users"))
            .expect("users descriptor")
            .columns
            .clone();
        let data = crate::row_format::encode_row(
            &descriptor,
            &[crate::query_manager::types::Value::Text("orphan".into())],
        )
        .expect("row encodes");
        // A parent batch id that does not exist anywhere — the apply must fail.
        let missing_parent = crate::row_histories::BatchId([9u8; 16]);
        let orphan = StoredRowBatch::new(
            row_id,
            branch.as_str(),
            vec![missing_parent],
            data.clone(),
            crate::metadata::RowProvenance::for_insert(row_id.to_string(), 1_000),
            std::collections::HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        let failed = apply_row_batch(
            &mut storage,
            row_id,
            &BranchName::new(branch.as_str()),
            orphan,
            &[],
        );
        assert!(
            failed.is_err(),
            "the orphan batch must fail its parent check, or this gates nothing"
        );
        let locator = storage
            .load_row_locator(row_id)
            .expect("locator readable")
            .expect("locator still present");
        assert_eq!(
            locator.origin_schema_hash,
            Some(stamped_hash),
            "a FAILED apply moved the row locator — batches stored without exact \
             locators under the stamped hash are now unreachable"
        );

        // The happy half: a valid batch aligns the locator to the resolved hash.
        let rooted = StoredRowBatch::new(
            row_id,
            branch.as_str(),
            Vec::new(),
            data,
            crate::metadata::RowProvenance::for_insert(row_id.to_string(), 1_100),
            std::collections::HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        apply_row_batch(
            &mut storage,
            row_id,
            &BranchName::new(branch.as_str()),
            rooted,
            &[],
        )
        .expect("the rooted batch applies");
        let locator = storage
            .load_row_locator(row_id)
            .expect("locator readable")
            .expect("locator present");
        assert_eq!(
            locator.origin_schema_hash,
            Some(schema_hash),
            "a successful apply must align the locator to the resolved hash"
        );

        let _ = std::fs::remove_file(&path);
    }
}
