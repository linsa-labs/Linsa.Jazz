/**
 * Transform WASM row results to typed TypeScript objects.
 */

import type { Value as WasmValue, WasmRow, WasmSchema } from "../drivers/types.js";
import type { ColumnType } from "../drivers/types.js";
import { relationsForSchema, type Relation } from "../codegen/relation-analyzer.js";
import { isProvenanceMagicTimestampColumn, magicColumnType } from "../magic-columns.js";
import { normalizeIncludeEntries, type NormalizedIncludeSpec } from "./query-builder-shape.js";
import { resolveSelectedColumns } from "./select-projection.js";

export type { WasmValue };

export interface IncludeSpec {
  [relationName: string]: unknown;
}

type BaseColumn = { name: string; columnType: ColumnType };

type IncludePlan = {
  relation: Relation;
  nested: IncludePlan[];
  projection?: readonly string[];
  /** The included table's columns under `projection`, in the order its row values arrive. */
  baseColumns: BaseColumn[];
};

/**
 * Everything about a query's output shape that does not depend on the row being
 * transformed: the columns of the output table and the include tree resolved against
 * the schema's relations. Resolving it walks every foreign key of the schema
 * (`analyzeRelations`), so it is built once per query, never once per row.
 */
type RowPlan = {
  baseColumns: BaseColumn[];
  includePlans: IncludePlan[];
};

function resolveBaseColumns(
  tableName: string,
  schema: WasmSchema,
  projection?: readonly string[],
): BaseColumn[] {
  const table = schema[tableName];
  if (!table) {
    throw new Error(`Unknown table "${tableName}" in schema`);
  }

  return resolveSelectedColumns(tableName, schema, projection)
    .map((columnName) => {
      const magicType = magicColumnType(columnName);
      if (magicType) {
        return { name: columnName, columnType: magicType };
      }
      const column = table.columns.find((candidate) => candidate.name === columnName);
      return column ? { name: column.name, columnType: column.column_type } : null;
    })
    .filter((column): column is BaseColumn => column !== null);
}

function toByteArray(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }

  if (Array.isArray(value)) {
    // Indexed loop, validating as it copies: this runs once per byte of every Bytea
    // read on React Native (rows arrive as JSON arrays of integers there), so the
    // `.map()` version's per-element closure call and intermediate array were the
    // dominant cost of megabyte reads.
    const length = value.length;
    const bytes = new Uint8Array(length);
    for (let i = 0; i < length; i++) {
      const entry: unknown = value[i];
      if (typeof entry !== "number" || !Number.isInteger(entry) || entry < 0 || entry > 255) {
        throw new Error("Invalid Bytea array value. Expected integers in range 0..255.");
      }
      bytes[i] = entry;
    }
    return bytes;
  }

  throw new Error("Invalid Bytea value. Expected Uint8Array or byte array.");
}

function buildIncludePlans(
  tableName: string,
  includes: NormalizedIncludeSpec,
  relationsByTable: Map<string, Relation[]>,
  schema: WasmSchema,
): IncludePlan[] {
  const relations = relationsByTable.get(tableName) || [];
  const plans: IncludePlan[] = [];

  for (const [relationName, spec] of Object.entries(includes)) {
    const relation = relations.find((candidate) => candidate.name === relationName);
    if (!relation) {
      throw new Error(`Unknown relation "${relationName}" on table "${tableName}"`);
    }

    const nested = buildIncludePlans(relation.toTable, spec.includes, relationsByTable, schema);
    const projection = spec.select.length > 0 ? spec.select : undefined;

    plans.push({
      relation,
      nested,
      projection,
      baseColumns: resolveBaseColumns(relation.toTable, schema, projection),
    });
  }

  return plans;
}

function transformIncludedValue(value: WasmValue, plan: IncludePlan): unknown {
  if (value.type !== "Array") {
    return unwrapValue(value);
  }

  const rows = value.value.map((entry) => {
    if (entry.type !== "Row") {
      return unwrapValue(entry);
    }
    // Row id is carried in the struct's `id` field
    const rowId = entry.value.id;
    const columnValues = entry.value.values;
    return transformRowValues(columnValues, plan.baseColumns, plan.nested, rowId);
  });

  return plan.relation.isArray ? rows : (rows[0] ?? null);
}

function transformRowValues(
  values: WasmValue[],
  baseColumns: BaseColumn[],
  includePlans: IncludePlan[],
  rowId?: string,
): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  if (rowId !== undefined) {
    obj.id = rowId;
  }

  for (let i = 0; i < baseColumns.length; i++) {
    const col = baseColumns[i];
    if (!col) continue;
    const value = values[i];
    if (value !== undefined) {
      obj[col.name] = unwrapValue(value, col.columnType, col.name);
    }
  }

  for (let i = 0; i < includePlans.length; i++) {
    const value = values[baseColumns.length + i];
    if (value === undefined) continue;
    const plan = includePlans[i];
    if (!plan) continue;
    obj[plan.relation.name] = transformIncludedValue(value, plan);
  }

  return obj;
}

function buildRowPlan(
  schema: WasmSchema,
  tableName: string,
  includes: IncludeSpec,
  projection: readonly string[] | undefined,
): RowPlan {
  if (!schema[tableName]) {
    throw new Error(`Unknown table "${tableName}" in schema`);
  }

  const includePlans =
    Object.keys(includes).length === 0
      ? []
      : buildIncludePlans(
          tableName,
          normalizeIncludeEntries(includes),
          relationsForSchema(schema),
          schema,
        );

  return { baseColumns: resolveBaseColumns(tableName, schema, projection), includePlans };
}

function timestampToDate(value: number, columnName?: string): Date {
  if (columnName && isProvenanceMagicTimestampColumn(columnName)) {
    return new Date(Math.trunc(value / 1_000));
  }
  return new Date(value);
}

export function unwrapValue(v: WasmValue, columnType?: ColumnType, columnName?: string): unknown {
  switch (v.type) {
    case "Text":
      if (columnType?.type === "Json") {
        try {
          return JSON.parse(v.value);
        } catch (error) {
          throw new Error(
            `Invalid stored JSON value: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
      return v.value;
    case "Uuid":
      return v.value;
    case "Boolean":
      return v.value;
    case "Integer":
    case "BigInt":
    case "Double":
      return v.value;
    case "Timestamp":
      return timestampToDate(v.value, columnName);
    case "Bytea":
      return toByteArray((v as { value: unknown }).value);
    case "Null":
      return null;
    case "Array":
      if (columnType?.type === "Array") {
        return v.value.map((entry) => unwrapValue(entry, columnType.element));
      }
      return v.value.map((entry) => unwrapValue(entry));
    case "Row":
      if (columnType?.type === "Row") {
        return v.value.values.map((entry, index) =>
          unwrapValue(entry, columnType.columns[index]?.column_type),
        );
      }
      return v.value.values.map((entry) => unwrapValue(entry));
  }
}

/**
 * Transform WasmRow[] to typed objects using schema column order.
 *
 * @param rows Array of WasmRow results from query
 * @param schema WasmSchema containing table definitions
 * @param tableName Name of the table being queried
 * @param includes Include tree from QueryBuilder._build() (if any)
 * @returns Array of typed objects with named properties
 */
export function transformRows<T>(
  rows: WasmRow[],
  schema: WasmSchema,
  tableName: string,
  includes: IncludeSpec = {},
  projection?: readonly string[],
): T[] {
  const plan = buildRowPlan(schema, tableName, includes, projection);

  return rows.map((row) => {
    return transformRowValues(
      row.values as WasmValue[],
      plan.baseColumns,
      plan.includePlans,
      row.id,
    ) as T;
  });
}

export function transformRow<T>(
  row: WasmRow,
  schema: WasmSchema,
  tableName: string,
  includes: IncludeSpec = {},
  projection?: readonly string[],
): T {
  const transformed = transformRows<T>([row], schema, tableName, includes, projection)[0];
  if (transformed === undefined) {
    throw new Error(`Failed to transform row for table "${tableName}"`);
  }
  return transformed;
}

/**
 * A row transformer for one query: the same output as `transformRow` called with these
 * arguments, with the query's shape resolved once instead of on every row.
 *
 * A live subscription transforms rows one at a time for as long as it lives, and
 * `transformRow` re-derives the schema's whole relation map for each of them. The shape is
 * resolved on the first row rather than here, so a query whose shape does not resolve
 * fails where `transformRow` would have failed: when a row arrives.
 */
export function createRowTransformer<T>(
  schema: WasmSchema,
  tableName: string,
  includes: IncludeSpec = {},
  projection?: readonly string[],
): (row: WasmRow) => T {
  let plan: RowPlan | undefined;

  return (row) => {
    plan ??= buildRowPlan(schema, tableName, includes, projection);
    return transformRowValues(
      row.values as WasmValue[],
      plan.baseColumns,
      plan.includePlans,
      row.id,
    ) as T;
  };
}
