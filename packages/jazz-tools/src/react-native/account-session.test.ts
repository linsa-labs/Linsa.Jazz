import { describe, expect, it } from "vitest";
import { serializeSchemaSource } from "../drivers/schema-wire.js";
import type { WasmSchema } from "../drivers/types.js";
import type { ClientMigrations } from "../runtime/client-migrations.js";
import { nativeAccountSchemaSource } from "./account-session.js";

const schema = {
  todos: { columns: [{ name: "title", column_type: { type: "Text" }, nullable: false }] },
} as unknown as WasmSchema;

const migrations: ClientMigrations = {
  schemas: [{ hash: "a", schema: { tables: {} } }],
  migrations: [{ fromHash: "a", toHash: "b", forward: [] }],
};

describe("the schema source a native scope opens", () => {
  it("is the app's tables when the app ships no migrations", () => {
    expect(nativeAccountSchemaSource(schema, undefined)).toBe(serializeSchemaSource(schema));
  });

  it("carries the app's migrations beside its tables", () => {
    expect(JSON.parse(nativeAccountSchemaSource(schema, migrations))).toEqual({
      ...JSON.parse(serializeSchemaSource(schema)),
      migrations,
    });
  });
});
