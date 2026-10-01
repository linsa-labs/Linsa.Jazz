/**
 * Tests for relation-analyzer.
 */

import { describe, it, expect } from "vitest";
import { analyzeRelations, relationsForSchema } from "./relation-analyzer.js";
import type { WasmSchema } from "../drivers/types.js";

function schemaOfUsersAndTodos(): WasmSchema {
  return {
    users: {
      columns: [{ name: "name", column_type: { type: "Text" }, nullable: false }],
    },
    todos: {
      columns: [
        { name: "title", column_type: { type: "Text" }, nullable: false },
        { name: "owner_id", column_type: { type: "Uuid" }, nullable: false, references: "users" },
      ],
    },
  };
}

const relationNames = (relations: Map<string, { name: string }[]>, table: string) =>
  (relations.get(table) ?? []).map((relation) => relation.name);

describe("relationsForSchema", () => {
  it("gives what analyzeRelations gives", () => {
    const schema = schemaOfUsersAndTodos();
    expect(relationsForSchema(schema)).toEqual(analyzeRelations(schema));
  });

  it("analyses a schema object once", () => {
    const schema = schemaOfUsersAndTodos();
    const first = relationsForSchema(schema);
    expect(relationsForSchema(schema)).toBe(first);
    // `analyzeRelations` builds a new map on every call; this is what the memo saves.
    expect(analyzeRelations(schema)).not.toBe(first);
  });

  it("keeps one analysis per schema object, not per schema content", () => {
    const one = schemaOfUsersAndTodos();
    const other = schemaOfUsersAndTodos();
    expect(relationsForSchema(other)).not.toBe(relationsForSchema(one));
    expect(relationsForSchema(other)).toEqual(relationsForSchema(one));
  });

  it("analyses again once the object has another column", () => {
    const schema = schemaOfUsersAndTodos();
    const before = relationsForSchema(schema);
    expect(relationNames(before, "users")).toEqual(["todosViaOwner"]);

    schema.todos!.columns.push({
      name: "reviewer_id",
      column_type: { type: "Uuid" },
      nullable: true,
      references: "users",
    });

    const after = relationsForSchema(schema);
    expect(after).not.toBe(before);
    expect(relationNames(after, "todos")).toEqual(["owner", "reviewer"]);
    expect(relationNames(after, "users")).toEqual(["todosViaOwner", "todosViaReviewer"]);
    expect(relationsForSchema(schema)).toBe(after);
  });

  it("analyses again once the object has another table", () => {
    const schema = schemaOfUsersAndTodos();
    const before = relationsForSchema(schema);

    schema.notes = {
      columns: [
        { name: "todo_id", column_type: { type: "Uuid" }, nullable: false, references: "todos" },
      ],
    };

    const after = relationsForSchema(schema);
    expect(after).not.toBe(before);
    expect(relationNames(after, "notes")).toEqual(["todo"]);
    expect(relationNames(after, "todos")).toEqual(["owner", "notesViaTodo"]);
  });
});
