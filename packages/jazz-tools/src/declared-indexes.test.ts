import { describe, expect, it } from "vitest";
import { col } from "./dsl.js";
import { defineApp, defineSliceableApp, defineTable } from "./typed-app.js";

const columns = () => ({
  chatId: col.ref("chats"),
  text: col.string().optional(),
  createdAtMs: col.timestamp(),
});

describe("declared indexes", () => {
  it("collects what the tables declare onto the app", () => {
    const app = defineApp({
      chats: { title: col.string() },
      messages: defineTable(columns())
        .compositeIndex("chatId", "createdAtMs")
        .trigramIndex("chatId", "text"),
    });

    expect(app.declaredIndexes).toEqual({
      messages: {
        composite: [["chatId", "createdAtMs"]],
        trigram: [["chatId", "text"]],
      },
    });
  });

  it("stays out of the runtime schema, so declaring one starts no new schema", () => {
    const plain = defineApp({
      chats: { title: col.string() },
      messages: defineTable(columns()),
    });
    const declared = defineApp({
      chats: { title: col.string() },
      messages: defineTable(columns()).compositeIndex("chatId", "createdAtMs"),
    });

    expect(declared.wasmSchema).toEqual(plain.wasmSchema);
    expect(plain.declaredIndexes).toEqual({});
  });

  it("survives indexOnly in either order", () => {
    const before = defineTable(columns())
      .indexOnly(["chatId"])
      .compositeIndex("chatId", "createdAtMs");
    const after = defineTable(columns())
      .compositeIndex("chatId", "createdAtMs")
      .indexOnly(["chatId"]);

    for (const table of [before, after]) {
      expect(table.indexedColumns).toEqual(["chatId"]);
      expect(table.declaredIndexes).toEqual({ composite: [["chatId", "createdAtMs"]] });
    }
  });

  it("refuses a column the table does not have", () => {
    expect(() =>
      defineTable(columns()).compositeIndex("chatId", "missing" as "createdAtMs"),
    ).toThrow(/unknown column "missing"/);
    expect(() => defineTable(columns()).trigramIndex("nope" as "chatId", "text")).toThrow(
      /unknown column "nope"/,
    );
  });

  it("is on a sliceable app too, for the whole schema", () => {
    const app = defineSliceableApp({
      chats: { title: col.string() },
      messages: defineTable(columns()).trigramIndex("chatId", "text"),
    });

    expect(app.declaredIndexes).toEqual({ messages: { trigram: [["chatId", "text"]] } });
    expect(app.slice("chats").declaredIndexes).toEqual(app.declaredIndexes);
  });
});
