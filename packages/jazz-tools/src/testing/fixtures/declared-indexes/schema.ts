import { col } from "../../../dsl.js";
import { defineApp, defineTable } from "../../../typed-app.js";

export const app = defineApp({
  chats: {
    title: col.string(),
  },
  messages: defineTable({
    chatId: col.ref("chats"),
    text: col.string().optional(),
    createdAtMs: col.timestamp(),
  })
    .compositeIndex("chatId", "createdAtMs")
    .trigramIndex("chatId", "text"),
});
