/**
 * Indexes an app declares next to its tables (`table.compositeIndex(...)`,
 * `table.trigramIndex(...)`). They stay out of the hashed schema — declaring one
 * starts no new schema generation — and travel with the permissions instead: every
 * store that applies the permissions head builds and maintains them.
 *
 * - `composite: [first, second]` — the rows of one `first` value in `second` order,
 *   so a query filtered on `first`, ordered by `second` and limited reads its page
 *   instead of every row of that value.
 * - `trigram: [scope, text]` — case-insensitive substring search on `text` within
 *   one `scope` value, so `contains` reads its candidates instead of the scope.
 *
 * This is the shape `POST /admin/permissions` takes and `GET /admin/permissions`
 * returns.
 */
export interface TableDeclaredIndexes {
  composite?: [string, string][];
  trigram?: [string, string][];
}

export type DeclaredIndexes = Record<string, TableDeclaredIndexes>;
