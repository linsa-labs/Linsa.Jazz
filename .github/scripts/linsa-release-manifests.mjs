#!/usr/bin/env node
// LINSA FORK: point the packages of one GitHub Release at each other.
//
//   node .github/scripts/linsa-release-manifests.mjs <release-base-url> <package-dir>...
//
// Consumers install these packages by Release URL. A dependency on another
// package of the same release must name that release's tarball: left as
// `workspace:*` (or as the shared version) it resolves from npm, and a
// consumer silently gets upstream's build of it, native binaries included
// (jazz-rn-ios / jazz-rn-android, the jazz-napi platform packages).
//
// Every package in the list becomes the asset `<name>.tgz` under the base
// URL; its manifest's dependencies and optionalDependencies on any listed
// package are rewritten to that URL. Peer ranges and any other `workspace:`
// range become the exact version. Prints `<dir>\t<asset>` per package.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [base, ...dirs] = process.argv.slice(2);
if (!base || !/^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/releases\/download\/[\w.-]+$/.test(base))
  throw new Error("usage: linsa-release-manifests.mjs <release-base-url> <package-dir>...");
if (dirs.length === 0) throw new Error("no package directories");

const asset = (name) => name.replace(/^@/, "").replace(/\//g, "__");
const read = (dir) => JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
const released = new Map(dirs.map((dir) => [read(dir).name, `${base}/${asset(read(dir).name)}.tgz`]));
if (released.size !== dirs.length) throw new Error("two package directories share a name");

for (const dir of dirs) {
  const pkg = read(dir);
  for (const field of ["dependencies", "optionalDependencies", "devDependencies", "peerDependencies"]) {
    const deps = pkg[field];
    if (!deps) continue;
    for (const [name, range] of Object.entries(deps)) {
      if (field !== "peerDependencies" && field !== "devDependencies" && released.has(name))
        deps[name] = released.get(name);
      else if (typeof range === "string" && range.startsWith("workspace:")) deps[name] = pkg.version;
    }
  }
  writeFileSync(join(dir, "package.json"), `${JSON.stringify(pkg, null, 2)}\n`);
  console.log(`${dir}\t${asset(pkg.name)}`);
}
