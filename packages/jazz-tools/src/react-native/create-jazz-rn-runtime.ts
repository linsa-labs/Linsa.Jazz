import type { WasmSchema } from "../drivers/types.js";
import type { DurabilityTier } from "../runtime/client.js";
import { type JazzRnRuntimeBinding, JazzRnRuntimeAdapter } from "./jazz-rn-runtime-adapter.js";
import { getJazzRnSync } from "./jazz-rn-loader.js";

export interface CreateJazzRnRuntimeOptions {
  schema: WasmSchema;
  appId: string;
  env?: string;
  userBranch?: string;
  tier?: DurabilityTier;
  dataPath?: string;
  /** See `DbConfig.releaseDeclaredIndexes`. */
  releaseDeclaredIndexes?: boolean;
}

export function createJazzRnRuntime(options: CreateJazzRnRuntimeOptions): JazzRnRuntimeAdapter {
  const jazzRn = getJazzRnSync();
  const runtime = new jazzRn.jazz_rn.RnRuntime(
    JSON.stringify(options.schema),
    options.appId,
    options.env ?? "dev",
    options.userBranch ?? "main",
    options.tier,
    options.dataPath,
    options.releaseDeclaredIndexes ?? false,
  );

  return new JazzRnRuntimeAdapter(runtime as unknown as JazzRnRuntimeBinding, options.schema);
}
