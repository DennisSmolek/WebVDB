// Loads the spike's TypeScript modules (and, through them, our packages'
// TS sources) from a plain node script, with no build step and no new
// dependency: Vite (a root devDependency) in SSR middleware mode transpiles
// on the fly and resolves `.js` specifiers to `.ts` sources.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

export const SPIKE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const REPO_ROOT = path.resolve(SPIKE_DIR, "../..");

/** Returns { load(relPathFromSpikeDir), close() }. */
export async function tsLoader() {
  const vite = await createServer({
    root: REPO_ROOT,
    configFile: false,
    logLevel: "error",
    appType: "custom",
    server: { middlewareMode: true, hmr: false, watch: null },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  return {
    load: (rel) => vite.ssrLoadModule("/" + path.relative(REPO_ROOT, path.join(SPIKE_DIR, rel))),
    close: () => vite.close(),
  };
}
