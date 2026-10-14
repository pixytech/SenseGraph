// Copies the web build of the engine next to the static assets so the
// workers can load it by URL. Works on Windows, macOS and Linux.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const from = resolve(root, "apps/web/wasm-pkg/sensegraph_core_bg.wasm");
const to = resolve(root, "apps/web/public/sensegraph_core_bg.wasm");
mkdirSync(dirname(to), { recursive: true });
copyFileSync(from, to);
console.log("copied sensegraph_core_bg.wasm to apps/web/public");
