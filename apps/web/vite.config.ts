import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// BASE_PATH is set by the Pages workflow, for example "/SenseGraph/".
export default defineConfig({
  base: process.env.BASE_PATH || "/",
  plugins: [react()],
  // The engine binary is served from a fixed file name, so its URL carries a build id.
  // Without it a browser can pair a new script with a cached older binary.
  define: {
    __WASM_VERSION__: JSON.stringify(Date.now().toString(36)),
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["sensegraph_core"],
  },
});
