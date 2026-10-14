import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// BASE_PATH is set by the Pages workflow, for example "/apps/sensegraph/".
export default defineConfig({
  base: process.env.BASE_PATH || "/",
  plugins: [react()],
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["sensegraph_core"],
  },
});
