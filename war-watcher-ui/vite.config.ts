import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import svgr from "vite-plugin-svgr";

// Vite config for React + SVGR
export default defineConfig({
  plugins: [react(), svgr()],
  server: {
    host: true,          // allow LAN access if needed
    port: 5173,
    strictPort: true,
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
});
