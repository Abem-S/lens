import basicSsl from "@vitejs/plugin-basic-ssl";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Phone cameras need a secure context, so `LENS_HTTPS=1` serves the dev server over
// self-signed HTTPS on the LAN (accept the warning once on the phone).
const https = process.env.LENS_HTTPS === "1";

export default defineConfig({
  plugins: [react(), ...(https ? [basicSsl()] : [])],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 900 },
  test: { environment: "jsdom" },
});
