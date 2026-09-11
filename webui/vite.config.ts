import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    watch: {
      // Avoid compiling partial imports while an editor is still writing a file.
      awaitWriteFinish: { stabilityThreshold: 200, pollInterval: 50 },
    },
    proxy: {
      '/api': 'http://127.0.0.1:8787',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        onlyExplicitManualChunks: true,
        manualChunks(id) {
          const path = id.replace(/\\/g, '/')
          if (!path.includes('/node_modules/')) return
          if (/\/node_modules\/(react|react-dom|scheduler)\//.test(path)) return 'react-vendor'
          if (/\/node_modules\/@xyflow\//.test(path)) return 'flow-vendor'
          if (/\/node_modules\/d3-[^/]+\//.test(path)) return 'd3-vendor'
          if (/\/node_modules\/(recharts|recharts-scale|react-smooth|victory-vendor)\//.test(path)) return 'charts-vendor'
        },
      },
    },
  },
})
