import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Relative base: the built site works at a domain root (Vercel, Netlify, Cloudflare Pages) as well as
// in a sub-folder (GitHub Pages). The v1 page is built alongside as /legacy/kart-lap-lab.html.
export default defineConfig({
  base: './',
  worker: { format: 'es' },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      input: { main: resolve(__dirname, 'index.html'), v1: resolve(__dirname, 'legacy/kart-lap-lab.html') },
    },
  },
});
