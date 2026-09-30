import { defineConfig } from 'astro/config';
import { preparePublic } from './scripts/prepare-public.mjs';
import { copyFile, mkdir, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://ita9naiwa.github.io',
  output: 'static',
  publicDir: './.astro-public',
  trailingSlash: 'ignore',
  build: { format: 'file' },
  vite: { plugins: [tailwindcss()] },
  integrations: [{
    name: 'published-assets',
    hooks: {
      'astro:config:setup': () => preparePublic(),
      'astro:build:done': async ({ dir }) => {
        // Keep old /page2/ links working alongside Astro's /page2.html pages.
        const output = fileURLToPath(dir);
        for (const name of await readdir(output)) {
          if (!/^page\d+\.html$/.test(name)) continue;
          const directory = path.join(output, name.replace(/\.html$/, ''));
          await mkdir(directory, { recursive: true });
          await copyFile(path.join(output, name), path.join(directory, 'index.html'));
        }
      },
    },
  }],
});
