import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.join(root, '.astro-public');

export async function preparePublic() {
  // Copy only public assets, never the repository root or draft/source folders.
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  for (const entry of ['assets', 'toy-examples', 'favicon.ico', 'googleeb4726a2bd6cb6e7.html']) {
    await cp(path.join(root, entry), path.join(output, entry), {
      recursive: true,
      filter: source => !path.relative(root, source).split(path.sep).some(part => /^_?drafts?$/i.test(part) || part.startsWith('.')),
    });
  }
  for (const app of ['itascanner', 'itaview']) {
    await cp(path.join(root, 'public', app), path.join(output, app), {
      recursive: true,
      filter: source => !path.basename(source).startsWith('.'),
    });
  }
  await writeFile(path.join(output, '.nojekyll'), '');
  await writeFile(path.join(output, 'robots.txt'), 'User-agent: *\nAllow: /\nDisallow: /404.html\n\nSitemap: https://ita9naiwa.github.io/sitemap.xml\n');
}
