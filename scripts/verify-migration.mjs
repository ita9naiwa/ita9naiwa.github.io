// Run after `npm run build`; verifies the deployable artifact, not just source filters.
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join, relative, resolve } from 'node:path';
import { parse } from 'yaml';

const root = resolve(import.meta.dirname, '..');
const dist = join(root, 'dist');
const origin = 'https://ita9naiwa.github.io';
assert(existsSync(dist), 'dist is missing: run npm run build first');
function files(directory) {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? files(path) : [path];
  });
}
function outputFor(url) {
  const path = join(dist, decodeURIComponent(new URL(url, origin).pathname));
  return existsSync(path) && statSync(path).isFile() ? path : join(path, 'index.html');
}
function requireRoute(url) {
  const path = outputFor(url);
  assert(existsSync(path), `Missing published route: ${url}`);
  return readFileSync(path, 'utf8');
}
const artifactFiles = files(dist);
const forbidden = /^(?:_drafts?|drafts|_posts|_site|_includes|_layouts|_sass|node_modules|vendor|docs|tools|scripts|src|test|일기_저장소|\.git|\.github)(?:\/|$)|(?:^|\/)_drafts?(?:\/|$)/;
for (const path of artifactFiles) {
  const name = relative(dist, path);
  assert(!forbidden.test(name), `Source/private directory published: ${name}`);
  assert(!/^(?:[1-4]\.md|push\.sh|run\.m|Gemfile(?:\.lock)?|package(?:-lock)?\.json|_config\.yml)$/.test(name), `Source/private file published: ${name}`);
  assert(!/\.map$/.test(name), `Source map published: ${name}`);
}

for (const url of ['/', '/about.html', '/archive.html', '/ml.html', '/mlsys.html', '/recsys.html', '/journals.html', '/paper_read.html', '/rl.html', '/404.html', '/feed.xml', '/sitemap.xml', '/robots.txt', '/favicon.ico', '/googleeb4726a2bd6cb6e7.html', '/toy-examples/temp2.html']) requireRoute(url);
// Regression cases: explicit dates, category case/spaces, Unicode, punctuation and double extensions.
for (const url of ['/tensorflow/2017/11/26/how-to-tensorflow-C++.html', '/통계/2019/06/07/EM.html', '/numeric calculation/2018/11/10/Einsum.html', '/recsys/2017/11/25/SGD-versus-ALS.html', '/일기/2017/11/25/first-writing.html', '/productivity/2018/06/20/productivity_tools_i_use.md.html']) requireRoute(url);

// If the old local Jekyll build exists, every dated page remains directly reachable.
const oldRoutes = files(join(root, '_site')).map(path => relative(join(root, '_site'), path)).filter(path => /\/\d{4}\/\d{2}\/\d{2}\/[^/]+\.html$/.test(`/${path}`));
for (const path of oldRoutes) requireRoute(`/${path}`);

const sources = files(join(root, '_posts')).filter(path => /\.(?:md|markdown|html)$/i.test(path)).map(path => {
  const raw = readFileSync(path, 'utf8');
  const frontmatter = raw.match(/^\uFEFF?---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  if (!frontmatter) return null;
  const metadata = parse(frontmatter[1]);
  const match = basename(path).match(/^(\d{4})-(\d{1,2})-(\d{1,2})-(.+)\.(?:md|markdown|html)$/i);
  assert(match, `Unrecognized post filename: ${path}`);
  const date = String(metadata.date || match.slice(1, 4).join('-')).match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  assert(date, `Unrecognized post date: ${path}`);
  const list = value => value == null ? [] : Array.isArray(value) ? value : [value];
  const categories = [...new Set([...list(metadata.category), ...list(metadata.categories)])].map(value => String(value).toLowerCase());
  const slug = (metadata.slug || match[4]).replace(/\s+/g, '-');
  const url = metadata.permalink || '/' + [...categories, date[1], date[2].padStart(2, '0'), date[3].padStart(2, '0'), `${slug}.html`].join('/');
  return { ...metadata, url, raw, source: path, excluded: metadata.published === false || metadata.draft === true || relative(join(root, '_posts'), path).split('/').some(part => part.startsWith('_') || part.startsWith('.')) };
}).filter(Boolean);
const posts = sources.filter(post => !post.excluded);
for (const post of sources.filter(post => post.excluded)) assert(!existsSync(outputFor(post.url)), `Unpublished post emitted: ${post.url}`);
assert(posts.length > 0, 'No published posts loaded');
const sitemap = requireRoute('/sitemap.xml');
let multilingual = 0;
for (const post of posts) {
  const html = requireRoute(post.url);
  assert(!post.source?.split('/').some(part => /^_drafts?$/.test(part)), `Draft loaded as a post: ${post.source}`);
  assert(!post.draft && post.published !== false, `Unpublished post loaded: ${post.url}`);
  assert(!html.includes('class="katex-error"'), `Math rendering error: ${post.url}`);
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length, `Duplicate heading/footnote IDs: ${post.url}`);
  for (const match of html.matchAll(/href="#([^"]+)"/g)) assert(ids.includes(decodeURIComponent(match[1])), `Broken fragment #${match[1]} in ${post.url}`);
  assert(html.includes('rel="canonical"'), `Missing canonical link: ${post.url}`);
  assert(sitemap.includes(new URL(post.url, origin).href) || sitemap.includes(`${origin}${post.url}`), `Post omitted from sitemap: ${post.url}`);
  if ((post.languages || []).length > 1) {
    multilingual++;
    for (const language of post.languages) assert(html.includes(`data-lang="${language}"`), `Translation missing: ${post.url} (${language})`);
    assert(html.includes('data-page-titles='), `Title translations missing: ${post.url}`);
    assert(!/<section\b[^>]*markdown=["']1["']/.test(html), `Unprocessed Markdown section: ${post.url}`);
  }
}

// Look for substantial draft-only paragraphs in HTML, feeds and search data, not only filenames.
const normalize = text => text.replace(/<[^>]*>/g, '').replace(/[^\p{L}\p{N}]/gu, '');
const publicSource = normalize(posts.map(post => post.raw).join('\n'));
const draftSources = [...files(join(root, '_draft')), ...files(join(root, '_drafts')), ...sources.filter(post => post.excluded).map(post => post.source)].filter(path => /\.(?:md|markdown|html)$/i.test(path));
const draftMarkers = draftSources.flatMap(path => readFileSync(path, 'utf8').split(/\r?\n/).map(normalize).filter(line => line.length >= 100 && !publicSource.includes(line)).slice(0, 2).map(marker => ({ path, marker })));
const textFiles = artifactFiles.filter(path => /\.(?:html|xml|json|txt)$/.test(path) && !relative(dist, path).startsWith('assets/'));
for (const path of textFiles) {
  const text = normalize(readFileSync(path, 'utf8'));
  for (const draft of draftMarkers) assert(!text.includes(draft.marker), `Draft content from ${relative(root, draft.path)} leaked into ${relative(dist, path)}`);
}

const htmlFiles = artifactFiles.filter(path => /\.html$/.test(path));
const missingAssets = new Set();
for (const path of htmlFiles) {
  const html = readFileSync(path, 'utf8');
  // Source-defined scripts/styles/images must survive the move; already-broken legacy assets are reported separately.
  for (const match of html.matchAll(/\b(?:src|href)=["']([^"']+)["']/g)) {
    const value = match[1].replaceAll('&amp;', '&');
    if (!value.startsWith('/assets/') && !value.startsWith('/_astro/')) continue;
    const url = new URL(value, origin);
    if (!existsSync(outputFor(url.href))) missingAssets.add(`${url.pathname} (in ${relative(dist, path)})`);
  }
  assert(!/\{%\s*(?:raw|endraw|include|assign|for|endfor)\b/.test(html), `Unprocessed Liquid: ${relative(dist, path)}`);
}
const introducedMissing = [...missingAssets].filter(item => {
  const path = decodeURIComponent(item.split(' (in ')[0]);
  return path.startsWith('/_astro/') || existsSync(join(root, path));
});
assert.equal(introducedMissing.length, 0, `Assets lost in migration:\n${introducedMissing.join('\n')}`);
if (missingAssets.size) console.warn(`Existing content references ${missingAssets.size} missing local assets (not introduced by migration):\n${[...missingAssets].join('\n')}`);
console.log(`Migration verified: ${posts.length} posts, ${multilingual} multilingual posts, ${oldRoutes.length} legacy routes, ${htmlFiles.length} HTML pages; no source/draft directories published.`);
