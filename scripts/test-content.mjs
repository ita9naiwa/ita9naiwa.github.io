// Isolated fixture: never creates or changes real posts/drafts.
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';

const fixture = mkdtempSync(join(tmpdir(), 'astro-blog-content-'));
const contentModule = new URL('../src/lib/content.ts', import.meta.url).href;
function write(name, source) {
  const path = join(fixture, name);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, source);
}
try {
  const privatePost = (flags = '') => `---\ntitle: PRIVATE_FIXTURE_SENTINEL\n${flags}---\nPrivate fixture body.\n`;
  for (const name of ['_draft/private.md', '_drafts/private.md', '_posts/_drafts/2026-09-30-private.md', '_posts/_draft/2026-09-30-private.md', '_posts/.hidden/2026-09-30-private.md', '_draft.md', '_drafts.md']) write(name, privatePost());
  for (const [name, flags] of [['published', 'published: false\n'], ['draft', 'draft: true\n']]) {
    write(`_posts/2026-09-30-${name}.md`, privatePost(flags));
    write(`${name}.md`, privatePost(flags));
  }
  write('1.md', 'Unfrontmattered personal notes must not become a page.');
  write('_posts/guide.md', 'Unfrontmattered writing guide must not become a post.');
  write('about.md', '---\ntitle: About fixture\n---\nPublic page.\n');
  write('topic.md', '---\ntitle: Topic fixture\n---\n<ul>\n{% for post in site.tags["fixture"] %}{{ post.title }}{% endfor %}\n</ul>\n');
  write('_posts/2026-09-30-public post.md', `---
title: Public fixture
date: "2025-1-2"
category: "Mixed Category"
tag: fixture
languages: [ko, en]
default_lang: ko
titles:
  en: 'Public fixture "in English" & more'
---
<section class="post-translation" data-lang="ko" data-default-language markdown="1">
## Shared heading
본문[^1] and $x^2$.

[^1]: 한국어 각주
</section>
<section class="post-translation" data-lang="en" markdown="1">
## Shared heading
Body[^1] and $$x^2$$.

Private investment: **$285.9 billion**, **$12.4 billion**, and **$1.8 billion**. Forecasts: **roughly $800 billion** and **roughly $139 billion**.

Numeric math: $2x$.

Invalid closing whitespace: $x $.

Invalid closing digit: $x$2.

[^1]: English footnote
</section>
`);
  execFileSync(process.execPath, ['--experimental-strip-types', '--input-type=module'], {
    cwd: fixture,
    stdio: ['pipe', 'inherit', 'inherit'],
    input: `
import assert from 'node:assert/strict';
const { getPosts, getPages } = await import(${JSON.stringify(contentModule)});
const posts = await getPosts();
const pages = await getPages();
assert.equal(posts.length, 1, 'Only the public fixture post should load');
assert.equal(pages.length, 2, 'Only public fixture pages should load');
const about = pages.find(page => page.url === '/about.html');
assert(about);
assert.equal(about.defaultLang, 'en', 'Keep the original site language for pages without explicit metadata');
const post = posts[0];
assert.equal(post.url, '/mixed category/2025/01/02/public-post.html');
assert.equal(post.titles.ko, 'Public fixture', 'Default title must fall back to the original title');
assert.equal(post.titles.en, 'Public fixture "in English" & more');
assert.equal(post.titles.ja, undefined, 'Missing titles must not be synthesized');
assert.equal(post.excerpts.ko, post.excerpt);
assert(post.excerpts.ko.includes('본문') && !post.excerpts.ko.includes('Body'));
assert(post.excerpts.en.includes('Body') && !post.excerpts.en.includes('본문'));
assert.equal(post.excerpts.ja, undefined, 'Missing excerpts must remain absent for consumer fallback');
assert(Object.values(post.excerpts).every(text => text.length <= 240 && !text.includes('<')));
assert.equal(about.excerpts.en, about.excerpt);
const topic = pages.find(page => page.url === '/topic.html');
assert(topic.html.includes('data-default-text="Public fixture"'));
const localizedTitles = topic.html.match(/data-localized-text="([^"]*)"/)[1].replaceAll('&quot;', '"').replaceAll('&amp;', '&');
assert.deepEqual(JSON.parse(localizedTitles), post.titles, 'Topic links must carry escaped translated titles for the same locale controller');
assert(!JSON.stringify([...posts, ...pages]).includes('PRIVATE_FIXTURE_SENTINEL'));
for (const language of ['ko', 'en']) assert(post.html.includes('data-lang="' + language + '"'));
assert(post.html.includes('<h2'), 'Markdown within translations should render');
assert(post.html.includes('class="katex"'), 'Math should render');
assert(!post.html.includes('katex-error'));
for (const amount of ['$285.9 billion', '$12.4 billion', '$1.8 billion', 'roughly $800 billion', 'roughly $139 billion']) {
  assert(post.html.includes('<strong>' + amount + '</strong>'), 'Currency must remain bold text, not become math');
}
assert.equal(post.html.split('class="katex"').length - 1, 3, 'Single-dollar, numeric, and double-dollar math must still render');
assert(post.html.includes('Invalid closing whitespace: $x $.'));
assert(post.html.includes('Invalid closing digit: $x$2.'));
assert(!post.html.includes('markdown="1"'));
assert(post.html.includes('한국어 각주') && post.html.includes('English footnote'));
const ids = [...post.html.matchAll(/\\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(ids).size, ids.length, 'Translations must not repeat heading or footnote IDs');
for (const match of post.html.matchAll(/href="#([^"]+)"/g)) assert(ids.includes(decodeURIComponent(match[1])), 'Every footnote link must resolve');
console.log('Content fixture verified: draft roots/nested folders, private flags and raw notes excluded; URLs, translations, footnotes and math preserved.');
`,
  });
} finally {
  rmSync(fixture, { recursive: true, force: true });
}
