// Run after npm run build. Uses an isolated browser profile and only the local build.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, stat, mkdir } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { chromium } from 'playwright';
import { getPosts } from '../src/lib/content.ts';

const root = path.resolve(import.meta.dirname, '../dist');
const screenshots = path.join(os.tmpdir(), 'astro-migration-preview');
await mkdir(screenshots, { recursive: true });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.xml': 'application/xml', '.png': 'image/png', '.woff2': 'font/woff2' };
const server = createServer(async (request, response) => {
  try {
    let file = path.resolve(root, '.' + decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
    if (!file.startsWith(root + path.sep)) { if (file !== root) throw new Error('Outside public build'); }
    if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html');
    response.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    response.end(await readFile(file));
  } catch { response.writeHead(404); response.end('Not found'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || (process.platform === 'darwin' ? 'chrome' : 'chromium'), headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, locale: 'en-US', colorScheme: 'light' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  // Comments/CDNs are third-party services, outside this local UI test.
  await page.route('**/*', route => route.request().url().startsWith(origin) ? route.continue() : route.abort());
  async function expectTheme(theme) {
    await page.waitForFunction(value => document.documentElement.dataset.theme === value, theme);
    assert.equal(await page.evaluate(() => localStorage.getItem('theme')), theme, 'Explicit theme preference persists');
  }
  async function toggleTheme(theme) {
    const button = page.locator('#theme-btn');
    assert(await button.getAttribute('aria-label'), 'Theme toggle needs an accessible name');
    await button.focus();
    await button.press('Enter');
    await expectTheme(theme);
  }
  async function noOverflow(label) {
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${label} overflows mobile viewport`);
  }
  async function selectLanguage(language) {
    if (!(await page.locator('#site-language').isVisible())) await page.locator('#menu-btn').click();
    await page.locator('#site-language').selectOption(language);
    await page.waitForFunction(value => localStorage.getItem('blog.language') === value && document.querySelector('#site-language')?.value === value, language);
  }
  async function expectLanguage(language) {
    assert.equal(await page.locator('#site-language').inputValue(), language, 'Global language selector reflects persisted choice');
    assert.equal(await page.evaluate(() => localStorage.getItem('blog.language')), language, 'Global language preference persists');
  }
  await page.goto(origin);
  await page.waitForFunction(() => document.documentElement.dataset.theme === 'light');
  assert.equal(await page.locator('.post-card').count(), 6);
  await page.screenshot({ path: path.join(screenshots, 'home.png'), fullPage: true });
  const lightBackground = await page.locator('body').evaluate(element => getComputedStyle(element).backgroundColor);
  await toggleTheme('dark');
  const darkBackground = await page.locator('body').evaluate(element => getComputedStyle(element).backgroundColor);
  assert.notEqual(lightBackground, darkBackground, 'Theme changes the rendered background');
  await page.screenshot({ path: path.join(screenshots, 'home-dark.png'), fullPage: true });
  await page.reload();
  await expectTheme('dark');
  await page.locator('a[rel="next"]').click();
  assert(page.url().endsWith('/page2.html'));
  await expectTheme('dark');
  await toggleTheme('light');
  assert.equal(await page.locator('.post-card').count(), 6);
  assert.equal((await page.goto(origin + '/page2/')).status(), 200);
  for (const url of ['/_draft/secret.md', '/_drafts/be-not-disclosed.md', '/1.md', '/scripts/translate-post', '/_posts/2026/2026-09-29-sad.md']) {
    assert.equal((await page.request.get(origin + url)).status(), 404, `Private path visible: ${url}`);
  }
  const posts = await getPosts();
  const sad = posts.find(post => post.id.endsWith('/2026-09-29-sad.md'));
  await page.goto(origin);
  await selectLanguage('en');
  const englishArchive = await page.locator('#menu-items a[href="/archive.html"]').textContent();
  const englishIntro = await page.locator('#hero p').first().textContent();
  await selectLanguage('ja');
  const sadCard = page.locator('.post-card').filter({ has: page.locator(`a[href="${sad.url}"]`) });
  assert.equal(await sadCard.locator('h2').textContent(), sad.titles.ja, 'Home card uses translated title');
  assert.equal(await sadCard.locator('.excerpt').textContent(), sad.excerpts.ja, 'Home card uses translated excerpt');
  assert.notEqual(await page.locator('#menu-items a[href="/archive.html"]').textContent(), englishArchive, 'Header labels translate');
  assert.notEqual(await page.locator('#hero p').first().textContent(), englishIntro, 'Home introduction translates');
  await page.reload();
  await expectLanguage('ja');
  assert.equal(await sadCard.locator('h2').textContent(), sad.titles.ja);
  await page.locator('a[rel="next"]').click();
  await expectLanguage('ja');
  assert.notEqual(await page.locator('#menu-items a[href="/archive.html"]').textContent(), englishArchive, 'Pagination preserves translated header');
  const monolingual = posts.find(post => post.languages.length <= 1);
  assert(monolingual, 'Need a source-only article to verify locale fallback');
  await page.goto(origin + monolingual.url);
  await expectLanguage('ja');
  assert.equal(await page.locator('.main__title h1').textContent(), monolingual.title, 'Missing translation preserves original article title');
  await page.goto(origin + sad.url);
  await expectLanguage('ja');
  assert.equal(await page.locator('.post-translation:visible').getAttribute('data-lang'), 'ja');
  const sadIndex = posts.findIndex(post => post.url === sad.url);
  for (const neighbor of [posts[sadIndex - 1], posts[sadIndex + 1]].filter(Boolean)) {
    assert((await page.locator(`.article-neighbors a[href="${neighbor.url}"]`).textContent()).includes(neighbor.titles.ja || neighbor.title), 'Adjacent post title follows global locale');
  }
  await selectLanguage('ko');
  await expectLanguage('ko');
  await page.goto(origin);
  await expectLanguage('ko');
  assert.equal(await sadCard.locator('h2').textContent(), sad.titles.ko || sad.title, 'Global choice on an article carries back to home');
  assert.equal(await sadCard.locator('.excerpt').textContent(), sad.excerpts.ko || sad.excerpt);
  await page.goto(origin + '/archive.html');
  await selectLanguage('ja');
  await page.locator('#post-search').fill(sad.titles.ja);
  assert.equal(await page.locator('.archive-list li:visible').count(), 1, 'Archive searches translated titles');
  assert.equal(await page.locator('.archive-list li:visible a').textContent(), sad.titles.ja);
  for (const post of posts.filter(post => post.languages.length > 1)) {
    await page.goto(origin + post.url);
    assert.equal(await page.locator('article [data-language], article .language-switcher').count(), 0, 'Article must use only the global header language selector');
    for (const language of [...post.languages, post.defaultLang]) {
      await selectLanguage(language);
      await expectLanguage(language);
      assert.equal(await page.locator('.main__title h1').textContent(), post.titles[language] || post.title);
      assert.equal(await page.title(), (post.titles[language] || post.title) + " - Hyunsung Lee's Blog");
      assert.equal(await page.locator('.post-translation:visible').count(), 1);
      assert.equal(await page.locator('.post-translation:visible').getAttribute('data-lang'), language);
      assert(await page.evaluate(() => {
        const active = document.querySelector('.post-translation:not([hidden])');
        const headings = new Set([...active.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]')].map(heading => heading.id));
        return [...document.querySelectorAll('.js-page-aside li')].every(item => {
          const link = item.querySelector('a[href^="#"]');
          return !link || item.hidden === !headings.has(decodeURIComponent(link.getAttribute('href').slice(1)));
        });
      }), `Contents must follow global locale: ${post.url} (${language})`);
    }
  }
  const codePost = posts.find(post => /<pre class="shiki[^]*?<span style="color:/.test(post.html));
  assert(codePost, 'A published code article is required for theme verification');
  await page.goto(origin + codePost.url);
  const codeColors = () => page.locator('pre.shiki:has(span[style])').first().evaluate(element => ({
    background: getComputedStyle(element).backgroundColor,
    token: getComputedStyle(element.querySelector('span[style]')).color,
  }));
  const lightCode = await codeColors();
  await toggleTheme('dark');
  const darkCode = await codeColors();
  assert.notEqual(lightCode.background, darkCode.background, 'Code block background adapts to dark mode');
  assert.notEqual(lightCode.token, darkCode.token, 'Syntax highlighting adapts to dark mode');
  await page.screenshot({ path: path.join(screenshots, 'code-dark.png'), fullPage: true });
  await toggleTheme('light');
  await page.goto(origin + sad.url);
  await selectLanguage('ko');
  await page.reload();
  assert.equal(await page.locator('.main__title h1').textContent(), sad.title, 'Language persists after refresh');
  await page.screenshot({ path: path.join(screenshots, 'article.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await noOverflow('Article');
  for (const [url, label] of [['/', 'Home'], [sad.url, 'Article'], ['/archive.html', 'Archive']]) {
    await page.goto(origin + url);
    for (const language of ['ko', 'ja', 'zh', 'en']) {
      await selectLanguage(language);
      await noOverflow(`${label} in ${language}`);
      if (await page.locator('#menu-btn').getAttribute('aria-expanded') === 'true') await page.locator('#menu-btn').press('Escape');
      await noOverflow(`${label} in ${language} with closed menu`);
    }
    const menu = page.locator('#menu-btn');
    assert(await menu.isVisible(), 'Mobile menu toggle should be visible');
    assert.equal(await menu.getAttribute('aria-controls'), 'menu-items');
    assert.equal(await menu.getAttribute('aria-expanded'), 'false');
    assert(!(await page.locator('#menu-items').isVisible()), 'Mobile navigation starts collapsed');
    await menu.focus();
    await menu.press('Enter');
    assert.equal(await menu.getAttribute('aria-expanded'), 'true');
    assert(await page.locator('#menu-items').isVisible(), 'Keyboard opens mobile navigation');
    await noOverflow(`${label} with open menu`);
    await menu.press('Escape');
    assert.equal(await menu.getAttribute('aria-expanded'), 'false');
    assert(!(await page.locator('#menu-items').isVisible()), 'Escape closes mobile navigation');
    await page.screenshot({ path: path.join(screenshots, `${label.toLowerCase()}-mobile.png`), fullPage: true });
  }
  await page.locator('#menu-btn').click();
  await page.locator('#menu-items a[href="/about.html"]').click();
  assert(page.url().endsWith('/about.html'), 'Mobile navigation reaches its destination');
  assert.equal(await page.locator('#menu-btn').getAttribute('aria-expanded'), 'false', 'Navigation resets mobile menu');
  await page.goto(origin + sad.url);
  await page.screenshot({ path: path.join(screenshots, 'mobile.png'), fullPage: true });
  await page.goto(origin + '/archive.html?tag=Thoughts');
  assert((await page.locator('.archive-list li:visible').count()) > 0);
  await page.locator('#post-search').fill('NO-SUCH-POST-183771');
  assert.equal(await page.locator('.archive-list li:visible').count(), 0);
  assert(await page.locator('#empty-results').isVisible());
  await page.locator('#post-search').fill('');
  await page.locator('[data-tag=""]').click();
  assert.equal(await page.locator('.archive-list li:visible').count(), posts.length);
  await page.goto(origin + '/temp.html');
  assert.equal(await page.locator('#method-picker').count(), 1);
  assert.deepEqual(errors, [], 'Browser runtime errors');
  console.log(`Browser verified: ${posts.filter(post => post.languages.length > 1).length} multilingual articles, title/body/persistence, global locale + translated cards/archive/search + source fallback, pagination, archive, light/dark persistence + code colors, keyboard mobile menu + overflow, private-path 404s. Screenshots: ${screenshots}`);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
