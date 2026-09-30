// Run: node scripts/test-language-switcher.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const post = fs.readFileSync(path.join(__dirname, '../_posts/2026/2026-09-29-sad.md'), 'utf8');
const originalTitle = post.match(/^title: "(.+)"$/m)[1];
const titles = Object.fromEntries([...post.matchAll(/^  (en|ja|zh): "(.+)"$/gm)].map(m => [m[1], m[2]]));
const languages = ['ko', 'ja', 'zh', 'en'];
function element(attributes = {}) {
  return {
    getAttribute: name => attributes[name],
    setAttribute: (name, value) => { attributes[name] = value; },
    classList: { add() {}, toggle() {} },
    addEventListener(name, callback) { this[name] = callback; },
  };
}
const sections = languages.map(lang => element({ 'data-lang': lang }));
const content = element({ 'data-default-language': 'ko', 'data-page-titles': JSON.stringify(titles), 'data-site-title': 'Blog' });
content.querySelectorAll = () => sections;
const heading = { textContent: originalTitle };
const listeners = {};
const document = {
  addEventListener(name, callback) { listeners[name] = callback; },
  dispatchEvent(event) { listeners[event.type]?.(event); },
  title: originalTitle + ' - Blog',
  documentElement: element({ 'data-blog-language': 'en' }),
  querySelector: selector => ({
    '.js-multilingual-content': content,
    '.main__title h1': heading,
  })[selector],
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/scripts/language-switcher.js'), 'utf8'), {
  CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
  document, navigator: { languages: ['en'] },
  window: { localStorage: { getItem() {}, setItem() {} } },
});
assert.equal(heading.textContent, titles.en);
for (const lang of ['ko', 'ja', 'ko', 'zh', 'ko', 'en', 'ko']) {
  document.dispatchEvent({ type: 'blog:languagechange', detail: { language: lang } });
  assert.equal(heading.textContent, titles[lang] || originalTitle);
  assert.equal(document.title, (titles[lang] || originalTitle) + ' - Blog');
  assert.equal(content.lang, lang);
  assert.deepEqual(sections.map(section => !section.hidden), languages.map(value => value === lang));
}
console.log('PASS: browser language and all language switches update heading, tab title, and body');

document.documentElement.lang = 'ja';
document.dispatchEvent({ type: 'blog:languagechange', detail: { language: 'ja' } });
assert.equal(heading.textContent, titles.ja);
document.dispatchEvent({ type: 'blog:languagechange', detail: { language: 'fr' } });
assert.equal(content.lang, 'ko');
assert.equal(document.documentElement.lang, 'ja', 'article fallback must not overwrite global preference');
