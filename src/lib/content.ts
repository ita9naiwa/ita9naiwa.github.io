import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import MarkdownIt from 'markdown-it';
import footnote from 'markdown-it-footnote';
import anchor from 'markdown-it-anchor';
import { parse } from 'yaml';
import { createHighlighter } from 'shiki';
import katex from 'katex';

export interface Post {
  id: string; title: string; url: string; date: string;
  tags: string[]; categories: string[]; languages: string[]; defaultLang: string;
  titles: Record<string, string>; html: string; excerpt: string; excerpts: Record<string, string>;
  headings: { depth: number; text: string; slug: string }[];
  comment: boolean; license?: boolean;
}

const root = process.cwd();
const strings = (value: unknown): string[] => value == null ? [] : Array.isArray(value) ? value.map(String) : [String(value)];
const escape = (value: string) => value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const plain = (html: string) => html.replace(/<[^>]*>/g, ' ').replace(/&(?:nbsp|amp|quot|lt|gt);/g, ' ').replace(/\s+/g, ' ').trim();
const slugify = (text: string) => text.toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').trim().replace(/\s+/g, '-') || 'section';

async function files(directory: string): Promise<string[]> {
  const entries = await readdir(path.join(root, directory), { withFileTypes: true });
  return (await Promise.all(entries.filter(entry => !entry.name.startsWith('_') && !entry.name.startsWith('.')).map(entry => {
    const file = path.posix.join(directory, entry.name);
    return entry.isDirectory() ? files(file) : /\.(md|markdown|html)$/.test(file) ? [file] : [];
  }))).flat();
}

async function source(file: string) {
  const text = await readFile(path.join(root, file), 'utf8');
  const match = text.match(/^\uFEFF?---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  if (!match) return null;
  const data = parse(match[1]) || {};
  if (data.published === false || data.draft === true) return null;
  const name = path.basename(file).replace(/\.(md|markdown|html)$/, '');
  const dated = name.match(/^(\d{4})-(\d{1,2})-(\d{1,2})-(.+)$/);
  const dateParts = String(data.date || (dated ? dated.slice(1, 4).join('-') : '')).match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  const date = dateParts ? `${dateParts[1]}-${dateParts[2].padStart(2, '0')}-${dateParts[3].padStart(2, '0')}` : '';
  const categories = [...new Set([...strings(data.category), ...strings(data.categories)])];
  const tags = [...new Set([...strings(data.tag), ...strings(data.tags)])];
  const url = data.permalink || (dated ? `/${[...categories.map(c => c.toLowerCase()), ...date.split('-'), `${String(data.slug || dated[4]).replace(/\s+/g, '-')}.html`].join('/')}` : `/${name}.html`);
  const defaultLang = String(data.default_lang || data.lang || 'en');
  const title = String(data.title ?? data.titles?.[defaultLang] ?? data.titles?.en ?? name);
  return { file, data, body: text.slice(match[0].length), id: file, title, url, date, categories, tags,
    languages: strings(data.languages), defaultLang, titles: { ...data.titles, [defaultLang]: data.titles?.[defaultLang] || title }, comment: data.comment !== false, license: file.startsWith('_posts/') && data.license !== false };
}

type Source = NonNullable<Awaited<ReturnType<typeof source>>>;

const renderer = (async () => {
  const highlighter = await createHighlighter({ themes: ['github-light', 'github-dark'], langs: ['python', 'cpp', 'bash', 'json', 'javascript', 'typescript', 'yaml'] });
  const md = new MarkdownIt({ html: true, linkify: true, highlight(code, language) {
    const lang = language === 'c++' ? 'cpp' : language;
    return highlighter.codeToHtml(code, { lang: highlighter.getLoadedLanguages().includes(lang) ? lang : 'text', themes: { light: 'github-light', dark: 'github-dark' } });
  }}).use(footnote).use(anchor, { slugify });
  // Kramdown uses $$ both inline and in display blocks; preserve both without touching code tokens.
  md.inline.ruler.before('escape', 'math', (state: any, silent: boolean) => {
    const start = state.pos;
    const delimiter = state.src.startsWith('$$', start) ? '$$' : state.src[start] === '$' && !/\s/.test(state.src[start + 1] || ' ') ? '$' : null;
    if (!delimiter) return false;
    let end = state.src.indexOf(delimiter, start + delimiter.length);
    while (end !== -1 && state.src[end - 1] === '\\') end = state.src.indexOf(delimiter, end + delimiter.length);
    if (end < 0 || (delimiter === '$' && (/\s/.test(state.src[end - 1]) || /\d/.test(state.src[end + 1] || '')))) return false;
    if (!silent) { const token = state.push('math_inline', 'math', 0); token.content = state.src.slice(start + delimiter.length, end); }
    state.pos = end + delimiter.length;
    return true;
  });
  md.block.ruler.before('fence', 'math_block', (state: any, start: number, end: number, silent: boolean) => {
    const first = state.src.slice(state.bMarks[start] + state.tShift[start], state.eMarks[start]);
    if (!first.startsWith('$$')) return false;
    let last = start;
    let content = first.slice(2);
    if (content.includes('$$') && content.indexOf('$$') !== content.length - 2) return false;
    if (!content.endsWith('$$')) {
      if (content.includes('$$')) return false;
      for (last = start + 1; last < end; last++) {
        const line = state.src.slice(state.bMarks[last], state.eMarks[last]);
        content += '\n' + line;
        if (line.includes('$$')) break;
      }
    }
    if (!content.endsWith('$$') || last >= end) return false;
    if (!silent) { const token = state.push('math_block', 'math', 0); token.content = content.slice(0, -2); token.block = true; token.map = [start, last + 1]; }
    state.line = last + 1;
    return true;
  });
  for (const kind of ['inline', 'block']) md.renderer.rules[`math_${kind}`] = (tokens: any, index: number) => katex.renderToString(tokens[index].content.replace(/\\(?:begin|end)\{eqnarray\*?\}/g, (environment: string) => environment.replace(/eqnarray\*?/, 'aligned')).replace(/^\\\(([\s\S]*)\\\)$/, '$1'), { displayMode: kind === 'block', throwOnError: false, strict: false, trust: false });
  return md;
})();

async function render(item: Source, all: Source[]): Promise<Post> {
  const md = await renderer;
  let body = item.body
    .replace(/\{%[-]?\s*(?:raw|endraw)\s*[-]?%\}/g, '')
    .replace(/\{\{\s*["']([^"']+)["']\s*\|\s*(?:absolute_url|relative_url)\s*\}\}/g, (_, url) => '/' + url.replace(/^\/+/, '').replace(/\s*\n\s*/g, ''))
    .replace(/\{\{\s*site\.baseurl\s*\}\}/g, '')
    .replace(/\{%\s*link\s+([^%]+?)\s*%\}/g, (_, file) => {
      const target = all.find(post => post.file === file.trim());
      if (!target) throw new Error(`Unresolved Jekyll link ${file} in ${item.file}`);
      return target.url;
    });
  // Root category pages all use this same Liquid loop; regenerate it from published posts only.
  body = body.replace(/\{%-?\s*for post in site\.tags\[["']([^"']+)["']\]\s*-?%\}[\s\S]*?\{%-?\s*endfor\s*-?%\}/g, (_, tag) => all.filter(post => post.tags.includes(tag)).map(post => `<li><h4><a href="${escape(post.url)}" data-localized-text="${escape(JSON.stringify(post.titles))}" data-default-text="${escape(post.title)}">${escape(post.title)}</a></h4></li>`).join('\n'))
    .replace(/<script>\s*\{%-?\s*include scripts\/home\.js\s*-?%\}\s*<\/script>/g, '');
  const headings: Post['headings'] = [];
  let sectionIndex = 0;
  const headingIds = new Set<string>();
  const markdown = (text: string) => {
    const env = { docId: `${path.basename(item.file)}-${sectionIndex++}` };
    const tokens = md.parse(text, env);
    for (let i = 0; i < tokens.length; i++) if (tokens[i].type === 'heading_open') {
      const base = tokens[i].attrGet('id') || 'section';
      let slug = base;
      let suffix = 1;
      while (headingIds.has(slug)) slug = `${base}-${suffix++}`;
      headingIds.add(slug);
      tokens[i].attrSet('id', slug);
      headings.push({ depth: Number(tokens[i].tag.slice(1)), text: plain(md.renderInline(tokens[i + 1].content)), slug });
    }
    return md.renderer.render(tokens, md.options, env);
  };
  const sections = /<section\b([^>]*\bmarkdown=["']1["'][^>]*)>([\s\S]*?)<\/section>/g;
  let html = '';
  let previous = 0;
  for (const match of body.matchAll(sections)) {
    html += markdown(body.slice(previous, match.index));
    html += `<section${match[1].replace(/\s+markdown=["']1["']/, '')}>${markdown(match[2])}</section>\n`;
    previous = match.index! + match[0].length;
  }
  html += item.file.endsWith('.html') ? body.slice(previous).replace(/<!DOCTYPE[^>]*>|<\/?(?:html|head|body)\b[^>]*>/gi, '') : markdown(body.slice(previous));
  const firstSection = html.match(/<section[^>]*data-default-language[^>]*>([\s\S]*?)<\/section>/)?.[1] || html;
  const excerpt = plain(firstSection.split('<!--more-->')[0]).slice(0, 240);
  const excerpts: Record<string, string> = { [item.defaultLang]: excerpt };
  for (const section of html.matchAll(/<section\b[^>]*\bdata-lang=["']([^"']+)["'][^>]*>([\s\S]*?)<\/section>/g)) {
    excerpts[section[1]] = plain(section[2].split('<!--more-->')[0]).slice(0, 240);
  }
  const { file, data, body: original, ...post } = item;
  return { ...post, html, excerpt, excerpts, headings };
}

async function loadPosts(): Promise<Post[]> {
  const posts = (await Promise.all((await files('_posts')).map(source))).filter((post): post is Source => post !== null).sort((a, b) => b.date.localeCompare(a.date) || b.file.localeCompare(a.file));
  return Promise.all(posts.map(post => render(post, posts)));
}

async function loadPages(): Promise<Post[]> {
  const names = (await readdir(root)).filter(name => !name.startsWith('_') && !name.startsWith('.') && /\.(md|markdown|html)$/.test(name) && !['index.html', 'archive.html', '404.html'].includes(name));
  const pages = (await Promise.all(names.map(source))).filter((page): page is Source => page !== null);
  const posts = (await Promise.all((await files('_posts')).map(source))).filter((post): post is Source => post !== null).sort((a, b) => b.date.localeCompare(a.date));
  return Promise.all(pages.map(page => render(page, posts)));
}

let postsCache: Promise<Post[]> | undefined;
let pagesCache: Promise<Post[]> | undefined;
export function getPosts(): Promise<Post[]> {
  return process.env.NODE_ENV === 'production' ? postsCache ??= loadPosts() : loadPosts();
}
export function getPages(): Promise<Post[]> {
  return process.env.NODE_ENV === 'production' ? pagesCache ??= loadPages() : loadPages();
}
