import { getPosts, getPages } from '../lib/content';
import type { APIRoute } from 'astro';
export const GET: APIRoute = async ({ site }) => {
  const [posts, pages] = await Promise.all([getPosts(), getPages()]);
  const paths = ['/', '/archive.html', ...posts.map(post => post.url), ...pages.map(page => page.url), ...Array.from({length: Math.max(0, Math.ceil(posts.length / 6) - 1)}, (_, i) => `/page${i + 2}.html`)];
  const xml = paths.map(path => `<url><loc>${new URL(path, site).href.replace(/&/g, '&amp;')}</loc></url>`).join('');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${xml}</urlset>`, {headers: {'Content-Type': 'application/xml; charset=utf-8'}});
};
