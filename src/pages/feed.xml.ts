import { getPosts } from '../lib/content';
import type { APIRoute } from 'astro';
const escape = (text: string) => text.replace(/[<>&"']/g, char => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&apos;'}[char]!));
export const GET: APIRoute = async ({ site }) => {
  const posts = await getPosts();
  const base = site!.toString();
  const items = posts.map(post => `<item><title>${escape(post.title)}</title><link>${escape(new URL(post.url, base).href)}</link><guid>${escape(new URL(post.url, base).href)}</guid><pubDate>${new Date(post.date).toUTCString()}</pubDate><description>${escape(post.excerpt)}</description></item>`).join('');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Hyunsung Lee's Blog</title><link>${base}</link><description>Notes on machine learning, systems, and life.</description>${items}</channel></rss>`, { headers: {'Content-Type': 'application/rss+xml; charset=utf-8'} });
};
