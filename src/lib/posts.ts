import { getCollection, type CollectionEntry } from 'astro:content';
import type { Language } from '../data/site';

export async function getPosts(lang?: Language) {
  const posts = await getCollection('blog', ({ data }) =>
    (!lang || data.lang === lang) && (import.meta.env.DEV || (!data.draft && data.date <= new Date()))
  );
  return posts.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

export function postUrl(post: CollectionEntry<'blog'>) {
  return `/${post.data.lang}/blog/${postSlug(post)}/`;
}

export function postSlug(post: CollectionEntry<'blog'>) {
  return post.id.replace(new RegExp(`^${post.data.lang}/`), '');
}

export function formatDate(date: Date, lang: Language) {
  return new Intl.DateTimeFormat(lang === 'zh' ? 'zh-CN' : 'en-GB', {
    year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC',
  }).format(date);
}

export function readingTime(body: string = '') {
  const han = body.match(/[\p{Script=Han}]/gu)?.length ?? 0;
  const words = body.replace(/[\p{Script=Han}]/gu, ' ').match(/\S+/g)?.length ?? 0;
  return Math.max(1, Math.ceil(han / 350 + words / 220));
}
