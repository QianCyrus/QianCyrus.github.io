import rss from '@astrojs/rss';
import { copy, site, languagePaths, type Language } from '../../data/site';
import { getPosts, postUrl } from '../../lib/posts';
import type { APIContext } from 'astro';
export const getStaticPaths = languagePaths;
export async function GET(context: APIContext) {
  const lang = context.params.lang as Language;
  const posts = (await getPosts(lang)).filter((post) => !post.data.draft && post.data.date <= new Date());
  return rss({
    title: `Cyrus · ${copy[lang].nav.blog}`,
    description: copy[lang].description,
    site: site.url,
    items: posts.map((post) => ({ title: post.data.title, description: post.data.description, pubDate: post.data.date, link: postUrl(post), categories: post.data.tags })),
    customData: `<language>${copy[lang].htmlLang}</language>`,
  });
}
