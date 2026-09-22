import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const [slug, language = 'both'] = process.argv.slice(2);
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || !['en', 'zh', 'both'].includes(language)) {
  console.error('Usage: npm run new-post -- my-post [en|zh|both]');
  process.exit(1);
}
const root = fileURLToPath(new URL('../src/content/blog/', import.meta.url));
const langs = language === 'both' ? ['en', 'zh'] : [language];
const date = new Date().toISOString().slice(0, 10);
for (const lang of langs) {
  const directory = resolve(root, lang);
  await mkdir(directory, { recursive: true });
  const path = resolve(directory, `${slug}.md`);
  const title = lang === 'zh' ? '在这里填写文章标题' : 'Your post title';
  const description = lang === 'zh' ? '用一句话介绍这篇文章。' : 'Describe this post in one sentence.';
  const body = `---\ntitle: "${title}"\ndescription: "${description}"\ndate: ${date}\nlang: ${lang}\ntranslationKey: ${slug}\ntags: []\ndraft: true\n---\n\n${lang === 'zh' ? '从这里开始写作。' : 'Start writing here.'}\n`;
  try {
    await writeFile(path, body, { flag: 'wx' });
    console.log(`Created ${path}`);
  } catch (error) {
    if (error.code === 'EEXIST') { console.log(`Already exists, skipped: ${path}`); }
    else throw error;
  }
}
