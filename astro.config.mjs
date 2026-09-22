import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://qiancyrus.github.io',
  output: 'static',
  trailingSlash: 'always',
  redirects: { '/': '/en/' },
  integrations: [sitemap({ filter: (page) => !page.endsWith('/404/') })],
  markdown: {
    shikiConfig: { theme: 'github-dark' },
  },
  devToolbar: { enabled: false },
});
