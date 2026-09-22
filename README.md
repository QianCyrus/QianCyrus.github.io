# Cyrus — personal site

中英文个人主页、博客与项目展示。Astro 静态构建，部署到 GitHub Pages。

## 本地预览

使用 Node.js 24 LTS（`.nvmrc`）或 >=22.12 的兼容版本。

```sh
npm ci
npm run dev
```

打开 http://localhost:4321/en/ 或 http://localhost:4321/zh/ 。根路径默认跳转英文版。

```sh
npm run build
npm run preview
```

`build` 会先做 Astro / TypeScript 检查。正式构建排除草稿和未来日期的文章。

## 修改个人信息与文案

- `src/data/site.ts`：名字、GitHub、中英文学校名称、研究方向及所有界面文案。
- `src/styles/global.css`：配色、字体、页面样式和移动端布局。
- `src/pages/[lang]/`：首页、关于、项目、博客页面。
- 项目页当前使用空状态，没有填入具体项目或个人成果。

## 写文章

```sh
npm run new-post -- my-first-post
```

这会创建 `src/content/blog/en/my-first-post.md` 和 `src/content/blog/zh/my-first-post.md`。
只写一种语言：`npm run new-post -- my-first-post zh`。

```yaml
---
title: "文章标题"
description: "文章摘要"
date: 2026-09-23
lang: zh
translationKey: my-first-post
tags: [推理加速]
draft: true
---
```

正文使用 Markdown。两种语言相同的 `translationKey` 会让语言按钮跳到对应译文；没有译文时跳到另一语言的博客列表。
写好后将 `draft` 改为 `false`，提交到 `main` 即自动发布。草稿只在 `npm run dev` 中显示；不要把私密草稿提交到公开仓库，源码本身仍可被读取。

图片放在 `public/images/`，在文章里写 `![图片说明](/images/example.webp)`。
博客有文章时自动显示搜索（标题、摘要、正文和标签）及标签筛选。每种语言都有独立 RSS。

## 发布到 GitHub Pages

1. 在 `QianCyrus` 下创建公开仓库 `QianCyrus.github.io`。
2. 在仓库 Settings → Pages → Build and deployment 中，把 Source 设为 **GitHub Actions**。
3. 将本项目提交并推送到 `main`。
4. 等待 `Deploy site to GitHub Pages` 工作流完成。
5. 访问 https://qiancyrus.github.io/ 。

自动部署使用 `.github/workflows/deploy.yml`。PR 只执行构建检查，不部署。

后续如果绑定自己的域名，要同步更新 `astro.config.mjs` 的 `site`、`src/data/site.ts` 的 `url` 和 `public/robots.txt` 的 Sitemap 地址，并在 GitHub Pages 设置中配置域名。

## 目录

```text
src/
  components/          共用组件与 SVG 图形
  content/blog/        Markdown 博客文章
  data/site.ts         个人信息与中英文文案
  layouts/Base.astro   导航、SEO、语言切换与页脚
  lib/posts.ts         文章筛选、路由、时间格式
  pages/[lang]/        中英文页面与 RSS
  styles/global.css    响应式样式
```

参考：[Astro 内容集合](https://docs.astro.build/en/guides/content-collections/) · [GitHub Pages 部署](https://docs.astro.build/en/guides/deploy/github/)
