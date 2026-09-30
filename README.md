# Hyunsung Lee's Blog

Static Astro blog, deployed to GitHub Pages. Node 24 is recommended (`.nvmrc`).

## Theme

Uses [AstroPaper](https://github.com/satnaing/astro-paper) 6.1.0, adapted from upstream commit `35cfa7fbe0b897306d27670d3819e55d5205f3dd`. The upstream color tokens, typography, icons and page layouts are integrated with this blog's existing URLs and multilingual content renderer. The MIT license is retained at `assets/licenses/astro-paper.txt`.

Light/dark mode follows the system preference until a reader chooses a theme, then remembers that choice. Google Sans Code is served locally. Archive search continues to use the existing title/tag filter; no external search service is needed.

## Development

```sh
npm ci
npm run dev
```

The development server is at http://localhost:4321. To build and validate the deployable site:

```sh
npm run check
npm run preview
```

`npm run test:browser` verifies light/dark mode, mobile navigation, language switching, titles, pagination, archive filtering, mobile layout, and private-path 404s against `dist`. On macOS it uses installed Chrome with an isolated temporary profile. Elsewhere, install the test browser with `npx playwright install chromium`. Screenshots go to the system temporary directory under `astro-migration-preview`.

## Writing and drafts

Keep writing Markdown in `_posts/YYYY/YYYY-MM-DD-slug.md`; existing frontmatter, translations, math, and footnotes are supported without rewriting the original posts. `scripts/translate-post` retains its confirmation and source-text protection, and now uses Node instead of Ruby.

Drafts stay in `_draft/` or `_drafts/`. They are never read into the site. `draft: true` and `published: false` also exclude a post or page from HTML, home, archive, RSS, and sitemap. Underscore-prefixed/hidden folders and root notes without frontmatter are excluded. There is no draft-preview flag.

Only `assets/`, `toy-examples/`, the favicon, and Google verification file are copied into the generated public directory. Repository source, translation scripts, personal notes, and diary folders are not deployed. This governs the built website; it does not change the visibility of files already committed to GitHub.

## Deployment

The workflow in `.github/workflows/deploy.yml` builds and tests pull requests. Pushes to `master` build, test, and deploy **only `dist/`** through GitHub Pages Actions.

For the first Astro deployment, set the repository's **Settings → Pages → Build and deployment → Source** to **GitHub Actions**. The repository previously used the legacy Jekyll build from `master`; that setting must be switched when this migration is published.

Existing post URLs and pathname-based Utterances comments are preserved. Old `/page2/` pagination links are retained alongside `/page2.html`. RSS remains at `/feed.xml`, and the sitemap remains at `/sitemap.xml`.

## Implementation

- `src/lib/content.ts`: published content, legacy Markdown compatibility, URLs.
- `src/pages/`, `src/components/`, `src/styles/`: Astro pages and UI.
- `scripts/prepare-public.mjs`: public asset allowlist.
- `scripts/verify-migration.mjs`: deployed artifact checks, including draft exclusion.

The old Jekyll build and theme are retired; their source remains in Git history. Five missing image references in old posts predate the migration and are reported by the verification script.

The header language selector controls Korean, Japanese, Chinese, or English across the entire site, including article content. Navigation, dates, cards, archive search, and adjacent post titles follow that preference across page loads. Existing translations supply article titles and excerpts; untranslated articles retain their original content. Preferences stay in browser storage; URLs and static metadata retain their original form.
