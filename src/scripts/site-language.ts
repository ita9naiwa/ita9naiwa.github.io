const labels = {
  en: {
    posts:'Posts', tags:'Tags', about:'About', search:'Search', searchPosts:'Search posts', skip:'Skip to content', language:'Language',
    recentPosts:'Recent Posts', allPosts:'All posts', heroDescription:'Notes on ML systems, recommender systems, and life.', topics:'Topics',
    mlSystems:'ML Systems', machineLearning:'Machine Learning', recommenderSystems:'Recommender Systems', thoughts:'Thoughts',
    newer:'Newer', older:'Older', contents:'Contents', comments:'Comments', viewSource:'View source on GitHub',
    licenseIntro:'This work is licensed under', newerPost:'Newer post', olderPost:'Older post', searchPlaceholder:'Title or tag…',
    noResults:'No posts found.', all:'All', notFound:'Page not found', notFoundDescription:'The page may have moved or is no longer available.', browsePosts:'Browse all posts',
    postCount:'{count} posts', pageCount:'Page {page} of {total}', mainNavigation:'Main navigation', pagination:'Pagination', adjacentPosts:'Adjacent posts', rssFeed:'RSS feed',
    openMenu:'Open menu', closeMenu:'Close menu', lightTheme:'Switch to light theme', darkTheme:'Switch to dark theme', pageTitle:'Blog — Page {page}',
  },
  ko: {
    posts:'글', tags:'태그', about:'소개', search:'검색', searchPosts:'글 검색', skip:'본문으로 건너뛰기', language:'언어',
    recentPosts:'최근 글', allPosts:'전체 글', heroDescription:'머신러닝 시스템, 추천 시스템, 그리고 삶에 관한 기록.', topics:'주제',
    mlSystems:'ML 시스템', machineLearning:'머신러닝', recommenderSystems:'추천 시스템', thoughts:'생각',
    newer:'최신 글', older:'이전 글', contents:'목차', comments:'댓글', viewSource:'GitHub에서 원문 보기',
    licenseIntro:'이 글의 이용 조건:', newerPost:'다음 글', olderPost:'이전 글', searchPlaceholder:'제목 또는 태그…',
    noResults:'검색 결과가 없어.', all:'전체', notFound:'페이지를 찾을 수 없어', notFoundDescription:'페이지가 이동했거나 더 이상 제공되지 않아.', browsePosts:'전체 글 보기',
    postCount:'글 {count}개', pageCount:'{page} / {total} 페이지', mainNavigation:'주요 메뉴', pagination:'페이지 이동', adjacentPosts:'이전·다음 글', rssFeed:'RSS 피드',
    openMenu:'메뉴 열기', closeMenu:'메뉴 닫기', lightTheme:'라이트 모드로 전환', darkTheme:'다크 모드로 전환', pageTitle:'블로그 — {page}페이지',
  },
  ja: {
    posts:'記事', tags:'タグ', about:'プロフィール', search:'検索', searchPosts:'記事を検索', skip:'本文へスキップ', language:'言語',
    recentPosts:'最近の記事', allPosts:'すべての記事', heroDescription:'機械学習システム、推薦システム、そして日々の記録。', topics:'トピック',
    mlSystems:'MLシステム', machineLearning:'機械学習', recommenderSystems:'推薦システム', thoughts:'雑記',
    newer:'新しい記事', older:'以前の記事', contents:'目次', comments:'コメント', viewSource:'GitHubで原文を見る',
    licenseIntro:'この記事のライセンス：', newerPost:'次の記事', olderPost:'前の記事', searchPlaceholder:'タイトルまたはタグ…',
    noResults:'記事が見つかりません。', all:'すべて', notFound:'ページが見つかりません', notFoundDescription:'ページが移動したか、公開が終了した可能性があります。', browsePosts:'すべての記事を見る',
    postCount:'{count}件の記事', pageCount:'{page} / {total} ページ', mainNavigation:'メインメニュー', pagination:'ページ移動', adjacentPosts:'前後の記事', rssFeed:'RSSフィード',
    openMenu:'メニューを開く', closeMenu:'メニューを閉じる', lightTheme:'ライトモードに切り替え', darkTheme:'ダークモードに切り替え', pageTitle:'ブログ — {page}ページ',
  },
  zh: {
    posts:'文章', tags:'标签', about:'关于', search:'搜索', searchPosts:'搜索文章', skip:'跳转到正文', language:'语言',
    recentPosts:'最新文章', allPosts:'全部文章', heroDescription:'关于机器学习系统、推荐系统与生活的记录。', topics:'主题',
    mlSystems:'ML系统', machineLearning:'机器学习', recommenderSystems:'推荐系统', thoughts:'随想',
    newer:'较新文章', older:'较早文章', contents:'目录', comments:'评论', viewSource:'在GitHub查看原文',
    licenseIntro:'本文采用以下许可：', newerPost:'下一篇', olderPost:'上一篇', searchPlaceholder:'标题或标签…',
    noResults:'未找到文章。', all:'全部', notFound:'找不到页面', notFoundDescription:'页面可能已移动或不再提供。', browsePosts:'浏览全部文章',
    postCount:'{count}篇文章', pageCount:'第 {page} / {total} 页', mainNavigation:'主导航', pagination:'分页导航', adjacentPosts:'上一篇与下一篇', rssFeed:'RSS订阅',
    openMenu:'打开菜单', closeMenu:'关闭菜单', lightTheme:'切换到浅色模式', darkTheme:'切换到深色模式', pageTitle:'博客 — 第{page}页',
  },
};
type Language = keyof typeof labels;
type Key = keyof typeof labels.en;
const normalize = (value: string | null | undefined) => value?.toLowerCase().split('-')[0] as Language;
const supported = (value: string): value is Language => Object.hasOwn(labels, value);
let saved;
try { saved = normalize(localStorage.getItem('blog.language')); } catch {}
let language: Language = saved && supported(saved) ? saved : (navigator.languages || [navigator.language]).map(normalize).find(supported) || 'en';
const originalTitle = document.title;
function translate(key: string, args: Record<string, unknown> = {}) {
  return (labels[language][key as Key] || labels.en[key as Key] || '').replace(/\{(\w+)\}/g, (_, name) => String(args[name] ?? ''));
}
function updateText() {
  document.querySelectorAll<HTMLElement>('[data-i18n]').forEach(element => {
    const value = translate(element.dataset.i18n!, JSON.parse(element.dataset.i18nArgs || '{}'));
    if (value) element.textContent = value;
  });
  for (const attribute of ['placeholder', 'aria-label', 'title']) {
    document.querySelectorAll<HTMLElement>(`[data-i18n-${attribute}]`).forEach(element => {
      const value = translate(element.getAttribute(`data-i18n-${attribute}`)!);
      if (value) element.setAttribute(attribute, value);
    });
  }
  document.querySelectorAll<HTMLElement>('[data-localized-text]').forEach(element => {
    const values = JSON.parse(element.dataset.localizedText || '{}');
    element.textContent = values[language] || element.dataset.defaultText || '';
  });
  document.querySelectorAll<HTMLTimeElement>('[data-localized-date]').forEach(element => {
    element.textContent = new Date(element.dataset.localizedDate!).toLocaleDateString(language, { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });
  });
  const menu = document.querySelector('#menu-btn');
  menu?.setAttribute('aria-label', translate(menu.getAttribute('aria-expanded') === 'true' ? 'closeMenu' : 'openMenu'));
  const theme = document.querySelector('#theme-btn');
  const themeLabel = translate(document.documentElement.dataset.theme === 'dark' ? 'lightTheme' : 'darkTheme');
  theme?.setAttribute('aria-label', themeLabel);
  theme?.setAttribute('title', themeLabel);
  const titleKey = ({ 'All Posts':'allPosts', 'Page not found':'notFound' } as Record<string,string>)[originalTitle.split(' - ')[0]];
  if (titleKey) document.title = translate(titleKey) + " - Hyunsung Lee's Blog";
  const pagination = originalTitle.match(/^Blog — Page (\d+)/);
  if (pagination) document.title = translate('pageTitle', {page: pagination[1]}) + " - Hyunsung Lee's Blog";
}
function apply(value: string, persist = false) {
  const selected = normalize(value);
  if (!supported(selected)) return;
  language = selected;
  if (persist) try { localStorage.setItem('blog.language', language); } catch {}
  document.documentElement.lang = language;
  document.documentElement.dataset.blogLanguage = language;
  document.querySelectorAll<HTMLSelectElement>('[data-site-language]').forEach(select => { select.value = language; });
  updateText();
  document.dispatchEvent(new CustomEvent('blog:languagechange', {detail: {language}}));
}
document.querySelectorAll<HTMLSelectElement>('[data-site-language]').forEach(select => select.addEventListener('change', () => apply(select.value, true)));
document.addEventListener('blog:uiupdate', updateText);
window.addEventListener('storage', event => { if (event.key === 'blog.language' && event.newValue) apply(event.newValue); });
apply(language);
