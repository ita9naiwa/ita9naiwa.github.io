// AstroPaper's theme behavior, adapted for full page loads and unavailable storage.
const preference = window.matchMedia('(prefers-color-scheme: dark)');
const button = document.querySelector<HTMLButtonElement>('#theme-btn');
let theme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
function storedTheme() {
  try { return localStorage.getItem('theme'); } catch { return null; }
}
function updateComments() {
  document.querySelector<HTMLIFrameElement>('.utterances-frame')?.contentWindow?.postMessage({ type: 'set-theme', theme: theme === 'dark' ? 'github-dark' : 'github-light' }, 'https://utteranc.es');
}
function reflect() {
  document.documentElement.dataset.theme = theme;
  document.documentElement.classList.toggle('dark', theme === 'dark');
  const label = `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`;
  button?.setAttribute('aria-label', label);
  button?.setAttribute('title', label);
  button?.setAttribute('aria-pressed', String(theme === 'dark'));
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', getComputedStyle(document.body).backgroundColor);
  updateComments();
  document.dispatchEvent(new Event('blog:uiupdate'));
}
if (button) {
  button.hidden = false;
  button.addEventListener('click', () => {
    theme = theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', theme); } catch { /* The toggle still works without persistence. */ }
    reflect();
  });
}
preference.addEventListener('change', event => {
  if (!['light', 'dark'].includes(storedTheme() || '')) {
    theme = event.matches ? 'dark' : 'light'; reflect();
  }
});
window.addEventListener('storage', event => {
  if (event.key === 'theme') {
    theme = event.newValue === 'dark' || (event.newValue !== 'light' && preference.matches) ? 'dark' : 'light'; reflect();
  }
});
// Capturing load also handles the asynchronously inserted Utterances iframe.
document.addEventListener('load', event => {
  if ((event.target as Element)?.classList?.contains('utterances-frame')) updateComments();
}, true);
reflect();
