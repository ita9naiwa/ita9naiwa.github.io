(function () {
  var content = document.querySelector('.js-multilingual-content');
  var switcher = document.querySelector('.js-language-switcher');
  if (!content || !switcher) return;

  var sections = Array.prototype.slice.call(
    content.querySelectorAll('.post-translation[data-lang]')
  );
  var buttons = Array.prototype.slice.call(
    switcher.querySelectorAll('[data-language]')
  );
  if (!sections.length) return;

  var available = sections.map(function (section) {
    return section.getAttribute('data-lang');
  });

  function normalize(language) {
    var value = (language || '').toLowerCase();
    if (value.indexOf('ko') === 0) return 'ko';
    if (value.indexOf('ja') === 0) return 'ja';
    if (value.indexOf('zh') === 0) return 'zh';
    if (value.indexOf('en') === 0) return 'en';
    return value.split('-')[0];
  }

  function browserLanguage() {
    var languages = navigator.languages || [navigator.language || navigator.userLanguage];
    for (var i = 0; i < languages.length; i += 1) {
      var candidate = normalize(languages[i]);
      if (available.indexOf(candidate) !== -1) return candidate;
    }
    return '';
  }

  function storedLanguage() {
    try {
      return normalize(window.localStorage.getItem('blog.language'));
    } catch (error) {
      return '';
    }
  }

  function pageTitles() {
    try {
      return JSON.parse(switcher.getAttribute('data-page-titles') || '{}') || {};
    } catch (error) {
      return {};
    }
  }

  function activate(language, persist) {
    var selected = available.indexOf(language) !== -1
      ? language
      : normalize(content.getAttribute('data-default-language'));
    if (available.indexOf(selected) === -1) selected = available[0];

    sections.forEach(function (section) {
      var active = section.getAttribute('data-lang') === selected;
      section.hidden = !active;
      section.setAttribute('aria-hidden', active ? 'false' : 'true');
    });
    buttons.forEach(function (button) {
      var active = button.getAttribute('data-language') === selected;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });

    content.classList.add('is-language-ready');
    document.documentElement.lang = selected;
    document.documentElement.setAttribute('data-blog-language', selected);

    var titles = pageTitles();
    if (titles[selected]) {
      var siteTitle = switcher.getAttribute('data-site-title');
      document.title = titles[selected] + (siteTitle ? ' - ' + siteTitle : '');
    }

    if (persist) {
      try { window.localStorage.setItem('blog.language', selected); } catch (error) {}
    }
  }

  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      activate(button.getAttribute('data-language'), true);
    });
  });

  var fallback = normalize(content.getAttribute('data-default-language'));
  activate(storedLanguage() || browserLanguage() || fallback, false);
}());
