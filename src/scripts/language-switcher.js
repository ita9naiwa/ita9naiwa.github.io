(function () {
  var content = document.querySelector('.js-multilingual-content');
  if (!content) return;

  var sections = Array.prototype.slice.call(
    content.querySelectorAll('.post-translation[data-lang]')
  );
  if (!sections.length) return;

  var available = sections.map(function (section) {
    return section.getAttribute('data-lang');
  });
  var visibleTitle = document.querySelector('.main__title h1');
  var originalTitle = visibleTitle ? visibleTitle.textContent : '';
  var originalDocumentTitle = document.title;
  var toc = document.querySelector('.js-page-aside');

  function normalize(language) {
    var value = (language || '').toLowerCase();
    if (value.indexOf('ko') === 0) return 'ko';
    if (value.indexOf('ja') === 0) return 'ja';
    if (value.indexOf('zh') === 0) return 'zh';
    if (value.indexOf('en') === 0) return 'en';
    return value.split('-')[0];
  }

  function pageTitles() {
    try {
      return JSON.parse(content.getAttribute('data-page-titles') || '{}') || {};
    } catch (error) {
      return {};
    }
  }

  function updateToc(activeSection) {
    if (!toc) return;

    var visibleHeadingIds = {};
    Array.prototype.forEach.call(
      activeSection.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]'),
      function (heading) { visibleHeadingIds[heading.id] = true; }
    );

    Array.prototype.forEach.call(toc.querySelectorAll('a[href^="#"]'), function (link) {
      var targetId;
      try {
        targetId = decodeURIComponent(link.getAttribute('href').slice(1));
      } catch (error) {
        targetId = link.getAttribute('href').slice(1);
      }
      var item = link.closest ? link.closest('li') : link.parentNode;
      if (item) item.hidden = !visibleHeadingIds[targetId];
    });
  }

  function activate(language) {
    var selected = available.indexOf(language) !== -1
      ? language
      : normalize(content.getAttribute('data-default-language'));
    if (available.indexOf(selected) === -1) selected = available[0];

    var activeSection;
    sections.forEach(function (section) {
      var active = section.getAttribute('data-lang') === selected;
      if (active) activeSection = section;
      section.hidden = !active;
      section.setAttribute('aria-hidden', active ? 'false' : 'true');
    });

    content.classList.add('is-language-ready');
    content.lang = selected;
    if (visibleTitle) visibleTitle.lang = selected;

    var titles = pageTitles();
    if (titles[selected]) {
      if (visibleTitle) visibleTitle.textContent = titles[selected];
      var siteTitle = content.getAttribute('data-site-title');
      document.title = titles[selected] + (siteTitle ? ' - ' + siteTitle : '');
    } else {
      if (visibleTitle) visibleTitle.textContent = originalTitle;
      document.title = originalDocumentTitle;
    }
    if (activeSection) updateToc(activeSection);

  }


  var fallback = normalize(content.getAttribute('data-default-language'));
  document.addEventListener('blog:languagechange', function (event) {
    activate(event.detail.language);
  });
  activate(document.documentElement.getAttribute('data-blog-language') || fallback);
}());
