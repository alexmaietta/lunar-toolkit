// Page changes without a full reload: fetch the next docs page (prefetched on hover), swap the
// side nav and content, and redo the theme's per-page setup. Anything unusual falls back to a
// normal page load.
(() => {
  const SKIP = /\/(search\.html|coverage\/|browser\/)/;
  const PARTS = ['.wy-nav-side', '.wy-nav-content-wrap'];
  const cache = new Map();

  const target = (a) => {
    if (!a || a.target || a.hasAttribute('download')) return null;
    const url = new URL(a.href, location.href);
    if (url.origin !== location.origin || SKIP.test(url.pathname)) return null;
    return url;
  };
  const page = (url) => {
    const key = url.href.split('#')[0];
    if (!cache.has(key)) {
      cache.set(key, fetch(key).then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.status)))));
      cache.get(key).catch(() => cache.delete(key));
    }
    return cache.get(key);
  };

  function setup() {
    const $ = window.jQuery;
    const nav = window.SphinxRtdTheme?.Navigation;
    $('div.rst-content table').addClass('docutils');
    $('table.docutils:not(.field-list,.footnote,.citation)').wrap("<div class='wy-table-responsive'></div>");
    $('.wy-menu-vertical ul').not('.simple').siblings('a').each(function () {
      const a = $(this);
      const button = $('<button class="toctree-expand" title="Open/close menu"></button>');
      button.on('click', (e) => { nav?.toggleCurrent(a); e.stopPropagation(); return false; });
      a.prepend(button);
    });
    if (nav) nav.navBar = $('div.wy-side-scroll:first');
    window.hljs?.highlightAll();
  }

  async function go(url, push) {
    const doc = new DOMParser().parseFromString(await page(url), 'text/html');
    const fresh = PARTS.map((s) => doc.querySelector(s));
    if (fresh.some((n) => !n)) throw new Error('not a docs page');
    const sideTop = document.querySelector('.wy-side-scroll')?.scrollTop ?? 0;
    // Push first, so relative links and images in the new content resolve against its URL
    if (push) history.pushState({}, '', url.href);
    PARTS.forEach((s, i) => document.querySelector(s).replaceWith(document.adoptNode(fresh[i])));
    document.title = doc.title;
    const base = /var base_url = "([^"]*)"/.exec(doc.documentElement.innerHTML);
    if (base) window.base_url = base[1];
    setup();
    const side = document.querySelector('.wy-side-scroll');
    if (side) side.scrollTop = sideTop;
    const anchor = url.hash && document.getElementById(decodeURIComponent(url.hash.slice(1)));
    if (anchor) anchor.scrollIntoView(); else if (push) window.scrollTo(0, 0);
  }

  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const url = target(e.target.closest('a[href]'));
    if (!url || (url.pathname === location.pathname && url.search === location.search)) return;
    e.preventDefault();
    go(url, true).catch(() => { location.href = url.href; });
  });
  document.addEventListener('mouseover', (e) => {
    const url = target(e.target.closest?.('a[href]'));
    if (url && url.pathname !== location.pathname) page(url).catch(() => {});
  });
  window.addEventListener('popstate', () => { go(new URL(location.href), false).catch(() => location.reload()); });
})();
