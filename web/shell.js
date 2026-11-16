/*
 * MacGym window shell: sidebar selection -> iframe src.
 *
 * The frame never changes; only the content region does. Electron's menu calls
 * window.macgymNav(view); anything else can postMessage
 * {type: 'macgym:nav', view, robot?}.
 */
(function () {
  'use strict';

  var TITLES = { playground: 'Playground', slides: 'Slides', viewer: 'Viewer', docs: 'Docs' };
  var ROBOTS = ['walker', 'cheetah', 'hopper', 'tripod', 'worm'];

  var frame = document.getElementById('view');
  var title = document.getElementById('title');
  var list = document.getElementById('list');
  var rows = Array.prototype.slice.call(document.querySelectorAll('.source-list__row'));

  var state = { view: null, robot: null };

  function srcFor(view, robot) {
    switch (view) {
      case 'slides':
        return '/deck2/index.html?embed=1';
      case 'viewer':
        return '/viewer?embed=1';
      case 'docs':
        return '/docs/';
      default:
        return '/playground?embed=1' + (robot ? '&robot=' + encodeURIComponent(robot) : '');
    }
  }

  function focusFrame() {
    // Arrow keys in Slides (and typing in the Playground editor) need focus
    // inside the frame, not on the sidebar row that was just clicked.
    try {
      frame.focus();
      if (frame.contentWindow) frame.contentWindow.focus();
    } catch (e) {
      /* frame not ready yet; the load handler tries again */
    }
  }

  function paint() {
    rows.forEach(function (row) {
      var selected = state.robot
        ? row.dataset.robot === state.robot
        : row.dataset.view === state.view;
      if (selected) row.setAttribute('aria-current', 'page');
      else row.removeAttribute('aria-current');
    });
    var name = TITLES[state.view] || 'MacGym';
    title.textContent = state.robot ? name + ' / ' + state.robot : name;
    document.title = 'MacGym: ' + name;
    document.body.dataset.view = state.view;
  }

  function nav(view, robot) {
    if (!TITLES[view]) view = 'playground';
    if (view !== 'playground' || ROBOTS.indexOf(robot) === -1) robot = null;

    var same = state.view === view && state.robot === robot;
    state.view = view;
    state.robot = robot;
    paint();

    if (!same) {
      frame.src = srcFor(view, robot);
      try {
        history.replaceState(null, '', '#' + view + (robot ? '/' + robot : ''));
      } catch (e) {
        /* file:// or sandboxed; not needed */
      }
    }
    focusFrame();
  }

  frame.addEventListener('load', focusFrame);

  rows.forEach(function (row) {
    row.addEventListener('click', function () {
      if (row.dataset.robot) nav('playground', row.dataset.robot);
      else nav(row.dataset.view);
    });
  });

  // Arrow keys walk the source list across section boundaries, like an outline view.
  list.addEventListener('keydown', function (event) {
    var target = event.target;
    if (!target.classList || !target.classList.contains('source-list__row')) return;
    var index = rows.indexOf(target);
    var next = null;
    if (event.key === 'ArrowDown') next = rows[Math.min(rows.length - 1, index + 1)];
    else if (event.key === 'ArrowUp') next = rows[Math.max(0, index - 1)];
    else if (event.key === 'Home') next = rows[0];
    else if (event.key === 'End') next = rows[rows.length - 1];
    if (next) {
      event.preventDefault();
      next.focus();
    }
  });

  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'macgym:nav') nav(data.view, data.robot);
    else if (data.type === 'macgym:present') present();
  });

  // Browser fallback for the Electron menu shortcuts (when the shell itself has focus).
  window.addEventListener('keydown', function (event) {
    if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return;
    var view = { 1: 'playground', 2: 'slides', 3: 'viewer', 4: 'docs' }[event.key];
    if (view) {
      event.preventDefault();
      nav(view);
    }
  });

  function present() {
    nav('slides');
  }

  /* --- hooks for the Electron main process -------------------------------- */

  window.macgymNav = function (view, robot) {
    nav(view, robot);
  };

  window.macgymFullscreen = function (on) {
    document.body.classList.toggle('is-fullscreen', !!on);
    // Full screen while on Slides is presenting: the deck takes the whole display.
    document.body.classList.toggle('is-presenting', !!on && state.view === 'slides');
    focusFrame();
  };

  /* --- boot --------------------------------------------------------------- */

  var hash = (location.hash || '').replace(/^#/, '').split('/');
  var params = new URLSearchParams(location.search);
  nav(params.get('view') || hash[0] || 'playground', params.get('robot') || hash[1] || null);
})();
