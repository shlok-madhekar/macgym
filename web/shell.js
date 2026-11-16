/*
 * MacGym window shell: sidebar selection -> iframe src, plus the Xcode-style
 * toolbar (Run / Stop, scheme, activity view).
 *
 * Contract with the playground iframe (XCODE.md):
 *   shell  -> iframe  {type:'macgym:cmd', cmd:'run'|'stop'|'setCopies', value?}
 *   iframe -> shell   {type:'macgym:status', view, state, robot, detail, copies, headline, sub, best}
 * Electron's menu calls window.macgymNav(view) and window.macgymCmd(cmd).
 */
(function () {
  'use strict';

  var TITLES = { playground: 'Playground', slides: 'Slides', viewer: 'Viewer', docs: 'Docs' };
  var ROBOTS = ['walker', 'cheetah', 'hopper', 'tripod', 'worm'];
  var COPIES = [1024, 4096, 16384];

  var $ = function (id) {
    return document.getElementById(id);
  };
  var frame = $('view');
  var list = $('list');
  var rows = Array.prototype.slice.call(document.querySelectorAll('.source-list__row'));
  var runBtn = $('tb-run');
  var stopBtn = $('tb-stop');
  var robotText = $('tb-robot');
  var robotSelect = $('tb-robot-select');
  var copiesText = $('tb-copies');
  var copiesSelect = $('tb-copies-select');
  var activity = $('activity');
  var actLead = $('act-lead');
  var actSep = $('act-sep');
  var actText = $('act-text');
  var actTrail = $('act-trail');

  var state = { view: null, robot: null };
  // What the iframe last told us. `robot` here is whatever is actually loaded,
  // which can differ from the sidebar pick once the user edits the file.
  var status = { state: 'idle', robot: null, copies: 4096 };

  function fmt(n) {
    return Number(n).toLocaleString('en-US');
  }

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
    // inside the frame, not on the control that was just clicked.
    try {
      frame.focus();
      if (frame.contentWindow) frame.contentWindow.focus();
    } catch (e) {
      /* frame not ready yet; the load handler tries again */
    }
  }

  function currentRobot() {
    return status.robot || state.robot || null;
  }

  /* --- painting ----------------------------------------------------------- */

  function paintSidebar() {
    var robot = state.view === 'playground' ? currentRobot() : null;
    rows.forEach(function (row) {
      if (row.dataset.view) {
        if (row.dataset.view === state.view) row.setAttribute('aria-current', 'page');
        else row.removeAttribute('aria-current');
      } else if (row.dataset.robot === robot) row.setAttribute('data-current-robot', '');
      else row.removeAttribute('data-current-robot');
    });
  }

  function setActivity(mode, lead, text, trail) {
    activity.dataset.state = mode;
    actLead.textContent = lead || '';
    actSep.hidden = !lead;
    actText.textContent = text || '';
    actTrail.textContent = trail || '';
    activity.title = [lead, text, trail].filter(Boolean).join('  ');
  }

  function paintToolbar() {
    var name = TITLES[state.view] || 'MacGym';
    document.title = 'MacGym: ' + name;
    document.body.dataset.view = state.view;

    if (state.view !== 'playground') {
      setActivity('title', '', name, '');
      return;
    }

    var robot = currentRobot();
    robotText.textContent = robot || 'robot';
    if (robot && ROBOTS.indexOf(robot) !== -1) robotSelect.value = robot;
    else robotSelect.selectedIndex = -1;

    var copies = Number(status.copies) || 4096;
    copiesText.textContent = fmt(copies) + ' copies';
    if (COPIES.indexOf(copies) !== -1) copiesSelect.value = String(copies);
    else copiesSelect.selectedIndex = -1;

    var busy = status.state === 'building' || status.state === 'training';
    stopBtn.disabled = !busy;
    runBtn.setAttribute('aria-pressed', busy ? 'true' : 'false');

    var who = robot || 'robot';
    if (status.state === 'building') {
      setActivity('building', '', 'Building ' + who + '\u2026', '');
    } else if (status.state === 'training') {
      var line = 'Training ' + who;
      if (status.headline) line += ': ' + status.headline + (status.sub ? ' ' + status.sub : '');
      setActivity('training', '', line, status.best || '');
    } else {
      setActivity('idle', 'MacGym', status.detail ? 'Ready \u00b7 ' + status.detail : 'Ready', '');
    }
  }

  function paint() {
    paintSidebar();
    paintToolbar();
  }

  /* --- navigation --------------------------------------------------------- */

  function nav(view, robot) {
    if (!TITLES[view]) view = 'playground';
    if (view !== 'playground' || ROBOTS.indexOf(robot) === -1) robot = null;

    // Re-selecting Playground itself keeps whatever robot is loaded.
    var same = state.view === view && (robot === null || robot === currentRobot());
    if (!same) {
      status = { state: 'idle', robot: robot, copies: status.copies };
      state.robot = robot;
    }
    state.view = view;
    paint();

    if (!same) {
      frame.src = srcFor(view, robot);
      try {
        history.replaceState(null, '', '#' + view + (robot ? '/' + robot : ''));
      } catch (e) {
        /* not needed */
      }
    }
    focusFrame();
  }

  function cmd(name, value) {
    if (state.view !== 'playground') {
      if (name !== 'run') return;
      nav('playground');
    }
    var msg = { type: 'macgym:cmd', cmd: name };
    if (value !== undefined) msg.value = value;
    try {
      frame.contentWindow.postMessage(msg, '*');
    } catch (e) {
      /* frame still loading */
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

  runBtn.addEventListener('click', function () {
    cmd('run');
  });
  stopBtn.addEventListener('click', function () {
    cmd('stop');
  });
  robotSelect.addEventListener('change', function () {
    nav('playground', robotSelect.value);
  });
  copiesSelect.addEventListener('change', function () {
    var n = Number(copiesSelect.value);
    status.copies = n;
    paintToolbar();
    cmd('setCopies', n);
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
    if (data.type === 'macgym:nav') {
      nav(data.view, data.robot);
    } else if (data.type === 'macgym:status') {
      // Only the frame we are showing may drive the toolbar.
      if (event.source && event.source !== frame.contentWindow && event.source !== window) return;
      if (data.view && data.view !== 'playground') return;
      status = {
        state: data.state || 'idle',
        robot: data.robot ? String(data.robot).replace(/\.py$/, '') : status.robot,
        detail: data.detail || '',
        copies: Number(data.copies) || status.copies,
        headline: data.headline || '',
        sub: data.sub || '',
        best: data.best || '',
      };
      if (state.view === 'playground') paint();
    }
  });

  // Same shortcuts as the Electron menu, for when the shell itself has focus
  // (or the shell is open in a plain browser).
  window.addEventListener('keydown', function (event) {
    if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return;
    var view = { 1: 'playground', 2: 'slides', 3: 'viewer', 4: 'docs' }[event.key];
    if (view) {
      event.preventDefault();
      nav(view);
    } else if (event.key === 'r' || event.key === 'R') {
      event.preventDefault();
      cmd('run');
    } else if (event.key === '.') {
      event.preventDefault();
      cmd('stop');
    }
  });

  /* --- hooks for the Electron main process -------------------------------- */

  window.macgymNav = function (view, robot) {
    nav(view, robot);
  };

  window.macgymCmd = function (name, value) {
    cmd(name, value);
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
