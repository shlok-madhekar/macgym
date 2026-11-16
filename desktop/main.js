/*
 * MacGym desktop shell.
 *
 * A real native macOS window in the Probe0 style: hiddenInset title bar,
 * traffic lights at {19,13}, AppKit 'sidebar' vibrancy behind a transparent
 * web layer, one opaque content pane. The renderer is the MacGym server's own
 * /shell.html over loopback; this process only owns the window, the menu and
 * (when nothing is already listening) the server's lifetime.
 */
const { app, BrowserWindow, Menu, nativeTheme, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');

const PORT = Number(process.env.MACGYM_PORT || 8765);
const ORIGIN = `http://localhost:${PORT}`;
const PYTHON_CANDIDATES = [
  process.env.MACGYM_PYTHON,
  '/Users/shlok/.aside/runtime/bin/python3',
  path.join(os.homedir(), '.aside/runtime/bin/python3'),
].filter(Boolean);

app.setName('MacGym');
nativeTheme.themeSource = 'dark';

let win = null;
let serverProc = null; // set only when WE spawned it

/* --- locate the repo ------------------------------------------------------ */

function findRoot() {
  if (process.env.MACGYM_ROOT) return process.env.MACGYM_ROOT;
  let dir = __dirname;
  for (let i = 0; i < 8; i += 1) {
    if (fs.existsSync(path.join(dir, 'macgym', 'server.py'))) return dir;
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return path.join(os.homedir(), 'macgym');
}

/* --- server --------------------------------------------------------------- */

function ping(timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(`${ORIGIN}/shell.html`, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode !== undefined && res.statusCode < 500);
    });
    req.on('timeout', () => req.destroy());
    req.on('error', () => resolve(false));
  });
}

async function ensureServer() {
  if (await ping()) return true;

  const root = findRoot();
  const python = PYTHON_CANDIDATES.find((p) => fs.existsSync(p)) || 'python3';
  const logPath = path.join(os.tmpdir(), 'macgym-desktop-server.log');
  const log = fs.openSync(logPath, 'a');
  serverProc = spawn(python, ['-m', 'macgym.server', '--port', String(PORT)], {
    cwd: root,
    stdio: ['ignore', log, log],
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });
  serverProc.on('exit', () => {
    serverProc = null;
  });

  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    if (await ping(1000)) return true;
    if (serverProc === null) return false; // died during startup
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

function stopServer() {
  if (serverProc && !serverProc.killed) {
    try {
      serverProc.kill('SIGTERM');
    } catch {
      /* already gone */
    }
  }
  serverProc = null;
}

/* --- window --------------------------------------------------------------- */

function js(code) {
  if (!win || win.isDestroyed()) return;
  win.webContents.executeJavaScript(code, true).catch(() => {});
}

function nav(view) {
  if (!win || win.isDestroyed()) return;
  if (win.isMinimized()) win.restore();
  win.show();
  js(`window.macgymNav && window.macgymNav(${JSON.stringify(view)})`);
}

function present() {
  if (!win || win.isDestroyed()) return;
  nav('slides');
  win.setFullScreen(true);
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 680,

    /* macOS chrome, same as Probe0's main window */
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 19, y: 13 },
    vibrancy: 'sidebar',
    visualEffectState: 'active',
    backgroundColor: '#00000000',
    transparent: false,
    roundedCorners: true,
    frame: true,
    title: 'MacGym',

    show: false,

    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webviewTag: false,
      spellcheck: false,
    },
  });

  win.once('ready-to-show', () => win.show());

  const focus = (focused) =>
    js(`document.body && document.body.classList.toggle('window-inactive', ${!focused})`);
  win.on('blur', () => focus(false));
  win.on('focus', () => focus(true));

  const fullscreen = (on) => js(`window.macgymFullscreen && window.macgymFullscreen(${on})`);
  win.on('enter-full-screen', () => fullscreen(true));
  win.on('leave-full-screen', () => fullscreen(false));

  // Anything that is not our own loopback origin opens in the real browser.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith(ORIGIN)) return { action: 'allow' };
    shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(ORIGIN)) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  win.on('closed', () => {
    win = null;
  });

  win.loadURL(`${ORIGIN}/shell.html`);
}

function failWindow(message) {
  win = new BrowserWindow({
    width: 560,
    height: 300,
    titleBarStyle: 'hiddenInset',
    vibrancy: 'sidebar',
    visualEffectState: 'active',
    backgroundColor: '#00000000',
    title: 'MacGym',
  });
  const html = `<!doctype html><html><body style="margin:0;padding:64px 32px 0;background:transparent;color:#e8e8e8;font:13px/18px -apple-system,system-ui;-webkit-app-region:drag"><div style="font-size:17px;font-weight:600;margin-bottom:8px">MacGym server did not start</div><div style="color:#9a9a9a">${message}</div></body></html>`;
  win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
}

/* --- menu ----------------------------------------------------------------- */

function buildMenu() {
  const view = (label, id, accelerator) => ({ label, accelerator, click: () => nav(id) });
  const template = [
    {
      label: 'MacGym',
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { role: 'fileMenu' },
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        view('Playground', 'playground', 'CmdOrCtrl+1'),
        view('Slides', 'slides', 'CmdOrCtrl+2'),
        view('Viewer', 'viewer', 'CmdOrCtrl+3'),
        view('Docs', 'docs', 'CmdOrCtrl+4'),
        { type: 'separator' },
        {
          label: 'Present Slides Full Screen',
          accelerator: 'CmdOrCtrl+Shift+F',
          click: present,
        },
        { type: 'separator' },
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [{ label: 'MacGym Docs', click: () => nav('docs') }],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

/* --- lifecycle ------------------------------------------------------------ */

async function start() {
  buildMenu();
  const ok = await ensureServer();
  if (ok) createWindow();
  else
    failWindow(
      `Nothing answered on ${ORIGIN} and <code>python3 -m macgym.server --port ${PORT}</code> did not come up. See ${path.join(os.tmpdir(), 'macgym-desktop-server.log')}.`,
    );
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });
  app.whenReady().then(start);
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) start();
  });
  app.on('window-all-closed', () => app.quit());
  app.on('will-quit', stopServer);
  process.on('exit', stopServer);
  for (const sig of ['SIGINT', 'SIGTERM']) {
    process.on(sig, () => {
      stopServer();
      app.quit();
    });
  }
}
