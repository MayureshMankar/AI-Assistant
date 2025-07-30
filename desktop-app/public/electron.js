const { app, BrowserWindow, Menu, shell, ipcMain, dialog } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log');
const Store = require('electron-store');

// Configure logging
log.transports.file.level = 'info';
autoUpdater.logger = log;

// Initialize electron store for settings
const store = new Store({
  defaults: {
    windowBounds: { width: 1400, height: 900 },
    openrouterApiKey: '',
    selectedModel: 'deepseek/deepseek-chat-v3-0324:free',
    theme: 'dark',
    autoSave: true,
    contextAwareness: true
  }
});

let mainWindow;

function createWindow() {
  // Get stored window bounds
  const { width, height } = store.get('windowBounds');

  // Create the browser window
  mainWindow = new BrowserWindow({
    width: width,
    height: height,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, '../build/icon.png'),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false, // Don't show until ready
    backgroundColor: '#1e1e1e' // Dark theme background
  });

  // Load the app
  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Focus on the app
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Save window bounds on resize/move
  mainWindow.on('resize', saveWindowBounds);
  mainWindow.on('move', saveWindowBounds);

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Set up the menu
  createMenu();
}

function saveWindowBounds() {
  if (mainWindow) {
    store.set('windowBounds', mainWindow.getBounds());
  }
}

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Session',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu-new-session');
          }
        },
        {
          label: 'Open File',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(mainWindow, {
              properties: ['openFile', 'multiSelections'],
              filters: [
                { name: 'Code Files', extensions: ['js', 'ts', 'py', 'java', 'cpp', 'c', 'go', 'rs', 'php', 'rb'] },
                { name: 'Text Files', extensions: ['txt', 'md', 'json', 'xml', 'yaml', 'yml'] },
                { name: 'All Files', extensions: ['*'] }
              ]
            });
            
            if (!result.canceled) {
              mainWindow.webContents.send('menu-open-files', result.filePaths);
            }
          }
        },
        { type: 'separator' },
        {
          label: 'Export Chat',
          accelerator: 'CmdOrCtrl+E',
          click: () => {
            mainWindow.webContents.send('menu-export-chat');
          }
        },
        { type: 'separator' },
        {
          label: 'Preferences',
          accelerator: 'CmdOrCtrl+,',
          click: () => {
            createPreferencesWindow();
          }
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectall' }
      ]
    },
    {
      label: 'AI Assistant',
      submenu: [
        {
          label: 'Open Chat',
          accelerator: 'CmdOrCtrl+Shift+A',
          click: () => {
            mainWindow.webContents.send('menu-open-chat');
          }
        },
        {
          label: 'Explain Code',
          accelerator: 'CmdOrCtrl+Shift+E',
          click: () => {
            mainWindow.webContents.send('menu-explain-code');
          }
        },
        {
          label: 'Debug Code',
          accelerator: 'CmdOrCtrl+Shift+D',
          click: () => {
            mainWindow.webContents.send('menu-debug-code');
          }
        },
        {
          label: 'Refactor Code',
          accelerator: 'CmdOrCtrl+Shift+R',
          click: () => {
            mainWindow.webContents.send('menu-refactor-code');
          }
        },
        { type: 'separator' },
        {
          label: 'Switch Model',
          click: () => {
            mainWindow.webContents.send('menu-switch-model');
          }
        },
        {
          label: 'Clear History',
          click: () => {
            mainWindow.webContents.send('menu-clear-history');
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
        { type: 'separator' },
        {
          label: 'Toggle Theme',
          accelerator: 'CmdOrCtrl+Shift+T',
          click: () => {
            const currentTheme = store.get('theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            store.set('theme', newTheme);
            mainWindow.webContents.send('menu-toggle-theme', newTheme);
          }
        }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About AI Coding Assistant Pro',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About AI Coding Assistant Pro',
              message: 'AI Coding Assistant Pro v2.0.0',
              detail: 'Professional AI-powered coding assistant with VS Code-like interface and free models.\n\nBuilt with Electron, React, and OpenRouter.\n\n© 2024 AI Assistant Pro Team'
            });
          }
        },
        {
          label: 'Check for Updates',
          click: () => {
            autoUpdater.checkForUpdatesAndNotify();
          }
        },
        { type: 'separator' },
        {
          label: 'OpenRouter Models',
          click: () => {
            shell.openExternal('https://openrouter.ai/models');
          }
        },
        {
          label: 'Get API Key',
          click: () => {
            shell.openExternal('https://openrouter.ai/keys');
          }
        },
        {
          label: 'Report Issue',
          click: () => {
            shell.openExternal('https://github.com/your-username/ai-coding-assistant/issues');
          }
        }
      ]
    }
  ];

  // macOS specific menu adjustments
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideothers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });

    // Window menu
    template[5].submenu = [
      { role: 'close' },
      { role: 'minimize' },
      { role: 'zoom' },
      { type: 'separator' },
      { role: 'front' }
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

function createPreferencesWindow() {
  const prefsWindow = new BrowserWindow({
    width: 600,
    height: 500,
    parent: mainWindow,
    modal: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    title: 'Preferences',
    resizable: false,
    minimizable: false,
    maximizable: false
  });

  // Load preferences page
  const prefsUrl = isDev 
    ? 'http://localhost:3000/preferences' 
    : `file://${path.join(__dirname, '../build/index.html#preferences')}`;
  
  prefsWindow.loadURL(prefsUrl);
}

// IPC handlers
ipcMain.handle('get-store-value', (event, key) => {
  return store.get(key);
});

ipcMain.handle('set-store-value', (event, key, value) => {
  store.set(key, value);
});

ipcMain.handle('get-all-settings', () => {
  return store.store;
});

ipcMain.handle('reset-settings', () => {
  store.clear();
  return store.store;
});

// File system operations
ipcMain.handle('read-file', async (event, filePath) => {
  try {
    const fs = require('fs').promises;
    const content = await fs.readFile(filePath, 'utf8');
    return { success: true, content };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('write-file', async (event, filePath, content) => {
  try {
    const fs = require('fs').promises;
    await fs.writeFile(filePath, content, 'utf8');
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('save-file-dialog', async (event, options) => {
  try {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
  } catch (error) {
    return { canceled: true, error: error.message };
  }
});

// App event handlers
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });

  // Check for updates
  if (!isDev) {
    autoUpdater.checkForUpdatesAndNotify();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Auto-updater events
autoUpdater.on('checking-for-update', () => {
  log.info('Checking for update...');
});

autoUpdater.on('update-available', (info) => {
  log.info('Update available.');
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'Update Available',
    message: `A new version (${info.version}) is available. It will be downloaded in the background.`,
    buttons: ['OK']
  });
});

autoUpdater.on('update-not-available', (info) => {
  log.info('Update not available.');
});

autoUpdater.on('error', (err) => {
  log.error('Error in auto-updater:', err);
});

autoUpdater.on('download-progress', (progressObj) => {
  let log_message = `Download speed: ${progressObj.bytesPerSecond}`;
  log_message = `${log_message} - Downloaded ${progressObj.percent}%`;
  log_message = `${log_message} (${progressObj.transferred}/${progressObj.total})`;
  log.info(log_message);
});

autoUpdater.on('update-downloaded', (info) => {
  log.info('Update downloaded');
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'Update Ready',
    message: 'Update downloaded. The application will restart to apply the update.',
    buttons: ['Restart Now', 'Later']
  }).then((result) => {
    if (result.response === 0) {
      autoUpdater.quitAndInstall();
    }
  });
});