# 🚀 AI Coding Assistant Pro - Installation & Usage Guide

## 📋 Overview

Your AI Coding Assistant has been completely transformed with **3 different ways to use it**:

1. **🌐 Web App** - Browser-based VS Code-like IDE
2. **🔌 VS Code Extension** - Native VS Code integration
3. **💻 Desktop App** - Standalone Electron application

All versions use **100% FREE models** from OpenRouter with **auto language detection** and **improved context handling**.

---

## 🎯 Key Features Added

### ✅ **Fixed Issues:**
- **Chat Display**: Messages now show properly with correct height and scrolling
- **Context Problems**: Conversation history and file context now maintained
- **Language Selection**: Automatically detects programming languages

### 🆕 **New Features:**
- **8 Free AI Models**: DeepSeek V3, Qwen3 variants, Gemini 2.0 Flash
- **VS Code-like Interface**: Activity bar, explorer, editor, status bar
- **Auto Language Detection**: No manual language selection needed
- **Context Awareness**: Maintains conversation history and file context
- **Keyboard Shortcuts**: Professional hotkeys like Cursor IDE
- **File Management**: Upload, open, and manage code files
- **Code Actions**: Explain, debug, refactor code with AI

---

## 🔧 Installation Options

## Option 1: 🌐 Web Application

### Quick Start
```bash
# Clone and run the web app
cd frontend
npm install
npm start

# In another terminal for backend
cd backend
pip install -r requirements.txt
python main.py
```

### Access
- **URL**: http://localhost:3000
- **Backend**: http://localhost:8000
- **Features**: Full VS Code-like interface in browser

---

## Option 2: 🔌 VS Code Extension

### Installation
```bash
# Build the extension
cd vscode-extension
npm install
npm run compile
npm run package
```

### Install in VS Code
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Click "..." → "Install from VSIX"
4. Select the generated `.vsix` file

### Setup
1. **Get Free API Key**: https://openrouter.ai/keys
2. **Configure Extension**:
   - Open Settings (Ctrl+,)
   - Search "AI Assistant"
   - Set your OpenRouter API key
   - Choose default model (all are free!)

### Usage
| Command | Shortcut | Description |
|---------|----------|-------------|
| Open AI Chat | `Ctrl+Shift+A` | Open chat panel |
| Explain Code | `Ctrl+Shift+E` | Explain selected code |
| Debug Code | `Ctrl+Shift+D` | Debug selected code |
| Refactor Code | `Ctrl+Shift+R` | Refactor selected code |
| Generate Code | `Ctrl+Shift+G` | Generate new code |

---

## Option 3: 💻 Desktop Application

### Build Instructions
```bash
# Setup desktop app
cd desktop-app
npm install

# Development
npm run dev

# Build for distribution
npm run dist           # Current platform
npm run dist-win       # Windows
npm run dist-mac       # macOS  
npm run dist-linux     # Linux
```

### Features
- **Native App**: Runs independently of browser
- **Auto Updates**: Built-in update system
- **File System**: Direct file access
- **System Integration**: Native menus and shortcuts
- **Settings Storage**: Persistent configuration

---

## 🎮 Free AI Models Available

| Model | Type | Context | Best For |
|-------|------|---------|----------|
| **DeepSeek V3** | Reasoning | 32K | Complex problems, reasoning |
| **DeepSeek R1 Distill 70B** | Reasoning | 8K | High performance coding |
| **Qwen3 30B A3B** | Balanced | 40K | General coding, thinking mode |
| **Qwen3 14B** | Balanced | 40K | Reasoning and dialogue |
| **Qwen3 8B** | Fast | 40K | Quick coding tasks |
| **Qwen3 4B** | Fast | 40K | Lightweight responses |
| **QwQ 32B** | Reasoning | 32K | Complex problem solving |
| **Gemini 2.0 Flash** | Multimodal | 1M | Large context, fast responses |

---

## ⚙️ Configuration

### 🔑 Required: OpenRouter API Key
1. **Get Free Key**: https://openrouter.ai/keys
2. **No Credit Card Required**
3. **Generous Free Tier**: Thousands of requests/month

### Web App Configuration
```bash
# Backend environment
cp backend/.env.example backend/.env
# Add your OpenRouter API key to .env
OPENROUTER_API_KEY=your_key_here
```

### VS Code Extension Configuration
```json
{
  "aiAssistant.openrouterApiKey": "your_key_here",
  "aiAssistant.defaultModel": "deepseek/deepseek-chat-v3-0324:free",
  "aiAssistant.autoDetectLanguage": true,
  "aiAssistant.contextAwareness": true
}
```

### Desktop App Configuration
- **Settings**: File → Preferences (Ctrl+,)
- **API Key**: Set in preferences
- **Auto-saved**: Settings persist between sessions

---

## 🎯 Usage Examples

### 💬 Smart Chat
```
You: "How do I optimize this Python function?"
AI: Analyzes your code with context and provides specific optimizations
```

### 🔍 Code Analysis
```
1. Select code in editor
2. Press Ctrl+Shift+E (or right-click → AI Assistant → Explain Code)
3. Get detailed explanation with best practices
```

### 🐛 Debug Code
```
1. Select buggy code
2. Press Ctrl+Shift+D
3. AI identifies issues and suggests fixes
```

### ⚡ Refactor Code
```
1. Select code to improve
2. Press Ctrl+Shift+R  
3. Get optimized, cleaner code
```

---

## 🔧 Advanced Features

### 📁 File Context
- **Auto-detection**: AI knows what files you have open
- **Smart Context**: Includes relevant file content in requests
- **Multi-file**: Understands relationships between files

### 🧠 Conversation Memory
- **Persistent**: Remembers conversation across the session
- **Context-aware**: References previous questions and answers
- **Smart**: Automatically maintains relevant context

### 🎨 Auto Language Detection
```javascript
// Auto-detects as JavaScript
function example() { return true; }

# Auto-detects as Python  
def example():
    return True
```

---

## 🚀 Development Workflow

### 1. **Setup Phase**
```bash
# Choose your preferred option
Web App:    cd frontend && npm start
Extension:  Install in VS Code
Desktop:    cd desktop-app && npm run dev
```

### 2. **Configure API**
- Get free OpenRouter API key
- Set in your chosen platform
- Test with a simple question

### 3. **Start Coding**
- Open your project files
- Use AI for explanations, debugging, refactoring
- Leverage auto language detection
- Maintain conversation context

---

## 📚 Tips & Best Practices

### 🎯 **Getting Best Results**
1. **Be Specific**: "Debug this Python function for memory leaks" vs "fix this"
2. **Provide Context**: Select relevant code before asking
3. **Use Follow-ups**: AI remembers conversation context
4. **Leverage File Context**: AI sees your open files

### ⌨️ **Keyboard Shortcuts**
- `Ctrl+Shift+A`: Open AI chat
- `Ctrl+Shift+E`: Explain selected code
- `Ctrl+Shift+D`: Debug code
- `Ctrl+Shift+R`: Refactor code
- `Ctrl+Enter`: Send message in chat

### 🔧 **Performance Tips**
- **Use appropriate model**: Qwen3-4B for simple tasks, DeepSeek V3 for complex
- **Manage context**: Clear history for fresh starts
- **File selection**: AI auto-includes relevant open files

---

## 🐛 Troubleshooting

### Common Issues

#### "API Key not configured"
- **Web App**: Check `backend/.env` file
- **VS Code**: Settings → AI Assistant → OpenRouter API Key
- **Desktop**: File → Preferences → API Key

#### "Model not responding"
- Try different model from the dropdown
- Check internet connection
- Verify API key is valid

#### "Context not working"
- Ensure files are open in editor
- Check context awareness setting is enabled
- Verify auto language detection is on

#### "Chat display issues"
- Clear browser cache (web app)
- Restart VS Code (extension)
- Restart app (desktop)

---

## 🆘 Support

### 📖 Documentation
- **OpenRouter Models**: https://openrouter.ai/models
- **API Documentation**: https://openrouter.ai/docs
- **Free API Key**: https://openrouter.ai/keys

### 🐛 Report Issues
- **GitHub**: https://github.com/your-username/ai-coding-assistant/issues
- **Include**: Platform, model used, error message
- **Logs**: Check browser console (web) or VS Code output (extension)

### 💡 Feature Requests
- **GitHub Discussions**: Share ideas and feedback
- **Community**: Join discussions with other users

---

## 🎉 What's New in v2.0

### ✅ **Major Improvements**
- **Fixed Chat Display**: Messages show completely, proper scrolling
- **Context Handling**: Maintains conversation and file context
- **Auto Language Detection**: No manual language selection
- **Free Models Only**: All 8 models are completely free
- **VS Code Interface**: Professional IDE-like experience
- **3 Installation Options**: Web, extension, desktop

### 🆕 **New Features**
- **File Explorer**: Upload and manage code files
- **Code Actions**: Right-click context menu for AI features
- **Keyboard Shortcuts**: Professional hotkeys
- **Settings Management**: Persistent configuration
- **Auto Updates**: Desktop app auto-updates

### 🔧 **Under the Hood**
- **Better Error Handling**: Graceful error messages
- **Performance**: Optimized API calls and context management
- **Security**: Secure API key storage
- **Reliability**: Improved connection handling

---

## 🎯 Next Steps

1. **Choose Your Platform**: Web app, VS Code extension, or desktop app
2. **Get Free API Key**: https://openrouter.ai/keys (no credit card required)
3. **Install & Configure**: Follow the installation guide above
4. **Start Coding**: Use AI to enhance your development workflow
5. **Explore Features**: Try different models and AI commands
6. **Share Feedback**: Help improve the assistant

---

## 🏆 Success! You now have a professional AI coding assistant with:

- ✅ **Working free models** (no more API costs)
- ✅ **Fixed chat display** (full message visibility)
- ✅ **Auto language detection** (no manual selection)
- ✅ **Context awareness** (remembers conversation & files)
- ✅ **VS Code-like interface** (professional experience)
- ✅ **Multiple platforms** (web, extension, desktop)
- ✅ **Professional features** (shortcuts, file management, settings)

**Happy coding with your AI assistant! 🚀**