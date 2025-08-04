"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const axios_1 = __importDefault(require("axios"));
// Free models available
const FREE_MODELS = [
    'deepseek/deepseek-chat-v3-0324:free',
    'deepseek/deepseek-r1-distill-llama-70b:free',
    'qwen/qwen3-30b-a3b:free',
    'qwen/qwen3-14b:free',
    'qwen/qwen3-8b:free',
    'qwen/qwen3-4b:free',
    'qwen/qwq-32b:free',
    'google/gemini-2.0-flash-exp:free'
];
// Language detection patterns
const LANGUAGE_PATTERNS = {
    python: [/def\s+\w+/, /import\s+\w+/, /from\s+\w+\s+import/, /class\s+\w+/, /#.*python/i],
    javascript: [/function\s+\w+/, /const\s+\w+/, /let\s+\w+/, /var\s+\w+/, /=>\s*{/, /console\.log/],
    typescript: [/interface\s+\w+/, /type\s+\w+/, /:\s*string/, /:\s*number/, /:\s*boolean/],
    java: [/public\s+class/, /public\s+static\s+void\s+main/, /System\.out\.println/, /import\s+java\./],
    cpp: [/#include\s*</, /using\s+namespace/, /int\s+main/, /cout\s*<</, /std::/],
    c: [/#include\s*</, /int\s+main/, /printf\s*\(/, /scanf\s*\(/],
    go: [/package\s+main/, /func\s+main/, /import\s+\(/, /fmt\.Print/],
    rust: [/fn\s+main/, /let\s+mut/, /use\s+std::/, /println!/],
    php: [/<\?php/, /echo\s+/, /\$\w+/, /function\s+\w+/],
    ruby: [/def\s+\w+/, /puts\s+/, /class\s+\w+/, /require\s+/],
    bash: [/#!\/bin\/bash/, /echo\s+/, /if\s+\[/, /for\s+\w+\s+in/],
    html: [/<html/, /<head/, /<body/, /<div/, /<script/],
    css: [/\.\w+\s*{/, /#\w+\s*{/, /@media/, /font-family:/],
    sql: [/SELECT\s+/i, /FROM\s+/i, /WHERE\s+/i, /INSERT\s+INTO/i, /CREATE\s+TABLE/i]
};
// Auto-detect programming language
function detectLanguage(code) {
    if (!code || code.trim().length < 10)
        return 'text';
    const lines = code.split('\n').slice(0, 10);
    const scores = {};
    for (const [lang, patterns] of Object.entries(LANGUAGE_PATTERNS)) {
        scores[lang] = 0;
        for (const pattern of patterns) {
            for (const line of lines) {
                if (pattern.test(line)) {
                    scores[lang] += 1;
                }
            }
        }
    }
    const maxScore = Math.max(...Object.values(scores));
    if (maxScore === 0)
        return 'text';
    const detectedLang = Object.keys(scores).find(lang => scores[lang] === maxScore);
    return detectedLang || 'text';
}
// Chat Webview Provider
class ChatWebviewProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
    }
    resolveWebviewView(webviewView) {
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
    }
    _getHtmlForWebview(webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            margin: 0;
            padding: 10px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 10px;
            padding: 10px;
            border: 1px solid var(--vscode-panel-border);
            border-radius: 5px;
        }

        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }

        .user-message {
            background-color: var(--vscode-inputOption-activeBackground);
            margin-left: 20px;
        }

        .ai-message {
            background-color: var(--vscode-editor-selectionBackground);
            margin-right: 20px;
        }

        .message-header {
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 12px;
            opacity: 0.8;
        }

        .input-container {
            display: flex;
            gap: 5px;
        }

        #messageInput {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            border-radius: 3px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            font-family: inherit;
            resize: vertical;
            min-height: 60px;
        }

        button {
            padding: 8px 15px;
            border: none;
            border-radius: 3px;
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            cursor: pointer;
            font-family: inherit;
        }

        button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .toolbar {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .model-info {
            font-size: 11px;
            opacity: 0.7;
        }

        pre {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
            font-family: var(--vscode-editor-font-family);
        }

        code {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 2px 4px;
            border-radius: 2px;
            font-family: var(--vscode-editor-font-family);
        }

        .loading {
            opacity: 0.7;
            font-style: italic;
        }

        .error {
            color: var(--vscode-errorForeground);
            background-color: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <div class="model-info">AI Assistant Pro • Free Models</div>
        <button id="clearBtn" onclick="clearHistory()">Clear</button>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="message ai-message">
            <div class="message-header">AI Assistant</div>
            <div>👋 Welcome! I'm your AI coding assistant. I can help you with:
            <br>• Code analysis and debugging
            <br>• Code generation and refactoring
            <br>• Best practices and explanations
            <br>• Multi-language support with auto-detection
            <br><br>What can I help you with today?</div>
        </div>
    </div>

    <div class="input-container">
        <textarea id="messageInput" placeholder="Ask me anything about coding... (Ctrl+Enter to send)" rows="3"></textarea>
        <div style="display: flex; flex-direction: column; gap: 5px;">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const vscode = acquireVsCodeApi();
        let isLoading = false;

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || isLoading) return;

            // Add user message to chat
            addMessage('You', message, 'user-message');
            input.value = '';

            // Show loading
            isLoading = true;
            const loadingDiv = addMessage('AI Assistant', 'Thinking...', 'ai-message loading');
            document.getElementById('sendBtn').disabled = true;

            // Send to extension
            vscode.postMessage({
                type: 'sendMessage',
                text: message
            });
        }

        function clearHistory() {
            document.getElementById('chatContainer').innerHTML = \`
                <div class="message ai-message">
                    <div class="message-header">AI Assistant</div>
                    <div>History cleared. How can I help you?</div>
                </div>
            \`;
            vscode.postMessage({ type: 'clearHistory' });
        }

        function addMessage(sender, content, className) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${className}\`;

            // Convert markdown to HTML for AI messages
            let htmlContent = content;
            if (className.includes('ai-message') && !className.includes('loading')) {
                htmlContent = convertMarkdownToHtml(content);
            }

            messageDiv.innerHTML = \`
                <div class="message-header">\${sender}</div>
                <div>\${htmlContent}</div>
            \`;

            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            return messageDiv;
        }

        function convertMarkdownToHtml(markdown) {
            // Simple markdown conversion for code blocks
            return markdown
                .replace(/\`\`\`(\\w+)?\\n([\\s\\S]*?)\\n\`\`\`/g, '<pre><code>$2</code></pre>')
                .replace(/\`([^\`]+)\`/g, '<code>$1</code>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/\\n/g, '<br>');
        }

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;

            switch (message.type) {
                case 'aiResponse':
                    // Remove loading message
                    const loadingMessages = document.querySelectorAll('.loading');
                    loadingMessages.forEach(msg => msg.remove());

                    // Add AI response
                    addMessage('AI Assistant', message.text, 'ai-message');
                    isLoading = false;
                    document.getElementById('sendBtn').disabled = false;
                    break;

                case 'error':
                    // Remove loading message
                    const errorLoadingMessages = document.querySelectorAll('.loading');
                    errorLoadingMessages.forEach(msg => msg.remove());

                    // Add error message
                    addMessage('Error', message.text, 'ai-message error');
                    isLoading = false;
                    document.getElementById('sendBtn').disabled = false;
                    break;

                case 'historyCleared':
                    break;
            }
        });

        // Handle keyboard shortcuts
        document.getElementById('messageInput').addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>`;
    }
}
// AI Assistant Provider
class AIAssistantProvider {
    constructor() {
        this.conversationHistory = [];
        this.config = this.loadConfig();
        this.sessionId = `session_${Date.now()}`;
    }
    loadConfig() {
        const config = vscode.workspace.getConfiguration('aiAssistant');
        return {
            openrouterApiKey: config.get('openrouterApiKey') || '',
            defaultModel: config.get('defaultModel') || 'deepseek/deepseek-chat-v3-0324:free',
            autoDetectLanguage: config.get('autoDetectLanguage') || true,
            contextAwareness: config.get('contextAwareness') || true,
            maxContextFiles: config.get('maxContextFiles') || 5,
            temperature: config.get('temperature') || 0.7,
            autoSaveConversations: config.get('autoSaveConversations') || true,
            showInlineCodeActions: config.get('showInlineCodeActions') || true
        };
    }
    sendMessage(message, context) {
        var _a, _b;
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.config.openrouterApiKey) {
                throw new Error('OpenRouter API key not configured. Please set it in VS Code settings.');
            }
            try {
                // Build context
                const contextData = yield this.buildContext(context);
                // Add to conversation history
                this.conversationHistory.push({ role: 'user', content: message });
                // Keep only last 20 messages for context
                if (this.conversationHistory.length > 20) {
                    this.conversationHistory = this.conversationHistory.slice(-20);
                }
                const response = yield axios_1.default.post('https://openrouter.ai/api/v1/chat/completions', {
                    model: this.config.defaultModel,
                    messages: [
                        {
                            role: 'system',
                            content: `You are an expert AI coding assistant integrated into VS Code. You help with coding, debugging, code analysis, and programming questions across all languages.
Context:
- Auto language detection: ${this.config.autoDetectLanguage}
- Context awareness: ${this.config.contextAwareness}
- Free model: ${this.config.defaultModel}
Guidelines:
- Provide detailed, practical coding help
- Use markdown code blocks with language specification
- Explain your reasoning and approach
- Be concise but thorough
- Maintain conversation context
- Focus on code quality and best practices
${contextData}`
                        },
                        ...this.conversationHistory.slice(-10),
                        { role: 'user', content: message }
                    ],
                    temperature: this.config.temperature,
                    max_tokens: 4000
                }, {
                    headers: {
                        'Authorization': `Bearer ${this.config.openrouterApiKey}`,
                        'Content-Type': 'application/json',
                        'HTTP-Referer': 'https://github.com/your-username/ai-coding-assistant',
                        'X-Title': 'AI Coding Assistant VS Code Extension'
                    }
                });
                const aiResponse = response.data.choices[0].message.content;
                // Add AI response to history
                this.conversationHistory.push({ role: 'assistant', content: aiResponse });
                return aiResponse;
            }
            catch (error) {
                if (((_a = error.response) === null || _a === void 0 ? void 0 : _a.status) === 401) {
                    throw new Error('Invalid OpenRouter API key. Please check your configuration.');
                }
                else if (((_b = error.response) === null || _b === void 0 ? void 0 : _b.status) === 429) {
                    throw new Error('Rate limit reached. Please try again later.');
                }
                else {
                    throw new Error(`AI request failed: ${error.message}`);
                }
            }
        });
    }
    buildContext(context) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.config.contextAwareness)
                return '';
            const contextParts = [];
            // Add active editor context
            const activeEditor = vscode.window.activeTextEditor;
            if (activeEditor) {
                const document = activeEditor.document;
                const language = this.config.autoDetectLanguage
                    ? detectLanguage(document.getText())
                    : document.languageId;
                contextParts.push(`Active File: ${document.fileName} (${language})`);
                const selection = activeEditor.selection;
                if (!selection.isEmpty) {
                    const selectedText = document.getText(selection);
                    contextParts.push(`Selected Code:\n\`\`\`${language}\n${selectedText}\n\`\`\``);
                }
            }
            // Add workspace context
            const workspaceFolders = vscode.workspace.workspaceFolders;
            if (workspaceFolders) {
                contextParts.push(`Workspace: ${workspaceFolders[0].name}`);
            }
            // Add open files context
            const openDocuments = vscode.workspace.textDocuments
                .filter(doc => !doc.isUntitled && doc.uri.scheme === 'file')
                .slice(0, this.config.maxContextFiles);
            if (openDocuments.length > 0) {
                contextParts.push(`Open Files (${openDocuments.length}):`);
                openDocuments.forEach(doc => {
                    const language = this.config.autoDetectLanguage
                        ? detectLanguage(doc.getText())
                        : doc.languageId;
                    contextParts.push(`- ${doc.fileName} (${language})`);
                });
            }
            return contextParts.join('\n');
        });
    }
}
// Extension activation
function activate(context) {
    const aiAssistant = new AIAssistantProvider();
    // Register chat webview provider
    const chatProvider = new ChatWebviewProvider(context.extensionUri);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider('aiAssistant.chatView', chatProvider));
    // Register commands
    const openChatCommand = vscode.commands.registerCommand('aiAssistant.openChat', () => {
        vscode.commands.executeCommand('aiAssistant.chatView.focus');
    });
    const explainCodeCommand = vscode.commands.registerCommand('aiAssistant.explainCode', () => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }
        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select some code to explain');
            return;
        }
        const language = detectLanguage(selectedText);
        try {
            const response = yield aiAssistant.sendMessage(`Please explain this ${language} code:\n\n\`\`\`${language}\n${selectedText}\n\`\`\``);
            // Show response in a new document
            const doc = yield vscode.workspace.openTextDocument({
                content: response,
                language: 'markdown'
            });
            vscode.window.showTextDocument(doc);
        }
        catch (error) {
            vscode.window.showErrorMessage(error.message);
        }
    }));
    const debugCodeCommand = vscode.commands.registerCommand('aiAssistant.debugCode', () => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }
        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select some code to debug');
            return;
        }
        const language = detectLanguage(selectedText);
        try {
            const response = yield aiAssistant.sendMessage(`Please debug and analyze this ${language} code for potential issues, bugs, and improvements:\n\n\`\`\`${language}\n${selectedText}\n\`\`\``);
            const doc = yield vscode.workspace.openTextDocument({
                content: response,
                language: 'markdown'
            });
            vscode.window.showTextDocument(doc);
        }
        catch (error) {
            vscode.window.showErrorMessage(error.message);
        }
    }));
    const refactorCodeCommand = vscode.commands.registerCommand('aiAssistant.refactorCode', () => __awaiter(this, void 0, void 0, function* () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }
        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select some code to refactor');
            return;
        }
        const language = detectLanguage(selectedText);
        try {
            const response = yield aiAssistant.sendMessage(`Please refactor and optimize this ${language} code for better performance, readability, and best practices:\n\n\`\`\`${language}\n${selectedText}\n\`\`\``);
            const doc = yield vscode.workspace.openTextDocument({
                content: response,
                language: 'markdown'
            });
            vscode.window.showTextDocument(doc);
        }
        catch (error) {
            vscode.window.showErrorMessage(error.message);
        }
    }));
    const generateCodeCommand = vscode.commands.registerCommand('aiAssistant.generateCode', () => __awaiter(this, void 0, void 0, function* () {
        const prompt = yield vscode.window.showInputBox({
            prompt: 'Describe what code you want to generate',
            placeHolder: 'e.g., "Create a Python function that sorts a list of dictionaries by a specific key"'
        });
        if (!prompt)
            return;
        try {
            const response = yield aiAssistant.sendMessage(`Please generate code based on this description: ${prompt}\n\nInclude explanations and best practices.`);
            const doc = yield vscode.workspace.openTextDocument({
                content: response,
                language: 'markdown'
            });
            vscode.window.showTextDocument(doc);
        }
        catch (error) {
            vscode.window.showErrorMessage(error.message);
        }
    }));
    const switchModelCommand = vscode.commands.registerCommand('aiAssistant.switchModel', () => __awaiter(this, void 0, void 0, function* () {
        const modelOptions = FREE_MODELS.map(model => {
            var _a;
            return ({
                label: ((_a = model.split('/')[1]) === null || _a === void 0 ? void 0 : _a.replace(':free', '')) || model,
                description: model,
                detail: 'Free model'
            });
        });
        const selected = yield vscode.window.showQuickPick(modelOptions, {
            placeHolder: 'Select AI model to use'
        });
        if (selected) {
            yield vscode.workspace.getConfiguration('aiAssistant').update('defaultModel', selected.description, vscode.ConfigurationTarget.Global);
            vscode.window.showInformationMessage(`AI model switched to: ${selected.label}`);
        }
    }));
    // Add all commands to subscriptions
    context.subscriptions.push(openChatCommand, explainCodeCommand, debugCodeCommand, refactorCodeCommand, generateCodeCommand, switchModelCommand);
    // Show welcome message
    vscode.window.showInformationMessage('AI Coding Assistant Pro activated! 🚀', 'Open Chat', 'View Commands').then(selection => {
        if (selection === 'Open Chat') {
            vscode.commands.executeCommand('aiAssistant.openChat');
        }
        else if (selection === 'View Commands') {
            vscode.commands.executeCommand('workbench.action.showCommands');
        }
    });
}
exports.activate = activate;
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map