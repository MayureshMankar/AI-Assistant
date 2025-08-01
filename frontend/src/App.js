import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Send, Code, Play, Bug, Terminal, Brain, Zap, Eye, FileCode, Settings, 
  Upload, Download, Save, FolderOpen, RefreshCw, AlertCircle, CheckCircle, 
  Copy, Trash2, Plus, Monitor, Search, MessageSquare, Edit3, Wand2, 
  FileText, ArrowRight, X, ChevronDown, ChevronUp, Maximize2, Minimize2,
  Sidebar, Menu, Activity, GitBranch, Package, Database, Globe
} from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Free working models from OpenRouter (verified)
const FREE_MODELS = [
  {
    id: 'google/gemini-2.0-flash-exp:free',
    name: 'Gemini 2.0 Flash Experimental',
    type: 'multimodal',
    context: '1,048,576',
    description: 'Google: Gemini 2.0 Flash Experimental (free)'
  },
  {
    id: 'qwen/qwen3-coder:free',
    name: 'Qwen3 Coder',
    type: 'programming',
    context: '262,144',
    description: 'Qwen: Qwen3 Coder (free)'
  },
  {
    id: 'tngtech/deepseek-r1t2-chimera:free',
    name: 'DeepSeek R1T2 Chimera',
    type: 'reasoning',
    context: '163,840',
    description: 'TNG: DeepSeek R1T2 Chimera (free)'
  },
  {
    id: 'deepseek/deepseek-r1-0528:free',
    name: 'DeepSeek R1 0528',
    type: 'reasoning',
    context: '163,840',
    description: 'DeepSeek: R1 0528 (free)'
  },
  {
    id: 'tngtech/deepseek-r1t-chimera:free',
    name: 'DeepSeek R1T Chimera',
    type: 'reasoning',
    context: '163,840',
    description: 'TNG: DeepSeek R1T Chimera (free)'
  },
  {
    id: 'microsoft/mai-ds-r1:free',
    name: 'MAI DS R1',
    type: 'reasoning',
    context: '163,840',
    description: 'Microsoft: MAI DS R1 (free)'
  },
  {
    id: 'deepseek/deepseek-r1:free',
    name: 'DeepSeek R1',
    type: 'reasoning',
    context: '163,840',
    description: 'DeepSeek: R1 (free)'
  },
  {
    id: 'z-ai/glm-4.5-air:free',
    name: 'GLM 4.5 Air',
    type: 'reasoning',
    context: '131,072',
    description: 'Z.AI: GLM 4.5 Air (free)'
  },
  {
    id: 'moonshotai/kimi-dev-72b:free',
    name: 'Kimi Dev 72b',
    type: 'reasoning',
    context: '131,072',
    description: 'Kimi Dev 72b (free)'
  },
  {
    id: 'deepseek/deepseek-r1-0528-qwen3-8b:free',
    name: 'Deepseek R1 0528 Qwen3 8B',
    type: 'reasoning',
    context: '131,072',
    description: 'Deepseek R1 0528 Qwen3 8B (free)'
  },
  {
    id: 'qwen/qwen3-235b-a22b:free',
    name: 'Qwen3 235B A22B',
    type: 'reasoning',
    context: '131,072',
    description: 'Qwen: Qwen3 235B A22B (free)'
  },
  {
    id: 'moonshotai/kimi-vl-a3b-thinking:free',
    name: 'Kimi VL A3B Thinking',
    type: 'reasoning',
    context: '131,072',
    description: 'Moonshot AI: Kimi VL A3B Thinking (free)'
  },
  {
    id: 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free',
    name: 'Llama 3.1 Nemotron Ultra 253B v1',
    type: 'reasoning',
    context: '131,072',
    description: 'NVIDIA: Llama 3.1 Nemotron Ultra 253B v1 (free)'
  }
];

// VS Code-like activity bar items
const ACTIVITY_BAR_ITEMS = [
  { id: 'explorer', icon: FolderOpen, label: 'Explorer', shortcut: 'Ctrl+Shift+E' },
  { id: 'search', icon: Search, label: 'Search', shortcut: 'Ctrl+Shift+F' },
  { id: 'git', icon: GitBranch, label: 'Source Control', shortcut: 'Ctrl+Shift+G' },
  { id: 'debug', icon: Bug, label: 'Run and Debug', shortcut: 'Ctrl+Shift+D' },
  { id: 'extensions', icon: Package, label: 'Extensions', shortcut: 'Ctrl+Shift+X' },
  { id: 'ai-chat', icon: Brain, label: 'AI Assistant', shortcut: 'Ctrl+Shift+A' },
  { id: 'terminal', icon: Terminal, label: 'Terminal', shortcut: 'Ctrl+`' }
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
  sql: [/SELECT\s+/, /FROM\s+/, /WHERE\s+/, /INSERT\s+INTO/, /CREATE\s+TABLE/i],
  json: [/{\s*"/, /\[\s*{/, /":\s*"/],
  yaml: [/---/, /:\s*$/, /^\s*-\s+/],
  markdown: [/^#\s+/, /^\*\s+/, /\[.*\]\(.*\)/, /```/]
};

// Auto-detect language from code
const detectLanguage = (code) => {
  if (!code || code.trim().length < 10) return 'text';
  
  const lines = code.split('\n').slice(0, 10); // Check first 10 lines
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
  const detectedLang = Object.keys(scores).find(lang => scores[lang] === maxScore);
  
  return maxScore > 0 ? detectedLang : 'text';
};

// Enhanced code editor component
const CodeEditor = ({ value, onChange, className = "", readOnly = false, placeholder }) => {
  const textareaRef = useRef(null);
  const [detectedLanguage, setDetectedLanguage] = useState('text');

  useEffect(() => {
    if (value) {
      const lang = detectLanguage(value);
      setDetectedLanguage(lang);
    }
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const newValue = value.substring(0, start) + '  ' + value.substring(end);
      onChange(newValue);
      
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 2;
      }, 0);
    }
  };

  return (
    <div className="relative h-full font-mono">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        readOnly={readOnly}
        className={`w-full h-full resize-none outline-none bg-gray-900 text-gray-100 p-4 leading-6 ${className}`}
        style={{ 
          tabSize: 2,
          fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
          fontSize: '14px'
        }}
      />
      {detectedLanguage !== 'text' && (
        <div className="absolute top-2 right-2 bg-blue-600 text-white text-xs px-2 py-1 rounded">
          {detectedLanguage}
        </div>
      )}
    </div>
  );
};

// Message component with better rendering
const Message = ({ message, onCopy, onApplyCode, onRegenerate }) => {
  const isUser = message.type === 'user';
  const isSystem = message.type === 'system';
  const [isExpanded, setIsExpanded] = useState(true);

  const extractCodeBlocks = (content) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)\n```/g;
    const blocks = [];
    let match;
    
    while ((match = codeBlockRegex.exec(content)) !== null) {
      blocks.push({
        language: match[1] || detectLanguage(match[2]),
        code: match[2]
      });
    }
    
    return blocks;
  };

  const renderContent = () => {
    const codeBlocks = extractCodeBlocks(message.content);
    
    if (codeBlocks.length === 0) {
      return (
        <div className="prose prose-gray prose-sm max-w-none text-gray-300 whitespace-pre-wrap">
          {message.content}
        </div>
      );
    }
    
    let lastIndex = 0;
    const parts = [];
    
    codeBlocks.forEach((block, index) => {
      const blockStart = message.content.indexOf('```', lastIndex);
      const blockEnd = message.content.indexOf('```', blockStart + 3) + 3;
      
      // Add text before code block
      if (blockStart > lastIndex) {
        const textContent = message.content.substring(lastIndex, blockStart).trim();
        if (textContent) {
          parts.push(
            <div key={`text-${index}`} className="prose prose-gray prose-sm max-w-none text-gray-300 whitespace-pre-wrap mb-4">
              {textContent}
            </div>
          );
        }
      }
      
      // Add code block
      parts.push(
        <div key={`code-${index}`} className="mb-4">
          <div className="flex items-center justify-between bg-gray-800 text-gray-300 px-4 py-2 text-sm border-b border-gray-700">
            <span className="flex items-center">
              <Code className="w-4 h-4 mr-2" />
              {block.language}
            </span>
            <div className="flex space-x-2">
              <button
                onClick={() => onCopy(block.code)}
                className="p-1 hover:bg-gray-700 rounded transition-colors"
                title="Copy code"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={() => onApplyCode(block.code, block.language)}
                className="p-1 hover:bg-gray-700 rounded transition-colors"
                title="Apply to editor"
              >
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="bg-gray-900 p-4 overflow-x-auto">
            <pre className="text-sm text-gray-100 leading-6">{block.code}</pre>
          </div>
        </div>
      );
      
      lastIndex = blockEnd;
    });
    
    // Add remaining text
    if (lastIndex < message.content.length) {
      const remainingText = message.content.substring(lastIndex).trim();
      if (remainingText) {
        parts.push(
          <div key="text-final" className="prose prose-gray prose-sm max-w-none text-gray-300 whitespace-pre-wrap">
            {remainingText}
          </div>
        );
      }
    }
    
    return parts;
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`max-w-4xl w-full ${
        isUser ? 'bg-blue-600 text-white' : 
        isSystem ? 'bg-gray-700 text-gray-300' :
        'bg-gray-800 text-gray-100 border border-gray-700'
      } rounded-lg`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              {!isUser && !isSystem && (
                <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                  <Brain className="w-3 h-3 text-white" />
                </div>
              )}
              <span className="text-sm font-medium">
                {isUser ? 'You' : isSystem ? 'System' : 'AI Assistant'}
              </span>
              <span className="text-xs opacity-70">{message.timestamp}</span>
              {message.model && (
                <span className="text-xs bg-gray-600 px-2 py-1 rounded">
                  {FREE_MODELS.find(m => m.id === message.model)?.name || message.model}
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              {!isUser && !isSystem && (
                <button
                  onClick={() => onRegenerate(message)}
                  className="p-1 hover:bg-gray-600 rounded transition-colors"
                  title="Regenerate response"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-1 hover:bg-gray-600 rounded transition-colors"
              >
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
          </div>
          
          {isExpanded && renderContent()}
        </div>
      </div>
    </div>
  );
};

// File explorer component
const FileExplorer = ({ files, onFileSelect, onFileUpload }) => {
  const fileInputRef = useRef(null);

  return (
    <div className="h-full bg-gray-800 text-gray-300">
      <div className="p-3 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">EXPLORER</span>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-1 hover:bg-gray-700 rounded"
            title="Upload file"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
      </div>
      
      <input
        type="file"
        ref={fileInputRef}
        onChange={onFileUpload}
        className="hidden"
        multiple
        accept=".txt,.js,.py,.java,.cpp,.c,.go,.rs,.php,.rb,.sh,.ts,.jsx,.tsx,.html,.css,.sql,.json,.yaml,.yml,.xml,.md"
      />
      
      <div className="p-2">
        {files.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <FolderOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No files opened</p>
            <p className="text-xs mt-1">Upload files to get started</p>
          </div>
        ) : (
          files.map((file, index) => (
            <div
              key={index}
              onClick={() => onFileSelect(file)}
              className="flex items-center space-x-2 p-2 hover:bg-gray-700 rounded cursor-pointer"
            >
              <FileText className="w-4 h-4" />
              <span className="text-sm truncate">{file.name}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

// Main VS Code-like IDE component
const VSCodeIDE = () => {
  // Core state
  const [activeActivity, setActiveActivity] = useState('ai-chat');
  const [selectedModel, setSelectedModel] = useState('deepseek/deepseek-chat-v3-0324:free');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [editorContent, setEditorContent] = useState('');
  const [conversationContext, setConversationContext] = useState([]);
  // Session ID state, always a valid UUID
  const [sessionId, setSessionId] = useState(() => {
    let id = localStorage.getItem('sessionId');
    // Validate UUID format
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    if (!id || !uuidRegex.test(id)) {
      id = generateUUID();
      localStorage.setItem('sessionId', id);
    }
    return id;
  });

  // Refs
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize with welcome message
  useEffect(() => {
    addMessage(
      `🚀 **AI Coding Assistant - VS Code Edition**

Welcome to your AI-powered coding environment! I'm here to help you with:

• **Code Analysis & Debugging** - Paste your code and I'll help fix issues
• **Code Generation** - Describe what you need and I'll write it
• **Code Explanation** - I'll explain complex code in simple terms
• **Best Practices** - Get suggestions for improving your code
• **Multi-language Support** - I auto-detect programming languages

**Quick Tips:**
- Upload files using the Explorer panel
- I maintain conversation context for better assistance
- Code blocks are automatically highlighted and can be copied
- Use natural language to describe what you need

What can I help you build today?`,
      'assistant',
      { model: selectedModel }
    );
  }, []);

  // Add message with context management
  const addMessage = useCallback((content, type = 'user', metadata = {}) => {
    const message = {
      id: Date.now() + Math.random(),
      content,
      type,
      timestamp: new Date().toLocaleTimeString(),
      ...metadata
    };
    
    setMessages(prev => [...prev, message]);
    
    // Update conversation context for AI
    if (type === 'user' || type === 'assistant') {
      setConversationContext(prev => [
        ...prev,
        {
          role: type === 'user' ? 'user' : 'assistant',
          content: content
        }
      ].slice(-20)); // Keep last 20 messages for context
    }
    
    return message;
  }, []);

  // Enhanced message sending with context
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    try {
      // Add user message
      addMessage(userMessage, 'user');

      // Prepare context for AI
      const contextMessages = [
        {
          role: "system",
          content: `You are an expert AI coding assistant integrated into a VS Code-like IDE. You help with coding, debugging, code analysis, and programming questions across all languages.

Current context:
- Session ID: ${sessionId}
- Active files: ${files.map(f => f.name).join(', ') || 'None'}
- Auto-detected language capabilities enabled
- Free model: ${FREE_MODELS.find(m => m.id === selectedModel)?.name}

Guidelines:
- Provide detailed, practical coding help
- Use markdown code blocks with language specification
- Explain your reasoning and approach
- Suggest best practices and improvements
- Be concise but thorough
- Auto-detect programming languages from code snippets
- Maintain conversation context for follow-up questions`
        },
        ...conversationContext.slice(-10), // Last 10 messages for context
        {
          role: "user",
          content: userMessage
        }
      ];

      // Call AI API with enhanced context
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: userMessage,
        model: selectedModel,
        session_id: sessionId,
        context: {
          conversation_history: contextMessages,
          files: files.map(f => ({ name: f.name, language: detectLanguage(f.content) })),
          active_file: activeFile?.name || null,
          editor_content: editorContent ? editorContent.substring(0, 1000) : null // Limit context size
        },
        temperature: 0.7,
        max_tokens: 4000
      });

      // Add AI response
      addMessage(response.data.response, 'assistant', {
        model: selectedModel,
        session_id: sessionId
      });

    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Something went wrong';
      addMessage(`❌ Error: ${errorMessage}`, 'system');
    } finally {
      setIsLoading(false);
    }
  };

  // File handling
  const handleFileUpload = (event) => {
    const uploadedFiles = Array.from(event.target.files);
    
    uploadedFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        const newFile = {
          name: file.name,
          content: content,
          type: file.type,
          size: file.size,
          lastModified: file.lastModified,
          language: detectLanguage(content)
        };
        
        setFiles(prev => [...prev, newFile]);
        addMessage(`📁 Uploaded file: ${file.name} (${newFile.language})`, 'system');
      };
      reader.readAsText(file);
    });
  };

  const handleFileSelect = (file) => {
    setActiveFile(file);
    setEditorContent(file.content);
    addMessage(`📂 Opened file: ${file.name}`, 'system');
  };

  // Utility functions
  const handleCopyCode = async (code) => {
    try {
      await navigator.clipboard.writeText(code);
      addMessage('📋 Code copied to clipboard!', 'system');
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleApplyCode = (code, language) => {
    setEditorContent(code);
    addMessage(`📝 Code applied to editor (${language})`, 'system');
  };

  const handleRegenerate = async (message) => {
    if (isLoading) return;
    
    // Find the user message before this assistant message
    const messageIndex = messages.findIndex(m => m.id === message.id);
    const userMessage = messages[messageIndex - 1];
    
    if (userMessage && userMessage.type === 'user') {
      setIsLoading(true);
      try {
        const response = await axios.post(`${API_BASE_URL}/api/chat`, {
          message: userMessage.content,
          model: selectedModel,
          session_id: sessionId,
          context: {
            conversation_history: conversationContext.slice(-10),
            regenerate: true
          },
          temperature: 0.8, // Slightly higher for variety
          max_tokens: 4000
        });

        // Replace the message
        setMessages(prev => prev.map(m => 
          m.id === message.id 
            ? { ...m, content: response.data.response, timestamp: new Date().toLocaleTimeString() }
            : m
        ));

      } catch (error) {
        console.error('Error regenerating:', error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'Enter':
            if (activeActivity === 'ai-chat') {
              e.preventDefault();
              handleSendMessage();
            }
            break;
          case '`':
            e.preventDefault();
            setActiveActivity('terminal');
            break;
          default:
            break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [activeActivity]);

  // Render sidebar panel content
  const renderSidebarContent = () => {
    switch (activeActivity) {
      case 'explorer':
        return (
          <FileExplorer
            files={files}
            onFileSelect={handleFileSelect}
            onFileUpload={handleFileUpload}
          />
        );
      
      case 'ai-chat':
        return (
          <div className="h-full bg-gray-800 text-gray-300 flex flex-col">
            <div className="p-3 border-b border-gray-700">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium">AI ASSISTANT</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="text-xs bg-gray-700 border border-gray-600 rounded px-2 py-1"
                >
                  {FREE_MODELS.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.type})
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="text-xs text-gray-500">
                Model: {FREE_MODELS.find(m => m.id === selectedModel)?.description}
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-3">
              {messages.map(message => (
                <Message 
                  key={message.id}
                  message={message}
                  onCopy={handleCopyCode}
                  onApplyCode={handleApplyCode}
                  onRegenerate={handleRegenerate}
                />
              ))}
              
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-700 rounded-lg p-3">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            
            <div className="p-3 border-t border-gray-700">
              <div className="flex space-x-2">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Ask me anything about coding... (Ctrl+Enter to send)"
                  className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm resize-none"
                  rows="3"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !input.trim()}
                  className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white p-2 rounded transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
              
              <div className="text-xs text-gray-500 mt-2">
                Free model • Context maintained • Auto language detection
              </div>
            </div>
          </div>
        );
      
      default:
        return (
          <div className="h-full bg-gray-800 text-gray-300 flex items-center justify-center">
            <div className="text-center">
              <div className="text-gray-500 mb-2">
                {ACTIVITY_BAR_ITEMS.find(item => item.id === activeActivity)?.icon && 
                  React.createElement(ACTIVITY_BAR_ITEMS.find(item => item.id === activeActivity).icon, { className: "w-12 h-12 mx-auto mb-2" })
                }
              </div>
              <p className="text-sm">
                {ACTIVITY_BAR_ITEMS.find(item => item.id === activeActivity)?.label || 'Panel'}
              </p>
              <p className="text-xs text-gray-500 mt-1">Coming soon...</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="h-screen bg-gray-900 text-gray-100 flex">
      {/* Activity Bar */}
      <div className="w-12 bg-gray-800 border-r border-gray-700 flex flex-col">
        {ACTIVITY_BAR_ITEMS.map(item => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setActiveActivity(item.id)}
              className={`w-12 h-12 flex items-center justify-center hover:bg-gray-700 transition-colors ${
                activeActivity === item.id ? 'bg-gray-700 border-r-2 border-blue-500' : ''
              }`}
              title={`${item.label} (${item.shortcut})`}
            >
              <Icon className="w-5 h-5" />
            </button>
          );
        })}
      </div>

      {/* Sidebar */}
      <div className="w-80 border-r border-gray-700">
        {renderSidebarContent()}
      </div>

      {/* Main Editor Area */}
      <div className="flex-1 flex flex-col">
        {/* Tab Bar */}
        {activeFile && (
          <div className="bg-gray-800 border-b border-gray-700 px-4 py-2">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4" />
              <span className="text-sm">{activeFile.name}</span>
              <span className="text-xs text-gray-500">({activeFile.language})</span>
            </div>
          </div>
        )}

        {/* Editor */}
        <div className="flex-1">
          {activeFile ? (
            <CodeEditor
              value={editorContent}
              onChange={setEditorContent}
              className="h-full"
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-gray-900">
              <div className="text-center">
                <Brain className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                <h2 className="text-xl font-semibold mb-2">AI Coding Assistant</h2>
                <p className="text-gray-400 mb-4">VS Code-like IDE with AI integration</p>
                <div className="space-y-2 text-sm text-gray-500">
                  <p>• Upload files using the Explorer panel</p>
                  <p>• Chat with AI using the Assistant panel</p>
                  <p>• Auto language detection enabled</p>
                  <p>• Free models with conversation context</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Status Bar */}
        <div className="bg-blue-600 text-white px-4 py-1 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <span>AI Assistant Ready</span>
            <span>Model: {FREE_MODELS.find(m => m.id === selectedModel)?.name}</span>
            {activeFile && <span>Language: {activeFile.language}</span>}
          </div>
          <div className="flex items-center space-x-4">
            <span>Session: {sessionId.slice(-8)}</span>
            <span>Context: {conversationContext.length} messages</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// UUID v4 generator
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

export default VSCodeIDE;