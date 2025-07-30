import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Send, Code, Play, Bug, Mic, FileCode, Users, Sparkles, Terminal, Brain, 
  Zap, GitBranch, Eye, Cpu, Settings, Upload, Download, Save, FolderOpen,
  RefreshCw, AlertCircle, CheckCircle, Clock, Copy, Trash2, Plus,
  Monitor, Globe, Database, Shield, BarChart3, Layers, Palette, Search,
  MessageSquare, Edit3, Wand2, Target, FileText, ArrowRight, X, 
  ChevronDown, ChevronUp, Maximize2, Minimize2, Split, Command
} from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Enhanced modes with Cursor-like features
const modes = [
  {
    id: 'chat',
    label: '💬 Smart Chat',
    icon: Brain,
    color: 'bg-gradient-to-r from-purple-500 to-blue-500',
    description: 'Intelligent conversations with context awareness',
    features: ['Context Memory', 'Multi-turn Dialogue', 'Code Understanding', 'File References'],
    shortcut: 'Cmd+L'
  },
  {
    id: 'composer',
    label: '🎼 AI Composer',
    icon: Edit3,
    color: 'bg-gradient-to-r from-green-500 to-teal-500',
    description: 'Multi-file editing with AI assistance',
    features: ['Multi-file Editing', 'Project Understanding', 'Smart Refactoring', 'Code Generation'],
    shortcut: 'Cmd+I'
  },
  {
    id: 'execute',
    label: '▶️ Code Runner',
    icon: Play,
    color: 'bg-gradient-to-r from-orange-500 to-red-500',
    description: 'Execute code with real-time feedback',
    features: ['11 Languages', 'Performance Metrics', 'Input/Output', 'Error Detection'],
    shortcut: 'Ctrl+R'
  },
  {
    id: 'debug',
    label: '🐛 AI Debugger',
    icon: Bug,
    color: 'bg-gradient-to-r from-red-500 to-pink-500',
    description: 'Intelligent debugging and error fixing',
    features: ['Auto Error Detection', 'Fix Suggestions', 'Performance Analysis', 'Security Scan'],
    shortcut: 'Ctrl+D'
  },
  {
    id: 'refactor',
    label: '⚡ Smart Refactor',
    icon: Zap,
    color: 'bg-gradient-to-r from-yellow-500 to-orange-500',
    description: 'AI-powered code optimization',
    features: ['Performance Boost', 'Clean Code', 'Best Practices', 'Modern Patterns'],
    shortcut: 'Ctrl+K'
  },
  {
    id: 'analyze',
    label: '🔍 Code X-Ray',
    icon: Eye,
    color: 'bg-gradient-to-r from-cyan-500 to-blue-500',
    description: 'Deep code analysis and insights',
    features: ['Complexity Analysis', 'Security Audit', 'Performance Review', 'Documentation'],
    shortcut: 'Ctrl+Shift+A'
  },
  {
    id: 'generate',
    label: '🏗️ Project Builder',
    icon: FileCode,
    color: 'bg-gradient-to-r from-indigo-500 to-purple-500',
    description: 'Generate complete projects and components',
    features: ['50+ Templates', 'Framework Integration', 'Best Practices', 'Full Structure'],
    shortcut: 'Ctrl+G'
  },
  {
    id: 'terminal',
    label: '🖥️ AI Terminal',
    icon: Terminal,
    color: 'bg-gradient-to-r from-gray-500 to-slate-500',
    description: 'Natural language terminal commands',
    features: ['Plain English', 'Command Generation', 'Auto Execution', 'History'],
    shortcut: 'Ctrl+`'
  }
];

// Enhanced language configurations
const languageConfigs = {
  python: { icon: '🐍', extension: '.py', color: 'text-blue-400' },
  javascript: { icon: '🟨', extension: '.js', color: 'text-yellow-400' },
  typescript: { icon: '🔷', extension: '.ts', color: 'text-blue-500' },
  java: { icon: '☕', extension: '.java', color: 'text-orange-600' },
  cpp: { icon: '⚡', extension: '.cpp', color: 'text-blue-600' },
  c: { icon: '🔧', extension: '.c', color: 'text-gray-600' },
  go: { icon: '🐹', extension: '.go', color: 'text-cyan-400' },
  rust: { icon: '🦀', extension: '.rs', color: 'text-orange-500' },
  php: { icon: '🐘', extension: '.php', color: 'text-purple-500' },
  ruby: { icon: '💎', extension: '.rb', color: 'text-red-500' },
  bash: { icon: '📟', extension: '.sh', color: 'text-green-400' },
  html: { icon: '🌐', extension: '.html', color: 'text-orange-400' },
  css: { icon: '🎨', extension: '.css', color: 'text-blue-400' },
  sql: { icon: '🗃️', extension: '.sql', color: 'text-cyan-500' }
};

// Enhanced AI models with latest options
const availableModels = [
  { id: 'deepseek/deepseek-chat-v3-0324', name: 'DeepSeek V3', type: 'reasoning', speed: 'fast' },
  { id: 'deepseek/deepseek-reasoner', name: 'DeepSeek Reasoner', type: 'reasoning', speed: 'medium' },
  { id: 'openai/gpt-4o', name: 'GPT-4o', type: 'balanced', speed: 'fast' },
  { id: 'openai/gpt-4-turbo', name: 'GPT-4 Turbo', type: 'advanced', speed: 'medium' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', type: 'creative', speed: 'fast' },
  { id: 'google/gemini-2.0-flash-exp', name: 'Gemini 2.0 Flash', type: 'multimodal', speed: 'fast' },
  { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B', type: 'open-source', speed: 'medium' },
  { id: 'mistral/mistral-large', name: 'Mistral Large', type: 'efficient', speed: 'fast' },
  { id: 'qwen/qwen-2.5-coder-32b-instruct', name: 'Qwen 2.5 Coder', type: 'coding', speed: 'fast' }
];

// Enhanced code editor with Cursor-like features
const CodeEditor = ({ 
  value, 
  onChange, 
  language, 
  placeholder, 
  className = "",
  readOnly = false,
  showLineNumbers = true,
  enableAutoComplete = true,
  enableSmartRewrite = true
}) => {
  const textareaRef = useRef(null);
  const [cursorPosition, setCursorPosition] = useState(0);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleKeyDown = (e) => {
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const newValue = value.substring(0, start) + '  ' + value.substring(end);
      onChange(newValue);
      
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 2;
      }, 0);
    }
    
    // Cursor-like shortcuts
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'k':
          e.preventDefault();
          // Trigger AI inline edit
          break;
        case 'l':
          e.preventDefault();
          // Trigger chat mode
          break;
        case 'i':
          e.preventDefault();
          // Trigger composer mode
          break;
        default:
          break;
      }
    }
  };

  const handleChange = (e) => {
    const newValue = e.target.value;
    onChange(newValue);
    setCursorPosition(e.target.selectionStart);
    
    // Smart rewrite simulation
    if (enableSmartRewrite) {
      // Auto-fix common typos and patterns
      const fixes = {
        'functin': 'function',
        'retrun': 'return',
        'consle': 'console',
        'imoprt': 'import'
      };
      
      let fixedValue = newValue;
      Object.entries(fixes).forEach(([typo, correct]) => {
        fixedValue = fixedValue.replace(new RegExp(typo, 'g'), correct);
      });
      
      if (fixedValue !== newValue) {
        onChange(fixedValue);
      }
    }
  };

  return (
    <div className="relative font-mono">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        readOnly={readOnly}
        className={`w-full resize-none outline-none ${className}`}
        style={{ 
          tabSize: 2,
          whiteSpace: 'pre',
          overflow: 'auto',
          lineHeight: '1.5'
        }}
      />
      
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute top-full left-0 bg-white border border-gray-200 rounded-lg shadow-lg z-10 max-w-md">
          {suggestions.map((suggestion, index) => (
            <div key={index} className="p-2 hover:bg-gray-100 cursor-pointer text-sm">
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Enhanced message renderer with better display
const MessageRenderer = ({ message, onCopy, onApplyCode }) => {
  const isUser = message.type === 'user';
  const isSystem = message.type === 'system';
  const isError = message.type === 'error';
  const [isExpanded, setIsExpanded] = useState(true);
  
  // Extract code blocks from message
  const extractCodeBlocks = (content) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)\n```/g;
    const blocks = [];
    let match;
    
    while ((match = codeBlockRegex.exec(content)) !== null) {
      blocks.push({
        language: match[1] || 'text',
        code: match[2]
      });
    }
    
    return blocks;
  };

  const renderContent = (content) => {
    const codeBlocks = extractCodeBlocks(content);
    
    if (codeBlocks.length === 0) {
      return <div className="prose prose-sm max-w-none whitespace-pre-wrap">{content}</div>;
    }
    
    let lastIndex = 0;
    const parts = [];
    
    codeBlocks.forEach((block, index) => {
      const blockStart = content.indexOf('```', lastIndex);
      const blockEnd = content.indexOf('```', blockStart + 3) + 3;
      
      // Add text before code block
      if (blockStart > lastIndex) {
        parts.push(
          <div key={`text-${index}`} className="prose prose-sm max-w-none whitespace-pre-wrap mb-4">
            {content.substring(lastIndex, blockStart).trim()}
          </div>
        );
      }
      
      // Add code block
      parts.push(
        <div key={`code-${index}`} className="relative mb-4">
          <div className="flex items-center justify-between bg-gray-800 text-white px-4 py-2 rounded-t-lg text-sm">
            <span className="flex items-center">
              <Code className="w-4 h-4 mr-2" />
              {block.language}
            </span>
            <div className="flex space-x-2">
              <button
                onClick={() => onCopy(block.code)}
                className="p-1 hover:bg-gray-700 rounded"
                title="Copy code"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={() => onApplyCode(block.code, block.language)}
                className="p-1 hover:bg-gray-700 rounded"
                title="Apply to editor"
              >
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-b-lg overflow-x-auto">
            <pre className="text-sm leading-relaxed">{block.code}</pre>
          </div>
        </div>
      );
      
      lastIndex = blockEnd;
    });
    
    // Add remaining text
    if (lastIndex < content.length) {
      parts.push(
        <div key="text-final" className="prose prose-sm max-w-none whitespace-pre-wrap">
          {content.substring(lastIndex).trim()}
        </div>
      );
    }
    
    return parts;
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`max-w-5xl w-full rounded-xl ${
        isUser ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white' : 
        isError ? 'bg-red-50 text-red-800 border border-red-200' :
        isSystem ? 'bg-gray-50 text-gray-600 border border-gray-200' :
        'bg-white border border-gray-200 shadow-sm'
      }`}>
        <div className="p-6">
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center space-x-3">
              {!isUser && !isSystem && (
                <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <Brain className="w-4 h-4 text-white" />
                </div>
              )}
              <div>
                <p className="font-semibold text-sm">
                  {isUser ? 'You' : isSystem ? 'System' : 'AI Assistant Pro'}
                </p>
                <p className="text-xs opacity-70">{message.timestamp}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              {message.language && (
                <span className="text-xs px-2 py-1 bg-black/10 rounded-md">
                  {languageConfigs[message.language]?.icon} {message.language}
                </span>
              )}
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-1 hover:bg-black/10 rounded"
              >
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
          </div>
          
          {isExpanded && (
            <div className="space-y-4">
              {renderContent(message.content)}
              
              {message.data && (
                <div className="border-t pt-4 mt-4">
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {JSON.stringify(message.data, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Enhanced file reference component
const FileReference = ({ onFileSelect, files = [], isLoading = false }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  
  const filteredFiles = files.filter(file => 
    file.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="border rounded-lg p-4 bg-gray-50">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium text-gray-700">📁 Reference Files</h4>
        <Search className="w-4 h-4 text-gray-400" />
      </div>
      
      <input
        type="text"
        placeholder="Search files... (type @ to reference)"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="w-full p-2 border rounded-md text-sm mb-3"
      />
      
      <div className="max-h-32 overflow-y-auto space-y-1">
        {isLoading ? (
          <div className="text-center py-2 text-gray-500">Loading files...</div>
        ) : filteredFiles.length > 0 ? (
          filteredFiles.map((file, index) => (
            <button
              key={index}
              onClick={() => onFileSelect(file)}
              className="w-full text-left p-2 hover:bg-gray-100 rounded text-sm flex items-center space-x-2"
            >
              <FileText className="w-4 h-4 text-gray-400" />
              <span>{file.name}</span>
            </button>
          ))
        ) : (
          <div className="text-center py-2 text-gray-500">No files found</div>
        )}
      </div>
    </div>
  );
};

// Main App Component
const AICodingAssistantPro = () => {
  // State management
  const [activeMode, setActiveMode] = useState('chat');
  const [selectedModel, setSelectedModel] = useState('deepseek/deepseek-chat-v3-0324');
  const [selectedLanguage, setSelectedLanguage] = useState('python');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [apiHealth, setApiHealth] = useState('checking');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showFileReference, setShowFileReference] = useState(false);
  const [referencedFiles, setReferencedFiles] = useState([]);
  const [executionResult, setExecutionResult] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [systemStats, setSystemStats] = useState({
    codeExecutions: 0,
    analysisCount: 0,
    sessionsCount: 0,
    uptime: '0h 0m'
  });

  // Refs
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);

  // Get current mode details
  const currentMode = modes.find(mode => mode.id === activeMode);
  const supportedLanguages = Object.keys(languageConfigs);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/health`, { timeout: 5000 });
        setApiHealth(response.status === 200 ? 'online' : 'offline');
      } catch (error) {
        setApiHealth('offline');
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Initialize session
  useEffect(() => {
    createNewSession();
    
    // Update system stats
    const interval = setInterval(() => {
      setSystemStats(prev => ({
        ...prev,
        uptime: `${Math.floor(Date.now() / 3600000)}h ${Math.floor((Date.now() % 3600000) / 60000)}m`
      }));
    }, 60000);
    
    return () => clearInterval(interval);
  }, []);

  // Message management
  const addMessage = (content, type = 'user', metadata = {}) => {
    const message = {
      id: Date.now() + Math.random(),
      content,
      type,
      timestamp: new Date().toLocaleTimeString(),
      ...metadata
    };
    setMessages(prev => [...prev, message]);
    return message;
  };

  const createNewSession = () => {
    const sessionId = `session_${Date.now()}`;
    const newSession = {
      id: sessionId,
      name: `Session ${sessions.length + 1}`,
      created: new Date().toISOString(),
      messageCount: 0
    };
    setSessions(prev => [...prev, newSession]);
    setCurrentSession(newSession);
    setMessages([]);
    setSystemStats(prev => ({ ...prev, sessionsCount: prev.sessionsCount + 1 }));
  };

  // Enhanced message handling
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    try {
      // Add user message
      addMessage(userMessage, 'user', { language: selectedLanguage });

      // Route to appropriate handler based on mode
      switch (activeMode) {
        case 'chat':
          await handleChatMode(userMessage, currentSession?.id);
          break;
        case 'composer':
          await handleComposerMode(userMessage, currentSession?.id);
          break;
        case 'execute':
          await handleExecuteMode(userMessage, currentSession?.id);
          break;
        case 'debug':
          await handleDebugMode(userMessage, currentSession?.id);
          break;
        case 'refactor':
          await handleRefactorMode(userMessage, currentSession?.id);
          break;
        case 'analyze':
          await handleAnalyzeMode(userMessage, currentSession?.id);
          break;
        case 'generate':
          await handleGenerateMode(userMessage, currentSession?.id);
          break;
        case 'terminal':
          await handleTerminalMode(userMessage, currentSession?.id);
          break;
        default:
          await handleChatMode(userMessage, currentSession?.id);
      }
    } catch (error) {
      console.error('Error:', error);
      addMessage(
        `❌ Error: ${error.response?.data?.detail || error.message || 'Something went wrong'}`,
        'error'
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Mode handlers
  const handleChatMode = async (message, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/chat`, {
      message: message,
      model: selectedModel,
      session_id: sessionId,
      context: { 
        language: selectedLanguage, 
        mode: activeMode,
        referenced_files: referencedFiles 
      },
      temperature: 0.7
    });
    
    addMessage(response.data.response, 'assistant', {
      type: 'chat',
      model: response.data.model_used,
      session_id: response.data.session_id
    });
  };

  const handleComposerMode = async (prompt, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/composer`, {
      prompt: prompt,
      model: selectedModel,
      language: selectedLanguage,
      session_id: sessionId,
      files: referencedFiles
    });
    
    addMessage('🎼 Multi-file editing suggestions generated!', 'assistant', {
      type: 'composer',
      suggestions: response.data.suggestions,
      files_modified: response.data.files_modified
    });
  };

  const handleExecuteMode = async (code, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/execute`, {
      language: selectedLanguage,
      code: code,
      session_id: sessionId,
      timeout: 15
    });
    
    setExecutionResult(response.data);
    addMessage('✅ Code executed successfully!', 'assistant', {
      type: 'execution',
      result: response.data,
      language: selectedLanguage
    });
    
    setSystemStats(prev => ({ ...prev, codeExecutions: prev.codeExecutions + 1 }));
  };

  const handleDebugMode = async (code, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/debug`, {
      code: code,
      language: selectedLanguage,
      session_id: sessionId
    });
    
    addMessage('🐛 Debug analysis complete!', 'assistant', {
      type: 'debug',
      analysis: response.data.analysis,
      fixes: response.data.suggested_fixes,
      language: selectedLanguage
    });
  };

  const handleRefactorMode = async (code, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/refactor`, {
      code: code,
      language: selectedLanguage,
      refactor_type: ['performance', 'readability', 'best_practices'],
      session_id: sessionId
    });
    
    addMessage('⚡ Refactoring suggestions ready!', 'assistant', {
      type: 'refactor',
      original_code: code,
      refactored_code: response.data.refactored_code,
      improvements: response.data.improvements
    });
  };

  const handleAnalyzeMode = async (code, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
      code: code,
      language: selectedLanguage,
      analysis_type: ['syntax', 'complexity', 'security', 'performance'],
      session_id: sessionId
    });
    
    setAnalysisResults(response.data);
    addMessage('🔍 Code analysis complete!', 'assistant', {
      type: 'analysis',
      data: response.data,
      language: selectedLanguage
    });
    
    setSystemStats(prev => ({ ...prev, analysisCount: prev.analysisCount + 1 }));
  };

  const handleGenerateMode = async (description, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/generate-project`, {
      description: description,
      language: selectedLanguage,
      framework: 'auto-detect',
      features: ['modern', 'scalable', 'secure']
    });
    
    addMessage('🏗️ Project structure generated!', 'assistant', {
      type: 'project',
      data: response.data,
      language: selectedLanguage
    });
  };

  const handleTerminalMode = async (command, sessionId) => {
    const response = await axios.post(`${API_BASE_URL}/api/terminal`, {
      natural_command: command,
      session_id: sessionId
    });
    
    addMessage('🖥️ Terminal command generated and executed!', 'assistant', {
      type: 'terminal',
      command: response.data.command,
      output: response.data.output,
      explanation: response.data.explanation
    });
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
    setInput(code);
    setSelectedLanguage(language || selectedLanguage);
    addMessage(`📝 Code applied to editor (${language || selectedLanguage})`, 'system');
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        setInput(content);
        addMessage(`📁 File uploaded: ${file.name}`, 'system');
        
        // Add to referenced files
        setReferencedFiles(prev => [...prev, {
          name: file.name,
          content: content,
          type: file.type
        }]);
      };
      reader.readAsText(file);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'Enter':
            e.preventDefault();
            handleSendMessage();
            break;
          case 'l':
            e.preventDefault();
            setActiveMode('chat');
            break;
          case 'i':
            e.preventDefault();
            setActiveMode('composer');
            break;
          case 'k':
            e.preventDefault();
            setActiveMode('refactor');
            break;
          case '`':
            e.preventDefault();
            setActiveMode('terminal');
            break;
          default:
            break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className={`${isFullscreen ? 'fixed inset-0 z-50' : 'h-screen'} bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex`}>
      {/* Enhanced Sidebar */}
      <div className={`${isFullscreen ? 'w-80' : 'w-96'} bg-black/20 backdrop-blur-xl border-r border-white/10 flex flex-col`}>
        {/* Header */}
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl flex items-center justify-center">
                <Cpu className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">AI Assistant Pro</h1>
                <p className="text-sm text-gray-400">Cursor-Enhanced Edition</p>
              </div>
            </div>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              {isFullscreen ? <Minimize2 className="w-5 h-5 text-white" /> : <Maximize2 className="w-5 h-5 text-white" />}
            </button>
          </div>
          
          {/* Mode Selection */}
          <div className="space-y-2">
            {modes.map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all group ${
                    activeMode === mode.id 
                      ? `${mode.color} text-white shadow-lg transform scale-105` 
                      : 'text-gray-300 hover:bg-white/5 hover:transform hover:scale-102'
                  }`}
                  title={mode.description}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className="w-5 h-5" />
                    <div className="text-left">
                      <div className="font-medium text-sm">{mode.label}</div>
                      <div className="text-xs opacity-70 group-hover:opacity-100">
                        {mode.features?.[0]}
                      </div>
                    </div>
                  </div>
                  <div className="text-xs opacity-60">
                    {mode.shortcut}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
        
        {/* Configuration Panel */}
        <div className="p-6 border-b border-white/10 space-y-4">
          {/* Model Selection */}
          <div>
            <h3 className="text-white font-medium mb-3 flex items-center">
              <Brain className="w-4 h-4 mr-2" />
              AI Model
            </h3>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {availableModels.map(model => (
                <option key={model.id} value={model.id} className="bg-gray-800">
                  {model.name} ({model.type})
                </option>
              ))}
            </select>
          </div>
          
          {/* Language Selection */}
          <div>
            <h3 className="text-white font-medium mb-3 flex items-center">
              <Code className="w-4 h-4 mr-2" />
              Language
            </h3>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {supportedLanguages.map(lang => (
                <option key={lang} value={lang} className="bg-gray-800">
                  {languageConfigs[lang]?.icon} {lang.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>
        
        {/* File Reference Panel */}
        {showFileReference && (
          <div className="p-6 border-b border-white/10">
            <FileReference
              onFileSelect={(file) => {
                setReferencedFiles(prev => [...prev, file]);
                setShowFileReference(false);
              }}
              files={[]} // Would be populated from backend
            />
          </div>
        )}
        
        {/* System Status */}
        <div className="flex-1 p-6 space-y-4">
          <div className="bg-white/5 rounded-xl p-4 border border-white/10">
            <h3 className="text-white font-medium mb-3 flex items-center">
              <Monitor className="w-4 h-4 mr-2" />
              System Status
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between text-gray-300">
                <span>API Status</span>
                <div className={`flex items-center space-x-2`}>
                  <div className={`w-2 h-2 rounded-full ${
                    apiHealth === 'online' ? 'bg-green-500' : 
                    apiHealth === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                  }`}></div>
                  <span className="text-xs">{apiHealth}</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between text-gray-300">
                <span>Sessions</span>
                <span className="text-xs">{sessions.length}</span>
              </div>
              
              <div className="flex items-center justify-between text-gray-300">
                <span>Executions</span>
                <span className="text-xs">{systemStats.codeExecutions}</span>
              </div>
              
              <div className="flex items-center justify-between text-gray-300">
                <span>Analyses</span>
                <span className="text-xs">{systemStats.analysisCount}</span>
              </div>
              
              <div className="flex items-center justify-between text-gray-300">
                <span>Uptime</span>
                <span className="text-xs">{systemStats.uptime}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Enhanced Header */}
        <div className="bg-black/10 backdrop-blur-xl border-b border-white/10 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center">
                {currentMode && <currentMode.icon className="w-6 h-6 mr-3" />}
                {currentMode?.label}
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                {currentMode?.description}
              </p>
              {currentMode?.features && (
                <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                  {currentMode.features.slice(0, 3).map((feature, index) => (
                    <span key={index} className="flex items-center">
                      <CheckCircle className="w-3 h-3 mr-1" />
                      {feature}
                    </span>
                  ))}
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-3">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
                accept=".txt,.js,.py,.java,.cpp,.c,.go,.rs,.php,.rb,.sh,.ts,.jsx,.tsx,.html,.css,.sql,.json,.yaml,.yml,.xml,.md"
              />
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 bg-white/10 text-gray-300 hover:bg-white/20 rounded-lg transition-all"
                title="Upload file (Drag & Drop supported)"
              >
                <Upload className="w-5 h-5" />
              </button>
              
              <button
                onClick={() => setShowFileReference(!showFileReference)}
                className={`p-2 rounded-lg transition-all ${
                  showFileReference ? 'bg-blue-500 text-white' : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
                title="Reference files (@)"
              >
                <FileText className="w-5 h-5" />
              </button>
              
              <button 
                className="p-2 bg-white/10 text-gray-300 hover:bg-white/20 rounded-lg transition-all"
                title="Download session"
                onClick={() => {
                  const data = JSON.stringify({ messages, session: currentSession }, null, 2);
                  const blob = new Blob([data], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `session_${currentSession?.id || 'current'}.json`;
                  a.click();
                }}
              >
                <Download className="w-5 h-5" />
              </button>
              
              <button
                onClick={createNewSession}
                className="p-2 bg-white/10 text-gray-300 hover:bg-white/20 rounded-lg transition-all"
                title="New session (Ctrl+N)"
              >
                <Plus className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Enhanced Messages Area with Fixed Height */}
        <div className="flex-1 overflow-y-auto p-6" style={{ height: 'calc(100vh - 200px)' }}>
          {messages.length === 0 && (
            <div className="text-center py-16">
              <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <currentMode?.icon className="w-10 h-10 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-4">
                Ready to revolutionize your coding! 🚀
              </h3>
              <p className="text-gray-400 max-w-md mx-auto mb-6">
                {currentMode?.description || 'Start a conversation and let AI assist you with your coding tasks.'}
              </p>
              {currentMode?.features && (
                <div className="flex flex-wrap justify-center gap-2 max-w-lg mx-auto mb-4">
                  {currentMode.features.map((feature, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 bg-white/10 text-gray-300 rounded-full text-sm"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              )}
              <div className="text-xs text-gray-500 mt-4">
                Keyboard shortcuts: {currentMode?.shortcut} • Ctrl+Enter to send • @ to reference files
              </div>
            </div>
          )}
          
          {messages.map(message => (
            <MessageRenderer 
              key={message.id}
              message={message}
              onCopy={handleCopyCode}
              onApplyCode={handleApplyCode}
            />
          ))}
          
          {isLoading && (
            <div className="flex justify-start mb-6">
              <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-lg max-w-md">
                <div className="flex items-center space-x-3">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                  <span className="text-sm text-gray-600">AI is processing your request...</span>
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  Model: {selectedModel.split('/')[1]} • Mode: {currentMode?.label} • Language: {selectedLanguage}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Enhanced Input Area */}
        <div className="bg-black/10 backdrop-blur-xl border-t border-white/10 p-6">
          <div className="flex items-end space-x-4">
            <div className="flex-1">
              <CodeEditor
                ref={inputRef}
                value={input}
                onChange={setInput}
                language={selectedLanguage}
                placeholder={
                  activeMode === 'chat' ? 'Ask me anything about coding... (@ to reference files, web, docs)' :
                  activeMode === 'composer' ? 'Describe the multi-file changes you want to make...' :
                  activeMode === 'execute' ? `Paste your ${selectedLanguage} code to run it...` :
                  activeMode === 'debug' ? 'Share your buggy code for analysis and fixes...' :
                  activeMode === 'refactor' ? 'Show me code that needs improvement and optimization...' :
                  activeMode === 'analyze' ? 'Submit code for deep analysis and insights...' :
                  activeMode === 'generate' ? 'Describe the project or component you want to generate...' :
                  activeMode === 'terminal' ? 'Describe what you want to do in natural language...' :
                  'Enter your message...'
                }
                className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none min-h-[120px] max-h-[300px]"
                enableAutoComplete={true}
                enableSmartRewrite={true}
                showLineNumbers={false}
              />
              
              {referencedFiles.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {referencedFiles.map((file, index) => (
                    <span key={index} className="inline-flex items-center px-2 py-1 bg-blue-500/20 text-blue-200 rounded-md text-xs">
                      <FileText className="w-3 h-3 mr-1" />
                      {file.name}
                      <button
                        onClick={() => setReferencedFiles(prev => prev.filter((_, i) => i !== index))}
                        className="ml-1 hover:bg-blue-500/30 rounded"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
            
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim() || apiHealth === 'offline'}
              className="bg-gradient-to-r from-purple-500 to-blue-500 text-white p-4 rounded-xl hover:from-purple-600 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 min-w-[56px]"
              title={apiHealth === 'offline' ? 'Backend server is offline' : `Send message (${currentMode?.shortcut})`}
            >
              {isLoading ? (
                <RefreshCw className="w-6 h-6 animate-spin" />
              ) : (
                <Send className="w-6 h-6" />
              )}
            </button>
          </div>
          
          <div className="flex items-center justify-between mt-4 text-xs text-gray-400">
            <div className="flex items-center space-x-4">
              <span className="flex items-center">
                <Brain className="w-3 h-3 mr-1" />
                {selectedModel.split('/')[1] || selectedModel}
              </span>
              <span>•</span>
              <span className="flex items-center">
                <Code className="w-3 h-3 mr-1" />
                {selectedLanguage} {languageConfigs[selectedLanguage]?.icon}
              </span>
              <span>•</span>
              <span className={`flex items-center ${apiHealth === 'online' ? 'text-green-400' : 'text-red-400'}`}>
                <Monitor className="w-3 h-3 mr-1" />
                {apiHealth === 'online' ? 'Online' : apiHealth === 'offline' ? 'Offline' : 'Checking...'}
              </span>
              {referencedFiles.length > 0 && (
                <>
                  <span>•</span>
                  <span className="flex items-center text-blue-400">
                    <FileText className="w-3 h-3 mr-1" />
                    {referencedFiles.length} file{referencedFiles.length !== 1 ? 's' : ''}
                  </span>
                </>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <span>Ctrl+Enter to send • @ to reference • {currentMode?.shortcut} for mode</span>
              <span className="flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                {currentSession ? 'Session active' : 'No session'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AICodingAssistantPro;