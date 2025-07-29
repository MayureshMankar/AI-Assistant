import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Send, Code, Play, Bug, Mic, FileCode, Users, Sparkles, Terminal, Brain, 
  Zap, GitBranch, Eye, Cpu, Settings, Upload, Download, Save, FolderOpen,
  RefreshCw, AlertCircle, CheckCircle, Clock, Copy, Trash2, Plus,
  Monitor, Globe, Database, Shield, BarChart3, Layers, Palette
} from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Enhanced code editor component
const CodeEditor = ({ 
  value, 
  onChange, 
  language, 
  placeholder, 
  className = "",
  readOnly = false 
}) => {
  const textareaRef = useRef(null);
  
  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const newValue = value.substring(0, start) + '  ' + value.substring(end);
      onChange(newValue);
      
      // Reset cursor position
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 2;
      }, 0);
    }
  };
  
  return (
    <div className="relative">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={`font-mono text-sm leading-relaxed ${className}`}
        style={{ 
          tabSize: 2,
          whiteSpace: 'pre',
          overflow: 'auto'
        }}
        readOnly={readOnly}
      />
      <div className="absolute top-2 right-2 flex space-x-1">
        <button
          onClick={() => navigator.clipboard.writeText(value)}
          className="p-1 bg-white/10 rounded hover:bg-white/20 text-gray-400"
          title="Copy code"
        >
          <Copy className="w-4 h-4" />
        </button>
        {language && (
          <span className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded">
            {language}
          </span>
        )}
      </div>
    </div>
  );
};

// Project template selector
const ProjectTemplates = ({ onSelect, language }) => {
  const templates = {
    python: [
      { id: 'flask-api', name: 'Flask REST API', description: 'RESTful API with Flask' },
      { id: 'django-web', name: 'Django Web App', description: 'Full-stack Django application' },
      { id: 'fastapi-modern', name: 'FastAPI Modern', description: 'Modern async API with FastAPI' },
      { id: 'ml-project', name: 'ML Project', description: 'Machine Learning project structure' },
      { id: 'cli-tool', name: 'CLI Tool', description: 'Command line interface tool' }
    ],
    javascript: [
      { id: 'react-app', name: 'React App', description: 'Modern React application' },
      { id: 'express-api', name: 'Express API', description: 'Node.js Express REST API' },
      { id: 'nextjs-full', name: 'Next.js Full-Stack', description: 'Full-stack Next.js app' },
      { id: 'electron-app', name: 'Electron App', description: 'Desktop application with Electron' },
      { id: 'vue-spa', name: 'Vue.js SPA', description: 'Single Page Application with Vue' }
    ],
    java: [
      { id: 'spring-boot', name: 'Spring Boot API', description: 'Enterprise Spring Boot application' },
      { id: 'maven-project', name: 'Maven Project', description: 'Standard Maven project structure' },
      { id: 'android-app', name: 'Android App', description: 'Android mobile application' }
    ],
    cpp: [
      { id: 'cmake-project', name: 'CMake Project', description: 'Modern C++ with CMake' },
      { id: 'game-engine', name: 'Game Engine', description: 'Basic game engine structure' }
    ]
  };
  
  const currentTemplates = templates[language] || [];
  
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-gray-300 mb-3">Project Templates</h4>
      {currentTemplates.map(template => (
        <button
          key={template.id}
          onClick={() => onSelect(template)}
          className="w-full text-left p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors border border-white/10"
        >
          <div className="font-medium text-white text-sm">{template.name}</div>
          <div className="text-xs text-gray-400 mt-1">{template.description}</div>
        </button>
      ))}
    </div>
  );
};

// Enhanced session manager
const SessionManager = ({ sessions, activeSession, onSessionSelect, onNewSession }) => (
  <div className="space-y-2">
    <div className="flex items-center justify-between">
      <h4 className="text-sm font-medium text-gray-300">Sessions</h4>
      <button
        onClick={onNewSession}
        className="p-1 bg-white/10 rounded hover:bg-white/20 text-gray-400"
        title="New session"
      >
        <Plus className="w-4 h-4" />
      </button>
    </div>
    <div className="space-y-1 max-h-32 overflow-y-auto">
      {sessions.map(session => (
        <button
          key={session.id}
          onClick={() => onSessionSelect(session)}
          className={`w-full text-left p-2 rounded text-sm transition-colors ${
            session.id === activeSession?.id 
              ? 'bg-blue-500/20 text-blue-200 border border-blue-500/30' 
              : 'text-gray-300 hover:bg-white/5'
          }`}
        >
          <div className="truncate">{session.name}</div>
          <div className="text-xs opacity-70">{session.mode}</div>
        </button>
      ))}
    </div>
  </div>
);

const AICodingAssistantPro = () => {
  // Enhanced state management
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [activeMode, setActiveMode] = useState('chat');
  const [isLoading, setIsLoading] = useState(false);
  const [codeContext, setCodeContext] = useState('');
  const [executionResult, setExecutionResult] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('deepseek/deepseek-chat-v3-0324');
  const [selectedLanguage, setSelectedLanguage] = useState('python');
  const [supportedLanguages, setSupportedLanguages] = useState([]);
  const [apiHealth, setApiHealth] = useState('checking');
  const [currentSession, setCurrentSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [projectConfig, setProjectConfig] = useState({
    description: '',
    framework: '',
    features: [],
    architecture: 'mvc'
  });
  const [analysisResults, setAnalysisResults] = useState(null);
  const [refactorResults, setRefactorResults] = useState(null);
  const [systemStats, setSystemStats] = useState({
    totalSessions: 0,
    codeExecutions: 0,
    analysisCount: 0
  });
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Enhanced modes with more features
  const modes = [
    { 
      id: 'chat', 
      label: 'Smart Chat', 
      icon: Brain, 
      color: 'bg-blue-500', 
      description: 'Intelligent conversation with context awareness',
      features: ['Multi-turn conversations', 'Code understanding', 'Problem solving']
    },
    { 
      id: 'execute', 
      label: 'Code Runner', 
      icon: Play, 
      color: 'bg-green-500', 
      description: 'Run and test code in multiple languages',
      features: ['Multi-language support', 'Input/output handling', 'Performance metrics']
    },
    { 
      id: 'debug', 
      label: 'AI Debugger', 
      icon: Bug, 
      color: 'bg-red-500', 
      description: 'AI-powered debugging and error analysis',
      features: ['Error detection', 'Bug fixing suggestions', 'Code quality checks']
    },
    { 
      id: 'refactor', 
      label: 'Code Optimizer', 
      icon: Zap, 
      color: 'bg-yellow-500', 
      description: 'Optimize and improve code structure',
      features: ['Performance optimization', 'Code modernization', 'Best practices']
    },
    { 
      id: 'analyze', 
      label: 'Code X-Ray', 
      icon: Eye, 
      color: 'bg-purple-500', 
      description: 'Deep code analysis and insights',
      features: ['Security scanning', 'Complexity analysis', 'Quality metrics']
    },
    { 
      id: 'generate', 
      label: 'Project Generator', 
      icon: Layers, 
      color: 'bg-indigo-500', 
      description: 'Generate complete project structures',
      features: ['Project templates', 'Best practices', 'Modern architecture']
    }
  ];

  // Enhanced language configurations
  const languageConfigs = {
    python: { icon: '🐍', color: 'text-blue-400', extension: '.py' },
    javascript: { icon: '🟨', color: 'text-yellow-400', extension: '.js' },
    typescript: { icon: '🔷', color: 'text-blue-500', extension: '.ts' },
    java: { icon: '☕', color: 'text-red-500', extension: '.java' },
    cpp: { icon: '⚡', color: 'text-blue-600', extension: '.cpp' },
    c: { icon: '🔧', color: 'text-gray-500', extension: '.c' },
    go: { icon: '🚀', color: 'text-cyan-400', extension: '.go' },
    rust: { icon: '🦀', color: 'text-orange-500', extension: '.rs' },
    php: { icon: '🐘', color: 'text-purple-400', extension: '.php' },
    ruby: { icon: '💎', color: 'text-red-400', extension: '.rb' },
    bash: { icon: '📟', color: 'text-green-400', extension: '.sh' }
  };

  // Effects
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    initializeApp();
  }, []);

  // Initialize application
  const initializeApp = async () => {
    try {
      await Promise.all([
        checkApiHealth(),
        fetchAvailableModels(),
        loadSessions()
      ]);
      
      addMessage('Welcome to CodeGenius AI Pro! 🚀 Your advanced coding assistant is ready.', 'assistant');
    } catch (error) {
      console.error('Initialization failed:', error);
    }
  };

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/health`);
      setApiHealth('online');
      setSupportedLanguages(response.data.supported_languages || []);
      return response.data;
    } catch (error) {
      setApiHealth('offline');
      console.error('API health check failed:', error);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/models`);
      setAvailableModels(response.data.models || []);
      setSupportedLanguages(response.data.languages || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const loadSessions = () => {
    const savedSessions = localStorage.getItem('ai_coding_sessions');
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions);
      setSessions(parsed);
      if (parsed.length > 0) {
        setCurrentSession(parsed[0]);
      }
    }
  };

  const saveSession = (session) => {
    const updatedSessions = sessions.filter(s => s.id !== session.id);
    updatedSessions.unshift(session);
    setSessions(updatedSessions);
    localStorage.setItem('ai_coding_sessions', JSON.stringify(updatedSessions));
  };

  const createNewSession = () => {
    const newSession = {
      id: Date.now().toString(),
      name: `Session ${new Date().toLocaleTimeString()}`,
      mode: activeMode,
      created: new Date().toISOString(),
      messages: []
    };
    setCurrentSession(newSession);
    setMessages([]);
    addMessage('New session started! How can I help you today?', 'assistant');
  };

  const addMessage = useCallback((content, type = 'user', data = null) => {
    const message = { 
      id: Date.now(), 
      content, 
      type, 
      timestamp: new Date().toLocaleTimeString(),
      data,
      mode: activeMode,
      language: selectedLanguage
    };
    
    setMessages(prev => {
      const updated = [...prev, message];
      
      // Update current session
      if (currentSession) {
        const updatedSession = {
          ...currentSession,
          messages: updated,
          lastActivity: new Date().toISOString()
        };
        saveSession(updatedSession);
      }
      
      return updated;
    });
  }, [activeMode, selectedLanguage, currentSession]);

  // Enhanced message handling
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage = input;
    addMessage(userMessage, 'user');
    setInput('');
    setIsLoading(true);

    try {
      const sessionId = currentSession?.id || Date.now().toString();
      
      switch (activeMode) {
        case 'chat':
          await handleEnhancedChat(userMessage, sessionId);
          break;
        case 'execute':
          await handleEnhancedExecution(userMessage, sessionId);
          break;
        case 'debug':
          await handleDebugMode(userMessage, sessionId);
          break;
        case 'refactor':
          await handleRefactorMode(userMessage, sessionId);
          break;
        case 'analyze':
          await handleAnalyzeMode(userMessage, sessionId);
          break;
        case 'generate':
          await handleGenerateMode(userMessage, sessionId);
          break;
      }
    } catch (error) {
      addMessage(`Error: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEnhancedChat = async (message, sessionId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message,
        model: selectedModel,
        session_id: sessionId,
        context: { language: selectedLanguage, mode: activeMode },
        temperature: 0.7
      });
      
      addMessage(response.data.response, 'assistant', {
        type: 'chat',
        model: response.data.model_used,
        session_id: response.data.session_id
      });
      
      // Auto-detect code in response
      if (response.data.response.includes('```')) {
        const extractedCode = extractCodeFromResponse(response.data.response);
        if (extractedCode) {
          setCodeContext(extractedCode);
        }
      }
    } catch (error) {
      throw error;
    }
  };

  const handleEnhancedExecution = async (code, sessionId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/execute`, {
        language: selectedLanguage,
        code: code,
        session_id: sessionId,
        timeout: 15
      });
      
      setExecutionResult(response.data);
      addMessage('Code executed successfully! ✅', 'assistant', {
        type: 'execution',
        result: response.data,
        language: selectedLanguage
      });
      
      // Update stats
      setSystemStats(prev => ({ ...prev, codeExecutions: prev.codeExecutions + 1 }));
    } catch (error) {
      throw error;
    }
  };

  const handleAnalyzeMode = async (code, sessionId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
        code: code,
        language: selectedLanguage,
        analysis_type: ['syntax', 'complexity', 'security', 'performance'],
        session_id: sessionId
      });
      
      setAnalysisResults(response.data);
      addMessage('Code analysis complete! 🔍', 'assistant', {
        type: 'analysis',
        data: response.data,
        language: selectedLanguage
      });
      
      setSystemStats(prev => ({ ...prev, analysisCount: prev.analysisCount + 1 }));
    } catch (error) {
      throw error;
    }
  };

  const handleRefactorMode = async (code, sessionId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/refactor`, {
        code: code,
        language: selectedLanguage,
        refactor_type: ['performance', 'readability', 'best_practices'],
        session_id: sessionId
      });
      
      setRefactorResults(response.data);
      addMessage('Code refactoring suggestions ready! ⚡', 'assistant', {
        type: 'refactor',
        data: response.data,
        language: selectedLanguage
      });
    } catch (error) {
      throw error;
    }
  };

  const handleGenerateMode = async (description, sessionId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate-project`, {
        description,
        language: selectedLanguage,
        framework: projectConfig.framework,
        features: projectConfig.features,
        architecture: projectConfig.architecture
      });
      
      addMessage('Project structure generated! 🏗️', 'assistant', {
        type: 'project',
        data: response.data,
        language: selectedLanguage
      });
    } catch (error) {
      throw error;
    }
  };

  const handleDebugMode = async (code, sessionId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: `Debug this ${selectedLanguage} code and provide detailed analysis:\n\n${code}`,
        model: selectedModel,
        session_id: sessionId
      });
      
      addMessage('Debug analysis complete! 🐛', 'assistant', {
        type: 'debug',
        analysis: response.data.response,
        language: selectedLanguage
      });
    } catch (error) {
      throw error;
    }
  };

  const extractCodeFromResponse = (response) => {
    const codeMatch = response.match(/```(?:\w+)?\n([\s\S]*?)\n```/);
    return codeMatch ? codeMatch[1] : '';
  };

  const handleTemplateSelect = (template) => {
    setProjectConfig(prev => ({
      ...prev,
      description: `Generate a ${template.name}: ${template.description}`,
      framework: template.id.includes('flask') ? 'flask' : 
                 template.id.includes('react') ? 'react' :
                 template.id.includes('spring') ? 'spring-boot' : ''
    }));
    setShowTemplates(false);
    setActiveMode('generate');
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        setInput(content);
        addMessage(`File uploaded: ${file.name}`, 'system');
      };
      reader.readAsText(file);
    }
  };

  // Enhanced message rendering
  const renderMessage = (message) => {
    const isUser = message.type === 'user';
    const isSystem = message.type === 'system';
    const isError = message.type === 'error';
    
    return (
      <div key={message.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 animate-fadeIn`}>
        <div className={`max-w-4xl w-full px-6 py-4 rounded-xl ${
          isUser ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white' : 
          isError ? 'bg-red-100 text-red-800 border border-red-300' :
          isSystem ? 'bg-gray-100 text-gray-600 border border-gray-300' :
          'bg-white border border-gray-200 shadow-lg'
        }`}>
          <div className="flex items-start space-x-4">
            {!isUser && !isSystem && (
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                <Brain className="w-5 h-5 text-white" />
              </div>
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-semibold">
                  {isUser ? 'You' : isSystem ? 'System' : 'CodeGenius AI Pro'}
                  <span className="text-xs ml-2 opacity-70">{message.timestamp}</span>
                </p>
                {message.language && (
                  <span className="text-xs px-2 py-1 bg-black/10 rounded">
                    {languageConfigs[message.language]?.icon} {message.language}
                  </span>
                )}
              </div>
              
              <div className="prose prose-sm max-w-none">
                <CodeEditor
                  value={message.content}
                  onChange={() => {}}
                  language={message.language}
                  className="bg-transparent border-none resize-none w-full"
                  readOnly={true}
                />
              </div>
              
              {message.data && renderMessageData(message.data)}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderMessageData = (data) => {
    switch (data.type) {
      case 'execution':
        return (
          <div className="mt-4 p-4 bg-gray-900 rounded-lg border">
            <h4 className="font-semibold text-sm mb-3 flex items-center text-green-400">
              <Terminal className="w-4 h-4 mr-2" />
              Execution Result ({data.language})
            </h4>
            <div className="space-y-2">
              {data.result.stdout && (
                <div>
                  <div className="text-xs text-gray-400 mb-1">Output:</div>
                  <pre className="text-sm bg-black/50 text-green-400 p-3 rounded overflow-x-auto">
                    {data.result.stdout}
                  </pre>
                </div>
              )}
              {data.result.stderr && (
                <div>
                  <div className="text-xs text-gray-400 mb-1">Errors:</div>
                  <pre className="text-sm bg-black/50 text-red-400 p-3 rounded overflow-x-auto">
                    {data.result.stderr}
                  </pre>
                </div>
              )}
              {data.result.error && (
                <div>
                  <div className="text-xs text-gray-400 mb-1">Error:</div>
                  <pre className="text-sm bg-black/50 text-red-400 p-3 rounded overflow-x-auto">
                    {data.result.error}
                  </pre>
                </div>
              )}
              <div className="flex items-center justify-between text-xs text-gray-400 mt-3">
                <span>Execution time: {data.result.execution_time}s</span>
                <span>Language: {data.result.language}</span>
              </div>
            </div>
          </div>
        );
        
      case 'analysis':
        return (
          <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
            <h4 className="font-semibold text-sm mb-3 flex items-center text-purple-700">
              <Eye className="w-4 h-4 mr-2" />
              Code Analysis Results
            </h4>
            <div className="prose prose-sm max-w-none text-gray-700">
              <pre className="whitespace-pre-wrap text-sm">{data.data.analysis?.ai_analysis || 'Analysis complete'}</pre>
            </div>
          </div>
        );
        
      case 'refactor':
        return (
          <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
            <h4 className="font-semibold text-sm mb-3 flex items-center text-yellow-700">
              <Zap className="w-4 h-4 mr-2" />
              Refactoring Suggestions
            </h4>
            <div className="prose prose-sm max-w-none text-gray-700">
              <pre className="whitespace-pre-wrap text-sm">{data.data.refactored_code}</pre>
            </div>
          </div>
        );
        
      case 'project':
        return (
          <div className="mt-4 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
            <h4 className="font-semibold text-sm mb-3 flex items-center text-indigo-700">
              <Layers className="w-4 h-4 mr-2" />
              Generated Project Structure
            </h4>
            <div className="prose prose-sm max-w-none text-gray-700">
              <pre className="whitespace-pre-wrap text-sm">{data.data.project_structure}</pre>
            </div>
          </div>
        );
        
      default:
        return (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
            <pre className="text-sm whitespace-pre-wrap text-gray-700">
              {data.analysis || data.suggestions || data.steps || data.data || 'Processing complete'}
            </pre>
          </div>
        );
    }
  };

  const currentMode = modes.find(m => m.id === activeMode);

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex">
      {/* Enhanced Sidebar */}
      <div className="w-96 bg-black/20 backdrop-blur-xl border-r border-white/10 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl flex items-center justify-center">
              <Cpu className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">CodeGenius AI Pro</h1>
              <p className="text-sm text-gray-400">v2.0 - Enhanced Edition</p>
            </div>
          </div>
          
          {/* Mode Selection */}
          <div className="space-y-2">
            {modes.map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all group ${
                    activeMode === mode.id 
                      ? `${mode.color} text-white shadow-lg transform scale-105` 
                      : 'text-gray-300 hover:bg-white/5 hover:transform hover:scale-102'
                  }`}
                  title={mode.description}
                >
                  <Icon className="w-5 h-5" />
                  <div className="flex-1 text-left">
                    <div className="font-medium">{mode.label}</div>
                    <div className="text-xs opacity-70 group-hover:opacity-100">
                      {mode.features?.[0]}
                    </div>
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
              <Settings className="w-4 h-4 mr-2" />
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
        
        {/* Session Management */}
        <div className="p-6 border-b border-white/10">
          <SessionManager
            sessions={sessions}
            activeSession={currentSession}
            onSessionSelect={setCurrentSession}
            onNewSession={createNewSession}
          />
        </div>
        
        {/* Project Templates */}
        {activeMode === 'generate' && (
          <div className="p-6 border-b border-white/10">
            <ProjectTemplates
              onSelect={handleTemplateSelect}
              language={selectedLanguage}
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
                  {currentMode.features.map((feature, index) => (
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
                accept=".txt,.js,.py,.java,.cpp,.c,.go,.rs,.php,.rb,.sh,.ts,.jsx,.tsx"
              />
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 bg-white/10 text-gray-300 hover:bg-white/20 rounded-lg transition-all"
                title="Upload file"
              >
                <Upload className="w-5 h-5" />
              </button>
              
              <button 
                className="p-2 bg-white/10 text-gray-300 hover:bg-white/20 rounded-lg transition-all"
                title="Download session"
              >
                <Download className="w-5 h-5" />
              </button>
              
              <button
                onClick={() => setShowTemplates(!showTemplates)}
                className={`p-2 rounded-lg transition-all ${
                  showTemplates ? 'bg-blue-500 text-white' : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
                title="Project templates"
              >
                <FolderOpen className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Enhanced Messages Area */}
        <div className="flex-1 overflow-y-auto p-6">
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
                <div className="flex flex-wrap justify-center gap-2 max-w-lg mx-auto">
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
            </div>
          )}
          
          {messages.map(renderMessage)}
          
          {isLoading && (
            <div className="flex justify-start mb-6">
              <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-lg animate-fadeIn max-w-md">
                <div className="flex items-center space-x-3">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                  <span className="text-sm text-gray-600">AI is processing your request...</span>
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  Model: {selectedModel.split('/')[1]} • Language: {selectedLanguage}
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
                value={input}
                onChange={setInput}
                language={selectedLanguage}
                placeholder={
                  activeMode === 'chat' ? 'Ask me anything about coding...' :
                  activeMode === 'execute' ? `Paste your ${selectedLanguage} code to run it...` :
                  activeMode === 'debug' ? 'Share your buggy code for analysis...' :
                  activeMode === 'refactor' ? 'Show me code that needs improvement...' :
                  activeMode === 'analyze' ? 'Submit code for deep analysis...' :
                  activeMode === 'generate' ? 'Describe the project you want to generate...' :
                  'Enter your message...'
                }
                className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none min-h-[120px]"
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim() || apiHealth === 'offline'}
              className="bg-gradient-to-r from-purple-500 to-blue-500 text-white p-4 rounded-xl hover:from-purple-600 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95"
              title={apiHealth === 'offline' ? 'Backend server is offline' : 'Send message'}
            >
              <Send className="w-6 h-6" />
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
            </div>
            <div className="flex items-center space-x-4">
              <span>Press Enter to send, Shift+Enter for new line</span>
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