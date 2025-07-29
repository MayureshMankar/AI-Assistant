import React, { useState, useEffect, useRef } from 'react';
import { Send, Code, Play, Bug, Mic, FileCode, Users, Sparkles, Terminal, Brain, Zap, GitBranch, Eye, Cpu, Settings } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const AICodingAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [activeMode, setActiveMode] = useState('chat');
  const [isLoading, setIsLoading] = useState(false);
  const [codeContext, setCodeContext] = useState('');
  const [executionResult, setExecutionResult] = useState(null);
  const [debugSession, setDebugSession] = useState(null);
  const [voiceMode, setVoiceMode] = useState(false);
  const [collaborators, setCollaborators] = useState([]);
  const [workflow, setWorkflow] = useState([]);
  const [codeAnalysis, setCodeAnalysis] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('deepseek/deepseek-chat-v3-0324');
  const [apiHealth, setApiHealth] = useState('checking');
  const messagesEndRef = useRef(null);

  const modes = [
    { id: 'chat', label: 'Smart Chat', icon: Brain, color: 'bg-blue-500', description: 'Intelligent conversation with context awareness' },
    { id: 'execute', label: 'Code Runner', icon: Play, color: 'bg-green-500', description: 'Run and test your code in real-time' },
    { id: 'debug', label: 'AI Debugger', icon: Bug, color: 'bg-red-500', description: 'AI-powered debugging and error analysis' },
    { id: 'refactor', label: 'Code Optimizer', icon: Zap, color: 'bg-yellow-500', description: 'Optimize and improve your code structure' },
    { id: 'analyze', label: 'Code X-Ray', icon: Eye, color: 'bg-purple-500', description: 'Deep code analysis and insights' },
    { id: 'workflow', label: 'AI Workflow', icon: GitBranch, color: 'bg-indigo-500', description: 'Automated development workflows' }
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    checkApiHealth();
    fetchAvailableModels();
    // Add welcome message
    addMessage('Welcome to CodeGenius AI! I\'m your revolutionary coding assistant. How can I help you today?', 'assistant');
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/health`);
      setApiHealth(response.status === 200 ? 'online' : 'offline');
    } catch (error) {
      setApiHealth('offline');
      console.error('API health check failed:', error);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/models`);
      setAvailableModels(response.data.free_models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
      setAvailableModels(['deepseek/deepseek-chat-v3-0324']); // fallback
    }
  };

  const addMessage = (content, type = 'user', data = null) => {
    setMessages(prev => [...prev, { 
      id: Date.now(), 
      content, 
      type, 
      timestamp: new Date().toLocaleTimeString(),
      data 
    }]);
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = input;
    addMessage(userMessage, 'user');
    setInput('');
    setIsLoading(true);

    try {
      switch (activeMode) {
        case 'chat':
          await handleChatMode(userMessage);
          break;
        case 'execute':
          await handleExecuteMode(userMessage);
          break;
        case 'debug':
          await handleDebugMode(userMessage);
          break;
        case 'refactor':
          await handleRefactorMode(userMessage);
          break;
        case 'analyze':
          await handleAnalyzeMode(userMessage);
          break;
        case 'workflow':
          await handleWorkflowMode(userMessage);
          break;
      }
    } catch (error) {
      addMessage(`Error: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChatMode = async (message) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: `Context: ${codeContext}\n\nQuery: ${message}`,
        model: selectedModel
      });
      
      addMessage(response.data.response, 'assistant');
      
      // Auto-detect if response contains code
      if (response.data.response.includes('```')) {
        setCodeContext(extractCodeFromResponse(response.data.response));
      }
    } catch (error) {
      throw error;
    }
  };

  const handleExecuteMode = async (code) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/execute`, {
        language: 'python',
        code: code
      });
      
      setExecutionResult(response.data);
      addMessage('Code executed successfully!', 'assistant', {
        type: 'execution',
        result: response.data
      });
    } catch (error) {
      throw error;
    }
  };

  const handleDebugMode = async (code) => {
    // For now, use chat mode with debug context
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: `Please debug this code and find potential issues:\n\n${code}`,
        model: selectedModel
      });
      
      addMessage('Debug analysis complete!', 'assistant', {
        type: 'debug',
        analysis: response.data.response
      });
    } catch (error) {
      throw error;
    }
  };

  const handleRefactorMode = async (code) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: `Please refactor this code to improve performance, readability, and best practices:\n\n${code}`,
        model: selectedModel
      });
      
      addMessage('Code refactoring suggestions ready!', 'assistant', {
        type: 'refactor',
        suggestions: response.data.response
      });
    } catch (error) {
      throw error;
    }
  };

  const handleAnalyzeMode = async (code) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: `Please analyze this code for performance, security, patterns, and provide detailed metrics:\n\n${code}`,
        model: selectedModel
      });
      
      addMessage('Comprehensive code analysis complete!', 'assistant', {
        type: 'analysis',
        data: response.data.response
      });
    } catch (error) {
      throw error;
    }
  };

  const handleWorkflowMode = async (task) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: `Generate a step-by-step development workflow for: ${task}`,
        model: selectedModel
      });
      
      addMessage('AI workflow generated!', 'assistant', {
        type: 'workflow',
        steps: response.data.response
      });
    } catch (error) {
      throw error;
    }
  };

  const extractCodeFromResponse = (response) => {
    const codeMatch = response.match(/```(?:python|javascript|java|cpp)?\n([\s\S]*?)\n```/);
    return codeMatch ? codeMatch[1] : '';
  };

  const toggleVoiceMode = () => {
    setVoiceMode(!voiceMode);
    if (!voiceMode) {
      addMessage('Voice mode activated. Speak your command...', 'system');
    } else {
      addMessage('Voice mode deactivated.', 'system');
    }
  };

  const renderMessage = (message) => {
    const isUser = message.type === 'user';
    const isSystem = message.type === 'system';
    const isError = message.type === 'error';
    
    return (
      <div key={message.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-fadeIn`}>
        <div className={`max-w-3xl px-4 py-3 rounded-lg ${
          isUser ? 'bg-blue-600 text-white' : 
          isError ? 'bg-red-100 text-red-800 border border-red-300' :
          isSystem ? 'bg-gray-100 text-gray-600 border border-gray-300' :
          'bg-white border border-gray-200 shadow-sm'
        }`}>
          <div className="flex items-start space-x-3">
            {!isUser && !isSystem && (
              <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                <Brain className="w-4 h-4 text-white" />
              </div>
            )}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium mb-1">
                {isUser ? 'You' : isSystem ? 'System' : 'CodeGenius AI'}
                <span className="text-xs ml-2 opacity-70">{message.timestamp}</span>
              </p>
              <div className="prose prose-sm max-w-none">
                <pre className="whitespace-pre-wrap font-sans">{message.content}</pre>
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
          <div className="mt-3 p-3 bg-gray-50 rounded border">
            <h4 className="font-medium text-sm mb-2 flex items-center">
              <Terminal className="w-4 h-4 mr-2" />
              Execution Result:
            </h4>
            <pre className="text-xs bg-black text-green-400 p-2 rounded overflow-x-auto code-block">
              {data.result.stdout || 'No output'}
              {data.result.stderr && (
                <span className="text-red-400">{data.result.stderr}</span>
              )}
              {data.result.error && (
                <span className="text-red-400">Error: {data.result.error}</span>
              )}
            </pre>
          </div>
        );
      
      default:
        return (
          <div className="mt-3 p-3 bg-gray-50 rounded border">
            <pre className="text-sm whitespace-pre-wrap">{data.analysis || data.suggestions || data.steps || data.data}</pre>
          </div>
        );
    }
  };

  const currentMode = modes.find(m => m.id === activeMode);

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* Sidebar */}
      <div className="w-80 bg-black/20 backdrop-blur-xl border-r border-white/10 flex flex-col sidebar">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl flex items-center justify-center">
              <Cpu className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">CodeGenius AI</h1>
              <p className="text-sm text-gray-400">Revolutionary Assistant</p>
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
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                    activeMode === mode.id 
                      ? `${mode.color} text-white shadow-lg` 
                      : 'text-gray-300 hover:bg-white/5'
                  }`}
                  title={mode.description}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{mode.label}</span>
                </button>
              );
            })}
          </div>
        </div>
        
        {/* Model Selection */}
        <div className="p-6 border-b border-white/10">
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
              <option key={model} value={model} className="bg-gray-800">
                {model.split('/')[1] || model}
              </option>
            ))}
          </select>
        </div>
        
        {/* Status Panel */}
        <div className="flex-1 p-6 space-y-4">
          <div className="bg-white/5 rounded-xl p-4 border border-white/10">
            <h3 className="text-white font-medium mb-3 flex items-center">
              <Sparkles className="w-4 h-4 mr-2" />
              System Status
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between text-gray-300">
                <span>API Status</span>
                <div className={`w-2 h-2 rounded-full ${
                  apiHealth === 'online' ? 'bg-green-500' : 
                  apiHealth === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                }`}></div>
              </div>
              <div className="flex items-center justify-between text-gray-300">
                <span>Context Awareness</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex items-center justify-between text-gray-300">
                <span>Real-time Analysis</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
            </div>
          </div>
          
          {collaborators.length > 0 && (
            <div className="bg-white/5 rounded-xl p-4 border border-white/10">
              <h3 className="text-white font-medium mb-3 flex items-center">
                <Users className="w-4 h-4 mr-2" />
                Collaborators ({collaborators.length})
              </h3>
              <div className="flex -space-x-2">
                {collaborators.map((collab, idx) => (
                  <div key={idx} className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-full border-2 border-white/20"></div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col main-content">
        {/* Header */}
        <div className="bg-black/10 backdrop-blur-xl border-b border-white/10 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">
                {currentMode?.label}
              </h2>
              <p className="text-sm text-gray-400">
                {currentMode?.description}
              </p>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={toggleVoiceMode}
                className={`p-2 rounded-lg transition-all ${
                  voiceMode ? 'bg-red-500 text-white' : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
                title="Toggle Voice Mode"
              >
                <Mic className="w-5 h-5" />
              </button>
              <button 
                className="p-2 bg-white/10 text-gray-300 hover:bg-white/20 rounded-lg transition-all"
                title="File Upload"
              >
                <FileCode className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Ready to revolutionize your coding!</h3>
              <p className="text-gray-400 max-w-md mx-auto">
                Start a conversation, run code, debug issues, or let AI analyze your codebase. 
                I'm here to make you a 10x developer.
              </p>
            </div>
          )}
          
          {messages.map(renderMessage)}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm animate-fadeIn">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  <span className="text-sm text-gray-600 ml-2">AI is thinking...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-black/10 backdrop-blur-xl border-t border-white/10 p-6">
          <div className="flex items-end space-x-4">
            <div className="flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder={
                  activeMode === 'chat' ? 'Ask me anything about coding...' :
                  activeMode === 'execute' ? 'Paste your code to run it...' :
                  activeMode === 'debug' ? 'Share your buggy code for analysis...' :
                  activeMode === 'refactor' ? 'Show me code that needs improvement...' :
                  activeMode === 'analyze' ? 'Submit code for deep analysis...' :
                  'Describe your development workflow...'
                }
                className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                rows={3}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim() || apiHealth === 'offline'}
              className="bg-gradient-to-r from-purple-500 to-blue-500 text-white p-3 rounded-xl hover:from-purple-600 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              title={apiHealth === 'offline' ? 'Backend server is offline' : 'Send message'}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          
          <div className="flex items-center justify-between mt-4 text-xs text-gray-400">
            <div className="flex items-center space-x-4">
              <span>Model: {selectedModel.split('/')[1] || selectedModel}</span>
              <span>•</span>
              <span>Context: {codeContext ? 'Active' : 'Empty'}</span>
              <span>•</span>
              <span className={apiHealth === 'online' ? 'text-green-400' : 'text-red-400'}>
                {apiHealth === 'online' ? 'Online' : apiHealth === 'offline' ? 'Offline' : 'Checking...'}
              </span>
            </div>
            <div>
              Press Enter to send, Shift+Enter for new line
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AICodingAssistant;