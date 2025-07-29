# 🚀 AI Coding Assistant Pro

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/MayureshMankar/AI-Assistant)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/react-18.0+-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

> **Next-generation AI-powered coding assistant with advanced multi-language support, intelligent analysis, real-time collaboration, and comprehensive development tools.**

![AI Coding Assistant Pro](https://via.placeholder.com/800x400/1a1a1a/ffffff?text=AI+Coding+Assistant+Pro)

## ✨ Features

### 🧠 **Intelligent AI Assistance**
- **Multi-Model Support**: Choose from 6+ advanced AI models (DeepSeek, Qwen, Gemini, Mistral)
- **Context-Aware Conversations**: Maintains context across sessions
- **Smart Code Understanding**: Analyzes and explains code in multiple languages
- **Advanced Reasoning**: Uses latest reasoning models for complex problem-solving

### 💻 **Multi-Language Code Execution**
- **11 Programming Languages**: Python, JavaScript, TypeScript, Java, C++, C, Go, Rust, PHP, Ruby, Bash
- **Real-Time Execution**: Run code instantly with performance metrics
- **Input/Output Handling**: Support for stdin/stdout operations
- **Security Sandboxing**: Safe code execution with timeouts and resource limits

### 🔍 **Advanced Code Analysis**
- **Syntax Analysis**: Real-time syntax checking and validation
- **Security Scanning**: Identify potential vulnerabilities
- **Performance Optimization**: Suggestions for better performance
- **Complexity Metrics**: Cyclomatic complexity and code quality scores
- **Best Practices**: Automated recommendations for improvement

### ⚡ **AI-Powered Refactoring**
- **Smart Refactoring**: Automated code improvements
- **Performance Optimization**: Enhance code efficiency
- **Modernization**: Update code to latest standards
- **Readability Enhancement**: Improve code clarity and structure

### 🏗️ **Project Generation**
- **30+ Project Templates**: Ready-to-use templates for various frameworks
- **Framework Integration**: Flask, Django, React, Express, Spring Boot, and more
- **Best Practices**: Generated code follows industry standards
- **Complete Structure**: Includes configuration, documentation, and setup

### 🛠️ **Enhanced Development Tools**
- **Smart Code Editor**: Syntax highlighting, auto-completion, and formatting
- **File Management**: Upload, download, and manage project files
- **Session Management**: Save and restore coding sessions
- **Version Control Integration**: Git workflow support

### 📊 **Monitoring & Analytics**
- **Real-Time Metrics**: Performance monitoring and usage statistics
- **Health Monitoring**: System status and service health checks
- **Error Tracking**: Comprehensive error logging and debugging
- **Usage Analytics**: Track usage patterns and optimization opportunities

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   PostgreSQL    │
│   (TypeScript)   │◄──►│    (Python)     │◄──►│    Database     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │     Redis       │    │   Monitoring    │
│  (Reverse Proxy) │    │   (Caching)     │    │  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Tech Stack

**Backend**
- **FastAPI**: Modern, fast web framework for Python
- **SQLAlchemy**: Advanced ORM with async support
- **PostgreSQL**: Robust relational database
- **Redis**: High-performance caching and session storage
- **OpenRouter**: Multi-model AI API integration

**Frontend**
- **React 18**: Modern UI framework with hooks
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide Icons**: Beautiful, customizable icons
- **Axios**: HTTP client for API communication

**Infrastructure**
- **Docker**: Containerization for all services
- **Nginx**: High-performance reverse proxy
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Data visualization and dashboards

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **Node.js 18+** and **Python 3.11+** (for local development)
- **OpenRouter API Key** ([Get it here](https://openrouter.ai/))

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/MayureshMankar/AI-Assistant.git
cd AI-Assistant

# Copy environment configuration
cp .env.example .env

# Edit .env file with your OpenRouter API key
nano .env  # or use your preferred editor
```

### 2. Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Initialize database
docker-compose exec backend python -c "from database import initialize_database; initialize_database()"

# Access the application
open http://localhost:3000
```

### 3. Manual Setup (Development)

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database import initialize_database; initialize_database()"

# Start backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Access at http://localhost:3000
```

## 📖 Usage Guide

### 🎯 Getting Started

1. **Open the Application**: Navigate to `http://localhost:3000`
2. **Select Mode**: Choose from 6 intelligent modes:
   - 🧠 **Smart Chat**: Intelligent conversations with context
   - ▶️ **Code Runner**: Execute code in 11+ languages
   - 🐛 **AI Debugger**: Debug and fix code issues
   - ⚡ **Code Optimizer**: Refactor and improve code
   - 👁️ **Code X-Ray**: Deep analysis and insights
   - 🏗️ **Project Generator**: Create complete projects

3. **Configure Settings**: 
   - Select your preferred AI model
   - Choose programming language
   - Set up session preferences

### 💬 Smart Chat Mode

```
You: "How do I implement a binary search in Python?"

AI: I'll help you implement a binary search algorithm in Python...

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

This implementation has O(log n) time complexity...
```

### ▶️ Code Execution

1. **Select Language**: Choose from 11 supported languages
2. **Write/Paste Code**: Use the enhanced code editor
3. **Execute**: Click run and see real-time results
4. **View Metrics**: Execution time, memory usage, and output

### 🔍 Code Analysis

```python
# Example: Analyze this Python function
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Analysis Results:**
- **Complexity**: O(2^n) - exponential time complexity
- **Security**: No vulnerabilities detected
- **Performance**: Consider memoization for better performance
- **Best Practices**: Add type hints and docstring

### 🏗️ Project Generation

1. **Select Template**: Choose from 30+ project templates
2. **Configure**: Set framework, features, and architecture
3. **Generate**: Get a complete, production-ready project structure
4. **Download**: Export your generated project

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Core API
OPENROUTER_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Security
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:3000

# Performance
MAX_FILE_SIZE=10485760
MAX_EXECUTION_TIMEOUT=30
```

### Advanced Configuration

#### Custom AI Models
```python
# Add custom models in backend/main.py
ENHANCED_MODELS = {
    "your-custom-model": {
        "type": "custom", 
        "code_analysis": True, 
        "max_tokens": 8192
    }
}
```

#### Language Support
```python
# Add new languages in backend/main.py
SUPPORTED_LANGUAGES = {
    "kotlin": {"extension": ".kt", "command": ["kotlinc"], "timeout": 15}
}
```

## 🐳 Docker Deployment

### Production Deployment

```bash
# Production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# With SSL/HTTPS
docker-compose -f docker-compose.prod.yml -f docker-compose.ssl.yml up -d
```

### Scaling

```bash
# Scale backend services
docker-compose up --scale backend=3 -d

# Load balancing with nginx
docker-compose -f docker-compose.scale.yml up -d
```

## 📊 Monitoring

### Health Checks

```bash
# Check application health
curl http://localhost:8000/api/health

# Database health
curl http://localhost:8000/api/health/database

# System metrics
curl http://localhost:8000/metrics
```

### Monitoring Dashboard

- **Grafana**: `http://localhost:3001` (admin/admin123)
- **Prometheus**: `http://localhost:9090`
- **Application Logs**: `docker-compose logs -f backend`

## 🔒 Security

### Security Features

- **Rate Limiting**: Configurable limits per user type
- **Input Validation**: Comprehensive request validation
- **Code Sandboxing**: Safe execution environment
- **CORS Protection**: Configurable origin restrictions
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Content sanitization

### Security Best Practices

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Use environment variables for secrets
export OPENROUTER_API_KEY="your-secure-key"

# Enable HTTPS in production
SSL_CERT_PATH=/etc/ssl/certs/cert.pem
SSL_KEY_PATH=/etc/ssl/private/key.pem
```

## 🧪 Development

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt
npm install --include=dev

# Run tests
pytest backend/tests/
npm test

# Code formatting
black backend/
npm run format

# Linting
flake8 backend/
npm run lint
```

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

### Adding New Features

1. **Backend**: Add endpoints in `backend/main.py`
2. **Frontend**: Create components in `frontend/src/`
3. **Database**: Update models in `backend/models.py`
4. **Tests**: Add tests in respective test directories

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Code Standards

- **Python**: Follow PEP 8, use Black formatting
- **JavaScript**: Use ESLint and Prettier
- **Commits**: Use conventional commit messages
- **Documentation**: Update docs for new features

## 📈 Performance

### Optimization Features

- **Caching**: Redis-based response caching
- **Connection Pooling**: Optimized database connections
- **Compression**: Gzip compression for responses
- **CDN Ready**: Static asset optimization
- **Lazy Loading**: Component-based code splitting

### Performance Metrics

- **API Response Time**: < 200ms average
- **Code Execution**: < 5s for most languages
- **Database Queries**: < 50ms average
- **Frontend Load**: < 2s initial load

## 🔄 API Reference

### Core Endpoints

#### Chat API
```http
POST /api/chat
Content-Type: application/json

{
  "message": "Explain async/await in JavaScript",
  "model": "deepseek/deepseek-chat-v3-0324",
  "session_id": "optional-session-id",
  "temperature": 0.7
}
```

#### Code Execution API
```http
POST /api/execute
Content-Type: application/json

{
  "language": "python",
  "code": "print('Hello, World!')",
  "timeout": 10,
  "input_data": ""
}
```

#### Code Analysis API
```http
POST /api/analyze
Content-Type: application/json

{
  "code": "def fibonacci(n): ...",
  "language": "python",
  "analysis_type": ["syntax", "complexity", "security"]
}
```

## 🚨 Troubleshooting

### Common Issues

#### "API Key not configured"
```bash
# Solution: Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"
# Or update .env file
```

#### Database Connection Failed
```bash
# Solution: Initialize database
docker-compose exec backend python -c "from database import initialize_database; initialize_database()"
```

#### Port Already in Use
```bash
# Solution: Change ports in docker-compose.yml
ports:
  - "3001:3000"  # Change from 3000:3000
  - "8001:8000"  # Change from 8000:8000
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# View detailed logs
docker-compose logs -f --tail=100
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenRouter** for AI model access
- **FastAPI** for the excellent web framework
- **React** team for the amazing frontend library
- **All contributors** who help make this project better

## 🔮 Roadmap

### Version 2.1 (Coming Soon)
- [ ] Real-time collaboration features
- [ ] Voice-to-code functionality
- [ ] Advanced debugging tools
- [ ] Mobile application
- [ ] VSCode extension

### Version 3.0 (Future)
- [ ] Custom AI model training
- [ ] Advanced project management
- [ ] Team collaboration features
- [ ] Enterprise authentication
- [ ] Advanced analytics dashboard

---

<div align="center">

**Made with ❤️ by [Mayuresh Mankar](https://github.com/MayureshMankar)**

[⭐ Star this repo](https://github.com/MayureshMankar/AI-Assistant/stargazers) | 
[🐛 Report Bug](https://github.com/MayureshMankar/AI-Assistant/issues) | 
[✨ Request Feature](https://github.com/MayureshMankar/AI-Assistant/issues)

</div> 