from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, Depends, BackgroundTasks, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import os
import io
import ast
import json
import asyncio
import aiofiles
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import tempfile
import sys
from datetime import datetime, timedelta
import time
from collections import defaultdict
from typing import List, Optional, Dict, Any
import uuid
import hashlib
import re
from pathlib import Path
import zipfile
import shutil
from sqlalchemy.orm import Session
from database import get_db, SessionLocal
from models import ChatSession, ChatMessage, CodeExecution, CodeAnalysis, FileUpload, APIUsage
import logging
from uuid import UUID
import traceback


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="AI Coding Assistant Pro",
    description="Advanced AI-powered coding assistant with multi-language support, intelligent analysis, and collaborative features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Supported programming languages with execution commands
SUPPORTED_LANGUAGES = {
    "python": {"extension": ".py", "command": [sys.executable], "timeout": 10},
    "javascript": {"extension": ".js", "command": ["node"], "timeout": 10},
    "typescript": {"extension": ".ts", "command": ["ts-node"], "timeout": 10},
    "java": {"extension": ".java", "command": ["java"], "timeout": 15},
    "cpp": {"extension": ".cpp", "command": ["g++", "-o", "temp", "temp.cpp", "&&", "./temp"], "timeout": 15},
    "c": {"extension": ".c", "command": ["gcc", "-o", "temp", "temp.c", "&&", "./temp"], "timeout": 15},
    "go": {"extension": ".go", "command": ["go", "run"], "timeout": 10},
    "rust": {"extension": ".rs", "command": ["rustc", "temp.rs", "&&", "./temp"], "timeout": 15},
    "php": {"extension": ".php", "command": ["php"], "timeout": 10},
    "ruby": {"extension": ".rb", "command": ["ruby"], "timeout": 10},
    "bash": {"extension": ".sh", "command": ["bash"], "timeout": 5}
}

# Replace the existing ENHANCED_MODELS with only working free models
ENHANCED_MODELS = {
    "google/gemini-2.0-flash-exp:free": {
        "type": "multimodal",
        "code_analysis": True,
        "max_tokens": 1048576,
        "description": "Google: Gemini 2.0 Flash Experimental (free)"
    },
    "qwen/qwen3-coder:free": {
        "type": "programming",
        "code_analysis": True,
        "max_tokens": 262144,
        "description": "Qwen: Qwen3 Coder (free)"
    },
    "tngtech/deepseek-r1t2-chimera:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 163840,
        "description": "TNG: DeepSeek R1T2 Chimera (free)"
    },
    "deepseek/deepseek-r1-0528:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 163840,
        "description": "DeepSeek: R1 0528 (free)"
    },
    "tngtech/deepseek-r1t-chimera:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 163840,
        "description": "TNG: DeepSeek R1T Chimera (free)"
    },
    "microsoft/mai-ds-r1:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 163840,
        "description": "Microsoft: MAI DS R1 (free)"
    },
    "deepseek/deepseek-r1:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 163840,
        "description": "DeepSeek: R1 (free)"
    },
    "z-ai/glm-4.5-air:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "Z.AI: GLM 4.5 Air (free)"
    },
    "moonshotai/kimi-dev-72b:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "Kimi Dev 72b (free)"
    },
    "deepseek/deepseek-r1-0528-qwen3-8b:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "Deepseek R1 0528 Qwen3 8B (free)"
    },
    "qwen/qwen3-235b-a22b:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "Qwen: Qwen3 235B A22B (free)"
    },
    "moonshotai/kimi-vl-a3b-thinking:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "Moonshot AI: Kimi VL A3B Thinking (free)"
    },
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": {
        "type": "reasoning",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "NVIDIA: Llama 3.1 Nemotron Ultra 253B v1 (free)"
    }
}

# Language detection patterns for auto-detection
LANGUAGE_PATTERNS = {
    'python': [r'def\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'class\s+\w+', r'#.*python'],
    'javascript': [r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+', r'=>\s*{', r'console\.log'],
    'typescript': [r'interface\s+\w+', r'type\s+\w+', r':\s*string', r':\s*number', r':\s*boolean'],
    'java': [r'public\s+class', r'public\s+static\s+void\s+main', r'System\.out\.println', r'import\s+java\.'],
    'cpp': [r'#include\s*<', r'using\s+namespace', r'int\s+main', r'cout\s*<<', r'std::'],
    'c': [r'#include\s*<', r'int\s+main', r'printf\s*\(', r'scanf\s*\('],
    'go': [r'package\s+main', r'func\s+main', r'import\s+\(', r'fmt\.Print'],
    'rust': [r'fn\s+main', r'let\s+mut', r'use\s+std::', r'println!'],
    'php': [r'<\?php', r'echo\s+', r'\$\w+', r'function\s+\w+'],
    'ruby': [r'def\s+\w+', r'puts\s+', r'class\s+\w+', r'require\s+'],
    'bash': [r'#!/bin/bash', r'echo\s+', r'if\s+\[', r'for\s+\w+\s+in'],
    'html': [r'<html', r'<head', r'<body', r'<div', r'<script'],
    'css': [r'\.\w+\s*{', r'#\w+\s*{', r'@media', r'font-family:'],
    'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO', r'CREATE\s+TABLE'],
    'json': [r'{\s*"', r'\[\s*{', r'":\s*"'],
    'yaml': [r'---', r':\s*$', r'^\s*-\s+'],
    'markdown': [r'^#\s+', r'^\*\s+', r'\[.*\]\(.*\)', r'```']
}

def detect_language(code):
    """Auto-detect programming language from code content"""
    if not code or len(code.strip()) < 10:
        return 'text'
    
    lines = code.split('\n')[:10]  # Check first 10 lines
    scores = {}
    
    for lang, patterns in LANGUAGE_PATTERNS.items():
        scores[lang] = 0
        for pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    scores[lang] += 1
    
    if not scores:
        return 'text'
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'text'
    
    detected_lang = max(scores, key=scores.get)
    return detected_lang

# Enhanced rate limiting with user-based limits
RATE_LIMITS = {
    "anonymous": {"requests": 30, "period": 60},
    "authenticated": {"requests": 100, "period": 60},
    "premium": {"requests": 500, "period": 60}
}

rate_limit_store = defaultdict(list)
active_sessions = {}

# Enhanced middleware
@app.middleware("http")
async def enhanced_logging_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path} - {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(f"[{request_id}] {response.status_code} - {process_time:.2f}ms")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] ERROR - {process_time:.2f}ms - {str(e)}")
        raise

@app.middleware("http")
async def enhanced_rate_limiter(request: Request, call_next):
    ip = request.client.host
    user_type = "anonymous"  # Could be enhanced with auth
    
    limits = RATE_LIMITS[user_type]
    now = time.time()
    
    # Clean old timestamps
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < limits["period"]]
    
    if len(rate_limit_store[ip]) >= limits["requests"]:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "retry_after": limits["period"],
                "limit": limits["requests"]
            }
        )
    
    rate_limit_store[ip].append(now)
    return await call_next(request)

# Enhanced Models
class EnhancedChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = "deepseek/deepseek-chat-v3-0324"
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4096, ge=1, le=32768)

class EnhancedCodeExecutionRequest(BaseModel):
    language: str = Field(..., pattern="^(python|javascript|typescript|java|cpp|c|go|rust|php|ruby|bash)$")
    code: str = Field(..., min_length=1, max_length=50000)
    session_id: Optional[str] = None
    input_data: Optional[str] = None
    timeout: Optional[int] = Field(10, ge=1, le=30)

class CodeAnalysisRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100000)
    language: str = "python"
    analysis_type: List[str] = ["syntax", "complexity", "security", "performance"]
    session_id: Optional[str] = None

class ProjectGenerationRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=1000)
    language: str = "python"
    framework: Optional[str] = None
    features: List[str] = []
    architecture: Optional[str] = "mvc"

class RefactorRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100000)
    language: str = "python"
    refactor_type: List[str] = ["performance", "readability", "best_practices"]
    session_id: Optional[str] = None

# Utility functions
async def analyze_code_syntax(code: str, language: str) -> Dict[str, Any]:
    """Analyze code syntax and structure"""
    analysis = {
        "syntax_errors": [],
        "warnings": [],
        "metrics": {},
        "suggestions": []
    }
    
    if language == "python":
        try:
            tree = ast.parse(code)
            analysis["metrics"]["lines"] = len(code.split('\n'))
            analysis["metrics"]["functions"] = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            analysis["metrics"]["classes"] = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            analysis["metrics"]["complexity"] = calculate_complexity(tree)
        except SyntaxError as e:
            analysis["syntax_errors"].append({
                "line": e.lineno,
                "message": e.msg,
                "type": "SyntaxError"
            })
    
    return analysis

def calculate_complexity(tree) -> int:
    """Calculate cyclomatic complexity"""
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return complexity

async def get_ai_response(prompt: str, model: str = "deepseek/deepseek-chat-v3-0324", **kwargs) -> str:
    """Enhanced AI response with better error handling"""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    try:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        model_config = ENHANCED_MODELS.get(model, ENHANCED_MODELS["deepseek/deepseek-chat-v3-0324"])
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", model_config["max_tokens"]),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"AI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

# Enhanced Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Coding Assistant Pro - Advanced Version",
        "version": "2.0.0",
        "features": [
            "Multi-language code execution",
            "Advanced code analysis",
            "AI-powered refactoring",
            "Project generation",
            "Real-time collaboration",
            "File management"
        ]
    }

# Update health check to show only working free models
@app.get("/api/health")
async def health_check():
    """Health check with working free models information"""
    try:
        return {
            "status": "online",
            "version": "2.0.0-vscode-edition",
            "features": {
                "chat": True,
                "code_analysis": True,
                "auto_language_detection": True,
                "context_handling": True,
                "free_models": True
            },
            "free_models": list(ENHANCED_MODELS.keys()),
            "models_count": len(ENHANCED_MODELS),
            "supported_languages": list(LANGUAGE_PATTERNS.keys()),
            "max_context": max(model['max_tokens'] for model in ENHANCED_MODELS.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Add endpoint to get available free models
@app.get("/api/models")
async def get_free_models():
    """Get list of available free models"""
    return {
        "models": [
            {
                "id": model_id,
                "name": model_id.split('/')[-1].replace(':free', '').replace('-', ' ').title(),
                "type": model_info["type"],
                "max_tokens": model_info["max_tokens"],
                "description": model_info["description"],
                "free": True,
                "code_analysis": model_info["code_analysis"]
            }
            for model_id, model_info in ENHANCED_MODELS.items()
        ],
        "total": len(ENHANCED_MODELS),
        "timestamp": datetime.utcnow().isoformat()
    }

# Auto language detection endpoint
@app.post("/api/detect-language")
async def detect_language_endpoint(request: dict):
    """Auto-detect programming language from code"""
    try:
        code = request.get("code", "")
        if not code:
            return {"language": "text", "confidence": 0}
        
        detected = detect_language(code)
        
        # Calculate confidence based on pattern matches
        lines = code.split('\n')[:10]
        patterns = LANGUAGE_PATTERNS.get(detected, [])
        matches = 0
        
        for pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    matches += 1
        
        confidence = min(matches / max(len(patterns), 1), 1.0)
        
        return {
            "language": detected,
            "confidence": confidence,
            "total_patterns": len(patterns) if detected in LANGUAGE_PATTERNS else 0,
            "matches_found": matches
        }
    except Exception as e:
        return {"language": "text", "confidence": 0, "error": str(e)}

# Set a valid default model from the current free models
DEFAULT_MODEL = list(ENHANCED_MODELS.keys())[0]

@app.post("/api/chat")
async def enhanced_chat(request: EnhancedChatRequest, db: Session = Depends(get_db)):
    """Enhanced chat with improved context handling for VS Code IDE interface"""
    try:
        # ✅ Convert session_id to UUID
        session_uuid = None
        if request.session_id:
            try:
                session_uuid = UUID(str(request.session_id))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid session_id format. Must be a valid UUID.")

        # ✅ Check if session exists before using it
        if session_uuid:
            session_exists = db.query(ChatSession.id).filter_by(id=session_uuid).scalar()
            if not session_exists:
                session_uuid = None  # Avoid foreign key error

        # ✅ Log API usage safely
        api_usage = APIUsage(
            endpoint="/api/chat",
            method="POST",
            model_used=request.model,
            session_id=session_uuid  # Now guaranteed to be None or valid
        )
        db.add(api_usage)
        db.commit()

        # Check if model is in our free models list
        if request.model not in ENHANCED_MODELS:
            request.model = DEFAULT_MODEL

        model_info = ENHANCED_MODELS[request.model]

        # Rest of your logic
        context_parts = []
        
        # Add system context
        system_context = f"""You are an expert AI coding assistant integrated into a VS Code-like IDE environment.

Environment Details:
- IDE: VS Code-like interface with file explorer, editor, and AI assistant
- Model: {request.model} ({model_info['description']})
- Auto language detection: Enabled
- Context limit: {model_info['max_tokens']} tokens
- Free model with full capabilities

Your capabilities:
- Code analysis and debugging across all programming languages
- Auto-detect programming languages from code snippets
- Provide detailed explanations and solutions
- Generate, refactor, and optimize code
- Best practices and code review
- Multi-file project assistance
- Context-aware responses based on conversation history

Guidelines:
- Use markdown code blocks with language specification
- Provide practical, actionable advice
- Explain your reasoning
- Be concise but thorough
- Maintain conversation context
- Focus on code quality and best practices"""
        
        context_parts.append(system_context)
        
        # Add file context if available
        if hasattr(request, 'context') and request.context:
            context_data = request.context
            
            # Add conversation history
            if 'conversation_history' in context_data:
                history = context_data['conversation_history']
                if isinstance(history, list) and len(history) > 0:
                    context_parts.append("\nConversation History:")
                    for msg in history[-10:]:  # Last 10 messages
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            role = msg['role'].upper()
                            content = msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content']
                            context_parts.append(f"{role}: {content}")
            
            # Add file context
            if 'files' in context_data and context_data['files']:
                context_parts.append(f"\nProject Files ({len(context_data['files'])}):")
                for file_info in context_data['files'][:5]:  # Limit to 5 files
                    if isinstance(file_info, dict):
                        name = file_info.get('name', 'unknown')
                        lang = file_info.get('language', 'unknown')
                        context_parts.append(f"- {name} ({lang})")
            
            # Add active file context
            if 'active_file' in context_data and context_data['active_file']:
                context_parts.append(f"\nActive File: {context_data['active_file']}")
            
            # Add editor content context (limited)
            if 'editor_content' in context_data and context_data['editor_content']:
                editor_content = context_data['editor_content']
                detected_lang = detect_language(editor_content)
                context_parts.append(f"\nEditor Content ({detected_lang}):")
                context_parts.append(f"```{detected_lang}\n{editor_content}\n```")
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Prepare the enhanced message
        enhanced_message = f"{full_context}\n\nUser Question: {request.message}"
        
        # Limit total context size to prevent token overflow
        max_context_length = model_info['max_tokens'] - 1000  # Reserve space for response
        if len(enhanced_message) > max_context_length:
            # Truncate context but keep user message
            truncated_context = full_context[:max_context_length - len(request.message) - 100]
            enhanced_message = f"{truncated_context}\n...\n\nUser Question: {request.message}"
        
        # Call OpenRouter API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": enhanced_message
                }
            ],
            temperature=request.temperature,
            max_tokens=min(4000, model_info['max_tokens'] // 2),  # Conservative limit
            stream=False
        )
        
        ai_response = response.choices[0].message.content
        
        # Generate or use session UUID
        session_uuid = request.session_id or uuid.uuid4()
        
        # Avoid duplicate session insert
        existing_session = db.query(ChatSession).filter(ChatSession.id == session_uuid).first()
        
        if not existing_session:
            chat_session = ChatSession(
                id=session_uuid,
                user_id=None,
                session_name="New Session",
                mode="chat"
            )
            db.add(chat_session)
            db.commit()
        
        return {
            "response": ai_response,
            "model_used": request.model,
            "session_id": request.session_id,
            "context_included": bool(hasattr(request, 'context') and request.context),
            "detected_language": detect_language(request.message) if '```' in request.message else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}\n{traceback.format_exc()}")
        # Fallback response for better user experience
        fallback_response = f"""I apologize, but I encountered an error processing your request. \n\nError details: {e}\n\nHowever, I'm still here to help! Please try:\n1. Simplifying your question\n2. Breaking it into smaller parts\n3. Providing specific code examples\n\nI can assist with:\n- Code debugging and analysis\n- Programming questions\n- Best practices\n- Code generation and refactoring\n\nWhat would you like help with?"""
        return {"response": fallback_response, "model": request.model, "error": str(e)}

@app.post("/api/execute")
async def enhanced_code_execution(request: EnhancedCodeExecutionRequest, db: Session = Depends(get_db)):
    """Enhanced multi-language code execution with security"""
    language = request.language.lower()
    
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{language}' not supported. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    lang_config = SUPPORTED_LANGUAGES[language]
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", 
            suffix=lang_config["extension"], 
            delete=False
        ) as tmp_file:
            tmp_file.write(request.code)
            tmp_path = tmp_file.name
        
        start_time = time.time()
        
        # Execute code with timeout
        if language == "python":
            cmd = [sys.executable, tmp_path]
        elif language == "javascript":
            cmd = ["node", tmp_path]
        elif language == "java":
            # Compile and run Java
            class_name = "TempClass"
            compile_result = subprocess.run(
                ["javac", tmp_path], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if compile_result.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "error": "Compilation failed",
                    "execution_time": 0
                }
            cmd = ["java", "-cp", os.path.dirname(tmp_path), "TempClass"]
        else:
            cmd = lang_config["command"] + [tmp_path]
        
        # Run with input data if provided
        input_data = request.input_data.encode() if request.input_data else None
        
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=request.timeout or lang_config["timeout"]
        )
        
        execution_time = time.time() - start_time
        
        # Clean up
        os.unlink(tmp_path)
        if language == "java":
            class_file = tmp_path.replace(".java", ".class")
            if os.path.exists(class_file):
                os.unlink(class_file)
        
        # Save execution result
        try:
            execution_record = CodeExecution(
                session_id=request.session_id or str(uuid.uuid4()),
                language=language,
                code=request.code,
                stdout=result.stdout,
                stderr=result.stderr,
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
                execution_time=execution_time
            )
            db.add(execution_record)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": "" if result.returncode == 0 else f"Process exited with code {result.returncode}",
            "execution_time": round(execution_time, 3),
            "language": language
        }
        
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return {
            "stdout": "",
            "stderr": "",
            "error": f"Execution timed out after {request.timeout or lang_config['timeout']} seconds",
            "execution_time": request.timeout or lang_config['timeout'],
            "language": language
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Code execution error: {str(e)}")
        return {
            "stdout": "",
            "stderr": "",
            "error": str(e),
            "execution_time": 0,
            "language": language
        }

@app.post("/api/analyze")
async def enhanced_code_analysis(request: CodeAnalysisRequest, db: Session = Depends(get_db)):
    """Enhanced code analysis with multiple analysis types"""
    try:
        analysis_results = {}
        
        # Syntax analysis
        if "syntax" in request.analysis_type:
            analysis_results["syntax"] = await analyze_code_syntax(request.code, request.language)
        
        # AI-powered analysis
        ai_analysis_prompt = f"""
        Analyze this {request.language} code for the following aspects: {', '.join(request.analysis_type)}
        
        Code:
        ```{request.language}
        {request.code}
        ```
        
        Provide detailed analysis including:
        1. Code quality assessment
        2. Performance optimization suggestions
        3. Security vulnerabilities
        4. Best practices recommendations
        5. Refactoring opportunities
        
        Format the response as structured analysis.
        """
        
        ai_analysis = await get_ai_response(ai_analysis_prompt)
        analysis_results["ai_analysis"] = ai_analysis
        
        # Save analysis
        try:
            analysis_record = CodeAnalysis(
                session_id=request.session_id or str(uuid.uuid4()),
                code=request.code,
                analysis_type=",".join(request.analysis_type),
                results=analysis_results,
                model_used="deepseek/deepseek-chat-v3-0324"
            )
            db.add(analysis_record)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
        
        return {
            "analysis": analysis_results,
            "language": request.language,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refactor")
async def enhanced_refactor(request: RefactorRequest, db: Session = Depends(get_db)):
    """AI-powered code refactoring"""
    try:
        refactor_prompt = f"""
        Refactor this {request.language} code focusing on: {', '.join(request.refactor_type)}
        
        Original Code:
        ```{request.language}
        {request.code}
        ```
        
        Please provide:
        1. Refactored code with improvements
        2. Explanation of changes made
        3. Performance impact assessment
        4. Maintainability improvements
        
        Focus on: {', '.join(request.refactor_type)}
        """
        
        refactored_response = await get_ai_response(refactor_prompt)
        
        return {
            "refactored_code": refactored_response,
            "original_code": request.code,
            "refactor_types": request.refactor_type,
            "language": request.language,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Refactor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-project")
async def generate_project(request: ProjectGenerationRequest):
    """Generate complete project structure with AI"""
    try:
        generation_prompt = f"""
        Generate a complete {request.language} project structure for: {request.description}
        
        Requirements:
        - Language: {request.language}
        - Framework: {request.framework or 'standard'}
        - Features: {', '.join(request.features) if request.features else 'basic functionality'}
        - Architecture: {request.architecture}
        
        Please provide:
        1. Project structure (folders and files)
        2. Main application code
        3. Configuration files
        4. Documentation (README)
        5. Dependencies/requirements
        6. Setup instructions
        
        Make it production-ready with best practices.
        """
        
        project_structure = await get_ai_response(generation_prompt)
        
        return {
            "project_structure": project_structure,
            "description": request.description,
            "language": request.language,
            "framework": request.framework,
            "features": request.features,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Project generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Enhanced Cursor-like Features ---

class ComposerRequest(BaseModel):
    prompt: str
    model: str = "deepseek/deepseek-chat-v3-0324"
    language: str = "python"
    session_id: Optional[str] = None
    files: List[dict] = []

class DebugRequest(BaseModel):
    code: str
    language: str = "python"
    session_id: Optional[str] = None
    error_message: Optional[str] = None

class TerminalRequest(BaseModel):
    natural_command: str
    session_id: Optional[str] = None
    context: Optional[dict] = {}

class FileContextRequest(BaseModel):
    file_path: str
    content: str
    operation: str = "reference"  # reference, analyze, modify

@app.post("/api/composer")
async def ai_composer(request: ComposerRequest, db: Session = Depends(get_db)):
    """AI Composer - Multi-file editing with AI assistance (Cursor Composer equivalent)"""
    try:
        # Log API usage
        api_usage = APIUsage(
            endpoint="/api/composer",
            model_used=request.model,
            language=request.language,
            session_id=request.session_id
        )
        db.add(api_usage)
        db.commit()

        composer_prompt = f"""
        You are an AI Composer assistant, equivalent to Cursor's Composer feature.
        Analyze the user's request and provide multi-file editing suggestions.

        User Request: {request.prompt}
        Language: {request.language}
        Referenced Files: {len(request.files)} files

        For this request, provide:
        1. **File Modifications**: List of files to modify with specific changes
        2. **New Files**: Any new files that need to be created
        3. **Refactoring**: Code structure improvements
        4. **Dependencies**: Any new dependencies or imports needed
        5. **Testing**: Suggested test modifications
        6. **Documentation**: Updates to comments or docs

        Format your response as structured suggestions with clear file paths and code changes.
        Focus on maintaining code quality, consistency, and best practices.
        """

        response = await get_ai_response(composer_prompt)
        
        return {
            "suggestions": response,
            "files_modified": [f["name"] for f in request.files],
            "language": request.language,
            "model_used": request.model,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Composer error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/debug")
async def ai_debugger(request: DebugRequest, db: Session = Depends(get_db)):
    """AI Debugger - Intelligent debugging and error fixing (Cursor Debug equivalent)"""
    try:
        # Log API usage
        api_usage = APIUsage(
            endpoint="/api/debug",
            language=request.language,
            session_id=request.session_id
        )
        db.add(api_usage)
        db.commit()

        debug_prompt = f"""
        You are an expert AI debugger, equivalent to Cursor's debugging features.
        Analyze the provided code and help identify and fix issues.

        Code to Debug:
        ```{request.language}
        {request.code}
        ```

        Error Message (if any): {request.error_message or "No specific error provided"}

        Please provide:
        1. **Error Analysis**: Identify potential bugs, syntax errors, logic issues
        2. **Root Cause**: Explain what's causing the problem
        3. **Fix Suggestions**: Provide corrected code with explanations
        4. **Prevention**: How to avoid similar issues in the future
        5. **Testing**: Suggested test cases to verify the fix
        6. **Performance**: Any performance improvements while fixing

        Focus on practical, actionable solutions that improve code quality.
        """

        analysis = await get_ai_response(debug_prompt)
        
        # Extract potential fixes (simple regex-based extraction)
        fixes = []
        if "```" in analysis:
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', analysis, re.DOTALL)
            fixes = [{"code": block.strip(), "description": "AI-suggested fix"} for block in code_blocks]

        return {
            "analysis": analysis,
            "suggested_fixes": fixes,
            "language": request.language,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/terminal")
async def ai_terminal(request: TerminalRequest, db: Session = Depends(get_db)):
    """AI Terminal - Natural language to terminal commands (Cursor Terminal equivalent)"""
    try:
        # Log API usage
        api_usage = APIUsage(
            endpoint="/api/terminal",
            session_id=request.session_id
        )
        db.add(api_usage)
        db.commit()

        terminal_prompt = f"""
        You are an AI terminal assistant, equivalent to Cursor's terminal AI features.
        Convert natural language requests into appropriate terminal commands.

        User Request: {request.natural_command}
        Context: {request.context}

        Provide:
        1. **Command**: The exact terminal command to run
        2. **Explanation**: What the command does
        3. **Safety**: Any warnings or considerations
        4. **Alternatives**: Other ways to achieve the same result
        5. **Follow-up**: Suggested next commands if applicable

        Prioritize safe, cross-platform commands when possible.
        If the request is unclear or potentially dangerous, ask for clarification.
        """

        response = await get_ai_response(terminal_prompt)
        
        # Extract command (simple extraction - in production, use more sophisticated parsing)
        command = "echo 'Command generated by AI'"
        if "Command:" in response:
            try:
                command_line = [line for line in response.split('\n') if line.strip().startswith('Command:')][0]
                command = command_line.split('Command:', 1)[1].strip().strip('`')
            except:
                pass

        # For safety, don't actually execute commands automatically
        # In a production environment, you'd want user confirmation
        output = "Command generated. Execute manually for safety."

        return {
            "command": command,
            "output": output,
            "explanation": response,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Terminal error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/file-context")
async def file_context(request: FileContextRequest, db: Session = Depends(get_db)):
    """File Context Management - Reference and analyze files (@file equivalent)"""
    try:
        context_prompt = f"""
        Analyze the provided file for context and reference.

        File: {request.file_path}
        Operation: {request.operation}

        Content:
        ```
        {request.content}
        ```

        Provide:
        1. **Summary**: Brief description of the file's purpose
        2. **Key Functions/Classes**: Main components
        3. **Dependencies**: Imports and external dependencies
        4. **Usage Examples**: How this code might be used
        5. **Integration Points**: How it connects to other files
        """

        analysis = await get_ai_response(context_prompt)
        
        return {
            "file_path": request.file_path,
            "analysis": analysis,
            "operation": request.operation,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"File context error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Enhanced Real-time Features ---

@app.websocket("/ws/ai-assist")
async def websocket_ai_assist(websocket: WebSocket):
    """Real-time AI assistance WebSocket (like Cursor Tab)"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "cursor_prediction":
                # Simulate cursor position prediction
                prediction = {
                    "type": "cursor_suggestion",
                    "suggestion": "// AI suggests next action",
                    "confidence": 0.85
                }
                await websocket.send_text(json.dumps(prediction))
            
            elif message["type"] == "autocomplete":
                # Simulate smart autocomplete
                suggestion = {
                    "type": "autocomplete_suggestion", 
                    "text": "def example_function():",
                    "language": message.get("language", "python")
                }
                await websocket.send_text(json.dumps(suggestion))
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/api/health/enhanced")
async def enhanced_health_check():
    """Enhanced health check with feature status"""
    try:
        return {
            "status": "online",
            "features": {
                "chat": True,
                "composer": True,
                "debugger": True,
                "terminal": True,
                "file_context": True,
                "websocket": True,
                "code_execution": True,
                "analysis": True
            },
            "models_available": len(ENHANCED_MODELS),
            "languages_supported": len(SUPPORTED_LANGUAGES),
            "uptime": "0h 0m",  # Would calculate actual uptime
            "version": "2.0.0-cursor-enhanced"
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/api/smart-rewrite")
async def smart_rewrite(request: dict):
    """Smart code rewriting (like Cursor's smart rewrites)"""
    try:
        code = request.get("code", "")
        language = request.get("language", "python")
        
        # Simple typo fixes and improvements
        improvements = {
            "functin": "function",
            "retrun": "return", 
            "consle": "console",
            "imoprt": "import",
            "classe": "class",
            "methdo": "method"
        }
        
        improved_code = code
        changes_made = []
        
        for typo, correct in improvements.items():
            if typo in improved_code:
                improved_code = improved_code.replace(typo, correct)
                changes_made.append(f"Fixed '{typo}' → '{correct}'")
        
        return {
            "original": code,
            "improved": improved_code,
            "changes": changes_made,
            "language": language
        }
        
    except Exception as e:
        logger.error(f"Smart rewrite error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add these enhanced endpoints to your existing backend:
class CodeAnalysisRequest(BaseModel):
    code: str

class CodeWithMetrics(BaseModel):
    code: str
    metrics: dict

class ProjectRequest(BaseModel):
    project_description: str

@app.post("/api/analyze_code")
async def analyze_code(request: CodeAnalysisRequest):
    """Deep code analysis with metrics, patterns, and security"""
    # Implement AI-powered code analysis
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/api/optimize_performance")
async def optimize_performance(request: CodeWithMetrics):
    """Performance optimization suggestions"""
    # Analyze code complexity and suggest improvements
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/api/generate_workflow")
async def generate_workflow(request: ProjectRequest):
    """AI-generated development workflow"""
    # Create intelligent development pipelines
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/api/voice_to_code")
async def voice_to_code(audio: UploadFile):
    """Convert speech to code with context awareness"""
    # Implement speech recognition + code generation
    raise HTTPException(status_code=501, detail="Not implemented yet")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
