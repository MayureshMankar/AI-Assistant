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
from database import get_db, SessionLocal, initialize_database
from models import ChatSession, ChatMessage, CodeExecution, CodeAnalysis, FileUpload, APIUsage
import logging
from uuid import UUID, uuid4
import traceback


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:

    logger.warning("⚠️ OPENROUTER_API_KEY not found in .env file! Some features may not work.")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# FastAPI App
app = FastAPI(
    title="AI Coding Assistant Pro",
    description="Advanced AI-powered coding assistant",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

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

# Enhanced free models that are working
ENHANCED_MODELS = {
    "google/gemini-2.0-flash-exp:free": {
        "type": "multimodal",
        "code_analysis": True,
        "max_tokens": 1048576,
        "description": "Google: Gemini 2.0 Flash Experimental (free)"
    },
    "qwen/qwen-2.5-coder-32b-instruct:free": {
        "type": "programming",
        "code_analysis": True,
        "max_tokens": 32768,
        "description": "Qwen: Qwen2.5 Coder 32B Instruct (free)"
    },
    "meta-llama/llama-3.2-3b-instruct:free": {
        "type": "general",
        "code_analysis": True,
        "max_tokens": 131072,
        "description": "Meta: Llama 3.2 3B Instruct (free)"
    },
    "microsoft/phi-3-mini-128k-instruct:free": {
        "type": "general",
        "code_analysis": True,
        "max_tokens": 128000,
        "description": "Microsoft: Phi-3 Mini 128K Instruct (free)"
    },
    "huggingfaceh4/zephyr-7b-beta:free": {
        "type": "general",
        "code_analysis": True,
        "max_tokens": 32768,
        "description": "HuggingFace: Zephyr 7B Beta (free)"
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
    logger.info(f"[{request_id}] {request.method} {request.url.path} - {request.client.host if request.client else 'unknown'}")
    
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
    ip = request.client.host if request.client else "unknown"
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
    model: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4096, ge=1, le=32768)

class EnhancedCodeExecutionRequest(BaseModel):
    language: str
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

async def get_ai_response(prompt: str, model: str = None, **kwargs) -> str:
    """Enhanced AI response with better error handling"""
    if not OPENROUTER_API_KEY:
        return "AI service not configured. Please provide a valid OpenRouter API key."
    
    # Use the first available model if none specified
    if not model or model not in ENHANCED_MODELS:
        model = list(ENHANCED_MODELS.keys())[0]
    
    try:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        model_config = ENHANCED_MODELS.get(model, list(ENHANCED_MODELS.values())[0])
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", min(4000, model_config["max_tokens"] // 2)),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"AI API error: {str(e)}")
        return f"AI service error: {str(e)}. Please try again or check your API configuration."

# Database initialization on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("🚀 Starting AI Coding Assistant Pro...")
    try:
        if initialize_database():
            logger.info("✅ Database initialized successfully")
        else:
            logger.warning("⚠️ Database initialization failed, but continuing...")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

# Enhanced Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Coding Assistant Pro - Advanced Version",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "Multi-language code execution",
            "Advanced code analysis", 
            "AI-powered refactoring",
            "Project generation",
            "Real-time collaboration",
            "File management",
            "Free AI models"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check with working free models information"""
    try:
        return {
            "status": "online",
            "version": "2.0.0-enhanced",
            "features": {
                "chat": True,
                "code_analysis": True,
                "auto_language_detection": True,
                "context_handling": True,
                "free_models": True,
                "code_execution": True
            },
            "free_models": list(ENHANCED_MODELS.keys()),
            "models_count": len(ENHANCED_MODELS),
            "supported_languages": list(LANGUAGE_PATTERNS.keys()),
            "max_context": max(model['max_tokens'] for model in ENHANCED_MODELS.values()),
            "api_key_configured": bool(OPENROUTER_API_KEY),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

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

# Enhanced chat endpoint with better context handling and AI response

@app.post("/api/chat")
async def enhanced_chat(request: EnhancedChatRequest, db: Session = Depends(get_db)):
    try:

        # Validate UUID if provided

        session_uuid = None
        if request.session_id:
            try:
                session_uuid = UUID(str(request.session_id))

                # Check if session exists, create if not
                existing_session = db.query(ChatSession).filter_by(id=session_uuid).first()
                if not existing_session:
                    new_session = ChatSession(
                        id=session_uuid,
                        user_id=None,
                        session_name="Chat Session",
                        mode="chat"
                    )
                    db.add(new_session)
                    db.commit()
            except (ValueError, Exception) as e:
                logger.warning(f"Invalid session_id, creating new: {e}")
                session_uuid = None

        # Create new session if needed
        if not session_uuid:
            session_uuid = uuid.uuid4()
            new_session = ChatSession(
                id=session_uuid,
                user_id=None,
                session_name="New Chat Session",
                mode="chat"
            )
            db.add(new_session)
            db.commit()

        # Validate model
        model_to_use = request.model if request.model and request.model in ENHANCED_MODELS else DEFAULT_MODEL
        model_info = ENHANCED_MODELS[model_to_use]

        # Build context
        context_parts = []
        
        # System context
        system_context = f"""
You are an expert AI coding assistant.
Model: {model_to_use} ({model_info['description']})
Your capabilities:
- Code generation and debugging
- Multi-language support
- Best practices guidance
- Architecture recommendations
"""
        context_parts.append(system_context)

        # Add user context if provided
        if request.context:
            if request.context.get("conversation_history"):
                context_parts.append("\nRecent conversation:")
                for msg in request.context["conversation_history"][-5:]:  # Last 5 messages
                    if isinstance(msg, dict):
                        role = msg.get("role", "").upper()
                        content = msg.get("content", "")[:200]  # Truncate long messages
                        context_parts.append(f"{role}: {content}")

            if request.context.get("editor_content"):
                editor_content = request.context["editor_content"]
                detected_lang = detect_language(editor_content)
                context_parts.append(f"\nCurrent editor content ({detected_lang}):\n```{detected_lang}\n{editor_content[:500]}\n```")

        # Add user message
        context_parts.append(f"\nUser Question: {request.message}")
        full_context = "\n".join(context_parts)

        # Truncate if too long
        max_context = model_info["max_tokens"] - 1000  # Reserve space for response
        if len(full_context) > max_context:
            full_context = full_context[:max_context]

        # Get AI response
        ai_response = await get_ai_response(
            full_context, 
            model_to_use, 
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Save to database (with error handling)
        try:
            # Save user message
            user_message = ChatMessage(
                session_id=session_uuid,
                role="user",
                content=request.message,
                model_used=model_to_use
            )
            db.add(user_message)
            
            # Save assistant response
            assistant_message = ChatMessage(
                session_id=session_uuid,
                role="assistant", 
                content=ai_response,
                model_used=model_to_use
            )
            db.add(assistant_message)
            
            # Log API usage
            api_usage = APIUsage(
                session_id=session_uuid,
                endpoint="/api/chat",
                method="POST",
                model_used=model_to_use,
                status_code=200
            )
            db.add(api_usage)
            
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
            db.rollback()

        return {
            "response": ai_response,
            "model_used": model_to_use,
            "session_id": str(session_uuid),
            "context_included": bool(request.context),

            "detected_language": detect_language(request.message) if "```" in request.message else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:

        logger.error(f"Enhanced chat error: {e}\n{traceback.format_exc()}")
        return {
            "response": f"I encountered an error: {str(e)}. Please try again.",
            "error": str(e),
            "model_used": model_to_use if 'model_to_use' in locals() else DEFAULT_MODEL,
            "session_id": str(session_uuid) if 'session_uuid' in locals() and session_uuid else None
        }


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
            compile_result = subprocess.run(
                ["javac", tmp_path], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if compile_result.returncode != 0:
                os.unlink(tmp_path)
                return {
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "error": "Compilation failed",
                    "execution_time": 0,
                    "language": language
                }
            # Find class name from file
            class_name = os.path.basename(tmp_path).replace('.java', '')
            cmd = ["java", "-cp", os.path.dirname(tmp_path), class_name]
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
        try:
            os.unlink(tmp_path)
            if language == "java":
                class_file = tmp_path.replace(".java", ".class")
                if os.path.exists(class_file):
                    os.unlink(class_file)
        except:
            pass
        
        # Save execution result (with error handling)
        try:
            session_id = request.session_id or str(uuid.uuid4())
            execution_record = CodeExecution(
                session_id=UUID(session_id) if session_id else uuid.uuid4(),
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
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {
            "stdout": "",
            "stderr": "",
            "error": f"Execution timed out after {request.timeout or lang_config['timeout']} seconds",
            "execution_time": request.timeout or lang_config['timeout'],
            "language": language
        }
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
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
        Analyze this {request.language} code for: {', '.join(request.analysis_type)}
        
        Code:
        ```{request.language}
        {request.code}
        ```
        
        Provide analysis for:
        1. Code quality and readability
        2. Performance optimization opportunities
        3. Security considerations
        4. Best practices compliance
        5. Refactoring suggestions
        
        Be specific and actionable in your recommendations.
        """
        
        ai_analysis = await get_ai_response(ai_analysis_prompt, DEFAULT_MODEL)
        analysis_results["ai_analysis"] = ai_analysis
        
        # Save analysis (with error handling)
        try:
            session_id = request.session_id or str(uuid.uuid4())
            analysis_record = CodeAnalysis(
                session_id=UUID(session_id) if session_id else uuid.uuid4(),
                code=request.code,
                analysis_type=",".join(request.analysis_type),
                results=analysis_results,
                model_used=DEFAULT_MODEL
            )
            db.add(analysis_record)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
        
        return {
            "analysis": analysis_results,
            "language": request.language,
            "analysis_types": request.analysis_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code analysis error: {str(e)}")
        return {
            "analysis": {"error": str(e)},
            "language": request.language,
            "analysis_types": request.analysis_type,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/refactor")
async def enhanced_refactor(request: RefactorRequest):
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
        
        refactored_response = await get_ai_response(refactor_prompt, DEFAULT_MODEL)
        
        return {
            "refactored_code": refactored_response,
            "original_code": request.code,
            "refactor_types": request.refactor_type,
            "language": request.language,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Refactor error: {str(e)}")
        return {
            "refactored_code": f"Error during refactoring: {str(e)}",
            "original_code": request.code,
            "refactor_types": request.refactor_type,
            "language": request.language,
            "timestamp": datetime.utcnow().isoformat()
        }

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
        
        project_structure = await get_ai_response(generation_prompt, DEFAULT_MODEL)
        
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
        return {
            "project_structure": f"Error generating project: {str(e)}",
            "description": request.description,
            "language": request.language,
            "framework": request.framework,
            "features": request.features,
            "timestamp": datetime.utcnow().isoformat()
        }

# --- Enhanced Cursor-like Features ---

class ComposerRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
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
        model_to_use = request.model if request.model and request.model in ENHANCED_MODELS else DEFAULT_MODEL
        
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

        response = await get_ai_response(composer_prompt, model_to_use)
        
        # Log API usage (with error handling)
        try:
            session_id = request.session_id or str(uuid.uuid4())
            api_usage = APIUsage(
                endpoint="/api/composer",
                method="POST",
                model_used=model_to_use,
                session_id=UUID(session_id) if session_id else None,
                status_code=200
            )
            db.add(api_usage)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database logging failed: {db_error}")
        
        return {
            "suggestions": response,
            "files_referenced": [f.get("name", "unknown") for f in request.files],
            "language": request.language,
            "model_used": model_to_use,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Composer error: {str(e)}")
        return {
            "suggestions": f"Error in composer: {str(e)}",
            "files_referenced": [f.get("name", "unknown") for f in request.files],
            "language": request.language,
            "model_used": DEFAULT_MODEL,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/debug")
async def ai_debugger(request: DebugRequest, db: Session = Depends(get_db)):
    """AI Debugger - Intelligent debugging and error fixing (Cursor Debug equivalent)"""
    try:
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

        analysis = await get_ai_response(debug_prompt, DEFAULT_MODEL)
        
        # Extract potential fixes (simple regex-based extraction)
        fixes = []
        if "```" in analysis:
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', analysis, re.DOTALL)
            fixes = [{"code": block.strip(), "description": "AI-suggested fix"} for block in code_blocks]

        # Log API usage (with error handling)
        try:
            session_id = request.session_id or str(uuid.uuid4())
            api_usage = APIUsage(
                endpoint="/api/debug",
                method="POST",
                session_id=UUID(session_id) if session_id else None,
                status_code=200
            )
            db.add(api_usage)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database logging failed: {db_error}")

        return {
            "analysis": analysis,
            "suggested_fixes": fixes,
            "language": request.language,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {
            "analysis": f"Debug analysis error: {str(e)}",
            "suggested_fixes": [],
            "language": request.language,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/terminal")
async def ai_terminal(request: TerminalRequest, db: Session = Depends(get_db)):
    """AI Terminal - Natural language to terminal commands (Cursor Terminal equivalent)"""
    try:
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

        response = await get_ai_response(terminal_prompt, DEFAULT_MODEL)
        
        # Extract command (simple extraction)
        command = "echo 'Command generated by AI'"
        if "Command:" in response:
            try:
                command_lines = [line for line in response.split('\n') if line.strip().startswith('Command:')]
                if command_lines:
                    command = command_lines[0].split('Command:', 1)[1].strip().strip('`')
            except:
                pass

        # For safety, don't execute commands automatically
        output = "Command generated. Execute manually for safety."

        # Log API usage (with error handling)
        try:
            session_id = request.session_id or str(uuid.uuid4())
            api_usage = APIUsage(
                endpoint="/api/terminal",
                method="POST",
                session_id=UUID(session_id) if session_id else None,
                status_code=200
            )
            db.add(api_usage)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database logging failed: {db_error}")

        return {
            "command": command,
            "output": output,
            "explanation": response,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Terminal error: {str(e)}")
        return {
            "command": "echo 'Error occurred'",
            "output": f"Terminal AI error: {str(e)}",
            "explanation": f"Error processing request: {str(e)}",
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/file-context")
async def file_context(request: FileContextRequest):
    """File Context Management - Reference and analyze files (@file equivalent)"""
    try:
        context_prompt = f"""
        Analyze the provided file for context and reference.

        File: {request.file_path}
        Operation: {request.operation}

        Content:
        ```
        {request.content[:2000]}  # Limit content to prevent token overflow
        ```

        Provide:
        1. **Summary**: Brief description of the file's purpose
        2. **Key Functions/Classes**: Main components
        3. **Dependencies**: Imports and external dependencies
        4. **Usage Examples**: How this code might be used
        5. **Integration Points**: How it connects to other files
        """

        analysis = await get_ai_response(context_prompt, DEFAULT_MODEL)
        
        return {
            "file_path": request.file_path,
            "analysis": analysis,
            "operation": request.operation,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"File context error: {str(e)}")
        return {
            "file_path": request.file_path,
            "analysis": f"File analysis error: {str(e)}",
            "operation": request.operation,
            "timestamp": datetime.utcnow().isoformat()
        }

# --- Enhanced Real-time Features ---

@app.websocket("/ws/ai-assist")
async def websocket_ai_assist(websocket: WebSocket):
    """Real-time AI assistance WebSocket (like Cursor Tab)"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "cursor_prediction":
                # Simulate cursor position prediction
                prediction = {
                    "type": "cursor_suggestion",
                    "suggestion": "// AI suggests completing this function",
                    "confidence": 0.85,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(prediction))
            
            elif message.get("type") == "autocomplete":
                # Simulate smart autocomplete
                language = message.get("language", "python")
                suggestion = {
                    "type": "autocomplete_suggestion", 
                    "text": f"def example_function(): # Auto-generated for {language}",
                    "language": language,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(suggestion))
            
            elif message.get("type") == "ping":
                # Handle ping/pong for connection health
                pong = {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(pong))
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

@app.get("/api/health/enhanced")
async def enhanced_health_check():
    """Enhanced health check with feature status"""
    try:
        # Test database connection
        db_status = "unknown"
        try:
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        return {
            "status": "online",
            "version": "2.0.0-enhanced",
            "features": {
                "chat": True,
                "composer": True,
                "debugger": True,
                "terminal": True,
                "file_context": True,
                "websocket": True,
                "code_execution": True,
                "analysis": True,
                "refactoring": True,
                "project_generation": True
            },
            "database": {
                "status": db_status,
                "type": "SQLite/PostgreSQL"
            },
            "api": {
                "openrouter_configured": bool(OPENROUTER_API_KEY),
                "models_available": len(ENHANCED_MODELS),
                "default_model": DEFAULT_MODEL
            },
            "languages_supported": len(SUPPORTED_LANGUAGES),
            "execution_engines": list(SUPPORTED_LANGUAGES.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/smart-rewrite")
async def smart_rewrite(request: dict):
    """Smart code rewriting (like Cursor's smart rewrites)"""
    try:
        code = request.get("code", "")
        language = request.get("language", "python")
        
        if not code:
            return {
                "original": "",
                "improved": "",
                "changes": ["No code provided"],
                "language": language
            }
        
        # Simple typo fixes and improvements
        improvements = {
            "functin": "function",
            "retrun": "return", 
            "consle": "console",
            "imoprt": "import",
            "classe": "class",
            "methdo": "method",
            "varialbe": "variable",
            "lenght": "length",
            "widht": "width",
            "heigth": "height"
        }
        
        improved_code = code
        changes_made = []
        
        for typo, correct in improvements.items():
            if typo in improved_code:
                improved_code = improved_code.replace(typo, correct)
                changes_made.append(f"Fixed '{typo}' → '{correct}'")
        
        # If no simple fixes, try AI-powered improvements
        if not changes_made and OPENROUTER_API_KEY:
            try:
                ai_prompt = f"""
                Improve this {language} code by fixing typos, improving style, and enhancing readability:
                
                ```{language}
                {code}
                ```
                
                Return only the improved code without explanations.
                """
                
                ai_improved = await get_ai_response(ai_prompt, DEFAULT_MODEL)
                if ai_improved and ai_improved != code:
                    improved_code = ai_improved
                    changes_made.append("AI-powered improvements applied")
            except Exception as ai_error:
                logger.warning(f"AI rewrite failed: {ai_error}")
        
        return {
            "original": code,
            "improved": improved_code,
            "changes": changes_made if changes_made else ["No improvements needed"],
            "language": language
        }
        
    except Exception as e:
        logger.error(f"Smart rewrite error: {str(e)}")
        return {
            "original": request.get("code", ""),
            "improved": request.get("code", ""),
            "changes": [f"Error: {str(e)}"],
            "language": request.get("language", "python")
        }

# Additional utility endpoints
@app.get("/api/sessions")
async def get_sessions(db: Session = Depends(get_db)):
    """Get recent chat sessions"""
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).limit(10).all()
        return {
            "sessions": [
                {
                    "id": str(session.id),
                    "name": session.session_name,
                    "mode": session.mode,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat() if session.updated_at else None
                }
                for session in sessions
            ],
            "total": len(sessions)
        }
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        return {"sessions": [], "total": 0, "error": str(e)}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session and its messages"""
    try:
        session_uuid = UUID(session_id)
        
        # Delete messages first
        db.query(ChatMessage).filter_by(session_id=session_uuid).delete()
        
        # Delete session
        session = db.query(ChatSession).filter_by(id=session_uuid).first()
        if session:
            db.delete(session)
            db.commit()
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        db.rollback()
        logger.error(f"Delete session error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn

    
    # Initialize database on startup
    logger.info("🚀 Starting AI Coding Assistant Pro...")
    try:
        initialize_database()
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "production") == "development"
    )
