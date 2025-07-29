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

# Enhanced AI models with capabilities
ENHANCED_MODELS = {
    "deepseek/deepseek-chat-v3-0324": {"type": "chat", "code_analysis": True, "max_tokens": 4096},
    "deepseek/deepseek-r1": {"type": "reasoning", "code_analysis": True, "max_tokens": 8192},
    "qwen/qwen3-coder": {"type": "code", "code_analysis": True, "max_tokens": 32768},
    "google/gemini-2.0-flash-exp": {"type": "multimodal", "code_analysis": True, "max_tokens": 8192},
    "mistralai/mistral-nemo": {"type": "chat", "code_analysis": True, "max_tokens": 128000},
    "qwen/qwen-2.5-72b-instruct": {"type": "instruct", "code_analysis": True, "max_tokens": 32768},
}

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
    language: str = Field(..., regex="^(python|javascript|typescript|java|cpp|c|go|rust|php|ruby|bash)$")
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

@app.get("/api/health")
async def enhanced_health_check():
    """Enhanced health check with system status"""
    try:
        # Test AI API
        ai_status = "online" if OPENROUTER_API_KEY else "offline"
        
        # Test database (if configured)
        db_status = "offline"
        try:
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            db_status = "online"
        except:
            pass
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "ai_api": ai_status,
                "database": db_status,
                "file_system": "online"
            },
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "available_models": list(ENHANCED_MODELS.keys())
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/api/models")
async def list_enhanced_models():
    """List available AI models with capabilities"""
    return {
        "models": [
            {
                "id": model_id,
                "name": model_id.split("/")[-1],
                "type": config["type"],
                "code_analysis": config["code_analysis"],
                "max_tokens": config["max_tokens"]
            }
            for model_id, config in ENHANCED_MODELS.items()
        ],
        "languages": list(SUPPORTED_LANGUAGES.keys())
    }

@app.post("/api/chat")
async def enhanced_chat(request: EnhancedChatRequest, db: Session = Depends(get_db)):
    """Enhanced chat with session management and context awareness"""
    try:
        # Create or get session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Build enhanced prompt with context
        context_prompt = ""
        if request.context:
            context_prompt = f"Context: {json.dumps(request.context)}\n\n"
        
        full_prompt = f"{context_prompt}User Query: {request.message}"
        
        # Get AI response
        ai_response = await get_ai_response(
            full_prompt,
            request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Save to database (if available)
        try:
            # Save user message
            user_msg = ChatMessage(
                session_id=session_id,
                role="user",
                content=request.message,
                model_used=request.model
            )
            db.add(user_msg)
            
            # Save AI response
            ai_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=ai_response,
                model_used=request.model
            )
            db.add(ai_msg)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
        
        return {
            "response": ai_response,
            "session_id": session_id,
            "model_used": request.model,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

# --- New Features (Placeholders) ---
@app.post("/api/codebase/embed")
async def embed_codebase(files: List[UploadFile]):
    """Upload & index code for contextual queries."""
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.websocket("/ws/collab")
async def websocket_collab(websocket: WebSocket):
    """Sync edits across users in real-time."""
    await websocket.accept()
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/api/voice")
async def voice_command(audio: UploadFile):
    """Convert speech to code/commands."""
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/api/debug")
async def debug_code(request: CodeWithError):
    """AI debugger with traceback analysis."""
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/api/run_workflow")
async def run_workflow(workflow: Workflow):
    """Chain AI actions."""
    raise HTTPException(status_code=501, detail="Not implemented yet")

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
