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
            model=request.model,
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
