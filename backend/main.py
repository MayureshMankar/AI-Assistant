from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import tempfile
import sys
from datetime import datetime
import time
from collections import defaultdict
from typing import List
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI(title="AI Coding Assistant Backend")

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

FREE_MODELS = [
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-coder",
    "deepseek/deepseek-r1",
    "qwen/qwen3-235b-a22b-2507",
    "tngtech/deepseek-r1t2-chimera",
    "moonshotai/kimi-k2",
    "google/gemini-2.0-flash-exp",
    "mistralai/mistral-nemo",
    "qwen/qwen-2.5-72b-instruct",
    "microsoft/mai-ds-r1",
    "moonshotai/kimi-dev-72b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "agentica-org/deepcoder-14b-preview",
    "moonshotai/kimi-vl-a3b-thinking",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
]

# --- Middlewares ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = None
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        print(f"{datetime.utcnow().isoformat()} | {request.client.host} | {request.method} {request.url.path} | {response.status_code} | {process_time:.2f}ms")
        return response
    except Exception as e:
        print(f"{datetime.utcnow().isoformat()} | ERROR | {request.method} {request.url.path} | {str(e)}")
        raise

RATE_LIMIT = 30
RATE_PERIOD = 60
rate_limit_store = defaultdict(list)

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    ip = request.client.host
    now = time.time()
    timestamps = rate_limit_store[ip]
    rate_limit_store[ip] = [t for t in timestamps if now - t < RATE_PERIOD]
    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
    rate_limit_store[ip].append(now)
    return await call_next(request)

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    model: str = None

class ChatResponse(BaseModel):
    response: str

class CodeExecutionRequest(BaseModel):
    language: str = "python"
    code: str

class CodeExecutionResponse(BaseModel):
    stdout: str
    stderr: str
    error: str = None

class CodeWithTask(BaseModel):
    code: str
    task: str

class CodeWithError(BaseModel):
    code: str
    error: str

class Workflow(BaseModel):
    steps: List[str]

# --- Endpoints ---
@app.get("/")
def read_root():
    return {"message": "AI Coding Assistant Backend is running."}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/models")
def list_models():
    return {"free_models": FREE_MODELS}

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not set.")
    
    model = request.model or "deepseek/deepseek-chat-v3-0324"
    
    if model not in FREE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not supported.")

    try:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"  # ✅ Correct custom base URL for OpenRouter
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.message}],
            max_tokens=4096
        )

        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        print(f"{datetime.utcnow().isoformat()} | ERROR | /api/chat | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/execute", response_model=CodeExecutionResponse)
def execute_code(request: CodeExecutionRequest):
    if request.language != "python":
        raise HTTPException(status_code=400, detail="Only Python code execution is supported.")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(request.code)
            tmp_path = tmp.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        os.remove(tmp_path)
        return CodeExecutionResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            error="" if result.returncode == 0 else f"Process exited with code {result.returncode}"
        )
    except subprocess.TimeoutExpired:
        return CodeExecutionResponse(stdout="", stderr="", error="Execution timed out.")
    except Exception as e:
        print(f"{datetime.utcnow().isoformat()} | ERROR | /api/execute | {str(e)}")
        return CodeExecutionResponse(stdout="", stderr="", error=str(e))

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

@app.post("/api/refactor")
async def refactor_code(request: CodeWithTask):
    """AI-powered code refactoring."""
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
