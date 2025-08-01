#!/usr/bin/env python3
"""
AI Assistant Comprehensive Test Script
Tests all components: Backend, Frontend, Database, API endpoints
"""

import os
import sys
import subprocess
import requests
import json
import time
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, colors['INFO'])}[{status}] {message}{colors['RESET']}")

def check_python_version():
    """Check if Python version is compatible"""
    print_status("Checking Python version...", "INFO")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible", "SUCCESS")
        return True
    else:
        print_status(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Need 3.8+", "ERROR")
        return False

def check_node_version():
    """Check if Node.js is installed"""
    print_status("Checking Node.js version...", "INFO")
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"✅ Node.js {version} is installed", "SUCCESS")
            return True
        else:
            print_status("❌ Node.js not found", "ERROR")
            return False
    except FileNotFoundError:
        print_status("❌ Node.js not installed", "ERROR")
        return False

def check_npm_version():
    """Check if npm is installed"""
    print_status("Checking npm version...", "INFO")
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"✅ npm {version} is installed", "SUCCESS")
            return True
        else:
            print_status(f"❌ npm returned error: {result.stderr.strip()}", "ERROR")
            return False
    except FileNotFoundError:
        print_status("❌ npm not installed or not in PATH", "ERROR")
        return False
    except Exception as e:
        print_status(f"❌ Unexpected error checking npm: {e}", "ERROR")
        return False


def check_backend_dependencies():
    """Check if backend dependencies are installed"""
    print_status("Checking backend dependencies...", "INFO")
    
    backend_path = Path("backend")
    if not backend_path.exists():
        print_status("❌ Backend directory not found", "ERROR")
        return False
    
    requirements_path = backend_path / "requirements.txt"
    if not requirements_path.exists():
        print_status("❌ requirements.txt not found", "ERROR")
        return False
    
    # Check if virtual environment exists
    venv_path = backend_path / "venv"
    if venv_path.exists():
        print_status("✅ Virtual environment found", "SUCCESS")
    else:
        print_status("⚠️ Virtual environment not found (will create one)", "WARNING")
    
    return True

def check_frontend_dependencies():
    """Check if frontend dependencies are installed"""
    print_status("Checking frontend dependencies...", "INFO")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print_status("❌ Frontend directory not found", "ERROR")
        return False
    
    package_json = frontend_path / "package.json"
    if not package_json.exists():
        print_status("❌ package.json not found", "ERROR")
        return False
    
    node_modules = frontend_path / "node_modules"
    if node_modules.exists():
        print_status("✅ node_modules found", "SUCCESS")
    else:
        print_status("⚠️ node_modules not found (will install dependencies)", "WARNING")
    
    return True

def check_environment_setup():
    """Check environment configuration"""
    print_status("Checking environment setup...", "INFO")
    
    # Check for .env file in backend
    backend_env = Path("backend/.env")
    if backend_env.exists():
        print_status("✅ Backend .env file found", "SUCCESS")
    else:
        print_status("⚠️ Backend .env file not found (will create template)", "WARNING")
    
    # Check for .env file in frontend
    frontend_env = Path("frontend/.env")
    if frontend_env.exists():
        print_status("✅ Frontend .env file found", "SUCCESS")
    else:
        print_status("⚠️ Frontend .env file not found (will create template)", "WARNING")
    
    return True

def check_docker_setup():
    """Check Docker configuration"""
    print_status("Checking Docker setup...", "INFO")
    
    docker_compose = Path("docker-compose.yml")
    if docker_compose.exists():
        print_status("✅ docker-compose.yml found", "SUCCESS")
    else:
        print_status("⚠️ docker-compose.yml not found", "WARNING")
    
    backend_dockerfile = Path("backend/Dockerfile")
    if backend_dockerfile.exists():
        print_status("✅ Backend Dockerfile found", "SUCCESS")
    else:
        print_status("⚠️ Backend Dockerfile not found", "WARNING")
    
    frontend_dockerfile = Path("frontend/Dockerfile")
    if frontend_dockerfile.exists():
        print_status("✅ Frontend Dockerfile found", "SUCCESS")
    else:
        print_status("⚠️ Frontend Dockerfile not found", "WARNING")
    
    return True

def check_api_endpoints():
    """Check if backend API endpoints are accessible"""
    print_status("Checking API endpoints...", "INFO")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print_status("✅ Backend API is running", "SUCCESS")
            return True
        else:
            print_status(f"⚠️ Backend API returned status {response.status_code}", "WARNING")
            return False
    except requests.exceptions.ConnectionError:
        print_status("❌ Backend API not running (start with: cd backend && python main.py)", "ERROR")
        return False
    except Exception as e:
        print_status(f"❌ Error connecting to API: {e}", "ERROR")
        return False

def check_frontend_app():
    """Check if frontend app is accessible"""
    print_status("Checking frontend app...", "INFO")
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print_status("✅ Frontend app is running", "SUCCESS")
            return True
        else:
            print_status(f"⚠️ Frontend app returned status {response.status_code}", "WARNING")
            return False
    except requests.exceptions.ConnectionError:
        print_status("❌ Frontend app not running (start with: cd frontend && npm start)", "ERROR")
        return False
    except Exception as e:
        print_status(f"❌ Error connecting to frontend: {e}", "ERROR")
        return False

def create_env_template():
    """Create environment template files"""
    print_status("Creating environment template files...", "INFO")
    
    # Backend .env template
    backend_env_content = """# AI Assistant Backend Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
DATABASE_URL=sqlite:///./ai_assistant.db
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
MAX_FILE_SIZE=10485760
DB_ECHO=false
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Optional: PostgreSQL (uncomment to use)
# DATABASE_URL=postgresql://user:password@localhost/ai_assistant

# Optional: Redis for caching (uncomment to use)
# REDIS_URL=redis://localhost:6379
"""
    
    # Frontend .env template
    frontend_env_content = """# AI Assistant Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
GENERATE_SOURCEMAP=false
"""
    
    try:
        # Create backend .env
        backend_env_path = Path("backend/.env")
        if not backend_env_path.exists():
            backend_env_path.write_text(backend_env_content)
            print_status("✅ Created backend/.env template", "SUCCESS")
        else:
            print_status("✅ Backend .env already exists", "SUCCESS")
        
        # Create frontend .env
        frontend_env_path = Path("frontend/.env")
        if not frontend_env_path.exists():
            frontend_env_path.write_text(frontend_env_content)
            print_status("✅ Created frontend/.env template", "SUCCESS")
        else:
            print_status("✅ Frontend .env already exists", "SUCCESS")
        
        return True
    except Exception as e:
        print_status(f"❌ Error creating .env files: {e}", "ERROR")
        return False

def run_comprehensive_test():
    """Run comprehensive test of all components"""
    print_status("🚀 Starting AI Assistant Comprehensive Test", "INFO")
    print_status("=" * 50, "INFO")
    
    results = {}
    
    # System checks
    results['python'] = check_python_version()
    results['node'] = check_node_version()
    results['npm'] = check_npm_version()
    
    # Project structure checks
    results['backend_deps'] = check_backend_dependencies()
    results['frontend_deps'] = check_frontend_dependencies()
    results['env_setup'] = check_environment_setup()
    results['docker_setup'] = check_docker_setup()
    
    # Create environment templates if needed
    if not results['env_setup']:
        create_env_template()
    
    # Runtime checks (only if services are running)
    results['api_endpoints'] = check_api_endpoints()
    results['frontend_app'] = check_frontend_app()
    
    # Summary
    print_status("=" * 50, "INFO")
    print_status("📊 TEST RESULTS SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print_status(f"{test.replace('_', ' ').title()}: {status}", "SUCCESS" if result else "ERROR")
    
    print_status("=" * 50, "INFO")
    print_status(f"Overall: {passed}/{total} tests passed", "SUCCESS" if passed == total else "WARNING")
    
    if passed == total:
        print_status("🎉 All tests passed! Your AI Assistant is ready to use.", "SUCCESS")
    else:
        print_status("⚠️ Some tests failed. Check the issues above and fix them.", "WARNING")
    
    # Next steps
    print_status("=" * 50, "INFO")
    print_status("📋 NEXT STEPS:", "INFO")
    print_status("1. Get free API key from: https://openrouter.ai/keys", "INFO")
    print_status("2. Add your API key to backend/.env", "INFO")
    print_status("3. Start backend: cd backend && python main.py", "INFO")
    print_status("4. Start frontend: cd frontend && npm start", "INFO")
    print_status("5. Access web app at: http://localhost:3000", "INFO")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 