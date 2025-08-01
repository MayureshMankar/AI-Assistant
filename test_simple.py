#!/usr/bin/env python3
"""
Simple AI Assistant Test Script
Quick check of basic functionality
"""

import os
import sys
import requests
import json
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

def check_project_structure():
    """Check if all required files exist"""
    print_status("Checking project structure...", "INFO")
    
    required_files = [
        "backend/main.py",
        "backend/requirements.txt",
        "frontend/package.json",
        "frontend/src/App.js",
        "docker-compose.yml",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_status(f"❌ Missing files: {missing_files}", "ERROR")
        return False
    else:
        print_status("✅ All required files found", "SUCCESS")
        return True

def check_backend_code():
    """Check if backend code looks correct"""
    print_status("Checking backend code...", "INFO")
    
    try:
        with open("backend/main.py", "r") as f:
            content = f.read()
            
        # Check for key components
        checks = [
            ("FastAPI app", "from fastapi import FastAPI"),
            ("OpenRouter API", "OPENROUTER_API_KEY"),
            ("Free models", "ENHANCED_MODELS"),
            ("Chat endpoint", "@app.post(\"/api/chat\")"),
            ("Health check", "@app.get(\"/api/health\")")
        ]
        
        for name, pattern in checks:
            if pattern in content:
                print_status(f"✅ {name} found", "SUCCESS")
            else:
                print_status(f"❌ {name} missing", "ERROR")
                return False
        
        return True
    except Exception as e:
        print_status(f"❌ Error reading backend code: {e}", "ERROR")
        return False

def check_frontend_code():
    """Check if frontend code looks correct"""
    print_status("Checking frontend code...", "INFO")
    
    try:
        with open("frontend/src/App.js", "r", encoding="utf-8") as f:
            content = f.read()

            
        # Check for key components
        checks = [
            ("React app", "import React"),
            ("VS Code interface", "VSCodeIDE"),
            ("Free models", "FREE_MODELS"),
            ("Chat interface", "handleSendMessage"),
            ("File upload", "handleFileUpload")
        ]
        
        for name, pattern in checks:
            if pattern in content:
                print_status(f"✅ {name} found", "SUCCESS")
            else:
                print_status(f"❌ {name} missing", "ERROR")
                return False
        
        return True
    except Exception as e:
        print_status(f"❌ Error reading frontend code: {e}", "ERROR")
        return False

def check_docker_setup():
    """Check Docker configuration"""
    print_status("Checking Docker setup...", "INFO")
    
    try:
        with open("docker-compose.yml", "r") as f:
            content = f.read()
            
        # Check for key services
        services = ["postgres", "redis", "backend", "frontend"]
        for service in services:
            if f"  {service}:" in content:
                print_status(f"✅ {service} service configured", "SUCCESS")
            else:
                print_status(f"❌ {service} service missing", "ERROR")
                return False
        
        return True
    except Exception as e:
        print_status(f"❌ Error reading Docker config: {e}", "ERROR")
        return False

def run_simple_test():
    """Run simple test of all components"""
    print_status("🧪 Running Simple AI Assistant Test", "INFO")
    print_status("=" * 40, "INFO")
    
    tests = [
        ("Project Structure", check_project_structure),
        ("Backend Code", check_backend_code),
        ("Frontend Code", check_frontend_code),
        ("Docker Setup", check_docker_setup)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print_status("=" * 40, "INFO")
    print_status("📊 TEST RESULTS", "INFO")
    print_status("=" * 40, "INFO")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print_status(f"{test_name}: {status}", "SUCCESS" if result else "ERROR")
    
    print_status("=" * 40, "INFO")
    print_status(f"Overall: {passed}/{total} tests passed", "SUCCESS" if passed == total else "WARNING")
    
    if passed == total:
        print_status("🎉 All basic tests passed! Your AI Assistant code is ready.", "SUCCESS")
        print_status("Next: Follow QUICK_START_GUIDE.md to run the application", "INFO")
    else:
        print_status("⚠️ Some tests failed. Check the issues above.", "WARNING")
    
    return passed == total

if __name__ == "__main__":
    success = run_simple_test()
    sys.exit(0 if success else 1) 