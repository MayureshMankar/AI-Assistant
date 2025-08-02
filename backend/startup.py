#!/usr/bin/env python3
"""
AI Coding Assistant Pro - Startup Script
Handles database initialization, health checks, and application startup
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/startup.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set"""
    logger.info("🔍 Checking environment configuration...")
    
    required_vars = []
    optional_vars = {
        'OPENROUTER_API_KEY': 'AI features will be limited',
        'DATABASE_URL': 'Will use SQLite by default',
        'REDIS_URL': 'Caching will be disabled',
        'SECRET_KEY': 'Will generate a temporary key'
    }
    
    missing_vars = []
    warnings = []
    
    # Check required variables
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    # Check optional variables
    for var, warning in optional_vars.items():
        if not os.getenv(var):
            warnings.append(f"⚠️ {var} not set: {warning}")
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info("✅ Environment check completed")
    return True

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating necessary directories...")
    
    directories = [
        'uploads',
        'logs',
        'temp',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory created/verified: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("📦 Checking dependencies...")
    
    critical_deps = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'pydantic',
        'python-dotenv'
    ]
    
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        logger.error(f"❌ Missing critical dependencies: {', '.join(missing_deps)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All critical dependencies are installed")
    return True

def initialize_database():
    """Initialize the database"""
    logger.info("🗃️ Initializing database...")
    
    try:
        from database import initialize_database, get_database_info
        
        if initialize_database():
            logger.info("✅ Database initialized successfully")
            
            # Log database info
            db_info = get_database_info()
            logger.info(f"Database type: {db_info.get('database_type', 'Unknown')}")
            logger.info(f"Tables: {len(db_info.get('tables', []))}")
            
            return True
        else:
            logger.error("❌ Database initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        return False

def test_ai_connection():
    """Test AI service connection"""
    logger.info("🤖 Testing AI service connection...")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("⚠️ No OpenRouter API key found, AI features will be limited")
            return True  # Not critical for startup
        
        # Test connection with a simple request
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Just verify the client can be created, don't make actual API call during startup
        logger.info("✅ AI service connection configured")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ AI service connection issue: {e}")
        return True  # Not critical for startup

def check_system_resources():
    """Check system resources"""
    logger.info("⚡ Checking system resources...")
    
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.percent}% used ({memory.available // 1024 // 1024} MB available)")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        logger.info(f"Disk: {disk.percent}% used ({disk.free // 1024 // 1024 // 1024} GB free)")
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU: {cpu_percent}% usage")
        
        # Warnings for low resources
        if memory.percent > 90:
            logger.warning("⚠️ High memory usage detected")
        if disk.percent > 90:
            logger.warning("⚠️ Low disk space available")
        if cpu_percent > 80:
            logger.warning("⚠️ High CPU usage detected")
            
        return True
        
    except ImportError:
        logger.info("psutil not available, skipping resource check")
        return True
    except Exception as e:
        logger.warning(f"Resource check failed: {e}")
        return True

def cleanup_temp_files():
    """Clean up temporary files from previous runs"""
    logger.info("🧹 Cleaning up temporary files...")
    
    temp_dirs = ['temp', 'uploads/temp']
    cleaned_count = 0
    
    for temp_dir in temp_dirs:
        temp_path = Path(temp_dir)
        if temp_path.exists():
            for file_path in temp_path.glob('*'):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to clean {file_path}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"✅ Cleaned up {cleaned_count} temporary files")
    else:
        logger.info("✅ No temporary files to clean")

def run_startup_checks():
    """Run all startup checks"""
    logger.info("🚀 Starting AI Coding Assistant Pro...")
    logger.info("=" * 50)
    
    checks = [
        ("Environment Configuration", check_environment),
        ("Directory Creation", create_directories),
        ("Dependencies Check", check_dependencies),
        ("Database Initialization", initialize_database),
        ("AI Service Connection", test_ai_connection),
        ("System Resources", check_system_resources),
        ("Cleanup", cleanup_temp_files)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            logger.error(f"❌ {check_name} failed with exception: {e}")
            failed_checks.append(check_name)
    
    logger.info("=" * 50)
    
    if failed_checks:
        logger.error(f"❌ Startup completed with issues: {', '.join(failed_checks)}")
        logger.warning("Some features may not work properly")
    else:
        logger.info("✅ All startup checks passed successfully!")
    
    logger.info("🎉 AI Coding Assistant Pro is ready to start!")
    return len(failed_checks) == 0

def main():
    """Main startup function"""
    try:
        success = run_startup_checks()
        
        if success:
            logger.info("Starting server...")
            # Import and run the FastAPI app
            import uvicorn
            from main import app
            
            port = int(os.getenv("PORT", 8000))
            host = os.getenv("HOST", "0.0.0.0")
            reload = os.getenv("ENVIRONMENT", "production") == "development"
            
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=reload,
                log_level=os.getenv("LOG_LEVEL", "info").lower()
            )
        else:
            logger.error("❌ Startup checks failed. Please fix the issues and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
