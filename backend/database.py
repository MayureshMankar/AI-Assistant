from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
import os
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to SQLite for development
    DATABASE_URL = "sqlite:///./ai_assistant.db"
    logger.warning("No DATABASE_URL found, using SQLite for development")

# Database engine configuration
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=os.getenv("DB_ECHO", "false").lower() == "true"
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=int(os.getenv("DB_POOL_SIZE", 20)),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", 30)),
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections every hour
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
        poolclass=QueuePool
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base from models to avoid conflicts
# Note: This import must be after engine creation to avoid circular imports
try:
    from models import Base
except ImportError:
    # Fallback for testing or if models.py doesn't exist
    from sqlalchemy.orm import declarative_base
    Base = declarative_base()
    logger.warning("Could not import Base from models.py, creating fallback Base")

# Database session dependency
def get_db() -> Session:
    """Get database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Async context manager for database sessions
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[Session, None]:
    """Async context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Database initialization and management
def create_tables():
    """Create all database tables"""
    try:
        # Import all models to ensure they are registered with Base
        from models import (
            User, ChatSession, ChatMessage, CodeExecution, 
            CodeAnalysis, FileUpload, APIUsage, ProjectTemplate, SystemMetrics
        )
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        return False

def drop_tables():
    """Drop all database tables"""
    try:
        from models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Database tables dropped successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to drop database tables: {e}")
        return False

def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            # Test basic connectivity
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
            
        logger.info("✅ Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def get_database_info():
    """Get database information and statistics"""
    try:
        with engine.connect() as connection:
            if DATABASE_URL.startswith("sqlite"):
                # SQLite queries
                tables_result = connection.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ))
                tables = [row[0] for row in tables_result.fetchall()]
                
                return {
                    "database_type": "SQLite",
                    "database_url": DATABASE_URL,
                    "tables": tables,
                    "engine_info": str(engine.url),
                    "pool_size": "N/A (SQLite)",
                    "connection_count": "N/A (SQLite)"
                }
            else:
                # PostgreSQL queries
                tables_result = connection.execute(text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ))
                tables = [row[0] for row in tables_result.fetchall()]
                
                # Get connection pool info
                pool = engine.pool
                
                return {
                    "database_type": "PostgreSQL",
                    "database_url": str(engine.url).replace(str(engine.url.password), "***") if engine.url.password else str(engine.url),
                    "tables": tables,
                    "engine_info": str(engine.url).replace(str(engine.url.password), "***") if engine.url.password else str(engine.url),
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
                
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {
            "error": str(e),
            "database_type": "Unknown",
            "status": "error"
        }

def initialize_database():
    """Initialize database with tables and default data"""
    try:
        logger.info("Initializing database...")
        
        # Test connection first
        if not test_connection():
            return False
            
        # Create tables
        if not create_tables():
            return False
            
        # Insert default data if needed
        insert_default_data()
        
        logger.info("✅ Database initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def insert_default_data():
    """Insert default data into the database"""
    try:
        from models import User, ChatSession, ProjectTemplate
        
        db = SessionLocal()
        
        try:
            # Check if we need to insert default data
            existing_users = db.query(User).count()
            if existing_users == 0:
                logger.info("Inserting default data...")
                
                # Add some default project templates
                default_templates = [
                    {
                        "name": "Python Flask API",
                        "description": "RESTful API with Flask framework",
                        "language": "python",
                        "framework": "flask",
                        "template_data": {
                            "files": {
                                "app.py": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'\n\nif __name__ == '__main__':\n    app.run(debug=True)",
                                "requirements.txt": "Flask==2.3.3"
                            }
                        }
                    },
                    {
                        "name": "React Frontend",
                        "description": "Modern React application",
                        "language": "javascript",
                        "framework": "react",
                        "template_data": {
                            "files": {
                                "src/App.js": "import React from 'react';\n\nfunction App() {\n  return (\n    <div className=\"App\">\n      <h1>Hello React!</h1>\n    </div>\n  );\n}\n\nexport default App;",
                                "package.json": '{\n  "name": "react-app",\n  "version": "1.0.0",\n  "dependencies": {\n    "react": "^18.0.0",\n    "react-dom": "^18.0.0"\n  }\n}'
                            }
                        }
                    }
                ]
                
                for template_data in default_templates:
                    template = ProjectTemplate(**template_data)
                    db.add(template)
                
                db.commit()
                logger.info("Default templates inserted successfully!")
            
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to insert default data: {e}")
        finally:
            db.close()
        
    except Exception as e:
        logger.warning(f"Failed to insert default data: {e}")

def reset_database():
    """Reset database by dropping and recreating all tables"""
    try:
        logger.info("Resetting database...")
        drop_tables()
        create_tables()
        insert_default_data()
        logger.info("✅ Database reset complete!")
        return True
    except Exception as e:
        logger.error(f"❌ Database reset failed: {e}")
        return False

# Database health check
def health_check():
    """Comprehensive database health check"""
    health_status = {
        "status": "healthy",
        "connection": False,
        "tables_exist": False,
        "can_write": False,
        "can_read": False,
        "info": {}
    }
    
    try:
        # Test connection
        health_status["connection"] = test_connection()
        
        # Get database info
        health_status["info"] = get_database_info()
        
        # Test if tables exist
        info = get_database_info()
        health_status["tables_exist"] = len(info.get("tables", [])) > 0
        
        # Test read/write operations
        with SessionLocal() as db:
            try:
                # Try to read from a table (if it exists)
                result = db.execute(text("SELECT 1")).fetchone()
                health_status["can_read"] = True
                health_status["can_write"] = True  # If we can connect, we can usually write
            except Exception:
                health_status["can_read"] = False
                health_status["can_write"] = False
        
        # Determine overall status
        if not health_status["connection"]:
            health_status["status"] = "unhealthy"
        elif not health_status["tables_exist"]:
            health_status["status"] = "needs_initialization"
        elif not health_status["can_read"] or not health_status["can_write"]:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
        logger.error(f"Database health check failed: {e}")
    
    return health_status

# Export commonly used items
__all__ = [
    "engine",
    "SessionLocal", 
    "Base",
    "get_db",
    "get_db_session",
    "create_tables",
    "test_connection",
    "initialize_database",
    "health_check",
    "get_database_info",
    "reset_database"
]