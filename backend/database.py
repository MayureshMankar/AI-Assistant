from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
import os
import logging
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to SQLite for development
    DATABASE_URL = "sqlite:///./ai_assistant.db"
    logger.info("No DATABASE_URL found, using SQLite for development")

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

# Database session dependency
def get_db() -> Generator[Session, None, None]:
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

def create_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        
        # Import models here to avoid any circular import issues
        from models import Base
        
        logger.info("Successfully imported Base model")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        return False

def initialize_database():
    """Initialize database with tables and default data"""
    try:
        logger.info("Initializing database...")
        
        # Test connection first
        if not test_connection():
            logger.error("Database connection failed")
            return False
            
        # Create tables
        if not create_tables():
            logger.error("Table creation failed")
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
        # Import models inside function to avoid circular imports
        from models import ProjectTemplate
        
        db = SessionLocal()
        
        try:
            # Check if we need to insert default data
            existing_templates = db.query(ProjectTemplate).count()
            if existing_templates == 0:
                logger.info("Inserting default data...")
                
                # Add some default project templates
                template1 = ProjectTemplate(
                    name="Python Flask API",
                    description="RESTful API with Flask framework",
                    language="python",
                    framework="flask",
                    template_data={
                        "files": {
                            "app.py": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'\n\nif __name__ == '__main__':\n    app.run(debug=True)",
                            "requirements.txt": "Flask==2.3.3"
                        }
                    }
                )
                
                template2 = ProjectTemplate(
                    name="React Frontend",
                    description="Modern React application",
                    language="javascript",
                    framework="react",
                    template_data={
                        "files": {
                            "src/App.js": "import React from 'react';\n\nfunction App() {\n  return (\n    <div className=\"App\">\n      <h1>Hello React!</h1>\n    </div>\n  );\n}\n\nexport default App;",
                            "package.json": '{\n  "name": "react-app",\n  "version": "1.0.0",\n  "dependencies": {\n    "react": "^18.0.0",\n    "react-dom": "^18.0.0"\n  }\n}'
                        }
                    }
                )
                
                template3 = ProjectTemplate(
                    name="Node.js Express API",
                    description="Express.js REST API server",
                    language="javascript",
                    framework="express",
                    template_data={
                        "files": {
                            "server.js": "const express = require('express');\nconst app = express();\nconst PORT = process.env.PORT || 3000;\n\napp.use(express.json());\n\napp.get('/', (req, res) => {\n  res.json({ message: 'Hello World!' });\n});\n\napp.listen(PORT, () => {\n  console.log(`Server running on port ${PORT}`);\n});",
                            "package.json": '{\n  "name": "express-api",\n  "version": "1.0.0",\n  "main": "server.js",\n  "dependencies": {\n    "express": "^4.18.0"\n  }\n}'
                        }
                    }
                )
                
                db.add(template1)
                db.add(template2)
                db.add(template3)
                db.commit()
                logger.info("✅ Default templates inserted successfully!")
            else:
                logger.info("Default data already exists, skipping insertion")
            
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to insert default data: {e}")
        finally:
            db.close()
        
    except Exception as e:
        logger.warning(f"Failed to insert default data: {e}")

def get_database_info():
    """Get database information for health checks"""
    try:
        db_type = "SQLite" if DATABASE_URL.startswith("sqlite") else "PostgreSQL"
        
        # Get table count
        with engine.connect() as connection:
            if DATABASE_URL.startswith("sqlite"):
                result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            else:
                result = connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public';"))
            
            tables = [row[0] for row in result.fetchall()]
        
        return {
            "database_type": db_type,
            "database_url": DATABASE_URL.split('@')[0] + '@***' if '@' in DATABASE_URL else DATABASE_URL,
            "tables": tables,
            "table_count": len(tables),
            "engine_info": str(engine.url),
            "pool_size": getattr(engine.pool, 'size', None),
            "connection_count": getattr(engine.pool, 'checked_in', None)
        }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {
            "database_type": "Unknown",
            "error": str(e)
        }

def cleanup_old_sessions(days=30):
    """Clean up old sessions and related data"""
    try:
        from models import ChatSession, ChatMessage, CodeExecution, CodeAnalysis
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        db = SessionLocal()
        try:
            # Find old sessions
            old_sessions = db.query(ChatSession).filter(
                ChatSession.created_at < cutoff_date
            ).all()
            
            if old_sessions:
                session_ids = [session.id for session in old_sessions]
                
                # Delete related data
                db.query(ChatMessage).filter(ChatMessage.session_id.in_(session_ids)).delete()
                db.query(CodeExecution).filter(CodeExecution.session_id.in_(session_ids)).delete()
                db.query(CodeAnalysis).filter(CodeAnalysis.session_id.in_(session_ids)).delete()
                
                # Delete sessions
                db.query(ChatSession).filter(ChatSession.id.in_(session_ids)).delete()
                
                db.commit()
                logger.info(f"✅ Cleaned up {len(old_sessions)} old sessions")
                return len(old_sessions)
            else:
                logger.info("No old sessions to clean up")
                return 0
                
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Session cleanup error: {e}")
        return 0

# Export commonly used items
__all__ = [
    "engine",
    "SessionLocal", 
    "get_db",
    "create_tables",
    "test_connection",
    "initialize_database",
    "get_database_info",
    "cleanup_old_sessions"

]