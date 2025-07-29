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
        from models import (
            Base, User, ChatSession, ChatMessage, CodeExecution, 
            CodeAnalysis, FileUpload, APIUsage, ProjectTemplate, SystemMetrics
        )
        
        logger.info("Successfully imported all models")
        
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
                
                db.add(template1)
                db.add(template2)
                db.commit()
                logger.info("Default templates inserted successfully!")
            else:
                logger.info("Default data already exists, skipping insertion")
            
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to insert default data: {e}")
        finally:
            db.close()
        
    except Exception as e:
        logger.warning(f"Failed to insert default data: {e}")

# Export commonly used items
__all__ = [
    "engine",
    "SessionLocal", 
    "get_db",
    "create_tables",
    "test_connection",
    "initialize_database"
]