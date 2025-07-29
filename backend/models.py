from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    preferred_model = Column(String(100), default="deepseek/deepseek-chat-v3-0324")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)  # nullable for anonymous users
    session_name = Column(String(200), default="New Session")
    mode = Column(String(50), nullable=False)  # chat, execute, debug, refactor, analyze, workflow
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    response_time = Column(Float)  # in seconds
    metadata = Column(JSON)  # store additional data like execution results
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CodeExecution(Base):
    __tablename__ = "code_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    language = Column(String(50), nullable=False)
    code = Column(Text, nullable=False)
    stdout = Column(Text)
    stderr = Column(Text)
    error = Column(Text)
    execution_time = Column(Float)  # in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CodeAnalysis(Base):
    __tablename__ = "code_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    code = Column(Text, nullable=False)
    analysis_type = Column(String(50), nullable=False)  # debug, refactor, analyze
    results = Column(JSON)  # store analysis results
    model_used = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    processed = Column(Boolean, default=False)
    embeddings = Column(JSON)  # store code embeddings for search
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class APIUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time = Column(Float)
    tokens_used = Column(Integer)
    model_used = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())