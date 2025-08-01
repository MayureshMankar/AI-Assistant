from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Create the base class for models - using a completely clean approach
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    preferred_model = Column(String(100), default="deepseek/deepseek-chat-v3-0324")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    session_name = Column(String(200), default="New Session")
    mode = Column(String(50), nullable=False)  # chat, execute, debug, refactor, analyze, workflow
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    response_time = Column(Float)  # in seconds
    message_extra_data = Column(JSON)  # store additional data like execution results - renamed to avoid any conflict
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CodeExecution(Base):
    __tablename__ = "code_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
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
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    code = Column(Text, nullable=False)
    analysis_type = Column(String(50), nullable=False)  # debug, refactor, analyze
    results = Column(JSON)  # store analysis results
    model_used = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
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
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=True, index=True)  # ✅ Add this
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time = Column(Float)
    tokens_used = Column(Integer)
    model_used = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ProjectTemplate(Base):
    __tablename__ = "project_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    language = Column(String(50), nullable=False)
    framework = Column(String(100))
    template_data = Column(JSON)  # store template structure and files
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    tags = Column(JSON)  # additional tags
    timestamp = Column(DateTime(timezone=True), server_default=func.now())