from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import TypeDecorator, CHAR
import uuid


# Custom UUID type that works with both SQLite and PostgreSQL
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(32), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value

# Create the base class for models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    preferred_model = Column(String(100), default="google/gemini-2.0-flash-exp:free")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    api_usage = relationship("APIUsage", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True, index=True)
    session_name = Column(String(200), default="New Session")
    mode = Column(String(50), nullable=False, default="chat")  # chat, execute, debug, refactor, analyze, workflow
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    code_executions = relationship("CodeExecution", back_populates="session", cascade="all, delete-orphan")
    code_analyses = relationship("CodeAnalysis", back_populates="session", cascade="all, delete-orphan")
    file_uploads = relationship("FileUpload", back_populates="session", cascade="all, delete-orphan")
    api_usage = relationship("APIUsage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    response_time = Column(Float)  # in seconds
    message_extra_data = Column(JSON)  # store additional data like execution results
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

class CodeExecution(Base):
    __tablename__ = "code_executions"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    language = Column(String(50), nullable=False)
    code = Column(Text, nullable=False)
    stdout = Column(Text)
    stderr = Column(Text)
    error = Column(Text)
    execution_time = Column(Float)  # in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="code_executions")

class CodeAnalysis(Base):
    __tablename__ = "code_analyses"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    code = Column(Text, nullable=False)
    analysis_type = Column(String(50), nullable=False)  # debug, refactor, analyze
    results = Column(JSON)  # store analysis results
    model_used = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="code_analyses")

class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    processed = Column(Boolean, default=False)
    embeddings = Column(JSON)  # store code embeddings for search
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="file_uploads")

class APIUsage(Base):
    __tablename__ = "api_usage"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(GUID(), ForeignKey("chat_sessions.id"), nullable=True, index=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False, default="POST")
    status_code = Column(Integer)
    response_time = Column(Float)
    tokens_used = Column(Integer)
    model_used = Column(String(100))
    language = Column(String(50))  # Programming language if applicable
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="api_usage")
    session = relationship("ChatSession", back_populates="api_usage")

class ProjectTemplate(Base):
    __tablename__ = "project_templates"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
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
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    tags = Column(JSON)  # additional tags
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

# Additional utility model for user preferences
class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, unique=True)
    preferred_language = Column(String(50), default="python")
    preferred_theme = Column(String(20), default="dark")
    auto_save = Column(Boolean, default=True)
    show_line_numbers = Column(Boolean, default=True)
    enable_autocomplete = Column(Boolean, default=True)
    enable_syntax_highlighting = Column(Boolean, default=True)
    font_size = Column(Integer, default=14)
    tab_size = Column(Integer, default=4)
    preferences_data = Column(JSON)  # Additional custom preferences
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Model for storing code snippets
class CodeSnippet(Base):
    __tablename__ = "code_snippets"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    language = Column(String(50), nullable=False)
    code = Column(Text, nullable=False)
    tags = Column(JSON)  # Array of tags
    is_public = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Model for tracking AI model performance
class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    endpoint = Column(String(100), nullable=False)
    average_response_time = Column(Float)
    success_rate = Column(Float)  # Percentage of successful requests
    total_requests = Column(Integer, default=0)
    total_failures = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    
# Index definitions for better performance
from sqlalchemy import Index

# Create indexes for frequently queried columns
Index('idx_chat_sessions_user_id', ChatSession.user_id)
Index('idx_chat_messages_session_id', ChatMessage.session_id)
Index('idx_code_executions_session_id', CodeExecution.session_id)
Index('idx_code_analyses_session_id', CodeAnalysis.session_id)
Index('idx_api_usage_user_id', APIUsage.user_id)
Index('idx_api_usage_session_id', APIUsage.session_id)
Index('idx_code_snippets_user_id', CodeSnippet.user_id)
Index('idx_code_snippets_language', CodeSnippet.language)

# Add created_at indexes for time-based queries
Index('idx_chat_sessions_created_at', ChatSession.created_at)
Index('idx_chat_messages_created_at', ChatMessage.created_at)
Index('idx_code_executions_created_at', CodeExecution.created_at)
Index('idx_api_usage_created_at', APIUsage.created_at)
