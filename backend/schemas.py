from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

# User schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    preferred_model: Optional[str] = None
    is_active: Optional[bool] = None

class User(UserBase):
    id: UUID
    is_active: bool
    preferred_model: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Chat session schemas
class ChatSessionBase(BaseModel):
    session_name: str = Field(..., max_length=200)
    mode: str = Field(..., max_length=50)

class ChatSessionCreate(ChatSessionBase):
    user_id: Optional[UUID] = None

class ChatSessionUpdate(BaseModel):
    session_name: Optional[str] = Field(None, max_length=200)
    mode: Optional[str] = Field(None, max_length=50)

class ChatSession(ChatSessionBase):
    id: UUID
    user_id: Optional[UUID] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Chat message schemas
class ChatMessageBase(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)

class ChatMessageCreate(ChatMessageBase):
    session_id: UUID
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    message_extra_data: Optional[Dict[str, Any]] = None

class ChatMessage(ChatMessageBase):
    id: UUID
    session_id: UUID
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    message_extra_data: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Code execution schemas
class CodeExecutionBase(BaseModel):
    language: str = Field(..., max_length=50)
    code: str = Field(..., min_length=1)

class CodeExecutionCreate(CodeExecutionBase):
    session_id: UUID

class CodeExecution(CodeExecutionBase):
    id: UUID
    session_id: UUID
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Code analysis schemas
class CodeAnalysisBase(BaseModel):
    code: str = Field(..., min_length=1)
    analysis_type: str = Field(..., max_length=50)

class CodeAnalysisCreate(CodeAnalysisBase):
    session_id: UUID
    model_used: Optional[str] = None

class CodeAnalysis(CodeAnalysisBase):
    id: UUID
    session_id: UUID
    results: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# File upload schemas
class FileUploadBase(BaseModel):
    filename: str = Field(..., max_length=255)
    file_path: str = Field(..., max_length=500)

class FileUploadCreate(FileUploadBase):
    session_id: UUID
    file_size: Optional[int] = None
    mime_type: Optional[str] = None

class FileUpload(BaseModel):
    id: UUID
    session_id: UUID
    filename: str
    file_path: str
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    processed: bool = False
    embeddings: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Project template schemas
class ProjectTemplateBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    language: str = Field(..., max_length=50)
    framework: Optional[str] = Field(None, max_length=100)

class ProjectTemplateCreate(ProjectTemplateBase):
    template_data: Optional[Dict[str, Any]] = None

class ProjectTemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    language: Optional[str] = Field(None, max_length=50)
    framework: Optional[str] = Field(None, max_length=100)
    template_data: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class ProjectTemplate(ProjectTemplateBase):
    id: UUID
    template_data: Optional[Dict[str, Any]] = None
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# API usage schemas
class APIUsageBase(BaseModel):
    endpoint: str = Field(..., max_length=100)
    method: str = Field(..., max_length=10)

class APIUsageCreate(APIUsageBase):
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None  # ✅ Add this
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None

class APIUsage(APIUsageBase):
    id: UUID
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None  # ✅ Add this
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# System metrics schemas
class SystemMetricsBase(BaseModel):
    metric_name: str = Field(..., max_length=100)
    metric_value: float
    metric_type: str = Field(..., max_length=50)

class SystemMetricsCreate(SystemMetricsBase):
    tags: Optional[Dict[str, Any]] = None

class SystemMetrics(SystemMetricsBase):
    id: UUID
    tags: Optional[Dict[str, Any]] = None
    timestamp: datetime

    class Config:
        from_attributes = True

# Response schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    supported_languages: List[str]
    available_models: List[str]

class DatabaseInfo(BaseModel):
    database_type: str
    database_url: str
    tables: List[str]
    engine_info: str
    pool_size: Optional[str] = None
    connection_count: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: str